import cobra
import cvxpy as cp
import numpy as np
from dataclasses import dataclass
import json
import scipy.sparse as sp
from carveme.reconstruction.scoring import reaction_scoring
from carveme.reconstruction.diamond import load_diamond_results
from carveme import config, project_dir
import pandas as pd
import logging
from pathlib import Path
import gurobipy as gp
from gurobipy import GRB
import argparse

# CONFIG
USE_LOGGING = True  # Set to False to not log at all
DISPLAY_DEBUG = True  # display log_debug() messages
VERBOSE_SOLVER = True

ALWAYS_ENFORCE_ATPM = False  # if False, solver will not enforce ATPM flux for non growth organisms
SINGLE_GROWTH_THRESHOLD = False  # if True, will use a single threshold for growth and non-growth
USE_REGULARIZATION = False  # if True, will add a micro-penalty to force the solver to use the smallest possible mu
USE_STRONG_DUALITY = True  # if True, will add a consraint enforcing strong (as opposed to weak) duality
EXPORT_PROBLEM = True  # if True, will export problem to .mps and .lp file formats (only works with new code)

ASSERT_TOLERANCE = 1e-7  # tolerance passed to all assert statements to ensure they are met
MODEL_TOLERANCE  = 1e-8  # tolerance passed directly to the parameters of the model on problem.solve()

# !!! NOTE: Model file names or Organism names CANNOT CONTAIN "#"

# BIOM_REACTION_ID = "BIOMASS_Ecoli_core_w_GAM"  # core ecoli model
# BIOM_REACTION_ID = "BIOMASS_Ec_iML1515_core_75p37M"  # iML1515
BIOM_REACTION_ID = "Growth"  # CarveMe models (json load -> R_Growth | xlm load -> Growth)
ATPM_REACTION_ID = "ATPM"
EXTRACELLS_SPACE = "C_e"  # space for extracellular metabolites (in CarveMe models)
# EXTRACELLS_SPACE = "e"  # ^ for iML1515

DBMODEL_FILENAME = "universal_model.json"  # universal model acting as a database
SCORES_DIRECTORY = "scores"  # directory that has all .tsv diamond output files for all organisms in the community
MODELS_DIRECTORY = "models"  # directory that has all .xml model files for all organisms in the community
# NOTE: the .xml and corresponding .tsv files must have the same name (apart from their file extension)

# region constants
# standard for reaction bounds + big M
M = 1000

# Flux thresholds
GROWTH_THRESHOLD = 0.05
NON_GROWTH_THRESHOLD = 0.01
MINIMAL_ATPM_FLUX = 8.39

# Weights
W_REM_RX = 3  # remove reaction
W_ADD_RX = 4  # add reaction
W_REV_RX = 2  # reverse reaction
W_EXC_EXP = 200  # exclude experiment
# endregion

# region LOGGING
if USE_LOGGING:
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s,%(msecs)03d [%(levelname)s] - %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler('glocom.log'),
            logging.StreamHandler()  # Also print to console
        ]
    )
    _logger = logging.getLogger(__name__)
else:
    _logger = None

def log_info(msg):
    """Log info message or ic if() logging disabled."""
    if USE_LOGGING:
        _logger.info(msg)

def log_debug(msg=""):
    """Log debug message (or ic() if logging disabled)."""
    if not DISPLAY_DEBUG:
        return
    if USE_LOGGING:
        _logger.debug(msg if msg else "[CHECKPOINT]")
# endregion

# region Helper functions and classes
def load_json_model(filename, is_DB: bool = False) -> cobra.Model:
    """Loads the specified JSON model assuring it runs on the newer versions of cobrapy."""

    with open(filename, "r") as file:
        file_as_dict = json.load(file)

    # Filter the reactions to not include exchange, demand, or sink reactions
    filtered_reactions = []
    for rxn in file_as_dict.get('reactions', []):
        rxn_id_lower = rxn['id'].lower()

        # Prevent solver keyword collisions (gurobi "St" bug)
        if rxn_id_lower == 'st':
            rxn['id'] = rxn['id'] + '_rxn'
            rxn['name'] = rxn['name'] + ' (Renamed)' # due to solver keyword collisions

        if is_DB:
            # Do not allow Exchange (EX), Demand (DM), or Sink (SK) reactions from the DB
            if not (rxn_id_lower.startswith("ex_") or
                    rxn_id_lower.startswith("dm_") or
                    rxn_id_lower.startswith("sk_") or
                    "biomass" in rxn_id_lower):
                filtered_reactions.append(rxn)

    if is_DB:
        # Replace the unedited reactions list with our filtered one
        file_as_dict['reactions'] = filtered_reactions

    # Clean the dictionaries to satisfy cobrapy's strict typing
    # We check metabolites, reactions, and genes for faulty annotations
    for component in ['metabolites', 'reactions', 'genes']:
        if component in file_as_dict:
            for item in file_as_dict[component]:
                if 'annotation' in item and not isinstance(item['annotation'], dict):
                    # If it's None, a list, or a string, force it into an empty dict
                    item['annotation'] = {}

    # Load the cleaned dictionary directly into a cobra Model
    model = cobra.io.model_from_dict(file_as_dict)

    return model

def load_xml_model(filename) -> cobra.Model:
    return cobra.io.read_sbml_model(filename)


class MockVar:
    """Mock CVXPY variable structure to maintain compatibility with evaluation methods"""
    def __init__(self, value):
        self.value = value

    def __getitem__(self, idx):
        # Intercepts array indexing (e.g., deltas_REM[i]) to return a new mock
        # containing the scalar value, mirroring CVXPY's slicing behavior.
        # (fixes print and evaluate results)
        return MockVar(self.value[idx])

@dataclass
class Experiment:
    reactions: list[cobra.Reaction]
    growth: dict[str, bool]  # ex.: {"Ecoli": True, "...": False}

# endregion


class GloCom:
    def __init__(self,
                 model: cobra.Model,
                 target_org: str,
                 organisms: list[str],
                 experiments: list[Experiment],
                 db_model: cobra.Model = None,
                 expected_result: str = None):
        self.model = model
        self.organisms = organisms
        self.n_organisms = len(organisms)
        self.target_org = target_org
        self.expected_result = expected_result # for testing

        # Track base reactions before any DB or reverse reactions are added
        n_rxns_pre = len(self.model.reactions)

        # Now allows passing a custom DB model for programmatic toy tests
        DB_model = db_model if db_model is not None else self.get_model_from_DB()
        log_info(f"Loaded {len(DB_model.reactions)} universal reactions from the database.")

        # Species-Specific Database Integration (TARGET ORGANISM ONLY)
        db_reactions_to_add = []
        for db_rxn in DB_model.reactions:
            loc_rxn = cobra.Reaction(f"{self.target_org}#DB#{db_rxn.id}")
            loc_rxn.name = f"[{self.target_org} Database] {db_rxn.name}"
            loc_rxn.lower_bound = db_rxn.lower_bound
            loc_rxn.upper_bound = db_rxn.upper_bound

            loc_mets = {}
            for met, coeff in db_rxn.metabolites.items():
                if met.compartment == EXTRACELLS_SPACE or met.id.endswith(EXTRACELLS_SPACE):
                    if met.id.endswith(f"_{EXTRACELLS_SPACE}"):
                        shared_id = met.id
                    elif met.id.endswith(f"#{EXTRACELLS_SPACE}"):  # failsafe for old tests
                        shared_id = f"{met.id[:-(len(EXTRACELLS_SPACE) + 1)]}_{EXTRACELLS_SPACE}"
                    elif met.id.endswith(EXTRACELLS_SPACE):
                        shared_id = f"{met.id[:-len(EXTRACELLS_SPACE)]}_{EXTRACELLS_SPACE}"
                    else:
                        shared_id = met.id
                    if shared_id in self.model.metabolites:
                        loc_met = self.model.metabolites.get_by_id(shared_id)
                    else:
                        loc_met = met.copy()
                        loc_met.id = shared_id
                        loc_met.compartment = EXTRACELLS_SPACE
                else:
                    internal_id = f"{self.target_org}#{met.id}"
                    if internal_id in self.model.metabolites:
                        loc_met = self.model.metabolites.get_by_id(internal_id)
                    else:
                        loc_met = met.copy()
                        loc_met.id = internal_id
                        loc_met.compartment = f"{self.target_org}#{met.compartment}"

                loc_mets[loc_met] = coeff

            loc_rxn.add_metabolites(loc_mets)
            db_reactions_to_add.append(loc_rxn)

        self.model.add_reactions(db_reactions_to_add)
        log_info(f"Added {len(db_reactions_to_add)} target-specific DB reactions.")
        n_rxns_merge = len(self.model.reactions)

        # Reversibility and Stoichiometric Matrix Math
        N = n_rxns_merge
        protected_rxns = [BIOM_REACTION_ID, ATPM_REACTION_ID]  # also used to have sink

        # Create an S matrix in dok format (for easy conversion into sparse csc later) directly from the cobrapy model
        S_dok = cobra.util.create_stoichiometric_matrix(self.model, array_type='dok')
        # Convert S_dok to csc sparse matrix for (memory) optimization purposes since most of the matrix is zeroes
        S_base = sp.csc_matrix(S_dok)
        # Horizontally join the previous matrix with its negative to add the reverse of all reactions in the model
        # (The model never gets these reaction objects - they exist only as numbers in this matrix and bounds arrays for cvxpy)
        self.S = sp.hstack([S_base, -S_base], format='csc')  # [S, -S]
        n_mets, n_rxns = self.S.shape

        self.experiments = experiments
        exp_len = len(self.experiments)

        # Reaction Bounds Processing
        unproc_ubs = np.zeros(n_rxns)
        orig_lbs = np.zeros(n_rxns)  # Original lower bounds
        orig_ubs = np.zeros(n_rxns)  # Original upper bounds
        for i, r in enumerate(self.model.reactions):
            # Forward bounds
            pre_ubs = max(0, r.upper_bound)
            if r.lower_bound < 0 and pre_ubs == 0:
                unproc_ubs[i] = M
            else:
                unproc_ubs[i] = pre_ubs
            orig_ubs[i] = r.upper_bound
            orig_lbs[i] = r.lower_bound

            # Reverse bounds mapping
            is_protected = any(p in r.id for p in protected_rxns) or str(r.id).lower().startswith("ex")
            is_target = r.id.startswith(f"{self.target_org}#")

            # Default strict reverse bound (0 if irreversible, -LB if reversible)
            unproc_ubs[N + i] = max(0, -r.lower_bound)

            if not is_protected and is_target and r.lower_bound >= 0:
                # If the reaction is irreversible (+ from target org and not ATPM/BIOM)
                #  -> set the reverse upper bound to 1000 (M) to make it reversible with dREV
                unproc_ubs[N + i] = M

            r._lower_bound = 0

        react_ids = [r.id for r in self.model.reactions]
        self.ubs = []
        for exp in self.experiments:
            exp_ubs = unproc_ubs.copy()
            for reaction in exp.reactions:
                try:
                    fwd_i = react_ids.index(reaction.id)
                    exp_ubs[fwd_i] = max(0, reaction.upper_bound)
                    exp_ubs[N + fwd_i] = max(0, -reaction.lower_bound)
                except ValueError:
                    pass
            self.ubs.append(exp_ubs)
        self.ubs = np.array(self.ubs)

        self.cATP = np.zeros(n_rxns)
        self.cREM = np.zeros(n_rxns)
        self.cREV = np.zeros(n_rxns)
        self.cADD = np.zeros(n_rxns)

        # Track biomass and ATPM index for each individual organism
        self.org_biomass_indices = {}
        self.org_atpm_indices = {}
        for org_name in self.organisms:
            try:
                idx = self.model.reactions.index(self.model.reactions.get_by_id(f"{org_name}#{BIOM_REACTION_ID}"))
                self.org_biomass_indices[org_name] = idx
            except KeyError:
                log_info(f"[!] WARNING: Biomass reaction for {org_name} not found.")
            try:
                atpm_idx = self.model.reactions.index(self.model.reactions.get_by_id(f"{org_name}#{ATPM_REACTION_ID}"))
                self.cATP[atpm_idx] = 1
                self.org_atpm_indices[org_name] = atpm_idx
            except KeyError:
                pass

        # Track reaction indices for each individual organism
        self.org_rxns_idx = {self.organisms[i]: [] for i in range(len(self.organisms))}

        # Populate Index Masks iteratively mapping N to N+i (TARGET ORG ONLY)
        for i, r in enumerate(self.model.reactions):
            is_db = (i >= n_rxns_pre)  # or rev
            rev_idx = N + i
            is_target = r.id.startswith(f"{self.target_org}#")
            orig_lb = orig_lbs[i]

            if not is_db:
                if is_target:
                    # Dynamically protect EX, BIOMASS, and ATPM regardless of specific ID variants
                    is_protected = (
                        any(p in r.id for p in protected_rxns)
                        # or "biomass" in str(r.id).lower()  # commented out -> now the program requires a file with only a single biomass reaction
                    )

                    if not is_protected:
                        self.cREM[i] = 1
                        if orig_lb < 0:  # use cached bounds
                            self.cREM[rev_idx] = 1
                            if orig_ubs[i] <= 0:  # less or equal because of the max() function when defining unproc_ubs
                                self.cREV[i] = 1
                                self.cREM[i] = 0
                        else:
                            self.cREV[rev_idx] = 1
                else:  # not target
                    # WARNING - solution is slightly janky, but works for now (org names should NOT have '#')
                    split = r.id.split("#")[0]
                    if split != "EX" and split in self.org_rxns_idx:
                        self.org_rxns_idx[split].append(i)
                        self.org_rxns_idx[split].append(i + N)
            else:  # from db
                self.cADD[i] = 1
                self.cADD[rev_idx] = 1
                # Possible future thing to worry about: EX reactions coming from the database
                #  -> these should be added by default and not be able to be removed

        # Identify all non-growth events to scale Dual Variables
        self.ng_events = []
        for i_exp, exp in enumerate(self.experiments):
            for org_name, grows in exp.growth.items():
                if not grows:
                    self.ng_events.append((i_exp, org_name))

        self.n_ng = len(self.ng_events)

        log_debug(f"  [DEBUG] Mathematical S-Matrix Constructed. Shape (Mets, Rxns): {self.S.shape}")
        log_debug(f"  [DEBUG] Biomass targets mapped to matrix columns: {self.org_biomass_indices}")
        log_debug(f"  [DEBUG] Total Non-Growth events to dualize: {self.n_ng}")

        # Global Constants and Thresholds
        pre_Wadd = n_rxns * [W_ADD_RX]
        pre_Wrem = n_rxns * [W_REM_RX]
        self.Wadd = self.get_weights(pre_Wadd, N, "add")
        self.Wrem = self.get_weights(pre_Wrem, N, "rem")
        self.Wrev = np.full(n_rxns, W_REV_RX)
        self.Wexp = np.full(exp_len, W_EXC_EXP)

        # CVXPY Variable Declarations
        self.v = cp.Variable((exp_len, n_rxns), name="fluxes", nonneg=True)
        self.deltas_REM = cp.Variable(n_rxns, name="binary_rem", boolean=True)
        self.deltas_ADD = cp.Variable(n_rxns, name="binary_add", boolean=True)
        self.deltas_REV = cp.Variable(n_rxns, name="binary_rev", boolean=True)
        self.deltas_EXP = cp.Variable(exp_len, name="binary_exp", boolean=True)

        if self.n_ng > 0:
            self.d_lmbda = cp.Variable((self.n_ng, n_mets), name="dual_lambda")
            self.d_mu = cp.Variable((self.n_ng, n_rxns), name="dual_mu", nonneg=True)
            self.d_xi = cp.Variable((self.n_ng, n_rxns), name="dual_xi", nonneg=True)
            self.z_REM = cp.Variable((self.n_ng, n_rxns), name="dual_lin_rem", nonneg=True)
            self.z_REV = cp.Variable((self.n_ng, n_rxns), name="dual_lin_rev", nonneg=True)
            self.z_ADD = cp.Variable((self.n_ng, n_rxns), name="dual_lin_add", nonneg=True)
            self.z_EXP_atpm = cp.Variable((self.n_ng, n_rxns), name="dual_lin_exp_atpm", nonneg=True)
            self.z_EXP_bio = cp.Variable((self.n_ng, self.n_organisms), name="dual_lin_exp_bio", nonneg=True)

        log_info(f" >>> GloCom Object Initialization Complete. <<<")

    @classmethod
    def _build_community_from_models(cls, org_models: dict[str, cobra.Model]) -> cobra.Model:
        """Core logic separated so it can be safely used by both directory and other loaders."""
        community = cobra.Model("Community_Model")

        shared_mets_dict = {}
        internal_mets_dict = {}

        for org_name, model in org_models.items():
            boundary_ids = set(r.id for r in model.boundary)

            new_reactions = []
            for rxn in model.reactions:
                # Safely strip boundaries, but strictly protect our core analytical reactions
                if rxn.id in boundary_ids:
                    if rxn.id not in [BIOM_REACTION_ID, ATPM_REACTION_ID] and rxn.id.startswith("EX_"):  # also used to have sink
                        if not rxn.id.startswith("EX_"): log_debug(f"  [DEBUG] {org_name}: Stripping boundary reaction '{rxn.id}'")
                        continue

                if rxn.upper_bound == 0:
                    if rxn.lower_bound == 0:
                        continue

                new_rxn = cobra.Reaction(f"{org_name}#{rxn.id}")
                new_rxn.name = f"[{org_name}] {rxn.name}"
                new_rxn.lower_bound = rxn.lower_bound
                new_rxn.upper_bound = rxn.upper_bound

                new_mets = {}
                for met, coeff in rxn.metabolites.items():
                    is_shared = (met.compartment == EXTRACELLS_SPACE) or (met.id.endswith(f"_{EXTRACELLS_SPACE}"))

                    if is_shared:
                        if met.id not in shared_mets_dict:
                            new_met = met.copy()
                            new_met.compartment = EXTRACELLS_SPACE
                            shared_mets_dict[met.id] = new_met
                        new_mets[shared_mets_dict[met.id]] = coeff
                    else:
                        new_met_id = f"{org_name}#{met.id}"
                        if new_met_id not in internal_mets_dict:
                            new_met = met.copy()
                            new_met.id = new_met_id
                            new_met.compartment = f"{org_name}#{met.compartment}"
                            internal_mets_dict[new_met_id] = new_met
                        new_mets[internal_mets_dict[new_met_id]] = coeff

                new_rxn.add_metabolites(new_mets)

                new_reactions.append(new_rxn)

            community.add_reactions(new_reactions)
            log_debug(f"  [DEBUG] {org_name}: Successfully merged {len(new_reactions)} reactions into community.")

        community_exchanges = []
        for met_id, met in shared_mets_dict.items():
            # Single reversible exchange reaction (after, it gets turned into two in [S, -S])
            # Positive flux = Export (leaves system). Negative flux = Import (enters system).
            ex_rxn = cobra.Reaction(f"EX#{met_id}")
            ex_rxn.name = f"ComEx Reaction for {met.name}"
            ex_rxn.lower_bound = 0  # always 0 until changed in each experiment for imports into the cell
            ex_rxn.upper_bound = M  # always 1000 we want to keep everything open for exporting into the environment

            community_exchanges.append(ex_rxn)

        # Add the empty reactions to the model FIRST to avoid implicit addition warnings
        community.add_reactions(community_exchanges)

        # Populate stoichiometry (Convention: -1 coefficient means export consumes it from the compartment)
        for met_id, met in shared_mets_dict.items():
            community.reactions.get_by_id(f"EX#{met_id}").add_metabolites({met: -1})

        log_debug(f"  [DEBUG] Community Build Final S-Matrix dimensions: {len(community.metabolites)}x{len(community.reactions)}")
        return community

    @classmethod
    def from_list(cls,
                  target_org: str,
                  org_list: list[str],
                  experiments: list[Experiment] = None,
                  db_model: cobra.Model = None,
                  expected_result: str = None) -> 'GloCom':
        """Builds a community model from a list of organism models in a directory (default: \'./models\')."""
        log_info(f"Building community model from directory: {MODELS_DIRECTORY}")
        target_dir = Path(MODELS_DIRECTORY)
        model_files = [f for f in target_dir.iterdir() if f.is_file() and f.suffix.lower() == '.xml']

        org_models = {}
        for f in model_files:
            if f.stem in org_list:
                model = load_xml_model(str(f))
                # The following code caused problems on toy models and was never tested on real ones
                # hoever it does not seem to be required, if an issue occurs related to this consider uncommenting

                # blocked_reacs = cobra.flux_analysis.find_blocked_reactions(model)
                # model.remove_reactions(blocked_reacs, remove_orphans = True) # Will remove orphan metabolites
                # model.repair()

                org_models[f.stem] = model

        community = cls._build_community_from_models(org_models)
        return cls(community, target_org, list(org_models.keys()), experiments, db_model, expected_result)

    def get_weights(self, pre_weights: list, N: int, operation: str) -> np.ndarray:
        """
        Parses genetic evidence scores and updates addition/removal weights.
        Updated to support localized community database reactions and their mathematical reverses.
        """
        diamond_filename = f"{SCORES_DIRECTORY}/{self.target_org}.tsv"

        try:
            # Load diamond file with blast results
            log_info(f" >> Loading diamond results from: {diamond_filename}")
            anotations = load_diamond_results(diamond_filename)
            bigg_gprs = project_dir + config.get("generated", "bigg_gprs")
            gprs = pd.read_csv(bigg_gprs)

            scores = reaction_scoring(anotations, gprs, debug_output=None)
            scores[0]["reaction"] = scores[0]["reaction"].apply(lambda x: x[2:] if x.startswith("R_") else x)
            if operation == "add":
                scores[0]["normalized_score"] = scores[0]["normalized_score"].apply(lambda x: round(1 + (1 / (1 + x)), 2))
            elif operation == "rem":
                scores[0]["normalized_score"] = scores[0]["normalized_score"].apply(lambda x: round(3 + (x / 3), 2))
            scores_dict: dict = scores[0].set_index("reaction").to_dict()["normalized_score"]

            # Database mismatch failsafe - diamond files were generated from another db other than BiGG
            if not scores_dict or all(pd.isna(v) for v in scores_dict.values()):
                log_info(" >> [!] WARNING: All genetic scores evaluated to NaN (Database ID mismatch likely). Defaulting to base weights.")
                # If this happens, make sure the chosen database file reaction ID types are the same as the model reaction ID types
                return np.array(pre_weights)
        except Exception as e:
            log_info(f" >> [!] ERROR while loading weights: {e}")
            return np.array(pre_weights)

        test_arr = []
        for i, rxn in enumerate(self.model.reactions):
            if operation == "add" and "#DB#" in rxn.id:  # if the reaction is from DB
                # rxn.id format: "orgA_DB_rxn1" or "rev_orgA_DB_rxn1"
                # Split at "#DB#" and take the right side to get the base BiGG ID
                base_id = rxn.id.split("#DB#", 1)[1]
                if result := scores_dict.get(base_id):
                    pre_weights[i] = result     # Forward reaction
                    pre_weights[N + i] = result # Reverse reaction
            elif operation == "rem" and "#DB#" not in rxn.id:
                test_arr.append(rxn.id)
                base_id = rxn.id.split(self.target_org + "#", 1)[-1]
                if result := scores_dict.get(base_id):
                    pre_weights[i] = result     # Forward reaction
                    pre_weights[N + i] = result # Reverse reaction

        return np.array(pre_weights)

    @staticmethod
    def get_model_from_DB() -> cobra.Model:
        """Gets the universal model from the database file."""
        return load_json_model(DBMODEL_FILENAME, True)

    def get_objective_function(self):
        """Defines the objective function for the optimization problem."""
        Wadd_masked = self.Wadd * self.cADD
        Wrem_masked = self.Wrem * self.cREM
        Wrev_masked = self.Wrev * self.cREV

        fun = self.deltas_ADD @ Wadd_masked \
            + self.deltas_REM @ Wrem_masked \
            + self.deltas_REV @ Wrev_masked \
            + self.deltas_EXP @ self.Wexp

        # Add a micro-penalty to force the solver to use the smallest possible mu (Tikhonov regularization)
        # NOTE: This does not work 100% well in the inner problem since we use weak duality as opposed to strong
        #       so I put it here instead and it works because it always forces sum(mu) to be as small as possible
        if USE_REGULARIZATION and self.n_ng > 0:
            fun += cp.sum(self.d_mu) * 1e-8  # changed because some terms in the biomass reaction are 10^-6

        return cp.Minimize(fun)

    def print_results(self, problem, return_string=False):
        """Prints the final results of the optimization problem."""
        final = []
        def print_log(message):
            if return_string:
                final.append(message)
            return log_info(message)

        if problem.status in ["infeasible", "unbounded"]:
            print_log(f" >> Problem failed: {problem.status}")
            return

        # new mu verification
        if self.n_ng > 0 and self.d_mu.value is not None:
            N = len(self.model.reactions)
            rxn_ids = [r.id for r in self.model.reactions]

            # Extract rounded deltas safely
            d_REM_val = np.round(self.deltas_REM.value) if self.deltas_REM.value is not None else np.zeros(N)
            d_REV_val = np.round(self.deltas_REV.value) if self.deltas_REV.value is not None else np.zeros(N)
            d_ADD_val = np.round(self.deltas_ADD.value) if self.deltas_ADD.value is not None else np.zeros(N)

            mu_matrix = self.d_mu.value

            for ng_idx, (exp_idx, org_ng) in enumerate(self.ng_events):
                for rxn_i in range(2 * N):
                    mu_val = mu_matrix[ng_idx, rxn_i]

                    # If Mu approaches Big-M (0.99 accounts for solver float inaccuracies, idk if it is needed)
                    if mu_val > (M * 0.99):
                        base_i = rxn_i if rxn_i < N else rxn_i - N
                        direction = "FWD" if rxn_i < N else "REV"
                        base_ub = self.ubs[exp_idx][rxn_i]
                        eff_ub = base_ub

                        # Calculate EFFECTIVE(!) upper bound *after* solver modifications
                        if self.cREM[rxn_i] == 1:
                            eff_ub = base_ub * (1 - d_REM_val[base_i])
                        elif self.cREV[rxn_i] == 1:
                            eff_ub = M * d_REV_val[base_i]
                        elif self.cADD[rxn_i] == 1:
                            eff_ub = M * d_ADD_val[base_i]

                        # If Effective UB > 0 (with some leeway)
                        if eff_ub > 1e-5:
                            print_log(f" [!] CRITICAL BIG-M LIMIT HIT")
                            print_log(f"     -> Organism: {org_ng} (Experiment {exp_idx})")
                            print_log(f"     -> Reaction: {rxn_ids[base_i]} ({direction})")
                            print_log(f"     -> Mu Value: {mu_val:.2f} | Big-M Limit: {M}")
                            print_log(f"     -> Effct UB: {eff_ub:.2f} (Base UB was {base_ub:.2f})")

        print_log(f" >> Problem value: {problem.value} ({problem.status})")
        print_log(" > Modifications:")

        N = len(self.model.reactions)
        for i, rxn in enumerate(self.model.reactions):
            rev_idx = N + i

            if self.cADD[i] == 1 and self.deltas_ADD[i].value is not None and self.deltas_ADD[i].value > 0.5:
                print_log(f"  [+] ADDED Database Reaction: {rxn.id}")
            if self.cREM[i] == 1 and self.deltas_REM[i].value is not None and self.deltas_REM[i].value > 0.5:
                print_log(f"  [-] REMOVED Model Reaction: {rxn.id}")
            if self.cREV[i] == 1 and self.deltas_REV[i].value is not None and self.deltas_REV[i].value > 0.5:
                print_log(f"  [<] REVERSED Model Reaction: {rxn.id}")

            if self.cADD[rev_idx] == 1 and self.deltas_ADD[rev_idx].value is not None and self.deltas_ADD[rev_idx].value > 0.5:
                print_log(f"  [+] ADDED Database Reaction (Reverse): rev_{rxn.id}")
            if self.cREV[rev_idx] == 1 and self.deltas_REV[rev_idx].value is not None and self.deltas_REV[rev_idx].value > 0.5:
                print_log(f"  [<] REVERSED Model Reaction: rev_{rxn.id}")
            if self.cREM[rev_idx] == 1 and self.deltas_REM[rev_idx].value is not None and self.deltas_REM[rev_idx].value > 0.5:
                print_log(f"  [-] REMOVED Model Reaction (Reverse bound): rev_{rxn.id}")

        for i_exp in range(len(self.experiments)):
            if self.deltas_EXP[i_exp].value is not None and self.deltas_EXP[i_exp].value > 0.5:
                print_log(f"  [x] EXPERIMENT ({i_exp}) EXCLUDED (Impossible to solve)")

        if return_string:
            return "\n".join(final)

    def evaluate_result(self) -> bool:
        """
        Parses the expected_result string and evaluates it against the solver's actual modifications.
        (This was used mainly for testing puroses)
        """
        if self.expected_result is None:
            return True

        # Parse Expected Modifications
        expected_mods = {"ADD": set(), "REM": set(), "REV": set(), "EXP": set()}
        if self.expected_result.strip():
            for part in self.expected_result.split(","):
                part = part.strip()
                if not part: continue
                try:
                    action, target = part.split(maxsplit=1)
                    expected_mods[action.upper()].add(target.strip())
                except ValueError:
                    log_info(f" [!] Invalid expected format: '{part}'. Example format: 'ADD OrgA_trans_Y'.")

        # Extract Actual Modifications
        actual_mods = {"ADD": set(), "REM": set(), "REV": set(), "EXP": set()}
        N = len(self.model.reactions)
        for i, rxn in enumerate(self.model.reactions):
            rev_idx = N + i

            # Forward direction modifications
            if self.cADD[i] == 1 and self.deltas_ADD[i].value is not None and self.deltas_ADD[i].value > 0.5:
                actual_mods["ADD"].add(rxn.id)
            if self.cREM[i] == 1 and self.deltas_REM[i].value is not None and self.deltas_REM[i].value > 0.5:
                actual_mods["REM"].add(rxn.id)
            if self.cREV[i] == 1 and self.deltas_REV[i].value is not None and self.deltas_REV[i].value > 0.5:
                actual_mods["REV"].add(f"rev_{rxn.id}")

            # Reverse direction modifications
            if self.cADD[rev_idx] == 1 and self.deltas_ADD[rev_idx].value is not None and self.deltas_ADD[rev_idx].value > 0.5:
                actual_mods["ADD"].add(f"rev_{rxn.id}")
            if self.cREV[rev_idx] == 1 and self.deltas_REV[rev_idx].value is not None and self.deltas_REV[rev_idx].value > 0.5:
                actual_mods["REV"].add(rxn.id)
            if self.cREM[rev_idx] == 1 and self.deltas_REM[rev_idx].value is not None and self.deltas_REM[rev_idx].value > 0.5:
                actual_mods["REM"].add(f"rev_{rxn.id}")

        for i_exp in range(len(self.experiments)):
            if self.deltas_EXP[i_exp].value is not None and self.deltas_EXP[i_exp].value > 0.5:
                actual_mods["EXP"].add(str(i_exp))

        # Compare and Report
        passed = True
        log_info("\n========= TEST EVALUATION =========")
        for action in ["ADD", "REM", "REV", "EXP"]:
            expected_set = expected_mods[action]
            actual_set = actual_mods[action]

            missing = expected_set - actual_set
            extra = actual_set - expected_set

            if missing or extra:
                passed = False
                if missing:
                    log_info(f" [FAIL] Expected {action} missing: {missing}")
                if extra:
                    log_info(f" [FAIL] Unexpected {action} found: {extra}")
            elif expected_set:
                log_info(f" [PASS] {action} perfectly matched: {expected_set}")

        if not any(expected_mods.values()) and not any(actual_mods.values()):
            log_info(" [PASS] 0 modifications expected and 0 modifications made.")

        log_info(f" >>> TEST RESULT: {'SUCCESS' if passed else 'FAILED'} <<<\n")
        return passed

    # DEBUG METHODS
    def get_active_fluxes(self, problem):
        """DEBUGGING: prints out all non-zero fluxes"""
        if problem.status == "optimal":
            for exp_i in range(len(self.experiments)):
                log_info(f"\n-- NON-ZERO FLUXES (Experiment {exp_i}) --")
                v_vals = self.v[exp_i].value
                N = len(self.model.reactions)
                for i, rxn in enumerate(self.model.reactions):
                    fwd_flux = v_vals[i]
                    rev_flux = v_vals[N + i]
                    # Filter out tiny floating-point artifacts
                    if fwd_flux > 1e-5:
                        log_info(f"{rxn.id} (FWD): {fwd_flux:.4f}")
                    if rev_flux > 1e-5:
                        log_info(f"{rxn.id} (REV): {rev_flux:.4f}")
            log_info("------------------------------------\n")

    def export_escher_fluxes(self, base_filename="escher_fluxes"):
        """
        DEBUGGING: Extracts the net fluxes (FWD - REV) for all reactions across ALL experiments,
        reconstructs their original base IDs, and exports them to separate CSV files for easier 
        visualization with the Escher software.
        """
        N = len(self.model.reactions)

        for exp_index in range(len(self.experiments)):
            v_vals = self.v[exp_index].value
            if v_vals is None:
                log_info(f" [!] Flux values are None for experiment {exp_index}. Skipping...")
                continue

            escher_data = []

            for i, rxn in enumerate(self.model.reactions):
                # Calculate net flux: Forward - Reverse
                fwd_flux = v_vals[i]
                rev_flux = v_vals[N + i]
                net_flux = fwd_flux - rev_flux

                # Reconstruct original IDs
                raw_id = rxn.id
                if raw_id.startswith("EX#"):
                    # Convert 'EX#h2o' -> 'EX_h2o'
                    clean_id = raw_id.replace("EX#", "EX_")
                elif "#" in raw_id:
                    # Convert 'e_coli_core#ATPM' -> 'ATPM'
                    # Also handles DB reactions like 'e_coli_core#DB#RXN1' -> 'RXN1'
                    clean_id = raw_id.split("#", 1)[-1]
                else:
                    clean_id = raw_id

                escher_data.append({
                    "reaction_id": clean_id,
                    "reaction_flux": net_flux
                })

            # Build DataFrame and save to CSV
            df_escher = pd.DataFrame(escher_data)
            filename = f"{base_filename}_exp_{exp_index}.csv"
            df_escher.to_csv(filename, index=False)

            log_info(f" [+] Successfully exported {len(df_escher)} net fluxes to '{filename}'.")

    # RUN METHOD
    def run(self):
        log_info(" >>> Compiling canonical GloCom model in Gurobi (Matrix Big-M) <<<")
        m = gp.Model("GloCom_Canonical")

        # Solver Parameters
        m.Params.OutputFlag = 1 if VERBOSE_SOLVER else 0
        m.Params.FeasibilityTol = MODEL_TOLERANCE
        m.Params.IntFeasTol = MODEL_TOLERANCE
        m.Params.MIPFocus = 1  # Focus on finding feasible integer solutions quickly
        m.Params.Presolve = 2  # Aggressive presolve for large genome-scale models
        m.Params.BarHomogeneous = 1  # suggested due to numerical issues with the community experiment
        # The below parameters were tried and for the limited experiments run with them they did not seem to help
        # m.Params.NumericFocus = 3    # Triggers Quad-Precision and careful basis factorization
        # m.Params.ScaleFlag = 2       # Aggressive matrix scaling
        # m.Params.MarkowitzTol = 0.5  # Prevents numerical instability in the Simplex pivot
        # m.Params.IntegralityFocus = 1

        exp_len = len(self.experiments)
        n_rxns = self.S.shape[1]
        n_mets = self.S.shape[0]

        # Primal Variables
        v = m.addMVar((exp_len, n_rxns), lb=0.0, ub=M, name="v")

        # Binary decision variables
        d_REM = m.addMVar(n_rxns, vtype=GRB.BINARY, name="d_REM")
        d_ADD = m.addMVar(n_rxns, vtype=GRB.BINARY, name="d_ADD")
        d_REV = m.addMVar(n_rxns, vtype=GRB.BINARY, name="d_REV")
        d_EXP = m.addMVar(exp_len, vtype=GRB.BINARY, name="d_EXP")

        # Dual and proxy variables (If Non-Growth events exist)
        if self.n_ng > 0:
            d_lmbda = m.addMVar((self.n_ng, n_mets), lb=-M, ub=M, name="d_lmbda")
            d_mu = m.addMVar((self.n_ng, n_rxns), lb=0.0, ub=M, name="d_mu")
            d_xi = m.addMVar((self.n_ng, n_rxns), lb=0.0, ub=M, name="d_xi")

            # Linearization Proxy Variables (z >= 0)
            z_REM = m.addMVar((self.n_ng, n_rxns), lb=0.0, ub=M, name="z_REM")
            z_REV = m.addMVar((self.n_ng, n_rxns), lb=0.0, ub=M, name="z_REV")
            z_ADD = m.addMVar((self.n_ng, n_rxns), lb=0.0, ub=M, name="z_ADD")
            z_EXP_atpm = m.addMVar((self.n_ng, n_rxns), lb=0.0, ub=M, name="z_EXP_atpm")
            z_EXP_bio = m.addMVar((self.n_ng, self.n_organisms), lb=0.0, ub=M, name="z_EXP_bio")

        S_sparse = self.S.tocsc()  # transform the format for optimization purposes

        # Primal Constraints
        for i_exp, experiment in enumerate(self.experiments):
            ubs = self.ubs[i_exp]

            # Mass balance and general upper bounds
            m.addConstr(S_sparse @ v[i_exp, :] == 0, name=f"mass_balance_exp{i_exp}")
            m.addConstr(v[i_exp, :] <= ubs, name=f"upper_bounds_exp{i_exp}")

            # Target network modifications coupling
            # REM: v_j <= u_j * (1 - delta_REM_j)
            m.addConstr(
                self.cREM * v[i_exp, :] <= self.cREM * ubs * (1.0 - d_REM),
                name=f"rem_coupling_exp{i_exp}"
            )
            # REV: v_j <= M * delta_REV_j
            m.addConstr(
                self.cREV * v[i_exp, :] <= self.cREV * M * d_REV,
                name=f"rev_coupling_exp{i_exp}"
            )
            # ADD: v_j <= M * delta_ADD_j
            m.addConstr(
                self.cADD * v[i_exp, :] <= self.cADD * M * d_ADD,
                name=f"add_coupling_exp{i_exp}"
            )

            # Growth and ATPM Enforcement
            for org_name in self.organisms:
                if org_name in self.org_atpm_indices:
                    if ALWAYS_ENFORCE_ATPM or experiment.growth.get(org_name, False):
                        atpm_idx = self.org_atpm_indices[org_name]
                        m.addConstr(
                            v[i_exp, atpm_idx] + M * d_EXP[i_exp] >= MINIMAL_ATPM_FLUX,
                            name=f"atpm_enforce_{org_name}_exp{i_exp}"
                        )

            for org_name, grows in experiment.growth.items():
                if grows and org_name in self.org_biomass_indices:
                    bio_idx = self.org_biomass_indices[org_name]
                    m.addConstr(
                        v[i_exp, bio_idx] + M * d_EXP[i_exp] >= GROWTH_THRESHOLD,
                        name=f"growth_enforce_{org_name}_exp{i_exp}"
                    )
                elif not grows and org_name != self.target_org:
                    for idx in self.org_rxns_idx[org_name]:
                        m.addConstr(v[i_exp, idx] == 0.0)

        # Inner Problem: Dual formulation and Linearization
        if self.n_ng > 0:
            for ng_idx, (i_exp, org_ng) in enumerate(self.ng_events):
                exp = self.experiments[i_exp]
                ubs = self.ubs[i_exp]

                cBIO_k = np.zeros(n_rxns)
                if org_ng in self.org_biomass_indices:
                    cBIO_k[self.org_biomass_indices[org_ng]] = 1.0

                # Dual Constraint: S^T * lambda + mu - xi >= c_bio
                m.addConstr(
                    S_sparse.T @ d_lmbda[ng_idx, :] + d_mu[ng_idx, :] - d_xi[ng_idx, :] >= cBIO_k,
                    name=f"dual_feasibility_{org_ng}_exp{i_exp}"
                )

                # Big-M Linearizations for Proxy Variables
                # REM Linearization: z_REM = mu * delta_REM
                m.addConstr(z_REM[ng_idx, :] <= M * d_REM)
                m.addConstr(z_REM[ng_idx, :] <= d_mu[ng_idx, :])
                m.addConstr(z_REM[ng_idx, :] >= d_mu[ng_idx, :] - M * (1.0 - d_REM))

                # REV Linearization: z_REV = mu * delta_REV
                m.addConstr(z_REV[ng_idx, :] <= M * d_REV)
                m.addConstr(z_REV[ng_idx, :] <= d_mu[ng_idx, :])
                m.addConstr(z_REV[ng_idx, :] >= d_mu[ng_idx, :] - M * (1.0 - d_REV))

                # ADD Linearization: z_ADD = mu * delta_ADD
                m.addConstr(z_ADD[ng_idx, :] <= M * d_ADD)
                m.addConstr(z_ADD[ng_idx, :] <= d_mu[ng_idx, :])
                m.addConstr(z_ADD[ng_idx, :] >= d_mu[ng_idx, :] - M * (1.0 - d_ADD))

                # ATPM Exclusion Linearization: z_EXP_atpm = xi * delta_EXP
                m.addConstr(z_EXP_atpm[ng_idx, :] <= M * d_EXP[i_exp])
                m.addConstr(z_EXP_atpm[ng_idx, :] <= d_xi[ng_idx, :])
                m.addConstr(z_EXP_atpm[ng_idx, :] >= d_xi[ng_idx, :] - M * (1.0 - d_EXP[i_exp]))

                # Dual Objective assembly
                # Static terms (non-modifiable reactions)
                c_STATIC = np.ones(n_rxns) - self.cREM - self.cREV - self.cADD
                term_default = (c_STATIC * ubs) @ d_mu[ng_idx, :]

                # Target modifications terms
                term_rem = (self.cREM * ubs) @ d_mu[ng_idx, :] - (self.cREM * ubs) @ z_REM[ng_idx, :]
                term_rev = (M * self.cREV) @ z_REV[ng_idx, :]
                term_add = (M * self.cADD) @ z_ADD[ng_idx, :]

                dual_UB = term_default + term_rem + term_rev + term_add

                # Lower Bound Deductions (ATPM and Growth enforcements)
                atpm_lb_array = np.zeros(n_rxns)
                cATP_growing = np.zeros(n_rxns)
                for org_name in self.organisms:
                    if org_name in self.org_atpm_indices:
                        if ALWAYS_ENFORCE_ATPM or exp.growth.get(org_name, False):
                            atpm_idx = self.org_atpm_indices[org_name]
                            atpm_lb_array[atpm_idx] = MINIMAL_ATPM_FLUX
                            cATP_growing[atpm_idx] = 1.0

                xi_atpm_term = atpm_lb_array @ d_xi[ng_idx, :]
                z_exp_atpm_term = (cATP_growing * M) @ z_EXP_atpm[ng_idx, :]
                dual_LB = xi_atpm_term - z_exp_atpm_term

                # Deductions for other growing organisms in the community
                for org_idx, org_other in enumerate(self.organisms):
                    if exp.growth.get(org_other, False) and org_other in self.org_biomass_indices:
                        bio_idx = self.org_biomass_indices[org_other]
                        xi_bio = d_xi[ng_idx, bio_idx]
                        z_bio = z_EXP_bio[ng_idx, org_idx]

                        m.addConstr(z_bio <= M * d_EXP[i_exp])
                        m.addConstr(z_bio <= xi_bio)
                        m.addConstr(z_bio >= xi_bio - M * (1.0 - d_EXP[i_exp]))

                        dual_LB += (xi_bio * GROWTH_THRESHOLD) - (z_bio * M)

                dual_obj = dual_UB - dual_LB

                # Verification Constraint
                thresh = GROWTH_THRESHOLD if SINGLE_GROWTH_THRESHOLD else NON_GROWTH_THRESHOLD
                m.addConstr(
                    dual_obj <= thresh + M * d_EXP[i_exp],
                    name=f"dual_verification_bound_{org_ng}_exp{i_exp}"
                )

                # Strong Duality enforcement (optional)
                if USE_STRONG_DUALITY:
                    v_bio_target = v[i_exp, self.org_biomass_indices[org_ng]] if org_ng in self.org_biomass_indices else 0.0
                    m.addConstr(
                        dual_obj == v_bio_target,
                        name=f"strong_duality_{org_ng}_exp{i_exp}"
                    )

        # Outer objective function
        Wadd_masked = self.Wadd * self.cADD
        Wrem_masked = self.Wrem * self.cREM
        Wrev_masked = self.Wrev * self.cREV

        obj = Wadd_masked @ d_ADD + Wrem_masked @ d_REM + Wrev_masked @ d_REV + self.Wexp @ d_EXP

        if USE_REGULARIZATION and self.n_ng > 0:
            obj += 1e-8 * d_mu.sum()

        m.setObjective(obj, GRB.MINIMIZE)

        if EXPORT_PROBLEM:
            log_info(" [+] Exporting compiled model to 'glocom_debug.mps' and 'glocom_debug.lp'...")
            m.write(f"glocom_debug.mps")  # Machine-readable exact matrix
            m.write(f"glocom_debug.lp")   # Human-readable algebraic equations
            # m.write(f"{CODE}_glocom_debug.mps")  # Machine-readable exact matrix
            # m.write(f"{CODE}_glocom_debug.lp")   # Human-readable algebraic equations
            # CODE came from console command

        # solve
        m.optimize()

        # parse solution
        class ProblemMock:
            def __init__(self, status, value):
                self.status = status
                self.value = value

        if m.Status in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
            self.deltas_ADD = MockVar(d_ADD.X)
            self.deltas_REM = MockVar(d_REM.X)
            self.deltas_REV = MockVar(d_REV.X)
            self.deltas_EXP = MockVar(d_EXP.X)

            v_reshaped = []
            for i in range(exp_len):
                v_reshaped.append(MockVar([v[i, j].X for j in range(n_rxns)]))
            self.v = v_reshaped

            if self.n_ng > 0:
                self.d_mu = MockVar(np.array([[d_mu[ng_idx, j].X for j in range(n_rxns)] for ng_idx in range(self.n_ng)]))

            if m.Status == GRB.SUBOPTIMAL: log_info(" >> WARNING: problem solved suboptimally <<")
            prob_mock = ProblemMock("optimal", m.ObjVal)
            # If you want to export the fluxes to a csv format or print them to the console uncomment below
            # self.export_escher_fluxes()
            # self.get_active_fluxes(prob_mock)
            self.print_results(prob_mock)
            return prob_mock, self.evaluate_result()
        elif m.Status == GRB.INFEASIBLE:
            log_info(" [!] Problem is INFEASIBLE. Computing IIS (Irreducible Inconsistent Subsystem)...")
            try:
                m.computeIIS()
                m.write(f"glocom_infeasible.ilp")
                # m.write(f"{CODE}_glocom_infeasible.ilp")
                log_info(" [+] IIS written to 'glocom_infeasible.ilp'. Open this file to see the exact conflicting constraints.")
            except gp.GurobiError as e:
                log_info(f" [!] Failed to compute IIS: {e}")
                
            prob_mock = ProblemMock("infeasible", None)
            self.print_results(prob_mock)
            return prob_mock, False
        else:
            prob_mock = ProblemMock("infeasible", None)
            self.print_results(prob_mock)
            return prob_mock, False
