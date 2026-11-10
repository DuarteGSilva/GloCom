# GloCom: Multilevel Microbial Community Model Refinement

**GloCom** is a Mixed-Integer Linear Programming (MILP) algorithm designed to refine genome-scale metabolic models (GSMMs) within microbial communities (or not). By leveraging bi-level optimization, linear programming duality, and a shared extracellular space, GloCom reconciles computational model predictions with observed growth and non-growth experimental data.

Unlike single-organism only, gap-filling tools, GloCom accounts for interspecies metabolic interactions, such as cross-feeding and resource competition, while restricting network modifications (reaction additions, removals, and reversals) to a specified target organism to maintain computational scalability.

## Key Features

- **Multilevel Integration:** Simultaneously processes growth ($\ge T_{\mathcal{G}}$) and non-growth ($\le T_{\mathcal{NG}}$) phenotypes across single-strain and community-level contexts.

- **Shared Compartment Modeling:** Simulates ecological interactions by linking individual species models to a unified extracellular environment.

- **Target-Organism Optimization:** Isolates network edits to a user-designated strain ($k^*$) within the community matrix.

- **BLAST-Based Biological Scoring:** Uses normalized DIAMOND/BLAST sequence alignment scores to weight reaction additions and removals based on genomic evidence.

- **Global Evaluation:** Evaluates all dataset conditions concurrently by harnessing the power of duality and MILPs, rather than relying on iterative, local "greedy" heuristics.

## Theoretical Framework

GloCom formulates an outer minimization problem subject to global community mass balance ($S_{com} \cdot v = 0$), thermodynamic reaction bounds, and dual verification constraints derived from an inner biomass maximization problem:

### Scoring Functions

Reaction modification costs are weighted according to normalized sequence similarity BLAST scores ($x$) generated against a reference database (e.g., BiGG):

**Reaction Addition Cost:**

$$
f_{add}(x) = 1 + \frac{1}{1 + x}
$$

**Reaction Removal Cost:**

$$
f_{rem}(x) = 3 + \frac{x}{3}
$$

## Prerequisites & Dependencies

### Hardware & Software Requirements

- **Python:** `3.9` or higher (version 3.14+ recommended)
- **MILP Solver:** [Gurobi Optimizer](https://www.gurobi.com/) (version 9.5+ recommended) with a valid academic or commercial license.
- **Sequence Alignment:** [DIAMOND](https://github.com/bbuchfink/diamond) (for generating genomic similarity scores).

### Python Packages

```bash
pip install cobra cvxpy numpy scipy pandas gurobipy carveme
```

## Directory Structure

To run GloCom, your workspace should be organized as follows:

```text
glocom/
├── scores/                 # Pre-made DIAMOND alignment score files (.tsv)
│   ├── organism1_name.tsv
│   └── organism2_name.tsv
├── models/                 # Pre-made single organism SBML (.xml) models
│   ├── organism1_name.xml
│   └── organism2_name.xml
├── universal_model.json    # Universal database model (e.g., BiGG)
├── glocom.py               # Main entry point to run GloCom
└── README.md
```

## Input Data Requirements

> **⚠️ IMPORTANT: Strict Naming Conventions**
> *   **No `#` in filenames or organism names:** The `#` symbol is strictly reserved as an internal delimiter by GloCom to map reactions to specific organisms (e.g., `OrgA#EX_h2o`). Using it in file or organism names will break the parser.
> *   **Exact Matching:** The `.xml` model file and the `.tsv` DIAMOND score file for a given organism **must** share the exact same base name.

1. **Genome-Scale Models (`/models`):** SBML format (`.xml`) for each individual organism/community member.
2. **Universal Reaction Database (`universal_model.json`):** BiGG database reference file used for candidate reaction additions.
3. **DIAMOND Alignment Scores (`/scores`):** Tab-separated files resulting from running protein sequences against the BiGG database via DIAMOND. 
4. **Experimental Configuration:** data defining:
   * Medium compositions (exchange flux lower/upper bounds).
   * Observed phenotype per strain (Growth = `1`, Non-Growth = `0`) per condition.

## Configuration & Thresholds

Several global configuration variables and default identifiers can be tweaked directly at the top of the `glocom.py` script:

### Model Compatibility (Identifiers)
If you are bringing your own models, ensure the following identifiers match (or update them in the script):
*   **Biomass ID:** Defaults to `"Growth"` (CarveMe default).
*   **ATPM ID:** Defaults to `"ATPM"`.
*   **Extracellular Space:** Defaults to `"C_e"`.

### Algorithm Thresholds
*   **Growth Threshold:** `0.05`
*   **Non-Growth Threshold:** `0.01`
*   **Minimal ATPM Flux:** `8.39`

### Core Execution Flags
*   `USE_STRONG_DUALITY` *(Default: `True`)*: Enforces strong duality (`dual_obj == v_bio_target`) in the inner problem, rather than just weak duality.
*   `USE_REGULARIZATION` *(Default: `False`)*: If set to `True`, applies a micro-penalty ($10^{-8}$) via Tikhonov regularization to handle unbounded dual variables.
*   `ALWAYS_ENFORCE_ATPM` *(Default: `False`)*: Determines if ATPM flux is enforced even for non-growing organisms.

## Solver Parameter Recommendations

Due to the numerical sensitivity of bi-level dual formulations in large MILP problems, the following Gurobi settings are pre-configured:

| Parameter | Recommended Value | Description |
| :--- | :--- | :--- |
| `FeasibilityTol` | `1e-8` | Tightens primal feasibility tolerance. |
| `IntFeasTol` | `1e-8` | Tightens integer feasibility tolerance. |
| `MIPFocus` | `1` | Prioritizes finding feasible solutions over proving optimality. |
| `Presolve` | `2` | Aggressive presolve to prune search space and reduce memory footprint. |
| `BarHomogeneous` | `1` | Improves numerical stability for interior point algorithms. |

## Test Example

A tiny AI-generated example is provided. It runs in seconds, needs no real genomes, and has a known answer, so you can check that your Gurobi/CarveMe/cobra installation and GloCom behave as expected before running real data. This example was verified and runs as intended given the instructions are followed correctly.

### What is available for this test

```text
glocom/
├── other files ...
├── run.py                  # Builds the experiments and runs GloCom    <-- run this file
├── universal_model.json    # Test reaction database (4 reactions)
└── models/OrgA.xml         # Test single-organism SBML model
```

A DIAMOND file containing the scores is not provided, so standard constant scores are used instead of the more complex scoring functions.

### The test organism

`OrgA` has a simplified network (shared extracellular metabolites end in `_e`, internal ones in `_c`):

| Reaction | What it does | Role in the example |
| :--- | :--- | :--- |
| `GLCt` | `glc_e -> glc_c` | Glucose uptake |
| `GLYC` | `glc_c + 2 adp_c -> 2 atp_c + 2 pyr_c` | Lumped glycolysis; makes ATP and biomass precursor |
| `XYLt` | `xyl_e -> xyl_c` | Xylose uptake, but `xyl_c` is a **dead end** |
| `ACEG` | `ace_e -> glc_c` | **Spurious** reaction (wrongly annotated); lets OrgA eat acetate |
| `ATPM` | `atp_c -> adp_c` | ATP maintenance (GloCom enforces a minimum flux of 8.39 when OrgA must grow) |
| `Growth` | `4 pyr_c + atp_c -> adp_c` | Biomass reaction |

The test database (`universal_model.json`) offers four candidate additions: `XYL2GLC` (`xyl_c -> glc_c`, the correct fix), `XYL2XUL` + `XUL2GLC` (a two-step bypass that also works but costs two additions), and `GLC2FRU` (an irrelevant decoy).

### The made-up experiments

| Experiment | Medium (max uptake) | Observed OrgA phenotype | Unmodified model | Why it conflicts |
| :--- | :--- | :--- | :--- | :--- |
| 0 | glucose (10) | Growth | Grows | No conflict; it protects the glucose pathway from being removed |
| 1 | xylose (10) | Growth | No growth | Xylose is a dead end, so a reaction must be **added** |
| 2 | acetate (10) | No growth | Grows (through `ACEG`) | The spurious reaction must be **removed** |

### How to run

```bash
python run.py
```

Logs go to the console and to `glocom.log`. Because `EXPORT_PROBLEM = True`, `glocom_debug.mps` and `glocom_debug.lp` are also written.

### Expected result

GloCom should suggest exactly two changes and nothing else:

```text
>> Problem value: 7.0 (optimal)
> Modifications:
 [+] ADDED Database Reaction: OrgA#DB#XYL2GLC
 [-] REMOVED Model Reaction: OrgA#ACEG
...
[PASS] ADD perfectly matched: {'OrgA#DB#XYL2GLC'}
[PASS] REM perfectly matched: {'OrgA#ACEG'}
>>> TEST RESULT: SUCCESS <<<
```

`run.py` exits with code 0 on success and 1 otherwise. No experiment should be excluded, no reaction reversed.

The objective is 7.0 because, with the default weights, one database addition costs 4 (`W_ADD_RX`) and one removal costs 3 (`W_REM_RX`). This is the unique cheapest solution: the two-step bypass would cost 8 for the additions alone, and excluding an experiment costs 200.
