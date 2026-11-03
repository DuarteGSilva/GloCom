# GloCom: Multilevel Microbial Community Model Refinement

**GloCom** is a Mixed-Integer Linear Programming (MILP) algorithm designed to refine genome-scale metabolic models (GSMMs) within microbial communities (or not). By leveraging bi-level optimization, linear programming duality, and a shared extracellular space, GloCom reconciles computational model predictions with observed growth and non-growth experimental data.

Unlike single-organism only, gap-filling tools, GloCom accounts for interspecies metabolic interactions, such as cross-feeding and resource competition, while restricting network modifications (reaction additions, removals, and reversals) to a specified target organism to maintain computational scalability.

## Key Features

* **Multilevel Integration:** Simultaneously processes growth ($\ge T_{\mathcal{G}}$) and non-growth ($\le T_{\mathcal{NG}}$) phenotypes across single-strain and community-level contexts.

* **Shared Compartment Modeling:** Simulates ecological interactions by linking individual species models to a unified extracellular environment.

* **Target-Organism Optimization:** Isolates network edits to a user-designated strain ($k^*$) within the community matrix.

* **BLAST-Based Biological Scoring:** Uses normalized DIAMOND/BLAST sequence alignment scores to weight reaction additions and removals based on genomic evidence.

* **Global Evaluation:** Evaluates all dataset conditions concurrently by harnessing the power of duality and MILPs, rather than relying on iterative, local "greedy" heuristics.

## Theoretical Framework

GloCom formulates an outer minimization problem subject to global community mass balance ($S_{com} \cdot v = 0$), thermodynamic reaction bounds, and dual verification constraints derived from an inner biomass maximization problem:

### Scoring Functions

Reaction modification costs are weighted according to normalized sequence similarity BLAST scores ($x$) generated against a reference database (e.g., BiGG):

* **Reaction Addition Cost:**

  $$
  f_{\text{add}}(x) = 1 + \frac{1}{1 + x}
  $$

* **Reaction Removal Cost:**

  $$
  f_{\text{rem}}(x) = 3 + \frac{x}{3}
  $$

## Prerequisites & Dependencies

### Hardware & Software Requirements

* **Python:** `3.9` or higher (version 3.14+ recommended)

* **MILP Solver:** [Gurobi Optimizer](https://www.gurobi.com/) (version 9.5+ recommended) with a valid academic or commercial license.

* **Sequence Alignment:** [DIAMOND](https://github.com/bbuchfink/diamond) (for generating genomic similarity scores).

### Python Packages

```
pip install gurobipy cobra pandas numpy
```

## Directory Structure

To run GloCom, your workspace should be organized as follows:

```
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

1. **Genome-Scale Models (`/models`):** SBML format (`.xml`) for each individual organism/community member.

2. **Universal Reaction Database (`universal_model.json`):** BiGG database reference file used for candidate reaction additions.

3. **DIAMOND Alignment Scores (`/scores`):** Tab-separated files resulting from running protein sequences against the BiGG database via DIAMOND. Filenames must match model file names.

4. **Experimental Configuration:** data  defining:

   * Medium compositions (exchange flux lower/upper bounds).

   * Observed phenotype per strain (Growth = `1`, Non-Growth = `0`) per condition.

## Solver Parameter Recommendations

Due to the numerical sensitivity of bi-level dual formulations in large MILP problems, the following Gurobi settings are pre-configured:

| Parameter | Recommended Value | Description | 
| ----- | ----- | ----- | 
| `FeasibilityTol` | `1e-8` | Tightens primal feasibility tolerance. | 
| `IntFeasTol` | `1e-8` | Tightens integer feasibility tolerance. | 
| `MIPFocus` | `1` | Prioritizes finding feasible solutions over proving optimality. | 
| `Presolve` | `2` | Aggressive presolve to prune search space and reduce memory footprint. | 
| `BarHomogeneous` | `1` | Improves numerical stability for interior point algorithms. | 
