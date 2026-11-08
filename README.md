# TrIM: Transformed Iterative Mondrian

TrIM is a Mondrian forest regressor that learns a linear transform of its inputs.
It estimates the expected gradient outer product (EGOP) `Ĥ` of the target function
with finite differences, then refits the forest on the transformed data

```
X @ Ĥ / two_one_norm(Ĥ)
```

This makes the forest's axis-aligned splits follow the directions the target
actually depends on. The code here runs simulation experiments that compare TrIM
with an untransformed Mondrian forest, the true-EGOP transform, eigen-projections
of `Ĥ`, and column subset selection (CSS). It also includes random-forest and
k-NN versions of the downstream learner.

## Setup

Requires Python 3.10.

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# conda
conda create -n trim python=3.10
conda activate trim
pip install -r requirements.txt

# or venv
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running experiments

Run every script from the repository root. The active scripts add `src/` to the
import path themselves and write their outputs to `results/<experiment>/`, which
they create if needed.

```bash
# quick check that everything runs
python -u experiments/orth_transform_main.py --smoke

# full run, logging to a file
mkdir -p results/orth_transform
python -u experiments/orth_transform_main.py > results/orth_transform/run.log 2>&1

# options: dimensions, target weights, output folder name
python -u experiments/orth_transform_main.py --dims 10 --weights 1,0.5 --out orth_transform_w1_05
```

| script | question | output |
|---|---|---|
| `experiments/orth_transform_main.py` | Does an orthogonal-column transform `X U Λ` beat TrIM's `X Ĥ`? | `results/orth_transform/` |
| `experiments/full_pipeline.py` | Full comparison: no transform, true H, TrIM, Weighted Mondrian, CSS k-sweep, eigen-projection. Set `ROTATION_MODE` for aligned or rotated data. | `results/full_pipeline/` |
| `experiments/mse_compare.py` | Mondrian forest vs. random forest test MSE, no transform. | `results/mse_compare/` |

## Repository layout

```
src/                  shared library code (imported by experiments, not run directly)
  Mondrian_RF/        MondrianForestRegressor, TrIM, WeightedMondrianForestRegressor
  RandomForest/       scikit-learn random-forest versions: RFRegressor, RFTrIM
  kNN_Euclidean/      k-NN downstream learner
  utils.py            SimulatedData (synthetic poly/max targets, aligned or rotated), angle metrics
  eigen_utils.py      eigendecomposition of H, top-k eigen-projection
  css_utils_*.py      column subset selection variants
experiments/          experiment drivers
  archive/            older drivers, kept for reference
```

### `src/` details

| file | contents |
|---|---|
| `Mondrian_RF/Mondrian_forest.py` | `MondrianForestRegressor`, `TrIM`, `WeightedMondrianForestRegressor` |
| `Mondrian_RF/utils.py` | Mondrian training, `evaluate_all_lifetimes` (lifetime sweep → test MSE), `two_one_norm` |
| `RandomForest/` | `RFRegressor`, `RFTrIM`. Its `utils.evaluate_all_lifetimes` sweeps `max_depth` of a random forest rather than Mondrian lifetime. |
| `kNN_Euclidean/` | k-NN version of the downstream learner |
| `utils.py` | `SimulatedData`, `get_angle_distance`, `Simulation*` helpers |
| `eigen_utils.py` | eigendecomposition of H, top-k eigen-projection |
| `css_utils_trueH.py` | full comparison: no transform / true H / TrIM / Weighted Mondrian / Weighted CSS / eigen |
| `css_utils_direct.py` | "direct" CSS in TrIM coordinates |
| `css_utils_max.py` | top-diagonal CSS for the aligned case (currently broken: imports a `TrIM` class from `RandomForest.RandomForest` that no longer exists) |

### Archived experiments

The scripts in `experiments/archive/` were written against an older version of
`Mondrian_forest.py` that had a backend switch (`backend=` / `EGOP_BACKEND`).
Most of them no longer run without changes. To try one:

```bash
PYTHONPATH=src python experiments/archive/<file>.py
```

| script | what it did |
|---|---|
| `0913_main.py` | EGOP angle error vs. n at fixed d |
| `0919_exp.py` | iterative sweep plus oracle floor (Mondrian) |
| `0923_exp.py` | same, random-forest backend (`RFTrIM`) |
| `high_dimension_main.py` | d-scan, aligned data, RF EGOP + RF downstream |
| `no_rotation_main.py`, `no_rotation_main_rf.py` | aligned-data version of the full pipeline |
| `rotation_main_kNN.py` | full pipeline, k-NN downstream |
| `rotation_main_rf.py` | rotated data, direct CSS, RF EGOP + RF downstream |
| `rotation_main_directcssrf.py` | direct CSS on estimated vs. true H, RF, multiple seeds |
| `rotation_main_mondrian_egop_dsweep.py` | Mondrian EGOP, iteration sweep |
| `rotation_main_rf_egop_dsweep.py` | RF EGOP, step-size sweep |
| `step_size_diag.py` | finite-difference step-size diagnostic |

## Related code

Recursive Feature Machines (RFM), which also use the EGOP to learn features, served
as a reference for this project. The scripts here don't need RFM. Its code is in the
authors' repository: https://github.com/aradha/recursive_feature_machines
