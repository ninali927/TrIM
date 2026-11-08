# TrIM project

TrIM (Transformed Iterative Mondrian): a Mondrian forest that estimates the EGOP Â
by finite differences, then refits on transformed data `X @ Â / two_one_norm(Â)`.

**Always run scripts from this folder (the project root)**, e.g.
`python -u experiments/orth_transform_main.py > results/orth_transform/run.log 2>&1`.
Active scripts add `src/` to the import path and write under `results/` themselves.

## Layout

```
src/            shared code (imported by experiments, not run directly)
experiments/    scripts that run experiments
  archive/      old drivers, kept for reference (several no longer run, see below)
results/        one folder per experiment; results/archive/ = all old outputs
notes/          LaTeX notes (computational complexity)
papers/         related papers (PDF)
external/       recursive_feature_machines: the RFM authors' repo (own git, untouched)
```

## src/

| file | what it is |
|---|---|
| `Mondrian_RF/Mondrian_forest.py` | `MondrianForestRegressor`, `TrIM`, `WeightedMondrianForestRegressor` |
| `Mondrian_RF/utils.py` | Mondrian training, `evaluate_all_lifetimes` (lifetime sweep → test MSE), `two_one_norm` |
| `RandomForest/` | sklearn-RF version: `RFRegressor`, `RFTrIM`; its `utils.evaluate_all_lifetimes` sweeps **max_depth** with an RF, **not** Mondrian |
| `kNN_Euclidean/` | k-NN version of the downstream learner |
| `utils.py` | `SimulatedData` (synthetic data; `poly`/`max`, aligned or `"orth"` rotation), `get_angle_distance`, `Simulation*` helpers |
| `eigen_utils.py` | eigendecomposition of H, top-k eigen projection |
| `css_utils_trueH.py` | full comparison: no transform / true H / TrIM / Weighted Mondrian / Weighted CSS / eigen |
| `css_utils_direct.py` | "Direct" CSS in TrIM coordinates |
| `css_utils_max.py` | top-diagonal CSS for the aligned case (**broken**: imports `TrIM` from `RandomForest.RandomForest`, which no longer exists) |

## experiments/ (active)

| script | question | output |
|---|---|---|
| `orth_transform_main.py` | Does an orthogonal-column transform `X U Λ` beat TrIM's `X Â`? (professor, Nov 2026) | `results/orth_transform/` |
| `full_pipeline.py` (was `rotation_main.py`) | Full comparison: no transform, true H, TrIM, Weighted Mondrian, CSS k-sweep, eigen. Set `ROTATION_MODE` for aligned/rotated | `results/full_pipeline/` |
| `mse_compare.py` | Mondrian vs random forest MSE with no transform | `results/mse_compare/` |

## experiments/archive/

| script | what it was | runs? |
|---|---|---|
| `0913_main.py` | Exp B: EGOP angle error vs n at fixed d | no (`EGOP_BACKEND`) |
| `0919_exp.py` | Exp B full: iterative sweep + oracle floor (Mondrian) | no (`backend=`) |
| `0923_exp.py` | Exp B, random-forest backend (`RFTrIM`) | probably |
| `high_dimension_main.py` | d-scan, aligned, RF EGOP + RF downstream | no (`backend=`) |
| `no_rotation_main.py` | aligned version of full pipeline | no (`css_utils_max`) |
| `no_rotation_main_rf.py` | aligned, RF downstream | no |
| `rotation_main_kNN.py` | full pipeline, k-NN downstream | maybe |
| `rotation_main_rf.py` | rotated, Direct CSS, RF EGOP + RF downstream | no (`backend=`) |
| `rotation_main_directcssrf.py` | Direct CSS on estimated vs true H, RF, seeds | no (`backend=`) |
| `rotation_main_mondrian_egop_dsweep.py` | Mondrian EGOP, iteration sweep | no (`EGOP_BACKEND`) |
| `rotation_main_rf_egop_dsweep.py` | RF EGOP, step-size sweep | no (`EGOP_BACKEND`) |
| `step_size_diag.py` | finite-difference step-size diagnostic | ? |

To run an archived script: `PYTHONPATH=src python experiments/archive/<file>.py` (from the project root).
"Broken" scripts were written against an older `Mondrian_forest.py` that had an RF backend switch.

## results/archive/

| folder | contents |
|---|---|
| `2026-05-06_css_logs/` | May–Jun CSS k-sweep logs (rotated/aligned, kNN, direct) |
| `2026-06_rf_css/` | Jun 29–Jul 1 RF and EGOP-RF CSS runs, bootstrap/max_features ablations, high-d |
| `2026-07_mondrian_css_eigen/` | Jul 10 full pipeline figures (CSS + eigen k-sweep, n=3200/6400) |
| `2026-07_egop_step_sweep/` | Jul 21 EGOP iteration/step sweeps (Mondrian & RF) |
| `2026-09_expB_mondrian/` | Sep Exp B (Mondrian): angle vs n, d; logs + csv/png |
| `2026-09_expB_rf/` | Sep Exp B (RF backend) |
| `2026-09_mse_compare/` | Sep Mondrian-vs-RF MSE |
