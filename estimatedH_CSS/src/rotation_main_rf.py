"""
rotation_main_rf.py
===================

Rotated-case experiment with Direct CSS and a full random-forest pipeline
(RF EGOP + RF downstream). Mirror of rotation_main_kNN.py, with three changes:

  ROTATION_MODE = "orth"                          -> rotated case
  from css_utils_direct import ...                -> Direct TrIM-coordinate CSS
  EGOP_BACKEND forced to "rf" (below)             -> RF EGOP

Why Direct CSS and not Weighted CSS here: Weighted CSS adds an internal rotation
(QR re-orthogonalization + in-subspace eigendecomposition). The downstream RF is
axis-aligned and therefore rotation-sensitive, so a WCSS rotation on top of the
already-rotated data would confound "did CSS pick the right columns" with "WCSS's
construction rotation". Direct CSS keeps the TrIM coordinate system (no extra
rotation), so it pairs cleanly with the RF.

Both stages are random forests:
  - downstream: css_utils_direct imports evaluate_all_lifetimes from
    RandomForest.utils (sweeps max_depth).
  - EGOP: this script sets Mondrian_RF.Mondrian_forest.EGOP_BACKEND = "rf" before
    any TrIM is constructed, so the finite-difference H estimate probes a random
    forest. (TrIM is still re-exported from Mondrian_RF; only the probed model
    changes.)

Expected result (consistency check, not a new claim): in the rotated case the
active subspace is a combination of many original coordinates, which axis-aligned
column selection cannot span at small k. CSS should therefore NOT beat the
references here -- unlike the aligned case. Confirming this under RF shows the
learner swap changed only speed, not the CSS science.
"""

import os
import numpy as np
from jax import vmap

# --- Force RF EGOP for the whole run, before anything constructs a TrIM. ------
import Mondrian_RF.Mondrian_forest as _mf
_mf.EGOP_BACKEND = "rf"

from utils import SimulatedData, SimulationAligned, get_angle_distance
from css_utils_direct import evaluation_comparison_with_direct_css
from RandomForest.utils import MAX_DEPTH_GRID, BOOTSTRAP, MAX_FEATURES


# =============================================================
# Settings
# =============================================================
DIM_IN       = 30
ACTIVE       = 2
LIFETIME     = 5
CSS_K        = [4, 15, 30]   # rotated-case sweep (matches the original report);
                             # 30 == DIM_IN -> full rank, where Direct CSS = TrIM
N_ITERATIONS = 1
WEIGHT_POWER = 1.0

STEP_SIZE    = 3.0
N_ESTIMATORS = 30
SAMPLE_RANGE = [3200, 6400]
FUN_STR      = "poly"
N_TOTAL      = 6400
N_TEST       = 3200

ROTATION_MODE = "orth"       # rotated case


print(f"STEP_SIZE = {STEP_SIZE}")
print(f"LIFETIME = {LIFETIME}")
print(f"DIM_IN = {DIM_IN}")
print(f"ACTIVE = {ACTIVE}")
print(f"CSS_K = {CSS_K}")
print(f"N_ITERATIONS = {N_ITERATIONS}")
print(f"WEIGHT_POWER = {WEIGHT_POWER}")
print(f"FUN_STR = {FUN_STR}")
print(f"ROTATION_MODE = {ROTATION_MODE}")
print(f"EGOP backend = {_mf.EGOP_BACKEND} (forced)   downstream = rf")
print(f"N_ESTIMATORS (trees) = {N_ESTIMATORS}")
print(f"RF MAX_DEPTH_GRID = {list(MAX_DEPTH_GRID)}")
print(f"RF BOOTSTRAP = {BOOTSTRAP}  (bagging)")
print(f"RF MAX_FEATURES = {MAX_FEATURES}  (feature subsampling)")


# =============================================================
# Output path
# =============================================================
out_dir = "RF_results"
os.makedirs(out_dir, exist_ok=True)

rotation_name = "aligned" if ROTATION_MODE is None else ROTATION_MODE
save_prefix = os.path.join(
    out_dir,
    f"{rotation_name}_{FUN_STR}_d{DIM_IN}_Nesti{N_ESTIMATORS}_directcss_rf"
)


# =============================================================
# Data: X ~ Uniform([-1, 1]^d)
# =============================================================
np.random.seed(0)
x_train = np.random.rand(N_TOTAL, DIM_IN) * 2 - 1
x_test  = np.random.rand(N_TEST, DIM_IN) * 2 - 1

fn = SimulatedData(
    DIM_IN,
    ACTIVE,
    rotation=ROTATION_MODE,
    fun=FUN_STR
)

y_train = np.array(vmap(fn.fun, in_axes=0, out_axes=0)(x_train))
y_test  = np.array(vmap(fn.fun, in_axes=0, out_axes=0)(x_test))
true_H  = np.array(fn.get_true_H(x_test))


# =============================================================
# Print EGOP angle distance: estimated H vs true H
# (RF-based now, since EGOP_BACKEND is forced to "rf" above.)
# =============================================================
angle_simulation = SimulationAligned(
    DIM_IN,
    ACTIVE,
    N_ESTIMATORS,
    1,              # SimulationAligned requires tries == 1
    LIFETIME,
    STEP_SIZE,
    SAMPLE_RANGE,
    x_train,
    x_test,
    y_train,
    y_test,
    true_H
)

angle_simulation.train()

print("\nEstimated EGOP angle distances:")
print(f"  Iteration 1:")
for n_sim in SAMPLE_RANGE:
    forest = angle_simulation.forests[n_sim][0]
    dist = get_angle_distance(forest.H, true_H, ACTIVE)
    print(f"    n = {n_sim}: angle distance = {dist:.6f}")

for it in range(2, N_ITERATIONS + 1):
    angle_simulation.reiterate()
    print(f"  Iteration {it}:")
    for n_sim in SAMPLE_RANGE:
        forest = angle_simulation.forests[n_sim][0]
        dist = get_angle_distance(forest.H, true_H, ACTIVE)
        print(f"    n = {n_sim}: angle distance = {dist:.6f}")


# =============================================================
# Run experiment: reference methods + Direct CSS (k sweep)
# Downstream learner is whatever css_utils_direct imports (RF here).
# =============================================================
evaluation_comparison_with_direct_css(
    x_train=x_train,
    y_train=y_train,
    x_test=x_test,
    y_test=y_test,
    true_H=true_H,
    dim_in=DIM_IN,
    active=ACTIVE,
    n_estimators=N_ESTIMATORS,
    lifetime=LIFETIME,
    step_size=STEP_SIZE,
    sample_range=SAMPLE_RANGE,
    save_path=save_prefix,
    css_random_state=0,
    css_k_list=CSS_K,
    n_iterations=N_ITERATIONS,
    weight_power=WEIGHT_POWER,
)

print(f"\nDone. Figures are in '{out_dir}/'.")