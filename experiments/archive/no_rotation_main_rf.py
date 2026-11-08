"""
no_rotation_main_RF.py
======================

Run the aligned (no-rotation) TrIM-style experiment with the STANDARD RANDOM
FOREST as the downstream learner, sweeping CSS over a list of k values. Mirror of
rotation_main_kNN.py; the only structural differences are:

  ROTATION_MODE = None                       -> aligned case (x0, x1 active)
  from css_utils_max import ...              -> top-diagonal CSS + Eigen baseline
  results saved under RF_results/

The downstream learner (Mondrian vs Euclidean k-NN vs standard RF) is selected
NOT here but by the two import lines at the top of css_utils_max.py:

    # Mondrian (original):
    from Mondrian_RF.Mondrian_forest import TrIM
    from Mondrian_RF.utils import evaluate_all_lifetimes, two_one_norm

    # Standard Random Forest (this experiment):
    from RandomForest.RandomForest import TrIM
    from RandomForest.utils import evaluate_all_lifetimes, two_one_norm

With the RandomForest imports active, evaluate_all_lifetimes sweeps max_depth
instead of lifetime. This import line controls ONLY the downstream learner; it
does NOT control how H is estimated. The EGOP backend is a SEPARATE switch,
EGOP_BACKEND in Mondrian_RF/Mondrian_forest.py ("mondrian" or "rf"). Because
TrIM is re-exported unchanged, the angle-distance block below uses whichever
backend EGOP_BACKEND selects -- Mondrian OR random forest -- exactly the same
estimator css_utils uses for the CSS/TrIM curves. The two are therefore always
consistent, but neither is hard-wired to Mondrian.

NOTE: under the RandomForest imports the plot x-axis labelled "Lifetime" is
actually max_depth (values/minima are correct; only the label text is a
misnomer).
"""

import os
import numpy as np
from jax import vmap

from utils import SimulatedData, SimulationAligned, get_angle_distance
from css_utils_max import evaluation_comparison_with_css

# RF configuration is recorded here for the log; the actual knobs live in
# RandomForest.utils (MAX_DEPTH_GRID / BOOTSTRAP / MAX_FEATURES) so css_utils
# needs no change beyond its import line.
from RandomForest.utils import MAX_DEPTH_GRID, BOOTSTRAP, MAX_FEATURES


# =============================================================
# Settings
# =============================================================
DIM_IN       = 30
ACTIVE       = 2
LIFETIME     = 5
CSS_K        = [2, 3, 4, 5]   # k sweep for the aligned case
N_ITERATIONS = 1   # 1 = initial fit only (no reiteration)
WEIGHT_POWER = 1.0

STEP_SIZE    = 3.0
N_ESTIMATORS = 30           # number of trees in the random forest
SAMPLE_RANGE = [3200, 6400]
FUN_STR      = "poly"       # "poly" or "max"
N_TOTAL      = 6400
N_TEST       = 3200

# None  = aligned case
# "orth" = rotated case
ROTATION_MODE = None


print(f"STEP_SIZE = {STEP_SIZE}")
print(f"LIFETIME = {LIFETIME}")
print(f"DIM_IN = {DIM_IN}")
print(f"ACTIVE = {ACTIVE}")
print(f"CSS_K = {CSS_K}")
print(f"N_ITERATIONS = {N_ITERATIONS}")
print(f"WEIGHT_POWER = {WEIGHT_POWER}")
print(f"FUN_STR = {FUN_STR}")
print(f"ROTATION_MODE = {ROTATION_MODE}")
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
    f"{rotation_name}_{FUN_STR}_d{DIM_IN}_Nesti{N_ESTIMATORS}_rf"
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
# (backend set by EGOP_BACKEND in Mondrian_forest.py; independent of the
#  downstream learner. Uses the same estimator as the CSS/TrIM curves.)
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
# Run experiment: paper methods + CSS + Eigen (k sweep)
# Downstream learner is whatever css_utils_max.py imports (standard RF here).
# =============================================================
evaluation_comparison_with_css(
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