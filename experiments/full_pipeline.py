"""
full_pipeline.py   (was rotation_main.py)
=================

Run from the project root:
    python -u experiments/full_pipeline.py > results/full_pipeline/run.log 2>&1

Run TrIM paper-style experiment with (Weighted) CSS on either aligned or
rotated data, sweeping CSS over a list of k values.

Set:
    ROTATION_MODE = None    -> aligned case, useful directions are original x0, x1
    ROTATION_MODE = "orth"  -> rotated case, useful directions are random linear combinations

This main.py uses css_utils_weighted.evaluation_comparison_with_css, which:
    - replaces the unweighted CSS with Weighted CSS (kept under the labels
      "CSS (estimated H)" / "CSS (true H)"),
    - sweeps CSS over css_k_list (k-independent methods computed once per n),
    - logs a k=d TrIM-vs-WeightedCSS equivalence diagnostic per sample size,
    - emits one 6-method comparison figure per (n, k) plus one
      "best MSE vs k" summary figure per n.
"""

# --- project paths: import from src/, write outputs under <project>/results/ ---
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
os.chdir(_ROOT)
# -------------------------------------------------------------------------------

import os
import numpy as np
from jax import vmap

from utils import SimulatedData, SimulationAligned, get_angle_distance
from css_utils_trueH import evaluation_comparison_with_css
from css_utils_direct import evaluation_comparison_with_direct_css


# =============================================================
# Settings
# =============================================================
DIM_IN       = 30
ACTIVE       = 2
LIFETIME     = 5
CSS_K        = [4, 8, 12, 16, 20, 24, 28, 30]   # k sweep; 30 == DIM_IN -> true k=d (all columns)
N_ITERATIONS = 1   # 1 = initial fit only (no reiteration)
                   # k = initial fit + (k - 1) reiterations
WEIGHT_POWER = 1.0 # 1.0 matches TrIM's eigenbasis scaling; 0.5 -> H^{1/2}

STEP_SIZE    = 3.0
N_ESTIMATORS = 30
SAMPLE_RANGE = [3200, 6400]
FUN_STR      = "poly"       # "poly" or "max"
N_TOTAL      = 6400
N_TEST       = 3200

# None = aligned case
# "orth" = rotated case
ROTATION_MODE = "orth"



print(f"STEP_SIZE = {STEP_SIZE}")
print(f"LIFETIME = {LIFETIME}")
print(f"DIM_IN = {DIM_IN}")
print(f"ACTIVE = {ACTIVE}")
print(f"CSS_K = {CSS_K}")
print(f"N_ITERATIONS = {N_ITERATIONS}")
print(f"WEIGHT_POWER = {WEIGHT_POWER}")
print(f"FUN_STR = {FUN_STR}")
print(f"ROTATION_MODE = {ROTATION_MODE}")


# =============================================================
# Output path
# =============================================================
out_dir = os.path.join("results", "full_pipeline")
os.makedirs(out_dir, exist_ok=True)

rotation_name = "aligned" if ROTATION_MODE is None else ROTATION_MODE
save_prefix = os.path.join(
    out_dir,
    f"{rotation_name}_{FUN_STR}_d{DIM_IN}_Nesti{N_ESTIMATORS}"
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
# Run experiment: paper methods + Weighted CSS (k sweep)
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