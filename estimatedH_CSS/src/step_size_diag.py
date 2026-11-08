"""
step_size_diag.py
==================

Quick diagnostic: is STEP_SIZE_DEFAULT=3.0 (copied from rotation_main_rf.py's
d=30 setup) actually reasonable at smaller d, or is it stepping outside the
training data's range ([-1, 1]^d, width 2) and probing pure RF extrapolation
noise instead of real gradient signal?

Runs ONLY the iterative sweep (no oracle -- not needed to see whether
iteration behaves monotonically) at d=10, across a few candidate step sizes,
single seed, single largest n for speed. Not a full experiment -- just enough
to pick a step_size before committing to the overnight d=10..100 sweep.

Run with:
    python -u step_size_diag.py
"""

import numpy as np
from jax import vmap

from RandomForest.RandomForest import RFTrIM
from utils import SimulatedData, get_angle_distance


D          = 10
ACTIVE     = 2
N_ESTIMATORS = 30
MAX_DEPTH  = None
BOOTSTRAP  = False
MAX_FEATURES = 1.0
FUN_STR    = "poly"
ROTATION_MODE = "orth"

N          = 6400          # single n, largest from the d=10 benchmark run
N_TEST     = 3200
N_ITERATIONS = 4
SEED       = 0
ROTATION_SEED_BASE = 1000

STEP_SIZES = [0.1, 0.3, 0.5, 1.0, 2.0, 3.0]


np.random.seed(SEED)
x_train = np.random.rand(N, D) * 2 - 1
x_test  = np.random.rand(N_TEST, D) * 2 - 1

np.random.seed(ROTATION_SEED_BASE + SEED)
fn = SimulatedData(D, ACTIVE, rotation=ROTATION_MODE, fun=FUN_STR)

y_train = np.array(vmap(fn.fun, in_axes=0, out_axes=0)(x_train))
y_test  = np.array(vmap(fn.fun, in_axes=0, out_axes=0)(x_test))
true_H  = np.array(fn.get_true_H(x_test))

print(f"d={D}, n={N}, data range is [-1, 1]^{D} (width 2 per dimension)\n")
print(f"{'step_size':>10}  " + "  ".join(f"iter{k}" for k in range(1, N_ITERATIONS + 1)))

for step_size in STEP_SIZES:
    forest = RFTrIM(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                     bootstrap=BOOTSTRAP, max_features=MAX_FEATURES,
                     step_size=step_size, random_state=(N + 1), iteration=1)
    forest.fit(x_train, y_train)
    dists = [float(get_angle_distance(forest.H, true_H, ACTIVE))]

    for _ in range(N_ITERATIONS - 1):
        forest.reiterate()
        dists.append(float(get_angle_distance(forest.H, true_H, ACTIVE)))

    marker = ""
    if all(dists[i] >= dists[i + 1] - 1e-6 for i in range(len(dists) - 1)):
        marker = "  <- monotonically improving"

    print(f"{step_size:>10.2f}  " + "  ".join(f"{d_:.4f}" for d_ in dists) + marker)

print("\nProbe half-width at this step_size = step_size / 2 (added/subtracted "
      "from each coordinate). Data half-width is 1.0 (range [-1, 1]). A probe "
      "half-width much larger than 1.0 means the finite-difference points can "
      "fall outside the training data's range.")
