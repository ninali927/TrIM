"""
rotation_main_mondrian_egop_Ksweep.py
=====================================
Mondrian-backend version of rotation_main_rf_egop_stepsweep.py. Two changes only:

  (1) EGOP_BACKEND = "mondrian"  (was "rf"). This is the runtime-optimized
      Mondrian path -- the actual thing PI wants compared against RF.
  (2) step swept -> single STEP_SIZE = 1.0, matching the step the RF runs
      actually used (SimulationAligned passed 1.0), so the two backends are
      compared at the same step, same d, same K, same n, same M.

Everything else -- K sweep (N_ITERATIONS=4), oracle line, SimulationAligned
path, d loop over {10,30,50} -- is identical to the RF script, so the Mondrian
table can be laid directly next to the RF table (question A: RF vs Mondrian
EGOP accuracy at each d).
"""

import numpy as np
from jax import vmap

# --- Force MONDRIAN EGOP for the whole run, before anything constructs a TrIM.
import Mondrian_RF.Mondrian_forest as _mf
_mf.EGOP_BACKEND = "mondrian"

from utils import SimulatedData, SimulationAligned, get_angle_distance


# =============================================================
# Settings
# =============================================================
ACTIVE       = 2
LIFETIME     = 5
N_ITERATIONS = 4            # K sweep: report angle after K = 1..4 reiterations

STEP_SIZE    = 1.0          # matches the step the RF runs actually used
ORACLE_H     = 1e-3         # oracle's own FD step (true f is smooth -> small)

N_ESTIMATORS = 30
SAMPLE_RANGE = [3200, 6400]
FUN_STR      = "poly"
N_TOTAL      = 6400
N_TEST       = 3200

ROTATION_MODE = "orth"      # rotated case

DIM_LIST     = [5, 10, 15, 20, 30, 50, 100]


print(f"STEP_SIZE (Mondrian surrogate) = {STEP_SIZE}")
print(f"ORACLE_H (true-f FD step) = {ORACLE_H}")
print(f"LIFETIME = {LIFETIME}")
print(f"DIM_LIST = {DIM_LIST}")
print(f"ACTIVE = {ACTIVE}")
print(f"N_ITERATIONS (max K) = {N_ITERATIONS}")
print(f"FUN_STR = {FUN_STR}")
print(f"ROTATION_MODE = {ROTATION_MODE}")
print(f"EGOP backend = {_mf.EGOP_BACKEND} (forced)")
print(f"N_ESTIMATORS (trees) = {N_ESTIMATORS}")


# =============================================================
# Oracle: EGOP built from the TRUE function via central finite differences.
# =============================================================
def oracle_egop_H(fn, X, h):
    """H_oracle = (1/n) sum_i grad_f(x_i) grad_f(x_i)^T, grad by central FD on fn.fun."""
    f_batch = vmap(fn.fun, in_axes=0, out_axes=0)
    n, d = X.shape
    grads = np.zeros((n, d))
    for j in range(d):
        Xp = X.copy(); Xp[:, j] += h
        Xm = X.copy(); Xm[:, j] -= h
        fp = np.array(f_batch(Xp))
        fm = np.array(f_batch(Xm))
        grads[:, j] = (fp - fm) / (2.0 * h)
    return grads.T @ grads / n


# =============================================================
# d-sweep
# =============================================================
for DIM_IN in DIM_LIST:
    print(f"\n{'='*55}")
    print(f"d = {DIM_IN}")
    print(f"{'='*55}")

    # ---- Data: X ~ Uniform([-1, 1]^d) ----
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

    # ---- Oracle line (surrogate removed) ----
    H_oracle = oracle_egop_H(fn, x_test, ORACLE_H)
    oracle_dist = get_angle_distance(H_oracle, true_H, ACTIVE)
    print(f"Oracle (FD on true f, near-exact grad): angle distance = {oracle_dist:.6f}")

    # ---- Mondrian-surrogate EGOP, reiteration K sweep ----
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

    def report(K):
        row = f"    K={K}  "
        for n_sim in SAMPLE_RANGE:
            forest = angle_simulation.forests[n_sim][0]
            dist = get_angle_distance(forest.H, true_H, ACTIVE)
            row += f"n={n_sim}: {dist:.6f}   "
        print(row)

    print("Mondrian surrogate EGOP (angle distance, lower=better):")
    report(1)
    for K in range(2, N_ITERATIONS + 1):
        angle_simulation.reiterate()
        report(K)


print("\nDone.")
