"""
rotation_main_rf_egop_stepsweep.py
==================================
Same as rotation_main_rf_egop_Ksweep_oracle.py, with ONE change: STEP_SIZE is
now a grid (STEP_GRID), and the whole per-d block runs once per step. Everything
else -- K sweep (N_ITERATIONS=4), oracle line, SimulationAligned path, d loop --
is identical, so results are on the same code path as the earlier runs.

Read: within each d, and within a single n column (n=3200 and n=6400 are
independent seeds under SimulationAligned, so don't compare across them), find
the step that gives the lowest angle distance. Step only keeps the central
difference from degenerating (too small -> both probes in one RF leaf ->
zero gradient; too large -> local curvature washed out); it does not fix the
rotated-ridge problem, so don't expect it to revive a floored d.
"""

import numpy as np
from jax import vmap

# --- Force RF EGOP for the whole run, before anything constructs a TrIM. ------
import Mondrian_RF.Mondrian_forest as _mf
_mf.EGOP_BACKEND = "rf"

from utils import SimulatedData, SimulationAligned, get_angle_distance


# =============================================================
# Settings
# =============================================================
ACTIVE       = 2
LIFETIME     = 5
N_ITERATIONS = 4            # K sweep: report angle after K = 1..4 reiterations

STEP_GRID    = [0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0]   # <-- swept
ORACLE_H     = 1e-3         # oracle's own FD step (true f is smooth -> small)

N_ESTIMATORS = 100
SAMPLE_RANGE = [3200, 6400]
FUN_STR      = "poly"
N_TOTAL      = 6400
N_TEST       = 3200

ROTATION_MODE = "orth"      # rotated case

DIM_LIST     = [10, 30, 50]


print(f"STEP_GRID (RF surrogate) = {STEP_GRID}")
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
# d-sweep x step-sweep
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

    # ---- Oracle line (surrogate removed) -- independent of step ----
    H_oracle = oracle_egop_H(fn, x_test, ORACLE_H)
    oracle_dist = get_angle_distance(H_oracle, true_H, ACTIVE)
    print(f"Oracle (FD on true f, near-exact grad): angle distance = {oracle_dist:.6f}")

    # ---- RF-surrogate EGOP: sweep step, K sweep inside each step ----
    for STEP_SIZE in STEP_GRID:
        print(f"\n  --- step_size = {STEP_SIZE} ---")

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

        print("  RF surrogate EGOP (angle distance, lower=better):")
        report(1)
        for K in range(2, N_ITERATIONS + 1):
            angle_simulation.reiterate()
            report(K)


print("\nDone.")