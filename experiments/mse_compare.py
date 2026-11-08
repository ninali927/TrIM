"""
mse_compare.py
==============

Downstream regression MSE, Mondrian forest vs random forest, with NO transform.
No H, no finite differences: train each forest on (X, y), predict y on the test
set, compare MSE. This checks the premise "RF is data-adaptive, so it fits
better", which the angle-distance sweeps do not test.

Same data protocol as the Experiment B sweeps: X ~ Uniform([-1,1]^d), seed 0,
rotation drawn on its own seed stream, n_max fixed at 12800 so the test set is
the same one those sweeps used.

Settings match the rest of the work:
    Mondrian : 30 trees, lifetime_max = 5
    RF       : 30 trees, bootstrap OFF, max_features = 1.0
               (RandomForest/utils.py defaults to bootstrap=True, so both are
               passed explicitly here)

"MSE" is the existing convention: the minimum test MSE over the complexity
sweep (Mondrian: over lifetime checkpoints; RF: over the max_depth grid), as in
np.min(res['mse']). Note the Mondrian sweep has many more checkpoints than the
RF grid has depths, so taking a minimum favours Mondrian slightly.

rel_mse = MSE / Var(y_test), so cells with different d / rotation can be read
on one scale (1.0 = no better than predicting the mean).

Loop order is n outermost (small n first), RF before Mondrian in every cell, and
each result is written to the CSV as soon as it finishes, so the run can be
stopped at any time without losing finished cells. Mondrian is the slow part.

Run with:
    python -u experiments/mse_compare.py > results/mse_compare/run.log 2>&1   (from the project root)

Output: results/mse_compare/mse_compare.csv
"""

# --- project paths: import from src/, write outputs under <project>/results/ ---
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
os.chdir(_ROOT)
# -------------------------------------------------------------------------------

import os
import csv
import time

import numpy as np
from jax import vmap

from utils import SimulatedData                                     # root utils.py
from Mondrian_RF.utils import evaluate_all_lifetimes as mondrian_eval
from RandomForest.utils import evaluate_all_lifetimes as rf_eval


# =============================================================
# Settings
# =============================================================
D_LIST         = [10, 30]            # extend once you see how long Mondrian takes
N_LIST         = [3200, 12800]       # ascending: cheap cells first
ROTATION_MODES = [None, "orth"]      # None = aligned, "orth" = rotated

N_MAX  = 12800                       # fixed so the test set matches Experiment B
N_TEST = 3200
SEED   = 0
ROTATION_SEED_BASE = 1000

ACTIVE       = 2
N_ESTIMATORS = 30
LIFETIME     = 5
FUN_STR      = "poly"

RF_BOOTSTRAP    = False
RF_MAX_FEATURES = 1.0

OUT_DIR  = os.path.join("results", "mse_compare")
OUT_PATH = os.path.join(OUT_DIR, "mse_compare.csv")
FIELDS = ["rotation", "d", "n", "backend", "best_mse", "rel_mse",
          "best_at", "seconds"]


def label(rotation):
    return "aligned" if rotation is None else "rotated"


def make_data(d, rotation):
    np.random.seed(SEED)
    x_train = np.random.rand(N_MAX, d) * 2 - 1
    x_test  = np.random.rand(N_TEST, d) * 2 - 1

    np.random.seed(ROTATION_SEED_BASE + SEED)
    fn = SimulatedData(d, ACTIVE, rotation=rotation, fun=FUN_STR)

    y_train = np.array(vmap(fn.fun, in_axes=0, out_axes=0)(x_train))
    y_test  = np.array(vmap(fn.fun, in_axes=0, out_axes=0)(x_test))
    return x_train, y_train, x_test, y_test


# =============================================================
# Run
# =============================================================
os.makedirs(OUT_DIR, exist_ok=True)
with open(OUT_PATH, "w", newline="") as f:
    csv.DictWriter(f, fieldnames=FIELDS).writeheader()

print(f"run started {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  D_LIST={D_LIST}  N_LIST={N_LIST}  rotations={[label(r) for r in ROTATION_MODES]}")
print(f"  trees={N_ESTIMATORS}  lifetime={LIFETIME}  "
      f"RF bootstrap={RF_BOOTSTRAP} max_features={RF_MAX_FEATURES}\n")

records = []

for n in N_LIST:
    for d in D_LIST:
        for rotation in ROTATION_MODES:
            x_train, y_train, x_test, y_test = make_data(d, rotation)
            xt, yt = x_train[:n], y_train[:n]
            var_test = float(np.var(y_test))

            # ---- RF (fast) ----
            t0 = time.time()
            res, _ = rf_eval(xt, yt, x_test, y_test, N_ESTIMATORS, LIFETIME,
                             bootstrap=RF_BOOTSTRAP, max_features=RF_MAX_FEATURES,
                             random_state=n + 1)
            t_rf = time.time() - t0
            i = int(np.argmin(res["mse"]))
            row_rf = {"rotation": label(rotation), "d": d, "n": n,
                      "backend": "RF", "best_mse": float(res["mse"][i]),
                      "rel_mse": float(res["mse"][i]) / var_test,
                      "best_at": res["times"][i], "seconds": t_rf}

            # ---- Mondrian (slow) ----
            t0 = time.time()
            res, _ = mondrian_eval(xt, yt, x_test, y_test, N_ESTIMATORS, LIFETIME,
                                   rng=np.random.RandomState(n + 1))
            t_m = time.time() - t0
            i = int(np.argmin(res["mse"]))
            row_m = {"rotation": label(rotation), "d": d, "n": n,
                     "backend": "Mondrian", "best_mse": float(res["mse"][i]),
                     "rel_mse": float(res["mse"][i]) / var_test,
                     "best_at": res["times"][i], "seconds": t_m}

            with open(OUT_PATH, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS)
                w.writerow(row_rf)
                w.writerow(row_m)
            records.extend([row_rf, row_m])

            print(f"[{label(rotation):7s} d={d:3d} n={n:6d}]  "
                  f"RF mse={row_rf['best_mse']:.4g} ({t_rf:6.1f}s)   "
                  f"Mondrian mse={row_m['best_mse']:.4g} ({t_m:7.1f}s)", flush=True)


# =============================================================
# Summary table
# =============================================================
print("\n" + "=" * 78)
print("Best test MSE over the complexity sweep (no transform)")
print("=" * 78)
print(f"{'data':>8} {'d':>4} {'n':>6} {'Mondrian':>11} {'RF':>11} {'RF/Mond':>9} "
      f"{'rel M':>8} {'rel RF':>8} {'sec M':>8} {'sec RF':>8}")
keys = sorted({(r["rotation"], r["d"], r["n"]) for r in records})
for rot, d, n in keys:
    m = next(r for r in records if (r["rotation"], r["d"], r["n"], r["backend"]) == (rot, d, n, "Mondrian"))
    f_ = next(r for r in records if (r["rotation"], r["d"], r["n"], r["backend"]) == (rot, d, n, "RF"))
    ratio = f_["best_mse"] / m["best_mse"] if m["best_mse"] > 0 else float("nan")
    print(f"{rot:>8} {d:>4} {n:>6} {m['best_mse']:>11.4g} {f_['best_mse']:>11.4g} {ratio:>9.2f} "
          f"{m['rel_mse']:>8.3f} {f_['rel_mse']:>8.3f} {m['seconds']:>8.1f} {f_['seconds']:>8.1f}")
print("\n  RF/Mond < 1 -> RF fits better;  > 1 -> Mondrian fits better")
print(f"\nDone. {OUT_PATH}")
