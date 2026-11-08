"""
directcss_trueH_seeds_rf.py
===========================

Construction-vs-estimation isolation for the ROTATED case, RF downstream,
with a seed loop reporting median MSE.

WHY THIS EXPERIMENT
-------------------
In the rotated case the estimated EGOP is itself failing (angle distance to
true_H ~1.41, near pi/2), so "Direct CSS approaches TrIM while CSS-X is floored"
could be caused by geometry (construction) OR by a bad H estimate -- the two are
confounded. To separate them we ALSO run both CSS variants on the TRUE H:

    CSS-X (true H)      : top-diagonal selection of k ORIGINAL columns using a
                          PERFECT H. If this is STILL floored, the failure is
                          construction (original axes cannot span a rotated
                          subspace), NOT estimation error.
    Direct CSS (true H) : RPQR-selected k columns of the PERFECT H (dense combos).
                          If this approaches the oracle, dense H-columns CAN span
                          the rotated subspace even at small k.

The contrast Direct-CSS(true H) vs CSS-X(true H) -- both using a perfect H --
is the load-bearing comparison: it isolates construction from estimation.

Estimated-H versions and the two ceilings are kept for reference:
    references (k-independent): TrIM (est H) ; transform by true H (oracle floor)
    per k: Direct CSS (est H) ; CSS-X (est H) ; Direct CSS (true H) ; CSS-X (true H)

Seed loop: the underlying rotation R is FIXED (same rotated problem); each seed
re-draws x_train / x_test, so this reports "same problem, resampled data" error
bars. MSE is summarized as median [min, max] over seeds; the single-seed jitter
and non-monotonicity seen before should wash out.

Both stages are random forests (EGOP backend="rf"; downstream via RandomForest).
"""

import os
import time
import numpy as np
from collections import defaultdict
from jax import vmap

from utils import SimulatedData, get_angle_distance
from Mondrian_RF.Mondrian_forest import TrIM                       # EGOP, backend="rf"
from RandomForest.utils import evaluate_all_lifetimes, two_one_norm  # RF downstream
from css_utils_direct import css_project_data_direct               # RPQR CSS on H


# =============================================================
# Settings
# =============================================================
DIM_IN        = 30
ACTIVE        = 2
CSS_K_LIST    = [4, 8, 12, 16]
SAMPLE_RANGE  = [3200, 6400]
N_TOTAL       = 6400
N_TEST        = 3200

N_ESTIMATORS  = 30
LIFETIME      = 5                     # ignored by RF; signature parity
STEP_SIZE     = 3.0
FUN_STR       = "poly"
ROTATION_MODE = "orth"                # "orth" = rotated (the case this targets); None = aligned

BOOTSTRAP     = True
MAX_FEATURES  = 1.0

CSS_RANDOM_STATE = 0
N_SEEDS       = 5                     # data-resample seeds for the median
SEEDS         = list(range(N_SEEDS))


def sum_of_ranges(X):
    X = np.asarray(X)
    return float((X.max(axis=0) - X.min(axis=0)).sum())


rotation_name = "aligned" if ROTATION_MODE is None else ROTATION_MODE
print(f"ROTATION_MODE = {ROTATION_MODE} ({rotation_name})")
print(f"DIM_IN={DIM_IN} ACTIVE={ACTIVE} CSS_K_LIST={CSS_K_LIST} N_SEEDS={N_SEEDS}")
print(f"N_ESTIMATORS={N_ESTIMATORS} STEP_SIZE={STEP_SIZE} FUN_STR={FUN_STR}")
print(f"EGOP backend=rf  downstream=rf  BOOTSTRAP={BOOTSTRAP}  MAX_FEATURES={MAX_FEATURES}")

# Fixed rotation R (built once): same rotated problem across all seeds.
fn = SimulatedData(DIM_IN, ACTIVE, rotation=ROTATION_MODE, fun=FUN_STR)


def measure(x_tr_p, y_tr, x_te_p, y_te):
    """Time the downstream RF fit/eval; return (runtime, best_mse, sum_of_ranges)."""
    sor = sum_of_ranges(x_tr_p)
    t0 = time.perf_counter()
    res, _ = evaluate_all_lifetimes(
        x_tr_p, y_tr, x_te_p, y_te, N_ESTIMATORS, LIFETIME,
        bootstrap=BOOTSTRAP, max_features=MAX_FEATURES)
    rt = time.perf_counter() - t0
    return rt, float(np.min(res["mse"])), sor


# key = (n, method_label, k) -> list over seeds of dict(mse, rt, sor)
agg = defaultdict(list)
ad_by_n = defaultdict(list)   # angle distances per n over seeds


for seed in SEEDS:
    np.random.seed(seed)
    x_train = np.random.rand(N_TOTAL, DIM_IN) * 2 - 1
    x_test  = np.random.rand(N_TEST, DIM_IN) * 2 - 1
    y_train = np.array(vmap(fn.fun, in_axes=0, out_axes=0)(x_train))
    y_test  = np.array(vmap(fn.fun, in_axes=0, out_axes=0)(x_test))
    true_H  = np.array(fn.get_true_H(x_test))
    nH        = true_H / two_one_norm(true_H)
    diag_true = np.argsort(np.diag(true_H))[::-1]   # top-diagonal order on TRUE H

    for n in SAMPLE_RANGE:
        x_tr = x_train[:n]
        y_tr = y_train[:n]

        # ---- RF EGOP ----
        forest = TrIM(n_estimators=N_ESTIMATORS, lifetime=LIFETIME,
                      step_size=STEP_SIZE, random_state=(n + 1) * 2 + seed,
                      backend="rf")
        forest.X = np.array(x_tr)
        forest.y = np.array(y_tr)
        forest.mf.set_random_state()
        forest.mf.fit(forest.X, forest.y)
        importance = forest._get_importance()
        H_hat = (importance @ importance.T) / x_tr.shape[0]
        forest.H = H_hat
        ad = get_angle_distance(H_hat, true_H, ACTIVE)
        ad_by_n[n].append(ad)
        diag_hat = np.argsort(np.diag(H_hat))[::-1]  # top-diagonal order on EST H

        # ---- references (k-independent) ----
        rt, mse, sor = measure(forest.transform(x_tr), y_tr,
                               forest.transform(x_test), y_test)
        agg[(n, "TrIM (est H)", DIM_IN)].append(dict(mse=mse, rt=rt, sor=sor))

        rt, mse, sor = measure(x_tr @ nH, y_tr, x_test @ nH, y_test)
        agg[(n, "transform true H (oracle)", DIM_IN)].append(dict(mse=mse, rt=rt, sor=sor))

        # ---- per k: 4 CSS variants ----
        for k in CSS_K_LIST:
            # Direct CSS (est H)
            xtr, xte, _, _ = css_project_data_direct(H_hat, x_tr, x_test,
                                                     css_k=k, random_state=CSS_RANDOM_STATE)
            rt, mse, sor = measure(xtr, y_tr, xte, y_test)
            agg[(n, "Direct CSS (est H)", k)].append(dict(mse=mse, rt=rt, sor=sor))

            # Direct CSS (true H)  <-- perfect H, dense columns
            xtr, xte, _, _ = css_project_data_direct(true_H, x_tr, x_test,
                                                     css_k=k, random_state=CSS_RANDOM_STATE)
            rt, mse, sor = measure(xtr, y_tr, xte, y_test)
            agg[(n, "Direct CSS (true H)", k)].append(dict(mse=mse, rt=rt, sor=sor))

            # CSS-X (est H): top-diag original columns via estimated H
            S = diag_hat[:k]
            rt, mse, sor = measure(x_tr[:, S], y_tr, x_test[:, S], y_test)
            agg[(n, "CSS-X (est H)", k)].append(dict(mse=mse, rt=rt, sor=sor))

            # CSS-X (true H): top-diag original columns via PERFECT H
            S = diag_true[:k]
            rt, mse, sor = measure(x_tr[:, S], y_tr, x_test[:, S], y_test)
            agg[(n, "CSS-X (true H)", k)].append(dict(mse=mse, rt=rt, sor=sor))

        print(f"[seed={seed} n={n}] EGOP angle_dist={ad:.4f}  (done)")


# =============================================================
# Aggregate + print
# =============================================================
def med_lo_hi(vals):
    a = np.array(vals, dtype=float)
    return float(np.median(a)), float(a.min()), float(a.max())


# ordered method list for printing
_ORDER = ["transform true H (oracle)", "TrIM (est H)",
          "Direct CSS (true H)", "CSS-X (true H)",
          "Direct CSS (est H)", "CSS-X (est H)"]

print("\n\n=========  ISOLATION SUMMARY (rotated, RF downstream, "
      f"median over {N_SEEDS} seeds)  =========")

for n in SAMPLE_RANGE:
    ad_med, ad_lo, ad_hi = med_lo_hi(ad_by_n[n])
    # median TrIM runtime for the rt/TrIM column
    trim_rt = float(np.median([r["rt"] for r in agg[(n, "TrIM (est H)", DIM_IN)]]))

    print(f"\n  n = {n}   EGOP angle_dist to true_H: "
          f"median {ad_med:.4f} [{ad_lo:.4f}, {ad_hi:.4f}]  (max possible pi/2=1.5708)")
    print(f"  {'Method / k':27s} | {'median MSE':>11s} | {'[min':>10s}, {'max]':>10s} | "
          f"{'med rt(s)':>9s} | {'rt/TrIM':>7s}")
    print("  " + "-" * 92)

    # references first (k = DIM_IN)
    for label in ["transform true H (oracle)", "TrIM (est H)"]:
        recs = agg[(n, label, DIM_IN)]
        m, lo, hi = med_lo_hi([r["mse"] for r in recs])
        rt = float(np.median([r["rt"] for r in recs]))
        print(f"  {label:27s} | {m:11.4e} | {lo:10.4e}, {hi:10.4e} | "
              f"{rt:9.3f} | {rt/trim_rt:7.3f}")
    print("  " + "-" * 92)

    # per-k methods, grouped so the true-H pair sits together
    for k in CSS_K_LIST:
        for label in ["Direct CSS (true H)", "CSS-X (true H)",
                      "Direct CSS (est H)", "CSS-X (est H)"]:
            recs = agg[(n, label, k)]
            m, lo, hi = med_lo_hi([r["mse"] for r in recs])
            rt = float(np.median([r["rt"] for r in recs]))
            tag = f"{label} k={k}"
            print(f"  {tag:27s} | {m:11.4e} | {lo:10.4e}, {hi:10.4e} | "
                  f"{rt:9.3f} | {rt/trim_rt:7.3f}")
        print("  " + "-" * 92)

print("\nHow to read the isolation:")
print("  * If CSS-X (true H) stays ~flat/floored across k while Direct CSS (true H)")
print("    drops toward the oracle as k grows -> the CSS-X failure is CONSTRUCTION")
print("    (original axes cannot span the rotated subspace), NOT H-estimation error,")
print("    since both use a perfect H.")
print("  * Compare (true H) vs (est H) rows at the same k to see how much the poor")
print("    rotated EGOP estimate degrades each method on top of the construction limit.")