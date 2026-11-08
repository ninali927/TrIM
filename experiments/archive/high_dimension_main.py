"""
dscan_main_rf.py
================

High-dimensional AMBIENT-DIMENSION SCAN for the aligned (no-rotation) case, with
BOTH stages on a standard random forest:

    EGOP (H) estimation : TrIM(..., backend="rf")          -> RandomForestRegressor probe
    downstream regressor: RandomForest.utils.evaluate_all_lifetimes  -> RandomForestRegressor

This is self-contained: the RF EGOP backend is requested explicitly via
backend="rf", and the RF downstream comes from importing evaluate_all_lifetimes
from RandomForest.utils. Nothing here depends on the EGOP_BACKEND module
constant or on css_utils' import line, so you do NOT need to flip any switch.

It compares four methods as the ambient dimension d grows, all sharing the
SAME RF EGOP estimate of H:

    no transformation   : RF on the raw d-dim input (baseline, no H used)
    transform by true H : oracle floor (analytic H, d-dim transform)
    TrIM                : full transform by the ESTIMATED H (d-dim output)
    CSS (top-diagonal)  : keep the k=ACTIVE coords with largest diag(H) (k-dim)

Two effects to look for:

  (1) RUNTIME GAP widens. TrIM and true H transform into d output columns, so
      their downstream RF cost grows with d; CSS keeps only k features, so its
      cost stays flat. CSS should pull ahead of BOTH TrIM and true H as d grows.

  (2) MSE. true H is the floor. The true-H -> TrIM gap isolates RF-EGOP
      estimation error (does it grow with d?); the TrIM -> CSS gap is
      transform-vs-selection in the aligned case (CSS preserves axis-alignment
      by construction). With MAX_FEATURES="sqrt" (rerun) each large-d split
      samples only ~sqrt(d) features; CSS, having pre-selected the active dims
      via H, should be the least affected.

For each (d, n) it records: RF-EGOP time, angle distance of the estimated H to
true_H (does RF EGOP stay accurate as d grows?), and best-MSE-over-max_depth plus
downstream runtime for transform-by-true-H, TrIM, and CSS (top-diagonal, k=ACTIVE).
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from jax import vmap

from utils import SimulatedData, get_angle_distance
from Mondrian_RF.Mondrian_forest import TrIM                       # EGOP (backend="rf" below)
from RandomForest.utils import evaluate_all_lifetimes, two_one_norm  # RF downstream


# =============================================================
# Settings
# =============================================================
DIM_LIST     = [1000]   # ambient dimensions to scan
ACTIVE       = 2
CSS_K        = ACTIVE                  # CSS keeps exactly the active-dim count
SAMPLE_RANGE = [3200, 6400]
N_TOTAL      = 6400
N_TEST       = 3200

N_ESTIMATORS = 30                      # trees, in both EGOP probe and downstream
LIFETIME     = 5                       # ignored by RF; passed for signature parity
STEP_SIZE    = 3.0                     # keep LARGE (RF EGOP is piecewise-constant)
FUN_STR      = "poly"
ROTATION_MODE = None                   # aligned case

# Downstream RF knobs (also the bagging vs feature-subsampling switches).
BOOTSTRAP    = True
MAX_FEATURES = "sqrt"               # 1.0 = all features (no feature subsampling).
                                       # set "sqrt" and rerun to expose effect (2) above.


print(f"DIM_LIST = {DIM_LIST}")
print(f"ACTIVE = {ACTIVE}   CSS_K = {CSS_K}")
print(f"SAMPLE_RANGE = {SAMPLE_RANGE}")
print(f"N_ESTIMATORS (trees) = {N_ESTIMATORS}")
print(f"STEP_SIZE = {STEP_SIZE}   FUN_STR = {FUN_STR}   ROTATION_MODE = {ROTATION_MODE}")
print(f"EGOP backend = rf (explicit)   downstream = rf")
print(f"RF BOOTSTRAP = {BOOTSTRAP}   RF MAX_FEATURES = {MAX_FEATURES}")


out_dir = "RF_dscan_results"
os.makedirs(out_dir, exist_ok=True)


# =============================================================
# Scan
# =============================================================
records = []  # one dict per (d, n)

for d in DIM_LIST:
    # Data: X ~ Uniform([-1, 1]^d); target depends on the first ACTIVE coords.
    np.random.seed(0)
    X = np.random.rand(N_TOTAL, d) * 2 - 1
    X_test = np.random.rand(N_TEST, d) * 2 - 1
    fn = SimulatedData(d, ACTIVE, rotation=ROTATION_MODE, fun=FUN_STR)
    y = np.array(vmap(fn.fun, in_axes=0, out_axes=0)(X))
    y_test = np.array(vmap(fn.fun, in_axes=0, out_axes=0)(X_test))
    true_H = np.array(fn.get_true_H(X_test))

    for n in SAMPLE_RANGE:
        x_tr = X[:n]
        y_tr = y[:n]

        # ---------- RF EGOP: estimate H (timed) ----------
        t0 = time.perf_counter()
        forest = TrIM(n_estimators=N_ESTIMATORS, lifetime=LIFETIME,
                      step_size=STEP_SIZE, random_state=(n + 1) * 2, backend="rf")
        forest.X = np.array(x_tr)
        forest.y = np.array(y_tr)
        forest.mf.set_random_state()
        forest.mf.fit(forest.X, forest.y)
        importance = forest._get_importance()
        H_hat = (importance @ importance.T) / x_tr.shape[0]
        egop_t = time.perf_counter() - t0
        ad = get_angle_distance(H_hat, true_H, ACTIVE)
        forest.H = H_hat  # needed by forest.transform (TrIM)

        # ---------- no transformation (RF on raw d-dim input, timed) ----------
        t0 = time.perf_counter()
        res_notrans, _ = evaluate_all_lifetimes(
            x_tr, y_tr, X_test, y_test, N_ESTIMATORS, LIFETIME,
            bootstrap=BOOTSTRAP, max_features=MAX_FEATURES)
        notrans_t = time.perf_counter() - t0
        notrans_mse = float(np.min(res_notrans["mse"]))

        # ---------- transform by true H (oracle floor, timed) ----------
        nH = true_H / two_one_norm(true_H)
        t0 = time.perf_counter()
        res_true, _ = evaluate_all_lifetimes(
            x_tr @ nH, y_tr, X_test @ nH, y_test, N_ESTIMATORS, LIFETIME,
            bootstrap=BOOTSTRAP, max_features=MAX_FEATURES)
        trueH_t = time.perf_counter() - t0
        trueH_mse = float(np.min(res_true["mse"]))

        # ---------- TrIM: full transform by estimated H (d-dim, timed) ----------
        t0 = time.perf_counter()
        x_tr_trim = forest.transform(x_tr)
        x_te_trim = forest.transform(X_test)
        res_trim, _ = evaluate_all_lifetimes(
            x_tr_trim, y_tr, x_te_trim, y_test, N_ESTIMATORS, LIFETIME,
            bootstrap=BOOTSTRAP, max_features=MAX_FEATURES)
        trim_t = time.perf_counter() - t0
        trim_mse = float(np.min(res_trim["mse"]))

        # ---------- CSS top-diagonal (k = ACTIVE, timed) ----------
        S = np.argsort(np.diag(H_hat))[::-1][:CSS_K]
        t0 = time.perf_counter()
        res_css, _ = evaluate_all_lifetimes(
            x_tr[:, S], y_tr, X_test[:, S], y_test, N_ESTIMATORS, LIFETIME,
            bootstrap=BOOTSTRAP, max_features=MAX_FEATURES)
        css_t = time.perf_counter() - t0
        css_mse = float(np.min(res_css["mse"]))

        records.append(dict(
            d=d, n=n, egop_t=egop_t, ad=ad, S=S.tolist(),
            notrans_mse=notrans_mse, notrans_t=notrans_t,
            trueH_mse=trueH_mse, trueH_t=trueH_t,
            trim_mse=trim_mse, trim_t=trim_t,
            css_mse=css_mse, css_t=css_t,
        ))

        print(f"\n[d={d:<5d} n={n}] EGOP(rf) {egop_t:8.3f}s  angle_dist {ad:.6f}  "
              f"CSS cols {S.tolist()}")
        print(f"    MSE   | no-trans {notrans_mse:.6e} | trueH {trueH_mse:.6e} | TrIM {trim_mse:.6e} "
              f"| CSS {css_mse:.6e} | CSS/TrIM {css_mse/trim_mse:.4f}")
        print(f"    time  | no-trans {notrans_t:8.3f}s | trueH {trueH_t:8.3f}s | TrIM {trim_t:8.3f}s "
              f"| CSS {css_t:8.3f}s | TrIM/CSS x{trim_t/css_t:.2f}")


# =============================================================
# Summary tables (one per n)
# =============================================================
for n in SAMPLE_RANGE:
    rows = [r for r in records if r["n"] == n]
    print(f"\n  ===== d-scan summary: n = {n} "
          f"(MAX_FEATURES={MAX_FEATURES}) =====")
    print("  d     | EGOP(s) | angle_dist | no-trans MSE | trueH MSE  | TrIM MSE   | CSS MSE    | "
          "CSS/TrIM | no-trans(s) | trueH(s) | TrIM(s) | CSS(s) | TrIM/CSS")
    print("  " + "-" * 132)
    for r in rows:
        print(f"  {r['d']:<5d} | {r['egop_t']:7.3f} | {r['ad']:10.6f} | "
              f"{r['notrans_mse']:12.4e} | {r['trueH_mse']:10.4e} | {r['trim_mse']:10.4e} | {r['css_mse']:10.4e} | "
              f"{r['css_mse']/r['trim_mse']:8.4f} | "
              f"{r['notrans_t']:11.3f} | {r['trueH_t']:8.3f} | {r['trim_t']:7.3f} | {r['css_t']:6.3f} | "
              f"x{r['trim_t']/r['css_t']:.2f}")
    print("  " + "-" * 132)


# =============================================================
# Figures (one set per n): runtime vs d, MSE vs d, angle distance vs d
# =============================================================
def _logfmt(ax):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.tick_params(labelsize=12)

for n in SAMPLE_RANGE:
    rows = sorted([r for r in records if r["n"] == n], key=lambda r: r["d"])
    ds = [r["d"] for r in rows]

    # runtime vs d
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(ds, [r["egop_t"] for r in rows], marker="o", label="EGOP (RF, estimate H)")
    ax.plot(ds, [r["notrans_t"] for r in rows], marker="o", label="no transformation (d-dim RF)")
    ax.plot(ds, [r["trueH_t"] for r in rows], marker="o", label="transform by true H (d-dim RF)")
    ax.plot(ds, [r["trim_t"] for r in rows], marker="o", label="TrIM (d-dim RF)")
    ax.plot(ds, [r["css_t"] for r in rows], marker="o", label=f"CSS (k={CSS_K} RF)")
    ax.set_title(f"Runtime vs ambient dimension  (n={n}, max_features={MAX_FEATURES})",
                 fontsize=12)
    ax.set_xlabel("ambient dimension d", fontsize=13)
    ax.set_ylabel("seconds", fontsize=13)
    _logfmt(ax); ax.legend(fontsize=10); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"dscan_runtime_n{n}.png"), dpi=130)
    plt.close(fig)

    # MSE vs d
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(ds, [r["trueH_mse"] for r in rows], marker="o", linestyle="--",
            label="transform by true H (oracle)")
    ax.plot(ds, [r["notrans_mse"] for r in rows], marker="o", label="no transformation")
    ax.plot(ds, [r["trim_mse"] for r in rows], marker="o", label="TrIM")
    ax.plot(ds, [r["css_mse"] for r in rows], marker="o", label=f"CSS (k={CSS_K})")
    ax.set_title(f"Best MSE vs ambient dimension  (n={n}, max_features={MAX_FEATURES})",
                 fontsize=12)
    ax.set_xlabel("ambient dimension d", fontsize=13)
    ax.set_ylabel("best MSE over max_depth", fontsize=13)
    _logfmt(ax); ax.legend(fontsize=10); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"dscan_mse_n{n}.png"), dpi=130)
    plt.close(fig)

    # angle distance vs d (does RF EGOP stay accurate?)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(ds, [r["ad"] for r in rows], marker="o", color="tab:purple")
    ax.set_title(f"RF-EGOP angle distance vs ambient dimension  (n={n})", fontsize=12)
    ax.set_xlabel("ambient dimension d", fontsize=13)
    ax.set_ylabel("angle distance to true_H", fontsize=13)
    ax.set_xscale("log"); ax.tick_params(labelsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"dscan_angle_n{n}.png"), dpi=130)
    plt.close(fig)

print(f"\nDone. Tables above; figures in '{out_dir}/'.")