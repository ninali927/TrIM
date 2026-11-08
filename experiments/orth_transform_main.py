"""
orth_transform_main.py
======================

Does an ORTHOGONAL-column transform beat TrIM's  X @ H_hat ?

    H_hat = U diag(lam) U^T        (estimated EGOP, eigen-decomposed)

Methods, same estimated H_hat, same Mondrian settings, sweep lifetime:
    1. no transform : X
    2. TrIM         : X @ H_hat       / two_one_norm(H_hat)
    3. orthogonal   : X @ U diag(lam) / two_one_norm(H_hat)   (all d directions)
    4. rotation XU  : X @ U    (orthonormal; its own two_one_norm is already 1)

WEIGHTS (--weights) sets the target  y = sum_i w_i z_i^4.  Equal weights give
lam_1 = lam_2, so U's top-2 eigenvectors are only defined up to a rotation;
e.g. --weights 1,0.5 makes them distinct (true lam_2/lam_1 = 0.25).
Diagnostic: eig_angle_i = angle (deg) between estimated u_i and true v_i.

2 and 3 share the same scalar, so they differ ONLY by the final rotation U^T
(X H_hat = (X U diag(lam)) U^T). U diag(lam) has orthogonal columns.

Within a seed, every method uses an identical Mondrian RNG stream, so MSE
differences come from the transform alone.

Usage (from the project root):
    python -u experiments/orth_transform_main.py > results/orth_transform/run.log 2>&1
    python -u experiments/orth_transform_main.py --smoke   (tiny run, checks the code works)
    python -u experiments/orth_transform_main.py --dims 10 --weights 1,0.5 --out orth_transform_w1_05
"""

# --- project paths: import from src/, write outputs under <project>/results/ ---
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
os.chdir(_ROOT)
# -------------------------------------------------------------------------------

import argparse
import os
import time

import jax.numpy as jnp

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jax import vmap

from utils import SimulatedData, get_angle_distance
from Mondrian_RF.Mondrian_forest import TrIM
from Mondrian_RF.utils import evaluate_all_lifetimes, two_one_norm


# =============================================================
# Settings
# =============================================================
DIMS         = [10, 20, 30]
ROTATIONS    = [None, "orth"]        # None = aligned
SAMPLE_RANGE = [3200, 6400]
SEEDS        = [0]                   # 1 seed for a first look; [0, 1, 2, 3, 4] for 5 seeds
ACTIVE       = 2
LIFETIME     = 5
N_ITERATIONS = 1                     # 1 = initial fit only
STEP_SIZE    = 3.0
N_ESTIMATORS = 30
FUN_STR      = "poly"
WEIGHTS      = None                  # None = equal weights (1, 1)
N_TOTAL      = 6400
N_TEST       = 3200

METHODS = ["no transform", "TrIM", "orthogonal", "rotation XU"]
COLORS  = {"no transform": "tab:blue", "TrIM": "tab:green", "orthogonal": "tab:red",
           "rotation XU": "tab:purple"}

OUT_DIR = os.path.join("results", "orth_transform")


# =============================================================
# TrIM with a selectable transform (usable in every iteration)
# =============================================================
def orth_matrix(H):
    """U diag(lam) / two_one_norm(H): orthogonal columns, eigenvalues descending."""
    H = 0.5 * (H + H.T)
    lam, U = np.linalg.eigh(H)
    order = np.argsort(lam)[::-1]
    lam = np.clip(lam[order], 0.0, None)
    U = U[:, order]
    return (U * lam) / two_one_norm(H)


def sorted_eig(H):
    H = 0.5 * (H + H.T)
    lam, U = np.linalg.eigh(H)
    order = np.argsort(lam)[::-1]
    return np.clip(lam[order], 0.0, None), U[:, order]


def eig_angles_deg(H, true_H, k):
    """Angle (degrees) between i-th estimated and i-th true eigenvector, i < k."""
    _, U = sorted_eig(H)
    _, V = sorted_eig(true_H)
    cos = np.abs(np.sum(U[:, :k] * V[:, :k], axis=0)).clip(0, 1)
    return np.degrees(np.arccos(cos))


class OrthTrIM(TrIM):
    """TrIM whose transform is either 'trim' (X H/||H||) or 'orth' (X U Lam/||H||)."""

    def __init__(self, transform_mode="trim", **kwargs):
        super().__init__(**kwargs)
        assert transform_mode in ("trim", "orth")
        self.transform_mode = transform_mode
        self._A, self._A_src = None, None

    def transform(self, X):
        if self.H is None or self.transform_mode == "trim":
            return super().transform(X)
        if self._A_src is not self.H:            # H changed -> recompute (cached)
            self._A, self._A_src = orth_matrix(self.H), self.H
        return np.matmul(X, self._A)


def estimate_H(x_tr, y_tr, mode, random_state):
    """Initial raw fit + H estimate + (N_ITERATIONS-1) reiterations (no final fit)."""
    f = OrthTrIM(transform_mode=mode, n_estimators=N_ESTIMATORS, lifetime=LIFETIME,
                 step_size=STEP_SIZE, random_state=random_state)
    f.X, f.y = np.array(x_tr), np.array(y_tr)
    f.mf.set_random_state()
    f.mf.fit(f.X, f.y)
    f.H = f.estimate_H_finite_diff()
    for _ in range(N_ITERATIONS - 1):
        f.reiterate()
    return f


# =============================================================
# One (d, rotation, seed): all n and all methods
# =============================================================
def run_cell(d, rotation, seed):
    np.random.seed(seed)
    x_train = np.random.rand(N_TOTAL, d) * 2 - 1
    x_test = np.random.rand(N_TEST, d) * 2 - 1
    fn = SimulatedData(d, ACTIVE, rotation=rotation, fun=FUN_STR, seed=seed)
    if WEIGHTS is not None:                      # y = sum_i w_i z_i^4
        fn.project = jnp.concatenate([jnp.array(WEIGHTS, dtype=float),
                                      jnp.zeros(d - ACTIVE)])
    y_train = np.array(vmap(fn.fun)(x_train))
    y_test = np.array(vmap(fn.fun)(x_test))
    true_H = np.array(fn.get_true_H(x_test))
    lam_true = sorted_eig(true_H)[0][:ACTIVE + 1]
    print(f"  [d={d} rot={rotation or 'aligned'} seed={seed}] true top eigenvalues: "
          f"{np.round(lam_true, 4).tolist()}", flush=True)

    rows, curves = [], {}
    for n in SAMPLE_RANGE:
        x_tr, y_tr = x_train[:n], y_train[:n]
        rs = (n + 1) * (seed + 1)

        # At 1 iteration both transforms share one H; otherwise each has its own.
        f_trim = estimate_H(x_tr, y_tr, "trim", rs)
        f_orth = f_trim if N_ITERATIONS == 1 else estimate_H(x_tr, y_tr, "orth", rs)
        A_orth = orth_matrix(f_orth.H)
        U_rot = sorted_eig(f_orth.H)[1]
        np.save(os.path.join(OUT_DIR, f"H_d{d}_{rotation or 'aligned'}_s{seed}_n{n}.npy"),
                f_trim.H)
        eig_ang = eig_angles_deg(f_trim.H, true_H, ACTIVE)
        lam_est = sorted_eig(f_trim.H)[0][:ACTIVE + 1]
        print(f"  n={n}: est top eigenvalues {np.round(lam_est, 4).tolist()}; "
              f"angle(u_i, true v_i) deg = {np.round(eig_ang, 1).tolist()}", flush=True)

        transforms = {
            "no transform": lambda X: X,
            "TrIM": lambda X: X @ (f_trim.H / two_one_norm(f_trim.H)),
            "orthogonal": lambda X: X @ A_orth,
            "rotation XU": lambda X: X @ U_rot,
        }
        H_of = {"no transform": None, "TrIM": f_trim.H, "orthogonal": f_orth.H,
                "rotation XU": f_orth.H}

        for m in METHODS:
            t0 = time.perf_counter()
            Xtr, Xte = transforms[m](x_tr), transforms[m](x_test)
            res, _ = evaluate_all_lifetimes(
                Xtr, y_tr, Xte, y_test, N_ESTIMATORS, LIFETIME,
                rng=np.random.RandomState(1000 + seed))   # same RNG for every method
            mse = np.asarray(res["mse"])
            i = int(np.argmin(mse))
            angle = (np.nan if H_of[m] is None
                     else float(get_angle_distance(H_of[m], true_H, ACTIVE)))
            sr = float((Xtr.max(0) - Xtr.min(0)).sum())
            rows.append(dict(d=d, rotation=rotation or "aligned", seed=seed, n=n,
                             method=m, best_mse=float(mse[i]),
                             best_lifetime=float(res["times"][i]),
                             mse_at_max_lifetime=float(mse[-1]),
                             angle=angle, sum_of_ranges=sr,
                             **{f"eig_angle_{i + 1}": float(eig_ang[i]) for i in range(ACTIVE)},
                             seconds=time.perf_counter() - t0))
            curves[(n, m)] = (np.asarray(res["times"]), mse)
            print(f"  d={d:>3} rot={rotation or 'aligned':>7} seed={seed} n={n:>5} "
                  f"{m:<13} best={mse[i]:.4e} @t={res['times'][i]:.2f} "
                  f"angle={angle:.4f} sum_ranges={sr:.2f}", flush=True)
    return rows, curves


def on_grid(times, mse, grid):
    """Step-interpolate an irregular MSE curve onto grid (NaN before first point)."""
    idx = np.searchsorted(times, grid, side="right") - 1
    out = mse[np.clip(idx, 0, None)].astype(float)
    out[idx < 0] = np.nan
    return out


# =============================================================
# Main
# =============================================================
def main():
    global DIMS, SAMPLE_RANGE, SEEDS, N_TOTAL, N_TEST, LIFETIME, OUT_DIR, WEIGHTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dims", type=str, help="e.g. 10 or 10,20")
    ap.add_argument("--seeds", type=str, help="e.g. 0 or 0,1,2,3,4")
    ap.add_argument("--weights", type=str, help="e.g. 1,0.5  (length ACTIVE)")
    ap.add_argument("--out", type=str, help="results subfolder name")
    args = ap.parse_args()
    if args.dims:
        DIMS = [int(v) for v in args.dims.split(",")]
    if args.seeds:
        SEEDS = [int(v) for v in args.seeds.split(",")]
    if args.weights:
        WEIGHTS = [float(v) for v in args.weights.split(",")]
        assert len(WEIGHTS) == ACTIVE
    if args.out:
        OUT_DIR = os.path.join("results", args.out)
    if args.smoke:
        DIMS, SAMPLE_RANGE, SEEDS = [5], [200], [0]
        N_TOTAL, N_TEST, LIFETIME = 200, 100, 2
        OUT_DIR = os.path.join("results", "orth_transform_smoke")
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"DIMS={DIMS} ROTATIONS={ROTATIONS} SAMPLE_RANGE={SAMPLE_RANGE} SEEDS={SEEDS}")
    print(f"ACTIVE={ACTIVE} LIFETIME={LIFETIME} N_ITERATIONS={N_ITERATIONS} "
          f"STEP_SIZE={STEP_SIZE} N_ESTIMATORS={N_ESTIMATORS} FUN_STR={FUN_STR} WEIGHTS={WEIGHTS}")

    grid = np.linspace(0, LIFETIME, 201)
    all_rows, grid_curves = [], {}
    csv_path = os.path.join(OUT_DIR, "per_seed.csv")
    for d in DIMS:
        for rot in ROTATIONS:
            for seed in SEEDS:
                rows, curves = run_cell(d, rot, seed)
                all_rows += rows
                pd.DataFrame(all_rows).to_csv(csv_path, index=False)  # save as we go
                for (n, m), (t, mse) in curves.items():
                    grid_curves.setdefault((d, rot, n, m), []).append(on_grid(t, mse, grid))

    df = pd.DataFrame(all_rows)
    summary = (df.groupby(["rotation", "d", "n", "method"], sort=False)
                 .agg(best_mse_mean=("best_mse", "mean"),
                      best_mse_std=("best_mse", "std"),
                      best_lifetime_mean=("best_lifetime", "mean"),
                      angle_mean=("angle", "mean"))
                 .reset_index())
    summary.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print("\n===== Summary (best MSE over lifetimes, mean ± std over seeds) =====")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4e}"))

    # One figure: rows = (rotation, n), cols = d; mean ± std over seeds.
    row_keys = [(rot, n) for rot in ROTATIONS for n in SAMPLE_RANGE]
    fig, axes = plt.subplots(len(row_keys), len(DIMS), squeeze=False,
                             figsize=(4.5 * len(DIMS), 3.4 * len(row_keys)))
    for r, (rot, n) in enumerate(row_keys):
        for c, d in enumerate(DIMS):
            ax = axes[r][c]
            for m in METHODS:
                Y = np.vstack(grid_curves[(d, rot, n, m)])
                mu, sd = np.nanmean(Y, 0), np.nanstd(Y, 0)
                ax.plot(grid, mu, color=COLORS[m], label=m, lw=2)
                ax.fill_between(grid, mu - sd, mu + sd, color=COLORS[m], alpha=0.15)
            ax.set_title(f"{rot or 'aligned'}, d={d}, n={n}", fontsize=11)
            ax.set_xlabel("Lifetime")
            ax.set_ylabel("Test MSE")
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-1, 1))
    axes[0][0].legend(fontsize=9)
    fig.suptitle(f"MSE vs lifetime (mean ± std over {len(SEEDS)} seeds, "
                 f"{N_ITERATIONS} iteration, weights={WEIGHTS or [1.0] * ACTIVE})", fontsize=13)
    fig.tight_layout()
    fig_path = os.path.join(OUT_DIR, "mse_vs_lifetime.png")
    fig.savefig(fig_path, dpi=130)
    print(f"\nSaved: {csv_path}, {os.path.join(OUT_DIR, 'summary.csv')}, {fig_path}")


if __name__ == "__main__":
    main()
