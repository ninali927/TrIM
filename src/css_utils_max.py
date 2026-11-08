"""
css_utils_max.py
================

k-sweep CSS baseline (top-diagonal selection) + Eigen-decomposition baseline,
for the aligned / no-rotation experiment. Structured to mirror the rotation
pipeline (css_utils_trueH) so the aligned and rotated drivers stay as similar
as possible.

CSS here = the ORIGINAL top-diagonal feature selection:
    pick the css_k original coordinates with the largest diagonal entries of the
    estimated H, keep those coordinates (no rotation of the data).

Eigen here = rank-k eigen-projection of the SAME estimated H (reused from
eigen_utils.eigen_project_data), at the SAME effective k as CSS, so the two are
compared at identical output dimension.

What this prints (kept from the original + new diagnostics):
    - estimated H diagonal + top-diagonal indices
    - estimated H eigenvalues (descending)            [NEW]
    - per-method TrIM runtime + sum_of_ranges
    - CSS selected columns, runtime, sum_of_ranges
    - CSS-vs-Eigen selection overlap diagnostic        [NEW]
      (in the aligned case these often coincide; this quantifies how much)
    - Eigen per-k best MSE + k=d sentinel
    - "Best MSE over max_depth" table for every method, every k

Figures:
    - one 6-method comparison figure per (n, k)
    - one "best MSE vs k" summary figure per n
"""

import os
import numpy as np
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from tqdm import tqdm

# Downstream learner selector (mirrors the kNN swap). TrIM is re-exported by
# each backend, so the EGOP (H) estimation stays Mondrian-based either way;
# only evaluate_all_lifetimes (the regression head) changes.
#
# Mondrian (original):
# from Mondrian_RF.Mondrian_forest import TrIM
# from Mondrian_RF.utils import evaluate_all_lifetimes, two_one_norm
#
# Standard Random Forest (this experiment):
from RandomForest.RandomForest import TrIM
from RandomForest.utils import evaluate_all_lifetimes, two_one_norm

from eigen_utils import eigen_select_topk, eigen_project_data, check_eigen_kd_sentinel


# ===========================================================================
# small shared helpers
# ===========================================================================

def _print_data_stats(X, label):
    """Print bounding-box stats: sum_of_ranges drives the Mondrian cut rate."""
    X = np.asarray(X)
    ranges = X.max(axis=0) - X.min(axis=0)
    print(
        f"  [{label}] shape={X.shape}, "
        f"sum_of_ranges={float(ranges.sum()):.4f}, "
        f"per_dim_ranges={np.round(ranges, 3).tolist()}"
    )


def _format_axis(ax):
    yf = ScalarFormatter(useOffset=False)
    yf.set_scientific(True)
    yf.set_powerlimits((-1, 1))
    ax.yaxis.set_major_formatter(yf)


# ===========================================================================
# Top-diagonal CSS (original logic, unchanged) + selection diagnostic
# ===========================================================================

def css_project_data(H, x_train, x_test, css_k):
    """
    Select the css_k original coordinates with the largest diagonal entries of
    H, then keep those coordinates. (No rotation of the data: output stays
    strictly axis-aligned.)

    Returns
    -------
    x_train_proj : (n_train, k)
    x_test_proj  : (n_test, k)
    S            : selected coordinate indices (descending diagonal)
    k            : effective number kept (min(css_k, d))
    """
    H = np.array(H, dtype=float)
    H = 0.5 * (H + H.T)

    d = H.shape[0]
    k = int(min(css_k, d))

    scores = np.diag(H)
    S = np.argsort(scores)[::-1][:k]
    return x_train[:, S], x_test[:, S], S, k


def _css_eigen_overlap(H, S_css, k_eff):
    """
    Diagnostic: how much do top-diagonal CSS and top-k eigen agree?

    In the aligned case the estimated H is near-diagonal, so the top-k
    eigenvectors are nearly standard basis vectors and the two methods often
    select the same coordinates. This quantifies it:
      - |S_css ∩ supp(eigen)| via each eigenvector's dominant coordinate
      - max angle of each top-k eigenvector away from its nearest axis (rad)
      - off-diag / diag Frobenius ratio of H (0 = exactly diagonal)
    """
    H = 0.5 * (np.array(H, float) + np.array(H, float).T)
    U_k, lam_k, k = eigen_select_topk(H, k_eff)

    # dominant coordinate of each top-k eigenvector
    dom = np.argmax(np.abs(U_k), axis=0) if k > 0 else np.array([], dtype=int)
    overlap = len(set(S_css.tolist()) & set(dom.tolist()))

    # angle of each eigenvector from its nearest coordinate axis
    if k > 0:
        max_abs = np.max(np.abs(U_k), axis=0)            # |cos| to nearest axis
        max_abs = np.clip(max_abs, 0.0, 1.0)
        angles = np.degrees(np.arccos(max_abs))
        max_angle = float(angles.max())
    else:
        max_angle = 0.0

    diag_fro = float(np.linalg.norm(np.diag(np.diag(H))))
    off_fro = float(np.linalg.norm(H - np.diag(np.diag(H))))
    off_ratio = off_fro / diag_fro if diag_fro > 1e-14 else 0.0

    print(f"    --- CSS vs Eigen selection diagnostic (k={k}) ---")
    print(f"      CSS top-diag columns      : {np.asarray(S_css).tolist()}")
    print(f"      Eigen dominant coords     : {dom.tolist()}")
    print(f"      selection overlap         : {overlap}/{k}")
    print(f"      max eigvec angle from axis: {max_angle:.3f} deg "
          f"(0 = perfectly axis-aligned)")
    print(f"      H off-diag/diag Frobenius : {off_ratio:.4f} "
          f"(0 = exactly diagonal)")
    print(f"    --- end CSS vs Eigen diagnostic ---")


# ===========================================================================
# plotting metadata
# ===========================================================================

_K_INDEP = [
    ("no transformation",   "tab:blue"),
    ("transform by true H", "tab:orange"),
    ("TrIM",                "tab:green"),
    ("Weighted Mondrian",   "tab:red"),
]
_K_DEP = [
    ("CSS (estimated H)",   "tab:purple", "-"),
    ("Eigen (estimated H)", "tab:brown",  "--"),
]


# ===========================================================================
# Main entry point: paper's 4 reference methods + CSS + Eigen, swept over k
# ===========================================================================

def evaluation_comparison_with_css(
    *,
    x_train, y_train, x_test, y_test, true_H,
    dim_in, active,
    n_estimators, lifetime, step_size,
    sample_range,
    save_path=None,
    css_random_state=0,
    css_k_list=None,
    n_iterations=1,
    weight_power=1.0,
    eigen_mode="kdim",
):
    """
    Aligned-case evaluation: 4 reference methods (k-independent) + top-diagonal
    CSS (k-dependent) + Eigen (k-dependent), swept over css_k_list. Mirrors the
    rotation pipeline's signature so no_rotation_main and rotation_main differ
    only by ROTATION_MODE.

    Parameters
    ----------
    css_k_list : list[int]
        k values to sweep. Pass [active] for a single k.
    n_iterations : int
        Number of EGOP iterations (1 = initial fit only). Kept for signature
        parity with the rotation pipeline; reiteration uses forest.reiterate().
    weight_power : float
        Eigenvalue exponent for the Eigen projection (1.0 = TrIM-style linear
        eigenbasis scaling). Does not affect top-diagonal CSS.
    eigen_mode : {"kdim", "reconstruct"}
        "kdim" -> k output columns (apples-to-apples with CSS).
    """
    all_curves = {}

    if css_k_list is None:
        css_k_list = [active]
    css_k_list = list(css_k_list)

    print(f"CSS_K_LIST = {css_k_list}")
    print(f"CSS random_state = {css_random_state}")
    print(f"N_ITERATIONS = {n_iterations}")
    print(f"WEIGHT_POWER = {weight_power}")

    for n_sim in tqdm(sample_range, desc="sample sizes"):
        x_tr = x_train[:n_sim]
        y_tr = y_train[:n_sim]
        curves = {}
        per_k_curves = {k: {} for k in css_k_list}

        print(f"\nn = {n_sim}")

        # ---------- 1. no transformation (k-independent) ----------
        res_no, _ = evaluate_all_lifetimes(
            x_tr, y_tr, x_test, y_test, n_estimators, lifetime)
        curves["no transformation"] = res_no

        # ---------- 2. transform by true H (k-independent) ----------
        nH = true_H / two_one_norm(true_H)
        res_true, _ = evaluate_all_lifetimes(
            x_tr @ nH, y_tr, x_test @ nH, y_test, n_estimators, lifetime)
        curves["transform by true H"] = res_true

        # ---------- Train ONE TrIM forest, get estimated H ----------
        forest = TrIM(
            n_estimators=n_estimators,
            lifetime=lifetime,
            step_size=step_size,
            random_state=(n_sim + 1) * 2,
        )
        forest.X = np.array(x_tr)
        forest.y = np.array(y_tr)
        forest.mf.set_random_state()
        forest.mf.fit(forest.X, forest.y)
        importance = forest._get_importance()
        forest.H = (importance @ importance.T) / x_tr.shape[0]

        # optional EGOP reiteration for parity with the rotation pipeline
        for _ in range(2, n_iterations + 1):
            forest.reiterate()
        H_hat = forest.H.copy()

        eigvals_desc = np.sort(np.linalg.eigvalsh(0.5 * (H_hat + H_hat.T)))[::-1]
        print("estimated H diagonal:", np.diag(H_hat))
        print("estimated H top diagonal indices:",
              np.argsort(np.diag(H_hat))[::-1])
        print("estimated H eigenvalues descending:", eigvals_desc)

        # ---------- 3. TrIM (k-independent) ----------
        _trim_start = time.perf_counter()
        _trim_xtr = forest.transform(x_tr)
        _trim_xte = forest.transform(x_test)
        _trim_transform_time = time.perf_counter() - _trim_start
        _print_data_stats(_trim_xtr, "TrIM x_tr")
        _trim_mf_start = time.perf_counter()
        res_trim, _ = evaluate_all_lifetimes(
            _trim_xtr, y_tr, _trim_xte, y_test, n_estimators, lifetime)
        _trim_mf_time = time.perf_counter() - _trim_mf_start
        _trim_runtime = _trim_transform_time + _trim_mf_time
        print(f"TrIM runtime (n={n_sim}): {_trim_runtime:.4f} s")
        print(f"  TrIM transform time : {_trim_transform_time:.4f} s")
        print(f"  TrIM RF time        : {_trim_mf_time:.4f} s")
        curves["TrIM"] = res_trim

        # ---------- 4. Weighted Mondrian (k-independent) ----------
        forest.H = np.diag(np.sum(importance ** 2, axis=1)) / x_tr.shape[0]
        res_wmf, _ = evaluate_all_lifetimes(
            forest.transform(x_tr), y_tr,
            forest.transform(x_test), y_test,
            n_estimators, lifetime)
        curves["Weighted Mondrian"] = res_wmf
        forest.H = H_hat  # restore estimated H

        # ---------- 5. CSS + Eigen sweep over k ----------
        css_eff_k = {}
        summary_rows = []   # (k_eff, css_mse, eig_mse, css_runtime, eig_runtime)
        for css_k_val in css_k_list:
            print(f"\n  ===== k sweep: k = {css_k_val} (n = {n_sim}) =====")

            # --- CSS (top-diagonal) ---
            _css_start = time.perf_counter()
            x_tr_c, x_te_c, S_css, k_eff = css_project_data(
                H_hat, x_tr, x_test, css_k=css_k_val)
            _css_proj_time = time.perf_counter() - _css_start
            css_eff_k[css_k_val] = k_eff
            print(f"  [k={css_k_val}] CSS top-diagonal selected columns "
                  f"(effective k={k_eff}): {S_css.tolist()}")
            _print_data_stats(x_tr_c, f"CSS x_tr k={css_k_val}")
            _css_mf_start = time.perf_counter()
            res_css, _ = evaluate_all_lifetimes(
                x_tr_c, y_tr, x_te_c, y_test, n_estimators, lifetime)
            _css_mf_time = time.perf_counter() - _css_mf_start
            _css_runtime = _css_proj_time + _css_mf_time
            print(f"  [k={css_k_val}] CSS runtime (n={n_sim}): "
                  f"{_css_runtime:.4f} s "
                  f"(proj {_css_proj_time:.4f} + rf {_css_mf_time:.4f})")
            per_k_curves[css_k_val]["CSS (estimated H)"] = res_css

            # --- CSS-vs-Eigen selection overlap (aligned-case diagnostic) ---
            _css_eigen_overlap(H_hat, S_css, k_eff)

            # --- Eigen (estimated H), same effective k ---
            _eig_start = time.perf_counter()
            x_tr_e, x_te_e, _, _ = eigen_project_data(
                H_hat, x_tr, x_test, eigen_k=k_eff,
                weight_power=weight_power, mode=eigen_mode)
            _eig_proj_time = time.perf_counter() - _eig_start
            _print_data_stats(x_tr_e, f"Eigen(est) x_tr k={k_eff}")
            _eig_mf_start = time.perf_counter()
            res_eig, _ = evaluate_all_lifetimes(
                x_tr_e, y_tr, x_te_e, y_test, n_estimators, lifetime)
            _eig_mf_time = time.perf_counter() - _eig_mf_start
            _eig_runtime = _eig_proj_time + _eig_mf_time
            print(f"  [k={css_k_val}] Eigen runtime (n={n_sim}): "
                  f"{_eig_runtime:.4f} s "
                  f"(proj {_eig_proj_time:.4f} + rf {_eig_mf_time:.4f})")
            per_k_curves[css_k_val]["Eigen (estimated H)"] = res_eig

            print(f"  [k={css_k_val}] Best MSE over max_depth:")
            for label, _c in _K_INDEP:
                print(f"    {label:22s}: {float(np.min(curves[label]['mse'])):.6e}")
            print(f"    {'CSS (estimated H)':22s}: "
                  f"{float(np.min(res_css['mse'])):.6e}")
            print(f"    {'Eigen (estimated H)':22s}: "
                  f"{float(np.min(res_eig['mse'])):.6e}")

            summary_rows.append((
                k_eff,
                float(np.min(res_css['mse'])),
                float(np.min(res_eig['mse'])),
                _css_runtime,
                _eig_runtime,
            ))

        # ---------- compact summary table ----------
        print(f"\n  ===== Compact summary: n = {n_sim} =====")
        print("  k  | CSS best MSE | Eigen best MSE | CSS/Eigen | "
              "CSS runtime | Eigen runtime")
        print("  " + "-" * 78)
        for k_eff, css_mse, eig_mse, css_rt, eig_rt in summary_rows:
            ratio = css_mse / eig_mse if eig_mse > 0 else float("inf")
            print(f"  {k_eff:<3d}| {css_mse:12.6e} | {eig_mse:14.6e} | "
                  f"{ratio:9.4f} | {css_rt:9.4f} s | {eig_rt:9.4f} s")
        print("  " + "-" * 78)

        # ---------- k=d eigen sentinel ----------
        print(f"\n  ===== eigen sentinel (n={n_sim}, eigen_mode={eigen_mode}) =====")
        check_eigen_kd_sentinel(H_hat, x_tr, dim_in, weight_power=weight_power)

        all_curves[n_sim] = {"ref": curves, "per_k": per_k_curves,
                             "eff_k": css_eff_k}

        # ---------- per-(n,k) comparison figures ----------
        if save_path is not None:
            for css_k_val in css_k_list:
                fig, ax = plt.subplots(figsize=(7.5, 5))
                for label, color in _K_INDEP:
                    ax.plot(curves[label]["times"], curves[label]["mse"],
                            label=label, color=color, linewidth=2)
                for label, color, style in _K_DEP:
                    cv = per_k_curves[css_k_val].get(label)
                    if cv is None:
                        continue
                    ax.plot(cv["times"], cv["mse"], label=label,
                            color=color, linewidth=2, linestyle=style)
                ax.set_title(
                    f"n={n_sim}, k={css_k_val} "
                    f"(w_pow={weight_power}, eigen={eigen_mode})", fontsize=13)
                ax.set_xlabel("max_depth", fontsize=14)
                ax.set_ylabel("MSE", fontsize=14)
                ax.tick_params(labelsize=12)
                ax.legend(fontsize=9)
                _format_axis(ax)
                fig.tight_layout()
                fig.savefig(f"{save_path}_n{n_sim}_k{css_k_val}.png", dpi=130)
                plt.close(fig)

        # ---------- best MSE vs k summary figure ----------
        if save_path is not None and len(css_k_list) > 1:
            fig, ax = plt.subplots(figsize=(8, 5.5))
            ks = list(css_k_list)
            for label, color, style in _K_DEP:
                if any(label not in per_k_curves[k] for k in ks):
                    continue
                best = [float(np.min(per_k_curves[k][label]["mse"])) for k in ks]
                ax.plot(ks, best, marker="o", color=color, linewidth=2,
                        linestyle=style, label=label)
            for label, color in _K_INDEP:
                ref = float(np.min(curves[label]["mse"]))
                ax.axhline(ref, color=color, linewidth=1.5, linestyle=":",
                           label=f"{label} (ref)")
            ax.set_title(
                f"Best MSE vs k  (n={n_sim}, w_pow={weight_power}, "
                f"eigen={eigen_mode})", fontsize=13)
            ax.set_xlabel("k", fontsize=14)
            ax.set_ylabel("Best MSE over max_depth", fontsize=14)
            ax.set_xticks(ks)
            ax.tick_params(labelsize=12)
            ax.legend(fontsize=9, loc="best")
            _format_axis(ax)
            fig.tight_layout()
            fig.savefig(f"{save_path}_n{n_sim}_bestMSE_vs_k.png", dpi=130)
            plt.close(fig)

    return all_curves