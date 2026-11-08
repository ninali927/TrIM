"""
eigen_utils.py
==============

Eigen-decomposition baseline for the TrIM / CSS comparison.

This module is imported by css_utils_trueH.py so that the *unchanged* main
(which calls evaluation_comparison_with_css) also produces the eigen comparison
figure "..._bestMSE_vs_k_eigen.png". To stay cheap, evaluation_comparison_with_eigen
REUSES the reference / CSS curves the CSS driver already computed and only adds
the genuinely-new work: the rank-k eigen projection of the *estimated* H plus one
Mondrian evaluation per k. Nothing is retrained, so total runtime is barely
affected. (This module does NOT import css_utils_trueH, so there is no circular
import.)

Eigen transform (estimated H only -- no true-H variant)
-------------------------------------------------------
  mode="kdim"  (default, apples-to-apples with CSS -- k output columns)
      A = U_k @ diag(Lambda_k ** weight_power)              # (d, k)
      Top-k eigenvectors scaled by their eigenvalues; k-dimensional output,
      directly comparable to CSS at the same effective k.

  mode="reconstruct"  (truncation diagnostic -- d output columns)
      A = U_k @ diag(Lambda_k ** weight_power) @ U_k^T      # (d, d)
      Rank-k spectral approximation in the original frame. At k=d, p=1 this
      equals H / two_one_norm(H) (the exact TrIM transform), so under Mondrian
      it reproduces full-TrIM MSE -- the clean k=d sentinel.

Normalization (identical to Weighted CSS, always on, no toggle)
---------------------------------------------------------------
Every transform is scaled by A -> A / two_one_norm(A). This codebase's
two_one_norm is the AVERAGE column L2 norm (= ||A||_{2,1,std} / ncols), so this
is exactly the per-output-dimension k/||A||_{2,1} normalization and leaves every
method with average column norm 1 (matching TrIM's H/two_one_norm(H)).
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from RandomForest.utils import evaluate_all_lifetimes, two_one_norm


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
# Eigen-decomposition selection + projection
# ===========================================================================

def eigen_select_topk(H, eigen_k):
    """
    Top-k symmetric eigendecomposition of H.

    Returns
    -------
    U_k    : (d, k) top-k eigenvectors (descending eigenvalue order)
    lam_k  : (k,)   corresponding eigenvalues, clipped to >= 0
    k      : effective number kept (min(eigen_k, d))
    """
    H = np.array(H, dtype=float)
    H = 0.5 * (H + H.T)
    d = H.shape[0]

    eigvals, eigvecs = np.linalg.eigh(H)          # ascending, real (symmetric)
    order = np.argsort(eigvals)[::-1]             # descending
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    k = int(min(eigen_k, d))
    if k <= 0:
        return np.zeros((d, 0)), np.zeros((0,)), 0

    U_k = eigvecs[:, :k]
    lam_k = np.clip(eigvals[:k], 0.0, None)        # H is PSD (Gram); guard noise
    return U_k, lam_k, k


def eigen_project_data(H, x_train, x_test, eigen_k, weight_power=1.0, mode="kdim"):
    """
    Project data onto the rank-k eigen-approximation of H. Always normalized by
    two_one_norm (identical to Weighted CSS) -- there is no normalize toggle.

    Parameters
    ----------
    eigen_k : int
        Number of top eigenpairs to keep. Pass CSS's effective k to compare the
        two methods at identical output dimension.
    weight_power : float, default 1.0
        Exponent on the eigenvalues. 1.0 matches TrIM's linear eigenbasis
        scaling; 0.0 -> pure orthonormal projection; 0.5 -> H^{1/2}-style.
    mode : {"kdim", "reconstruct"}, default "kdim"

    Returns
    -------
    x_train_proj, x_test_proj, A, lam_k
    """
    H = np.array(H, dtype=float)
    H = 0.5 * (H + H.T)

    U_k, lam_k, k = eigen_select_topk(H, eigen_k)
    if k == 0:
        return x_train.copy(), x_test.copy(), np.zeros((H.shape[0], 0)), lam_k

    scale = np.power(lam_k, weight_power)          # (k,)

    if mode == "kdim":
        A = U_k * scale                            # (d, k): col j scaled by scale[j]
    elif mode == "reconstruct":
        A = (U_k * scale) @ U_k.T                  # (d, d): U_k diag(scale) U_k^T
    else:
        raise ValueError(f"unknown mode={mode!r} "
                         f"(expected 'kdim' or 'reconstruct')")

    nA = two_one_norm(A)                           # average column norm
    if nA > 1e-14:
        A = A / nA                                 # -> average column norm 1

    return x_train @ A, x_test @ A, A, lam_k


# ===========================================================================
# k=d sentinel: reconstruct-mode eigen should reproduce the TrIM transform
# ===========================================================================

def check_eigen_kd_sentinel(H_final, x_tr, dim_in, weight_power=1.0):
    """
    Diagnostic (no Mondrian training). At k=d, reconstruct-mode A (p=1) equals
    the TrIM transform H/two_one_norm(H) to machine precision; kdim-mode differs
    only by the rotation U_k (Mondrian-only effect). Also surfaces the
    two_one_norm convention.
    """
    print(f"  --- check_eigen_kd_sentinel (k=d={dim_in}, "
          f"weight_power={weight_power}) ---")

    H = 0.5 * (H_final + H_final.T)
    M_trim = H / two_one_norm(H)

    col_sum = float(np.linalg.norm(H, axis=0).sum())
    t = two_one_norm(H) * H.shape[1]
    which = "columns" if np.isclose(t, col_sum) else "rows"
    print(f"    two_one_norm sums over {which} "
          f"(=> normalize divides by avg {which[:-1]} norm)")

    _, _, A_rec, _ = eigen_project_data(
        H, x_tr, x_tr, dim_in, weight_power=weight_power, mode="reconstruct")
    _, _, A_kdim, _ = eigen_project_data(
        H, x_tr, x_tr, dim_in, weight_power=weight_power, mode="kdim")

    rec_err = float(np.abs(A_rec - M_trim).max())
    print(f"    max|reconstruct(k=d) - TrIM transform| = {rec_err:.3e} "
          f"({'OK: reproduces TrIM' if rec_err < 1e-8 else 'WARN: check H PSD'})")

    def _sr(M):
        Z = x_tr @ M
        return float((Z.max(0) - Z.min(0)).sum())

    print(f"    sum_of_ranges: reconstruct={_sr(A_rec):.4f}  "
          f"kdim={_sr(A_kdim):.4f}  TrIM={_sr(M_trim):.4f}")
    print(f"  --- end check_eigen_kd_sentinel ---")

    return {"reconstruct_vs_trim_maxabs": rec_err}


# ===========================================================================
# Integration entry point (called by evaluation_comparison_with_css)
# ===========================================================================

# k-independent reference methods (drawn as horizontal lines on the eigen plot).
_K_INDEP = [
    ("no transformation",   "tab:blue"),
    ("transform by true H", "tab:orange"),
    ("TrIM",                "tab:green"),
    ("Weighted Mondrian",   "tab:red"),
]
# k-dependent curves on the eigen plot. No Eigen (true H) -- estimated H only.
# CSS curves are reused from the CSS driver; only Eigen (estimated H) is new.
_K_DEP = [
    ("CSS (estimated H)",   "tab:purple", "-"),
    ("CSS (true H)",        "tab:pink",   "-"),
    ("Eigen (estimated H)", "tab:brown",  "--"),
]


def evaluation_comparison_with_eigen(
    *,
    H_final, x_tr, y_tr, x_test, y_test,
    n_estimators, lifetime, dim_in,
    css_k_list, css_eff_k,
    k_indep_curves, per_k_curves,
    n_sim, save_path,
    weight_power=1.0, eigen_mode="kdim",
):
    """
    Add the eigen-decomposition baseline (estimated H) to an ALREADY-computed
    per-sample-size comparison, then write the "..._bestMSE_vs_k_eigen.png"
    figure. Designed to be called once per sample size from inside
    css_utils_trueH.evaluation_comparison_with_css, reusing the reference and
    CSS curves it has already produced (no retraining).

    Parameters
    ----------
    H_final : (d, d)
        The estimated EGOP H used by TrIM / CSS for this sample size.
    x_tr, y_tr, x_test, y_test :
        Train/test data for this sample size (x_tr already sliced to n_sim).
    css_k_list : list[int]
        The k values swept by the CSS driver.
    css_eff_k : dict[int, int]
        Effective k actually selected by CSS for each css_k value (len(S_css)).
        Eigen uses this exact k so the two methods match in output dimension.
    k_indep_curves : dict[label -> result]
        The four reference curves already computed by the CSS driver.
    per_k_curves : dict[k -> {label -> result}]
        CSS curves already computed by the CSS driver. This function ADDS
        "Eigen (estimated H)" into each per_k_curves[k] in place.
    n_sim, save_path, weight_power, eigen_mode :
        As in the CSS driver. eigen_mode="kdim" (k-dim, like CSS) or
        "reconstruct" (d-dim truncation; reproduces TrIM at k=d).

    Returns
    -------
    per_k_curves : the same dict, now including the Eigen (estimated H) curves.
    """
    print(f"\n  ===== eigen baseline (n={n_sim}, eigen_mode={eigen_mode}) =====")
    check_eigen_kd_sentinel(H_final, x_tr, dim_in, weight_power=weight_power)

    for css_k_val in css_k_list:
        k_eff = int(css_eff_k.get(css_k_val, min(css_k_val, dim_in)))

        x_tr_eig, x_te_eig, _, _ = eigen_project_data(
            H_final, x_tr, x_test, eigen_k=k_eff,
            weight_power=weight_power, mode=eigen_mode,
        )
        _print_data_stats(x_tr_eig, f"Eigen(est) x_tr k={k_eff}")
        res_eig, _ = evaluate_all_lifetimes(
            x_tr_eig, y_tr, x_te_eig, y_test, n_estimators, lifetime)
        per_k_curves[css_k_val]["Eigen (estimated H)"] = res_eig
        print(f"  [k={css_k_val}] Eigen (est H) best MSE: "
              f"{float(np.min(res_eig['mse'])):.6e}")

        # per-(n,k) plot: references + CSS(est/true) + Eigen(est)
        if save_path is not None:
            fig, ax = plt.subplots(figsize=(7.5, 5))
            for label, color in _K_INDEP:
                ax.plot(k_indep_curves[label]["times"],
                        k_indep_curves[label]["mse"],
                        label=label, color=color, linewidth=2)
            for label, color, style in _K_DEP:
                curve = per_k_curves[css_k_val].get(label)
                if curve is None:
                    continue
                ax.plot(curve["times"], curve["mse"], label=label,
                        color=color, linewidth=2, linestyle=style)
            ax.set_title(
                f"n={n_sim}, k={css_k_val} "
                f"(w_pow={weight_power}, eigen={eigen_mode})", fontsize=13)
            ax.set_xlabel("Lifetime", fontsize=14)
            ax.set_ylabel("MSE", fontsize=14)
            ax.tick_params(labelsize=12)
            ax.legend(fontsize=9)
            _format_axis(ax)
            fig.tight_layout()
            fig.savefig(f"{save_path}_n{n_sim}_eigk{css_k_val}.png", dpi=130)
            plt.close(fig)

    # best MSE vs k summary -> the "..._bestMSE_vs_k_eigen.png" figure.
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
            ref = float(np.min(k_indep_curves[label]["mse"]))
            ax.axhline(ref, color=color, linewidth=1.5, linestyle=":",
                       label=f"{label} (ref)")
        ax.set_title(
            f"Best MSE vs k  (n={n_sim}, w_pow={weight_power}, "
            f"eigen={eigen_mode})", fontsize=13)
        ax.set_xlabel("k", fontsize=14)
        ax.set_ylabel("Best MSE over lifetimes", fontsize=14)
        ax.set_xticks(ks)
        ax.tick_params(labelsize=12)
        ax.legend(fontsize=9, loc="best")
        _format_axis(ax)
        fig.tight_layout()
        fig.savefig(f"{save_path}_n{n_sim}_bestMSE_vs_k_eigen.png", dpi=130)
        plt.close(fig)

    return per_k_curves