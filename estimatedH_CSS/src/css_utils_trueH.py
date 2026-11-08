"""
css_utils_weighted.py
======================

CSS baseline using the WEIGHTED CSS variant only (the unweighted /
variance-standardized CSS has been removed).

Weighted CSS:
    RPQR column selection on H  ->  orthonormal basis U of the selected
    columns  ->  restore H's magnitude information via an H^{weight_power}
    transform restricted to the selected subspace:

        H_proj = U^T H U                     # (k x k) restriction of H to span(U)
        H_proj = V Λ V^T                     # symmetric eigendecomposition
        W = U @ V @ diag(Λ^{weight_power})   # weight_power=1.0 -> matches TrIM's
                                             #   per-eigendirection scaling (x @ Λ)
        x -> x @ W

    W is normalized by its (2,1)-norm, mirroring the scale control used by
    "transform by true H" (nH = true_H / two_one_norm(true_H)) and by TrIM
    (H / two_one_norm(H)).

Why weight_power=1.0 matches TrIM:
    TrIM.transform is x @ (H / two_one_norm(H)) -- a *linear* map. In H's
    eigenbasis this scales coordinate i by eigenvalue λ_i. Weighted CSS's
    transform W = U V diag(Λ^p) scales coordinate i by λ_i^p, so p=1.0
    reproduces TrIM's scaling shape (up to the global scalar fixed by the
    differing 2,1-norm denominators -- see check_kd_equivalence, which
    quantifies that residual scalar at k=d).

Methods plotted (Weighted CSS now occupies the slot previously labelled
"CSS (estimated H)"):
    1. no transformation
    2. transform by true H
    3. TrIM
    4. Weighted Mondrian
    5. CSS (estimated H)   -- now produced by Weighted CSS on the estimated H
    6. CSS (true H)        -- now produced by Weighted CSS on the true H
"""

import numpy as np
import time
from numpy.linalg import qr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from tqdm import tqdm

from Mondrian_RF.Mondrian_forest import TrIM
from Mondrian_RF.utils import evaluate_all_lifetimes, two_one_norm
# from kNN_Euclidean.kNN import TrIM
# from kNN_Euclidean.utils import evaluate_all_lifetimes, two_one_norm

# Eigen baseline. Imported here so the UNCHANGED main triggers it via
# evaluation_comparison_with_css. eigen_utils does NOT import this module, so
# there is no circular import.
from eigen_utils import evaluation_comparison_with_eigen


def _print_data_stats(X, label):
    """Print bounding-box stats: sum_of_ranges drives the Mondrian cut rate."""
    X = np.asarray(X)
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    ranges = hi - lo
    print(
        f"  [{label}] shape={X.shape}, "
        f"sum_of_ranges={float(ranges.sum()):.4f}, "
        f"per_dim_ranges={np.round(ranges, 3).tolist()}"
    )


# ===========================================================================
# k=d equivalence check: TrIM vs Weighted CSS transform geometry
# ===========================================================================

def check_kd_equivalence(H_final, x_tr, dim_in, weight_power=1.0):
    """
    Diagnostic (no Mondrian training): at k=d, compare the TrIM transform
    M_trim = H / two_one_norm(H) against the Weighted CSS transform
    W = U V diag(Λ^weight_power) / two_one_norm(W).

    Prints:
      - singular values of each transform (the per-eigendirection scaling),
      - their ratio (a near-constant vector means the two transforms differ
        only by a global scalar -> identical *shape*, MSE curve only shifts
        along the lifetime axis),
      - std/mean of that ratio (closer to 0 = more purely a global scalar),
      - sum_of_ranges of x_tr under each transform (this is what actually
        drives the Mondrian cut rate, so similar values predict similar MSE
        at a given lifetime).

    All output goes through print(), so it lands in the run log alongside the
    per-sample-size experiment output.
    """
    print(f"  --- check_kd_equivalence (k=d={dim_in}, weight_power={weight_power}) ---")

    H = 0.5 * (H_final + H_final.T)

    # --- TrIM transform matrix ---
    M_trim = H / two_one_norm(H)

    # --- Weighted CSS at k=d (all columns selected) ---
    C = H[:, :dim_in]
    U, _ = qr(C)
    H_proj = 0.5 * (U.T @ H @ U + (U.T @ H @ U).T)
    eigvals_p, V = np.linalg.eigh(H_proj)
    eigvals_p = np.maximum(eigvals_p, 0.0)
    W = U @ V @ np.diag(eigvals_p ** weight_power)
    w_norm = two_one_norm(W)
    if w_norm > 1e-14:
        W = W / w_norm

    # Singular values = per-eigendirection scaling of each transform.
    s_trim = np.sort(np.linalg.svd(M_trim, compute_uv=False))[::-1]
    s_wcss = np.sort(np.linalg.svd(W, compute_uv=False))[::-1]
    ratio = s_wcss / np.maximum(s_trim, 1e-12)
    rel_spread = float(np.std(ratio) / np.mean(ratio)) if np.mean(ratio) != 0 else float("nan")

    print(f"    TrIM singular values : {np.round(s_trim, 6).tolist()}")
    print(f"    WCSS singular values : {np.round(s_wcss, 6).tolist()}")
    print(f"    ratio (const => only a global scalar differs): "
          f"{np.round(ratio, 4).tolist()}")
    print(f"    ratio std/mean (->0 = pure global scalar): {rel_spread:.6e}")

    # sum_of_ranges: the quantity that actually drives the cut rate.
    def _sum_ranges(M):
        Z = x_tr @ M
        return float((Z.max(0) - Z.min(0)).sum())

    sr_trim = _sum_ranges(M_trim)
    sr_wcss = _sum_ranges(W)
    print(f"    TrIM transformed sum_of_ranges : {sr_trim:.6f}")
    print(f"    WCSS transformed sum_of_ranges : {sr_wcss:.6f}")
    if sr_trim > 1e-14:
        print(f"    sum_of_ranges ratio (WCSS/TrIM): {sr_wcss / sr_trim:.6f}")
    print(f"    => similar sum_of_ranges predicts similar MSE at k=d "
          f"(same lifetime => comparable cut rate)")
    print(f"  --- end check_kd_equivalence ---")

    return {
        "s_trim": s_trim,
        "s_wcss": s_wcss,
        "ratio": ratio,
        "ratio_rel_spread": rel_spread,
        "sum_ranges_trim": sr_trim,
        "sum_ranges_wcss": sr_wcss,
    }


# ===========================================================================
# Randomized RPQR CSS / subspace selection
# ===========================================================================

def _rpqr(Q, k, rng):
    """
    Randomly-pivoted QR on Q^T to select k row indices of Q.
    Returns the list of selected indices.
    """
    M = Q.T.copy()
    S = []

    for _ in range(k):
        col_norms_sq = np.sum(M ** 2, axis=0)
        total = col_norms_sq.sum()

        if total < 1e-14:
            break

        probs = col_norms_sq / total
        j = int(rng.choice(M.shape[1], p=probs))
        S.append(j)

        v = M[:, j:j + 1]
        denom = float((v.T @ v).item())

        if denom > 1e-14:
            M -= v @ (v.T @ M) / denom

        M[:, j] = 0

    return S


def _css_select_basis(H, css_k, random_state=0):
    """
    RPQR column-subset selection on H.

    Returns
    -------
    U : (d, k) orthonormal basis of the selected H columns
    S : selected column indices
    k : effective number of selected columns (may be < css_k)
    """
    rng = np.random.RandomState(random_state)

    H = np.array(H, dtype=float)
    H = 0.5 * (H + H.T)

    d = H.shape[0]
    k = min(css_k, d, np.linalg.matrix_rank(H))

    if k == 0:
        return np.zeros((d, 0)), [], 0

    Omega = rng.randn(d, k)
    B = H @ Omega

    Q, _ = qr(B)
    Q = Q[:, :k]

    S = _rpqr(Q, k, rng)

    if len(S) < k:
        remaining = [j for j in range(d) if j not in S]
        S = S + remaining[:(k - len(S))]

    C = H[:, S]
    U, _ = qr(C)
    U = U[:, :k]

    return U, S, k


def css_project_data_weighted(
    H, x_train, x_test, css_k, random_state=0, weight_power=1.0
):
    """
    WEIGHTED CSS.

    RPQR column selection on H, then an H-magnitude-preserving transform
    restricted to the selected subspace:

        H_proj = U^T H U                     # (k, k) restriction of H to span(U)
        H_proj = V Λ V^T                     # symmetric eigendecomposition
        W = U @ V @ diag(Λ^{weight_power})
        x -> x @ W

    Rotating by V aligns the projected coordinates with H's eigendirections
    *within* the subspace (U's columns are not themselves eigenvectors of H,
    so this rotation is necessary for the per-coordinate scaling to be
    meaningful). Scaling by Λ^{weight_power} reproduces, on the retained
    subspace, the magnitude weighting that gives TrIM / "transform by true H"
    their advantage. weight_power=1.0 matches TrIM's linear x @ H scaling in
    H's eigenbasis.

    W is normalized by its (2,1)-norm so the overall data scale matches the
    other H-based transforms, while the relative per-direction weighting is
    preserved.

    Parameters
    ----------
    weight_power : float, default 1.0
        Exponent applied to the subspace eigenvalues. 1.0 weights linearly by
        H (matches TrIM's eigenbasis scaling); 0.5 gives an H^{1/2}-style
        transform; 0.0 recovers a pure orthonormal projection (no magnitude
        weighting).

    Returns
    -------
    x_train_proj : (n_train, k)
    x_test_proj  : (n_test,  k)
    W            : (d, k)  the weighted transform (after 2,1-norm scaling)
    S            : selected column indices
    """
    H = np.array(H, dtype=float)
    H = 0.5 * (H + H.T)

    U, S, k = _css_select_basis(H, css_k, random_state=random_state)

    if k == 0:
        return x_train.copy(), x_test.copy(), U, S

    # Restrict H to the selected subspace and align to its eigenbasis.
    H_proj = U.T @ H @ U
    H_proj = 0.5 * (H_proj + H_proj.T)
    eigvals_p, V = np.linalg.eigh(H_proj)            # ascending eigenvalues
    eigvals_p = np.maximum(eigvals_p, 0.0)           # clip tiny negatives

    scale = eigvals_p ** weight_power
    # Guard against zero-scale collapsing a direction entirely.
    scale = np.where(scale < 1e-12, 1e-12, scale)

    W = U @ V @ np.diag(scale)                       # (d, k)

    # Normalize overall scale the same way the true-H transform does, keeping
    # relative per-direction weighting intact.
    w_norm = two_one_norm(W)
    if w_norm > 1e-14:
        W = W / w_norm

    x_tr_proj = x_train @ W
    x_te_proj = x_test @ W

    return x_tr_proj, x_te_proj, W, S


# ===========================================================================
# Main entry point: paper's 4-method evaluation + Weighted CSS
# ===========================================================================

_PLOT_STYLE = [
    ("no transformation",   "tab:blue"),
    ("transform by true H", "tab:orange"),
    ("TrIM",                "tab:green"),
    ("Weighted Mondrian",   "tab:red"),
    ("CSS (estimated H)",   "tab:purple"),
    ("CSS (true H)",        "tab:pink"),
]


def _format_axis(ax):
    yf = ScalarFormatter(useOffset=False)
    yf.set_scientific(True)
    yf.set_powerlimits((-1, 1))
    ax.yaxis.set_major_formatter(yf)


def evaluation_comparison_with_css(
    *,
    x_train, y_train, x_test, y_test, true_H,
    dim_in, active,
    n_estimators, lifetime, step_size,
    sample_range,
    save_path=None,
    css_random_state=0,
    css_k=None,
    css_k_list=None,
    n_iterations=1,
    weight_power=1.0,
):
    """
    Replicates the paper's evaluation; CSS is Weighted CSS, swept over k.

    k-independent methods (computed once per sample size):
        1. no transformation
        2. transform by true H
        3. TrIM                (uses H after n_iterations EGOP iterations)
        4. Weighted Mondrian   (diagonal of H after n_iterations EGOP iterations)

    k-dependent methods (recomputed for each k in css_k_list):
        5. CSS (estimated H)   (Weighted CSS on the estimated H)
        6. CSS (true H)        (Weighted CSS on the true H)

    For each k, the full RPQR pipeline runs independently:
        Omega = rng.randn(d, k); B = H @ Omega; Q = qr(B); S = _rpqr(Q, k);
        C = H[:, S]; U = qr(C); W = U V diag(Λ^weight_power).
    NOTE: RPQR selection is NOT nested -- the columns chosen at k=10 are not a
    superset of those at k=3, so best-MSE-vs-k need not be monotone. Only at
    k=d (k >= dim_in, capped by matrix_rank) does selection become trivial
    (all columns), removing the selection randomness.

    Plots produced (when save_path is given):
        - one 6-method comparison figure per (n_sim, k)
        - one "best MSE vs k" summary figure per n_sim: k-dependent methods
          drawn as lines over k, k-independent methods as horizontal reference
          lines (TrIM, true-H transform, Weighted Mondrian, no transformation)

    Parameters
    ----------
    css_k_list : list[int] or None
        The k values to sweep. If None, falls back to [css_k] (or [active] if
        css_k is also None), reproducing single-k behaviour. Each k is capped
        internally at min(k, dim_in, matrix_rank(H)); pass dim_in to hit k=d.
    css_k : int or None
        Backwards-compatible single-k entry point. Ignored if css_k_list is
        given. If both are None, defaults to active.
    n_iterations : int, default 1
        EGOP iterations on the estimated H before TrIM / Weighted Mondrian /
        Weighted CSS (estimated H). 1 == single fit only.
    weight_power : float, default 1.0
        Exponent on subspace eigenvalues in Weighted CSS. 1.0 matches TrIM's
        eigenbasis scaling; 0.5 -> H^{1/2}-style.

    Returns
    -------
    all_curves : dict
        all_curves[n_sim] = {
            "k_independent": {label: result_dict, ...},   # methods 1-4
            "per_k": {k: {label: result_dict, ...}, ...}, # methods 5-6 by k
        }
    """
    # Resolve the k sweep list.
    if css_k_list is None:
        if css_k is None:
            css_k_list = [active]
        else:
            css_k_list = [css_k]
    css_k_list = list(css_k_list)

    if n_iterations < 1:
        raise ValueError("n_iterations must be >= 1")

    print(f"CSS_K_LIST = {css_k_list}")
    print(f"CSS random_state = {css_random_state}")
    print(f"N_ITERATIONS = {n_iterations}")
    print(f"WEIGHT_POWER = {weight_power}")

    # Labels for the k-independent reference methods.
    _K_INDEP = [
        ("no transformation",   "tab:blue"),
        ("transform by true H", "tab:orange"),
        ("TrIM",                "tab:green"),
        ("Weighted Mondrian",   "tab:red"),
    ]
    # k-dependent methods (the two CSS variants).
    _K_DEP = [
        ("CSS (estimated H)",   "tab:purple"),
        ("CSS (true H)",        "tab:pink"),
    ]

    all_curves = {}

    for n_sim in tqdm(sample_range, desc="sample sizes"):
        x_tr = x_train[:n_sim]
        y_tr = y_train[:n_sim]

        k_indep_curves = {}
        per_k_curves = {}
        css_eff_k = {}   # css_k_val -> effective k selected (len(S_css)); eigen reuses this

        # ================= k-INDEPENDENT methods (once per n_sim) =================

        # ---------- 1. no transformation ----------
        res_no, _ = evaluate_all_lifetimes(
            x_tr, y_tr, x_test, y_test, n_estimators, lifetime
        )
        k_indep_curves["no transformation"] = res_no

        # ---------- 2. transform by true H ----------
        nH = true_H / two_one_norm(true_H)
        res_true, _ = evaluate_all_lifetimes(
            x_tr @ nH, y_tr, x_test @ nH, y_test, n_estimators, lifetime
        )
        k_indep_curves["transform by true H"] = res_true

        # ---------- Train ONE TrIM forest, get importance ----------
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

        # ---------- Compute estimated H at iteration 1 ----------
        forest.H = (importance @ importance.T) / x_tr.shape[0]
        H_hat = forest.H.copy()

        print(f"\nn = {n_sim}")
        print("estimated H (iteration 1) diagonal:", np.diag(H_hat))
        print("estimated H (iteration 1) top diagonal indices:",
              np.argsort(np.diag(H_hat))[::-1])

        H_sym = 0.5 * (H_hat + H_hat.T)
        eigvals = np.sort(np.linalg.eigvalsh(H_sym))[::-1]
        print("estimated H (iteration 1) eigenvalues descending:", eigvals)

        # ---------- EGOP iterations ----------
        if n_iterations > 1:
            print(f"\nRunning {n_iterations - 1} reiteration(s) to refine H "
                  f"(target {n_iterations} total iterations)...")
            for it in range(n_iterations - 1):
                _iter_start = time.perf_counter()
                forest.reiterate()
                _iter_time = time.perf_counter() - _iter_start
                top_idx = np.argsort(np.diag(forest.H))[::-1][:5]
                H_sym_it = 0.5 * (forest.H + forest.H.T)
                eigvals_it = np.sort(np.linalg.eigvalsh(H_sym_it))[::-1][:5]
                print(f"  iteration {it + 2}: time={_iter_time:.3f}s, "
                      f"top diag idx={top_idx.tolist()}, "
                      f"top eigvals={np.round(eigvals_it, 6).tolist()}")
            print(f"estimated H after {n_iterations} iterations, "
                  f"top diagonal indices: "
                  f"{np.argsort(np.diag(forest.H))[::-1].tolist()}")

        H_final = forest.H.copy()

        # ---------- k=d equivalence diagnostic (no Mondrian training) ----------
        check_kd_equivalence(H_final, x_tr, dim_in, weight_power=weight_power)

        # ---------- 3. TrIM ----------
        # forest.H is already H_final
        _trim_start = time.perf_counter()
        _trim_xtr = forest.transform(x_tr)
        _trim_xte = forest.transform(x_test)
        _trim_transform_time = time.perf_counter() - _trim_start
        _print_data_stats(_trim_xtr, "TrIM x_tr")
        _trim_mf_start = time.perf_counter()
        res_trim, _ = evaluate_all_lifetimes(
            _trim_xtr, y_tr,
            _trim_xte, y_test,
            n_estimators, lifetime,
        )
        _trim_mf_time = time.perf_counter() - _trim_mf_start
        _trim_runtime = _trim_transform_time + _trim_mf_time
        _trim_events = len(res_trim['times']) * 50
        print(f"TrIM runtime (n={n_sim}): {_trim_runtime:.4f} s")
        print(f"  TrIM transform time : {_trim_transform_time:.4f} s")
        print(f"  TrIM Mondrian time  : {_trim_mf_time:.4f} s")
        print(f"  TrIM approx events  : {_trim_events}")
        k_indep_curves["TrIM"] = res_trim

        # ---------- 4. Weighted Mondrian ----------
        forest.H = np.diag(np.diag(H_final))
        res_wmf, _ = evaluate_all_lifetimes(
            forest.transform(x_tr), y_tr,
            forest.transform(x_test), y_test,
            n_estimators, lifetime,
        )
        k_indep_curves["Weighted Mondrian"] = res_wmf
        # Restore forest.H so any later read sees H_final.
        forest.H = H_final.copy()

        # ================= k-DEPENDENT methods (sweep over k) =================
        for css_k_val in css_k_list:
            print(f"\n  ===== k sweep: k = {css_k_val} (n = {n_sim}) =====")
            curves_k = {}

            # ---------- 5. Weighted CSS using H_final (estimated) ----------
            _css_start = time.perf_counter()
            x_tr_css, x_te_css, W_css, S_css = css_project_data_weighted(
                H_final,
                x_tr,
                x_test,
                css_k=css_k_val,
                random_state=css_random_state,
                weight_power=weight_power,
            )
            _css_proj_time = time.perf_counter() - _css_start
            print(f"  [k={css_k_val}] CSS (estimated H) selected columns "
                  f"(effective k={len(S_css)}): {S_css}")
            css_eff_k[css_k_val] = len(S_css)
            _print_data_stats(x_tr_css, f"CSS (est H) x_tr k={css_k_val}")

            _css_mf_start = time.perf_counter()
            res_css, _ = evaluate_all_lifetimes(
                x_tr_css, y_tr, x_te_css, y_test, n_estimators, lifetime
            )
            _css_mf_time = time.perf_counter() - _css_mf_start
            _css_runtime = _css_proj_time + _css_mf_time
            print(f"  [k={css_k_val}] CSS runtime (n={n_sim}): "
                  f"{_css_runtime:.4f} s "
                  f"(proj {_css_proj_time:.4f} + mondrian {_css_mf_time:.4f})")
            curves_k["CSS (estimated H)"] = res_css

            # ---------- 6. Weighted CSS using true H ----------
            x_tr_css_true, x_te_css_true, W_css_true, S_css_true = \
                css_project_data_weighted(
                    true_H,
                    x_tr,
                    x_test,
                    css_k=css_k_val,
                    random_state=css_random_state,
                    weight_power=weight_power,
                )
            print(f"  [k={css_k_val}] CSS (true H) selected columns "
                  f"(effective k={len(S_css_true)}): {S_css_true}")

            res_css_true, _ = evaluate_all_lifetimes(
                x_tr_css_true, y_tr, x_te_css_true, y_test,
                n_estimators, lifetime
            )
            curves_k["CSS (true H)"] = res_css_true

            per_k_curves[css_k_val] = curves_k

            # ---------- per-(n,k) MSE summary ----------
            print(f"  [k={css_k_val}] Best MSE over lifetimes:")
            for label, _ in _K_INDEP:
                print(f"    {label:22s}: "
                      f"{float(np.min(k_indep_curves[label]['mse'])):.6e}")
            for label, _ in _K_DEP:
                print(f"    {label:22s}: "
                      f"{float(np.min(curves_k[label]['mse'])):.6e}")

            # ---------- per-(n,k) comparison plot (all 6 methods) ----------
            if save_path is not None:
                fig, ax = plt.subplots(figsize=(7, 5))
                for label, color in _K_INDEP:
                    ax.plot(
                        k_indep_curves[label]["times"],
                        k_indep_curves[label]["mse"],
                        label=label, color=color, linewidth=2,
                    )
                for label, color in _K_DEP:
                    ax.plot(
                        curves_k[label]["times"],
                        curves_k[label]["mse"],
                        label=label, color=color, linewidth=2,
                    )
                ax.set_title(
                    f"Sample Size {n_sim}, CSS k={css_k_val} "
                    f"(w_pow={weight_power})",
                    fontsize=14,
                )
                ax.set_xlabel("neighbors", fontsize=15)
                ax.set_ylabel("MSE", fontsize=15)
                ax.tick_params(labelsize=13)
                ax.legend(fontsize=10)
                _format_axis(ax)
                fig.tight_layout()
                fig.savefig(f"{save_path}_n{n_sim}_cssk{css_k_val}.png", dpi=130)
                plt.close(fig)

        all_curves[n_sim] = {
            "k_independent": k_indep_curves,
            "per_k": per_k_curves,
        }

        # ---------- best MSE vs k summary plot (per n_sim) ----------
        if save_path is not None and len(css_k_list) > 1:
            fig, ax = plt.subplots(figsize=(8, 5.5))

            ks = list(css_k_list)

            # k-dependent methods: lines over k.
            for label, color in _K_DEP:
                best_mse = [
                    float(np.min(per_k_curves[k][label]["mse"])) for k in ks
                ]
                ax.plot(ks, best_mse, marker="o", color=color,
                        linewidth=2, label=label)

            # k-independent methods: horizontal reference lines.
            for label, color in _K_INDEP:
                ref = float(np.min(k_indep_curves[label]["mse"]))
                ax.axhline(ref, color=color, linewidth=1.5, linestyle="--",
                           label=f"{label} (ref)")

            ax.set_title(
                f"Best MSE vs CSS k  (Sample Size {n_sim}, w_pow={weight_power})",
                fontsize=14,
            )
            ax.set_xlabel("CSS k", fontsize=14)
            ax.set_ylabel("Best MSE over lifetimes", fontsize=14)
            ax.set_xticks(ks)
            ax.tick_params(labelsize=12)
            ax.legend(fontsize=9, loc="best")
            _format_axis(ax)
            fig.tight_layout()
            fig.savefig(f"{save_path}_n{n_sim}_bestMSE_vs_k.png", dpi=130)
            plt.close(fig)

        # ---------- eigen-decomposition comparison (estimated H) ----------
        # Reuses the reference + CSS curves above; only adds Eigen (estimated H)
        # and writes "..._bestMSE_vs_k_eigen.png". No retraining, no true-H eigen.
        evaluation_comparison_with_eigen(
            H_final=H_final, x_tr=x_tr, y_tr=y_tr, x_test=x_test, y_test=y_test,
            n_estimators=n_estimators, lifetime=lifetime, dim_in=dim_in,
            css_k_list=css_k_list, css_eff_k=css_eff_k,
            k_indep_curves=k_indep_curves, per_k_curves=per_k_curves,
            n_sim=n_sim, save_path=save_path,
            weight_power=weight_power, eigen_mode="kdim",
        )

    return all_curves