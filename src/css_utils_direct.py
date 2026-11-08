"""
css_utils_direct.py
===================

Direct TrIM-coordinate CSS.

Motivation
----------
Weighted CSS (css_utils_trueH.css_project_data_weighted) does NOT reduce to
TrIM at k=d, because it re-expresses the selected subspace in a *new*
coordinate frame:

    U   = qr(H[:, S])            # re-orthonormalize the selected columns
    W   = U V diag(Lambda^p)     # rotate within the subspace + rescale

That re-orthonormalization + within-subspace rotation V changes the
*orientation* of the transformed coordinates. A Mondrian forest cuts
axis-aligned in the transformed frame, so it is NOT rotation invariant:
two transforms with identical singular-value spectra (identical cut budget)
but different orientation produce different partitions and different MSE.
This is exactly why Weighted CSS at k=d sits above TrIM even though
check_kd_equivalence reports a matching spectrum.

Direct TrIM-coordinate CSS avoids this entirely by staying in TrIM's frame:

    # use CSS/RPQR ONLY to choose which TrIM coordinates to keep
    S = _css_select_basis(H, k)[1]          # selected column indices of H
    A = H[:, S] / two_one_norm(H)           # raw selected columns of H, (d, k)
    x_train_css = x_train @ A
    x_test_css  = x_test  @ A

Column j of the full TrIM transform is x @ (H[:, j] / two_one_norm(H)); this
method just keeps the k columns indexed by S. No re-orthonormalization, no
within-subspace rotation, no eigenvalue rescaling -- the retained coordinates
are TrIM's own coordinates, verbatim.

Properties
----------
* k < d  -> output dimension is k  -> still a dimensionality reduction.
* k = d  -> S is all columns (a permutation of range(d)), so
            A = H / two_one_norm(H) up to a column permutation. Hence
            x @ A is the full TrIM transform up to column ORDER. The Mondrian
            partition law is invariant to column permutation, so MSE matches
            TrIM (up to finite-forest sampling noise).
            Likewise Direct CSS (true H) at k=d matches "transform by true H".
  (The k=d match requires H to be full rank so that S really contains all
  columns; _css_select_basis caps k at matrix_rank(H). The estimated H here is
  full rank, and the true H is empirically full rank in this experiment.)

Methods plotted
---------------
k-independent reference methods (computed once per sample size):
    1. no transformation
    2. transform by true H
    3. TrIM
    4. Weighted Mondrian
k-dependent methods (swept over k):
    5. Direct CSS (est H)   <- should hit the TrIM ref line at k=d
    6. Direct CSS (true H)  <- should hit the "transform by true H" ref at k=d
    7. CSS (true H)         <- Weighted CSS on true H, kept as a baseline
"""

import time
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# Downstream learner selector (mirrors the kNN / css_utils_max swap). TrIM is
# re-exported by each backend, so EGOP (H) estimation stays controlled by the
# Mondrian_RF.Mondrian_forest EGOP_BACKEND constant; only evaluate_all_lifetimes
# (the regression head) changes here.
#
# Mondrian (original):
from Mondrian_RF.Mondrian_forest import TrIM
from Mondrian_RF.utils import evaluate_all_lifetimes, two_one_norm
#
# Standard Random Forest (this experiment):
# from RandomForest.RandomForest import TrIM
# from RandomForest.utils import evaluate_all_lifetimes, two_one_norm

# Reuse the shared helpers from the weighted-CSS module so selection logic,
# bounding-box logging, axis formatting, the k=d diagnostic, and the weighted
# baseline stay byte-for-byte consistent across the two experiments.
from css_utils_trueH import (
    _css_select_basis,
    _print_data_stats,
    _format_axis,
    check_kd_equivalence,
    css_project_data_weighted,
)


# ===========================================================================
# Direct TrIM-coordinate CSS projection
# ===========================================================================

def css_project_data_direct(H, x_train, x_test, css_k, random_state=0):
    """
    Direct TrIM-coordinate CSS.

    Select k column indices S of H via the SAME RPQR logic used by Weighted
    CSS (_css_select_basis), then build the transform directly from the raw
    selected columns of H, normalized by the global (2,1)-norm of the FULL H:

        A = H[:, S] / two_one_norm(H)        # (d, k)
        x -> x @ A

    Using the full-H (2,1)-norm (rather than two_one_norm(H[:, S])) is what
    makes the k=d transform numerically identical to TrIM's
    H / two_one_norm(H), not merely identical up to a scalar -- so the MSE-vs-
    lifetime curves overlie, they do not just share a shape.

    Returns
    -------
    x_train_proj : (n_train, k)
    x_test_proj  : (n_test,  k)
    A            : (d, k)  the transform (raw selected H columns, scaled)
    S            : selected column indices
    """
    H = np.array(H, dtype=float)
    H = 0.5 * (H + H.T)              # no-op for the symmetric H here; safe guard

    # We only need the indices S (and the effective k); discard U.
    _U, S, k = _css_select_basis(H, css_k, random_state=random_state)

    if k == 0:
        return x_train.copy(), x_test.copy(), np.zeros((H.shape[0], 0)), S

    denom = two_one_norm(H)
    if denom < 1e-14:
        denom = 1.0

    A = H[:, S] / denom             # (d, k) -- k retained TrIM coordinates

    x_tr_proj = x_train @ A
    x_te_proj = x_test @ A
    return x_tr_proj, x_te_proj, A, S


# ===========================================================================
# Main entry point
# ===========================================================================

def evaluation_comparison_with_direct_css(
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
    Same evaluation harness as evaluation_comparison_with_css, but the swept
    CSS method is Direct TrIM-coordinate CSS. Signature is identical so this is
    a drop-in replacement (only the import in main.py changes).

    k-independent methods (once per sample size):
        1. no transformation
        2. transform by true H
        3. TrIM                (H after n_iterations EGOP iterations)
        4. Weighted Mondrian   (diag of H after n_iterations EGOP iterations)

    k-dependent methods (recomputed per k in css_k_list):
        5. Direct CSS (est H)
        6. Direct CSS (true H)
        7. CSS (true H)        (Weighted CSS on true H -- baseline for contrast)

    weight_power is only consumed by the Weighted CSS baseline (#7) and by the
    check_kd_equivalence diagnostic; the Direct method has no weight parameter.

    Plots produced (when save_path is given):
        - one comparison figure per (n_sim, k): MSE vs lifetime, all methods
        - one "best MSE vs k" summary figure per n_sim: k-dependent methods as
          lines over k, k-independent methods as horizontal reference lines.
          At k=d the Direct (est H) line should land on the TrIM ref and the
          Direct (true H) line on the transform-by-true-H ref.

    Returns
    -------
    all_curves : dict
        all_curves[n_sim] = {
            "k_independent": {label: result_dict, ...},
            "per_k": {k: {label: result_dict, ...}, ...},
        }
    """
    # Resolve the k sweep list.
    if css_k_list is None:
        css_k_list = [active] if css_k is None else [css_k]
    css_k_list = list(css_k_list)

    if n_iterations < 1:
        raise ValueError("n_iterations must be >= 1")

    print(f"CSS_K_LIST = {css_k_list}")
    print(f"CSS random_state = {css_random_state}")
    print(f"N_ITERATIONS = {n_iterations}")
    print(f"WEIGHT_POWER = {weight_power} (baseline Weighted CSS only)")
    print("CSS variant = Direct TrIM-coordinate CSS")

    # k-independent reference methods.
    _K_INDEP = [
        ("no transformation",   "tab:blue"),
        ("transform by true H", "tab:orange"),
        ("TrIM",                "tab:green"),
        ("Weighted Mondrian",   "tab:red"),
    ]
    # k-dependent methods.
    _K_DEP = [
        ("Direct CSS (est H)",  "tab:purple"),
        ("Direct CSS (true H)", "tab:cyan"),
        ("CSS (true H)",        "tab:pink"),   # Weighted CSS baseline
    ]
    # Markers for the best-MSE-vs-k plot.
    _K_DEP_MARKER = {
        "Direct CSS (est H)":  "o",
        "Direct CSS (true H)": "s",
        "CSS (true H)":        "^",
    }

    all_curves = {}

    for n_sim in tqdm(sample_range, desc="sample sizes"):
        x_tr = x_train[:n_sim]
        y_tr = y_train[:n_sim]

        k_indep_curves = {}
        per_k_curves = {}

        # ============== k-INDEPENDENT methods (once per n_sim) ==============

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
        print(f"TrIM runtime (n={n_sim}): {_trim_runtime:.4f} s")
        print(f"  TrIM transform time : {_trim_transform_time:.4f} s")
        print(f"  TrIM RF time        : {_trim_mf_time:.4f} s")
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

            # ---------- 5. Direct CSS using H_final (estimated) ----------
            _d_start = time.perf_counter()
            x_tr_d, x_te_d, A_d, S_d = css_project_data_direct(
                H_final, x_tr, x_test,
                css_k=css_k_val,
                random_state=css_random_state,
            )
            _d_proj_time = time.perf_counter() - _d_start
            print(f"  [k={css_k_val}] Direct CSS (est H) selected columns "
                  f"(effective k={len(S_d)}): {S_d}")
            _print_data_stats(x_tr_d, f"Direct CSS (est H) x_tr k={css_k_val}")

            _d_mf_start = time.perf_counter()
            res_d, _ = evaluate_all_lifetimes(
                x_tr_d, y_tr, x_te_d, y_test, n_estimators, lifetime
            )
            _d_mf_time = time.perf_counter() - _d_mf_start
            print(f"  [k={css_k_val}] Direct CSS (est H) runtime (n={n_sim}): "
                  f"{_d_proj_time + _d_mf_time:.4f} s "
                  f"(proj {_d_proj_time:.4f} + rf {_d_mf_time:.4f})")
            curves_k["Direct CSS (est H)"] = res_d

            # ---------- 6. Direct CSS using true H ----------
            x_tr_dt, x_te_dt, A_dt, S_dt = css_project_data_direct(
                true_H, x_tr, x_test,
                css_k=css_k_val,
                random_state=css_random_state,
            )
            print(f"  [k={css_k_val}] Direct CSS (true H) selected columns "
                  f"(effective k={len(S_dt)}): {S_dt}")
            _print_data_stats(x_tr_dt, f"Direct CSS (true H) x_tr k={css_k_val}")

            res_dt, _ = evaluate_all_lifetimes(
                x_tr_dt, y_tr, x_te_dt, y_test, n_estimators, lifetime
            )
            curves_k["Direct CSS (true H)"] = res_dt

            # ---------- 7. Weighted CSS using true H (baseline) ----------
            x_tr_w, x_te_w, W_w, S_w = css_project_data_weighted(
                true_H, x_tr, x_test,
                css_k=css_k_val,
                random_state=css_random_state,
                weight_power=weight_power,
            )
            print(f"  [k={css_k_val}] CSS (true H) [weighted] selected columns "
                  f"(effective k={len(S_w)}): {S_w}")
            res_w, _ = evaluate_all_lifetimes(
                x_tr_w, y_tr, x_te_w, y_test, n_estimators, lifetime
            )
            curves_k["CSS (true H)"] = res_w

            per_k_curves[css_k_val] = curves_k

            # ---------- per-(n,k) MSE summary ----------
            print(f"  [k={css_k_val}] Best MSE over max_depth:")
            for label, _ in _K_INDEP:
                print(f"    {label:22s}: "
                      f"{float(np.min(k_indep_curves[label]['mse'])):.6e}")
            for label, _ in _K_DEP:
                print(f"    {label:22s}: "
                      f"{float(np.min(curves_k[label]['mse'])):.6e}")

            # ---------- per-(n,k) comparison plot (MSE vs lifetime) ----------
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
                    f"Sample Size {n_sim}, Direct CSS k={css_k_val}",
                    fontsize=14,
                )
                ax.set_xlabel("max_depth", fontsize=15)
                ax.set_ylabel("MSE", fontsize=15)
                ax.tick_params(labelsize=13)
                ax.legend(fontsize=9)
                _format_axis(ax)
                fig.tight_layout()
                fig.savefig(f"{save_path}_n{n_sim}_directcssk{css_k_val}.png",
                            dpi=130)
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
                ax.plot(ks, best_mse, marker=_K_DEP_MARKER[label], color=color,
                        linewidth=2, markersize=7, label=label)

            # k-independent methods: horizontal reference lines.
            for label, color in _K_INDEP:
                ref = float(np.min(k_indep_curves[label]["mse"]))
                ax.axhline(ref, color=color, linewidth=1.5, linestyle="--",
                           label=f"{label} (ref)")

            ax.set_title(
                f"Best MSE vs k  (Direct TrIM-CSS, Sample Size {n_sim})",
                fontsize=14,
            )
            ax.set_xlabel("CSS k", fontsize=14)
            ax.set_ylabel("Best MSE over max_depth", fontsize=14)
            ax.set_xticks(ks)
            ax.tick_params(labelsize=12)
            ax.legend(fontsize=9, loc="best")
            _format_axis(ax)
            fig.tight_layout()
            fig.savefig(f"{save_path}_n{n_sim}_directCSS_bestMSE_vs_k.png",
                        dpi=130)
            plt.close(fig)

    return all_curves