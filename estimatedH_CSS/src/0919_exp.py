"""
main_expB_full.py
=================

Experiment B, complete, in one run.

For each (d, seed) the data is generated ONCE and then used for both halves,
so the two sets of numbers are aligned cell for cell by construction. There is
no second script and nothing to keep in sync.

Half 1 -- iterative sweep
    The reiteration as it actually runs: cold start, then K-1 reiterations,
    recording the EGOP angle distance at every (n, k).

Half 2 -- oracle floor
    One estimation step with the transform pinned to true_H. This is what the
    iteration would produce if it had already converged to the correct
    transform, so it is the floor of the estimator itself at this n. No
    iteration can beat it.

Reading the comparison (printed at the end, and in expB_compare.csv):

    best_iter ~= oracle   ->  the iteration has converged. The estimator is
                              the binding constraint; more iterations will
                              not help.
    best_iter >> oracle   ->  the iteration is far from its own ceiling. That
                              is an algorithmic gap, not a wall -- most likely
                              the initial estimate falling outside the domain
                              of attraction.

Run with:
    python -u main_expB_full.py > 0919_expB_full.log 2>&1

Outputs (under results/expB/):
    expB_raw.csv        one row per (d, seed, n, method, iteration)
    expB_summary.csv    median / IQR across seeds, per method and iteration
    expB_compare.csv    best iterative vs oracle, with the ratio
    expB_d{d}.png       iterative curves plus the oracle floor, per d
    expB_all.png        final-iteration curves and oracle floors, all d

METRIC RESOLUTION. get_angle_distance ends in arccos near 1, which takes the
square root of the floating-point rounding error. Under JAX's default float32
that puts a noise floor of roughly 3e-4 on this metric. Any value within a few
multiples of that is indistinguishable from zero and must NOT be read as "the
estimator is nearly perfect" -- only as "below the resolution of the measure".
The analytic_check column measures that floor directly: it is an analytic-
gradient EGOP whose true angle distance is exactly zero, so whatever it prints
is pure numerical noise.
"""

import os
import csv
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jax import vmap, grad

# Force the EGOP backend before anything constructs a TrIM.
EGOP_BACKEND = "mondrian"          # "mondrian" or "rf"

from Mondrian_RF.Mondrian_forest import TrIM
from utils import SimulatedData, SimulationAligned, get_angle_distance


# =============================================================
# Settings
# =============================================================
D_LIST = [50, 55]          # ascending: cheap cells finish first

N_LIST_DEFAULT  = [3200, 6400, 12800]
N_LIST_OVERRIDE = {}                    # e.g. {100: [800, 1600, 3200, 6400]}
N_LIST_BY_D = {d: list(N_LIST_OVERRIDE.get(d, N_LIST_DEFAULT)) for d in D_LIST}

SEEDS        = [0]
N_ITERATIONS = 4                        # 1 = cold start only; K = fit + (K-1) reiterations
N_TEST       = 3200

ROTATION_SEED_BASE = 1000

ACTIVE        = 2
LIFETIME      = 5
N_ESTIMATORS  = 30
FUN_STR       = "poly"
ROTATION_MODE = None                  # None = aligned, "orth" = rotated

STEP_SIZE_DEFAULT = 1.0
STEP_SIZE_BY_D    = {}

RUN_ORACLE         = True
RUN_ANALYTIC_CHECK = True               # cheap; measures the metric noise floor

OUT_DIR = os.path.join("results", "mondrian_aligned")


def step_size_for(d):
    return float(STEP_SIZE_BY_D.get(d, STEP_SIZE_DEFAULT))


# =============================================================
# Setup
# =============================================================
os.makedirs(OUT_DIR, exist_ok=True)
RAW_PATH     = os.path.join(OUT_DIR, "expB_raw.csv")
SUMMARY_PATH = os.path.join(OUT_DIR, "expB_summary.csv")
COMPARE_PATH = os.path.join(OUT_DIR, "expB_compare.csv")

print("\n" + "=" * 72)
print(f"run started {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 72)

RAW_FIELDS = ["d", "seed", "n", "method", "iteration", "angle_distance",
              "analytic_check", "step_size", "seconds"]

print("Experiment B (full): iterative sweep + oracle floor")
print(f"  EGOP_BACKEND  = {EGOP_BACKEND} (forced)")
print(f"  D_LIST        = {D_LIST}")
print(f"  SEEDS         = {SEEDS}")
print(f"  N_ITERATIONS  = {N_ITERATIONS}")
print(f"  RUN_ORACLE    = {RUN_ORACLE}")
print(f"  ACTIVE        = {ACTIVE}")
print(f"  LIFETIME      = {LIFETIME}")
print(f"  N_ESTIMATORS  = {N_ESTIMATORS}")
print(f"  FUN_STR       = {FUN_STR}")
print(f"  ROTATION_MODE = {ROTATION_MODE}")
print(f"  N_TEST        = {N_TEST}")
for _d in D_LIST:
    print(f"  d={_d}: n={N_LIST_BY_D[_d]}, step_size={step_size_for(_d):.4f}")
print()

with open(RAW_PATH, "w", newline="") as f:
    csv.DictWriter(f, fieldnames=RAW_FIELDS).writeheader()

records = []


def append_rows(rows):
    """Extend the in-memory record set and flush to disk immediately, so a
    long run can be inspected or killed mid-way without losing work."""
    records.extend(rows)
    with open(RAW_PATH, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RAW_FIELDS)
        for r in rows:
            writer.writerow(r)


def oracle_H(x, y, true_H, step_size, random_state):
    """One estimation step with the transform pinned to true_H.

    Identical to TrIM.reiterate() except that self.H is true_H rather than a
    previous estimate. Verified bitwise against reiterate() in sanity_oracle.py.

    transform() normalizes by two_one_norm internally, so true_H is passed raw.
    _get_importance perturbs in ORIGINAL coordinates and calls transform() on
    each probe, which keeps the probes consistent with the fitted space.
    """
    forest = TrIM(n_estimators=N_ESTIMATORS, lifetime=LIFETIME,
                  step_size=step_size, random_state=random_state)
    forest.X = np.array(x)
    forest.y = np.array(y)
    forest.H = np.array(true_H)
    forest.mf.set_random_state()
    forest.mf.fit(forest.transform(forest.X), forest.y)
    H = forest.estimate_H_finite_diff()
    del forest
    return H


# =============================================================
# Main sweep
# =============================================================
for d in D_LIST:
    n_list = sorted(N_LIST_BY_D[d])
    n_max = max(n_list)
    step_size = step_size_for(d)

    for seed in SEEDS:
        print(f"\n[d={d}] seed={seed}  (n_max={n_max}, h={step_size:.4f})")

        # ---- data, generated once and shared by both halves --------------
        # X first, from the data seed.
        np.random.seed(seed)
        x_train = np.random.rand(n_max, d) * 2 - 1
        x_test  = np.random.rand(N_TEST, d) * 2 - 1

        # Re-seed on a dedicated stream before drawing the rotation. Without
        # this, SimulatedData consumes whatever RNG state is left after the X
        # draws -- which depends on n_max and N_TEST -- so changing the n
        # sweep would silently change the rotation, and with it true_H.
        # Now the rotation is a function of `seed` alone.
        np.random.seed(ROTATION_SEED_BASE + seed)
        fn = SimulatedData(d, ACTIVE, rotation=ROTATION_MODE, fun=FUN_STR)

        y_train = np.array(vmap(fn.fun, in_axes=0, out_axes=0)(x_train))
        y_test  = np.array(vmap(fn.fun, in_axes=0, out_axes=0)(x_test))
        true_H  = np.array(fn.get_true_H(x_test))

        # ---- half 1: the iterative sweep ---------------------------------
        sim = SimulationAligned(
            d, ACTIVE, N_ESTIMATORS,
            1,                 # tries must be 1
            LIFETIME, step_size,
            n_list,            # sample_range == the n sweep
            x_train, x_test, y_train, y_test, true_H,
        )

        t0 = time.time()
        sim.train()
        t_fit = time.time() - t0
        plt.close("all")       # Simulation.train() draws via plot_dist

        rows = []
        for n in n_list:
            H_est = sim.forests[n][0].H
            rows.append({
                "d": d, "seed": seed, "n": n,
                "method": "iterative", "iteration": 1,
                "angle_distance": float(get_angle_distance(H_est, true_H, ACTIVE)),
                "analytic_check": "",
                "step_size": step_size, "seconds": t_fit,
            })
        append_rows(rows)
        print(f"    iter 1  ({t_fit:8.1f}s): " +
              "  ".join(f"n={r['n']}:{r['angle_distance']:.4f}" for r in rows))

        for it in range(2, N_ITERATIONS + 1):
            t0 = time.time()
            sim.reiterate()
            t_it = time.time() - t0
            plt.close("all")

            rows = []
            for n in n_list:
                H_est = sim.forests[n][0].H
                rows.append({
                    "d": d, "seed": seed, "n": n,
                    "method": "iterative", "iteration": it,
                    "angle_distance": float(get_angle_distance(H_est, true_H, ACTIVE)),
                    "analytic_check": "",
                    "step_size": step_size, "seconds": t_it,
                })
            append_rows(rows)
            print(f"    iter {it}  ({t_it:8.1f}s): " +
                  "  ".join(f"n={r['n']}:{r['angle_distance']:.4f}" for r in rows))

        del sim

        # ---- half 2: the oracle floor, same data -------------------------
        if RUN_ORACLE:
            rows = []
            for n in n_list:
                t0 = time.time()
                H_or = oracle_H(x_train[:n], y_train[:n], true_H,
                                step_size, random_state=(n + 1))
                t_or = time.time() - t0
                dist = float(get_angle_distance(H_or, true_H, ACTIVE))

                # Analytic EGOP on the same n points. Its true angle distance
                # is exactly zero, so this reads out the metric's noise floor.
                check = ""
                if RUN_ANALYTIC_CHECK:
                    W = np.array(vmap(grad(fn.fun), in_axes=0, out_axes=0)(x_train[:n]))
                    check = float(get_angle_distance(W.T @ W / n, true_H, ACTIVE))

                rows.append({
                    "d": d, "seed": seed, "n": n,
                    "method": "oracle", "iteration": "",
                    "angle_distance": dist,
                    "analytic_check": check,
                    "step_size": step_size, "seconds": t_or,
                })
            append_rows(rows)
            print("    oracle          : " +
                  "  ".join(f"n={r['n']}:{r['angle_distance']:.4f}" for r in rows))
            if RUN_ANALYTIC_CHECK:
                noise = max(r["analytic_check"] for r in rows)
                print(f"    metric noise floor (analytic, true value 0): {noise:.2e}")

        del x_train, x_test, y_train, y_test, true_H, fn


# =============================================================
# Summary: median / IQR across seeds, plus local log-log slope
# =============================================================
def cell(d, n, method, it):
    return [r["angle_distance"] for r in records
            if r["d"] == d and r["n"] == n
            and r["method"] == method and r["iteration"] == it]


summary = []
for d in D_LIST:
    n_list = sorted(N_LIST_BY_D[d])
    keys = [("iterative", it) for it in range(1, N_ITERATIONS + 1)]
    if RUN_ORACLE:
        keys.append(("oracle", ""))

    for method, it in keys:
        prev_n, prev_med = None, None
        for n in n_list:
            vals = cell(d, n, method, it)
            if not vals:
                continue
            med = float(np.median(vals))
            q1, q3 = np.percentile(vals, [25, 75])

            # d log(err) / d log(n) against the previous n. ~0 means flat.
            slope = ""
            if prev_med is not None and prev_med > 0 and med > 0:
                slope = float((np.log(med) - np.log(prev_med)) /
                              (np.log(n) - np.log(prev_n)))

            summary.append({
                "d": d, "method": method, "iteration": it, "n": n,
                "n_seeds": len(vals), "median": med,
                "q1": float(q1), "q3": float(q3),
                "loglog_slope_vs_prev_n": slope,
            })
            prev_n, prev_med = n, med

with open(SUMMARY_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    writer.writeheader()
    writer.writerows(summary)


def med_of(d, n, method, it):
    for s in summary:
        if (s["d"] == d and s["n"] == n
                and s["method"] == method and s["iteration"] == it):
            return s["median"]
    return None


print("\n" + "=" * 72)
print(f"Summary (median over {len(SEEDS)} seed(s))")
print("=" * 72)
print(f"{'d':>5} {'method':>10} {'iter':>5} {'n':>7} {'median':>10} {'IQR':>20} {'slope':>8}")
for s in summary:
    slope = ("       -" if s["loglog_slope_vs_prev_n"] == ""
             else f"{s['loglog_slope_vs_prev_n']:8.3f}")
    iqr = f"[{s['q1']:.4f}, {s['q3']:.4f}]"
    it = "-" if s["iteration"] == "" else s["iteration"]
    print(f"{s['d']:>5} {s['method']:>10} {it:>5} {s['n']:>7} "
          f"{s['median']:>10.4f} {iqr:>20} {slope}")


# =============================================================
# Comparison: best iterative vs oracle floor
# =============================================================
compare = []
if RUN_ORACLE:
    for d in D_LIST:
        for n in sorted(N_LIST_BY_D[d]):
            iters = [(it, med_of(d, n, "iterative", it))
                     for it in range(1, N_ITERATIONS + 1)]
            iters = [(it, m) for it, m in iters if m is not None]
            orc = med_of(d, n, "oracle", "")
            if not iters or orc is None:
                continue

            best_it, best_med = min(iters, key=lambda p: p[1])
            first_med = iters[0][1]

            compare.append({
                "d": d, "n": n,
                "iter1": first_med,
                "best_iter": best_med,
                "best_at_k": best_it,
                "oracle": orc,
                "ratio_best_over_oracle": (best_med / orc) if orc > 0 else "",
                "iter_gain_k1_to_best": (1.0 - best_med / first_med) if first_med > 0 else "",
            })

    with open(COMPARE_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(compare[0].keys()))
        writer.writeheader()
        writer.writerows(compare)

    print("\n" + "=" * 72)
    print("Best iterative vs oracle floor")
    print("=" * 72)
    print(f"{'d':>5} {'n':>7} {'k=1':>9} {'best':>9} {'@k':>3} "
          f"{'oracle':>9} {'ratio':>8} {'iter gain':>10}")
    for c in compare:
        ratio = ("       -" if c["ratio_best_over_oracle"] == ""
                 else f"{c['ratio_best_over_oracle']:8.1f}")
        gain = ("         -" if c["iter_gain_k1_to_best"] == ""
                else f"{100 * c['iter_gain_k1_to_best']:9.1f}%")
        print(f"{c['d']:>5} {c['n']:>7} {c['iter1']:>9.4f} {c['best_iter']:>9.4f} "
              f"{c['best_at_k']:>3} {c['oracle']:>9.4f} {ratio} {gain}")

    print("\n  ratio near 1  -> iteration has reached the estimator's floor")
    print("  ratio large   -> iteration is far from its own ceiling")
    print("  iter gain     -> how much the reiterations bought over the cold start")


# =============================================================
# Figures
# =============================================================
def series(d, method, it):
    rows = [s for s in summary
            if s["d"] == d and s["method"] == method and s["iteration"] == it]
    rows.sort(key=lambda r: r["n"])
    return (np.array([r["n"] for r in rows], dtype=float),
            np.array([r["median"] for r in rows]),
            np.array([r["q1"] for r in rows]),
            np.array([r["q3"] for r in rows]))


for d in D_LIST:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for it in range(1, N_ITERATIONS + 1):
        n, med, q1, q3 = series(d, "iterative", it)
        if n.size == 0:
            continue
        ax.plot(n, med, marker="o", label=f"iteration {it}")
        if len(SEEDS) > 1:
            ax.fill_between(n, q1, q3, alpha=0.18)
    if RUN_ORACLE:
        n, med, q1, q3 = series(d, "oracle", "")
        if n.size:
            ax.plot(n, med, marker="s", linestyle="--", color="k",
                    label="oracle floor (true H)")
            if len(SEEDS) > 1:
                ax.fill_between(n, q1, q3, alpha=0.15, color="k")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n (training samples)")
    ax.set_ylabel("EGOP angle distance to true H")
    ax.set_title(f"EGOP error vs n   (d={d}, rotation={ROTATION_MODE}, "
                 f"h={step_size_for(d):.3f})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, f"expB_d{d}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")

fig, ax = plt.subplots(figsize=(6.5, 4.5))
for i, d in enumerate(D_LIST):
    color = f"C{i}"
    n, med, _, _ = series(d, "iterative", N_ITERATIONS)
    if n.size:
        ax.plot(n, med, marker="o", color=color, label=f"d={d}")
    if RUN_ORACLE:
        n, med, _, _ = series(d, "oracle", "")
        if n.size:
            ax.plot(n, med, marker="s", linestyle="--", color=color, alpha=0.6)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("n (training samples)")
ax.set_ylabel("EGOP angle distance to true H")
ax.set_title(f"Solid: iteration {N_ITERATIONS}.  Dashed: oracle floor.  "
             f"(rotation={ROTATION_MODE})")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=9)
fig.tight_layout()
path = os.path.join(OUT_DIR, "expB_all.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"wrote {path}")

print(f"\nDone. Raw: {RAW_PATH}")
print(f"      Summary: {SUMMARY_PATH}")
if RUN_ORACLE:
    print(f"      Compare: {COMPARE_PATH}")