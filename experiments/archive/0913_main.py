"""
main_expB.py
============

Experiment B: at fixed d, how does EGOP estimation error scale with n?

For each d in D_LIST, holds everything else fixed and sweeps n upward,
recording the subspace angle distance between the forest-estimated H and
the true H, for each reiteration round.

This script runs NO downstream CSS/TrIM evaluation. It only measures EGOP
quality, so the cost per (d, n) is one forest fit plus (N_ITERATIONS - 1)
refits -- not a full 6-method comparison.

Run with:
    python -u 0913_main.py > 0913_expB.log 2>&1

Outputs (under results/expB/):
    expB_raw.csv        one row per (d, seed, n, iteration)
    expB_summary.csv    median / IQR across seeds + local log-log slope
    expB_d{d}.png       angle distance vs n, one line per iteration
    expB_all.png        all d on one axis, final iteration only

Assumptions carried over from main.py (verify once before a long run):
  - SimulationAligned slices x_train[:n] internally for each n in
    sample_range, so one dataset of size max(N_LIST) covers the whole sweep.
  - SimulationAligned requires tries == 1.
  - SimulatedData draws its rotation from the global numpy RNG. This script
    re-seeds on a separate stream (ROTATION_SEED_BASE) right before building
    it, so the rotation depends on `seed` alone and not on n_max. That makes
    the n sweep self-consistent, but it also means the rotation here differs
    from scripts that let SimulatedData inherit the post-X RNG state -- so
    per-(d, seed) numbers will NOT match those scripts even at equal n.
"""

import os
import csv
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jax import vmap

# Force the EGOP backend for the whole run, BEFORE anything constructs a TrIM.
# Must happen before `from utils import ...`.
EGOP_BACKEND = "mondrian"          # "mondrian" or "rf"

import Mondrian_RF.Mondrian_forest as _mf
_mf.EGOP_BACKEND = EGOP_BACKEND

from utils import SimulatedData, SimulationAligned, get_angle_distance


# =============================================================
# Settings
# =============================================================
D_LIST = [65, 70]

# Same n sweep for every d, so curves are directly comparable across d.
# To shorten the sweep for one specific d, add an entry to N_LIST_OVERRIDE.
N_LIST_DEFAULT  = [400, 800, 1600, 3200, 6400, 12800]
N_LIST_OVERRIDE = {}    # e.g. {500: [800, 1600, 3200, 6400]}

N_LIST_BY_D = {d: list(N_LIST_OVERRIDE.get(d, N_LIST_DEFAULT)) for d in D_LIST}

SEEDS        = [0]   # est-H variance is high; median over seeds
N_ITERATIONS = 4                 # 1 = initial fit only; K = fit + (K-1) reiterations
N_TEST       = 3200              # true_H is estimated on x_test; matches other scripts

# Rotation is drawn from its own seed stream so it depends on `seed` only,
# not on how large the n sweep happens to be. See the data block below.
ROTATION_SEED_BASE = 1000

ACTIVE        = 2
LIFETIME      = 5
N_ESTIMATORS  = 30
FUN_STR       = "poly"
ROTATION_MODE = "orth"           # None = aligned, "orth" = rotated

# Flat 1.0 across d, matching the step used in the K-sweep / RF-vs-Mondrian
# runs, so Experiment B numbers sit next to those without a step confound.
# Override per-d only if a calibration says a given d needs something else.
STEP_SIZE_DEFAULT = 1.0
STEP_SIZE_BY_D    = {}           # e.g. {500: 0.5}

OUT_DIR = os.path.join("results", "expB")


def step_size_for(d):
    return float(STEP_SIZE_BY_D.get(d, STEP_SIZE_DEFAULT))


# =============================================================
# Setup
# =============================================================
os.makedirs(OUT_DIR, exist_ok=True)
RAW_PATH     = os.path.join(OUT_DIR, "expB_raw.csv")
SUMMARY_PATH = os.path.join(OUT_DIR, "expB_summary.csv")

print("\n" + "=" * 70)
print(f"run started {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

RAW_FIELDS = ["d", "seed", "n", "iteration", "angle_distance",
              "step_size", "fit_seconds"]

print("Experiment B: EGOP error vs n at fixed d")
print(f"  EGOP_BACKEND  = {_mf.EGOP_BACKEND} (forced)")
print(f"  D_LIST        = {D_LIST}")
print(f"  SEEDS         = {SEEDS}")
print(f"  N_ITERATIONS  = {N_ITERATIONS}")
print(f"  ACTIVE        = {ACTIVE}")
print(f"  LIFETIME      = {LIFETIME}")
print(f"  N_ESTIMATORS  = {N_ESTIMATORS}")
print(f"  FUN_STR       = {FUN_STR}")
print(f"  ROTATION_MODE = {ROTATION_MODE}")
print(f"  N_TEST        = {N_TEST}")
for _d in D_LIST:
    print(f"  d={_d}: n={N_LIST_BY_D[_d]}, step_size={step_size_for(_d):.4f}")
print()

# Fresh raw file with header; rows are appended and flushed after every
# (d, seed) block so a long run can be inspected or killed mid-way.
with open(RAW_PATH, "w", newline="") as f:
    csv.DictWriter(f, fieldnames=RAW_FIELDS).writeheader()

records = []


def append_rows(rows):
    records.extend(rows)
    with open(RAW_PATH, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RAW_FIELDS)
        for r in rows:
            writer.writerow(r)


# =============================================================
# Main sweep
# =============================================================
for d in D_LIST:
    n_list = sorted(N_LIST_BY_D[d])
    n_max = max(n_list)
    step_size = step_size_for(d)

    for seed in SEEDS:
        print(f"[d={d}] seed={seed}  (n_max={n_max}, h={step_size:.4f})")

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

        sim = SimulationAligned(
            d,
            ACTIVE,
            N_ESTIMATORS,
            1,                 # tries must be 1
            LIFETIME,
            step_size,
            n_list,            # sample_range == the n sweep
            x_train,
            x_test,
            y_train,
            y_test,
            true_H,
        )

        t0 = time.time()
        sim.train()
        t_fit = time.time() - t0

        rows = []
        for n in n_list:
            H_est = sim.forests[n][0].H
            rows.append({
                "d": d, "seed": seed, "n": n, "iteration": 1,
                "angle_distance": float(get_angle_distance(H_est, true_H, ACTIVE)),
                "step_size": step_size, "fit_seconds": t_fit,
            })
        append_rows(rows)
        print(f"    iter 1  ({t_fit:8.1f}s): " +
              "  ".join(f"n={r['n']}:{r['angle_distance']:.4f}" for r in rows))

        for it in range(2, N_ITERATIONS + 1):
            t0 = time.time()
            sim.reiterate()
            t_it = time.time() - t0

            rows = []
            for n in n_list:
                H_est = sim.forests[n][0].H
                rows.append({
                    "d": d, "seed": seed, "n": n, "iteration": it,
                    "angle_distance": float(get_angle_distance(H_est, true_H, ACTIVE)),
                    "step_size": step_size, "fit_seconds": t_it,
                })
            append_rows(rows)
            print(f"    iter {it}  ({t_it:8.1f}s): " +
                  "  ".join(f"n={r['n']}:{r['angle_distance']:.4f}" for r in rows))

        del x_train, x_test, y_train, y_test, true_H, sim, fn


# =============================================================
# Summary: median / IQR across seeds, plus local log-log slope
# =============================================================
def cell(d, n, it):
    return [r["angle_distance"] for r in records
            if r["d"] == d and r["n"] == n and r["iteration"] == it]


summary = []
for d in D_LIST:
    n_list = sorted(N_LIST_BY_D[d])
    for it in range(1, N_ITERATIONS + 1):
        prev_n, prev_med = None, None
        for n in n_list:
            vals = cell(d, n, it)
            if not vals:
                continue
            med = float(np.median(vals))
            q1, q3 = np.percentile(vals, [25, 75])

            # d log(err) / d log(n) between this n and the previous one.
            # ~0 means the curve has flattened.
            slope = ""
            if prev_med is not None and prev_med > 0 and med > 0:
                slope = float((np.log(med) - np.log(prev_med)) /
                              (np.log(n) - np.log(prev_n)))

            summary.append({
                "d": d, "iteration": it, "n": n, "n_seeds": len(vals),
                "median": med, "q1": float(q1), "q3": float(q3),
                "loglog_slope_vs_prev_n": slope,
            })
            prev_n, prev_med = n, med

with open(SUMMARY_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    writer.writeheader()
    writer.writerows(summary)

print(f"\nSummary (median over {len(SEEDS)} seed(s)):")
print(f"{'d':>6} {'iter':>5} {'n':>7} {'median':>10} {'IQR':>20} {'slope':>8}")
for s in summary:
    slope = ("       -" if s["loglog_slope_vs_prev_n"] == ""
             else f"{s['loglog_slope_vs_prev_n']:8.3f}")
    iqr = f"[{s['q1']:.4f}, {s['q3']:.4f}]"
    print(f"{s['d']:>6} {s['iteration']:>5} {s['n']:>7} "
          f"{s['median']:>10.4f} {iqr:>20} {slope}")


# =============================================================
# Figures
# =============================================================
def series(d, it):
    rows = [s for s in summary if s["d"] == d and s["iteration"] == it]
    rows.sort(key=lambda r: r["n"])
    return (np.array([r["n"] for r in rows], dtype=float),
            np.array([r["median"] for r in rows]),
            np.array([r["q1"] for r in rows]),
            np.array([r["q3"] for r in rows]))


for d in D_LIST:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for it in range(1, N_ITERATIONS + 1):
        n, med, q1, q3 = series(d, it)
        if n.size == 0:
            continue
        ax.plot(n, med, marker="o", label=f"iteration {it}")
        if len(SEEDS) > 1:
            ax.fill_between(n, q1, q3, alpha=0.18)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n (training samples)")
    ax.set_ylabel("EGOP angle distance to true H")
    ax.set_title(f"EGOP error vs n   (d={d}, rotation={ROTATION_MODE}, h={step_size_for(d):.3f})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUT_DIR, f"expB_d{d}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")

fig, ax = plt.subplots(figsize=(6.5, 4.5))
for d in D_LIST:
    n, med, q1, q3 = series(d, N_ITERATIONS)
    if n.size == 0:
        continue
    ax.plot(n, med, marker="o", label=f"d={d}")
    if len(SEEDS) > 1:
        ax.fill_between(n, q1, q3, alpha=0.18)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("n (training samples)")
ax.set_ylabel("EGOP angle distance to true H")
ax.set_title(f"EGOP error vs n at iteration {N_ITERATIONS} (rotation={ROTATION_MODE})")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
path = os.path.join(OUT_DIR, "expB_all.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"wrote {path}")

print(f"\nDone. Raw: {RAW_PATH}  Summary: {SUMMARY_PATH}")