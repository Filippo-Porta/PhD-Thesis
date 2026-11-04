"""
Aggregate per-run breakthrough.csv / tortuosity.csv (written by
compute_breakthrough_run.py), in TWO modes:

SINGLE-SECTOR MODE (unchanged, one sector at a time):
    python aggregate_breakthrough.py \\
        --outdir /hpc/home/filippo.porta/metrics/low --label low
    Produces the per-sector breakthrough grid + tortuosity plots
    described below, written inside that sector's own directory.

COMBINED MODE (NEW -- overlay low/medium/high on the same axes, one
figure per direction, for direct cross-sector comparison):
    python aggregate_breakthrough.py \\
        --data low:/hpc/home/filippo.porta/metrics/low \\
        --data medium:/hpc/home/filippo.porta/metrics/medium \\
        --data high:/hpc/home/filippo.porta/metrics/high \\
        --combined-outdir /hpc/home/filippo.porta/figures/combined
    Produces, for each direction X/Y/Z, ONE figure with all sectors
    overlaid (colored by sector, SECTOR_COLOR), instead of one figure
    per sector: mean curve (bold) + 95% CI (dashed) per sector, no
    individual-run traces (three sectors' worth would clutter the
    axes) -- exactly the "accorpare i 3 settori per direzione" request.

Reads, from each <outdir>:
    run_*/breakthrough.csv   (run,direction,plane,particle_file,
                              arrival_time,traj_length)
    run_*/tortuosity.csv     (run,direction,plane_from,plane_to,
                              particle_file,delta_l,delta_x,delta_t,
                              tortuosity)

Single-sector mode produces, in <outdir>:
    all_breakthrough.csv, all_tortuosity.csv   -- concatenated tables
    breakthrough_summary.csv                   -- t10/t50/t90 per
                                                  direction per plane
    tortuosity_summary.csv                     -- mean/std chi per
                                                  direction per segment
    btc_<DIR>.png / .pdf   -- CDF (top row) + CCDF (bottom row) grid,
                              one column per selected control plane,
                              ensemble mean (bold) + 95% CI band
                              (gray) + individual runs (thin, faint)
    tortuosity_<DIR>.png / .pdf -- <chi(x)> vs traveled distance,
                              mean +/- 95% CI across runs, with the
                              per-run values as faint scatter
    tortuosity_pdf_<DIR>.png / .pdf -- psi(chi) distribution: one KDE
                              curve per realization (thin), ensemble
                              mean (bold), 95% CI (dashed), Mean/
                              Variance text box -- matching the
                              per-realization tortuosity-distribution
                              figures in your references

Combined mode produces, in <combined-outdir>:
    tortuosity_pdf_combined_<DIR>.png / .pdf   -- psi(chi), all
                              sectors overlaid, Mean/Variance per
                              sector in the text box
    tortuosity_vsx_combined_<DIR>.png / .pdf   -- <chi(x)> vs x, all
                              sectors overlaid
    tortuosity_combined_summary.csv            -- sector,direction,
                              plane,n_runs,mean_chi,ci_lo,ci_hi
    tortuosity_pdf_combined_summary.csv         -- sector,direction,
                              n_runs,mean_chi,variance_chi

particles are injected flux-weighted at t=0 (per your methods text),
so each particle already carries equal statistical weight -- no
separate weight column is needed here (unlike fig5_btc_ccdf.py's
partime.dat, which does carry an explicit flux-weight column).
"""

import argparse
import csv
import glob
import math
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

DIRECTIONS = ("X", "Y", "Z")
DIR_COLORS = {"X": "tab:red", "Y": "tab:green", "Z": "tab:blue"}
SECTOR_COLOR = {"low": "tab:blue", "medium": "tab:orange",
                "high": "tab:red"}
SECTOR_ORDER = ("low", "medium", "high")


# ----------------------------------------------------------------------
# Loading (streamed with csv.reader, not DictReader -- these tables can
# be large: tens of runs x thousands of particles x ~10 planes each)
# ----------------------------------------------------------------------

def load_breakthrough(outdir):
    """dict[direction][plane][run] -> list of arrival_time."""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    paths = sorted(glob.glob(os.path.join(outdir, "run_*",
                                          "breakthrough.csv")))
    if not paths:
        raise SystemExit(f"No run_*/breakthrough.csv found in "
                         f"{outdir!r}")
    n_rows = 0
    for path in paths:
        run_id = os.path.basename(os.path.dirname(path))
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            idx = {name: i for i, name in enumerate(header)}
            for row in reader:
                try:
                    d = row[idx["direction"]]
                    plane = float(row[idx["plane"]])
                    t = float(row[idx["arrival_time"]])
                except (ValueError, IndexError):
                    continue
                data[d][plane][run_id].append(t)
                n_rows += 1
    print(f"Loaded {n_rows} breakthrough rows from {len(paths)} runs.")
    return data


def load_tortuosity(outdir):
    """
    Returns (by_plane, by_run):
      by_plane[direction][plane_to] -> list of (run, tortuosity)
          (used by the vs-distance plot / asymptotic convergence check)
      by_run[direction][run] -> list of tortuosity, POOLED over every
          plane pair -- used for the psi(chi) distribution plot, one
          KDE curve per realization.
    """
    by_plane = defaultdict(lambda: defaultdict(list))
    by_run = defaultdict(lambda: defaultdict(list))
    paths = sorted(glob.glob(os.path.join(outdir, "run_*",
                                          "tortuosity.csv")))
    if not paths:
        print(f"WARNING: no run_*/tortuosity.csv found in {outdir!r} "
             f"-- skipping tortuosity plots.")
        return by_plane, by_run
    n_rows = 0
    for path in paths:
        run_id = os.path.basename(os.path.dirname(path))
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            idx = {name: i for i, name in enumerate(header)}
            for row in reader:
                try:
                    d = row[idx["direction"]]
                    plane_to = float(row[idx["plane_to"]])
                    chi = float(row[idx["tortuosity"]])
                except (ValueError, IndexError):
                    continue
                by_plane[d][plane_to].append((run_id, chi))
                by_run[d][run_id].append(chi)
                n_rows += 1
    print(f"Loaded {n_rows} tortuosity rows from {len(paths)} runs.")
    return by_plane, by_run


# ----------------------------------------------------------------------
# Concatenated tables (all_breakthrough.csv, all_tortuosity.csv)
# ----------------------------------------------------------------------

def concat_csvs(outdir, filename, out_name):
    paths = sorted(glob.glob(os.path.join(outdir, "run_*", filename)))
    if not paths:
        return None
    out_path = os.path.join(outdir, out_name)
    with open(out_path, "w", newline="") as fout:
        writer = None
        for path in paths:
            with open(path, newline="") as fin:
                reader = csv.reader(fin)
                header = next(reader)
                if writer is None:
                    writer = csv.writer(fout)
                    writer.writerow(header)
                for row in reader:
                    writer.writerow(row)
    print(f"-> {out_path}")
    return out_path


# ----------------------------------------------------------------------
# Ensemble CDF helpers (unweighted -- particles are already injected
# flux-proportionally, per the methods section)
# ----------------------------------------------------------------------

def cdf_at(times_sorted, t_query):
    """Fraction of `times_sorted` <= each value in t_query."""
    n = len(times_sorted)
    return np.searchsorted(times_sorted, t_query, side="right") / n


def ensemble_curves(run_times, n_grid=60, tmax_percentile=90,
                    t_range=None):
    """
    run_times: dict[run] -> list of arrival times (one direction, one
    plane). Returns (t_grid, mean_cdf, lo_cdf, hi_cdf, per_run_curves)
    on a shared log-time grid, using a robust upper bound (percentile
    of per-run max, not the absolute max -- same rationale as
    fig5_btc_ccdf.py: a few outlier runs otherwise stretch the grid).
    per_run_curves is a list of (t_grid, cdf) for the thin background
    traces.

    t_range: if given as (tmin, tmax), use THIS grid instead of
    computing one locally from run_times. Pass the same global
    (tmin, tmax) -- computed once across every sector/plane/direction
    -- to every call so all panels share one time axis: otherwise a
    small-n group (e.g. a sector with fewer realizations) can have its
    OWN percentile-based tmax stretched far past where its ensemble
    has any real information, leaving a long flat, meaningless tail
    (a numerical residual, not decay) that looks wildly different from
    a same-position panel built on a better-populated group.
    """
    runs = {r: np.sort(np.asarray(t)) for r, t in run_times.items()
           if len(t)}
    if not runs:
        return None

    # ALWAYS check this, regardless of whether t_range is given: if
    # EVERY run has only t<=0 at this plane, it's the injection
    # boundary (trivial arrival time by definition, not real decay
    # information) and must be reported as "no data" -- this check
    # used to live only in the locally-computed branch below, so
    # passing a shared t_range silently bypassed it, letting a
    # degenerate all-zero group get plotted as a spurious flat CDF=1
    # line (and an empty CCDF, since 1-1=0 everywhere) instead of the
    # correct "no data" placeholder.
    any_positive = any(np.any(t > 0) for t in runs.values())
    if not any_positive:
        return None

    if t_range is not None:
        tmin, tmax = t_range
    else:
        tmin = min(t[0] for t in runs.values())
        run_maxs = [t[-1] for t in runs.values()]
        tmax = np.percentile(run_maxs, tmax_percentile)
        if tmin <= 0:
            positive_mins = [t[t > 0][0] for t in runs.values()
                            if np.any(t > 0)]
            tmin = min(positive_mins)  # safe: any_positive already True
    if tmax <= tmin:
        # degenerate: no meaningful spread to build a log-time grid on
        # (e.g. a plane immediately adjacent to the injection boundary
        # where almost every run's max is still ~0)
        return None
    t_grid = np.geomspace(tmin, tmax, n_grid)

    mat = np.array([cdf_at(t, t_grid) for t in runs.values()])
    mean_cdf = mat.mean(axis=0)
    lo_cdf = np.percentile(mat, 2.5, axis=0)
    hi_cdf = np.percentile(mat, 97.5, axis=0)

    per_run_curves = [(t, cdf_at(t, t)) for t in runs.values()]
    return t_grid, mean_cdf, lo_cdf, hi_cdf, per_run_curves, len(runs)


def crossing_time(t_grid, curve, threshold):
    """First t_grid value where `curve` reaches `threshold` (linear
    interpolation in log-t)."""
    above = np.where(curve >= threshold)[0]
    if not len(above):
        return None
    i = above[0]
    if i == 0:
        return t_grid[0]
    t0, t1 = t_grid[i - 1], t_grid[i]
    y0, y1 = curve[i - 1], curve[i]
    frac = (threshold - y0) / (y1 - y0) if y1 != y0 else 0.0
    return 10 ** (np.log10(t0) + frac * (np.log10(t1) - np.log10(t0)))


# ----------------------------------------------------------------------
# Breakthrough grid: CDF (top) + CCDF (bottom), one column per plane
# ----------------------------------------------------------------------

def plot_breakthrough_grid(data, direction, planes, outdir, label,
                           n_grid, tmax_percentile, summary_rows):
    color = DIR_COLORS[direction]
    n_cols = len(planes)
    fig, axes = plt.subplots(2, n_cols, figsize=(3.6 * n_cols, 6.4),
                             sharex="col")
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    any_plotted = False
    for col, plane in enumerate(planes):
        run_times = data.get(direction, {}).get(plane, {})
        result = ensemble_curves(run_times, n_grid, tmax_percentile)
        ax_cdf, ax_ccdf = axes[0][col], axes[1][col]

        # axis formatting ALWAYS applied, whether or not this column
        # has data -- otherwise a "no data" column stays on
        # matplotlib's default linear scale (never set explicitly
        # except as a side-effect of semilogx()/loglog(), which only
        # run when there IS data), producing garbled ticks
        ax_cdf.set_xscale("log")
        ax_ccdf.set_xscale("log")
        ax_ccdf.set_yscale("log")
        ax_cdf.set_ylim(0, 1)
        ax_cdf.xaxis.set_major_locator(LogLocator(base=10.0,
                                                  numticks=20))
        ax_ccdf.xaxis.set_major_locator(LogLocator(base=10.0,
                                                    numticks=20))
        ax_ccdf.yaxis.set_major_locator(LogLocator(base=10.0,
                                                    numticks=20))
        ax_cdf.set_title(f"$x$ = {plane:g} m", fontsize=10)
        ax_ccdf.set_xlabel("travel time")
        ax_cdf.grid(True, which="both", alpha=0.15)
        ax_ccdf.grid(True, which="both", alpha=0.15)

        if result is None:
            for ax in (ax_cdf, ax_ccdf):
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                       transform=ax.transAxes, fontsize=8, color="0.5")
            continue
        any_plotted = True
        t_grid, mean_cdf, lo_cdf, hi_cdf, per_run, n_runs = result

        for t, cdf in per_run:
            ax_cdf.semilogx(t, cdf, color=color, lw=0.5, alpha=0.15)
            mask = (1 - cdf) > 0
            ax_ccdf.loglog(t[mask], (1 - cdf)[mask], color=color,
                          lw=0.5, alpha=0.15)

        ax_cdf.fill_between(t_grid, lo_cdf, hi_cdf, color="0.5",
                            alpha=0.3, linewidth=0, zorder=3,
                            label="95% CI (realizations)")
        ax_cdf.semilogx(t_grid, mean_cdf, color=color, lw=2.2, zorder=5,
                       label=f"mean (n={n_runs} runs)")

        ccdf_mean = 1 - mean_cdf
        ccdf_lo = 1 - hi_cdf
        ccdf_hi = 1 - lo_cdf
        keep = ccdf_mean > 0
        ax_ccdf.fill_between(t_grid[keep], ccdf_lo[keep],
                            ccdf_hi[keep], color="0.5", alpha=0.3,
                            linewidth=0, zorder=3)
        ax_ccdf.loglog(t_grid[keep], ccdf_mean[keep], color=color,
                      lw=2.2, zorder=5)

        t10 = crossing_time(t_grid, mean_cdf, 0.1)
        t50 = crossing_time(t_grid, mean_cdf, 0.5)
        t90 = crossing_time(t_grid, mean_cdf, 0.9)
        summary_rows.append({"direction": direction, "plane": plane,
                            "n_runs": n_runs, "t10": t10, "t50": t50,
                            "t90": t90})

    axes[0][0].set_ylabel("CDF")
    axes[1][0].set_ylabel("CCDF")
    handles, legend_labels = axes[0][0].get_legend_handles_labels()
    if handles:
        axes[0][0].legend(fontsize=7, loc="lower right")
    else:
        # first column had no data (e.g. injection plane) -- put the
        # legend on the first column that does have a labeled curve
        for c in range(n_cols):
            handles, legend_labels = axes[0][c].get_legend_handles_labels()
            if handles:
                axes[0][c].legend(fontsize=7, loc="lower right")
                break

    title = f"Breakthrough curves, direction {direction}"
    if label:
        title += f" -- {label}"
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    base = os.path.join(outdir, f"btc_{direction}")
    if any_plotted:
        fig.savefig(f"{base}.png", dpi=300)
        fig.savefig(f"{base}.pdf")
        print(f"-> {base}.png / {base}.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------
# COMBINED breakthrough grid: same layout (CDF top, CCDF bottom, one
# column per plane), but ONE figure per direction with all sectors
# overlaid (colored by SECTOR_COLOR) instead of one figure per sector.
# No individual-run traces (three sectors' worth would clutter the
# axes, same call as the tortuosity combined plots) -- just mean
# (bold) + 95% CI (shaded) per sector.
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# COMBINED breakthrough grid: same layout (CDF top, CCDF bottom, one
# column per plane), but ONE figure per direction with all sectors
# overlaid (colored by SECTOR_COLOR) instead of one figure per sector.
# No individual-run traces (three sectors' worth would clutter the
# axes, same call as the tortuosity combined plots) -- just mean
# (bold) + 95% CI (shaded) per sector.
#
# Split into compute (no plotting) + render (no computation) so the
# caller can scan all THREE DIRECTIONS first, find a single shared
# CCDF y-axis, and only then draw -- otherwise each panel/direction
# auto-scales to its own truncation point and the decay rates aren't
# visually comparable (CDF axes are already a fixed 0-1, so this only
# matters for the log-scale CCDF row).
# ----------------------------------------------------------------------

def compute_global_time_range(sector_bt_data, directions, planes,
                              tmax_percentile=90):
    """
    Robust (tmin, tmax) computed ONCE across every sector, every
    direction, and every plane -- so every panel of every direction's
    combined breakthrough figure shares the exact same time axis
    (mirrors fig5_btc_ccdf.py's original "global x-axis" design).

    tmax is the given percentile of ALL per-run max arrival times
    pooled together (not per-group), so one small-n or one
    pathologically slow group can no longer stretch a single panel's
    grid far past where it has real information.
    """
    all_mins, all_maxs = [], []
    for sector_data in sector_bt_data.values():
        for direction in directions:
            for plane in planes:
                run_times = sector_data.get(direction, {}).get(plane, {})
                for t in run_times.values():
                    t = np.asarray(t)
                    if len(t) == 0:
                        continue
                    positive = t[t > 0]
                    if len(positive):
                        all_mins.append(float(positive.min()))
                        # only count this run's max toward tmax if it
                        # has SOME positive time -- a run that's
                        # entirely t<=0 (the injection boundary) would
                        # otherwise contribute a spurious max=0 that
                        # drags the percentile down for no reason
                        all_maxs.append(float(t.max()))
    if not all_mins or not all_maxs:
        return None
    tmin = min(all_mins)
    tmax = float(np.percentile(all_maxs, tmax_percentile))
    if tmax <= tmin:
        return None
    return tmin, tmax


def compute_breakthrough_stats(sector_bt_data, direction, planes,
                               n_grid, tmax_percentile, t_range=None):
    """Returns dict[plane][sector] = (t_grid, mean_cdf, lo_cdf,
    hi_cdf, n_runs) -- only entries where data exists."""
    stats = {p: {} for p in planes}
    for plane in planes:
        for sector in SECTOR_ORDER:
            run_times = sector_bt_data.get(sector, {}).get(
                direction, {}).get(plane, {})
            result = ensemble_curves(run_times, n_grid, tmax_percentile,
                                    t_range=t_range)
            if result is None:
                continue
            t_grid, mean_cdf, lo_cdf, hi_cdf, _, n_runs = result
            stats[plane][sector] = (t_grid, mean_cdf, lo_cdf, hi_cdf,
                                   n_runs)
    return stats


def render_breakthrough_grid_combined(stats, direction, planes, outdir,
                                      ccdf_ylim, t_range, summary_rows):
    n_cols = len(planes)
    fig, axes = plt.subplots(2, n_cols, figsize=(3.6 * n_cols, 6.4),
                             sharex="col")
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    any_plotted = False
    for col, plane in enumerate(planes):
        ax_cdf, ax_ccdf = axes[0][col], axes[1][col]
        pstats = stats.get(plane, {})

        # --- axis formatting ALWAYS applied, whether or not this
        #     column has data -- previously the log scale itself was
        #     only ever set as a side-effect of semilogx()/loglog()
        #     calls, which never ran for a "no data" column, leaving
        #     it on matplotlib's default LINEAR scale (that's what
        #     produced the "0.00  1.00" garbled ticks: a LogLocator
        #     applied to a linear axis) -----------------------------
        ax_cdf.set_xscale("log")
        ax_ccdf.set_xscale("log")
        ax_ccdf.set_yscale("log")

        if t_range is not None:
            ax_cdf.set_xlim(*t_range)
            ax_ccdf.set_xlim(*t_range)
        ax_cdf.set_ylim(0, 1)
        if ccdf_ylim is not None:
            ax_ccdf.set_ylim(*ccdf_ylim)

        # force every decade to get a tick (10^0, 10^1, ... included)
        # instead of matplotlib's default thinning, which otherwise
        # skips some decades to avoid crowding in a narrow column
        ax_cdf.xaxis.set_major_locator(LogLocator(base=10.0,
                                                  numticks=20))
        ax_ccdf.xaxis.set_major_locator(LogLocator(base=10.0,
                                                    numticks=20))
        ax_ccdf.yaxis.set_major_locator(LogLocator(base=10.0,
                                                    numticks=20))

        ax_cdf.set_title(f"$x$ = {plane:g} m", fontsize=10)
        ax_ccdf.set_xlabel("travel time")
        ax_cdf.grid(True, which="both", alpha=0.15)
        ax_ccdf.grid(True, which="both", alpha=0.15)

        if not pstats:
            for ax in (ax_cdf, ax_ccdf):
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                       transform=ax.transAxes, fontsize=8, color="0.5")
            continue
        any_plotted = True

        for sector in SECTOR_ORDER:
            if sector not in pstats:
                continue
            t_grid, mean_cdf, lo_cdf, hi_cdf, n_runs = pstats[sector]
            color = SECTOR_COLOR[sector]

            ax_cdf.fill_between(t_grid, lo_cdf, hi_cdf, color=color,
                               alpha=0.20, linewidth=0, zorder=3)
            ax_cdf.semilogx(t_grid, mean_cdf, color=color, lw=2.2,
                           zorder=5, label=f"{sector} (n={n_runs})")

            ccdf_mean = 1 - mean_cdf
            ccdf_lo = 1 - hi_cdf
            ccdf_hi = 1 - lo_cdf
            keep = ccdf_mean > 0
            ax_ccdf.fill_between(t_grid[keep], ccdf_lo[keep],
                                ccdf_hi[keep], color=color, alpha=0.20,
                                linewidth=0, zorder=3)
            ax_ccdf.loglog(t_grid[keep], ccdf_mean[keep], color=color,
                          lw=2.2, zorder=5)

            t10 = crossing_time(t_grid, mean_cdf, 0.1)
            t50 = crossing_time(t_grid, mean_cdf, 0.5)
            t90 = crossing_time(t_grid, mean_cdf, 0.9)
            summary_rows.append({"sector": sector, "direction": direction,
                                "plane": plane, "n_runs": n_runs,
                                "t10": t10, "t50": t50, "t90": t90})

    axes[0][0].set_ylabel("CDF")
    axes[1][0].set_ylabel("CCDF")
    for c in range(n_cols):
        handles, _ = axes[0][c].get_legend_handles_labels()
        if handles:
            axes[0][c].legend(fontsize=7, loc="lower right")
            break

    fig.suptitle(f"Breakthrough curves across sectors -- "
                f"direction {direction}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    base = os.path.join(outdir, f"btc_combined_{direction}")
    if any_plotted:
        fig.savefig(f"{base}.png", dpi=300)
        fig.savefig(f"{base}.pdf")
        print(f"-> {base}.png / {base}.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------
# Manual Gaussian KDE (no scipy dependency -- HPC conda envs vary)
# ----------------------------------------------------------------------

def gaussian_kde(samples, grid, max_samples=20000, rng=None):
    """
    Silverman-bandwidth Gaussian KDE, evaluated at `grid`. Subsamples
    to `max_samples` for speed if there are more points than that
    (bias-free: a uniform random subsample of a large ensemble gives
    the same density estimate, just noisier -- fine at these sizes).
    """
    samples = np.asarray(samples, dtype=float)
    n = len(samples)
    if n < 2:
        return np.zeros_like(grid)
    if n > max_samples:
        rng = rng or np.random.default_rng(0)
        samples = rng.choice(samples, size=max_samples, replace=False)
        n = max_samples
    std = samples.std(ddof=1)
    if std <= 0:
        return np.zeros_like(grid)
    bw = 1.06 * std * n ** (-1 / 5)
    diffs = (grid[:, None] - samples[None, :]) / bw
    dens = np.exp(-0.5 * diffs ** 2).sum(axis=1) / (n * bw * math.sqrt(2 * math.pi))
    return dens


# ----------------------------------------------------------------------
# COMPUTE-ONLY helpers: return numbers, draw nothing. This lets the
# caller collect stats for ALL THREE DIRECTIONS (or, in combined mode,
# all sectors) FIRST, work out one shared y-axis, and only then
# render -- so panels are always on a directly comparable scale
# instead of each auto-scaling to its own data.
# ----------------------------------------------------------------------

def compute_vsx_stats(per_plane):
    """
    per_plane: dict[plane] -> list of (run_id, chi).

    Per-run aggregate is the MEDIAN chi across that run's particles at
    this plane -- robust to the rare particle with a pathological
    detour (huge chi), which would otherwise dominate a per-run MEAN
    and blow up the y-axis/CI for the whole plot. The ensemble center
    across runs is then the MEAN of those per-run medians, so it is
    still <chi(x)> ("ensemble-averaged") in the sense of your methods
    text, just no longer at the mercy of a single outlier particle.

    Returns (planes, means, los, his, ns) -- one entry per plane.
    """
    planes = sorted(per_plane)
    means, los, his, ns = [], [], [], []
    for plane in planes:
        by_run = defaultdict(list)
        for run_id, chi in per_plane[plane]:
            by_run[run_id].append(chi)
        run_medians = [float(np.median(v)) for v in by_run.values()]
        m = float(np.mean(run_medians))
        if len(run_medians) > 1:
            lo = float(np.percentile(run_medians, 2.5))
            hi = float(np.percentile(run_medians, 97.5))
        else:
            lo = hi = m
        means.append(m)
        los.append(lo)
        his.append(hi)
        ns.append(len(run_medians))
    return planes, means, los, his, ns


def compute_pdf_stats(runs, chi_max=4.0, n_grid=200):
    """
    runs: dict[run_id] -> list of chi (pooled over every plane pair).
    Returns (grid, mean_curve, lo_curve, hi_curve, curves,
    overall_mean, overall_var, n_runs). `curves` is the full
    per-run KDE matrix (for the thin background traces in
    single-sector mode; combined mode ignores it).
    """
    grid = np.linspace(1.0, chi_max, n_grid)
    curves = np.array([gaussian_kde(v, grid) for v in runs.values()])
    mean_curve = curves.mean(axis=0)
    lo_curve = np.percentile(curves, 2.5, axis=0)
    hi_curve = np.percentile(curves, 97.5, axis=0)
    pooled = np.concatenate(list(runs.values()))
    return (grid, mean_curve, lo_curve, hi_curve, curves,
           float(pooled.mean()), float(pooled.var()), len(runs))


# ----------------------------------------------------------------------
# RENDER-ONLY helpers: draw one sector/direction's curve onto a given
# Axes, plus small "finalize" helpers for labels/legend/limits/saving.
# Reused by both single-sector mode (one color, one call) and combined
# mode (looped once per sector, several colors on one Axes).
# ----------------------------------------------------------------------

def render_vsx(ax, planes, means, los, his, color, label,
               scatter_xy=None):
    if scatter_xy is not None and scatter_xy[0]:
        xs, ys = scatter_xy
        ax.scatter(xs, ys, s=4, color=color, alpha=0.06, linewidths=0,
                  zorder=1)
    ax.fill_between(planes, los, his, color=color, alpha=0.22,
                    linewidth=0, zorder=3)
    ax.plot(planes, means, color=color, lw=2.2, marker="o",
           markersize=4, zorder=5, label=label)


def render_pdf(ax, grid, mean_curve, lo_curve, hi_curve, color, label,
              curves=None):
    if curves is not None:
        for curve in curves:
            ax.plot(grid, curve, color=color, lw=0.6, alpha=0.25)
    ax.plot(grid, mean_curve, color=color, lw=2.4, label=label)
    ax.plot(grid, lo_curve, color=color, lw=1.0, ls="--", alpha=0.8)
    ax.plot(grid, hi_curve, color=color, lw=1.0, ls="--", alpha=0.8)


def finalize_vsx_axes(ax, title, ylim):
    """Legend at upper right: the vs-distance curves are flat, not
    peaked at one edge like the PDFs, so this never covers data."""
    ax.axhline(1.0, color="0.3", ls=":", lw=1,
              label="straight path (χ=1)")
    ax.set_xlabel("traveled distance $x$ [m]")
    ax.set_ylabel(r"local tortuosity $\chi$")
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2)


def finalize_pdf_axes(ax, title, chi_max, ylim):
    """Legend at upper right: chi distributions peak near chi=1 (the
    physical minimum, left edge) and decay rightward, so upper-right
    is the one corner that is reliably clear of data -- unlike
    upper-left, which used to sit right on top of the tallest 95% CI
    peaks and hide them.

    x-axis starts a bit BEFORE 1.0 (chi's physical minimum, a
    straight path) with a dotted vertical line marking it -- purely
    cosmetic, so the curve doesn't look "cut" flush against the left
    frame; the underlying KDE itself is unchanged."""
    margin = 0.06 * (chi_max - 1.0)
    ax.set_xlim(1.0 - margin, chi_max)
    ax.axvline(1.0, color="0.4", ls=":", lw=1, zorder=1)
    ax.text(1.0, ax.get_ylim()[1] if ylim is None else ylim[1],
           r" $\chi_{min}=1$", fontsize=7, color="0.4",
           ha="left", va="top")
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(bottom=0)
    ax.set_xlabel(r"Tortuosity $\chi$")
    ax.set_ylabel(r"$\psi(\chi)$")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.2)


def save_fig(fig, outdir, filename):
    base = os.path.join(outdir, filename)
    fig.savefig(f"{base}.png", dpi=300)
    fig.savefig(f"{base}.pdf")
    print(f"-> {base}.png / {base}.pdf")
    plt.close(fig)


def detect_trivial_planes(bt_data_list, directions, candidate_planes):
    """
    bt_data_list: list of `load_breakthrough()`-style dicts (one per
    sector in combined mode, or a single one-item list in
    single-sector mode).

    Returns the SET of plane values that are trivial (arrival_time
    <= 0 for EVERY particle, pooled across every one of those dicts,
    every run) in AT LEAST ONE of `directions` -- i.e. that
    direction's injection boundary. Different directions can have
    their injection boundary at OPPOSITE ends of the same numeric
    plane range (e.g. X's is at the domain minimum, Y's turned out to
    be at the domain maximum instead) -- excluding whichever planes
    are trivial for ANY direction from the shared default selection
    means no panel in the combined X/Y/Z figures ever lands on a
    guaranteed "no data" column by surprise, while still showing the
    IDENTICAL set of planes across all three direction figures.
    """
    trivial = set()
    for direction in directions:
        for plane in candidate_planes:
            any_positive = False
            for bt_data in bt_data_list:
                run_times = bt_data.get(direction, {}).get(plane, {})
                for t in run_times.values():
                    if any(x > 0 for x in t):
                        any_positive = True
                        break
                if any_positive:
                    break
            if not any_positive:
                trivial.add(plane)
    return trivial


def choose_default_planes(all_planes, trivial_planes, n_show=4):
    """All_planes minus whichever are trivial in some direction, then
    n_show values evenly spread across what's left."""
    pool = [p for p in all_planes if p not in trivial_planes]
    if not pool:
        pool = all_planes  # degenerate fallback, shouldn't normally happen
    n_show = min(n_show, len(pool))
    idxs = np.linspace(0, len(pool) - 1, n_show).round()
    idxs = sorted(set(int(i) for i in idxs))
    return [pool[i] for i in idxs]


def run_single_sector(outdir, label, planes_arg, n_grid,
                     tmax_percentile, skip_concat):
    """Original single-sector behavior, unchanged."""
    if not skip_concat:
        concat_csvs(outdir, "breakthrough.csv", "all_breakthrough.csv")
        concat_csvs(outdir, "tortuosity.csv", "all_tortuosity.csv")

    bt_data = load_breakthrough(outdir)
    tort_by_plane, tort_by_run = load_tortuosity(outdir)

    all_planes = sorted({p for d in bt_data.values() for p in d})
    if planes_arg:
        planes = [float(x) for x in planes_arg.split(",")]
    else:
        # auto-detect which planes are trivial (t<=0 for every
        # particle) in ANY of X/Y/Z -- different directions can have
        # their injection boundary at OPPOSITE ends of the domain, so
        # excluding just the global minimum isn't enough (still
        # selectable explicitly via --planes if you want to see one)
        trivial_planes = detect_trivial_planes([bt_data], DIRECTIONS,
                                               all_planes)
        if trivial_planes:
            print(f"Excluding trivial (injection-boundary) planes "
                 f"from default selection: {sorted(trivial_planes)}")
        planes = choose_default_planes(all_planes, trivial_planes)
    print(f"Control planes shown in breakthrough grid: {planes}")

    bt_summary, tort_summary, pdf_summary = [], [], []

    # ---- PASS 1: compute tortuosity stats for X/Y/Z, track a single
    #      shared y-axis across the three directions -------------------
    vsx_stats, vsx_scatter = {}, {}
    pdf_stats = {}
    vsx_max, pdf_max = 0.0, 0.0

    for direction in DIRECTIONS:
        per_plane = tort_by_plane.get(direction, {})
        if per_plane:
            vsx_planes, means, los, his, ns = compute_vsx_stats(per_plane)
            vsx_stats[direction] = (vsx_planes, means, los, his, ns)
            if his:
                vsx_max = max(vsx_max, max(his))
            xs, ys = [], []
            for plane, entries in per_plane.items():
                for run_id, chi in entries:
                    xs.append(plane)
                    ys.append(chi)
            vsx_scatter[direction] = (xs, ys)

        runs = {r: v for r, v in tort_by_run.get(direction, {}).items()
               if len(v) >= 5}
        if runs:
            stats = compute_pdf_stats(runs)
            pdf_stats[direction] = stats
            pdf_max = max(pdf_max, float(stats[3].max()))

    vsx_ylim = (0, vsx_max * 1.08) if vsx_max > 0 else None
    pdf_ylim = (0, pdf_max * 1.08) if pdf_max > 0 else None

    # ---- PASS 2: breakthrough grid (unchanged) + tortuosity renders --
    for direction in DIRECTIONS:
        if direction not in bt_data:
            print(f"WARNING: no breakthrough data for direction "
                 f"{direction}")
        else:
            plot_breakthrough_grid(bt_data, direction, planes, outdir,
                                  label, n_grid, tmax_percentile,
                                  bt_summary)

        color = DIR_COLORS[direction]

        if direction in vsx_stats:
            dplanes, means, los, his, ns = vsx_stats[direction]
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            render_vsx(ax, dplanes, means, los, his, color,
                     r"$\langle\chi(x)\rangle$",
                     scatter_xy=vsx_scatter.get(direction))
            title = f"Ensemble-averaged tortuosity, direction {direction}"
            if label:
                title += f" -- {label}"
            finalize_vsx_axes(ax, title, vsx_ylim)
            fig.tight_layout()
            save_fig(fig, outdir, f"tortuosity_{direction}")
            for plane, m, lo, hi, n in zip(dplanes, means, los, his, ns):
                tort_summary.append({"direction": direction,
                                    "plane": plane, "n_runs": n,
                                    "mean_chi": m, "ci_lo": lo,
                                    "ci_hi": hi})

        if direction in pdf_stats:
            (grid, mean_curve, lo_curve, hi_curve, curves,
            overall_mean, overall_var, n_runs) = pdf_stats[direction]
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            render_pdf(ax, grid, mean_curve, lo_curve, hi_curve, color,
                     f"mean (n={n_runs} runs)", curves=curves)
            ax.text(0.97, 0.05, f"Mean: {overall_mean:.2f}\n"
                                f"Variance: {overall_var:.2f}",
                   transform=ax.transAxes, ha="right", va="bottom",
                   fontsize=9,
                   bbox=dict(facecolor="white", edgecolor="0.7",
                            alpha=0.85))
            title = f"Tortuosity distribution, direction {direction}"
            if label:
                title += f" -- {label}"
            finalize_pdf_axes(ax, title, chi_max=4.0, ylim=pdf_ylim)
            fig.tight_layout()
            save_fig(fig, outdir, f"tortuosity_pdf_{direction}")
            pdf_summary.append({"direction": direction,
                               "n_runs": n_runs,
                               "mean_chi": overall_mean,
                               "variance_chi": overall_var})

    if bt_summary:
        path = os.path.join(outdir, "breakthrough_summary.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["direction", "plane",
                                              "n_runs", "t10", "t50",
                                              "t90"])
            w.writeheader()
            w.writerows(bt_summary)
        print(f"-> {path}")

    if tort_summary:
        path = os.path.join(outdir, "tortuosity_summary.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["direction", "plane",
                                              "n_runs", "mean_chi",
                                              "ci_lo", "ci_hi"])
            w.writeheader()
            w.writerows(tort_summary)
        print(f"-> {path}")

    if pdf_summary:
        path = os.path.join(outdir, "tortuosity_pdf_summary.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["direction", "n_runs",
                                              "mean_chi", "variance_chi"])
            w.writeheader()
            w.writerows(pdf_summary)
        print(f"-> {path}")


def run_combined(sector_paths, combined_outdir, planes_arg=None,
                 n_grid=60, tmax_percentile=90, metrics_outdir=None):
    """sector_paths: dict label -> outdir. Overlays sectors per
    direction, into combined_outdir, with ONE shared y-axis across
    X/Y/Z for each plot type (so panels are directly comparable).
    Figures go to combined_outdir; summary CSVs go to metrics_outdir
    (defaults to the common parent of the --data paths, e.g. the
    shared .../metrics/ root, so CSVs land next to the per-run data
    they were aggregated from, not mixed in with the figures)."""
    os.makedirs(combined_outdir, exist_ok=True)

    if metrics_outdir is None:
        try:
            metrics_outdir = os.path.commonpath(list(sector_paths.values()))
        except ValueError:
            metrics_outdir = combined_outdir  # different drives etc.
    os.makedirs(metrics_outdir, exist_ok=True)
    print(f"Figures -> {combined_outdir}\nSummary CSVs -> {metrics_outdir}")

    sector_by_plane, sector_by_run = {}, {}
    sector_bt_data = {}
    for label, path in sector_paths.items():
        print(f"\n[{label}] loading tortuosity from {path}")
        by_plane, by_run = load_tortuosity(path)
        sector_by_plane[label] = by_plane
        sector_by_run[label] = by_run
        print(f"[{label}] loading breakthrough from {path}")
        sector_bt_data[label] = load_breakthrough(path)

    # ---- PASS 1: compute stats for every (direction, sector), track
    #      one shared y-axis per plot type across all of X/Y/Z --------
    vsx_stats = {d: {} for d in DIRECTIONS}
    pdf_stats = {d: {} for d in DIRECTIONS}
    vsx_summary, pdf_summary = [], []
    vsx_max, pdf_max = 0.0, 0.0

    for direction in DIRECTIONS:
        for sector in SECTOR_ORDER:
            per_plane = sector_by_plane.get(sector, {}).get(direction, {})
            if per_plane:
                planes, means, los, his, ns = compute_vsx_stats(per_plane)
                vsx_stats[direction][sector] = (planes, means, los, his)
                if his:
                    vsx_max = max(vsx_max, max(his))
                for plane, m, lo, hi, n in zip(planes, means, los,
                                               his, ns):
                    vsx_summary.append({"sector": sector,
                                       "direction": direction,
                                       "plane": plane, "n_runs": n,
                                       "mean_chi": m, "ci_lo": lo,
                                       "ci_hi": hi})

            runs = {r: v for r, v in
                   sector_by_run.get(sector, {}).get(direction, {}).items()
                   if len(v) >= 5}
            if runs:
                grid, mean_curve, lo_curve, hi_curve, _, overall_mean, \
                    overall_var, n_runs = compute_pdf_stats(runs)
                pdf_stats[direction][sector] = (grid, mean_curve,
                                               lo_curve, hi_curve,
                                               overall_mean, n_runs)
                pdf_max = max(pdf_max, float(hi_curve.max()))
                pdf_summary.append({"sector": sector,
                                   "direction": direction,
                                   "n_runs": n_runs,
                                   "mean_chi": overall_mean,
                                   "variance_chi": overall_var})

    vsx_ylim = (0, vsx_max * 1.08) if vsx_max > 0 else None
    pdf_ylim = (0, pdf_max * 1.08) if pdf_max > 0 else None

    # ---- choose which planes to show in the combined breakthrough
    #      grid: auto-detect which planes are trivial (t<=0 for every
    #      particle) in ANY of X/Y/Z across ANY sector -- different
    #      directions can have their injection boundary at OPPOSITE
    #      ends of the domain, so excluding just the global minimum
    #      isn't enough (still selectable explicitly via --planes) --
    all_planes = sorted({p for bt in sector_bt_data.values()
                         for d in bt.values() for p in d})
    if planes_arg:
        bt_planes = [float(x) for x in planes_arg.split(",")]
    else:
        trivial_planes = detect_trivial_planes(
            list(sector_bt_data.values()), DIRECTIONS, all_planes)
        if trivial_planes:
            print(f"Excluding trivial (injection-boundary) planes "
                 f"from default selection: {sorted(trivial_planes)}")
        bt_planes = choose_default_planes(all_planes, trivial_planes)
    print(f"Control planes shown in combined breakthrough grid: "
         f"{bt_planes}")

    bt_summary = []

    # ---- PASS 1 (breakthrough): ONE global time range across every
    #      sector, plane, AND direction -- computed first, before any
    #      per-group grid is built, so every panel of every direction
    #      figure shares the exact same x-axis (mirrors the "global
    #      x-axis" approach in fig5_btc_ccdf.py). Without this, a
    #      small-n group (e.g. a sector with fewer realizations) can
    #      have ITS OWN percentile-based tmax stretched far past where
    #      it has real information, producing a long flat, meaningless
    #      tail that isn't comparable to a same-position panel built
    #      from a better-populated group ---------------------------
    global_t_range = compute_global_time_range(
        sector_bt_data, DIRECTIONS, bt_planes, tmax_percentile)
    if global_t_range is not None:
        print(f"Global breakthrough time axis (shared by every panel, "
             f"every direction): [{global_t_range[0]:.3e}, "
             f"{global_t_range[1]:.3e}]")

    # ---- compute stats for X/Y/Z on that SAME global grid, and also
    #      find one shared CCDF y-axis across all of them (the
    #      log-scale row auto-scales per panel/direction otherwise;
    #      the CDF row is already a fixed 0-1, so it needs no such
    #      pass) -----------------------------------------------------
    bt_stats_by_dir = {}
    ccdf_global_min = None
    for direction in DIRECTIONS:
        stats = compute_breakthrough_stats(sector_bt_data, direction,
                                          bt_planes, n_grid,
                                          tmax_percentile,
                                          t_range=global_t_range)
        bt_stats_by_dir[direction] = stats
        for pstats in stats.values():
            for (t_grid, mean_cdf, lo_cdf, hi_cdf,
                n_runs) in pstats.values():
                ccdf_mean = 1 - mean_cdf
                keep = ccdf_mean > 0
                if keep.any():
                    m = float(ccdf_mean[keep].min())
                    ccdf_global_min = (m if ccdf_global_min is None
                                      else min(ccdf_global_min, m))

    if ccdf_global_min is not None and ccdf_global_min > 0:
        # round DOWN to the nearest power of ten for a clean log axis,
        # e.g. 0.00043 -> 1e-4, with a bit of headroom at the top
        bottom = 10 ** math.floor(math.log10(ccdf_global_min))
        ccdf_ylim = (bottom, 1.5)
    else:
        ccdf_ylim = None

    # ---- PASS 2: render, same y-axis for X/Y/Z in each plot type ----
    for direction in DIRECTIONS:
        render_breakthrough_grid_combined(
            bt_stats_by_dir[direction], direction, bt_planes,
            combined_outdir, ccdf_ylim, global_t_range, bt_summary)

        stats = vsx_stats[direction]
        if stats:
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            for sector in SECTOR_ORDER:
                if sector not in stats:
                    continue
                planes, means, los, his = stats[sector]
                render_vsx(ax, planes, means, los, his,
                         SECTOR_COLOR[sector], sector)
            finalize_vsx_axes(
                ax, f"Ensemble-averaged tortuosity across sectors -- "
                   f"direction {direction}", vsx_ylim)
            fig.tight_layout()
            save_fig(fig, combined_outdir,
                    f"tortuosity_vsx_combined_{direction}")

        pstats = pdf_stats[direction]
        if pstats:
            fig, ax = plt.subplots(figsize=(6, 4.6))
            for sector in SECTOR_ORDER:
                if sector not in pstats:
                    continue
                grid, mean_curve, lo_curve, hi_curve, overall_mean, \
                    n_runs = pstats[sector]
                render_pdf(ax, grid, mean_curve, lo_curve, hi_curve,
                         SECTOR_COLOR[sector],
                         f"{sector} (n={n_runs}, μ={overall_mean:.2f})")
            finalize_pdf_axes(
                ax, f"Tortuosity distribution across sectors -- "
                   f"direction {direction}", chi_max=4.0,
                ylim=pdf_ylim)
            fig.tight_layout()
            save_fig(fig, combined_outdir,
                    f"tortuosity_pdf_combined_{direction}")

    if bt_summary:
        path = os.path.join(metrics_outdir,
                           "breakthrough_combined_summary.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["sector", "direction",
                                              "plane", "n_runs", "t10",
                                              "t50", "t90"])
            w.writeheader()
            w.writerows(bt_summary)
        print(f"-> {path}")

    if pdf_summary:
        path = os.path.join(metrics_outdir,
                           "tortuosity_pdf_combined_summary.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["sector", "direction",
                                              "n_runs", "mean_chi",
                                              "variance_chi"])
            w.writeheader()
            w.writerows(pdf_summary)
        print(f"-> {path}")

    if vsx_summary:
        path = os.path.join(metrics_outdir,
                           "tortuosity_combined_summary.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["sector", "direction",
                                              "plane", "n_runs",
                                              "mean_chi", "ci_lo",
                                              "ci_hi"])
            w.writeheader()
            w.writerows(vsx_summary)
        print(f"-> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=None,
                    help="SINGLE-SECTOR mode: sector metrics dir, e.g. "
                        "/hpc/home/filippo.porta/metrics/low -- must "
                        "contain run_*/breakthrough.csv (and "
                        "run_*/tortuosity.csv)")
    ap.add_argument("--label", default="",
                    help="sector label for plot titles (single-sector "
                        "mode only), e.g. 'low'")
    ap.add_argument("--data", action="append", default=None,
                    help="COMBINED mode (repeatable): label:path, e.g. "
                        "--data low:/.../metrics/low --data "
                        "medium:/.../metrics/medium --data "
                        "high:/.../metrics/high")
    ap.add_argument("--combined-outdir", default=None,
                    help="where combined-mode FIGURES are written "
                        "(required if --data is used)")
    ap.add_argument("--metrics-outdir", default=None,
                    help="[combined mode] where the summary CSVs are "
                        "written -- separate from --combined-outdir "
                        "so figures and data tables don't mix. "
                        "Default: the common parent directory of the "
                        "--data paths (e.g. .../metrics/ if you passed "
                        ".../metrics/low, .../metrics/medium, etc.)")
    ap.add_argument("--planes", default=None,
                    help="[single-sector mode] comma-separated list "
                        "of control planes to show in the "
                        "breakthrough grid, e.g. '-15,-5,5,15'. "
                        "Default: 4 planes evenly spread across the "
                        "ones present in the data.")
    ap.add_argument("--n-grid", type=int, default=60)
    ap.add_argument("--tmax-percentile", type=float, default=90)
    ap.add_argument("--skip-concat", action="store_true",
                    help="[single-sector mode] skip writing "
                        "all_breakthrough.csv / all_tortuosity.csv "
                        "(large files, slow on network filesystems) "
                        "-- plots/summaries are produced either way")
    args = ap.parse_args()

    if args.data:
        if not args.combined_outdir:
            raise SystemExit("--combined-outdir is required when "
                             "using --data (combined mode)")
        sector_paths = {}
        for spec in args.data:
            label, path = spec.split(":", 1)
            sector_paths[label] = path
        unknown = set(sector_paths) - set(SECTOR_COLOR)
        if unknown:
            raise SystemExit(f"Unknown sector label(s) {unknown} -- "
                             f"expected one of {list(SECTOR_COLOR)} "
                             f"(edit SECTOR_COLOR in the script to "
                             f"add more)")
        run_combined(sector_paths, args.combined_outdir, args.planes,
                    args.n_grid, args.tmax_percentile,
                    metrics_outdir=args.metrics_outdir)
    elif args.outdir:
        run_single_sector(args.outdir, args.label, args.planes,
                         args.n_grid, args.tmax_percentile,
                         args.skip_concat)
    else:
        raise SystemExit("Provide either --outdir (single-sector "
                         "mode) or one or more --data label:path "
                         "(combined mode)")


if __name__ == "__main__":
    main()
