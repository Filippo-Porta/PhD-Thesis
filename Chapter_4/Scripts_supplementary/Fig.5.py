
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import (apply_style, save, DIR_LABEL, SECTOR_COLOR,
                         panel_label)
import matplotlib.pyplot as plt


def fit_tail_slope(t, y, frac_of_peak=0.3, min_points=6):
    """
    Fits a power law y ~ t^slope to the TAIL of a log-log curve
    (Sweeney et al. 2023-style annotated exponent). The tail is taken
    from the point where y first drops below `frac_of_peak` times its
    peak value, through to the last positive value. Returns
    (slope, t_fit, y_fit) for drawing a guide line, or None if there
    aren't enough tail points to fit reliably.
    """
    mask = y > 0
    t, y = np.asarray(t)[mask], np.asarray(y)[mask]
    if len(t) < min_points:
        return None
    peak = y.max()
    start = np.argmax(y < frac_of_peak * peak)
    if start == 0 or len(t) - start < min_points:
        start = max(0, len(t) - min_points)
    tt, yy = t[start:], y[start:]
    if len(tt) < min_points:
        return None
    logt, logy = np.log10(tt), np.log10(yy)
    slope, intercept = np.polyfit(logt, logy, 1)
    t_fit = np.array([tt[0], tt[-1]])
    y_fit = 10 ** (intercept + slope * np.log10(t_fit))
    return slope, t_fit, y_fit


def load_partime(path):
    """Returns (times, weights) arrays; skips header line."""
    times, weights = [], []
    with open(path, encoding="utf-8-sig") as f:
        for i, line in enumerate(f):
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                w = float(parts[1])
                t = float(parts[2])
            except ValueError:
                continue  # header
            if t > 0:
                times.append(t)
                weights.append(w)
    return np.array(times), np.array(weights)


def ccdf_step(times, weights):
    """Sorted (t, ccdf) step function for one run, flux-weighted."""
    order = np.argsort(times)
    t = times[order]
    w = weights[order]
    w = w / w.sum()
    ccdf = 1.0 - np.cumsum(w)
    return t, np.clip(ccdf, 0.0, 1.0)


def ccdf_at(times, weights, t_query):
    """
    Evaluate P(T > t_query) directly (not via the sorted step array) --
    robust for querying at arbitrary common grid points across runs
    with different time supports.
    """
    w = weights / weights.sum()
    out = np.empty(len(t_query))
    for i, tq in enumerate(t_query):
        out[i] = w[times > tq].sum()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="append", required=True,
                    help="SECTOR:DIR:path/to/partime.dat (repeatable -- "
                        "one entry PER RUN, same sector+dir kept as "
                        "separate ensemble members, not pooled)")
    ap.add_argument("--n-grid", type=int, default=60,
                    help="points in the common log-time grid used to "
                        "compute the ensemble median/IQR band")
    ap.add_argument("--tmax-percentile", type=float, default=90,
                    help="use this percentile of PER-RUN max times "
                        "(not the absolute max) to set the grid's "
                        "upper bound -- a few outlier runs (particles "
                        "trapped in a pathologically slow path) "
                        "otherwise stretch the shared grid, squashing "
                        "all other runs into a sliver of the axis. "
                        "Outlier runs' own thin traces are still drawn "
                        "in full, just excluded from setting the grid.")
    ap.add_argument("--annotate-slope", action="store_true", default=True,
                    help="fit and annotate the power-law tail slope on "
                        "the ensemble median (Sweeney et al. 2023 style)")
    ap.add_argument("--no-annotate-slope", dest="annotate_slope",
                    action="store_false")
    ap.add_argument("--xlim-min", type=float, default=None,
                    help="manually fix the shared x-axis lower bound "
                        "[years] -- use this to keep axes identical "
                        "across separate script runs/figures (e.g. "
                        "producing low/medium/high separately but "
                        "wanting them visually comparable). If not "
                        "given, computed automatically from ALL data "
                        "passed in THIS run.")
    ap.add_argument("--xlim-max", type=float, default=None,
                    help="manually fix the shared x-axis upper bound "
                        "[years] -- see --xlim-min")
    ap.add_argument("--figdir", default="figures")
    args = ap.parse_args()

    apply_style()

    # pool[(sector, dir)] = list of (times, weights), ONE PER RUN
    pool = {}
    for spec in args.data:
        sector, d, path = spec.split(":", 2)
        d = d.lower()
        t, w = load_partime(path)
        pool.setdefault((sector, d), []).append((t, w))
        print(f"[{sector}/{d}] {path}: {len(t)} particles")

    sectors = sorted({s for s, _ in pool},
                     key=lambda s: ["low", "medium", "high"].index(s)
                     if s in ("low", "medium", "high") else 99)

    # -- GLOBAL time-axis range, shared identically by every panel --
    # Computed once across ALL sectors and ALL directions combined (not
    # per-panel), so the x-axis is directly comparable across direction
    # AND across intensity. A few pathologically slow particles would
    # otherwise stretch individual panels by very different amounts.
    if args.xlim_min is not None and args.xlim_max is not None:
        global_tmin, global_tmax = args.xlim_min, args.xlim_max
        print(f"Using manually fixed x-axis: [{global_tmin:.3e}, "
             f"{global_tmax:.3e}] years")
    else:
        all_tmins, all_tmaxs = [], []
        for runs in pool.values():
            for t, _ in runs:
                if len(t):
                    all_tmins.append(t.min())
                    all_tmaxs.append(t.max())
        global_tmin = args.xlim_min or min(all_tmins)
        global_tmax = args.xlim_max or np.percentile(
            all_tmaxs, args.tmax_percentile)
        print(f"Global x-axis (from all sectors/directions, "
             f"{args.tmax_percentile:.0f}th pct of per-run max): "
             f"[{global_tmin:.3e}, {global_tmax:.3e}] years")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), sharex="col")
    summary_rows = []

    for col, (d, letter) in enumerate(zip(("x", "y", "z"), "abc")):
        ax_cdf, ax = axes[0][col], axes[1][col]
        panel_label(ax, letter)
        for sector in sectors:
            runs = pool.get((sector, d))
            if not runs:
                continue
            color = SECTOR_COLOR.get(sector, "0.3")

            # -- thin individual-run curves, background (CCDF e CDF) --
            for t, w in runs:
                rt, rccdf = ccdf_step(t, w)
                mask = rccdf > 0
                ax.loglog(rt[mask], rccdf[mask], color=color, lw=0.7,
                         alpha=0.25, zorder=2)
                ax_cdf.semilogx(rt, 1.0 - rccdf, color=color, lw=0.7,
                               alpha=0.25, zorder=2)

            # -- ensemble median + IQR band, foreground --
            if len(runs) >= 2:
                run_tmins = [t.min() for t, _ in runs if len(t)]
                run_tmaxs = [t.max() for t, _ in runs if len(t)]
                tmin = min(run_tmins)
                # robust upper bound: a percentile of per-run maxima,
                # not the absolute max -- see --tmax-percentile help
                tmax = np.percentile(run_tmaxs, args.tmax_percentile)
                n_excluded = sum(1 for m in run_tmaxs if m > tmax)
                if n_excluded:
                    print(f"  [{sector}/{d}] grid tmax = {tmax:.3e} "
                         f"({args.tmax_percentile:.0f}th pct of per-run "
                         f"max) -- {n_excluded} outlier run(s) extend "
                         f"beyond it as thin traces only")
                t_grid = np.geomspace(tmin, tmax, args.n_grid)
                mat = np.array([ccdf_at(t, w, t_grid) for t, w in runs])
                med = np.median(mat, axis=0)
                q25 = np.percentile(mat, 25, axis=0)
                q75 = np.percentile(mat, 75, axis=0)

                keep = med > 0
                ax.fill_between(t_grid[keep], q25[keep], q75[keep],
                                color=color, alpha=0.22, zorder=3,
                                linewidth=0)
                ax.loglog(t_grid[keep], med[keep], color=color, lw=2.6,
                         zorder=5, label=f"{sector} (n={len(runs)})")

                cdf_med = 1.0 - med
                cdf_q25 = 1.0 - q75
                cdf_q75 = 1.0 - q25
                ax_cdf.fill_between(t_grid, cdf_q25, cdf_q75, color=color,
                                    alpha=0.22, zorder=3, linewidth=0)
                ax_cdf.semilogx(t_grid, cdf_med, color=color, lw=2.6, zorder=5)

                # -- numeric summary, printed to stdout (not read off
                # the plot by eye): t10/t50/t90 arrival times (where
                # the ensemble-median CCDF crosses 0.9/0.5/0.1 -- i.e.
                # the time by which 10%/50%/90% of the flux-weighted
                # mass has arrived) and the fitted tail slope. t50 is
                # the "bulk" arrival time; the tail slope describes
                # late-time behavior; these are two DISTINCT physical
                # aspects of transport, and both are separate from the
                # tail-extent (grid tmax) printed above, which is a
                # plotting-grid construction, not a physical time.
                tk, mk = t_grid[keep], med[keep]

                def crossing_time(ccdf_threshold):
                    below = np.where(mk <= ccdf_threshold)[0]
                    if not len(below):
                        return None
                    i = below[0]
                    if i == 0:
                        return tk[0]
                    t0, t1 = tk[i - 1], tk[i]
                    y0, y1 = mk[i - 1], mk[i]
                    frac = ((ccdf_threshold - y0) / (y1 - y0)
                           if y1 != y0 else 0)
                    return 10 ** (np.log10(t0)
                                 + frac * (np.log10(t1) - np.log10(t0)))

                t10 = crossing_time(0.9)   # time by which 10% has arrived
                t50 = crossing_time(0.5)   # median arrival time
                t90 = crossing_time(0.1)   # time by which 90% has arrived

                summary_rows.append((sector, d, len(runs), t10, t50, t90))

                if args.annotate_slope:

                    fit = fit_tail_slope(t_grid[keep], med[keep])
                    if fit is not None:
                        slope, t_fit, y_fit = fit
                        summary_rows[-1] = summary_rows[-1] + (slope,)
                        ax.loglog(t_fit, y_fit, color=color, lw=1.6,
                                 ls="--", zorder=6, alpha=0.9)
                        # vertical offset per sector avoids label
                        # collisions when two sectors' tails end close
                        # together (points instead of data units, so
                        # the offset is independent of the y-axis scale)
                        sector_idx = sectors.index(sector)
                        y_offset = -14 * sector_idx
                        ax.annotate(f"{slope:.2f}",
                                   xy=(t_fit[-1], y_fit[-1]),
                                   xytext=(6, y_offset),
                                   textcoords="offset points",
                                   color=color, fontsize=11,
                                   fontweight="bold", va="center")
            else:
                # single run: just show it prominently, no band
                t, w = runs[0]
                rt, rccdf = ccdf_step(t, w)
                mask = rccdf > 0
                ax.loglog(rt[mask], rccdf[mask], color=color, lw=2.6,
                         zorder=5, label=f"{sector} (n=1)")
                ax_cdf.semilogx(rt, 1.0 - rccdf, color=color, lw=2.6, zorder=5)

        ax_cdf.set_title(f"Flow {DIR_LABEL[d]}")
        ax.set_xlabel(r"travel time $t$ [years]")
        ax.grid(True, which="both", alpha=0.18)
        ax_cdf.grid(True, which="both", alpha=0.18)
        ax.set_xlim(global_tmin, global_tmax)
        ax_cdf.set_xlim(global_tmin, global_tmax)
        ax_cdf.set_ylim(0, 1)

    axes[0][0].set_ylabel("CDF")
    axes[1][0].set_ylabel(r"CCDF, $P(T > t)$")
    axes[0][0].legend(loc="upper left", fontsize=10)

    print("\n=== Summary table (t10/t50/t90 arrival times + tail slope) ===")
    print(f"{'sector':<8} {'dir':<4} {'n_runs':<7} "
         f"{'t10 [yr]':<12} {'t50 [yr]':<12} {'t90 [yr]':<12} "
         f"{'tail_slope':<10}")

    def fmt(x):
        return f"{x:.3e}" if x is not None else "n/a"

    for row in summary_rows:
        if len(row) == 7:
            sector, d, n, t10, t50, t90, slope = row
            slope_str = f"{slope:.2f}"
        else:
            sector, d, n, t10, t50, t90 = row
            slope_str = "n/a"
        print(f"{sector:<8} {d:<4} {n:<7} {fmt(t10):<12} {fmt(t50):<12} "
             f"{fmt(t90):<12} {slope_str:<10}")

    fig.tight_layout()
    os.makedirs(args.figdir, exist_ok=True)
    save(fig, os.path.join(args.figdir, "fig5_btc_cdf_ccdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
