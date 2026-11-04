#!/usr/bin/env python3

import argparse
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import apply_style, save, DIR_LABEL, SECTOR_COLOR, panel_label
import matplotlib.pyplot as plt

DOMAIN_RUNS_DIR = {
    "low":    "/hpc/scratch/filippo.porta/runs_10k/low",
    "medium": "/hpc/scratch/filippo.porta/runs_10k/medium",
    "high":   "/hpc/scratch/filippo.porta/runs_10k/high",
}

PHYSICAL_TIME_CUTOFF_YEARS = 1e3
REFERENCE_DOMAIN = "low"
REFERENCE_DIRECTION = "x"
CONFIDENCE_LEVEL = 0.95
Y_AXIS_MIN = 5e-10

# limiti manuali dell'asse X normalizzato (t / t_peak): se lasciati a
# None, vengono calcolati automaticamente dal minimo e dal massimo
# effettivi dei dati. Impostati qui ai valori scelti per allargare
# leggermente la vista rispetto al range stretto calcolato in automatico.
X_AXIS_MIN = 1e-7
X_AXIS_MAX = 1e10


def load_partime(path):
    times, weights = [], []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                w = float(parts[1])
                t = float(parts[2])
            except ValueError:
                continue
            if t > 0:
                times.append(t)
                weights.append(w)
    return np.array(times), np.array(weights)


def per_run_kde(t, w, n_grid=150, bw=0.15):
    mask = t <= PHYSICAL_TIME_CUTOFF_YEARS
    t, w = t[mask], w[mask]
    if len(t) < 5:
        return None

    logt = np.log10(t)
    grid_log = np.linspace(logt.min(), logt.max(), n_grid)
    ww = w / w.sum()
    dens = np.zeros_like(grid_log)
    for lt, wt in zip(logt, ww):
        dens += wt * np.exp(-0.5 * ((grid_log - lt) / bw) ** 2)
    dens /= (bw * np.sqrt(2 * np.pi))
    return 10 ** grid_log, dens


def gather_runs(domain, direction):
    base = DOMAIN_RUNS_DIR[domain]
    runs = sorted(
        d for d in os.listdir(base)
        if d.isdigit() and os.path.exists(os.path.join(base, d, "STATUS_OK")))
    curves = []
    for run in runs:
        partime = os.path.join(base, run, "transport", direction.upper(),
                               "partime.dat")
        if not os.path.exists(partime):
            continue
        t, w = load_partime(partime)
        if len(t) < 10:
            continue
        result = per_run_kde(t, w)
        if result is not None:
            curves.append(result)
    return curves


def ensemble_mean_ci(curves, n_grid=200, confidence=CONFIDENCE_LEVEL):
    all_min = min(c[0].min() for c in curves)
    all_max = max(c[0].max() for c in curves)
    grid = np.geomspace(all_min, all_max, n_grid)
    log_grid = np.log10(grid)

    mat = np.array([
        np.interp(log_grid, np.log10(tg), dens, left=0, right=0)
        for tg, dens in curves])

    n = mat.shape[0]
    mean = mat.mean(axis=0)
    sem = mat.std(axis=0, ddof=1) / np.sqrt(n)
    tval = stats.t.ppf(0.5 + confidence / 2, df=n - 1)
    ci_low = mean - tval * sem
    ci_high = mean + tval * sem
    ci_low = np.clip(ci_low, 0, None)
    return grid, mean, ci_low, ci_high


def plot_normalized(ax, domain, direction, curves, ensemble, ref_time, color):
    for tg, dens in curves:
        ax.loglog(tg / ref_time, dens, color=color, lw=0.6, alpha=0.2, zorder=2)

    grid, mean, ci_low, ci_high = ensemble
    ax.fill_between(grid / ref_time, ci_low, ci_high, color=color,
                    alpha=0.25, zorder=3, linewidth=0)
    ax.loglog(grid / ref_time, mean, color=color, lw=2.4, zorder=5,
             label=f"{domain} (n={len(curves)} runs)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figdir", default="figures")
    args = ap.parse_args()

    apply_style()
    os.makedirs(args.figdir, exist_ok=True)

    domains = ["low", "medium", "high"]
    directions = ["x", "y", "z"]

    all_curves, all_ensembles = {}, {}
    for domain in domains:
        for direction in directions:
            curves = gather_runs(domain, direction)
            if not curves:
                continue
            all_curves[(domain, direction)] = curves
            all_ensembles[(domain, direction)] = ensemble_mean_ci(curves)

    ref_grid, ref_mean, _, _ = all_ensembles[(REFERENCE_DOMAIN, REFERENCE_DIRECTION)]
    t_peak_ref = ref_grid[np.argmax(ref_mean)]
    print(f"Tempo di picco della MEDIA di riferimento "
         f"({REFERENCE_DOMAIN}/{REFERENCE_DIRECTION.upper()}) = "
         f"{t_peak_ref:.4e} anni\n")

    print(f"{'domain':<8}{'dir':<5}{'n_runs':<8}{'own_peak_time [yr]'}")
    for (domain, direction), ens in all_ensembles.items():
        grid, mean, _, _ = ens
        own_peak = grid[np.argmax(mean)]
        print(f"{domain:<8}{direction.upper():<5}"
             f"{len(all_curves[(domain, direction)]):<8}{own_peak:.4e}")

    xmin_fixed, xmax_fixed = np.inf, -np.inf
    for (domain, direction), curves in all_curves.items():
        for tg, dens in curves:
            xmin_fixed = min(xmin_fixed, (tg / t_peak_ref).min())
            xmax_fixed = max(xmax_fixed, (tg / t_peak_ref).max())

    if X_AXIS_MIN is not None:
        xmin_fixed = X_AXIS_MIN
    if X_AXIS_MAX is not None:
        xmax_fixed = X_AXIS_MAX

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), sharey=True)
    for ax, direction, letter in zip(axes, directions, "abc"):
        panel_label(ax, letter)
        for domain in domains:
            key = (domain, direction)
            if key not in all_curves:
                continue
            plot_normalized(ax, domain, direction, all_curves[key],
                           all_ensembles[key], t_peak_ref,
                           SECTOR_COLOR.get(domain, "0.3"))
        ax.set_title(f"Flow {DIR_LABEL[direction]}")
        ax.set_xlabel(r"$t / t_{peak}$ (fixed ref: low, X, ensemble mean)")
        ax.grid(True, which="both", alpha=0.18)
        ax.set_ylim(bottom=Y_AXIS_MIN)
        ax.set_xlim(xmin_fixed, xmax_fixed)

    axes[0].set_ylabel("probability density")
    axes[0].legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    save(fig, os.path.join(args.figdir, "fig7_pdf_10k_fixed_ref"))
    plt.close(fig)

    print(f"\nSalvata figura in {args.figdir}/fig7_pdf_10k_fixed_ref")


if __name__ == "__main__":
    main()
