#!/usr/bin/env python3


import os
import csv
import glob
import math
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DOMAINS = {
    "low": {
        "metrics": "/hpc/home/filippo.porta/metrics/fixed/low",
        "runs":    "/hpc/scratch/filippo.porta/runs/runs_fixed",
    },
    "medium": {
        "metrics": "/hpc/home/filippo.porta/metrics/fixed/medium",
        "runs":    "/hpc/archive/G_NEXT/runs/runs_fixed",
    },
    "high": {
        "metrics": "/hpc/home/filippo.porta/metrics/fixed/high",
        "runs":    "/hpc/scratch/filippo.porta/runs_fixed_high",
    },
}
# FIX 21/09: path allineati a fig3_topology_population_fixed2.py e
# run_fig5_btc_final.py (entrambi verificati corretti). I path
# originali (senza "fixed"/"runs_fixed") leggevano dati con la
# numerazione fratture pre-fix (vedi stato-lavori-manoscritto-FP.md
# §1: output_backbone/dfnGen_output = rete potata dell'ultima
# direzione girata, rinumerata da 1, incompatibile con gli ID di
# backbone_<DIR>.dat, presi dalla rete madre). NB: "high" non e'
# semplicemente <vecchio path>/runs_fixed -- e' un albero diverso,
# copiato cosi' com'e' dagli altri due script per coerenza.
DOMAIN_ORDER = ["low", "medium", "high"]
DIRECTIONS = ["x", "y", "z"]
DIR_COLOR  = {"x": "tab:blue", "y": "tab:orange", "z": "tab:green"}
DIR_MARKER = {"x": "o", "y": "s", "z": "^"}
DIR_LABEL  = {"x": "Flow X", "y": "Flow Y", "z": "Flow Z"}


# ============================================================
# LETTURA raggi (header-safe, path risolto via symlink)
# ============================================================
def find_original_radii(run_path):
    link = os.path.join(run_path, "output_backbone",
                        "dfnGen_output", "connectivity.dat")
    if os.path.islink(link):
        real_dir = os.path.dirname(os.path.realpath(link))
        cand = os.path.join(real_dir, "radii_Final.dat")
        if os.path.exists(cand):
            return cand
    matches = glob.glob(os.path.join(run_path, "intensity*",
                                     "dfnGen_output", "radii_Final.dat"))
    return matches[0] if matches else None


def read_radii(path):
    """Header-safe: salta righe non numeriche iniziali, poi
    riga numerica k -> fracture k."""
    radii = {}
    fid = 0
    with open(path) as f:
        for line in f:
            s = line.split()
            if not s:
                continue
            try:
                r = float(s[0])
            except ValueError:
                if fid == 0:
                    continue
                raise ValueError(f"riga non numerica in mezzo ai dati: {line!r}")
            fid += 1
            radii[fid] = r
    return radii


# ============================================================
# path_stats.csv (n_paths AGGIORNATO factor=1.5, non metrics.csv)
# ============================================================
def load_path_stats(run_metrics_dir):
    """Ritorna {direction: (n_paths, capped)} da path_stats.csv."""
    path = os.path.join(run_metrics_dir, "path_stats.csv")
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            d = row["direction"].strip().lower()
            out[d] = (int(row["n_paths"]), int(row.get("capped", 0)))
    return out


def load_centrality(path):
    Ci = {d: {} for d in DIRECTIONS}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            d = row["direction"].strip().lower()
            if d in Ci:
                Ci[d][int(row["fracture"])] = float(row["C_i"])
    return Ci


try:
    from scipy.stats import spearmanr
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


def spearman(xs, ys):
    """Spearman rho. Con scipy: gestisce correttamente i pareggi
    (essenziale qui, dato l'alto numero di C_i=0 identici). Senza
    scipy: fallback a correlazione di Pearson sui ranghi, che non
    corregge i pareggi -- leggermente distorto ma accettabile."""
    n = len(xs)
    if n < 3:
        return float("nan")
    if HAVE_SCIPY:
        rho, _ = spearmanr(xs, ys)
        return float(rho)
    rx = np.argsort(np.argsort(xs)).astype(float)
    ry = np.argsort(np.argsort(ys)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    denom = math.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom else float("nan")


# ============================================================
# RACCOLTA DATI PER DOMINIO
# ============================================================
def collect_domain(domain, min_n_paths):
    cfg = DOMAINS[domain]
    metrics_dir, runs_dir = cfg["metrics"], cfg["runs"]
    pts = {d: ([], []) for d in DIRECTIONS}

    runs = sorted(
        [x.replace("run_", "") for x in os.listdir(metrics_dir)
         if x.startswith("run_")], key=lambda x: int(x))

    print(f"\n{'='*60}\n{domain.upper()}  ({len(runs)} run)\n{'='*60}")

    for run in runs:
        run_metrics = os.path.join(metrics_dir, f"run_{run}")
        cent_file = os.path.join(run_metrics, "centrality.csv")
        run_path = os.path.join(runs_dir, run)

        if not os.path.exists(cent_file):
            continue
        radii_file = find_original_radii(run_path)
        if radii_file is None:
            continue

        try:
            radii = read_radii(radii_file)
            Ci = load_centrality(cent_file)
        except Exception as e:
            print(f"  run {run}: errore ({e}), salto")
            continue

        pstats = load_path_stats(run_metrics)
        refix_backup = os.path.join(run_metrics, "centrality_f12_backup.csv")
        was_refixed = os.path.exists(refix_backup)

        for d in DIRECTIONS:
            n_paths, capped = pstats.get(d, (0, 0))

            # path_stats.csv NON viene mai aggiornato da refix_capped_v2.py:
            # se questa run e' stata rifissata (backup presente), il vecchio
            # flag capped=1 di path_stats.csv e' STANTIO (viene dal primo
            # tentativo fallito a cap=20000) e va ignorato -- il refix con
            # cap piu' alto e' gia' stato verificato non-capped altrimenti
            # non avrebbe sovrascritto centrality.csv (vedi refix_run()).
            if d == "y" and was_refixed:
                capped = 0
                n_paths = max(n_paths, min_n_paths)  # bypassa il filtro soglia

            if n_paths < min_n_paths:
                continue  # C_i degenere (troppo pochi path enumerati)
            if capped:
                print(f"  run {run} dir {d.upper()}: ancora CAPPED, salto")
                continue

            r_list, c_list = pts[d]
            for fid, ci in Ci[d].items():
                if fid in radii:
                    r_list.append(radii[fid])
                    c_list.append(ci)

    return {d: (np.array(v[0]), np.array(v[1])) for d, v in pts.items()}


# ============================================================
# PANNELLO PER DOMINIO
# ============================================================
def draw_panel(ax, ddata, title):
    all_r = np.concatenate([ddata[d][0] for d in DIRECTIONS if len(ddata[d][0])])
    all_c = np.concatenate([ddata[d][1] for d in DIRECTIONS if len(ddata[d][1])])
    rho = spearman(all_r, all_c)

    for d in DIRECTIONS:
        r, c = ddata[d]
        if len(r) == 0:
            continue
        ax.scatter(r, c, s=16, facecolors="none", edgecolors=DIR_COLOR[d],
                  marker=DIR_MARKER[d], alpha=0.30, linewidths=0.7,
                  label=DIR_LABEL[d], zorder=2)

    if len(all_r):
        edges = np.geomspace(all_r.min(), max(all_r.max(), all_r.min()*1.01), 13)
        for q, ls, lw, col, lbl in ((90, "-", 2.6, "black", "90th pct"),
                                    (75, "--", 2.0, "0.35", "75th pct"),
                                    (50, ":", 1.6, "0.55", "Median")):
            bc, bq = [], []
            for i in range(len(edges) - 1):
                m = (all_r >= edges[i]) & (all_r < edges[i+1])
                if m.sum() >= 10:   # almeno 10 punti per bin (era 5:
                                    # troppo pochi per il 90 percentile)
                    bc.append(math.sqrt(edges[i]*edges[i+1]))
                    bq.append(np.percentile(all_c[m], q))
            if bc:
                ax.plot(bc, bq, ls=ls, lw=lw, color=col, zorder=5,
                       label=lbl)

    for d in DIRECTIONS:
        r, c = ddata[d]
        if len(r) == 0:
            continue
        m = c >= 0.99
        if m.any():
            ax.scatter(r[m], np.full(m.sum(), 1.0), s=38,
                      facecolors=DIR_COLOR[d], edgecolors="k",
                      marker=DIR_MARKER[d], linewidths=0.6,
                      alpha=0.95, zorder=6)

    ax.set_xscale("log")
    ax.set_xlabel(r"radius / $r_0$")
    ax.grid(True, which="both", alpha=0.15)
    ax.set_title(title, fontweight="bold")
    ax.text(0.03, 0.97, f"$\\rho$={rho:.2f}\nn={len(all_r)}",
           transform=ax.transAxes, va="top", ha="left", fontsize=9,
           bbox=dict(boxstyle="round", facecolor="white",
                    edgecolor="0.6", alpha=0.9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    ap.add_argument("--min-n-paths", type=int, default=10, dest="min_n_paths")
    args = ap.parse_args()

    all_data = {dom: collect_domain(dom, args.min_n_paths)
               for dom in DOMAIN_ORDER}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, dom in zip(axes, DOMAIN_ORDER):
        letter = chr(97 + DOMAIN_ORDER.index(dom))
        draw_panel(ax, all_data[dom], f"({letter}) {dom}-intensity")
    axes[0].set_ylabel(r"simple-path centrality $C_i$")
    axes[-1].legend(loc="center right", fontsize=8, ncols=1)

    fig.suptitle("Fracture radius vs simple-path centrality "
                "(2-core backbone)", fontsize=13, fontweight="bold")
    fig.tight_layout()

    os.makedirs(args.out, exist_ok=True)
    out_png = os.path.join(args.out, "fig4_merged.png")
    out_pdf = os.path.join(args.out, "fig4_merged.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"\nSalvato:\n  {out_png}\n  {out_pdf}")


if __name__ == "__main__":
    main()
