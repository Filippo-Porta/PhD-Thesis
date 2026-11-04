#!/usr/bin/env python3
"""
fig3_topology_population.py
-------------------------------
Figura 3: metriche topologiche a livello di popolazione (tutte le
run, tutti i domini, tutte le direzioni).

FIX rispetto a compute_topo.py / topology_metrics_corrected.csv:
  degree_mean era gia' corretto (2-core), ma n_c e n_paths in quel
  file venivano da metrics.csv (vecchia pipeline: current-flow-
  threshold backbone, factor=1.2) -- rete e criterio DIVERSI dal
  2-core/factor=1.5 usato ovunque nel resto del paper.

  Qui tutte e tre le metriche sono ricalcolate sullo STESSO grafo
  2-core:
    degree_mean : media del grado (s/t esclusi)
    n_c         : node connectivity fresco (nx.node_connectivity)
    n_paths     : letto da path_stats.csv (factor=1.5, gia' 2-core,
                  gia' aggiornato dal refix)

Pannelli:
  (a) mean node degree     -- boxplot + jitter
  (b) node connectivity     -- boxplot + jitter
  (c) simple paths n_paths  -- ECDF log-x (il boxplot distorce una
      distribuzione che spazia ordini di grandezza)

Uso:
    python fig3_topology_population.py --out figures/fig3_topology
"""

import os
import csv
import argparse
import numpy as np
import networkx as nx
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
# NB: sui dati fixed non esiste il concetto di "refix" (centrality
# gia' corretta al primo giro con factor=1.5 e cap adeguato) -- il
# bypass del flag capped stantio in load_centrality_values() resta
# nel codice per compatibilita' ma non scattera' mai (nessun
# centrality_f12_backup.csv presente), comportamento comunque
# corretto: se capped=1 nel nuovo path_stats.csv e' un capped VERO.
DOMAIN_ORDER = ["low", "medium", "high"]
DOMAIN_COLOR = {"low": "#4C72B0", "medium": "#DD8452", "high": "#C44E52"}
DIRECTIONS = ["X", "Y", "Z"]

BOUNDARY = {
    "X": ("left.dat",   "right.dat"),
    "Y": ("front.dat",  "back.dat"),
    "Z": ("bottom.dat", "top.dat"),
}


# ============================================================
# GRAFO 2-CORE (identico al resto della pipeline)
# ============================================================
def read_ids(path):
    ids = set()
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            s = line.split()
            if s:
                ids.add(int(s[0]))
    return ids


def build_2core_graph(run_path, direction):
    dfn = os.path.join(run_path, "output_backbone", "dfnGen_output")
    connectivity = os.path.join(dfn, "connectivity.dat")
    src_name, snk_name = BOUNDARY[direction]
    source = os.path.join(dfn, src_name)
    target = os.path.join(dfn, snk_name)
    backbone_file = os.path.join(run_path, f"backbone_{direction}.dat")

    for p in (connectivity, source, target, backbone_file):
        if not os.path.exists(p):
            return None

    G = nx.Graph()
    with open(connectivity, encoding="utf-8-sig") as f:
        for i, line in enumerate(f):
            u = i + 1
            for v in (int(x) for x in line.split()):
                G.add_edge(u, v)

    G.add_node("s"); G.add_node("t")
    for n in read_ids(source):
        if n in G: G.add_edge("s", n)
    for n in read_ids(target):
        if n in G: G.add_edge("t", n)

    H = G.subgraph(read_ids(backbone_file) | {"s", "t"}).copy()

    changed = True
    while changed:
        changed = False
        remove = [n for n in H.nodes()
                  if n not in ("s", "t") and H.degree(n) < 2]
        if remove:
            H.remove_nodes_from(remove)
            changed = True
    return H


def degree_mean_2core(H2):
    """Grado medio SOLO fratture (s/t esclusi anche dagli archi,
    coerente con z_frac_only stabilito settimane fa)."""
    nodes = [n for n in H2.nodes() if isinstance(n, int)]
    if not nodes:
        return None
    H2_frac = H2.subgraph(nodes)
    degs = [H2_frac.degree(n) for n in nodes]
    return float(np.mean(degs))


def load_n_paths(metrics_dir, run, direction):
    """n_paths da path_stats.csv (factor=1.5, 2-core, gia' refixato)."""
    path = os.path.join(metrics_dir, f"run_{run}", "path_stats.csv")
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["direction"].strip().lower() == direction.lower():
                if int(row.get("capped", 0)):
                    return None  # ancora capped, scarta
                return int(row["n_paths"])
    return None


def load_centrality_values(metrics_dir, run, direction):
    """Tutti i C_i > 0 di quella run/direzione (stessa logica di
    fig4_merged.py: bypassa il flag capped stantio se il refix e'
    andato a buon fine, verificato dalla presenza del backup)."""
    run_metrics = os.path.join(metrics_dir, f"run_{run}")
    cent_file = os.path.join(run_metrics, "centrality.csv")
    if not os.path.exists(cent_file):
        return []

    path_stats = os.path.join(run_metrics, "path_stats.csv")
    was_refixed = os.path.exists(
        os.path.join(run_metrics, "centrality_f12_backup.csv"))

    if os.path.exists(path_stats):
        with open(path_stats, newline="") as f:
            for row in csv.DictReader(f):
                if row["direction"].strip().lower() != direction.lower():
                    continue
                capped = int(row.get("capped", 0))
                if capped and not (direction.lower() == "y" and was_refixed):
                    return []  # ancora capped e non rifissata, scarta

    vals = []
    with open(cent_file, newline="") as f:
        for row in csv.DictReader(f):
            if row["direction"].strip().lower() != direction.lower():
                continue
            ci = float(row["C_i"])
            if ci > 0:
                vals.append(ci)
    return vals


# ============================================================
# RACCOLTA DATI
# ============================================================
def collect_domain(domain):
    cfg = DOMAINS[domain]
    metrics_dir, runs_dir = cfg["metrics"], cfg["runs"]
    runs = sorted(
        [x.replace("run_", "") for x in os.listdir(metrics_dir)
         if x.startswith("run_")], key=lambda x: int(x))

    data = {d: {"degree": [], "n_c": [], "n_paths": [], "C_i": []}
            for d in DIRECTIONS}

    print(f"\n{'='*60}\n{domain.upper()}  ({len(runs)} run)\n{'='*60}")
    for run in runs:
        for direction in DIRECTIONS:
            H2 = build_2core_graph(os.path.join(runs_dir, run), direction)
            if H2 is None:
                continue

            deg = degree_mean_2core(H2)
            if deg is None:
                continue

            try:
                nc = nx.node_connectivity(H2, "s", "t")
            except Exception:
                nc = None

            npaths = load_n_paths(metrics_dir, run, direction)
            ci_vals = load_centrality_values(metrics_dir, run, direction)

            if deg is not None:
                data[direction]["degree"].append(deg)
            if nc is not None:
                data[direction]["n_c"].append(nc)
            if npaths is not None:
                data[direction]["n_paths"].append(npaths)
            data[direction]["C_i"].extend(ci_vals)

        print(f"  run {run}: OK")

    return data


# ============================================================
# PLOT HELPERS
# ============================================================
def boxplot_panel(ax, all_data, key, ylabel, title, logscale=False):
    positions, values, colors = [], [], []
    width = 0.22
    n_dom = len(DOMAIN_ORDER)

    for di, direction in enumerate(DIRECTIONS):
        for si, domain in enumerate(DOMAIN_ORDER):
            vals = all_data[domain][direction][key]
            if not vals:
                continue
            pos = di + (si - (n_dom - 1) / 2) * width
            positions.append(pos)
            values.append(vals)
            colors.append(DOMAIN_COLOR[domain])

    if not values:
        ax.set_title(title + " (no data)")
        return

    bp = ax.boxplot(values, positions=positions, widths=width * 0.85,
                    patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.35)
    for median, color in zip(bp["medians"], colors):
        median.set_color(color); median.set_linewidth(2)

    rng = np.random.default_rng(0)
    for pos, vals, color in zip(positions, values, colors):
        jitter = rng.uniform(-width*0.3, width*0.3, size=len(vals))
        ax.scatter(np.full(len(vals), pos) + jitter, vals,
                  s=10, color=color, alpha=0.5, zorder=3,
                  edgecolors="none")

    ax.set_xticks(range(len(DIRECTIONS)))
    ax.set_xticklabels([f"Flow {d}" for d in DIRECTIONS])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    if logscale:
        ax.set_yscale("log")
    ax.grid(alpha=0.25)


def boxplot_panel_dense(ax, all_data, key, ylabel, title, logscale=False):
    """Come boxplot_panel, ma per metriche con MOLTI punti per box
    (es. C_i: una frattura ciascuno, centinaia/migliaia per box) --
    niente jitter individuale (illeggibile a quella densita'),
    solo showfliers per vedere gli outlier."""
    positions, values, colors = [], [], []
    width = 0.22
    n_dom = len(DOMAIN_ORDER)

    for di, direction in enumerate(DIRECTIONS):
        for si, domain in enumerate(DOMAIN_ORDER):
            vals = all_data[domain][direction][key]
            if not vals:
                continue
            pos = di + (si - (n_dom - 1) / 2) * width
            positions.append(pos)
            values.append(vals)
            colors.append(DOMAIN_COLOR[domain])

    if not values:
        ax.set_title(title + " (no data)")
        return

    bp = ax.boxplot(values, positions=positions, widths=width * 0.85,
                    patch_artist=True, showfliers=True,
                    flierprops=dict(marker=".", markersize=2, alpha=0.3))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.35)
    for median, color in zip(bp["medians"], colors):
        median.set_color(color); median.set_linewidth(2)
    for flier, color in zip(bp["fliers"], colors):
        flier.set_markerfacecolor(color); flier.set_markeredgecolor(color)

    ax.set_xticks(range(len(DIRECTIONS)))
    ax.set_xticklabels([f"Flow {d}" for d in DIRECTIONS])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    if logscale:
        ax.set_yscale("log")
    ax.grid(alpha=0.25)


def ecdf_panel(ax, all_data, key, xlabel, title, logscale=True):
    for direction, ls in zip(DIRECTIONS, ["-", "--", ":"]):
        for domain in DOMAIN_ORDER:
            vals = np.sort(all_data[domain][direction][key])
            if len(vals) == 0:
                continue
            y = np.arange(1, len(vals) + 1) / len(vals)
            ax.plot(vals, y, ls=ls, color=DOMAIN_COLOR[domain], lw=1.6,
                   alpha=0.85)

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=DOMAIN_COLOR[d], lw=2, label=d)
              for d in DOMAIN_ORDER]
    handles += [Line2D([0], [0], color="k", lw=1.6, ls=ls,
                       label=f"Flow {d}")
               for d, ls in zip(DIRECTIONS, ["-", "--", ":"])]
    ax.legend(handles=handles, fontsize=7, loc="lower right", ncol=2)

    if logscale:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("ECDF")
    ax.set_title(title, fontweight="bold")
    ax.grid(alpha=0.25, which="both")


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fig3_topology_fixed")
    args = ap.parse_args()

    all_data = {dom: collect_domain(dom) for dom in DOMAIN_ORDER}

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10))

    boxplot_panel(axes[0][0], all_data, "degree", r"$\langle k \rangle$",
                 "(a) Mean node degree")
    boxplot_panel(axes[0][1], all_data, "n_c", r"$n_c$",
                 "(b) Node connectivity")
    axes[0][2].axis("off")  # vuoto, per bilanciare la griglia

    ecdf_panel(axes[1][0], all_data, "n_paths", r"$N_{paths}$",
              "(c) Simple paths (ECDF)", logscale=True)
    ecdf_panel(axes[1][1], all_data, "C_i", r"Path centrality $C_i$  ($C_i>0$)",
              "(d) Fracture centrality (ECDF)", logscale=False)
    boxplot_panel_dense(axes[1][2], all_data, "C_i",
                       r"Path centrality $C_i$  ($C_i>0$)",
                       "(e) Fracture centrality (boxplot)")

    from matplotlib.patches import Patch
    dom_handles = [Patch(facecolor=DOMAIN_COLOR[d], alpha=0.35, label=d)
                  for d in DOMAIN_ORDER]
    axes[0][0].legend(handles=dom_handles, loc="upper left", fontsize=8)

    fig.suptitle("Population-level topological metrics "
                "(2-core, FIXED backbone, all realizations)", fontsize=13, fontweight="bold")
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(args.out + ".pdf", bbox_inches="tight")
    print(f"\nSalvato:\n  {args.out}.png\n  {args.out}.pdf")


if __name__ == "__main__":
    main()
