#!/usr/bin/env python3
"""
fig6_final.py
----------------
Figura 6 definitiva sui dati FIXED: traiettoria della particella con
chi piu' vicina al 99esimo percentile statistico, per ciascun dominio
(low/medium/high), direzione Y.

Legge direttamente i part_N.inp (AVS UCD, gia' raccolti in
fig6_supplementary_raw_data/) -- stessa logica di lettura gia'
validata (parser AVS UCD a blocchi separati nodi/celle/dati).

Layout: 3 righe (low/medium/high) x 2 colonne (3D + chi vs y) --
compattato da 3 colonne a 2 come suggerito nella review (il pannello
"pathline distance vs y" era quasi un passaggio algebrico di chi,
poca informazione aggiuntiva).

Uso:
    python fig6_final.py --out figures/fig6_final
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

SUPP_ROOT = "/hpc/home/filippo.porta/figures/combined/fig6_supplementary_raw_data"

DOMAIN_CONFIG = [
    {"label": "low",    "particle": 406, "chi": 14.5},
    {"label": "medium", "particle": 243, "chi": 14.20},
    {"label": "high",   "particle": 726, "chi": 10.99},
]

DIRECTION = "y"
AXIS_IDX = {"x": 0, "y": 1, "z": 2}
SECONDS_PER_YEAR = 3.15576e7


# ============================================================
# LETTURA part_N.inp (AVS UCD, formato reale, blocchi separati)
# ============================================================
def read_particle_inp(path):
    with open(path) as f:
        lines = f.readlines()

    i = 0
    while lines[i].lstrip().startswith("#"):
        i += 1
    header = lines[i].split()
    n_nodes, n_cells = int(header[0]), int(header[1])
    i += 1

    X = np.empty(n_nodes); Y = np.empty(n_nodes); Z = np.empty(n_nodes)
    for k in range(n_nodes):
        p = lines[i + k].split()
        X[k], Y[k], Z[k] = float(p[1]), float(p[2]), float(p[3])
    i += n_nodes + n_cells

    comp_line = lines[i].split()
    ncomp = int(comp_line[0])
    i += 1
    names = [lines[i + c].split(",")[0].strip() for c in range(ncomp)]
    i += ncomp

    data = np.empty((n_nodes, ncomp))
    for k in range(n_nodes):
        p = lines[i + k].split()
        data[k, :] = [float(v) for v in p[1:1 + ncomp]]

    col = {nm: data[:, c] for c, nm in enumerate(names)}
    velocity = col.get("velocity", np.zeros(n_nodes))

    return X, Y, Z, velocity


def cumulative_length(X, Y, Z):
    d = np.sqrt(np.diff(X)**2 + np.diff(Y)**2 + np.diff(Z)**2)
    return np.concatenate(([0.0], np.cumsum(d)))


def local_tortuosity_profile(coord, L, X, Y, Z, spacing=5.0, domain_half=25.0):
    """Tortuosita' locale tra control plane consecutivi (first
    crossing), stessa logica Lagrangiana usata ovunque nella
    pipeline."""
    planes = np.arange(-domain_half, domain_half + spacing, spacing)
    lam = {}
    for i in range(len(coord) - 1):
        c1, c2 = coord[i], coord[i+1]
        if c1 == c2:
            continue
        lo, hi = (c1, c2) if c1 < c2 else (c2, c1)
        for p in planes:
            pf = float(p)
            if pf in lam or not (lo <= pf <= hi):
                continue
            alpha = (pf - c1) / (c2 - c1)
            lam[pf] = L[i] + alpha * (L[i+1] - L[i])

    centers, chis = [], []
    for i in range(len(planes) - 1):
        p1, p2 = float(planes[i]), float(planes[i+1])
        if p1 not in lam or p2 not in lam:
            continue
        dx = abs(p2 - p1)
        if dx <= 0:
            continue
        centers.append(0.5 * (p1 + p2))
        chis.append(abs(lam[p2] - lam[p1]) / dx)
    return np.array(centers), np.array(chis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fig6_final")
    ap.add_argument("--domain-size", type=float, default=25.0, dest="ds")
    args = ap.parse_args()
    DS = args.ds

    domain_data = []
    g_dist_max = -np.inf
    g_logv_min, g_logv_max = np.inf, -np.inf
    g_tort_max = -np.inf

    for cfg in DOMAIN_CONFIG:
        label, pid = cfg["label"], cfg["particle"]
        path = os.path.join(SUPP_ROOT, label, "trajectories", f"part_{pid}.inp")
        X, Y, Z, velocity_raw = read_particle_inp(path)
        L = cumulative_length(X, Y, Z)
        coord = (X, Y, Z)[AXIS_IDX[DIRECTION]]
        velocity_ms = velocity_raw / SECONDS_PER_YEAR

        t_centers, t_vals = local_tortuosity_profile(coord, L, X, Y, Z)
        print(f"{label} (particle {pid}): {len(X)} punti, "
             f"chi max nel profilo = {t_vals.max() if len(t_vals) else float('nan'):.2f}")

        domain_data.append({
            "label": label, "pid": pid, "L": L, "X": X, "Y": Y, "Z": Z,
            "coord": coord, "velocity": velocity_ms,
            "t_centers": t_centers, "t_vals": t_vals,
        })

        g_dist_max = max(g_dist_max, L.max())
        lv = np.log10(np.where((velocity_ms > 0) & np.isfinite(velocity_ms),
                               velocity_ms, np.nan))
        g_logv_min = min(g_logv_min, np.nanmin(lv))
        g_logv_max = max(g_logv_max, np.nanmax(lv))
        if len(t_vals):
            g_tort_max = max(g_tort_max, t_vals.max())

    if not np.isfinite(g_tort_max) or g_tort_max <= 0:
        g_tort_max = 1.0

    cmap = plt.cm.viridis
    norm = plt.Normalize(g_logv_min, g_logv_max)

    fig = plt.figure(figsize=(15.5, 4.6 * len(DOMAIN_CONFIG)))

    for row, data in enumerate(domain_data):
        L, X, Y, Z = data["L"], data["X"], data["Y"], data["Z"]
        coord = data["coord"]

        ax3d = fig.add_subplot(len(DOMAIN_CONFIG), 3, row*3 + 1, projection="3d")
        logv = np.log10(np.where(data["velocity"] > 0, data["velocity"], np.nan))
        logv = np.nan_to_num(logv, nan=g_logv_min)

        pts = np.column_stack([X, Y, Z])
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        seg_colors = cmap(norm(logv[:-1]))
        lc = Line3DCollection(segs, colors=seg_colors, linewidths=1.6)
        ax3d.add_collection3d(lc)

        ax3d.scatter(X[0], Y[0], Z[0], marker="o", s=75, color="lime",
                    edgecolors="black", zorder=200)
        ax3d.scatter(X[-1], Y[-1], Z[-1], marker="s", s=75, color="red",
                    edgecolors="black", zorder=200)

        ax3d.set_xlabel("x [m]", fontsize=8); ax3d.set_ylabel("y [m]", fontsize=8)
        ax3d.set_zlabel("z [m]", fontsize=8)
        ax3d.set_title(f"{data['label']}-intensity  (particle {data['pid']})",
                       fontsize=11, fontweight="bold")
        ax3d.set_xlim(-DS, DS); ax3d.set_ylim(-DS, DS); ax3d.set_zlim(-DS, DS)
        ax3d.set_box_aspect([1, 1, 1])
        ax3d.view_init(elev=18, azim=-60)
        tk = [-DS, -DS/2, 0, DS/2, DS]
        ax3d.set_xticks(tk); ax3d.set_yticks(tk); ax3d.set_zticks(tk)
        ax3d.tick_params(labelsize=6, pad=1)

        # --- pathline distance vs y ---
        ax2 = fig.add_subplot(len(DOMAIN_CONFIG), 3, row*3 + 2)
        ax2.plot(coord, L, color="k", lw=1.4)
        ax2.set_xlabel(f"{DIRECTION} [m]", fontsize=9)
        ax2.set_ylabel(r"Pathline distance $\lambda$ [m]", fontsize=9)
        ax2.grid(alpha=0.3); ax2.tick_params(labelsize=7)
        ax2.set_xlim(-DS, DS); ax2.set_ylim(0, g_dist_max)

        # --- local tortuosity vs y ---
        ax3 = fig.add_subplot(len(DOMAIN_CONFIG), 3, row*3 + 3)
        ax3.plot(data["t_centers"], data["t_vals"], color="k", lw=1.0,
                alpha=0.7, zorder=5)
        ax3.scatter(data["t_centers"], data["t_vals"], s=32, color="k", zorder=10)
        if len(data["t_vals"]):
            imax = int(np.argmax(data["t_vals"]))
            ax3.scatter([data["t_centers"][imax]], [data["t_vals"][imax]],
                       marker="*", s=320, color="gold", edgecolors="black",
                       linewidths=0.8, zorder=20)
        ax3.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.6)
        ax3.set_xlabel(f"{DIRECTION} [m]", fontsize=9)
        ax3.set_ylabel(r"Local tortuosity $\chi_{i,j}$", fontsize=9)
        ax3.grid(alpha=0.3); ax3.tick_params(labelsize=7)
        ax3.set_xlim(-DS, DS); ax3.set_ylim(0, g_tort_max * 1.08)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.get_axes(), shrink=0.5, pad=0.02,
                        location="right", aspect=40)
    cbar.set_label(r"$\log_{10}(|v_p|\ \mathrm{[m/s]})$", fontsize=9)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="lime",
               markeredgecolor="k", markersize=9, label="Inlet"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="red",
               markeredgecolor="k", markersize=9, label="Outlet"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
               markeredgecolor="k", markersize=15, label=r"max $\chi_{i,j}$"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3,
              frameon=True, fontsize=10, bbox_to_anchor=(0.5, 0.985))

    fig.suptitle("Representative local Lagrangian tortuosity trajectories "
                f"(flow Y, 99th-percentile particles, FIXED backbone)",
                fontsize=13, fontweight="bold", y=1.03)

    # spazio esplicito per suptitle/legend, evita sovrapposizione
    # con le etichette y dei pannelli sottostanti
    fig.subplots_adjust(hspace=0.45, wspace=0.35, top=0.93)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(args.out + ".pdf", bbox_inches="tight")
    print(f"\nSalvato:\n  {args.out}.png\n  {args.out}.pdf")


if __name__ == "__main__":
    main()
