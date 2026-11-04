#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_fig8_grid.py
=================

Una sola figura 3x3 dai file gia' prodotti in vt_out/.

  righe    = domini (low, medium, high)
  colonne  = direzioni di flusso (X, Y, Z)
  pannello = densita' congiunta di (chi, u), con la legge di potenza fittata
             e i valori di b e rho annotati

Tre differenze rispetto alle figure attuali, che sono quelle che rendevano i
pannelli non confrontabili:

  1. ASSI CONDIVISI. Tutti i pannelli hanno gli stessi limiti, calcolati
     sull'insieme dei dati. Cosi' l'estensione in chi lungo Y si vede come
     differenza, non come diverso zoom.
  2. SCALA COLORE CONDIVISA. Gli istogrammi sono normalizzati a densita' di
     probabilita' (conteggi / totale / area del bin), quindi pannelli con
     numerosita' diversa sono comparabili, e c'e' una sola colorbar.
  3. NUMERI NEI PANNELLI. b e rho come media +/- sd sulle realizzazioni,
     presi da vt_per_run.csv: il risultato si legge nella figura.

USO
---
  python3 make_fig8_grid.py                  # figura principale, u vs chi
  python3 make_fig8_grid.py --which v        # versione supplementare, v vs chi
  python3 make_fig8_grid.py --outdir vt_out --min-n 50000

La versione "v" serve a mostrare una volta sola il vincolo v = u/chi: il bordo
superiore netto dei pannelli e' proprio quella retta.
"""

import os
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Font incorporati come TrueType e non come subset Type 3: quasi tutti gli
# editori lo richiedono, e con Type 3 il testo del PDF non resta selezionabile.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

DOMAINS = ["low", "medium", "high"]
DIRECTIONS = ["X", "Y", "Z"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
LINE = "#eb6834"          # slot 2 della palette validata, su fondo scuro


def load_per_run(path, min_n, verbose=True):
    """
    Media +/- sd sulle realizzazioni, da vt_per_run.csv.

    Due accortezze che non sono opzionali:

    1. INSIEME COMUNE. Il filtro su min_n agisce per direzione e puo' quindi
       rompere l'appaiamento anche su un file generato con --paired: una
       realizzazione scartata lungo Y resta lungo X. Qui si interseca a valle,
       cosi' tutti e nove i pannelli riportano le STESSE reti.

    2. RETTA COERENTE CON b. La retta disegnata e' la media delle regressioni
       per realizzazione: per ogni run a_r = log10(geo_u) - b_r*log10(geo_chi),
       poi si mediano a_r e b_r. Ancorare invece alla media della densita'
       pooled mescolerebbe due stimatori diversi (la pendenza viene dai fit per
       run, l'ancoraggio da tutti i segmenti insieme).
    """
    if not os.path.isfile(path):
        return {}
    rows = list(csv.DictReader(open(path), delimiter=";"))
    out = {}
    for dom in DOMAINS:
        keep = {D: set(r["run"] for r in rows if r["domain"] == dom
                       and r["direction"] == D and float(r["n"]) >= min_n)
                for D in DIRECTIONS}
        common = set.intersection(*keep.values()) if keep else set()
        if verbose:
            sizes = "/".join(str(len(keep[D])) for D in DIRECTIONS)
            flag = "" if all(len(keep[D]) == len(common) for D in DIRECTIONS) \
                else "   <- filtro rompeva l'appaiamento, intersecato"
            print(f"[{dom}] realizzazioni per direzione {sizes} "
                  f"-> insieme comune {len(common)}{flag}")
        if len(common) < 2:
            continue
        for D in DIRECTIONS:
            sel = [r for r in rows if r["domain"] == dom
                   and r["direction"] == D and r["run"] in common]
            b = np.array([float(r["slope_u_vs_chi"]) for r in sel])
            sp = np.array([float(r["spearman_logu_logchi"]) for r in sel])
            gc = np.array([float(r["geo_chi"]) for r in sel])
            gu = np.array([float(r["geo_u"]) for r in sel])
            aa = np.log10(gu) - b * np.log10(gc)      # intercetta per run
            out[(dom, D)] = dict(b=b.mean(), b_sd=b.std(ddof=1),
                                 a=aa.mean(),
                                 rho=sp.mean(), rho_sd=sp.std(ddof=1),
                                 n=len(sel), runs=sorted(common, key=int))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="vt_out")
    ap.add_argument("--which", choices=["u", "v"], default="u")
    ap.add_argument("--min-n", type=float, default=50000)
    ap.add_argument("--clip-pct", type=float, default=99.5,
                    help="percentile delle marginali che fissa i limiti degli "
                         "assi. Abbassalo (99.5) per stringere ancora, "
                         "alzalo (99.99) per mostrare piu' coda.")
    ap.add_argument("--full-range", action="store_true",
                    help="non restringere gli assi alla regione coperta da "
                         "tutte le celle. Serve solo per diagnostica: i "
                         "pannelli mostrano il bordo del proprio istogramma "
                         "come se fosse un dato.")
    ap.add_argument("--vmin-pct", type=float, default=60.0,
                    help="percentile dei bin non vuoti usato come vmin della "
                         "scala colore. Piu' alto = figura piu' pulita ma "
                         "code meno visibili. Prova 2, 15, 25, 40.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    key = "hist_uchi" if a.which == "u" else "hist_vchi"
    rkey = "logu_range" if a.which == "u" else "logv_range"
    sym = "u" if a.which == "u" else "v"

    stats = load_per_run(os.path.join(a.outdir, "vt_per_run.csv"), a.min_n)

    # --- carica tutto e determina i limiti comuni ----------------------
    data = {}
    for dom in DOMAINS:
        for D in DIRECTIONS:
            fn = os.path.join(a.outdir, f"vt_hist_{dom}_{D}.npz")
            if not os.path.isfile(fn):
                continue
            z = np.load(fn)
            H = z[key].astype(float)
            if H.sum() == 0:
                continue
            cr, yr = z["logchi_range"], z[rkey]
            ny, nx = H.shape[1], H.shape[0]
            dxb = (cr[1] - cr[0]) / nx
            dyb = (yr[1] - yr[0]) / ny
            data[(dom, D)] = dict(pdf=(H / H.sum() / (dxb * dyb)).T,
                                  cr=cr, yr=yr)
    if not data:
        raise SystemExit(f"nessun vt_hist_*.npz utilizzabile in {a.outdir}/")

    # limiti ROBUSTI: percentili delle marginali, non l'ultimo bin popolato.
    # Altrimenti un singolo bin isolato in un pannello allarga gli assi di tutti.
    def _q(weights, lo_edge, hi_edge, q):
        c = np.cumsum(weights)
        if c[-1] <= 0:
            return lo_edge
        c = c / c[-1]
        centres = lo_edge + (np.arange(weights.size) + 0.5) * \
            (hi_edge - lo_edge) / weights.size
        return float(np.interp(q, c, centres))

    xhi, ylo, yhi, vmaxs = [], [], [], []
    p = a.clip_pct / 100.0
    for d in data.values():
        mx = d["pdf"].sum(axis=0)          # marginale in chi
        my = d["pdf"].sum(axis=1)          # marginale in u (o v)
        xhi.append(_q(mx, d["cr"][0], d["cr"][1], p))
        ylo.append(_q(my, d["yr"][0], d["yr"][1], 1 - p))
        yhi.append(_q(my, d["yr"][0], d["yr"][1], p))
        vmaxs.append(d["pdf"].max())
    # In chi si usa tutto il range comune, non un percentile. La coda lungo Y
    # e' lunga e sottile, e qualunque taglio percentuale la fa arrivare a
    # filo del bordo: sembra che la nuvola prosegua oltre. Mostrando l'intero
    # range il riquadro coincide con il binning, quindi in chi non c'e' nulla
    # di tagliato, e lo spazio vuoto a destra nei pannelli X e Z non e'
    # sprecato: e' la differenza fra le direzioni.
    XLIM = (0.0, max(d["cr"][1] for d in data.values()))
    span = max(yhi) - min(ylo)
    YLIM = (min(ylo) - 0.05 * span, max(yhi) + 0.05 * span)

    # FINESTRA COMUNE A TUTTE LE CELLE.
    # I range degli istogrammi sono stati scelti cella per cella dai percentili
    # dei dati, quindi sono diversi fra loro (es. medium/Z parte da log10 u =
    # 6.25, high/X da 4.25). Con assi condivisi piu' larghi del range di una
    # cella, il bordo dell'istogramma di quella cella cade DENTRO il pannello e
    # si vede come un taglio orizzontale netto: sembra una soglia fisica del
    # campo di velocita' e non lo e'. Qui gli assi vengono ristretti alla
    # regione coperta da tutte e nove, cosi' nessun pannello mostra il proprio
    # bordo e i nove sono davvero confrontabili.
    if not a.full_range:
        cx_lo = max(d["cr"][0] for d in data.values())
        cx_hi = min(d["cr"][1] for d in data.values())
        cy_lo = max(d["yr"][0] for d in data.values())
        cy_hi = min(d["yr"][1] for d in data.values())
        XLIM = (max(XLIM[0], cx_lo), min(XLIM[1], cx_hi))
        YLIM = (max(YLIM[0], cy_lo), min(YLIM[1], cy_hi))

    # quanta massa resta dentro il riquadro disegnato, cella per cella:
    # va dichiarato in didascalia, non nascosto
    print("segmenti dentro il riquadro disegnato:")
    worst = 1.0
    for dom in DOMAINS:
        cells = []
        for D in DIRECTIONS:
            d = data.get((dom, D))
            if d is None:
                cells.append(f"{D}: n/a"); continue
            ny, nx = d["pdf"].shape
            cx = d["cr"][0] + (np.arange(nx) + 0.5) * (d["cr"][1] - d["cr"][0]) / nx
            cy = d["yr"][0] + (np.arange(ny) + 0.5) * (d["yr"][1] - d["yr"][0]) / ny
            sub = d["pdf"][np.ix_((cy >= YLIM[0]) & (cy <= YLIM[1]),
                                  (cx >= XLIM[0]) & (cx <= XLIM[1]))]
            f = sub.sum() / d["pdf"].sum()
            worst = min(worst, f)
            cells.append(f"{D}: {100*f:.2f}%")
        print("  " + dom.ljust(8) + "   ".join(cells))
    print(f"  minimo su tutti i pannelli: {100*worst:.2f}%")
    # vmin: taglia la coda di bin quasi vuoti, altrimenti domina il rumore
    allpos = np.concatenate([d["pdf"][d["pdf"] > 0].ravel()
                             for d in data.values()])
    NORM = LogNorm(vmin=np.percentile(allpos, a.vmin_pct), vmax=max(vmaxs))
    print(f"assi: x 0-{XLIM[1]:.2f}  y {YLIM[0]:.2f}-{YLIM[1]:.2f} "
          f"(percentile {a.clip_pct})")
    print(f"scala colore: vmin={NORM.vmin:.3g} (percentile {a.vmin_pct}) "
          f"vmax={NORM.vmax:.3g}  -> {np.log10(NORM.vmax/NORM.vmin):.1f} decadi")

    # --- disegna -------------------------------------------------------
    fig, axes = plt.subplots(3, 3, figsize=(10.4, 9.2),
                             sharex=True, sharey=True)
    im = None
    for i, dom in enumerate(DOMAINS):
        for j, D in enumerate(DIRECTIONS):
            ax = axes[i][j]
            ax.set_facecolor("#f4f3f0")
            d = data.get((dom, D))
            if d is None:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                        transform=ax.transAxes, color=INK2)
                continue
            M = np.ma.masked_where(d["pdf"] <= 0, d["pdf"])
            im = ax.imshow(M, origin="lower", aspect="auto", norm=NORM,
                           extent=[d["cr"][0], d["cr"][1],
                                   d["yr"][0], d["yr"][1]],
                           cmap="magma", interpolation="nearest")
            st = stats.get((dom, D))
            if st:
                # supporto in chi del pannello: taglia la retta al
                # percentile 99.9 della marginale, niente estrapolazione
                colm = d["pdf"].sum(axis=0)
                cxm = d["cr"][0] + (np.arange(colm.size) + 0.5) * \
                    (d["cr"][1] - d["cr"][0]) / colm.size
                cdf = np.cumsum(colm) / colm.sum()
                xmax = float(cxm[np.searchsorted(cdf, 0.999)]) \
                    if cdf[-1] > 0 else XLIM[1]
                x = np.linspace(XLIM[0], min(xmax, XLIM[1]), 40)
                # media delle regressioni per realizzazione: stessa fonte
                # per pendenza e intercetta. Per il pannello v vale
                # log v = log u - log chi, quindi la pendenza cala di 1.
                slope = st["b"] if a.which == "u" else st["b"] - 1.0
                ax.plot(x, st["a"] + slope * x, color=LINE, lw=2.0,
                        solid_capstyle="round", zorder=4)
                lab = (rf"$b={st['b']:.2f}\pm{st['b_sd']:.2f}$" "\n"
                       rf"$\rho={st['rho']:.2f}\pm{st['rho_sd']:.2f}$")
                # in basso a destra: e' l'unico angolo sparso in tutti e nove
                # i pannelli. In alto a destra la scritta finiva sopra la
                # retta e, nella riga low, dentro la nuvola.
                ax.text(0.965, 0.075, lab, transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=9, color=INK,
                        linespacing=1.5, zorder=6,
                        bbox=dict(boxstyle="round,pad=0.38", facecolor="white",
                                  edgecolor=GRID, linewidth=0.7, alpha=0.93))
            if i == 0:
                ax.set_title(f"flow direction {D}", color=INK, fontsize=11,
                             pad=8)
            if j == 2:
                ax.text(1.075, 0.5, f"{dom}\nintensity", transform=ax.transAxes,
                        rotation=270, ha="left", va="center", color=INK,
                        fontsize=12, fontweight="semibold", linespacing=1.4)
            for s in ("top", "right", "left", "bottom"):
                ax.spines[s].set_color(GRID)
            ax.tick_params(colors=INK2, length=3, labelsize=9)

    axes[0][0].set_xlim(*XLIM)
    axes[0][0].set_ylim(*YLIM)
    for j in range(3):
        axes[2][j].set_xlabel(r"$\log_{10}\chi$", color=INK, fontsize=11)
    for i in range(3):
        axes[i][0].set_ylabel(rf"$\log_{{10}}\,{sym}$   [m yr$^{{-1}}$]",
                              color=INK, fontsize=11)

    # la banda in basso deve contenere barra + tacche + etichetta:
    # con rect troppo stretto l'etichetta della colorbar finisce fuori figura
    fig.tight_layout(rect=[0, 0.105, 0.945, 1])
    cax = fig.add_axes([0.12, 0.062, 0.68, 0.015])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("joint probability density of segments", color=INK2,
                 fontsize=9)
    cb.ax.tick_params(colors=INK2, labelsize=8)
    cb.outline.set_edgecolor(GRID)

    # PDF per la sottomissione (vettoriale per assi e testo, la densita' resta
    # una immagine raster: e' un istogramma, non ha senso vettorializzarla) e
    # PNG per guardarla al volo. Con --out si impone un singolo formato.
    if a.out:
        outs = [a.out]
    else:
        base = os.path.join(a.outdir, f"fig8_grid_{a.which}chi")
        outs = [base + ".pdf", base + ".png"]
    for o in outs:
        fig.savefig(o, dpi=300, facecolor="white",
                    bbox_inches=None)          # niente crop: il layout e' gia' fissato
        print("->", o)


if __name__ == "__main__":
    main()
