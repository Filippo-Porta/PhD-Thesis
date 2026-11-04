#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
effective_permeability.py
=========================

Permeabilita' efficace direzionale calcolata DALLA PORTATA MISURATA, e
separazione degli effetti di permeabilita' da quelli di topologia.

PERCHE' DALLA PORTATA
---------------------
Ogni cartella transport/<DIR>/ contiene inputflux_m3s, cioe' la portata
volumetrica totale attraverso la faccia di ingresso. Da li'

    K = Q * mu * L / (A * dP)

con tutti i termini noti e verificati. Non passa da effective_perm(), quindi
non dipende da come quella funzione viene chiamata.

LA SEPARAZIONE CHE SERVE
------------------------
La portata si puo' scrivere Q = <u> * A*, dove A* = Q/<u> e' un'AREA DI FLUSSO
APPARENTE. Da qui il rapporto fra due direzioni si fattorizza in tre pezzi:

    dt_A / dt_B  =  (chi_A/chi_B)  *  (K_B/K_A)  *  (A*_A/A*_B)
                     tortuosita'      permeabilita'   area di flusso

  - tortuosita'   : quanto e' piu' lungo il percorso            (geometria locale)
  - permeabilita' : quanto meno fluido passa                    (proprieta' idraulica)
  - area di flusso: su quanta sezione quel fluido si distribuisce (topologia del backbone)

ATTENZIONE, ONESTA' SULLA TERZA VOCE. L'identita' e' esatta per costruzione,
perche' A* e' DEFINITA come Q/<u>. Ma <u> qui e' la media geometrica delle
velocita' per segmento, mentre la relazione Q = u*A vale per la media
aritmetica pesata sul flusso. A* non e' quindi l'area geometrica della sezione:
e' un termine di bilancio che assorbe sia l'area vera sia lo scarto fra le due
medie. Va presentata come area apparente, non come misura diretta.

USO
---
  python3 effective_permeability.py --root /hpc/scratch/filippo.porta/runs_10k
  python3 effective_permeability.py --root ... --paired
  python3 effective_permeability.py --root ... --paired --vt vt_out/vt_per_run.csv

Con --vt aggancia i risultati velocita'-tortuosita' e stampa la decomposizione
a tre fattori. Senza, produce solo le permeabilita'.

Dipendenze: numpy. Nessuno scipy: i test sono implementati a mano.
"""

import os
import csv
import math
import argparse
import itertools
import numpy as np

DIRECTIONS = ["X", "Y", "Z"]
DOMAINS = ["low", "medium", "high"]

# --- parametri fisici, tutti verificati ------------------------------------
MU_DEFAULT = 8.90e-4      # viscosita' dinamica [Pa s], acqua a 25 C
L_DEFAULT = 50.0          # lunghezza del dominio lungo il flusso [m]
A_DEFAULT = 2500.0        # area della faccia [m2] = 50 x 50
DP_DEFAULT = 1.0e6        # salto di pressione EFFETTIVAMENTE risolto [Pa]
DARCY = 9.869233e-13      # [m2] per la conversione


# ---------------------------------------------------------------- statistica
def wilcoxon(a, b):
    """Signed-rank a due code. Esatto per n<=20, altrimenti normale."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[d != 0]
    n = d.size
    if n < 5:
        return float("nan"), n
    ad = np.abs(d)
    order = np.argsort(ad)
    r = np.empty(n)
    r[order] = np.arange(1, n + 1)
    s = ad[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and s[j] == s[i]:
            j += 1
        if j - i > 1:
            r[order[i:j]] = r[order[i:j]].mean()
        i = j
    W = min(r[d > 0].sum(), r[d < 0].sum())
    tot = r.sum()
    if n <= 20:
        cnt = 0
        for signs in itertools.product((0, 1), repeat=n):
            w = sum(r[k] for k in range(n) if signs[k])
            if min(w, tot - w) <= W:
                cnt += 1
        return cnt / 2 ** n, n
    mu_ = n * (n + 1) / 4.0
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (W - mu_) / sd
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))), n


def mannwhitney(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = a.size, b.size
    allv = np.concatenate([a, b])
    order = np.argsort(allv)
    r = np.empty(allv.size)
    r[order] = np.arange(1, allv.size + 1)
    s = allv[order]
    i = 0
    while i < allv.size:
        j = i + 1
        while j < allv.size and s[j] == s[i]:
            j += 1
        if j - i > 1:
            r[order[i:j]] = r[order[i:j]].mean()
        i = j
    U = r[:na].sum() - na * (na + 1) / 2.0
    mu_ = na * nb / 2.0
    sd = math.sqrt(na * nb * (na + nb + 1) / 12.0)
    z = (U - mu_) / sd
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def star(p):
    return "**" if p < 0.01 else ("*" if p < 0.05 else "ns")


# ------------------------------------------------------------------ lettura
def read_flux(path):
    try:
        with open(path) as f:
            return float(f.read().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def collect(root, domains, paired):
    """Ritorna {(dom, D, run): Q}."""
    out = {}
    for dom in domains:
        dr = os.path.join(root, dom)
        if not os.path.isdir(dr):
            print(f"[{dom}] cartella assente: {dr}")
            continue
        runs = sorted((x for x in os.listdir(dr)
                       if os.path.isdir(os.path.join(dr, x))),
                      key=lambda s: (len(s), s))
        found = {D: {} for D in DIRECTIONS}
        for r in runs:
            for D in DIRECTIONS:
                q = read_flux(os.path.join(dr, r, "transport", D,
                                           "inputflux_m3s"))
                if q is not None and q > 0:
                    found[D][r] = q
        if paired:
            common = set.intersection(*[set(found[D]) for D in DIRECTIONS])
            sizes = "/".join(str(len(found[D])) for D in DIRECTIONS)
            print(f"[{dom}] portate trovate {sizes} -> appaiate {len(common)}")
        else:
            common = set().union(*[set(found[D]) for D in DIRECTIONS])
            print(f"[{dom}] portate trovate "
                  + "/".join(str(len(found[D])) for D in DIRECTIONS))
        for D in DIRECTIONS:
            for r, q in found[D].items():
                if r in common:
                    out[(dom, D, r)] = q
    return out


# ----------------------------------------------------------------- analisi
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/hpc/scratch/filippo.porta/runs_10k")
    ap.add_argument("--domains", default=",".join(DOMAINS))
    ap.add_argument("--paired", action="store_true",
                    help="solo i run con tutte e tre le direzioni")
    ap.add_argument("--vt", default=None,
                    help="vt_per_run.csv, per la decomposizione a tre fattori")
    ap.add_argument("--min-n", type=float, default=50000,
                    help="soglia sui segmenti, applicata ai dati --vt")
    ap.add_argument("--mu", type=float, default=MU_DEFAULT)
    ap.add_argument("--length", type=float, default=L_DEFAULT)
    ap.add_argument("--area", type=float, default=A_DEFAULT)
    ap.add_argument("--dp", type=float, default=DP_DEFAULT)
    ap.add_argument("--outdir", default="keff_out")
    a = ap.parse_args()

    doms = [d.strip() for d in a.domains.split(",") if d.strip()]
    os.makedirs(a.outdir, exist_ok=True)

    print(f"K = Q*mu*L/(A*dP)  con  mu={a.mu:.3g} Pa s   L={a.length:g} m   "
          f"A={a.area:g} m2   dP={a.dp:.3g} Pa\n")
    Q = collect(a.root, doms, a.paired)
    if not Q:
        raise SystemExit("nessuna portata trovata: controlla --root")

    K = {k: q * a.mu * a.length / (a.area * a.dp) for k, q in Q.items()}

    # --- per run --------------------------------------------------------
    path = os.path.join(a.outdir, "keff_per_run.csv")
    with open(path, "w") as f:
        f.write("domain;direction;run;Q_m3s;K_m2;K_darcy\n")
        for (dom, D, r), k in sorted(K.items()):
            f.write(f"{dom};{D};{r};{Q[(dom,D,r)]:.6g};{k:.6g};"
                    f"{k/DARCY:.4g}\n")
    print(f"\n-> {path}")

    def vals(dom, D):
        return np.array([K[(dom, D, r)] for (d2, D2, r) in K
                         if d2 == dom and D2 == D])

    def per_run(dom, D):
        return {r: K[(dom, D, r)] for (d2, D2, r) in K
                if d2 == dom and D2 == D}

    print("\n" + "=" * 74)
    print("PERMEABILITA' EFFICACE  [m2]   media geometrica x/ sd geometrica")
    print("=" * 74)
    print(f"  {'dominio':<9}" + "".join(f"{D:>21}" for D in DIRECTIONS))
    summ = [["domain", "direction", "n", "K_geomean_m2", "K_geosd_factor",
             "K_geomean_darcy", "K_min_m2", "K_max_m2"]]
    for dom in doms:
        cells = []
        for D in DIRECTIONS:
            v = vals(dom, D)
            if v.size < 2:
                cells.append("-"); continue
            lg = np.log(v)
            gm, gs = np.exp(lg.mean()), np.exp(lg.std(ddof=1))
            cells.append(f"{gm:.3e} x/{gs:.2f}")
            summ.append([dom, D, v.size, f"{gm:.6g}", f"{gs:.4g}",
                         f"{gm/DARCY:.4g}", f"{v.min():.6g}", f"{v.max():.6g}"])
        print(f"  {dom:<9}" + "".join(f"{c:>21}" for c in cells))
    with open(os.path.join(a.outdir, "keff_summary.csv"), "w") as f:
        f.write("\n".join(";".join(map(str, r)) for r in summ) + "\n")
    print(f"-> {a.outdir}/keff_summary.csv")

    # --- anisotropia, appaiata ------------------------------------------
    print("\n" + "=" * 74)
    print("ANISOTROPIA DI PERMEABILITA'   (rapporti per realizzazione)")
    print("=" * 74)
    for dom in doms:
        for A_, B_ in (("X", "Y"), ("Z", "Y"), ("Z", "X")):
            ga, gb = per_run(dom, A_), per_run(dom, B_)
            c = sorted(set(ga) & set(gb), key=int)
            if len(c) < 5:
                continue
            ratio = np.array([ga[r] / gb[r] for r in c])
            lg = np.log(ratio)
            p, n = wilcoxon([ga[r] for r in c], [gb[r] for r in c])
            print(f"  {dom:<8} K_{A_}/K_{B_} = {np.exp(lg.mean()):6.2f} "
                  f"x/{np.exp(lg.std(ddof=1)):.2f}   n={n:2d}  "
                  f"p={p:.4f} {star(p)}")

    print("\n" + "=" * 74)
    print("EFFETTO DELL'INTENSITA' (Mann-Whitney, realizzazioni indipendenti)")
    print("=" * 74)
    for D in DIRECTIONS:
        line = f"  {D}: "
        for dom in doms:
            v = vals(dom, D)
            line += f" {dom[:3]} {np.exp(np.log(v).mean()):.2e}"
        ps = []
        for d1, d2 in zip(doms[:-1], doms[1:]):
            v1, v2 = vals(d1, D), vals(d2, D)
            if v1.size > 2 and v2.size > 2:
                ps.append(f"{d1[:3]}-{d2[:3]} p={mannwhitney(v1, v2):.4f}")
        print(line + "   | " + "  ".join(ps))

    # --- decomposizione a tre fattori -----------------------------------
    if a.vt and os.path.isfile(a.vt):
        rows = list(csv.DictReader(open(a.vt), delimiter=";"))
        vt = {}
        for dom in doms:
            keep = {D: set(r["run"] for r in rows if r["domain"] == dom
                           and r["direction"] == D
                           and float(r["n"]) >= a.min_n) for D in DIRECTIONS}
            com = set.intersection(*keep.values()) if keep else set()
            for r in rows:
                if r["domain"] == dom and r["run"] in com:
                    vt[(dom, r["direction"], r["run"])] = (
                        float(r["geo_chi"]), float(r["geo_u"]))
        print("\n" + "=" * 74)
        print("DECOMPOSIZIONE A TRE FATTORI DEL RITARDO")
        print("  dt_A/dt_B = (chi_A/chi_B) * (K_B/K_A) * (A*_A/A*_B)")
        print("               tortuosita'   permeabilita'  area di flusso")
        print("  A* = Q/<u> e' un'area APPARENTE: vedi l'intestazione dello script")
        print("=" * 74)
        dec = [["domain", "pair", "n", "dt_ratio", "f_tortuosity",
                "f_permeability", "f_flow_area"]]
        for dom in doms:
            for A_, B_ in (("Y", "X"), ("Y", "Z")):
                c = sorted({r for (d2, D2, r) in vt if d2 == dom and D2 == A_}
                           & {r for (d2, D2, r) in vt if d2 == dom and D2 == B_}
                           & {r for (d2, D2, r) in K if d2 == dom and D2 == A_}
                           & {r for (d2, D2, r) in K if d2 == dom and D2 == B_},
                           key=int)
                if len(c) < 5:
                    continue
                ft, fk, fa, tot = [], [], [], []
                for r in c:
                    ca, ua = vt[(dom, A_, r)]
                    cb, ub = vt[(dom, B_, r)]
                    ka, kb = K[(dom, A_, r)], K[(dom, B_, r)]
                    Aa, Ab = Q[(dom, A_, r)] / ua, Q[(dom, B_, r)] / ub
                    ft.append(ca / cb); fk.append(kb / ka); fa.append(Aa / Ab)
                    tot.append((ca / cb) * (ub / ua))
                g = lambda v: np.exp(np.mean(np.log(v)))
                gsd = lambda v: np.exp(np.std(np.log(v), ddof=1))
                print(f"  {dom:<7} {A_}/{B_} (n={len(c):2d}):  dt x{g(tot):5.2f}"
                      f"  =  tort x{g(ft):4.2f}  *  perm x{g(fk):5.2f}"
                      f"  *  area x{g(fa):4.2f}")
                dec.append([dom, f"{A_}/{B_}", len(c), f"{g(tot):.4g}",
                            f"{g(ft):.4g}", f"{g(fk):.4g}", f"{g(fa):.4g}"])
        with open(os.path.join(a.outdir, "keff_decomposition.csv"), "w") as f:
            f.write("\n".join(";".join(map(str, r)) for r in dec) + "\n")
        print(f"\n-> {a.outdir}/keff_decomposition.csv")
        print("\n  Lettura: 'perm' e' quanto meno fluido passa lungo Y, 'area' e'")
        print("  su quanta sezione quel fluido si distribuisce. Se area < 1 il")
        print("  backbone piu' stretto compensa in parte la permeabilita' minore.")
    elif a.vt:
        print(f"\n[!] {a.vt} non trovato: salto la decomposizione.")


if __name__ == "__main__":
    main()
