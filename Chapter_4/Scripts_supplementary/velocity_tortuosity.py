#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
velocity_tortuosity.py  (v2 - lettore adattato a dfnTrans outcontroldir)
=======================================================================

Analisi congiunta velocita' di trasporto locale <-> tortuosita' locale,
per segmenti fra control plane consecutivi (Eq. 14-16 del manoscritto).

FORMATO DATI
------------
transport/<DIR>/outcontroldir/part_control_<N>   (un file per particella)
colonne:  0 t | 1 x | 2 y | 3 z | 4 Vx | 5 Vy | 6 Vz | 7 lambda_cum | 8 frac | 9 apertura
Una riga per ogni attraversamento di control plane (piani ogni dx = 5 m).
Se una particella riattraversa lo stesso piano, si tiene la PRIMA traversata
(t minimo), coerentemente con la definizione di lambda(x_i; a) nel manoscritto.

IMPOSTAZIONE (leggere prima di usare)
-------------------------------------
Con control plane equispaziati (dx = 5 m costante):

    chi_i = dlambda_i / dx          (Eq. 14, tortuosita' locale)
    v_i   = dx / dt_i               (Eq. 16, velocita' di trasporto locale)
    u_i   = dlambda_i / dt_i        (velocita' media VERA lungo la pathline)

ATTENZIONE alle unita': dfnTrans riporta i tempi in ANNI e le velocita' in
m/yr (coerente con la Tabella 3 del manoscritto). Le conclusioni - b, beta,
i rapporti fra direzioni, la decomposizione della varianza - sono comunque
invarianti per fattore moltiplicativo costante sulla velocita'.

da cui l'identita' esatta

    v_i = u_i / chi_i       <=>     log v = log u - log chi

Il grafico (v, chi) e' quindi in larga parte una tautologia: mostra una
nuvola ~1/chi anche con campo di velocita' uniforme. Lo script lo produce
comunque (riga alta) ma la parte informativa e':

  (a) u vs chi  -> relazione DECONFUSA. Pendenza b in u ~ chi^b:
        b ~  0  : la tortuosita' rallenta solo allungando il percorso
        b <  0  : i tratti tortuosi sono ANCHE intrinsecamente lenti
                  (ricircolazione, aperture piccole) -> rallentamento
                  super-geometrico, coerente con Sherman et al. (2020)
        b >  0  : i tratti tortuosi sono su canali veloci (effetto smorzato)

  (b) decomposizione della varianza
        Var(log v) = Var(log u) + Var(log chi) - 2 Cov(log u, log chi)

  (c) decomposizione del tempo di percorrenza fra direzioni:
        dt_i = dx * chi_i / u_i
      -> il rapporto fra i tempi si fattorizza in
         (contributo tortuosita') x (contributo velocita')

USO
---
  # prova veloce su 2 run per dominio, 2000 particelle ciascuno
  python3 velocity_tortuosity.py --run --root /hpc/scratch/filippo.porta/runs_10k \\
          --max-runs 2 --max-particles 2000

  # produzione (tutte le particelle, 5 run per dominio: gia' ~500k segmenti)
  python3 velocity_tortuosity.py --run --root /hpc/scratch/filippo.porta/runs_10k \\
          --max-runs 5

  python3 velocity_tortuosity.py --plot

ATTENZIONE alle prestazioni: ci sono ~10^4 file minuscoli per run/direzione.
Il collo di bottiglia e' l'apertura dei file sul filesystem condiviso, non il
calcolo. Con 5 run per dominio hai gia' potenza statistica abbondante
(~5x10^5 segmenti per dominio/direzione): non serve girare tutti i 44 run.

Dipendenze: numpy. matplotlib solo per --plot. Nessuno scipy.
"""

import os
import sys
import time
import argparse
import numpy as np

# ----------------------------------------------------------------------
# CONFIGURAZIONE
# ----------------------------------------------------------------------
DIRECTIONS = ["X", "Y", "Z"]
DOMAINS = ["low", "medium", "high"]
DEFAULT_ROOT = "/hpc/scratch/filippo.porta/runs_10k"

DX_DEFAULT = 5.0                 # spaziatura control plane [m]
CTRL_SUBDIR = "outcontroldir"
CTRL_PREFIX = "part_control_"

# indice di colonna della coordinata lungo la direzione di flusso
COORD_COL = {"X": 1, "Y": 2, "Z": 3}
COL_TIME = 0
COL_LAMBDA = 7
COL_APERTURE = 9

# tolleranza: una riga e' su un control plane se dista < TOL_FRAC*dx dal piano
TOL_FRAC = 0.05

# binning istogrammi 2D (log10)
NBINS = 220
# I range degli istogrammi 2D sono determinati AUTOMATICAMENTE dal primo
# blocco di dati (percentili 0.05-99.95 con margine). Niente valori cablati:
# le unita' di dfnTrans sono m/yr e un range sbagliato svuota gli istogrammi.
HIST_PCT = (0.05, 99.95)
HIST_PAD = 0.5          # margine in decadi

SAMPLE_MAX = 300000
OUTDIR = "vt_out"        # tutti gli output finiscono qui

# Range degli istogrammi imposti dall'esterno (--range-chi/-u/-v). Se None,
# ogni cella sceglie il proprio dai percentili. Il default e' comodo per
# guardare una cella per volta, ma produce nove range DIVERSI: in una figura
# a assi condivisi il bordo dell'istogramma di una cella cade dentro il
# pannello e si legge come una soglia fisica. Per la figura finale conviene
# imporre un range unico a tutte e nove.
FIX_RC = FIX_RV = FIX_RU = None

# ----------------------------------------------------------------------
# STATISTICA (senza scipy)
# ----------------------------------------------------------------------


def _rankdata(a):
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=float)
    ranks[order] = np.arange(1, a.size + 1, dtype=float)
    sa = a[order]
    i = 0
    while i < a.size:
        j = i + 1
        while j < a.size and sa[j] == sa[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = ranks[order[i:j]].mean()
        i = j
    return ranks


def spearman(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < 3:
        return float("nan")
    rx = _rankdata(x) - (x.size + 1) / 2.0
    ry = _rankdata(y) - (y.size + 1) / 2.0
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def _auto_range(a, lo_at_zero=False):
    """Range in log10 dai percentili, arrotondato all'esterno."""
    lo, hi = np.percentile(a, HIST_PCT)
    lo, hi = lo - HIST_PAD, hi + HIST_PAD
    if lo_at_zero:
        lo = 0.0
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = (0.0, 2.0) if lo_at_zero else (lo - 1.0, lo + 1.0)
    return (float(np.floor(lo * 4) / 4), float(np.ceil(hi * 4) / 4))


class Moments:
    """Statistiche per singolo run: solo momenti, niente istogrammi."""

    def __init__(self):
        self.n = 0
        for k in ("sc", "sc2", "su", "su2", "sv", "suc"):
            setattr(self, k, 0.0)
        self.s = None

    def add(self, chi, v, u):
        lc, lv, lu = np.log(chi), np.log(v), np.log(u)
        self.n += lc.size
        self.sc += lc.sum(); self.sc2 += (lc * lc).sum()
        self.su += lu.sum(); self.su2 += (lu * lu).sum()
        self.sv += lv.sum()
        self.suc += (lu * lc).sum()
        keep = min(lc.size, 200000)
        idx = (np.random.choice(lc.size, keep, replace=False)
               if keep < lc.size else np.arange(lc.size))
        self.s = np.column_stack([lc[idx], lu[idx]])

    def stats(self):
        n = self.n
        if n < 50:
            return None
        mc, mu = self.sc / n, self.su / n
        Vc = max(self.sc2 / n - mc * mc, 0.0)
        Vu = max(self.su2 / n - mu * mu, 0.0)
        Cuc = self.suc / n - mu * mc
        nan = float("nan")
        return dict(
            n=n, geo_chi=np.exp(mc), geo_u=np.exp(mu), geo_v=np.exp(self.sv / n),
            var_log_chi=Vc, var_log_u=Vu,
            slope_u_vs_chi=(Cuc / Vc) if Vc > 0 else nan,
            pearson_logu_logchi=(Cuc / np.sqrt(Vu * Vc)) if Vu * Vc > 0 else nan,
            spearman_logu_logchi=(spearman(self.s[:, 0], self.s[:, 1])
                                  if self.s is not None else nan))


class Accum:
    """Momenti in spazio log accumulati in streaming (esatti, memoria O(1))."""

    def __init__(self):
        self.n = 0
        for k in ("sv", "sv2", "sc", "sc2", "su", "su2", "svc", "suc"):
            setattr(self, k, 0.0)
        self.na = 0
        self.sa = self.sa2 = self.sac = 0.0
        self.rc = self.rv = self.ru = None      # range log10, fissati al 1o add
        self.n_out = 0
        self.hist_vchi = np.zeros((NBINS, NBINS), np.int64)
        self.hist_uchi = np.zeros((NBINS, NBINS), np.int64)
        self.sample = []
        self.n_sample = 0
        self.dt_sample = []

    def add(self, chi, v, u, dt, ap=None):
        lc, lv, lu = np.log(chi), np.log(v), np.log(u)
        self.n += lc.size
        self.sv += lv.sum(); self.sv2 += (lv * lv).sum()
        self.sc += lc.sum(); self.sc2 += (lc * lc).sum()
        self.su += lu.sum(); self.su2 += (lu * lu).sum()
        self.svc += (lv * lc).sum()
        self.suc += (lu * lc).sum()

        if ap is not None:
            good = np.isfinite(ap) & (ap > 0)
            if good.any():
                la = np.log(ap[good])
                lcg = lc[good]
                self.na += la.size
                self.sa += la.sum(); self.sa2 += (la * la).sum()
                self.sac += (la * lcg).sum()

        L = np.log(10.0)
        c10, v10, u10 = lc / L, lv / L, lu / L
        if self.rc is None:
            self.rc = FIX_RC or _auto_range(c10, lo_at_zero=True)
            self.rv = FIX_RV or _auto_range(v10)
            self.ru = FIX_RU or _auto_range(u10)
        h, _, _ = np.histogram2d(c10, v10, bins=NBINS,
                                 range=[self.rc, self.rv])
        self.hist_vchi += h.astype(np.int64)
        h, _, _ = np.histogram2d(c10, u10, bins=NBINS,
                                 range=[self.rc, self.ru])
        self.hist_uchi += h.astype(np.int64)
        self.n_out += int(lc.size - h.sum())

        keep = min(lc.size, max(0, SAMPLE_MAX - self.n_sample))
        if keep > 0:
            idx = (np.random.choice(lc.size, keep, replace=False)
                   if keep < lc.size else np.arange(lc.size))
            self.sample.append(np.column_stack([lc[idx], lv[idx], lu[idx]]))
            self.dt_sample.append(dt[idx])
            self.n_sample += keep

    def stats(self):
        n = self.n
        if n < 10:
            return None
        mv, mc, mu = self.sv / n, self.sc / n, self.su / n
        Vv = max(self.sv2 / n - mv * mv, 0.0)
        Vc = max(self.sc2 / n - mc * mc, 0.0)
        Vu = max(self.su2 / n - mu * mu, 0.0)
        Cvc = self.svc / n - mv * mc
        Cuc = self.suc / n - mu * mc
        nan = float("nan")
        out = dict(
            n=int(n),
            geo_chi=np.exp(mc), geo_v=np.exp(mv), geo_u=np.exp(mu),
            var_log_chi=Vc, var_log_v=Vv, var_log_u=Vu,
            cov_logu_logchi=Cuc,
            pearson_logv_logchi=(Cvc / np.sqrt(Vv * Vc)) if Vv * Vc > 0 else nan,
            pearson_logu_logchi=(Cuc / np.sqrt(Vu * Vc)) if Vu * Vc > 0 else nan,
            slope_v_vs_chi=(Cvc / Vc) if Vc > 0 else nan,
            slope_u_vs_chi=(Cuc / Vc) if Vc > 0 else nan,
            frac_var_from_u=(Vu / Vv) if Vv > 0 else nan,
            frac_var_from_chi=(Vc / Vv) if Vv > 0 else nan,
            frac_var_from_cov=(-2.0 * Cuc / Vv) if Vv > 0 else nan,
        )
        # apertura (diagnostico ausiliario)
        if self.na > 20:
            ma = self.sa / self.na
            Va = max(self.sa2 / self.na - ma * ma, 0.0)
            Ca = self.sac / self.na - ma * mc
            out["pearson_logap_logchi"] = (Ca / np.sqrt(Va * Vc)
                                           if Va * Vc > 0 else nan)
        else:
            out["pearson_logap_logchi"] = nan

        if self.sample:
            S = np.vstack(self.sample)
            out["spearman_logv_logchi"] = spearman(S[:, 0], S[:, 1])
            out["spearman_logu_logchi"] = spearman(S[:, 0], S[:, 2])
            out["median_dt"] = float(np.median(np.concatenate(self.dt_sample)))
        else:
            out["spearman_logv_logchi"] = nan
            out["spearman_logu_logchi"] = nan
            out["median_dt"] = nan
        return out


# ----------------------------------------------------------------------
# LETTURA dfnTrans
# ----------------------------------------------------------------------

def read_particle(path, ccol, dx):
    """
    Legge un file part_control_<N>.
    Ritorna dict {indice_piano: (t, lambda, apertura)} con la PRIMA traversata.
    """
    planes = {}
    try:
        with open(path, "r", errors="replace") as f:
            for line in f:
                if not line:
                    continue
                c0 = line[0]
                if not (c0.isdigit() or c0 in "+-. \t"):
                    continue          # riga di intestazione
                p = line.split()
                if len(p) < 8:
                    continue
                try:
                    t = float(p[COL_TIME])
                    c = float(p[ccol])
                    lam = float(p[COL_LAMBDA])
                except ValueError:
                    continue          # intestazione con spazio iniziale
                q = c / dx
                k = int(round(q))
                if abs(q - k) > TOL_FRAC:
                    continue          # non e' su un control plane
                prev = planes.get(k)
                if prev is None or t < prev[0]:
                    ap = np.nan
                    if len(p) > COL_APERTURE:
                        try:
                            ap = float(p[COL_APERTURE])
                        except ValueError:
                            pass
                    planes[k] = (t, lam, ap)
    except OSError:
        return None
    return planes


def segments_from_particle(planes, dx):
    """
    Ordina i piani per tempo di prima traversata (= ordine di valle) e
    costruisce i segmenti fra piani ADIACENTI.
    Ritorna liste (chi, v, u, dt, ap).
    """
    if planes is None or len(planes) < 2:
        return None
    items = sorted(planes.items(), key=lambda kv: kv[1][0])   # per t crescente
    out = []
    for (k0, (t0, l0, a0)), (k1, (t1, l1, _)) in zip(items[:-1], items[1:]):
        if abs(k1 - k0) != 1:
            continue                     # piani non adiacenti: salta
        dt = t1 - t0
        dl = l1 - l0
        if dt <= 0 or dl <= 0:
            continue
        chi = dl / dx
        if chi < 0.999:
            continue                     # impossibile: scarto
        out.append((max(chi, 1.0), dx / dt, dl / dt, dt, a0))
    return out or None


def profile_from_particle(planes, dx):
    """
    Per ogni piano attraversato, ritorna
      (d, chi_local, chi_cum)
    con d = distanza dall'inlet lungo la direzione di flusso,
        chi_local = tortuosita' del segmento che TERMINA in quel piano,
        chi_cum   = lambda(d)/d  (tortuosita' cumulata dall'inlet).
    chi_local e' None se il piano precedente manca.
    """
    if planes is None or len(planes) < 2:
        return None
    items = sorted(planes.items(), key=lambda kv: kv[1][0])
    k0 = items[0][0]
    lam0 = items[0][1][1]
    out = []
    prev = items[0]
    for k, (t, lam, _a) in items[1:]:
        d = abs(k - k0) * dx
        if d <= 0:
            continue
        cum = (lam - lam0) / d
        loc = None
        if abs(k - prev[0]) == 1:
            dl = lam - prev[1][1]
            if dl > 0:
                loc = max(dl / dx, 1.0)
        if cum >= 0.999:
            out.append((d, loc, max(cum, 1.0)))
        prev = (k, (t, lam, _a))
    return out or None


def process_profile(trans_dir, direction, dx, max_particles=None):
    """Accumula i profili ⟨chi_local(x)⟩ e ⟨chi_cum(x)⟩ su tutte le particelle."""
    cdir = os.path.join(trans_dir, CTRL_SUBDIR)
    if not os.path.isdir(cdir):
        return None
    try:
        names = [e.name for e in os.scandir(cdir)
                 if e.is_file() and e.name.startswith(CTRL_PREFIX)]
    except OSError:
        return None
    if not names:
        return None
    if max_particles and len(names) > max_particles:
        names = list(np.random.choice(names, max_particles, replace=False))
    ccol = COORD_COL[direction]
    agg = {}          # d -> [n_loc, sum_loc, sum_log_loc, n_cum, sum_cum, sum_log_cum]
    for nm in names:
        pr = profile_from_particle(read_particle(
            os.path.join(cdir, nm), ccol, dx), dx)
        if not pr:
            continue
        for d, loc, cum in pr:
            a = agg.setdefault(round(d, 6), [0, 0.0, 0.0, 0, 0.0, 0.0])
            if loc is not None:
                a[0] += 1; a[1] += loc; a[2] += np.log(loc)
            a[3] += 1; a[4] += cum; a[5] += np.log(cum)
    return agg or None


def run_profiles(root, domains, dx, max_runs, max_particles):
    os.makedirs(OUTDIR, exist_ok=True)
    table = {}
    for dom in domains:
        dr = os.path.join(root, dom)
        if not os.path.isdir(dr):
            continue
        runs = sorted((x for x in os.listdir(dr)
                       if os.path.isdir(os.path.join(dr, x))),
                      key=lambda s_: (len(s_), s_))
        for D in DIRECTIONS:
            tot, used = {}, []
            for r in runs:
                if max_runs and len(used) >= max_runs:
                    break
                td = os.path.join(dr, r, "transport", D)
                if not os.path.isdir(td):
                    continue
                agg = process_profile(td, D, dx, max_particles)
                if agg is None:
                    continue
                for d, a in agg.items():
                    t = tot.setdefault(d, [0, 0.0, 0.0, 0, 0.0, 0.0])
                    for i in range(6):
                        t[i] += a[i]
                used.append(r)
                print(f"  [{dom}/{D}] run {r} ok", flush=True)
            if not tot:
                continue
            table[(dom, D)] = tot
            print(f"[{dom}/{D}] profilo da {len(used)} run, "
                  f"{len(tot)} piani", flush=True)

    if not table:
        print("Nessun profilo calcolato.")
        return
    path = os.path.join(OUTDIR, "vt_profile.csv")
    with open(path, "w") as f:
        f.write("domain;direction;distance_m;n_local;mean_chi_local;"
                "geo_chi_local;n_cum;mean_chi_cum;geo_chi_cum\n")
        for dom in DOMAINS:
            for D in DIRECTIONS:
                tot = table.get((dom, D))
                if not tot:
                    continue
                for d in sorted(tot):
                    nl, sl, gl, nc, sc, gc = tot[d]
                    ml = sl / nl if nl else float("nan")
                    xl = np.exp(gl / nl) if nl else float("nan")
                    mc = sc / nc if nc else float("nan")
                    xc = np.exp(gc / nc) if nc else float("nan")
                    f.write(f"{dom};{D};{d:g};{nl};{ml:.6g};{xl:.6g};"
                            f"{nc};{mc:.6g};{xc:.6g}\n")
    print(f"-> {path}")
    _print_profile(table)
    try:
        _plot_profile(table, domains)
    except ImportError:
        print("[!] matplotlib assente: figura del profilo non prodotta.")
    except Exception as exc:
        print(f"[!] figura del profilo non prodotta: {exc}")


def _print_profile(table):
    print("\n" + "=" * 74)
    print("MEDIA ARITMETICA DELLA TORTUOSITA' LOCALE PER PIANO")
    print("  (se e' piatta lungo x il sistema e' stazionario; "
          "se deriva, niente asintoto)")
    print("=" * 74)
    for dom in DOMAINS:
        for D in DIRECTIONS:
            tot = table.get((dom, D))
            if not tot:
                continue
            ds = sorted(tot)
            vals = [tot[d][1] / tot[d][0] if tot[d][0] else float("nan")
                    for d in ds]
            body = " ".join(f"{v:5.2f}" for v in vals)
            fin = [v for v in vals if np.isfinite(v)]
            drift = (max(fin) - min(fin)) / np.mean(fin) * 100 if fin else float("nan")
            print(f"  {dom+'/'+D:<10} {body}   escursione {drift:.0f}%")


def _plot_profile(table, domains):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = {"X": "tab:blue", "Y": "tab:red", "Z": "tab:green"}
    doms = [d for d in DOMAINS if d in domains
            and any((d, D) in table for D in DIRECTIONS)]
    if not doms:
        return
    fig, axes = plt.subplots(2, len(doms), figsize=(4.4 * len(doms), 7.0),
                             squeeze=False)
    for j, dom in enumerate(doms):
        for D in DIRECTIONS:
            tot = table.get((dom, D))
            if not tot:
                continue
            ds = np.array(sorted(tot))
            loc = np.array([tot[d][1] / tot[d][0] if tot[d][0] else np.nan
                            for d in ds])
            cum = np.array([tot[d][4] / tot[d][3] if tot[d][3] else np.nan
                            for d in ds])
            axes[0][j].plot(ds, loc, "o-", color=cols[D], ms=4, label=D)
            axes[1][j].plot(ds, cum, "o-", color=cols[D], ms=4, label=D)
        axes[0][j].set_title(f"{dom}-intensity")
        for i in (0, 1):
            axes[i][j].grid(alpha=0.3)
            axes[i][j].set_xlabel("distanza dall'inlet  [m]")
        axes[0][j].legend(fontsize=8)
    axes[0][0].set_ylabel(r"$\langle\chi_i\rangle$ locale per segmento")
    axes[1][0].set_ylabel(r"$\langle\lambda(x)/x\rangle$ cumulata")
    fig.suptitle("Alto: test di stazionarieta' (media locale per segmento).   "
                 "Basso: tortuosita' cumulata (si appiattisce per costruzione).",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUTDIR, "fig_tortuosity_profile.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("->", out)


def process_run_dir(trans_dir, direction, dx, max_particles=None):
    """Elabora transport/<DIR>/outcontroldir. Ritorna array (chi,v,u,dt,ap)."""
    cdir = os.path.join(trans_dir, CTRL_SUBDIR)
    if not os.path.isdir(cdir):
        return None
    try:
        names = [e.name for e in os.scandir(cdir)
                 if e.is_file() and e.name.startswith(CTRL_PREFIX)]
    except OSError:
        return None
    if not names:
        return None
    if max_particles and len(names) > max_particles:
        names = list(np.random.choice(names, max_particles, replace=False))

    ccol = COORD_COL[direction]
    rows = []
    for nm in names:
        pl = read_particle(os.path.join(cdir, nm), ccol, dx)
        seg = segments_from_particle(pl, dx)
        if seg:
            rows.extend(seg)
    if not rows:
        return None
    A = np.asarray(rows, dtype=float)
    return A[:, 0], A[:, 1], A[:, 2], A[:, 3], A[:, 4]


# ----------------------------------------------------------------------
# RUN
# ----------------------------------------------------------------------

def _paired_runs(dom_root, runs):
    """Run presenti in TUTTE e tre le direzioni."""
    ok = []
    for r in runs:
        if all(os.path.isdir(os.path.join(dom_root, r, "transport", D,
                                          CTRL_SUBDIR))
               for D in DIRECTIONS):
            ok.append(r)
    return ok


def run_all(root, domains, dx, max_runs, max_particles,
            save_samples=False, paired=False):
    os.makedirs(OUTDIR, exist_ok=True)
    results = {}
    per_run = []
    for dom in domains:
        dom_root = os.path.join(root, dom)
        if not os.path.isdir(dom_root):
            print(f"[{dom}] cartella assente: {dom_root}")
            continue
        runs = sorted((d for d in os.listdir(dom_root)
                       if os.path.isdir(os.path.join(dom_root, d))),
                      key=lambda s: (len(s), s))
        if paired:
            allr = runs
            runs = _paired_runs(dom_root, runs)
            print(f"[{dom}] appaiamento: {len(runs)} run su {len(allr)} "
                  f"hanno tutte e tre le direzioni")
            if not runs:
                print(f"[{dom}] nessun run completo, salto")
                continue
        for D in DIRECTIONS:
            acc = Accum()
            used, t0 = [], time.time()
            for r in runs:
                if max_runs and len(used) >= max_runs:
                    break
                td = os.path.join(dom_root, r, "transport", D)
                if not os.path.isdir(td):
                    continue
                res = process_run_dir(td, D, dx, max_particles)
                if res is None:
                    continue
                chi, v, u, dt, ap = res
                acc.add(chi, v, u, dt, ap)
                m = Moments(); m.add(chi, v, u)
                ms = m.stats()
                if ms:
                    ms.update(domain=dom, direction=D, run=r)
                    per_run.append(ms)
                used.append(r)
                print(f"  [{dom}/{D}] run {r}: {chi.size:,} segmenti "
                      f"({time.time()-t0:.0f} s)", flush=True)
            st = acc.stats()
            if st is None:
                print(f"[{dom}/{D}] nessun segmento valido")
                continue
            st.update(domain=dom, direction=D, n_runs=len(used),
                      runs=",".join(used))
            results[(dom, D)] = st
            np.savez_compressed(
                os.path.join(OUTDIR, f"vt_hist_{dom}_{D}.npz"),
                hist_vchi=acc.hist_vchi, hist_uchi=acc.hist_uchi,
                logchi_range=np.array(acc.rc if acc.rc else (0.0, 2.0)),
                logv_range=np.array(acc.rv if acc.rv else (0.0, 1.0)),
                logu_range=np.array(acc.ru if acc.ru else (0.0, 1.0)))
            if acc.n_out:
                print(f"   [i] {acc.n_out:,} segmenti fuori dal range "
                      f"dell'istogramma ({100*acc.n_out/st['n']:.2f}%)")
            if save_samples and acc.sample:
                np.savez_compressed(
                    os.path.join(OUTDIR, f"vt_sample_{dom}_{D}.npz"),
                    sample=np.vstack(acc.sample))
            print(f"[{dom}/{D}] TOT {st['n']:,} segmenti da {len(used)} run | "
                  f"<chi>g={st['geo_chi']:.3f}  <u>g={st['geo_u']:.3e} m/yr | "
                  f"b(u~chi^b)={st['slope_u_vs_chi']:+.3f}  "
                  f"r(logu,logchi)={st['pearson_logu_logchi']:+.3f}",
                  flush=True)
    if not results:
        print("\nNessun risultato. Controlla --root e la presenza di "
              "transport/<DIR>/outcontroldir.")
        return results
    _write_summary(results)
    _write_per_run(per_run)
    _write_decomposition(results)
    _print_variance(results)
    try:
        make_plots(domains)
    except ImportError:
        print("\n[!] matplotlib non disponibile: figure non prodotte.")
        print("    Gli istogrammi sono comunque in " + OUTDIR +
              "/vt_hist_*.npz (pochi kB l'uno):")
        print("    caricali altrove, oppure installa matplotlib e rilancia")
        print("    con --plot (non ricalcola nulla).")
    except Exception as exc:
        print(f"\n[!] figure non prodotte: {exc}")
    return results


def run_hist_only(root, domains, dx, max_particles):
    """
    Rigenera SOLO i vt_hist_*.npz, senza toccare nessun CSV.

    A cosa serve: gli istogrammi servono unicamente a disegnare la figura,
    mentre b, rho e tutte le statistiche stanno in vt_per_run.csv. Se gli
    .npz si perdono non c'e' motivo di rifare l'analisi completa.

    Due vincoli, non negoziabili:

    1. STESSI RUN. L'elenco delle realizzazioni viene letto da
       vt_per_run.csv, non ricostruito dal filesystem. Cosi' la nuvola
       disegnata e i numeri annotati nel pannello descrivono le stesse
       reti. Ricostruirlo dal disco reintrodurrebbe proprio il disallineamento
       che l'appaiamento serviva a evitare.

    2. NESSUNA SCRITTURA SUI CSV. Con --max-particles i momenti sarebbero
       calcolati su un sottocampione: sovrascrivere vt_per_run.csv
       degraderebbe silenziosamente le statistiche gia' buone. Qui non
       viene scritto.

    Il sottocampionamento delle particelle e' casuale (process_run_dir usa
    np.random.choice), quindi la densita' resta non distorta: cambia solo
    la sgranatura dell'istogramma.
    """
    per_run_path = os.path.join(OUTDIR, "vt_per_run.csv")
    if not os.path.isfile(per_run_path):
        raise SystemExit(
            f"--hist-only ha bisogno di {per_run_path} per sapere quali run "
            "usare.\nSe hai perso anche quello, serve un --run completo.")

    runs, nseg = {}, {}
    with open(per_run_path) as f:
        cols = f.readline().rstrip("\n").split(";")
        ic = {c: i for i, c in enumerate(cols)}
        for line in f:
            p = line.rstrip("\n").split(";")
            if len(p) < len(cols):
                continue
            k = (p[ic["domain"]], p[ic["direction"]])
            runs.setdefault(k, []).append(p[ic["run"]])
            nseg[k] = nseg.get(k, 0.0) + float(p[ic["n"]])

    print(f"[hist-only] elenco run da {per_run_path}")
    for dom in domains:
        sizes = "/".join(str(len(runs.get((dom, D), []))) for D in DIRECTIONS)
        print(f"  {dom:<8} realizzazioni per direzione (X/Y/Z): {sizes}")
    if max_particles:
        print(f"[hist-only] sottocampionamento a {max_particles} particelle "
              "per run: influisce solo sulla sgranatura della figura")

    os.makedirs(OUTDIR, exist_ok=True)
    for dom in domains:
        dom_root = os.path.join(root, dom)
        for D in DIRECTIONS:
            rl = runs.get((dom, D))
            if not rl:
                print(f"[{dom}/{D}] nessun run in vt_per_run.csv, salto")
                continue
            acc, used, t0 = Accum(), [], time.time()
            for r in rl:
                td = os.path.join(dom_root, r, "transport", D)
                res = process_run_dir(td, D, dx, max_particles)
                if res is None:
                    print(f"  [{dom}/{D}] run {r}: dati assenti, salto")
                    continue
                chi, v, u, dt, ap = res
                acc.add(chi, v, u, dt, ap)
                used.append(r)
            st = acc.stats()
            if st is None:
                print(f"[{dom}/{D}] nessun segmento valido")
                continue
            np.savez_compressed(
                os.path.join(OUTDIR, f"vt_hist_{dom}_{D}.npz"),
                hist_vchi=acc.hist_vchi, hist_uchi=acc.hist_uchi,
                logchi_range=np.array(acc.rc if acc.rc else (0.0, 2.0)),
                logv_range=np.array(acc.rv if acc.rv else (0.0, 1.0)),
                logu_range=np.array(acc.ru if acc.ru else (0.0, 1.0)))
            frac = 100.0 * st["n"] / nseg[(dom, D)] if nseg.get((dom, D)) else float("nan")
            print(f"[{dom}/{D}] {st['n']:,} segmenti da {len(used)} run "
                  f"({frac:.0f}% di quelli usati per le statistiche) "
                  f"-> vt_hist_{dom}_{D}.npz  [{time.time()-t0:.0f} s]",
                  flush=True)
    print(f"\n-> istogrammi rigenerati in {OUTDIR}/. "
          "Nessun CSV e' stato modificato.")


def _write_summary(results):
    cols = ["domain", "direction", "n_runs", "n",
            "geo_chi", "geo_v", "geo_u", "median_dt",
            "var_log_chi", "var_log_v", "var_log_u",
            "pearson_logv_logchi", "pearson_logu_logchi",
            "spearman_logv_logchi", "spearman_logu_logchi",
            "slope_v_vs_chi", "slope_u_vs_chi",
            "frac_var_from_u", "frac_var_from_chi", "frac_var_from_cov",
            "pearson_logap_logchi", "runs"]
    with open(os.path.join(OUTDIR, "vt_summary.csv"), "w") as f:
        f.write(";".join(cols) + "\n")
        for dom in DOMAINS:
            for D in DIRECTIONS:
                st = results.get((dom, D))
                if st is None:
                    continue
                f.write(";".join(
                    st[c] if isinstance(st[c], str)
                    else (str(st[c]) if isinstance(st[c], int)
                          else f"{st[c]:.6g}")
                    for c in cols) + "\n")
    print(f"\n-> {OUTDIR}/vt_summary.csv")


def _write_per_run(per_run):
    """Una riga per (dominio, direzione, run): serve per media +/- sd sulle
    realizzazioni e per i test appaiati, come nel resto del manoscritto."""
    if not per_run:
        return
    cols = ["domain", "direction", "run", "n", "geo_chi", "geo_v", "geo_u",
            "var_log_chi", "var_log_u", "slope_u_vs_chi",
            "pearson_logu_logchi", "spearman_logu_logchi"]
    path = os.path.join(OUTDIR, "vt_per_run.csv")
    with open(path, "w") as f:
        f.write(";".join(cols) + "\n")
        for d in per_run:
            f.write(";".join(
                str(d[c]) if isinstance(d[c], (str, int)) else f"{d[c]:.6g}"
                for c in cols) + "\n")
    print(f"-> {path}   ({len(per_run)} righe)")

    # riepilogo: media +/- sd sulle realizzazioni, non sui segmenti
    print("\n" + "=" * 74)
    print("b(u~chi^b) COME MEDIA +/- SD SULLE REALIZZAZIONI")
    print("  (l'unita' di replicazione e' il run, non il singolo segmento)")
    print("=" * 74)
    print(f"  {'dom':<9}" + "".join(f"{D:>18}" for D in DIRECTIONS))
    for dom in DOMAINS:
        cells = []
        for D in DIRECTIONS:
            vals = [d["slope_u_vs_chi"] for d in per_run
                    if d["domain"] == dom and d["direction"] == D
                    and np.isfinite(d["slope_u_vs_chi"])]
            cells.append(f"{np.mean(vals):+.3f}+/-{np.std(vals, ddof=1):.3f}"
                         if len(vals) > 1 else "     -")
        if any(c.strip() != "-" for c in cells):
            print(f"  {dom:<9}" + "".join(f"{c:>18}" for c in cells))


def _write_decomposition(results):
    lines = ["domain;pair;ratio_dt_total;factor_tortuosity;factor_velocity"]
    print("\n" + "=" * 74)
    print("DECOMPOSIZIONE DEL TEMPO PER SEGMENTO   dt = dx * chi / u")
    print("   rapporto A/B = (chi_A/chi_B) x (u_B/u_A)   [medie geometriche]")
    print("=" * 74)
    for dom in DOMAINS:
        for a, b in (("Y", "X"), ("Y", "Z")):
            sa, sb = results.get((dom, a)), results.get((dom, b))
            if sa is None or sb is None:
                continue
            ft = sa["geo_chi"] / sb["geo_chi"]
            fv = sb["geo_u"] / sa["geo_u"]
            lines.append(f"{dom};{a}/{b};{ft*fv:.4g};{ft:.4g};{fv:.4g}")
            print(f"  {dom:<7} {a}/{b}:  dt x{ft*fv:7.2f}  =  "
                  f"tortuosita' x{ft:5.2f}   *   velocita' x{fv:7.2f}")
    with open(os.path.join(OUTDIR, "vt_time_decomposition.csv"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"-> {OUTDIR}/vt_time_decomposition.csv")


def _print_variance(results):
    print("\n" + "=" * 74)
    print("DECOMPOSIZIONE DI Var(log v) = Var(log u) + Var(log chi) "
          "- 2Cov(log u, log chi)")
    print("=" * 74)
    print(f"  {'dom/dir':<12}{'Var(log v)':>12}{'da u':>9}{'da chi':>9}"
          f"{'da cov':>9}   b(u~chi^b)")
    for dom in DOMAINS:
        for D in DIRECTIONS:
            st = results.get((dom, D))
            if st is None:
                continue
            print(f"  {dom+'/'+D:<12}{st['var_log_v']:>12.3f}"
                  f"{100*st['frac_var_from_u']:>8.0f}%"
                  f"{100*st['frac_var_from_chi']:>8.0f}%"
                  f"{100*st['frac_var_from_cov']:>8.0f}%"
                  f"   {st['slope_u_vs_chi']:+.3f}")


# ----------------------------------------------------------------------
# PLOT
# ----------------------------------------------------------------------

def _tight(ax, H, xr, yr, pad=0.12):
    filled = np.asarray(H) > 0
    if not filled.any():
        return
    ny, nx = filled.shape
    rows = np.where(filled.any(axis=1))[0]
    cols = np.where(filled.any(axis=0))[0]
    dy = (yr[1] - yr[0]) / ny
    dxx = (xr[1] - xr[0]) / nx
    y0 = yr[0] + rows[0] * dy
    y1 = yr[0] + (rows[-1] + 1) * dy
    x1 = xr[0] + (cols[-1] + 1) * dxx
    ax.set_ylim(y0 - pad * (y1 - y0), y1 + pad * (y1 - y0))
    ax.set_xlim(xr[0], min(xr[1], x1 + 0.05 * max(x1 - xr[0], 1e-9)))


_STAT = None


def _load_stat(dom, D):
    global _STAT
    if _STAT is None:
        _STAT = {}
        sp = os.path.join(OUTDIR, "vt_summary.csv")
        if os.path.isfile(sp):
            with open(sp) as f:
                hdr = f.readline().strip().split(";")
                for line in f:
                    v = line.rstrip("\n").split(";")
                    if len(v) != len(hdr):
                        continue
                    d = {}
                    for k, x in zip(hdr, v):
                        try:
                            d[k] = float(x)
                        except ValueError:
                            d[k] = x
                    _STAT[(d["domain"], d["direction"])] = d
    return _STAT.get((dom, D))


def make_plots(domains):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    for dom in domains:
        fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0),
                                 squeeze=False)
        drawn = False
        for j, D in enumerate(DIRECTIONS):
            fn = os.path.join(OUTDIR, f"vt_hist_{dom}_{D}.npz")
            if not os.path.isfile(fn):
                axes[0][j].set_visible(False)
                axes[1][j].set_visible(False)
                continue
            z = np.load(fn)
            if z["hist_vchi"].sum() == 0:
                print(f"    [!] {os.path.basename(fn)} e' vuoto "
                      f"(range sbagliato): rilancia --run")
                axes[0][j].set_visible(False)
                axes[1][j].set_visible(False)
                continue
            drawn = True
            cr, vr, ur = z["logchi_range"], z["logv_range"], z["logu_range"]
            st = _load_stat(dom, D)

            ax = axes[0][j]
            H = np.ma.masked_where(z["hist_vchi"].T == 0, z["hist_vchi"].T)
            ax.imshow(H, origin="lower", aspect="auto", norm=LogNorm(),
                      extent=[cr[0], cr[1], vr[0], vr[1]], cmap="viridis")
            if st:
                x = np.linspace(cr[0], cr[1], 50)
                ax.plot(x, np.log10(st["geo_u"]) - x, "w--", lw=1.6,
                        label=r"$v=\langle u\rangle_g/\chi$  (identita')")
                ax.legend(loc="upper right", fontsize=8, framealpha=0.65)
            ax.set_title(f"{dom} - {D}")
            if j == 0:
                ax.set_ylabel(r"$\log_{10}\,v$   [m yr$^{-1}$]")
            _tight(ax, z["hist_vchi"].T, cr, vr)

            ax = axes[1][j]
            H = np.ma.masked_where(z["hist_uchi"].T == 0, z["hist_uchi"].T)
            ax.imshow(H, origin="lower", aspect="auto", norm=LogNorm(),
                      extent=[cr[0], cr[1], ur[0], ur[1]], cmap="magma")
            if st:
                x = np.linspace(cr[0], cr[1], 50)
                b = st["slope_u_vs_chi"]
                a = np.log10(st["geo_u"]) - b * np.log10(st["geo_chi"])
                ax.plot(x, a + b * x, "c-", lw=1.9,
                        label=rf"$u\propto\chi^{{{b:+.2f}}}$")
                ax.legend(loc="upper right", fontsize=8, framealpha=0.65)
            ax.set_xlabel(r"$\log_{10}\chi$")
            if j == 0:
                ax.set_ylabel(r"$\log_{10}\,u$   [m yr$^{-1}$]")
            _tight(ax, z["hist_uchi"].T, cr, ur)

        if not drawn:
            plt.close(fig)
            continue
        fig.suptitle(
            f"{dom}-intensity   |   alto: velocita' di trasporto $v$ "
            f"(legata a $\\chi$ per definizione)   |   "
            f"basso: velocita' lungo la pathline $u$ (relazione deconfusa)",
            fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out = os.path.join(OUTDIR, f"fig_velocity_tortuosity_{dom}.png")
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print("->", out)


# ----------------------------------------------------------------------
# INSPECT
# ----------------------------------------------------------------------

def inspect(root, domains, dx):
    print("=" * 74)
    print("INSPECT  root =", root)
    print("=" * 74)
    for dom in domains:
        dr = os.path.join(root, dom)
        if not os.path.isdir(dr):
            print(f"\n[{dom}] assente")
            continue
        runs = sorted((d for d in os.listdir(dr)
                       if os.path.isdir(os.path.join(dr, d))),
                      key=lambda s: (len(s), s))
        print(f"\n[{dom}] {len(runs)} run")
        if not runs:
            continue
        for D in DIRECTIONS:
            td = os.path.join(dr, runs[0], "transport", D)
            cdir = os.path.join(td, CTRL_SUBDIR)
            if not os.path.isdir(cdir):
                print(f"  [{D}] outcontroldir assente")
                continue
            names = [e.name for e in os.scandir(cdir)
                     if e.is_file() and e.name.startswith(CTRL_PREFIX)]
            print(f"  [{D}] {len(names)} particelle")
            if not names:
                continue
            pl = read_particle(os.path.join(cdir, names[0]),
                               COORD_COL[D], dx)
            if not pl:
                print("       nessun piano riconosciuto (controlla --dx)")
                continue
            items = sorted(pl.items(), key=lambda kv: kv[1][0])
            print(f"       {names[0]}: {len(pl)} piani  "
                  f"idx(ordine di valle)={[k for k, _ in items]}")
            seg = segments_from_particle(pl, dx)
            if seg:
                print(f"       {len(seg)} segmenti; primi 3 "
                      f"(chi, v, u): " +
                      "  ".join(f"({s[0]:.2f}, {s[1]:.2e}, {s[2]:.2e})"
                                for s in seg[:3]))
    print("\n" + "=" * 74)


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--hist-only", action="store_true",
                    help="rigenera SOLO i vt_hist_*.npz, sugli stessi run "
                         "elencati in vt_per_run.csv. Non scrive alcun CSV: "
                         "serve a recuperare gli istogrammi persi senza "
                         "rifare l'analisi ne' degradare le statistiche.")
    ap.add_argument("--profile", action="store_true",
                    help="profilo di tortuosita' locale e cumulata vs distanza")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--domains", default=",".join(DOMAINS))
    ap.add_argument("--dx", type=float, default=DX_DEFAULT)
    ap.add_argument("--max-runs", type=int, default=5,
                    help="run per dominio/direzione (default 5; 0 = tutti)")
    ap.add_argument("--max-particles", type=int, default=None,
                    help="sottocampiona le particelle per run (default: tutte)")
    ap.add_argument("--outdir", default="vt_out",
                    help="cartella di output (default vt_out)")
    ap.add_argument("--paired", action="store_true",
                    help="usa solo i run che hanno TUTTE e tre le direzioni")
    ap.add_argument("--save-samples", action="store_true",
                    help="salva anche i campioni grezzi (file grossi)")
    ap.add_argument("--range-chi", default=None, metavar="LO,HI",
                    help="range in log10(chi) imposto a TUTTE le celle. "
                         "Senza questo ogni cella usa il proprio, e la figura "
                         "a assi condivisi mostra i bordi come fossero dati.")
    ap.add_argument("--range-u", default=None, metavar="LO,HI",
                    help="range in log10(u) imposto a tutte le celle")
    ap.add_argument("--range-v", default=None, metavar="LO,HI",
                    help="range in log10(v) imposto a tutte le celle")
    a = ap.parse_args()

    doms = [d.strip() for d in a.domains.split(",") if d.strip()]
    global OUTDIR, FIX_RC, FIX_RV, FIX_RU
    OUTDIR = a.outdir

    def _rng(s, name):
        if not s:
            return None
        try:
            lo, hi = (float(x) for x in s.split(","))
        except ValueError:
            raise SystemExit(f"--range-{name} vuole due numeri: LO,HI")
        if hi <= lo:
            raise SystemExit(f"--range-{name}: HI deve essere > LO")
        return (lo, hi)

    FIX_RC = _rng(a.range_chi, "chi")
    FIX_RU = _rng(a.range_u, "u")
    FIX_RV = _rng(a.range_v, "v")
    if FIX_RC or FIX_RU or FIX_RV:
        print(f"[range imposti] chi={FIX_RC}  u={FIX_RU}  v={FIX_RV}")
    np.random.seed(0)

    if a.inspect:
        inspect(a.root, doms, a.dx)
    elif a.run:
        run_all(a.root, doms, a.dx, a.max_runs or None, a.max_particles,
                a.save_samples, a.paired)
    elif a.hist_only:
        run_hist_only(a.root, doms, a.dx, a.max_particles)
    elif a.profile:
        run_profiles(a.root, doms, a.dx, a.max_runs or None, a.max_particles)
    elif a.plot:
        make_plots(doms)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
