#!/usr/bin/env python3

import os, csv, time, argparse, sys
from collections import Counter
import networkx as nx

RUNS_ROOT    = "/hpc/scratch/filippo.porta/runs_10k"
METRICS_ROOT = "/hpc/home/filippo.porta/metrics/fixed_10k"

DOMAINS    = ["low", "medium", "high"]
DIRECTIONS = ["x", "y", "z"]
FACTORS    = [1.2, 1.3, 1.5, 1.75, 2.0]
REFERENCE_FACTOR = 1.5
MAX_PATHS  = 20000

BOUNDARY = {"x": ("left.dat","right.dat"),
            "y": ("front.dat","back.dat"),
            "z": ("bottom.dat","top.dat")}


# ---------------------------------------------------------------- utils
def read_ids(path):
    ids = set()
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            s = line.split()
            if s:
                ids.add(int(s[0]))
    return ids


def find_mother_dir(run_path):
    """intensity<N>/dfnGen_output: la rete MADRE, stessa gen_dir dello
    script di generazione. NON output_backbone/dfnGen_output."""
    for name in sorted(os.listdir(run_path)):
        if name.startswith("intensity") and not name.endswith("_backup"):
            p = os.path.join(run_path, name, "dfnGen_output")
            if os.path.isdir(p):
                return p
    return None


def build_graph(run_path, direction):
    """Grafo della rete madre, ristretto al backbone direzionale,
    con s/t, ridotto al 2-core (che in queste reti non rimuove nulla)."""
    dfn = find_mother_dir(run_path)
    if dfn is None:
        return None
    src_name, snk_name = BOUNDARY[direction]
    conn   = os.path.join(dfn, "connectivity.dat")
    source = os.path.join(dfn, src_name)
    target = os.path.join(dfn, snk_name)
    bbfile = os.path.join(run_path, f"backbone_{direction.upper()}.dat")
    for p in (conn, source, target, bbfile):
        if not os.path.exists(p):
            return None

    G = nx.Graph()
    with open(conn, encoding="utf-8-sig") as f:
        for i, line in enumerate(f):
            u = i + 1
            for v in (int(x) for x in line.split()):
                G.add_edge(u, v)

    bb = read_ids(bbfile)
    if bb and max(bb) > G.number_of_nodes():
        return None                      # numerazioni incompatibili

    G.add_node("s"); G.add_node("t")
    for n in read_ids(source):
        if n in G: G.add_edge("s", n)
    for n in read_ids(target):
        if n in G: G.add_edge("t", n)

    H = G.subgraph(bb | {"s","t"}).copy()
    changed = True
    while changed:
        changed = False
        rm = [n for n in H.nodes() if n not in ("s","t") and H.degree(n) < 2]
        if rm:
            H.remove_nodes_from(rm); changed = True
    return H


# ------------------------------------------------------- centralita'
def length_ratio_centrality(G, factor, max_paths=MAX_PATHS, max_seconds=None):
    """Ritorna (centrality, n_paths, shortest_len, capped, elapsed).
    capped = 1 se fermato da max_paths, 2 se fermato dal limite di tempo."""
    if not (G.has_node("s") and G.has_node("t")) or not nx.has_path(G,"s","t"):
        return None, 0, None, 0, 0.0
    t0 = time.time()
    paths, shortest_len, capped = [], None, 0
    for p in nx.shortest_simple_paths(G, "s", "t"):
        if shortest_len is None:
            shortest_len = len(p)
        if len(p) > factor * shortest_len:
            break
        paths.append(p)
        if len(paths) >= max_paths:
            capped = 1
            break
        if max_seconds and (len(paths) % 200 == 0) and (time.time()-t0) > max_seconds:
            capped = 2
            break
    counts = Counter()
    for p in paths:
        for node in p[1:-1]:
            counts[node] += 1
    n = len(paths)
    cent = {v: counts[v]/n for v in G.nodes if v not in ("s","t")} if n else {}
    return cent, n, shortest_len, capped, time.time()-t0


def spearman(a, b):
    """Correlazione di rango, senza scipy. a e b: liste appaiate."""
    def rank(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0]*len(x); i = 0
        while i < len(order):
            j = i
            while j+1 < len(order) and x[order[j+1]] == x[order[i]]:
                j += 1
            avg = (i + j)/2.0 + 1
            for k in range(i, j+1):
                r[order[k]] = avg
            i = j + 1
        return r
    if len(a) < 3:
        return float("nan")
    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra)/len(ra), sum(rb)/len(rb)
    num = sum((x-ma)*(y-mb) for x, y in zip(ra, rb))
    da  = sum((x-ma)**2 for x in ra) ** 0.5
    db  = sum((y-mb)**2 for y in rb) ** 0.5
    return num/(da*db) if da and db else float("nan")


def topo_metrics(H):
    """Grado medio (solo archi frattura-frattura) e node connectivity s-t."""
    F = H.subgraph([n for n in H.nodes if n not in ("s","t")])
    degs = [d for _, d in F.degree()]
    k_mean = sum(degs)/len(degs) if degs else float("nan")
    try:
        nc = nx.node_connectivity(H, "s", "t")
    except Exception:
        nc = float("nan")
    return len(F), k_mean, nc


def list_runs(domain):
    base = os.path.join(RUNS_ROOT, domain)
    if not os.path.isdir(base):
        return []
    out = []
    for name in sorted(os.listdir(base)):
        p = os.path.join(base, name)
        if os.path.isdir(p) and os.path.isdir(os.path.join(p, "flow")):
            try:
                out.append((int(name), p))
            except ValueError:
                pass
    return sorted(out)


def usable_runs(domain, paired):
    """
    Run utilizzabili. Con paired=True tiene solo quelli in cui TUTTE e tre
    le direzioni danno un grafo costruibile e connesso s-t.

    Serve perche' la Tabella 1 mette X, Y e Z fianco a fianco: se una
    direzione salta in alcuni run, le tre colonne finiscono per descrivere
    reti diverse e il confronto direzionale non e' piu' appaiato.
    """
    runs = list_runs(domain)
    if not paired:
        return runs
    ok = []
    for rid, rpath in runs:
        good = True
        for d in DIRECTIONS:
            H = build_graph(rpath, d)
            if H is None or not (H.has_node("s") and H.has_node("t")) \
                    or not nx.has_path(H, "s", "t"):
                good = False
                break
        if good:
            ok.append((rid, rpath))
    print(f"[{domain}] {len(ok)} run su {len(runs)} hanno tutte e tre "
          f"le direzioni connesse")
    return ok


def _mstd(v):
    if not v:
        return float("nan"), float("nan")
    m = sum(v)/len(v)
    if len(v) < 2:
        return m, 0.0
    return m, (sum((x-m)**2 for x in v)/(len(v)-1))**0.5


# ------------------------------------------------------ sensitivity
def run_sensitivity(runs_per_domain, out_csv, max_seconds, paired=False):
    rows = []
    for dom in DOMAINS:
        runs = usable_runs(dom, paired)[:runs_per_domain]
        print(f"\n=== {dom.upper()}  ({len(runs)} run) ===")
        for rid, rpath in runs:
            for d in DIRECTIONS:
                H = build_graph(rpath, d)
                if H is None:
                    print(f"  run{rid}/{d.upper()}: grafo non costruibile")
                    continue
                nfrac, k_mean, nc = topo_metrics(H)

                # --- riferimento a 1.5, calcolato PER PRIMO ---
                ref, nref, slen, cref, eref = length_ratio_centrality(
                    H, REFERENCE_FACTOR, max_seconds=max_seconds)
                if not ref:
                    print(f"  run{rid}/{d.upper()}: nessun path s-t")
                    continue

                for fac in FACTORS:
                    if fac == REFERENCE_FACTOR:
                        cent, npaths, sl, capped, el = ref, nref, slen, cref, eref
                    else:
                        cent, npaths, sl, capped, el = length_ratio_centrality(
                            H, fac, max_seconds=max_seconds)
                    if not cent or npaths == 0:
                        continue
                    keys = sorted(cent)
                    rho = spearman([cent[k] for k in keys],
                                   [ref[k] for k in keys])
                    vals = [cent[k] for k in keys]
                    rows.append(dict(domain=dom, run=rid, direction=d.upper(),
                        factor=fac, n_fractures=nfrac,
                        Lmin_vertices=sl,
                        Lmin_fractures=(sl-2 if sl else ""),
                        max_len_vertices=int(fac*sl) if sl else "",
                        max_len_fractures=(int(fac*sl)-2 if sl else ""),
                        n_paths=npaths, capped=capped,
                        cpu_time_s=round(el,3),
                        pct_Ci_below_0p1=round(100*sum(1 for v in vals if v<0.1)/len(vals),2),
                        pct_Ci_equal_1=round(100*sum(1 for v in vals if v>=0.999)/len(vals),2),
                        spearman_vs_1p5=round(rho,4) if rho==rho else ""))
                    flag = {0:"", 1:" [CAP PATHS]", 2:" [CAP TEMPO]"}[capped]
                    print(f"  run{rid}/{d.upper()} f={fac:<5} "
                          f"Lmin={sl:<3}(<={int(fac*sl)}) paths={npaths:<6} "
                          f"t={el:6.2f}s rho={rho:.4f}{flag}")

    if not rows:
        print("nessun risultato"); return
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=";")
        w.writeheader(); w.writerows(rows)
    print(f"\n-> {out_csv}  ({len(rows)} righe)")

    print("\n=== RIEPILOGO per fattore ===")
    print("  Le due colonne a destra sono l'argomento vero: se restano stabili,")
    print("  i risultati di Tabella 1 non dipendono dalla scelta della soglia.")
    print(f"{'factor':>7}{'paths medio':>13}{'capped':>8}{'rho medio':>11}"
          f"{'t [s]':>9}{'% Ci<0.1':>20}{'% Ci=1':>17}")
    for fac in FACTORS:
        sub = [r for r in rows if r['factor'] == fac]
        if not sub: continue
        rhos = [r['spearman_vs_1p5'] for r in sub if r['spearman_vs_1p5'] != ""]
        lo_m, lo_s = _mstd([r['pct_Ci_below_0p1'] for r in sub])
        eq_m, eq_s = _mstd([r['pct_Ci_equal_1'] for r in sub])
        print(f"{fac:>7}{sum(r['n_paths'] for r in sub)/len(sub):>13.0f}"
              f"{sum(1 for r in sub if r['capped']):>8}"
              f"{(sum(rhos)/len(rhos) if rhos else float('nan')):>11.4f}"
              f"{sum(r['cpu_time_s'] for r in sub)/len(sub):>9.1f}"
              f"{lo_m:>14.1f} +/-{lo_s:<5.1f}"
              f"{eq_m:>11.2f} +/-{eq_s:<5.2f}")

    print("\n=== L_min per dominio e direzione (fratture attraversate) ===")
    for dom in DOMAINS:
        out = []
        for dr in ["X","Y","Z"]:
            v = [r['Lmin_fractures'] for r in rows
                 if r['domain']==dom and r['direction']==dr and r['factor']==REFERENCE_FACTOR]
            out.append(f"{dr}: {sum(v)/len(v):.1f}" if v else f"{dr}: -")
        print(f"  {dom:8s} " + "   ".join(out))


# ------------------------------------------------------------ final
def run_final(out_csv, max_seconds=None, paired=False):
    rows = []
    for dom in DOMAINS:
        runs = usable_runs(dom, paired)
        print(f"\n=== {dom.upper()}  ({len(runs)} run) ===")
        for rid, rpath in runs:
            for d in DIRECTIONS:
                H = build_graph(rpath, d)
                if H is None:
                    continue
                nfrac, k_mean, nc = topo_metrics(H)
                cent, npaths, slen, capped, el = \
                    length_ratio_centrality(H, REFERENCE_FACTOR, max_seconds=max_seconds)
                if not cent:
                    continue
                vals = list(cent.values())
                rows.append(dict(domain=dom, run=rid, direction=d.upper(),
                    n_fractures=nfrac, mean_degree=round(k_mean,4),
                    node_connectivity=nc,
                    shortest_path_len_vertices=slen, n_paths=npaths,
                    capped=int(capped),
                    pct_Ci_below_0p1=round(100*sum(1 for v in vals if v<0.1)/len(vals),2),
                    pct_Ci_equal_1=round(100*sum(1 for v in vals if v>=0.999)/len(vals),2)))
                # centrality per frattura
                od = os.path.join(METRICS_ROOT, dom, f"run_{rid}")
                os.makedirs(od, exist_ok=True)
                mode = "a" if d != DIRECTIONS[0] else "w"
                with open(os.path.join(od,"centrality.csv"), mode, newline="") as f:
                    w = csv.writer(f)
                    if mode == "w": w.writerow(["direction","fracture","C_i"])
                    for k in sorted(cent): w.writerow([d, k, f"{cent[k]:.6f}"])
                print(f"  run{rid}/{d.upper()}: n={nfrac} <k>={k_mean:.3f} "
                      f"nc={nc} paths={npaths} cap={int(capped)}")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=";")
        w.writeheader(); w.writerows(rows)
    print(f"\n-> {out_csv}  ({len(rows)} righe)")

    tab = [["domain","direction","n_real","N_fractures","mean_degree",
            "nc_mean","pct_Ci_equal_1","pct_Ci_below_0p1"]]
    print("\n" + "="*80)
    print("PRONTO PER LA TABELLA 1   (media +/- sd sulle realizzazioni)")
    print("="*80)
    print(f"  {'dom':<7}{'dir':<5}{'n':>4}{'N fratture':>17}{'<k>':>17}"
          f"{'nc':>16}{'% Ci=1':>9}{'% Ci<0.1':>10}")
    for dom in DOMAINS:
        for dr in ("X","Y","Z"):
            sub = [r for r in rows if r['domain']==dom and r['direction']==dr]
            if not sub: continue
            nf = _mstd([r['n_fractures'] for r in sub])
            kk = _mstd([r['mean_degree'] for r in sub])
            nc = _mstd([r['node_connectivity'] for r in sub])
            e1 = _mstd([r['pct_Ci_equal_1'] for r in sub])[0]
            lo = _mstd([r['pct_Ci_below_0p1'] for r in sub])[0]
            print(f"  {dom:<7}{dr:<5}{len(sub):>4}"
                  f"{nf[0]:>11.1f} +/-{nf[1]:<5.1f}"
                  f"{kk[0]:>11.3f} +/-{kk[1]:<5.3f}"
                  f"{nc[0]:>10.2f} +/-{nc[1]:<5.2f}"
                  f"{e1:>9.2f}{lo:>10.2f}")
            tab.append([dom, dr, len(sub), f"{nf[0]:.1f} +/- {nf[1]:.1f}",
                        f"{kk[0]:.3f} +/- {kk[1]:.3f}",
                        f"{nc[0]:.2f} +/- {nc[1]:.2f}",
                        f"{e1:.2f}", f"{lo:.2f}"])
    tpath = out_csv.replace(".csv", "_table1.csv")
    with open(tpath, "w", newline="") as f:
        f.write("\n".join(";".join(map(str,r)) for r in tab) + "\n")
    print(f"\n-> {tpath}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--runs-per-domain", type=int, default=5)
    ap.add_argument("--max-seconds", type=float, default=0,
                    help="limite di tempo per singola enumerazione. "
                         "Default 0 = nessun limite: l'enumerazione si ferma "
                         "solo sulla soglia di lunghezza o su MAX_PATHS.")
    ap.add_argument("--paired", action="store_true",
                    help="solo i run con tutte e tre le direzioni connesse")
    ap.add_argument("--runs-root", default=RUNS_ROOT)
    ap.add_argument("--metrics-root", default=METRICS_ROOT)
    a = ap.parse_args()
    RUNS_ROOT, METRICS_ROOT = a.runs_root, a.metrics_root
    if not (a.sensitivity or a.final):
        ap.error("scegli --sensitivity oppure --final")
    os.makedirs(METRICS_ROOT, exist_ok=True)
    if a.sensitivity:
        run_sensitivity(a.runs_per_domain,
                        os.path.join(METRICS_ROOT,"centrality_sensitivity.csv"),
                        a.max_seconds or None, a.paired)
    if a.final:
        run_final(os.path.join(METRICS_ROOT,"topology_summary.csv"),
                  a.max_seconds or None, a.paired)
