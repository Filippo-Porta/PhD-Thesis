#!/usr/bin/env python3


import os
import re
import sys
import csv
import argparse
import traceback
import numpy as np
import networkx as nx

RUNS_ROOT    = "/hpc/scratch/filippo.porta/runs_10k"
METRICS_ROOT = "/hpc/home/filippo.porta/metrics/fixed_10k"

DOMAIN_LIST = ["low", "medium", "high"]
DIR_LIST    = ["X", "Y", "Z"]

# file di connettivita' del grafo (ID ORIGINALI della rete madre)
BOUNDARY = {
    "X": ("left.dat",   "right.dat"),
    "Y": ("front.dat",  "back.dat"),
    "Z": ("bottom.dat", "top.dat"),
}

# facce della mesh (.ex). Piu' nomi possibili: la convenzione cambia fra
# versioni di dfnWorks. Il primo che esiste vince.
BOUNDARY_EX = {
    "X": (["boundary_left_w.ex",  "pboundary_left_w.ex"],
          ["boundary_right_e.ex", "pboundary_right_e.ex"]),
    "Y": (["boundary_front_n.ex", "pboundary_front_n.ex",
           "boundary_front_s.ex", "boundary_front.ex"],
          ["boundary_back_s.ex",  "pboundary_back_s.ex",
           "boundary_back_n.ex",  "boundary_back.ex"]),
    "Z": (["boundary_bottom.ex",  "pboundary_bottom.ex"],
          ["boundary_top.ex",     "pboundary_top.ex"]),
}


class RunError(Exception):
    pass


def fail(msg):
    raise RunError(msg)


def first_existing_dir(cands, what):
    for p in cands:
        if os.path.isdir(p):
            return p
    fail(f"{what}: nessuna di queste cartelle esiste:\n      "
         + "\n      ".join(cands))


def first_existing_file(cands, what):
    for p in cands:
        if os.path.isfile(p):
            return p
    fail(f"{what}: nessuno di questi file esiste:\n      "
         + "\n      ".join(cands))


# ==================================================================
# RISOLUZIONE DEI PERCORSI
# ==================================================================
def find_mother_dir(run_path):
    """dfnGen_output della rete MADRE (intensity<N>/dfnGen_output), cioe' la
    stessa gen_dir usata dallo script di generazione per costruire i
    backbone direzionali. NON output_backbone/dfnGen_output, che contiene
    la rete potata dell'ULTIMA direzione girata, rinumerata da 1."""
    cands = []
    for name in sorted(os.listdir(run_path)):
        if name.endswith("_backup"):
            continue
        p = os.path.join(run_path, name, "dfnGen_output")
        if name.startswith("intensity") and os.path.isdir(p):
            cands.append(p)
    if not cands:
        fail(f"nessuna cartella intensity*/dfnGen_output in {run_path}: "
             f"non e' possibile ricostruire la rete madre")
    if len(cands) > 1:
        fail(f"piu' reti madri candidate: {cands}")
    return cands[0]


def resolve_paths(run_path, direction):
    D, d = direction.upper(), direction.lower()

    flow_dir = first_existing_dir([
        os.path.join(run_path, "flow", D),
        os.path.join(run_path, "flow", d),
        os.path.join(run_path, f"output_backbone_{D}"),
        os.path.join(run_path, "output_backbone"),
    ], "cartella con la soluzione di flusso")

    base = os.path.basename(flow_dir).upper()
    trusted = (base == D) or base.endswith("_" + D)

    # grafo e raggi: SEMPRE dalla rete madre
    dfn_dir = find_mother_dir(run_path)

    radii_file = None
    for c in ("radii_Final.dat", "radii.dat"):
        p = os.path.join(dfn_dir, c)
        if os.path.isfile(p):
            radii_file = p
            break

    # aperture: solo se gia' indicizzato come la mesh della direzione.
    # output_backbone/aperture.dat e' della rete potata di un'ALTRA
    # direzione e rinumerato: non va mai usato.
    aperture_file = None
    p = os.path.join(flow_dir, "aperture.dat")
    if os.path.isfile(p):
        aperture_file = p

    poly_info = os.path.join(flow_dir, "poly_info.dat")
    poly_info = poly_info if os.path.isfile(poly_info) else None

    backbone_file = first_existing_file([
        os.path.join(run_path, f"backbone_{D}.dat"),
        os.path.join(flow_dir, f"backbone_{D}.dat"),
    ], f"backbone_{D}.dat")

    return {"flow_dir": flow_dir, "trusted": trusted,
            "aperture_file": aperture_file, "poly_info": poly_info,
            "radii_file": radii_file, "dfn_dir": dfn_dir,
            "backbone_file": backbone_file}


SOLUTION_PATTERNS = ["dfn_steady_pressure_{d}.in",
                     "dfn_steady_pressure_{d}.out",
                     "output_backbone_effective_perm_{d}.txt",
                     "PTDFN_control_{D}_patched.dat",
                     "PTDFN_control_{D}.dat"]


def detect_solution_direction(flow_dir):
    files = set(os.listdir(flow_dir))
    found = {dd.upper() for dd in ("x", "y", "z")
             for pat in SOLUTION_PATTERNS
             if pat.format(d=dd, D=dd.upper()) in files}
    if len(found) == 1:
        return found.pop()
    return None


def find_runs(runs_root, domain):
    base = os.path.join(runs_root, domain)
    if not os.path.isdir(base):
        fail(f"cartella dominio inesistente: {base}")
    out = []
    for name in sorted(os.listdir(base)):
        p = os.path.join(base, name)
        if not os.path.isdir(p):
            continue
        if os.path.isdir(os.path.join(p, "flow")) or \
           os.path.isdir(os.path.join(p, "output_backbone")):
            m = re.search(r"(\d+)", name)
            out.append((int(m.group(1)) if m else name, p))
    if not out:
        fail(f"nessuna run in {base} (contenuto: {sorted(os.listdir(base))[:20]})")
    out.sort(key=lambda t: (isinstance(t[0], str), t[0]))
    return out


def resolve_run_path(runs_root, domain, run):
    base = os.path.join(runs_root, domain)
    for cand in (str(run), f"run_{run}", f"run{run}"):
        p = os.path.join(base, cand)
        if os.path.isdir(p):
            return p
    for rid, p in find_runs(runs_root, domain):
        if rid == run:
            return p
    fail(f"run {run} non trovata in {base}")


# ==================================================================
# LETTURE
# ==================================================================
def get_cell_info(flow_dir):
    with open(os.path.join(flow_dir, "full_mesh_vol_area.uge")) as f:
        num_nodes = int(f.readline().split()[1])
        volumes = [float(f.readline().split()[4]) for _ in range(num_nodes)]
        num_conn = int(f.readline().split()[1])
    return num_nodes, num_conn, volumes


def get_cell_fractures(flow_dir):
    return np.loadtxt(os.path.join(flow_dir, "materialid.dat"),
                      dtype=int, skiprows=3).flatten()


def get_general(flow_dir):
    p = np.loadtxt(os.path.join(flow_dir, "params.txt")).flatten()
    return int(p[0]), p[4] * p[5] * p[6]


def read_aperture_file(aperture_file):
    """Formato dfnWorks: header 'aperture' + righe '-7 0 0 valore'
    -> frattura LOCALE 1. Fallback su formato a colonna singola."""
    rows = []
    with open(aperture_file, encoding="utf-8-sig") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                continue
    if not rows:
        fail(f"nessun dato numerico in {aperture_file}")
    raw = np.array(rows)
    if raw.shape[1] == 1:
        raw = raw.flatten()
    if raw.ndim == 1:
        return raw.flatten()
    b = np.zeros(len(raw))
    for row in raw:
        fid = -int(row[0]) - 6
        if not (1 <= fid <= len(raw)):
            fail(f"material id inatteso in {aperture_file}: {int(row[0])}")
        b[fid - 1] = row[3]
    return b


def read_radii_final(path):
    """radii_Final.dat della rete madre: righe di commento iniziali, poi una
    riga per frattura ORIGINALE. La prima colonna numerica e' il raggio.
    Restituisce un array 1-based (indice 0 non usato)."""
    vals = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            try:
                vals.append(float(parts[0]))
            except ValueError:
                continue          # riga di intestazione
    if not vals:
        fail(f"nessun raggio leggibile in {path}")
    r = np.empty(len(vals) + 1)
    r[0] = np.nan
    r[1:] = vals
    return r


def apertures_from_radii(P, num_frac_local, local_to_orig, coeff, expo, verbose):
    """b = coeff * r^expo, con r preso dalla rete madre tramite la mappatura
    locale -> originale di backbone_<DIR>.dat."""
    if not P["radii_file"]:
        fail(f"radii_Final.dat non trovato in {P['dfn_dir']}")
    r_orig = read_radii_final(P["radii_file"])

    b = np.zeros(num_frac_local)
    r_used = np.zeros(num_frac_local)
    for lid in range(1, num_frac_local + 1):
        oid = local_to_orig.get(lid)
        if oid is None:
            fail(f"frattura locale {lid} assente da "
                 f"{os.path.basename(P['backbone_file'])}")
        if not (1 <= oid < len(r_orig)):
            fail(f"ID originale {oid} fuori range in "
                 f"{os.path.basename(P['radii_file'])} "
                 f"({len(r_orig)-1} fratture)")
        r_used[lid - 1] = r_orig[oid]
        b[lid - 1] = coeff * r_orig[oid] ** expo

    if verbose:
        print(f"  aperture   : calcolate da "
              f"{os.path.basename(P['radii_file'])} via "
              f"{os.path.basename(P['backbone_file'])}")
        print(f"               r = {r_used.min():.4f} - {r_used.max():.4f} m")
        print(f"               b = {coeff:g} * r^{expo:g}"
              f"  ->  {b.min():.5e} - {b.max():.5e} m")
    return b


def get_apertures(P, num_frac_local, local_to_orig, coeff, expo, verbose):
    """Ordine di preferenza:
       a) aperture.dat gia' indicizzato come la mesh della direzione
       b) aperture.dat della rete di riferimento, mappato con backbone_<DIR>.dat
       c) ricalcolo da radii_Final.dat, mappato con backbone_<DIR>.dat
    Quando sono disponibili sia (b) sia (c), i due vengono confrontati."""
    b_file = read_aperture_file(P["aperture_file"]) if P["aperture_file"] else None

    # --- (a) corrispondenza diretta ---
    if b_file is not None and len(b_file) == num_frac_local:
        if verbose:
            print(f"  aperture   : {P['aperture_file']}  "
                  f"(indicizzato come la mesh, {len(b_file)} fratture)")
        return b_file

    oids = [local_to_orig[k] for k in range(1, num_frac_local + 1)]
    max_oid = max(oids)

    # --- (c) dai raggi, serve anche come verifica di (b) ---
    b_rad = None
    if P["radii_file"]:
        r_all = read_radii_final(P["radii_file"])
        if max_oid < len(r_all):
            r_used = np.array([r_all[o] for o in oids])
            b_rad = coeff * r_used ** expo

    # --- (b) aperture.dat mappato ---
    b_map = None
    if b_file is not None and max_oid <= len(b_file):
        b_map = np.array([b_file[o - 1] for o in oids])

    if b_map is not None and b_rad is not None:
        rel = np.max(np.abs(b_map - b_rad) / b_map)
        if verbose:
            print(f"  aperture   : {os.path.basename(P['aperture_file'])} "
                  f"mappato con {os.path.basename(P['backbone_file'])}")
            print(f"               verifica con {coeff:g}*r^{expo:g}: "
                  f"scarto relativo max = {rel:.2e}")
        if rel > 1e-6:
            fail(f"aperture.dat e {coeff:g}*r^{expo:g} non coincidono "
                 f"(scarto max {rel:.3e}): la mappatura o la legge "
                 f"apertura-raggio non sono quelle attese")
        return b_map

    if b_map is not None:
        if verbose:
            print(f"  aperture   : {os.path.basename(P['aperture_file'])} "
                  f"mappato con {os.path.basename(P['backbone_file'])} "
                  f"(nessun radii_Final.dat per la verifica)")
        return b_map

    if b_rad is not None:
        if verbose:
            print(f"  aperture   : calcolate da "
                  f"{os.path.basename(P['radii_file'])} via "
                  f"{os.path.basename(P['backbone_file'])}")
            print(f"               b = {coeff:g}*r^{expo:g}  ->  "
                  f"{b_rad.min():.5e} - {b_rad.max():.5e} m")
        return b_rad

    fail(f"impossibile ricavare le aperture: aperture.dat "
         f"{'assente' if b_file is None else f'con {len(b_file)} fratture'}, "
         f"radii_Final.dat "
         f"{'assente' if not P['radii_file'] else 'incompatibile'}, "
         f"ID massimo richiesto dalla mappatura = {max_oid}")


def load_darcy_vel(flow_dir):
    return np.loadtxt(os.path.join(flow_dir, "darcyvel.dat"))


def find_ex_file(flow_dir, cands):
    for n in cands:
        p = os.path.join(flow_dir, n)
        if os.path.exists(p):
            return p
    avail = sorted(f for f in os.listdir(flow_dir) if f.endswith(".ex"))
    fail(f"file di bordo non trovato fra {cands}\n"
         f"    .ex presenti: {avail if avail else 'NESSUNO'}")


def load_boundary_nodes(path):
    a = np.loadtxt(path, skiprows=1)
    return (a if a.ndim == 1 else a[:, 0]).astype(int)


# ==================================================================
# FORMULE
# ==================================================================
class Cell:
    __slots__ = ("frac", "boundary", "sa")
    def __init__(self, frac, boundary, sa):
        self.frac, self.boundary, self.sa = frac, boundary, sa


class Frac:
    __slots__ = ("sa", "Q")
    def __init__(self):
        self.sa, self.Q = 0.0, 0.0


def make_cell_array(num_nodes, b, frac_id, volumes):
    return [Cell(int(frac_id[i]), False,
                 2.0 * volumes[i] / b[int(frac_id[i]) - 1])
            for i in range(num_nodes)]


def mark_boundary_cells(cells, flow_dir, direction, verbose):
    in_c, out_c = BOUNDARY_EX[direction]
    inflow, outflow = find_ex_file(flow_dir, in_c), find_ex_file(flow_dir, out_c)
    n = 0
    for path in (inflow, outflow):
        for node in load_boundary_nodes(path):
            if not (1 <= node <= len(cells)):
                fail(f"nodo fuori range in {os.path.basename(path)}: {node}")
            if not cells[node - 1].boundary:
                n += 1
            cells[node - 1].boundary = True
    if verbose:
        print(f"  facce      : {os.path.basename(inflow)} / "
              f"{os.path.basename(outflow)}  ({n} celle su {len(cells)})")
    return cells


def make_frac_array(num_frac_local, cells):
    fracs = [Frac() for _ in range(num_frac_local)]
    for c in cells:
        fracs[c.frac - 1].sa += c.sa
    return fracs


def compute_Q(num_conn, fracs, cells, darcyvel, use_boundary):
    for i in range(num_conn):
        i1 = int(darcyvel[i, 0]) - 1
        i2 = int(darcyvel[i, 1]) - 1
        Q = darcyvel[i, 2] * darcyvel[i, 4]
        if (cells[i1].frac != cells[i2].frac) or \
           (use_boundary and (cells[i1].boundary or cells[i2].boundary)):
            fracs[cells[i1].frac - 1].Q += 0.5 * abs(Q)
            fracs[cells[i2].frac - 1].Q += 0.5 * abs(Q)
    return fracs


def p32_of(sa, V):
    return sum(sa) / V


def dq_of(sa, Q, V):
    top = sum(s * q for s, q in zip(sa, Q))
    bot = sum(s * q ** 2 for s, q in zip(sa, Q))
    return 0.0 if bot == 0 else (1.0 / V) * top ** 2 / bot


# ==================================================================
# 2-CORE
# ==================================================================
def read_ids(path):
    ids = set()
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            s = line.split()
            if s:
                ids.add(int(s[0]))
    return ids


def load_local_to_original(backbone_file):
    m = {}
    with open(backbone_file) as f:
        for lid, line in enumerate(f, start=1):
            s = line.strip()
            if s:
                m[lid] = int(s)
    return m


def two_core_original_ids(dfn_dir, backbone_file, direction):
    src_name, snk_name = BOUNDARY[direction]
    G = nx.Graph()
    with open(os.path.join(dfn_dir, "connectivity.dat"), encoding="utf-8-sig") as f:
        for i, line in enumerate(f):
            u = i + 1
            for v in (int(x) for x in line.split()):
                G.add_edge(u, v)
    G.add_node("s"); G.add_node("t")
    for n in read_ids(os.path.join(dfn_dir, src_name)):
        if n in G: G.add_edge("s", n)
    for n in read_ids(os.path.join(dfn_dir, snk_name)):
        if n in G: G.add_edge("t", n)

    H = G.subgraph(read_ids(backbone_file) | {"s", "t"}).copy()
    changed = True
    while changed:
        changed = False
        rm = [n for n in H.nodes() if n not in ("s", "t") and H.degree(n) < 2]
        if rm:
            H.remove_nodes_from(rm)
            changed = True
    return {n for n in H.nodes() if isinstance(n, int)}


# ==================================================================
# UN CASO
# ==================================================================
def process_one(domain, run_id, run_path, direction, verbose=True,
                skip_dir_check=False, coeff=5.0e-4, expo=0.5):
    P = resolve_paths(run_path, direction)
    flow_dir = P["flow_dir"]

    if verbose:
        print(f"  flusso     : {flow_dir}")
        print(f"  rete madre : {P['dfn_dir']}")
        print(f"  backbone   : {P['backbone_file']}")

    if not skip_dir_check and not P["trusted"]:
        solved = detect_solution_direction(flow_dir)
        if solved is None:
            fail(f"la cartella {os.path.basename(flow_dir)} non dichiara la "
                 f"direzione e non e' possibile dedurla dai nomi dei file. "
                 f"Usa --skip-dir-check solo se sei certo che sia {direction}.")
        if solved != direction:
            fail(f"INCOERENZA: richiesta --direction {direction} ma il flusso "
                 f"in {os.path.basename(flow_dir)} e' risolto per {solved}.")

    num_nodes, num_conn, volumes = get_cell_info(flow_dir)
    num_frac_local, V = get_general(flow_dir)

    l2o = load_local_to_original(P["backbone_file"])
    if len(l2o) != num_frac_local:
        fail(f"{os.path.basename(P['backbone_file'])} ha {len(l2o)} righe ma "
             f"la mesh di {direction} ha {num_frac_local} fratture: "
             f"mappatura locale->originale non valida")

    b = get_apertures(P, num_frac_local, l2o, coeff, expo, verbose)
    frac_id = get_cell_fractures(flow_dir)
    darcyvel = load_darcy_vel(flow_dir)

    cells = make_cell_array(num_nodes, b, frac_id, volumes)
    cells = mark_boundary_cells(cells, flow_dir, direction, verbose)

    fr_nb = compute_Q(num_conn, make_frac_array(num_frac_local, cells),
                      cells, darcyvel, use_boundary=False)
    fr_b = compute_Q(num_conn, make_frac_array(num_frac_local, cells),
                     cells, darcyvel, use_boundary=True)

    n_mother = sum(1 for line in
                   open(os.path.join(P["dfn_dir"], "connectivity.dat"),
                        encoding="utf-8-sig") if line.strip())
    max_oid = max(l2o.values())
    if max_oid > n_mother:
        fail(f"ID massimo in {os.path.basename(P['backbone_file'])} = "
             f"{max_oid} ma la rete madre ha {n_mother} fratture: "
             f"numerazioni incompatibili")

    core = two_core_original_ids(P["dfn_dir"], P["backbone_file"], direction)
    sa, Qnb, Qb = [], [], []
    for lid in range(1, num_frac_local + 1):
        oid = l2o.get(lid)
        if oid is not None and oid in core:
            sa.append(fr_nb[lid - 1].sa)
            Qnb.append(fr_nb[lid - 1].Q)
            Qb.append(fr_b[lid - 1].Q)

    if not sa:
        fail("intersezione vuota fra 2-core e mesh di flusso")
    if len(sa) != len(core) and verbose:
        print(f"  ATTENZIONE : il 2-core ha {len(core)} fratture ma solo "
              f"{len(sa)} sono nella mesh di flusso ({num_frac_local} "
              f"fratture). P32* e dQ sono calcolati su queste "
              f"{len(sa)}, non sulle {len(core)} della Tabella 1.")

    p32 = p32_of(sa, V)
    dq_nb, dq_b = dq_of(sa, Qnb, V), dq_of(sa, Qb, V)
    delta = 100.0 * (dq_b - dq_nb) / dq_nb if dq_nb > 0 else float("nan")

    return {"domain": domain, "run": run_id, "direction": direction,
            "n_mother": n_mother,
            "n_backbone": num_frac_local,
            "n_2core": len(core), "n_flowmesh": num_frac_local,
            "n_matched": len(sa),
            "frac_2core_in_mesh": len(sa) / len(core),
            "P32_twosided": p32, "P32_onesided": p32 / 2.0,
            "dQ_noboundary": dq_nb, "dQ_boundary": dq_b,
            "ratio_noboundary": dq_nb / p32, "ratio_boundary": dq_b / p32,
            "pct_change_dQ": delta}


def inspect(run_path, direction):
    D = direction.upper()
    print(f"=== INSPECT  {run_path}  direzione {D} ===\n")
    for sub in ("", "flow", f"flow/{D}", "output_backbone",
                "output_backbone/dfnGen_output"):
        p = os.path.join(run_path, sub) if sub else run_path
        print(f"--- {p}")
        if not os.path.isdir(p):
            print("    (inesistente)\n"); continue
        names = sorted(os.listdir(p))
        for i in range(0, len(names), 4):
            print("    " + "  ".join(f"{n:28s}" for n in names[i:i+4]))
        print()

    def nfrac(pt):
        try:
            return int(np.loadtxt(pt).flatten()[0])
        except Exception as e:
            return f"errore: {e}"

    print("--- numero di fratture dichiarato")
    for sub in (f"flow/{D}", "output_backbone"):
        pt = os.path.join(run_path, sub, "params.txt")
        if os.path.isfile(pt):
            print(f"    {sub}/params.txt : {nfrac(pt)}")

    bb = os.path.join(run_path, f"backbone_{D}.dat")
    if os.path.isfile(bb):
        n = sum(1 for l in open(bb) if l.strip())
        print(f"    backbone_{D}.dat      : {n} righe")

    try:
        md = find_mother_dir(run_path)
        nm = sum(1 for l in open(os.path.join(md, "connectivity.dat"),
                                 encoding="utf-8-sig") if l.strip())
        print(f"--- rete madre : {md}\n    {nm} fratture\n")
    except Exception as e:
        md = None
        print(f"--- rete madre non trovata: {e}\n")

    rf = os.path.join(md, "radii_Final.dat") if md else os.path.join(
        run_path, "output_backbone", "dfnGen_output", "radii_Final.dat")
    if os.path.isfile(rf):
        try:
            r = read_radii_final(rf)
            print(f"\n--- radii_Final.dat : {len(r)-1} fratture, "
                  f"r = {np.nanmin(r[1:]):.4f} - {np.nanmax(r[1:]):.4f} m")
            b = 5e-4 * r[1:] ** 0.5
            print(f"    b = 5e-4*sqrt(r) -> {b.min():.5e} - {b.max():.5e} m")
        except Exception as e:
            print(f"\n--- radii_Final.dat non leggibile: {e}")

    for sub, name in ((f"flow/{D}", "poly_info.dat"),
                      ("output_backbone", "aperture.dat")):
        p = os.path.join(run_path, sub, name)
        if not os.path.isfile(p):
            continue
        print(f"\n--- {sub}/{name}: prime 3 righe")
        with open(p) as f:
            for i, line in enumerate(f):
                if i >= 3: break
                print("    " + line.rstrip())
        try:
            arr = np.loadtxt(p)
            if arr.ndim == 1: arr = arr.reshape(1, -1)
            print(f"    shape = {arr.shape}")
            for j in range(arr.shape[1]):
                c = arr[:, j]
                print(f"      col {j}: min={c.min():12.5g}  max={c.max():12.5g}"
                      f"  {'interi' if np.allclose(c, np.round(c)) else ''}")

        except Exception as e:
            print(f"    (non leggibile come tabella: {e})")
    print()


FIELDS = ["domain", "run", "direction", "n_mother", "n_backbone",
          "n_2core", "n_flowmesh",
          "n_matched", "frac_2core_in_mesh",
          "P32_twosided", "P32_onesided", "dQ_noboundary", "dQ_boundary",
          "ratio_noboundary", "ratio_boundary", "pct_change_dQ"]


def print_single(r):
    print("\n--- RISULTATO (2-core) -------------------------------------")
    print(f"fratture nella rete madre  = {r['n_mother']}")
    print(f"fratture nel backbone {r['direction']}    = {r['n_backbone']}")
    print(f"fratture nel 2-core        = {r['n_2core']}   <- come Tabella 1")
    print(f"fratture nella mesh flusso = {r['n_flowmesh']}")
    print(f"usate per P32* e dQ        = {r['n_matched']}"
          f"   ({100*r['frac_2core_in_mesh']:.1f} % del 2-core)")
    print(f"P32* (due facce)     = {r['P32_twosided']:.6f}  m^-1")
    print(f"P32* (una faccia)    = {r['P32_onesided']:.6f}  m^-1\n")
    print(f"{'':18s} {'no-boundary (v1)':>18s} {'boundary (Hyman)':>18s}")
    print(f"{'dQ':18s} {r['dQ_noboundary']:18.6f} {r['dQ_boundary']:18.6f}")
    print(f"{'dQ/P32*':18s} {r['ratio_noboundary']:18.6f} {r['ratio_boundary']:18.6f}")
    print(f"\nvariazione di dQ dovuta al flag di bordo: {r['pct_change_dQ']:+.2f} %")
    print(">>> trascurabile." if abs(r["pct_change_dQ"]) < 1.0
          else ">>> NON trascurabile: usare la colonna 'boundary'.")
    print("------------------------------------------------------------")


# ==================================================================
# MAIN
# ==================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=DOMAIN_LIST)
    ap.add_argument("--run", type=int)
    ap.add_argument("--direction", choices=DIR_LIST)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--runs-root", default=RUNS_ROOT)
    ap.add_argument("--metrics-root", default=METRICS_ROOT)
    ap.add_argument("--skip-dir-check", action="store_true")
    ap.add_argument("--inspect", action="store_true",
                    help="stampa cosa c'e' nelle cartelle e nei file, senza calcolare")
    ap.add_argument("--aperture-coeff", type=float, default=5.0e-4,
                    help="coefficiente in b = coeff * r^exp (default 5e-4)")
    ap.add_argument("--aperture-exp", type=float, default=0.5,
                    help="esponente in b = coeff * r^exp (default 0.5)")

    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    print(f"runs-root    : {args.runs_root}")
    print(f"metrics-root : {args.metrics_root}\n")

    if not args.all:
        if not (args.domain and args.run and args.direction):
            ap.error("servono --domain, --run e --direction, oppure --all")
        rp = resolve_run_path(args.runs_root, args.domain, args.run)
        print(f"domain={args.domain} run={args.run} direction={args.direction}")
        print(f"run path   : {rp}")
        if args.inspect:
            inspect(rp, args.direction); return
        r = process_one(args.domain, args.run, rp, args.direction,
                        skip_dir_check=args.skip_dir_check,
                        coeff=args.aperture_coeff, expo=args.aperture_exp)
        print_single(r)
        out = os.path.join(args.metrics_root, args.domain, f"run_{args.run}",
                           f"dq_2core_{args.direction}.dat")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            for k in FIELDS:
                f.write(f"{k}: {r[k]}\n")
        print(f"\n-> {out}")
        return

    domains = [args.domain] if args.domain else DOMAIN_LIST
    dirs = [args.direction] if args.direction else DIR_LIST
    rows, errors = [], []

    for dom in domains:
        runs = find_runs(args.runs_root, dom)
        print(f"[{dom}] {len(runs)} run")
        for rid, rp in runs:
            for d in dirs:
                tag = f"{dom}/run{rid}/{d}"
                try:
                    r = process_one(dom, rid, rp, d, verbose=False,
                                    skip_dir_check=args.skip_dir_check,
                                    coeff=args.aperture_coeff,
                                    expo=args.aperture_exp)
                    rows.append(r)
                    print(f"  {tag:20s} 2core={r['n_2core']:4d} "
                          f"mesh={r['n_flowmesh']:4d} usate={r['n_matched']:4d}"
                          f"  dQ/P32* nb={r['ratio_noboundary']:.4f} "
                          f"b={r['ratio_boundary']:.4f} "
                          f"({r['pct_change_dQ']:+.2f} %)")
                except Exception as e:
                    errors.append((tag, str(e).splitlines()[0]))
                    print(f"  {tag:20s} ERRORE: {str(e).splitlines()[0]}")

    csv_path = args.csv or os.path.join(args.metrics_root, "dq_2core_summary.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader(); w.writerows(rows)

    print("\n=== RIEPILOGO ===")
    print(f"riusciti : {len(rows)}")
    print(f"falliti  : {len(errors)}")
    for tag, msg in errors:
        print(f"   {tag}: {msg}")
    part = [r for r in rows if r["n_matched"] < r["n_2core"]]
    if part:
        fr = [r["frac_2core_in_mesh"] for r in part]
        print(f"\nATTENZIONE: in {len(part)}/{len(rows)} casi il 2-core e' piu'"
              f" grande della mesh di flusso.")
        print(f"  frazione del 2-core effettivamente usata per P32*/dQ: "
              f"min {min(fr):.2f}, media {sum(fr)/len(fr):.2f}, max {max(fr):.2f}")
        print("  => Tabella 1 e Tabella 2 non sono sullo stesso insieme "
              "di fratture.")

    ch = [abs(r["pct_change_dQ"]) for r in rows
          if r["pct_change_dQ"] == r["pct_change_dQ"]]
    if ch:
        print(f"\neffetto del flag di bordo su dQ: max {max(ch):.2f} %, "
              f"medio {sum(ch)/len(ch):.2f} %")
        print(">>> trascurabile ovunque: i numeri della v1 vanno bene."
              if max(ch) < 1.0 else ">>> usare le colonne '*_boundary'.")
    print(f"\n-> {csv_path}")


if __name__ == "__main__":
    try:
        main()
    except RunError as e:
        print(f"\n*** ERRORE: {e}\n", file=sys.stderr)
        sys.exit(1)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
