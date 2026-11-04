"""
Shared library for DFN backbone graph metrics.

Directory layout per realization (N = 1, 2, 3, ...):

    /hpc/archive/G_NEXT/runs/N/intensity13/
        backbone.dat                     <- fracture IDs kept by current-flow threshold
        dfngen_logfile.txt               <- P32 before/after isolated-fracture removal
        dfnGen_output/radii_Final.dat    <- xRadius yRadius Family (full final network)

    /hpc/archive/G_NEXT/runs/N/output_backbone/dfnGen_output/
        connectivity.dat, fracture_info.dat
        left.dat, right.dat, front.dat, back.dat, bottom.dat, top.dat

Metrics per direction (x, y, z):
    n_c   -- node connectivity between virtual source and target
    C_i   -- simple-path centrality, length-ratio criterion (factor=1.2)
    plus dominance ratio C_rank1 / C_rank2

Per realization (direction-independent):
    P32 (before/after isolated removal, parsed from dfnGen log)
    P32* (backbone-only intensity, radii_Final x backbone.dat cross-ref)

Requires: networkx.
"""

import os
import re
import math
from collections import Counter

import networkx as nx

DOMAIN_VOLUME = 50.0 * 50.0 * 50.0   # m^3, cubic domain, side 50 m

DIRECTION_BOUNDARIES = {
    "x": ("left.dat", "right.dat"),
    "y": ("front.dat", "back.dat"),
    "z": ("bottom.dat", "top.dat"),
}


# ----------------------------------------------------------------------
# File loading
# ----------------------------------------------------------------------

def load_id_list(path):
    """One fracture ID per line -> set of ints. Loud errors on bad files."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path!r}")
    if os.path.isdir(path):
        raise IsADirectoryError(f"Expected file, found directory: {path!r}")
    if os.path.getsize(path) == 0:
        raise ValueError(f"Empty file (0 bytes): {path!r}")
    with open(path, encoding="utf-8-sig") as f:
        return {int(line.strip()) for line in f if line.strip()}


def load_connectivity(path):
    """
    connectivity.dat: line i (1-indexed) lists neighbors of fracture i.
    Returns (edges, neighbor_counts) where neighbor_counts[i] = number of
    neighbors listed on line i (used for the fracture_info cross-check).
    """
    edges = []
    neighbor_counts = {}
    with open(path, encoding="utf-8-sig") as f:
        for i, line in enumerate(f, start=1):
            neigh = [int(x) for x in line.split()]
            neighbor_counts[i] = len(neigh)
            for j in neigh:
                if i < j:
                    edges.append((i, j))
    return edges, neighbor_counts


def load_radii(path):
    """
    radii_Final.dat -> list of (rx, ry) tuples, index 0 = fracture 1.
    Skips the two header lines (title + format description).
    """
    radii = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                rx, ry = float(parts[0]), float(parts[1])
            except ValueError:
                continue  # header lines
            radii.append((rx, ry))
    if not radii:
        raise ValueError(f"No radii parsed from {path!r}")
    return radii


def load_fracture_info(path):
    """
    fracture_info.dat: header 'num_connections perm aperture', then one
    line per fracture. Returns list of (n_conn, perm, aperture),
    index 0 = fracture 1.
    """
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                rows.append((int(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                continue  # header
    if not rows:
        raise ValueError(f"No rows parsed from {path!r}")
    return rows


# ----------------------------------------------------------------------
# Sanity checks (file-numbering consistency)
# ----------------------------------------------------------------------

def check_backbone_indices(backbone, radii):
    """Every backbone ID must be a valid 1-based row of radii_Final.dat."""
    bad = [i for i in backbone if i < 1 or i > len(radii)]
    if bad:
        raise AssertionError(
            f"backbone.dat contains IDs outside radii_Final.dat range "
            f"(n_radii={len(radii)}): e.g. {bad[:5]} -- numbering "
            f"conventions do not match, P32* would be wrong.")


def check_connectivity_vs_info(neighbor_counts, fracture_info,
                                tol_mismatches=None):
    """
    NON-FATAL diagnostic only (does not raise).

    Originally intended as a numbering cross-check, assuming
    num_connections (fracture_info.dat) = fracture-fracture neighbors
    (connectivity.dat) + boundary intersections. Empirically false on
    real HPC output: num_connections is sometimes LOWER than the raw
    neighbor count (e.g., 0 despite 1-2 geometric neighbors), most
    likely because it reflects post-meshing/FRAM connectivity (small
    intersections below the meshing resolution h can be dropped) while
    connectivity.dat reflects raw geometric adjacency.

    The numbering itself is NOT in question here -- that is verified
    separately, and robustly, by check_aperture_radius_correlation
    (validated at 100% on both backbone and non-backbone rows). This
    function is kept only to print a heads-up count, in case an
    unusually large mismatch signals something worth a second look.
    """
    n = min(len(fracture_info), len(neighbor_counts))
    n_mismatch = sum(
        1 for i in range(1, n + 1)
        if fracture_info[i - 1][0] > neighbor_counts.get(i, -1) + 6
    )
    if n_mismatch:
        print(f"  [info] {n_mismatch}/{n} fractures have "
              f"num_connections > neighbors+6 (diagnostic only, "
              f"not treated as an error -- see docstring).")


def check_aperture_radius_correlation(radii, fracture_info,
                                       alpha=5e-4, beta=0.5,
                                       rel_tol=0.05, min_frac_ok=0.95):
    """
    Aperture model: b = alpha * r^beta. Checks fracture_info's aperture
    column against radii, informationally.

    NOTE: fracture_info.dat's aperture/perm columns are written by
    DFNGen for the MOTHER network, before hydraulic properties are
    assigned -- they are structurally 0 at this stage (confirmed on
    real data: both archive and scratch runs show aperture=0 here).
    Real per-fracture apertures only exist later, per-backbone, in
    aperture.dat (used by compute_dq_hyman.py). This function can
    therefore NEVER validate row correspondence via this column --
    it is informational only, never a hard gate.
    """
    n = min(len(radii), len(fracture_info))
    ok = 0
    for i in range(n):
        r = radii[i][0]
        b_expected = alpha * (r ** beta)
        b_actual = fracture_info[i][2]
        if b_expected > 0 and abs(b_actual - b_expected) / b_expected <= rel_tol:
            ok += 1
    frac_ok = ok / n if n else 0.0
    return frac_ok


# ----------------------------------------------------------------------
# Graph construction
# ----------------------------------------------------------------------

def build_directional_graph(edges, backbone, backbone_dir, direction):
    """
    Backbone graph for one flow direction, with virtual source 's' and
    target 't' attached to fractures touching the corresponding faces.
    `edges` comes from the full-network connectivity; only edges with
    both endpoints in `backbone` are kept.
    """
    source_name, target_name = DIRECTION_BOUNDARIES[direction]
    source_fx = load_id_list(os.path.join(backbone_dir, source_name)) & backbone
    target_fx = load_id_list(os.path.join(backbone_dir, target_name)) & backbone

    if not source_fx:
        raise ValueError(f"No backbone fracture touches the "
                         f"{source_name} boundary (direction {direction}).")
    if not target_fx:
        raise ValueError(f"No backbone fracture touches the "
                         f"{target_name} boundary (direction {direction}).")

    G = nx.Graph()
    G.add_nodes_from(backbone)
    G.add_edges_from((u, v) for u, v in edges
                     if u in backbone and v in backbone)

    G.add_node('s')
    G.add_node('t')
    G.add_edges_from(('s', f) for f in source_fx)
    G.add_edges_from((f, 't') for f in target_fx)
    return G


# ----------------------------------------------------------------------
# Option A: directional backbone extraction from the mother network
# ----------------------------------------------------------------------

def build_mother_graph(edges, n_fractures, gen_dir, direction):
    """
    Graph of the FULL mother network for one direction: all fractures,
    fracture-fracture edges, plus virtual s/t attached to the boundary
    pair of that direction (boundary files live in the mother network's
    dfnGen_output directory).
    """
    source_name, target_name = DIRECTION_BOUNDARIES[direction]
    source_fx = load_id_list(os.path.join(gen_dir, source_name))
    target_fx = load_id_list(os.path.join(gen_dir, target_name))

    G = nx.Graph()
    G.add_nodes_from(range(1, n_fractures + 1))
    G.add_edges_from(edges)
    G.add_node('s')
    G.add_node('t')
    G.add_edges_from(('s', f) for f in source_fx if 1 <= f <= n_fractures)
    G.add_edges_from((f, 't') for f in target_fx if 1 <= f <= n_fractures)
    return G


def current_flow_backbone(G, thrs=1e-16):
    """
    Exact translation of dfnWorks' current_flow_threshold(G, "s", "t",
    weight=None, thrs=1e-16), confirmed against its actual source:

        cf = nx.edge_current_flow_betweenness_centrality_subset(
                 G, sources=["s"], targets=["t"], weight=None)
        currentflow_edges = [(u, v) for (u, v), d in cf.items() if d > thrs]
        H = G.edge_subgraph(currentflow_edges)

    Note this is EDGE current-flow betweenness (not node-level, which
    was an earlier -- incorrect -- attempt here): thresholding is
    applied to edges, and a fracture (node) is retained in the backbone
    iff it touches at least one surviving edge. Returns the set of
    retained fracture IDs.
    """
    if not nx.has_path(G, 's', 't'):
        raise ValueError("No path between source and target -- "
                         "network does not percolate in this direction.")
    comp = nx.node_connected_component(G, 's')
    H_full = G.subgraph(comp).copy()

    cf = nx.edge_current_flow_betweenness_centrality_subset(
        H_full, sources=['s'], targets=['t'], weight=None)
    currentflow_edges = [(u, v) for (u, v), d in cf.items() if d > thrs]
    H = H_full.edge_subgraph(currentflow_edges).copy()

    return {n for n in H.nodes if n not in ('s', 't')}


def induced_backbone_graph(G, backbone):
    """Subgraph induced by backbone fractures + s/t (edges to s/t kept)."""
    keep = set(backbone) | {'s', 't'}
    return G.subgraph(keep).copy()


# ----------------------------------------------------------------------
# Topological metrics
# ----------------------------------------------------------------------

def node_connectivity(G):
    return nx.node_connectivity(G, 's', 't')


def length_ratio_paths_centrality(G, factor=1.2, max_paths=20000):
    """
    C_i = fraction of "reasonably direct" s-t simple paths through node i,
    where a path qualifies if its length <= factor * shortest path length.

    factor=1.2 validated empirically on both sparse and dense sectors:
    complete enumeration (no cap), stable ranking, signal of the dominant
    fracture not yet diluted by long circuitous paths (which is what
    happens at factor >= 1.5 on dense networks and >= 2.0 on sparse ones).

    Returns (centrality_dict, n_paths, shortest_len, capped_flag).
    """
    gen = nx.shortest_simple_paths(G, source='s', target='t')
    paths = []
    shortest_len = None
    capped = False

    for p in gen:
        if shortest_len is None:
            shortest_len = len(p)
        if len(p) > factor * shortest_len:
            break
        paths.append(p)
        if len(paths) >= max_paths:
            capped = True
            break

    counts = Counter()
    for p in paths:
        for node in p[1:-1]:
            counts[node] += 1

    n_paths = len(paths)
    centrality = {n: counts[n] / n_paths
                  for n in G.nodes if n not in ('s', 't')}
    return centrality, n_paths, shortest_len, capped


def dominance_ratio(centrality):
    """C_rank1 / C_rank2 -- scale-robust channelization indicator."""
    ranked = sorted(centrality.values(), reverse=True)
    if len(ranked) < 2 or ranked[1] == 0:
        return float('nan')
    return ranked[0] / ranked[1]


def degree_stats(G):
    """Mean and variance of the fracture-node degree distribution
    (virtual s/t excluded from the node list AND their edges ignored)."""
    degs = []
    for n in G.nodes:
        if n in ('s', 't'):
            continue
        d = sum(1 for nb in G.neighbors(n) if nb not in ('s', 't'))
        degs.append(d)
    n = len(degs)
    mean = sum(degs) / n
    var = sum((d - mean) ** 2 for d in degs) / n
    return mean, var


# ----------------------------------------------------------------------
# P32
# ----------------------------------------------------------------------

def parse_dfngen_log_p32(log_path):
    """
    (p32_before, p32_after) isolated-fracture removal, from dfnGen log.
    Either value may be None if the section is missing.
    """
    section = None
    p32_before, p32_after = None, None
    pattern = re.compile(r"Total Fracture Intensity \(P32\):\s*([\d.eE+-]+)")
    with open(log_path, encoding="utf-8-sig") as f:
        for line in f:
            if "Statistics Before Isolated Fractures Removed" in line:
                section = "before"
            elif "Statistics After Isolated Fractures Removed" in line:
                section = "after"
            m = pattern.search(line)
            if m:
                val = float(m.group(1))
                if section == "before" and p32_before is None:
                    p32_before = val
                elif section == "after" and p32_after is None:
                    p32_after = val
    return p32_before, p32_after


def backbone_p32(radii, backbone, domain_volume=DOMAIN_VOLUME):
    """
    P32* = sum of areas (pi * rx * ry, elliptical) of backbone fractures
    divided by domain volume. radii index 0 = fracture 1 (1-based IDs).
    """
    total_area = sum(math.pi * radii[i - 1][0] * radii[i - 1][1]
                     for i in backbone)
    return total_area / domain_volume


def kcore_degree_stats(G, k=2):
    """
    TRUE k-core (default k=2) mean/variance of degree, for comparison
    against the current-flow-threshold backbone. Unlike current-flow
    threshold, k-core enforces a strict minimum degree >= k on every
    retained node by recursive removal -- a genuinely different
    reduction, useful to check how much of any degree discrepancy is
    due to the extraction method vs. the underlying network itself.
    """
    core = nx.k_core(G, k=k, core_number=None)
    degs = [d for n, d in core.degree() if n not in ('s', 't')]
    if not degs:
        return None, None, 0
    mean = sum(degs) / len(degs)
    var = sum((d - mean) ** 2 for d in degs) / len(degs)
    return mean, var, len(degs)


def two_core_prune(H, protected=('s', 't')):
    """
    Additional 2-core pruning ON TOP OF an already-extracted backbone
    (e.g. the current-flow-threshold backbone), replicating the extra
    step used in the original analysis script: recursively remove any
    node (other than the protected source/target) whose degree drops
    below 2, where degree counts ALL edges including those to s/t.

    This is a genuinely different (stricter) reduction than current-flow
    threshold alone: current-flow threshold can retain degree-1
    fractures (e.g. boundary fractures with a single interior
    connection) as long as they carry non-negligible current; this
    extra pass strips those out, which can trigger a cascade of further
    removals inward from the boundary. Confirmed empirically to differ
    substantially between realizations (e.g. run 1: no change; run 4:
    224 -> 61 fractures).

    Returns a new graph (copy), s/t always retained.
    """
    H = H.copy()
    changed = True
    while changed:
        changed = False
        to_remove = [n for n in H.nodes()
                    if n not in protected and H.degree(n) < 2]
        if to_remove:
            H.remove_nodes_from(to_remove)
            changed = True
    return H
