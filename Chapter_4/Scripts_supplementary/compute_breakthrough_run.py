
import argparse
import csv
import glob
import os
import sys
import time
import traceback

# ----------------------------------------------------------------------
# Sector -> base directory (confirmed from your `ls` on the three roots)
# ----------------------------------------------------------------------
SECTOR_BASE = {
    "low": "/hpc/scratch/filippo.porta",
    "medium": "/hpc/archive/G_NEXT",
    "high": "/hpc/group/G_NEXT/filippo.porta",
}

DIRECTIONS = ("X", "Y", "Z")

# column index (after str.split()) of the position matching each
# transport direction: time=0, x=1, y=2, z=3
POS_COLUMN = {"X": 1, "Y": 2, "Z": 3}

FIELDNAMES = ["run", "direction", "plane", "particle_file",
             "arrival_time", "traj_length"]

TORT_FIELDNAMES = ["run", "direction", "plane_from", "plane_to",
                   "particle_file", "delta_l", "delta_x", "delta_t",
                   "tortuosity"]


def make_planes(half_domain=25.0, spacing=5.0):
    """-25, -20, ..., 20, 25 by default -- matches Delta x = 5 m on a
    cubic domain of side 50 (see dfn_metrics.DOMAIN_VOLUME)."""
    n = int(round(2 * half_domain / spacing)) + 1
    return [round(-half_domain + i * spacing, 6) for i in range(n)]


def nearest_plane(x, planes, tol):
    """Snap x to the nearest control plane if within tol, else None."""
    best, best_d = None, None
    for p in planes:
        d = abs(x - p)
        if best_d is None or d < best_d:
            best, best_d = p, d
    if best_d is not None and best_d <= tol:
        return best
    return None


def parse_particle_file(path, pos_col, planes, tol):
    """
    Reads one part_control_<k> file. Returns dict
    {plane: (arrival_time, traj_length)} -- FIRST crossing per plane
    only. A dict (not a list) so tortuosity can look up any two planes
    directly, regardless of the temporal order rows arrived in.
    """
    crossings = {}
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                t = float(parts[0])
                pos = float(parts[pos_col])
                traj_len = float(parts[7])
            except ValueError:
                continue  # stray header/blank line
            plane = nearest_plane(pos, planes, tol)
            if plane is None or plane in crossings:
                continue
            crossings[plane] = (t, traj_len)
    return crossings


def tortuosity_segments(crossings, planes):
    """
    For every pair of SPATIALLY ADJACENT planes (consecutive entries
    in `planes`) that this particle's first-crossing dict both cover,
    yield (plane_from, plane_to, delta_l, delta_x, delta_t,
    tortuosity). Uses lambda (traj_length) and arrival_time at each
    plane's first crossing, per chi(x_i,j;a) = |Dl| / |Dx|.
    """
    for p_from, p_to in zip(planes[:-1], planes[1:]):
        if p_from not in crossings or p_to not in crossings:
            continue
        t_from, l_from = crossings[p_from]
        t_to, l_to = crossings[p_to]
        delta_x = abs(p_to - p_from)
        if delta_x <= 0:
            continue
        delta_l = abs(l_to - l_from)
        delta_t = abs(t_to - t_from)
        yield (p_from, p_to, delta_l, delta_x, delta_t,
              delta_l / delta_x)


def process_direction(run, direction, transport_dir, planes, tol,
                      writer, tort_writer):
    """One direction: loop over every part_control_* file, write both
    breakthrough rows and tortuosity segment rows."""
    pos_col = POS_COLUMN[direction]
    outcontrol_dir = os.path.join(transport_dir, direction, "outcontroldir")

    if not os.path.isdir(outcontrol_dir):
        print(f"  [dir {direction}] MISSING: {outcontrol_dir}",
              file=sys.stderr)
        return 0, 0, 0

    files = glob.glob(os.path.join(outcontrol_dir, "part_control_*"))
    n_particles = len(files)
    n_rows = 0
    n_segments = 0

    for fp in files:
        fname = os.path.basename(fp)
        try:
            crossings = parse_particle_file(fp, pos_col, planes, tol)
        except Exception:
            print(f"  [dir {direction}] failed on {fname}:\n"
                 f"{traceback.format_exc()}", file=sys.stderr)
            continue

        for plane, (t, traj_len) in crossings.items():
            writer.writerow({
                "run": run, "direction": direction, "plane": plane,
                "particle_file": fname, "arrival_time": t,
                "traj_length": traj_len,
            })
            n_rows += 1

        for (p_from, p_to, delta_l, delta_x, delta_t,
            chi) in tortuosity_segments(crossings, planes):
            tort_writer.writerow({
                "run": run, "direction": direction,
                "plane_from": p_from, "plane_to": p_to,
                "particle_file": fname, "delta_l": delta_l,
                "delta_x": delta_x, "delta_t": delta_t,
                "tortuosity": chi,
            })
            n_segments += 1

    return n_particles, n_rows, n_segments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--sector", required=True, choices=list(SECTOR_BASE),
                    help="low | medium | high -- selects the base path")
    ap.add_argument("--base", default=None,
                    help="override the sector's default base path "
                        "(normally not needed)")
    ap.add_argument("--half-domain", type=float, default=25.0,
                    help="domain half-extent [m] -- default matches "
                        "the cubic side-50 domain (dfn_metrics.py)")
    ap.add_argument("--spacing", type=float, default=5.0,
                    help="control-plane spacing [m], matches the "
                        "paper's Delta x = 5 m")
    ap.add_argument("--tol", type=float, default=0.5,
                    help="max distance [m] from a plane for a row to "
                        "be assigned to it -- generous by default "
                        "since outcontroldir rows are ALREADY "
                        "control-plane crossings (not raw trajectory "
                        "points), the only noise is dfnTrans' own "
                        "interpolation")
    ap.add_argument("--outdir", default="/hpc/home/filippo.porta/metrics",
                    help="root metrics dir -- data are written to "
                        "<outdir>/<sector>/run_<run>/breakthrough.csv")
    args = ap.parse_args()

    base = args.base or SECTOR_BASE[args.sector]
    transport_dir = os.path.join(base, "runs", str(args.run), "transport")

    planes = make_planes(args.half_domain, args.spacing)

    run_outdir = os.path.join(args.outdir, args.sector, f"run_{args.run}")
    os.makedirs(run_outdir, exist_ok=True)
    out_csv = os.path.join(run_outdir, "breakthrough.csv")
    tort_csv = os.path.join(run_outdir, "tortuosity.csv")

    print(f"[{args.sector}/run {args.run}] transport_dir={transport_dir}")
    print(f"[{args.sector}/run {args.run}] control planes: {planes}")

    t0 = time.time()
    with open(out_csv, "w", newline="") as f, \
         open(tort_csv, "w", newline="") as ft:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        tort_writer = csv.DictWriter(ft, fieldnames=TORT_FIELDNAMES)
        tort_writer.writeheader()

        for direction in DIRECTIONS:
            td0 = time.time()
            n_particles, n_rows, n_segments = process_direction(
                args.run, direction, transport_dir, planes, args.tol,
                writer, tort_writer)
            print(f"  [{direction}] {n_particles} particle files, "
                 f"{n_rows} plane-crossings, {n_segments} tortuosity "
                 f"segments written ({time.time() - td0:.1f}s)")

    print(f"[{args.sector}/run {args.run}] done -> {out_csv}, "
         f"{tort_csv} ({time.time() - t0:.1f}s total)")


if __name__ == "__main__":
    main()
