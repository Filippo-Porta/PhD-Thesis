#!/usr/bin/env python3
"""
recompute_fixed_breakthrough.py
------------------------------------
Ricalcola breakthrough.csv + tortuosity.csv (Sherman et al. 2020, come
in compute_breakthrough_run.py originale -- stessa logica, stesse
funzioni riusate via import) per TUTTE le run con STATUS_OK in
runs_fixed, salvando in metrics/fixed/{domain}/run_N/.

Non modifica compute_breakthrough_run.py: lo importa e chiama
direttamente process_direction()/make_planes(), passando il
transport_dir giusto per la struttura di runs_fixed (che non combacia
con lo schema base/runs/N/transport dello script originale).

Uso:
    python recompute_fixed_breakthrough.py --domain low
    python recompute_fixed_breakthrough.py --all
"""

import os
import sys
import csv
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compute_breakthrough_run as cbr

DOMAIN_RUNS_DIR = {
    "low":    "/hpc/scratch/filippo.porta/runs_10k/low",
    "medium": "/hpc/scratch/filippo.porta/runs_10k/medium",
    "high":   "/hpc/scratch/filippo.porta/runs_10k/high",
}

METRICS_OUT_ROOT = "/hpc/home/filippo.porta/metrics/fixed_10k"
DIRECTIONS = ("X", "Y", "Z")


def process_run(domain, run, run_path, planes, tol=0.5):
    transport_dir = os.path.join(run_path, "transport")
    out_dir = os.path.join(METRICS_OUT_ROOT, domain, f"run_{run}")
    os.makedirs(out_dir, exist_ok=True)

    out_csv = os.path.join(out_dir, "breakthrough.csv")
    tort_csv = os.path.join(out_dir, "tortuosity.csv")

    with open(out_csv, "w", newline="") as f_out, \
         open(tort_csv, "w", newline="") as f_tort:

        writer = csv.DictWriter(f_out, fieldnames=cbr.FIELDNAMES)
        writer.writeheader()
        tort_writer = csv.DictWriter(f_tort, fieldnames=cbr.TORT_FIELDNAMES)
        tort_writer.writeheader()

        totals = {"particles": 0, "rows": 0, "segments": 0}
        for direction in DIRECTIONS:
            n_particles, n_rows, n_segments = cbr.process_direction(
                run, direction, transport_dir, planes, tol, writer, tort_writer)
            totals["particles"] += n_particles
            totals["rows"] += n_rows
            totals["segments"] += n_segments
            print(f"  [{domain} run {run}] dir {direction}: "
                 f"{n_particles} particelle, {n_rows} crossing, "
                 f"{n_segments} segmenti tortuosita'")

    print(f"  [{domain} run {run}] TOTALE: {totals} -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=list(DOMAIN_RUNS_DIR))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--half-domain", type=float, default=25.0)
    ap.add_argument("--spacing", type=float, default=5.0)
    ap.add_argument("--tol", type=float, default=0.5)
    args = ap.parse_args()

    planes = cbr.make_planes(args.half_domain, args.spacing)
    domains = list(DOMAIN_RUNS_DIR) if args.all else [args.domain]

    for domain in domains:
        base = DOMAIN_RUNS_DIR[domain]
        runs = sorted(
            d for d in os.listdir(base)
            if d.isdigit() and os.path.exists(os.path.join(base, d, "STATUS_OK")))

        print(f"\n{'='*60}\n{domain.upper()}  ({len(runs)} run con STATUS_OK)\n{'='*60}")
        for run in runs:
            run_path = os.path.join(base, run)
            process_run(domain, int(run), run_path, planes, args.tol)

    print("\nFatto.")


if __name__ == "__main__":
    main()
