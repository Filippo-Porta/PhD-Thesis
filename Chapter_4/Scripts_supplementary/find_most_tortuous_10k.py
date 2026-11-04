#!/usr/bin/env python3
"""
find_most_tortuous_fixed.py
-------------------------------
Trova la particella con tortuosita' locale massima per ciascuna
direzione (X, Y, Z) e dominio, riusando tortuosity.csv gia'
calcolato da recompute_fixed_breakthrough.py -- nessun ricalcolo,
solo scansione dei CSV esistenti.

Uso:
    python find_most_tortuous_fixed.py --domain low
    python find_most_tortuous_fixed.py --all
"""

import os
import csv
import argparse
import numpy as np

METRICS_ROOT_FIXED = "/hpc/home/filippo.porta/metrics/fixed_10k"
METRICS_ROOT_OLD = "/hpc/home/filippo.porta/metrics/fixed"
DOMAIN_ORDER = ["low", "medium", "high"]
DIRECTIONS = ["X", "Y", "Z"]


def scan_domain(domain, root, percentile=99):
    domain_dir = os.path.join(root, domain)
    all_rows = {d: [] for d in DIRECTIONS}  # {direction: [(chi, run, particle_file, p_from, p_to), ...]}

    for run_dir in sorted(d for d in os.listdir(domain_dir) if d.startswith("run_")):
        tort_file = os.path.join(domain_dir, run_dir, "tortuosity.csv")
        if not os.path.exists(tort_file):
            continue
        run = run_dir.replace("run_", "")

        with open(tort_file, newline="") as f:
            for row in csv.DictReader(f):
                d = row["direction"].strip().upper()
                if d not in DIRECTIONS:
                    continue
                chi = float(row["tortuosity"])
                all_rows[d].append((chi, run, row["particle_file"],
                                   row["plane_from"], row["plane_to"]))

    result = {}
    for d in DIRECTIONS:
        if not all_rows[d]:
            result[d] = None
            continue
        chis = [r[0] for r in all_rows[d]]
        p_val = np.percentile(chis, percentile)
        # particella con chi PIU' VICINO al percentile (non il massimo assoluto)
        closest = min(all_rows[d], key=lambda r: abs(r[0] - p_val))
        result[d] = closest + (p_val, max(chis))  # aggiunge anche il valore
                                                    # del percentile e il vero
                                                    # massimo, per contesto
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=DOMAIN_ORDER)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--old", action="store_true",
                    help="usa i vecchi dati bacati (metrics/{domain}) "
                        "invece di metrics/fixed, per confronto")
    ap.add_argument("--percentile", type=float, default=99,
                    help="cerca la particella con chi piu' vicino a "
                        "questo percentile, invece del massimo assoluto "
                        "(piu' robusto, meno dominato da un singolo "
                        "evento raro)")
    args = ap.parse_args()

    root = METRICS_ROOT_OLD if args.old else METRICS_ROOT_FIXED
    domains = DOMAIN_ORDER if args.all else [args.domain]

    print(f"\n[sorgente dati: {'VECCHI (bacati)' if args.old else 'FIXED'}] "
         f"{root}")
    print(f"[criterio: particella piu' vicina al {args.percentile:.0f}° "
         f"percentile -- non il massimo assoluto]\n")
    print(f"{'Domain':<8}{'Dir':<5}{'chi_scelto':<12}{'chi_'+str(int(args.percentile))+'pct':<12}"
         f"{'chi_MAX':<12}{'Run':<8}{'Particle':<18}{'Plane':<15}")
    print("-" * 100)

    for domain in domains:
        best = scan_domain(domain, root, args.percentile)
        for direction in DIRECTIONS:
            if best[direction] is None:
                print(f"{domain:<8}{direction:<5}  (nessun dato)")
                continue
            chi, run, pfile, p_from, p_to, p_val, chi_max = best[direction]
            print(f"{domain:<8}{direction:<5}{chi:<12.3f}{p_val:<12.3f}"
                 f"{chi_max:<12.3f}{run:<8}{pfile:<18}{p_from}->{p_to}")


if __name__ == "__main__":
    main()
