#!/usr/bin/env python3
"""
pct_low_centrality.py
-------------------------
Calcola la percentuale di fratture con C_i < 0.1, per dominio,
pooled su tutte le direzioni e run -- serve per il testo su Fig. 3c.

Uso:
    python pct_low_centrality.py
"""

import os
import csv

METRICS_ROOT = "/hpc/home/filippo.porta/metrics/fixed"
DOMAIN_ORDER = ["low", "medium", "high"]
DIRECTIONS = ["X", "Y", "Z"]
THRESHOLD = 0.1


def main():
    print(f"{'Domain':<8}{'Dir':<5}{'N fractures':<14}{'%% Ci<'+str(THRESHOLD):<12}")
    print("-" * 40)

    for domain in DOMAIN_ORDER:
        domain_dir = os.path.join(METRICS_ROOT, domain)
        by_dir = {d: {"total": 0, "below": 0} for d in DIRECTIONS}

        for run_dir in sorted(d for d in os.listdir(domain_dir) if d.startswith("run_")):
            cent_file = os.path.join(domain_dir, run_dir, "centrality.csv")
            if not os.path.exists(cent_file):
                continue
            with open(cent_file, newline="") as f:
                for row in csv.DictReader(f):
                    d = row["direction"].strip().upper()
                    if d not in DIRECTIONS:
                        continue
                    ci = float(row["C_i"])
                    by_dir[d]["total"] += 1
                    if ci < THRESHOLD:
                        by_dir[d]["below"] += 1

        for direction in DIRECTIONS:
            t, b = by_dir[direction]["total"], by_dir[direction]["below"]
            pct = 100 * b / t if t else float("nan")
            print(f"{domain:<8}{direction:<5}{t:<14}{pct:<12.2f}")


if __name__ == "__main__":
    main()
