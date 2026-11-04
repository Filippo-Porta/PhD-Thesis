#!/usr/bin/env python3
"""
completion_rate.py
----------------------
Calcola la percentuale di particelle che completano il transito
(arrivano a partime.dat) rispetto al totale seminato (particelle
con almeno un crossing in outcontroldir), per dominio e direzione,
sommando su tutte le run.

Uso:
    python completion_rate.py
"""

import os
import glob

DOMAIN_RUNS_DIR = {
    "low":    "/hpc/scratch/filippo.porta/runs_10k/low",
    "medium": "/hpc/scratch/filippo.porta/runs_10k/medium",
    "high":   "/hpc/scratch/filippo.porta/runs_10k/high",
}
DIRECTIONS = ["X", "Y", "Z"]


def count_completed(transport_dir, direction):
    partime = os.path.join(transport_dir, direction, "partime.dat")
    if not os.path.exists(partime):
        return 0
    with open(partime) as f:
        return sum(1 for _ in f)


def count_seeded(transport_dir, direction):
    outcontrol = os.path.join(transport_dir, direction, "outcontroldir")
    if not os.path.isdir(outcontrol):
        return 0
    return len(glob.glob(os.path.join(outcontrol, "part_control_*")))


def main():
    print(f"{'Domain':<8}{'Dir':<5}{'Seeded':<10}{'Completed':<12}{'%% completion':<12}")
    print("-" * 50)

    for domain, base in DOMAIN_RUNS_DIR.items():
        runs = sorted(
            d for d in os.listdir(base)
            if d.isdigit() and os.path.exists(os.path.join(base, d, "STATUS_OK")))

        for direction in DIRECTIONS:
            total_seeded, total_completed = 0, 0
            for run in runs:
                transport_dir = os.path.join(base, run, "transport")
                total_seeded += count_seeded(transport_dir, direction)
                total_completed += count_completed(transport_dir, direction)

            pct = 100 * total_completed / total_seeded if total_seeded else float("nan")
            print(f"{domain:<8}{direction:<5}{total_seeded:<10}"
                 f"{total_completed:<12}{pct:<12.2f}")


if __name__ == "__main__":
    main()
