#!/usr/bin/env python3


import os
import re
import argparse
import subprocess
import time

TARGET = 20
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))

DOMAIN_ROOTS = {
    "low":    "/hpc/scratch/filippo.porta/runs_10k/low",
    "medium": "/hpc/scratch/filippo.porta/runs_10k/medium",
    "high":   "/hpc/scratch/filippo.porta/runs_10k/high",
}

# memoria richiesta per la mesh completa: low e' piu' rada e basta
# meno, medium e high richiedono lo stesso carico della generazione
# originale (verificato empiricamente fino a 400 GB)
DOMAIN_MEM = {
    "low":    "100G",
    "medium": "450G",
    "high":   "450G",
}


def scan_status(root):
    """Ritorna (successi, tutti_i_seed_gia_tentati)."""
    if not os.path.isdir(root):
        os.makedirs(root, exist_ok=True)
        return 0, set()

    tried = set()
    successes = 0
    for name in os.listdir(root):
        m = re.match(r"^(\d+)(_FAILED)?$", name)
        if not m:
            continue
        seed = int(m.group(1))
        tried.add(seed)
        if m.group(2) is None:
            status_ok = os.path.join(root, name, "STATUS_OK")
            if os.path.isfile(status_ok):
                successes += 1
    return successes, tried


def submit_batch(domain, root, seeds, round_id):
    """Scrive tasks file + script SLURM per questo batch, lo
    sottomette con --wait (blocca finche' l'intero array non
    finisce), poi ritorna."""
    tasks_file = os.path.join(PIPELINE_DIR, f"tasks_{domain}_r{round_id}.txt")
    with open(tasks_file, "w") as f:
        for s in seeds:
            f.write(f"{s}\n")

    script_path = os.path.join(PIPELINE_DIR, f"submit_{domain}_r{round_id}.sh")
    with open(os.path.join(PIPELINE_DIR, "submit_batch_template.sh")) as f:
        template = f.read()

    script = (template
             .replace("DOMAIN_PLACEHOLDER", domain)
             .replace("N_PLACEHOLDER", str(len(seeds)))
             .replace("TASKFILE_PLACEHOLDER", tasks_file)
             .replace("ROOT_PLACEHOLDER", root)
             .replace("MEM_PLACEHOLDER", DOMAIN_MEM[domain]))
    with open(script_path, "w") as f:
        f.write(script)

    print(f"\n[{domain}] Round {round_id}: sottometto {len(seeds)} nuove run "
         f"(seed: {seeds})")
    print(f"  Attendo che il batch finisca (sbatch --wait)...")

    result = subprocess.run(["sbatch", "--wait", script_path],
                           cwd=PIPELINE_DIR, capture_output=True, text=True)
    print(f"  sbatch stdout: {result.stdout.strip()}")
    if result.returncode != 0:
        print(f"  ATTENZIONE sbatch stderr: {result.stderr.strip()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=list(DOMAIN_ROOTS))
    ap.add_argument("--target", type=int, default=TARGET)
    args = ap.parse_args()

    root = DOMAIN_ROOTS[args.domain]
    round_id = 0

    while True:
        successes, tried = scan_status(root)
        print(f"\n{'='*60}")
        print(f"[{args.domain}] successi finora: {successes}/{args.target}  "
             f"(seed gia' tentati: {len(tried)})")
        print(f"{'='*60}")

        missing = args.target - successes
        if missing <= 0:
            print(f"\n*** {args.domain}: raggiunto il target di "
                 f"{args.target} run riuscite. Fine. ***")
            break

        round_id += 1
        next_seed = (max(tried) + 1) if tried else 1
        new_seeds = list(range(next_seed, next_seed + missing))

        submit_batch(args.domain, root, new_seeds, round_id)
        time.sleep(5)  # margine di sicurezza prima di ricontare


if __name__ == "__main__":
    main()
