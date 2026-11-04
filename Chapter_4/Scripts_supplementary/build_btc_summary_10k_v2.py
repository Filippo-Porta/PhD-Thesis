
import argparse
import csv
import os
import sys

DEFAULT_SRC = ("/hpc/home/filippo.porta/metrics/fixed_10k/"
               "breakthrough_combined_summary.csv")

DIRECTIONS = ("X", "Y", "Z")
SECTOR_ORDER = {"low": 0, "medium": 1, "high": 2}

# piani candidati come uscita, uno per verso di flusso
CANDIDATES = (20.0, -20.0)


def load_rows(src):
    if not os.path.exists(src):
        sys.exit(f"ERRORE: file non trovato: {src}")
    with open(src, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"ERRORE: file vuoto: {src}")
    for col in ("sector", "direction", "plane", "t10", "t50", "t90"):
        if col not in rows[0]:
            sys.exit(f"ERRORE: colonna '{col}' assente in {src}.\n"
                     f"        colonne trovate: {list(rows[0])}")
    return rows


def median_t50_at(rows, direction, plane):
    """t50 mediano su tutti i settori, per una data direzione e piano."""
    vals = [float(r["t50"]) for r in rows
            if r["direction"] == direction and float(r["plane"]) == plane]
    if not vals:
        return None
    vals.sort()
    n = len(vals)
    return vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def detect_outlet_planes(rows, verbose=False):
    """
    Per ciascuna direzione, sceglie come piano di uscita quello fra +20 e
    -20 con il t50 mediano piu' grande: il piano di uscita e' l'ultimo
    raggiunto, quindi ha i tempi maggiori.
    """
    outlet = {}
    print("Rilevamento del verso di flusso (t50 mediano sui settori):")
    print(f"  {'dir':<5}{'t50 @ +20':<16}{'t50 @ -20':<16}"
          f"{'uscita':<10}{'rapporto'}")
    print("  " + "-" * 60)

    for d in DIRECTIONS:
        t_pos = median_t50_at(rows, d, 20.0)
        t_neg = median_t50_at(rows, d, -20.0)

        if t_pos is None and t_neg is None:
            sys.exit(f"ERRORE: nessuna riga per la direzione {d} "
                     f"ai piani +20 / -20.")
        if t_pos is None:
            outlet[d], ratio = -20.0, float("nan")
        elif t_neg is None:
            outlet[d], ratio = 20.0, float("nan")
        else:
            outlet[d] = 20.0 if t_pos >= t_neg else -20.0
            lo, hi = sorted((t_pos, t_neg))
            ratio = hi / lo if lo > 0 else float("nan")

        s_pos = f"{t_pos:.3e}" if t_pos is not None else "n/d"
        s_neg = f"{t_neg:.3e}" if t_neg is not None else "n/d"
        print(f"  {d:<5}{s_pos:<16}{s_neg:<16}"
              f"{outlet[d]:<10.1f}{ratio:.1f}x")

    print()
    if len(set(outlet.values())) > 1:
        flipped = [d for d in DIRECTIONS if outlet[d] < 0]
        print(f"NOTA: il verso del flusso non e' lo stesso in tutte le "
              f"direzioni.\n      Piano di uscita a -20 per: "
              f"{', '.join(flipped)}.\n      Verificare con:  head -3 "
              f"<run>/transport/{flipped[0]}/outcontroldir/part_control_1\n"
              f"      (la coordinata dell'asse di flusso deve partire da "
              f"+25 e decrescere)\n")
    return outlet


def print_table(selected, csv_out=None):
    has_n = "n_runs" in selected[0]

    header = f"{'Domain':<9}{'Dir':<5}"
    if has_n:
        header += f"{'n_runs':<9}"
    header += f"{'plane':<8}{'t10 [yr]':<14}{'t50 [yr]':<14}{'t90 [yr]'}"
    print(header)
    print("-" * len(header))

    for r in selected:
        line = f"{r['sector']:<9}{r['direction']:<5}"
        if has_n:
            line += f"{r['n_runs']:<9}"
        line += (f"{float(r['plane']):<8.0f}"
                 f"{float(r['t10']):<14.3e}"
                 f"{float(r['t50']):<14.3e}"
                 f"{float(r['t90']):.3e}")
        print(line)

    if csv_out:
        cols = ["sector", "direction", "plane", "t10", "t50", "t90"]
        if has_n:
            cols.insert(2, "n_runs")
        with open(csv_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(selected)
        print(f"\nTabella salvata in: {csv_out}")


def print_ratios(selected):
    """Rapporto t50(Y) / t50(X) e t50(Y) / t50(Z) per settore."""
    by = {(r["sector"], r["direction"]): float(r["t50"]) for r in selected}
    sectors = sorted({r["sector"] for r in selected},
                     key=lambda s: SECTOR_ORDER.get(s, 99))

    print("\nRapporti dei tempi mediani di arrivo:")
    print(f"  {'Domain':<9}{'t50(Y)/t50(X)':<18}{'t50(Y)/t50(Z)'}")
    print("  " + "-" * 45)
    for s in sectors:
        y, x, z = by.get((s, "Y")), by.get((s, "X")), by.get((s, "Z"))
        rx = f"{y / x:.1f}x" if y and x else "n/d"
        rz = f"{y / z:.1f}x" if y and z else "n/d"
        print(f"  {s:<9}{rx:<18}{rz}")
    print("\n(questi sono i numeri da usare al posto di HHH nel testo)")


def print_all_planes(rows):
    print("\nTutti i piani, per controllo:")
    print(f"  {'Domain':<9}{'Dir':<5}{'plane':<9}{'t10':<13}"
          f"{'t50':<13}{'t90'}")
    print("  " + "-" * 62)
    key = lambda r: (SECTOR_ORDER.get(r["sector"], 99),
                     r["direction"], float(r["plane"]))
    for r in sorted(rows, key=key):
        print(f"  {r['sector']:<9}{r['direction']:<5}"
              f"{float(r['plane']):<9.0f}{float(r['t10']):<13.3e}"
              f"{float(r['t50']):<13.3e}{float(r['t90']):.3e}")


def main():
    ap = argparse.ArgumentParser(
        description="Tabella t10/t50/t90 al piano di uscita, con "
                    "rilevamento automatico del verso di flusso.")
    ap.add_argument("--src", default=DEFAULT_SRC,
                    help="breakthrough_combined_summary.csv da leggere")
    ap.add_argument("--plane", action="append", default=[], metavar="DIR=VAL",
                    help="forza il piano di uscita di una direzione, "
                         "es. --plane Y=-20 (ripetibile)")
    ap.add_argument("--csv-out", default=None,
                    help="salva la tabella finale in questo file CSV")
    ap.add_argument("--verbose", action="store_true",
                    help="stampa anche tutti i piani intermedi")
    args = ap.parse_args()

    rows = load_rows(args.src)
    print(f"Sorgente: {args.src}")
    print(f"Righe lette: {len(rows)}\n")

    outlet = detect_outlet_planes(rows, args.verbose)

    for spec in args.plane:
        if "=" not in spec:
            sys.exit(f"ERRORE: formato non valido per --plane: {spec} "
                     f"(atteso DIR=VAL, es. Y=-20)")
        d, v = spec.split("=", 1)
        d = d.strip().upper()
        if d not in DIRECTIONS:
            sys.exit(f"ERRORE: direzione sconosciuta: {d}")
        outlet[d] = float(v)
        print(f"Piano forzato da riga di comando: {d} -> {outlet[d]:.1f}")

    selected = [r for r in rows
                if float(r["plane"]) == outlet.get(r["direction"])]
    if not selected:
        sys.exit("ERRORE: nessuna riga selezionata. Controllare i piani "
                 "disponibili con --verbose.")

    expected = len(DIRECTIONS) * len({r["sector"] for r in rows})
    if len(selected) != expected:
        print(f"ATTENZIONE: selezionate {len(selected)} righe, "
              f"attese {expected}.\n")

    selected.sort(key=lambda r: (r["direction"],
                                 SECTOR_ORDER.get(r["sector"], 99)))

    print()
    print_table(selected, args.csv_out)
    print_ratios(selected)

    if args.verbose:
        print_all_planes(rows)


if __name__ == "__main__":
    main()
