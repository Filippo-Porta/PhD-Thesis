#!/usr/bin/env python3


import argparse
import csv
import sys

# --------------------------------------------------------------------
# Parametri fisici. VERIFICARE prima dell'uso.
# --------------------------------------------------------------------

MU = 1.0e-3          # viscosita' dinamica dell'acqua [Pa s]
DP = 5.0e4           # salto di pressione imposto [Pa] (2.0e6 - 1.95e6)
L = 50.0             # lunghezza del dominio nella direzione di flusso [m]
SEC_PER_YEAR = 365.25 * 24 * 3600

# NOTA su L: il dominio e' un cubo di lato 50. Le particelle entrano al
# bordo (-25) e il piano di uscita usato per t50 e' a +20 lungo X e Z e
# a -20 lungo Y, quindi la distanza effettivamente percorsa e' 45 m, non
# 50. Usare --length 45 per il confronto piu' corretto con t50. La
# scelta non cambia le conclusioni, perche' e' un fattore comune alle
# tre direzioni, ma cambia il valore assoluto di t50 normalizzato.


# --------------------------------------------------------------------
# Dati. Sostituire se cambiano.
# --------------------------------------------------------------------

# t50 [anni], dal piano di uscita, campagna 10k
T50 = {
    ("low", "X"): 8.99e-07, ("low", "Y"): 5.62e-06, ("low", "Z"): 6.63e-07,
    ("medium", "X"): 6.95e-07, ("medium", "Y"): 5.79e-06, ("medium", "Z"): 6.56e-07,
    ("high", "X"): 7.13e-07, ("high", "Y"): 3.88e-06, ("high", "Z"): 7.33e-07,
}

# t10 e t90, servono per il rapporto di dispersione
T10 = {
    ("low", "X"): 3.53e-07, ("low", "Y"): 1.69e-06, ("low", "Z"): 2.78e-07,
    ("medium", "X"): 2.99e-07, ("medium", "Y"): 1.72e-06, ("medium", "Z"): 3.06e-07,
    ("high", "X"): 3.14e-07, ("high", "Y"): 1.39e-06, ("high", "Z"): 3.22e-07,
}
T90 = {
    ("low", "X"): 3.50e-06, ("low", "Y"): 2.28e-05, ("low", "Z"): 3.07e-06,
    ("medium", "X"): 2.65e-06, ("medium", "Y"): 2.13e-05, ("medium", "Z"): 3.20e-06,
    ("high", "X"): 3.37e-06, ("high", "Y"): 1.53e-05, ("high", "Z"): 4.07e-06,
}

# Permeabilita' effettiva [m^2], medie sulle realizzazioni, da
# Totale_basso.xlsx / Totale_medio.csv / Totale_alto.xlsx
# (Porta et al., in press). n = 50, 53, 50 realizzazioni: nel file
# medium le run 67 e 70 hanno K = NaN e sono state escluse.
K = {
    ("low", "X"): 2.273e-11,
    ("low", "Y"): 3.275e-12,
    ("low", "Z"): 2.620e-11,
    ("medium", "X"): 5.948e-11,
    ("medium", "Y"): 8.753e-12,
    ("medium", "Z"): 6.475e-11,
    ("high", "X"): 9.097e-11,
    ("high", "Y"): 1.926e-11,
    ("high", "Z"): 9.473e-11,
}

SECTORS = ("low", "medium", "high")
DIRECTIONS = ("X", "Y", "Z")


# --------------------------------------------------------------------

def load_phi(path):
    phi = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            phi[(r["sector"], r["direction"])] = float(r["phi"])
    return phi


def parse_phi_inline(specs):
    """--phi-inline low:X=1.2e-5"""
    phi = {}
    for s in specs:
        try:
            key, val = s.split("=")
            sector, direction = key.split(":")
            phi[(sector.strip(), direction.strip().upper())] = float(val)
        except ValueError:
            sys.exit(f"ERRORE: formato non valido: {s} "
                     f"(atteso settore:DIR=valore)")
    return phi


def t_adv_years(k, phi, length):
    """Tempo advettivo caratteristico, in anni."""
    return phi * MU * length ** 2 / (k * DP) / SEC_PER_YEAR


def section(title):
    print(f"\n{title}")
    print("=" * len(title))


def test_1_raw_ratios():
    section("1. Rapporti grezzi: tempi contro permeabilita'")
    print("Se il ritardo lungo Y fosse dovuto alla sola permeabilita',")
    print("le due colonne dovrebbero coincidere.\n")
    print(f"  {'settore':<9}{'t50(Y)/t50(X)':<16}{'K_xx/K_yy':<14}{'scarto'}")
    print("  " + "-" * 52)
    for s in SECTORS:
        ty, tx = T50.get((s, "Y")), T50.get((s, "X"))
        kx, ky = K.get((s, "X")), K.get((s, "Y"))
        if None in (ty, tx, kx, ky):
            print(f"  {s:<9}dati mancanti")
            continue
        rt, rk = ty / tx, kx / ky
        print(f"  {s:<9}{rt:<16.1f}{rk:<14.1f}{rt / rk:+.0%}"
              .replace("+", "x  ") if False else
              f"  {s:<9}{rt:<16.1f}{rk:<14.1f}{rt / rk:.2f}")
    print("\n  scarto = rapporto dei tempi / rapporto delle permeabilita'")
    print("  valore 1.0 significa che K spiega tutto il ritardo")


def test_2_no_porosity():
    section("2. Normalizzazione senza porosita'")
    print("t50 * k, cioe' i tempi corretti per la sola permeabilita'.")
    print("Valori uguali fra direzioni significano che K spiega tutto.\n")
    print(f"  {'settore':<9}{'X':<14}{'Y':<14}{'Z':<14}{'Y/X'}")
    print("  " + "-" * 60)
    for s in SECTORS:
        vals = {}
        for d in DIRECTIONS:
            t, k = T50.get((s, d)), K.get((s, d))
            vals[d] = t * k if (t is not None and k is not None) else None
        cells = "".join(f"{v:<14.3e}" if v is not None else f"{'n/d':<14}"
                        for v in (vals["X"], vals["Y"], vals["Z"]))
        ratio = (f"{vals['Y'] / vals['X']:.2f}"
                 if vals["Y"] and vals["X"] else "n/d")
        print(f"  {s:<9}{cells}{ratio}")


def test_3_full(phi, length):
    section("3. Normalizzazione completa con porosita'")
    print(f"t_adv = phi * mu * L^2 / (k * dP), con L = {length} m\n")
    print(f"  {'settore':<9}{'dir':<5}{'phi':<12}{'t_adv [yr]':<14}"
          f"{'t50 [yr]':<14}{'t50/t_adv'}")
    print("  " + "-" * 68)
    norm = {}
    for s in SECTORS:
        for d in DIRECTIONS:
            t, k, p = T50.get((s, d)), K.get((s, d)), phi.get((s, d))
            if None in (t, k, p):
                print(f"  {s:<9}{d:<5}dati mancanti")
                continue
            ta = t_adv_years(k, p, length)
            norm[(s, d)] = t / ta
            print(f"  {s:<9}{d:<5}{p:<12.3e}{ta:<14.3e}{t:<14.3e}"
                  f"{t / ta:.2f}")

    print(f"\n  {'settore':<9}{'(t50/t_adv)_Y / (t50/t_adv)_X'}")
    print("  " + "-" * 45)
    for s in SECTORS:
        y, x = norm.get((s, "Y")), norm.get((s, "X"))
        if y and x:
            print(f"  {s:<9}{y / x:.2f}")

    print("\n  Lettura del risultato")
    print("  se il rapporto resta ben sopra 1, la topologia aggiunge un")
    print("     controllo oltre la velocita' bulk: tesi confermata")
    print("  se il rapporto e' circa 1, la permeabilita' spiega la scala")
    print("     temporale, e l'argomento va spostato sulla FORMA della BTC")
    print("     (vedi test 4)")


def test_4_shape():
    section("4. Forma della distribuzione, indipendente dalla scala")
    print("t90/t10 misura la dispersione ed e' insensibile a qualunque")
    print("riscalamento temporale. Se differisce fra direzioni, la")
    print("topologia controlla la forma della BTC anche nel caso in cui")
    print("il test 3 dovesse collassare.\n")
    print(f"  {'settore':<9}{'X':<12}{'Y':<12}{'Z':<12}{'Y/X'}")
    print("  " + "-" * 52)
    for s in SECTORS:
        vals = {}
        for d in DIRECTIONS:
            a, b = T10.get((s, d)), T90.get((s, d))
            vals[d] = b / a if (a and b) else None
        cells = "".join(f"{v:<12.2f}" if v else f"{'n/d':<12}"
                        for v in (vals["X"], vals["Y"], vals["Z"]))
        ratio = (f"{vals['Y'] / vals['X']:.2f}"
                 if vals["Y"] and vals["X"] else "n/d")
        print(f"  {s:<9}{cells}{ratio}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi", help="CSV con colonne sector,direction,phi")
    ap.add_argument("--phi-inline", action="append", default=[],
                    metavar="SET:DIR=VAL",
                    help="porosita' passata a mano, ripetibile")
    ap.add_argument("--length", type=float, default=L,
                    help=f"lunghezza del dominio [m], default {L}. "
                         f"Usare 45 per coerenza con il piano di uscita")
    args = ap.parse_args()

    missing = [k for k, v in K.items() if v is None]
    if missing:
        print("ATTENZIONE: permeabilita' mancanti per "
              f"{', '.join(f'{s}/{d}' for s, d in missing)}.")
        print("Completare il dizionario K prima di usare i risultati.\n")

    test_1_raw_ratios()
    test_2_no_porosity()

    phi = {}
    if args.phi:
        phi.update(load_phi(args.phi))
    phi.update(parse_phi_inline(args.phi_inline))

    if phi:
        test_3_full(phi, args.length)
    else:
        section("3. Normalizzazione completa con porosita'")
        print("Saltata: nessuna porosita' fornita.")
        print("Vedi in fondo allo script come calcolarla.")

    test_4_shape()


if __name__ == "__main__":
    main()


# ====================================================================
# SNIPPET DA AGGIUNGERE A compute_dq_2core.py PER OTTENERE phi
# ====================================================================
#
# compute_dq_2core.py legge gia' le aree S_f e le aperture b_f delle
# fratture del backbone, perche' servono per d_Q. La porosita' e' la
# stessa somma, pesata diversamente:
#
#     d_Q  = (1/V) * (sum S_f Q_f)^2 / (sum S_f Q_f^2)
#     phi  = (1/V) *  sum S_f b_f
#
# Quindi bastano poche righe nel punto in cui gia' cicli sulle fratture:
#
#     phi = sum(f.sa * f.b for f in fracs) / V
#
# dove V e' il volume del dominio (50^3 = 125000 se il lato e' 50 m) e
# fracs e' la stessa lista usata per d_Q.
#
# Per esportarlo in CSV nel formato atteso da questo script:
#
#     import csv
#     with open("phi_backbone.csv", "w", newline="") as fh:
#         w = csv.writer(fh)
#         w.writerow(["sector", "direction", "phi"])
#         for (sector, direction), value in results.items():
#             w.writerow([sector, direction, f"{value:.6e}"])
#
# CONTROLLO DI SANITA': con aperture dell'ordine di 1e-3 m e P32 del
# backbone fra 0.1 e 0.7 1/m, phi dovrebbe risultare fra 1e-5 e 1e-3.
# Valori fuori da questo intervallo indicano un errore di unita' o di
# normalizzazione.
