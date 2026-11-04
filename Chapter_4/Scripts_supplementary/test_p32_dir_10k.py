#!/usr/bin/env python3
"""test_p32_dir_fixed_generic.py
------------------------------------
Versione generalizzata e corretta dello script di generazione DFN,
parametrica su --domain e --run.

FIX: i tre backbone X/Y/Z vengono precalcolati UNA VOLTA con
dfn_metrics.py (current_flow_backbone, indipendente), bypassando
DFN.create_graph()/DFN.current_flow_threshold() -- verificati
bacati (X/Y/Z sempre identici) su tutte le ~100 run esistenti.

Genera UNA run pulita per (domain, run_number), con:
  seed = run_number
  home = <root_dominio>/runs/runs_fixed/<run_number>

Uso (dal container dfnWorks):
    apptainer exec "$DFNWORKS_CONTAINER" python3 test_p32_dir_fixed_generic.py \
        --domain low --run 1
"""

import os
import sys
import argparse
import shutil
import glob
import numpy as _np
import shutil as _shutil
import shutil as _sh2
from pydfnworks import *

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)
from dfn_metrics import load_connectivity, build_mother_graph, current_flow_backbone

# ==============================================================================
# --- CONFIG PER DOMINIO ---
# ==============================================================================
DOMAIN_CONFIG = {
    "low": {
        "root": "/hpc/scratch/filippo.porta/runs_10k/low",
        "control_source": "/hpc/scratch/filippo.porta/control_10k",
        "p32": (0.33, 0.22, 0.06),
    },
    "medium": {
        "root": "/hpc/scratch/filippo.porta/runs_10k/medium",
        "control_source": "/hpc/scratch/filippo.porta/control_10k",
        "p32": (0.48, 0.42, 0.2),
    },
    "high": {
        "root": "/hpc/scratch/filippo.porta/runs_10k/high",
        "control_source": "/hpc/scratch/filippo.porta/control_10k",
        "p32": (0.74, 0.33, 0.33),
    },
}

CONTROL_FILES = [
    "PTDFN_control_X.dat", "PTDFN_control_Y.dat", "PTDFN_control_Z.dat",
    "dfn_steady_pressure_x.in", "dfn_steady_pressure_y.in",
    "dfn_steady_pressure_z.in",
]

FRACTURE_FAMILIES = [
    # shape params comuni, cambia solo p32 per dominio
    dict(alpha=1.12, kappa=31.94, probability=0.34, dip=68, strike=63.53,
        max_radius=23.59),
    dict(alpha=1.23, kappa=52.98, probability=0.38, dip=81, strike=90.98,
        max_radius=18.8),
    dict(alpha=1.7, kappa=6.04, probability=0.28, dip=51, strike=144.79,
        max_radius=13.226),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=list(DOMAIN_CONFIG))
    ap.add_argument("--run", required=True, type=int)
    args = ap.parse_args()

    cfg = DOMAIN_CONFIG[args.domain]
    p32_values = cfg["p32"]
    home = os.path.join(cfg["root"], str(args.run))
    seed = 0  # SEMPRE 0: run_number e' solo il nome della cartella,
              # mai il seed. seed=0 genera automaticamente una rete
              # diversa ad ogni chiamata.

    try:
        run_pipeline(args.domain, args.run, home, seed, p32_values)
        # successo: marca esplicitamente la run come valida
        with open(os.path.join(home, "STATUS_OK"), "w") as f:
            f.write("completed successfully\n")
        print(f"\n*** run {args.domain}/{args.run}: SUCCESSO ***")

    except Exception as e:
        import traceback
        err_txt = traceback.format_exc()
        print(f"\n*** ERRORE in {args.domain}/{args.run}: {e} ***")
        print(err_txt)

        # non cancelliamo subito: rinominiamo per poter ispezionare
        # la cartella rotta senza perdere l'informazione sull'errore
        failed_home = home.rstrip(os.sep) + "_FAILED"
        if os.path.isdir(failed_home):
            _shutil.rmtree(failed_home)
        if os.path.isdir(home):
            os.rename(home, failed_home)
            with open(os.path.join(failed_home, "ERROR_LOG.txt"), "w") as f:
                f.write(err_txt)
            print(f"  Cartella rinominata -> {failed_home}")
            print(f"  (ispezionala, poi cancellala manualmente quando confermato)")

        # esce con errore, ma NON blocca gli altri task del job array
        # (sono processi SLURM indipendenti)
        sys.exit(1)


def run_pipeline(domain, run, home, seed, p32_values):
    """Tutta la pipeline di generazione, isolata in una funzione cosi'
    puo' essere avvolta in try/except da main()."""
    os.makedirs(home, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"# domain={domain}  run={run}  seed={seed}")
    print(f"# home={home}")
    print(f"{'#'*70}\n")

    # --------------------------------------------------------------------
    # --- Copia i file di controllo (control PTDFN + dfn_steady_pressure)
    # --- dalla cartella sorgente del dominio dentro home, PRIMA di tutto
    # --------------------------------------------------------------------
    control_source = DOMAIN_CONFIG[domain]["control_source"]
    print(f"Copio i file di controllo da {control_source} -> {home}")
    for fname in CONTROL_FILES:
        src = os.path.join(control_source, fname)
        dst = os.path.join(home, fname)
        if not os.path.isfile(src):
            raise FileNotFoundError(f"File di controllo mancante: {src}")
        _shutil.copy2(src, dst)
        print(f"  copiato: {fname}")

    intensity_value = 13
    jobname = os.path.join(home, f"intensity{intensity_value}")

    # --------------------------------------------------------------------
    # --- DFN Generation ---
    # --------------------------------------------------------------------
    DFN = DFNWORKS(jobname, ncpu=8)
    DFN.params["domainSize"]["value"] = [50, 50, 50]
    DFN.params["domainSizeIncrease"]["value"] = [24, 24, 24]
    DFN.params["ignoreBoundaryFaces"]["value"] = False
    DFN.params["boundaryFaces"]["value"] = [1, 1, 1, 1, 1, 1]
    DFN.params["stopCondition"]["value"] = 1
    DFN.params["rejectsPerFracture"]["value"] = 2
    DFN.params["keepOnlyLargestCluster"]["value"] = 1
    DFN.params["seed"]["value"] = seed
    DFN.params["h"]["value"] = 0.1
    DFN.params["angleOption"]["value"] = "degree"
    DFN.params["orientationOption"]["value"] = 2

    for fam, p32 in zip(FRACTURE_FAMILIES, p32_values):
        DFN.add_fracture_family(
            shape="ell", distribution="tpl",
            alpha=fam["alpha"], kappa=fam["kappa"],
            probability=fam["probability"], aspect=1.0,
            dip=fam["dip"], strike=fam["strike"],
            max_radius=fam["max_radius"], min_radius=1, p32=p32,
            hy_variable="aperture", hy_function="correlated",
            hy_params={"alpha": 5e-4, "beta": 0.5},
        )

    # --------------------------------------------------------------------
    # --- DFN creation ---
    # --------------------------------------------------------------------
    DFN.make_working_directory(delete=True)
    DFN.print_domain_parameters()
    DFN.check_input()
    DFN.create_network()
    DFN.to_pickle()

    dfn_orig_dir = os.path.join(home, f"intensity{intensity_value}")
    dfn_backup_dir = os.path.join(home, f"intensity{intensity_value}_backup")

    if os.path.isdir(dfn_orig_dir):
        if os.path.isdir(dfn_backup_dir):
            _shutil.rmtree(dfn_backup_dir)
        _shutil.copytree(dfn_orig_dir, dfn_backup_dir)
        print(f"Backup DFN originale -> {dfn_backup_dir}")

    _perm_backup = _np.copy(DFN.perm) if hasattr(DFN, "perm") else None
    _aperture_backup = _np.copy(DFN.aperture) if hasattr(DFN, "aperture") else None

    # --------------------------------------------------------------------
    # --- FIX: precalcolo dei tre backbone CORRETTI ----------------------
    # --------------------------------------------------------------------
    print(f"\n{'='*60}\n PRECALCOLO BACKBONE (fix)\n{'='*60}")

    gen_dir = os.path.join(dfn_orig_dir, "dfnGen_output")
    conn_file = os.path.join(gen_dir, "connectivity.dat")
    with open(conn_file) as f:
        n_mother = sum(1 for _ in f)
    edges, _ = load_connectivity(conn_file)

    backbone_files = {}
    for direction in ["x", "y", "z"]:
        Gm = build_mother_graph(edges, n_mother, gen_dir, direction)
        bb = current_flow_backbone(Gm, thrs=1e-16)
        out_path = os.path.join(home, f"backbone_{direction.upper()}.dat")
        with open(out_path, "w") as f:
            for frac_id in sorted(bb):
                f.write(f"{frac_id}\n")
        backbone_files[direction.upper()] = out_path
        print(f"  backbone_{direction.upper()}.dat: {len(bb)} fratture")

    print(f"{'='*60}\n")

    # --------------------------------------------------------------------
    # --- Configurazione direzioni ---
    # --------------------------------------------------------------------
    pressure_in = 2e6
    pressure_out = 1e6
    number_of_particles = 10**4
    graph_base = os.path.join(home, "graph_transport")
    transport_base = os.path.join(home, "transport")
    inflow_pressure = 2e6
    outflow_pressure = 1.95e6

    directions_cfg = {
        "X": {"source": "left", "sink": "right",
             "flow_in": "dfn_steady_pressure_x.in",
             "control": "PTDFN_control_X.dat",
             "boundary_ex": "boundary_left_w.ex"},
        "Y": {"source": "front", "sink": "back",
             "flow_in": "dfn_steady_pressure_y.in",
             "control": "PTDFN_control_Y.dat",
             "boundary_ex": "boundary_front_n.ex"},
        "Z": {"source": "bottom", "sink": "top",
             "flow_in": "dfn_steady_pressure_z.in",
             "control": "PTDFN_control_Z.dat",
             "boundary_ex": "boundary_bottom.ex"},
    }

    K_eff = []

    def get_keff(filename):
        if not os.path.isfile(filename):
            return None
        with open(filename) as f:
            for line in f.readlines():
                if "effective perm" in line:
                    return float(line.split()[-1])
        return None

    def move_graph_outputs(partime_name, frac_seq_name, out_dir):
        search_dirs = [dfn_orig_dir, os.path.join(home, "output_backbone"), home]
        candidates = [f"{partime_name}.hdf5", f"{frac_seq_name}.hdf5",
                     f"{partime_name}.dat", f"{frac_seq_name}.dat", "graph_flow.hdf5"]
        for fname in candidates:
            for sd in search_dirs:
                src_f = os.path.join(sd, fname)
                if os.path.isfile(src_f):
                    shutil.move(src_f, os.path.join(out_dir, fname))
                    print(f"  Salvato: {fname}")
                    break

    # --------------------------------------------------------------------
    # --- Pipeline per direzione ---
    # --------------------------------------------------------------------
    for DIR, dcfg in directions_cfg.items():
        print(f"\n{'='*60}\n DIREZIONE {DIR} ({dcfg['source']} -> {dcfg['sink']})\n{'='*60}")

        print(f"\n[Step 1] Backbone {DIR} (fix)")
        _sh2.copy2(backbone_files[DIR],
                  os.path.join(dfn_orig_dir, "backbone.dat"))
        print(f"  {sum(1 for _ in open(backbone_files[DIR]))} fratture")

        print(f"\n[Step 2] Graph transport {DIR}")
        graph_out = os.path.join(graph_base, DIR)
        os.makedirs(graph_out, exist_ok=True)
        partime_name = f"graph_partime_{DIR}"
        frac_seq_name = f"graph_frac_sequence_{DIR}"
        try:
            G_flow = DFN.run_graph_flow(dcfg["source"], dcfg["sink"],
                                        pressure_in, pressure_out)
            DFN.run_graph_transport(G_flow, number_of_particles,
                                    partime_name, frac_seq_name)
            move_graph_outputs(partime_name, frac_seq_name, graph_out)
            print(f"  Graph transport {DIR} OK")
        except Exception as e:
            print(f"  ATTENZIONE: graph transport {DIR} fallito: {e}")

        print(f"\n[Step 3] BACKBONE mesh + PFLOTRAN + dfnTrans per {DIR}")
        backbone_jobname = os.path.join(home, "output_backbone")
        src_path = dfn_orig_dir + os.sep

        flow_in_file = os.path.join(home, dcfg["flow_in"])
        control_file = os.path.join(home, dcfg["control"])

        BACKBONE = DFNWORKS(
            jobname=backbone_jobname,
            pickle_file=os.path.join(src_path, f"intensity{intensity_value}.pkl"),
            ncpu=8, dfnFlow_file=flow_in_file, dfnTrans_file=control_file,
        )
        BACKBONE.prune_file = os.path.join(src_path, "backbone.dat")
        BACKBONE.path = src_path
        BACKBONE.jobname = backbone_jobname
        BACKBONE.local_jobname = "output_backbone"
        if not BACKBONE.jobname.endswith(os.sep):
            BACKBONE.jobname += os.sep

        BACKBONE.assign_hydraulic_properties()
        BACKBONE.make_working_directory(delete=True)
        BACKBONE.visual_mode = False
        BACKBONE.params["h"]["value"] = 0.1

        BACKBONE.mesh_network(min_dist=1.0, max_dist=5.0, max_resolution_factor=10)
        BACKBONE.clean_up_files_after_prune()
        BACKBONE.dfn_flow()

        backbone_dir = os.path.join(home, "output_backbone")
        BACKBONE.effective_perm(inflow_pressure, outflow_pressure,
                                dcfg["boundary_ex"], DIR.lower())
        keff_src = os.path.join(backbone_dir, "output_backbone_effective_perm.txt")
        keff_renamed = os.path.join(backbone_dir,
                                    f"output_backbone_effective_perm_{DIR.lower()}.txt")
        if os.path.isfile(keff_src):
            os.rename(keff_src, keff_renamed)
        keff_val = get_keff(keff_renamed)
        K_eff.append({DIR: keff_val})
        print(f"  Keff_{DIR} = {keff_val}")

        BACKBONE.parse_pflotran_vtk_python()

        flow_out = os.path.join(home, "flow", DIR)
        os.makedirs(flow_out, exist_ok=True)
        flow_files_to_save = ["darcyvel.dat", "cellinfo.dat", "materialid.dat",
                             "full_mesh_vol_area.uge", "aperture.dat",
                             "params.txt", "poly_info.dat"]
        flow_files_to_save += [os.path.basename(p) for p in
                              glob.glob(os.path.join(backbone_dir, "boundary_*.ex"))]
        for fname in flow_files_to_save:
            src_f = os.path.join(backbone_dir, fname)
            if os.path.isfile(src_f):
                _shutil.copy2(src_f, os.path.join(flow_out, fname))
            else:
                print(f"  [attenzione] {fname} non trovato")
        print(f"  File di flusso {DIR} salvati -> {flow_out}")

        trans_out = os.path.join(transport_base, DIR)
        traj_dir = os.path.join(trans_out, "trajectories")
        os.makedirs(traj_dir, exist_ok=True)

        ctrl_patched = os.path.join(home, f"PTDFN_control_{DIR}_patched.dat")
        with open(control_file, "r") as f:
            lines = f.readlines()
        with open(ctrl_patched, "w") as f:
            for line in lines:
                s = line.strip()
                if s.startswith("out_dir:"):
                    f.write(f"out_dir: {trans_out}\n")
                elif s.startswith("out_path:"):
                    f.write("out_path: trajectories\n")
                else:
                    f.write(line)

        BACKBONE.dfnTrans_file = ctrl_patched
        BACKBONE.local_dfnTrans_file = os.path.basename(ctrl_patched)
        BACKBONE.dfn_trans()
        print(f"  dfnTrans {DIR} OK")

        del BACKBONE

        if os.path.isdir(dfn_backup_dir):
            if os.path.isdir(dfn_orig_dir):
                _shutil.rmtree(dfn_orig_dir)
            _shutil.copytree(dfn_backup_dir, dfn_orig_dir)
            print("  DFN originale ripristinato da backup")

        if _perm_backup is not None:
            DFN.perm = _np.copy(_perm_backup)
        if _aperture_backup is not None:
            DFN.aperture = _np.copy(_aperture_backup)

    del DFN

    print(f"\n{'='*60}\nKeff risultanti (run {run}, {domain}):")
    for k in K_eff:
        print(f"  {k}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
