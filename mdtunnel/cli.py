from __future__ import annotations

import argparse
import json
import os
import time

from .param import ParamOptions, build_forcefield_and_generators
from .spheres import load_spheres
from .pipeline import RunOptions, run_pipeline, run_pipeline_amber
from .docking import run_vina_docking, VinaOptions, run_vina_rescore_frames
from .amber_prep import prep_complex_with_ambertools, AmberPrepOptions, pre_clean_protein_pdb
from .ligand_prep import protonate_ligand


def _merge_bidirectional(outdir: str, enter_dir: str, exit_dir: str, orig_spheres):
    import shutil
    import numpy as np
    import pandas as pd

    df_en = pd.read_csv(os.path.join(enter_dir, "energy_profile.csv"))
    df_ex = pd.read_csv(os.path.join(exit_dir, "energy_profile.csv"))
    n = len(orig_spheres)
    if len(df_en) != n or len(df_ex) != n:
        n = min(len(df_en), len(df_ex), n)
        df_en = df_en.iloc[:n]
        df_ex = df_ex.iloc[:n]
        orig_spheres = orig_spheres[:n]

    rows = []
    frames_dir = os.path.join(outdir, "frames")
    if os.path.isdir(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)

    traj_path = os.path.join(outdir, "substrate_traj.pdb")
    with open(traj_path, "w") as traj:
        for j in range(n):
            ex_step = n - 1 - j
            en_step = j
            e_en = float(df_en.iloc[en_step]["E_kJ_mol"]) if "E_kJ_mol" in df_en.columns else float("inf")
            e_ex = float(df_ex.iloc[ex_step]["E_kJ_mol"]) if "E_kJ_mol" in df_ex.columns else float("inf")
            use = "enter" if e_en <= e_ex else "exit"
            e_sel = e_en if use == "enter" else e_ex
            act = np.array([orig_spheres[-1].x, orig_spheres[-1].y, orig_spheres[-1].z], dtype=float)
            here = np.array([orig_spheres[j].x, orig_spheres[j].y, orig_spheres[j].z], dtype=float)
            dist_A = float(np.linalg.norm(here - act))
            rows.append({
                "step": j,
                "distance_A": dist_A,
                "E_kJ_mol": e_sel,
                "E_kcal_mol": e_sel / 4.184,
                "source": use,
            })
            frame_sel = os.path.join(enter_dir if use == "enter" else exit_dir, "frames", f"step_{(en_step if use=='enter' else ex_step):04d}.pdb")
            shutil.copyfile(frame_sel, os.path.join(frames_dir, f"step_{j:04d}.pdb"))
            for line in open(frame_sel):
                if (line.startswith("ATOM") or line.startswith("HETATM")) and line[17:20].strip() == "LIG":
                    traj.write(line)
            traj.write("TER\n")

    dfm = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
    dfm["dE_kJ_mol"] = dfm["E_kJ_mol"] - float(dfm.iloc[0]["E_kJ_mol"]) if len(dfm) else 0.0
    dfm["dE_kcal_mol"] = dfm["dE_kJ_mol"]/4.184
    # Simple spike smoothing: replace isolated spikes larger than 5 kcal/mol above neighbor avg
    try:
        y = dfm["dE_kcal_mol"].to_numpy().copy()
        for i in range(1, len(y)-1):
            nb = 0.5*(y[i-1] + y[i+1])
            if y[i] - nb > 5.0:
                y[i] = nb
        dfm["dE_kcal_mol"] = y
        dfm["dE_kJ_mol"] = y * 4.184
    except Exception:
        pass
    dfm.to_csv(os.path.join(outdir, "energy_profile.csv"), index=False)


def _run_bidirectional_and_merge(args, orig_spheres):
    import numpy as np
    # Prepare dirs
    out_enter = os.path.join(args.out, "enter")
    out_exit = os.path.join(args.out, "exit")
    os.makedirs(out_enter, exist_ok=True)
    os.makedirs(out_exit, exist_ok=True)

    s_enter = list(orig_spheres)
    s_exit = list(reversed(orig_spheres))

    # Pre-clean protein for docking/Amber (drop altLocs, standardize residues)
    prep_root = os.path.join(args.out, "amber_prep")
    os.makedirs(prep_root, exist_ok=True)
    protein_clean = pre_clean_protein_pdb(args.protein, prep_root)

    # Optional docking once at first sphere of enter path
    # Separate files for parameterization vs coordinates
    param_ligand_path = args.ligand
    ligand_path = args.ligand  # coordinates file; may be replaced by docked PDB
    translate_first_to_center = True
    timings: dict[str, float | int] = {}
    if args.dock_first:
        first = s_enter[0]
        cx, cy, cz = first.x, first.y, first.z
        box = args.box_size if args.box_size is not None else max(2.0 * float(first.r) + 6.0, 20.0)
        mgl = args.mgltools_root or os.environ.get("MGLTOOLS_ROOT")
        vina_opts = VinaOptions(
            vina_bin=args.vina_bin,
            exhaustiveness=args.exhaustiveness,
            num_modes=args.num_modes,
            box_size=args.box_size,
            mgltools_root=mgl,
        )
        dock_dir = os.path.join(args.out, "docking")
        t0 = time.time()
        docked_pdb = run_vina_docking(protein_clean, args.ligand, (cx, cy, cz), box, dock_dir, vina_opts)
        timings["docking_s"] = time.time() - t0
        ligand_path = docked_pdb
        translate_first_to_center = False

    # Protonate if requested (applies to both paths)
    if not args.keep_ligand_as_is:
        lig_prot = os.path.join(args.out, "ligand_protonated.pdb")
        try:
            param_ligand_path = protonate_ligand(param_ligand_path, lig_prot)
        except Exception as e:
            raise SystemExit(f"Ligand protonation failed: {e}")

    # Build common run options base
    run_opts_base = RunOptions(
        kcom=args.kcom,
        kpos_protein=args.kpos_protein,
        kpos_backbone=args.kpos_backbone,
        kpos_sidechain=args.kpos_sidechain,
        platform=args.platform,
        reparam_per_step=args.reparam_per_step,
        translate_first_to_center=translate_first_to_center,
        active_center_nm=orig_spheres[-1].center_nm(),
        min_tolerance_kj_per_nm=args.min_tol_kj,
        min_max_iter=args.min_max_iter,
        pre_min_md_ps=args.pre_min_md_ps,
        com_target_nm=args.com_target_A * 0.1,
        com_max_k=args.com_max_k,
        com_scale=args.com_scale,
        kcom_ramp_scale=args.kcom_ramp_scale,
        relax_pocket_margin_nm=args.relax_pocket_margin_A * 0.1,
        relax_pocket_scale=args.relax_pocket_scale,
    )

    # Run pipelines
    t1 = time.time()
    if args.ambertools_prep:
        prep_opts = AmberPrepOptions(
            antechamber_bin=args.antechamber_bin,
            parmchk2_bin=args.parmchk2_bin,
            tleap_bin=args.tleap_bin,
            ligand_charge=args.ligand_charge,
            ligand_resname=args.ligand_resname,
            charge_method=getattr(args, 'ligand_charge_method', 'bcc'),
        )
        amber_dir = os.path.join(args.out, "amber_prep")
        prmtop, inpcrd = prep_complex_with_ambertools(protein_clean, ligand_path, amber_dir, prep_opts)
        ro = run_opts_base; ro.exiting = True
        run_pipeline_amber(prmtop, inpcrd, s_exit, out_exit, ro, ligand_resname=args.ligand_resname)
        ro = run_opts_base; ro.exiting = False
        run_pipeline_amber(prmtop, inpcrd, s_enter, out_enter, ro, ligand_resname=args.ligand_resname)
    else:
        param_opts = ParamOptions(protein_ff=args.protein_ff, ligand_param=args.ligand_param)
        ff, ligand_mol, _ = build_forcefield_and_generators(param_opts, param_ligand_path)
        ro = run_opts_base; ro.exiting = True
        run_pipeline(args.protein, ligand_path, s_exit, ff, ligand_mol, out_exit, ro)
        ro = run_opts_base; ro.exiting = False
        run_pipeline(args.protein, ligand_path, s_enter, ff, ligand_mol, out_enter, ro)
    timings = {"minimization_total_s": time.time() - t1, "spheres": len(orig_spheres)} | timings

    _merge_bidirectional(args.out, out_enter, out_exit, orig_spheres)
    # Merge per-branch timings
    for sub in (out_enter, out_exit):
        tp = os.path.join(sub, "timings_pipeline.json")
        if os.path.isfile(tp):
            try:
                pip = json.load(open(tp))
                for k in ("prepare_system_s", "minimization_total_s"):
                    if k in pip:
                        timings[k] = float(timings.get(k, 0.0)) + float(pip[k])
            except Exception:
                pass
    # Plot merged FF profile
    try:
        from .pipeline import _plot_energy_profile
        _plot_energy_profile(args.out)
    except Exception:
        pass
    open(os.path.join(args.out, "timings.json"), "w").write(json.dumps(timings, indent=2))

def _run_vina_rescoring_and_plot(args, frames_dir: str):
    import os
    import pandas as pd
    import numpy as np
    t0 = time.time()
    mgl = args.mgltools_root or os.environ.get("MGLTOOLS_ROOT")
    vina_opts = VinaOptions(vina_bin=args.vina_bin, mgltools_root=mgl)
    vina_csv = run_vina_rescore_frames(args.protein, frames_dir, args.out, vina_opts)
    vina_rescore_s = time.time() - t0
    try:
        # Avoid system libjpeg/libtiff conflicts during plotting
        os.environ.pop("LD_LIBRARY_PATH", None)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ep_vina = pd.read_csv(vina_csv)
        ep_ff = None
        ff_here = os.path.join(args.out, "energy_profile.csv")
        ff_parent = os.path.join(os.path.dirname(frames_dir), "energy_profile.csv")
        if os.path.isfile(ff_here):
            ep_ff = pd.read_csv(ff_here)
        elif os.path.isfile(ff_parent):
            ep_ff = pd.read_csv(ff_parent)

        if ep_ff is not None and len(ep_ff) > 0:
            n = min(len(ep_ff), len(ep_vina))
            ep_ff = ep_ff.iloc[:n].copy().reset_index(drop=True)
            ep_vina = ep_vina.iloc[:n].copy().reset_index(drop=True)
            x_dist = ep_ff.get("distance_A", (ep_ff.get("distance_nm", ep_ff["step"]) * 10.0)).to_numpy()
            # Normalize FF to first step if not already delta
            y_ff = ep_ff.get("dE_kcal_mol", (ep_ff["E_kJ_mol"] / 4.184)).to_numpy()
            if "dE_kcal_mol" not in ep_ff.columns and len(y_ff) > 0:
                y_ff = y_ff - y_ff[0]
            # Normalize Vina so first step is 0.0 kcal/mol
            y_vina = ep_vina["vina_kcal_mol"].to_numpy()
            if len(y_vina) > 0:
                y_vina = y_vina - y_vina[0]
            fig, ax_l = plt.subplots(figsize=(7.0, 4.5))
            c1, c2 = "tab:blue", "tab:orange"
            ax_l.plot(x_dist, y_ff, color=c1, marker="o", label="FF ΔE")
            ax_l.set_xlabel("Distance from active site (Å)")
            ax_l.set_ylabel("FF ΔE (kcal/mol)", color=c1)
            ax_l.tick_params(axis='y', labelcolor=c1)
            ax_r = ax_l.twinx()
            ax_r.plot(x_dist, y_vina, color=c2, marker="s", label="Vina score")
            ax_r.set_ylabel("Vina (kcal/mol)", color=c2)
            ax_r.tick_params(axis='y', labelcolor=c2)
            lines = ax_l.get_lines() + ax_r.get_lines()
            ax_l.legend(lines, [ln.get_label() for ln in lines], loc="best")
            ax_l.set_title("FF vs Vina along tunnel")
            fig.tight_layout()
            out_png = os.path.join(args.out, "energy_ff_vs_vina.png")
            out_svg = os.path.join(args.out, "energy_ff_vs_vina.svg")
            try:
                plt.savefig(out_png, dpi=150)
            except Exception as e_png:
                try:
                    plt.savefig(out_svg)
                except Exception as e_svg:
                    open(os.path.join(args.out, "plot_debug.txt"), "w").write(f"PNG fail: {e_png}\nSVG fail: {e_svg}\n")
            plt.close(fig)
        else:
            ep_vina = pd.read_csv(vina_csv)
            x = ep_vina["step"].to_numpy(); y = ep_vina["vina_kcal_mol"].to_numpy()
            if len(y) > 0:
                y = y - y[0]
            fig, ax = plt.subplots(figsize=(7.0, 4.0))
            ax.plot(x, y, color="tab:orange", marker="s", label="Vina score")
            ax.set_xlabel("Step index"); ax.set_ylabel("Vina (kcal/mol)"); ax.legend(loc="best")
            fig.tight_layout()
            out_png = os.path.join(args.out, "energy_ff_vs_vina.png")
            out_svg = os.path.join(args.out, "energy_ff_vs_vina.svg")
            try:
                plt.savefig(out_png, dpi=150)
            except Exception as e_png:
                try:
                    plt.savefig(out_svg)
                except Exception as e_svg:
                    open(os.path.join(args.out, "plot_debug.txt"), "w").write(f"PNG fail: {e_png}\nSVG fail: {e_svg}\n")
            plt.close(fig)
    except Exception as e:
        try:
            open(os.path.join(args.out, "plot_debug.txt"), "w").write(f"plotting failed: {e}\n")
        except Exception:
            pass

    # Ensure timings.json exists and has vina_rescore_s
    tj = os.path.join(args.out, "timings.json")
    data = {}
    if os.path.isfile(tj):
        try:
            data = json.load(open(tj))
        except Exception:
            data = {}
    data.setdefault("prepare_system_s", 0.0)
    data.setdefault("docking_s", 0.0)
    data.setdefault("minimization_total_s", 0.0)
    data["vina_rescore_s"] = float(vina_rescore_s)
    open(tj, "w").write(json.dumps(data, indent=2))


def main():
    p = argparse.ArgumentParser(description="MDTunnel: sphere-by-sphere ligand minimization in tunnels")
    p.add_argument("--protein", required=True, help="Protein PDB path")
    p.add_argument("--ligand", required=True, help="Ligand file (PDB/SDF/MOL2)")
    p.add_argument("--spheres", required=True, help="CAVER spheres CSV or PDB (resname SPH)")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--protein-ff", default="amber14/protein.ff14SB.xml", help="Protein forcefield XML (OpenMM)")
    p.add_argument("--ligand-param", default="gaff-am1bcc", choices=["gaff-am1bcc", "amoeba", "qmmm"], help="Ligand parameterization path")
    p.add_argument("--kcom", type=float, default=2000.0, help="COM restraint k (kJ/mol/nm^2)")
    p.add_argument("--kpos-protein", type=float, default=10000.0, help="Protein position restraint k (kJ/mol/nm^2)")
    p.add_argument("--kpos-backbone", type=float, default=None, help="Backbone position restraint k (kJ/mol/nm^2)")
    p.add_argument("--kpos-sidechain", type=float, default=None, help="Sidechain position restraint k (kJ/mol/nm^2)")
    p.add_argument("--platform", default=None, help="OpenMM platform: CUDA|OpenCL|CPU")
    p.add_argument("--reparam-per-step", action="store_true", help="Rebuild ligand parameters every step")
    p.add_argument("--direction", default="exit", choices=["exit", "enter"], help="Traversal direction")
    p.add_argument("--min-tol-kj", type=float, default=0.01, help="Energy minimization tolerance (kJ/mol/nm)")
    p.add_argument("--min-max-iter", type=int, default=20000, help="Max iterations for minimization")
    p.add_argument("--pre-min-md-ps", type=float, default=0.0, help="Restrained MD before minimization (ps)")
    p.add_argument("--com-target-A", type=float, default=0.1, help="Target COM distance from sphere center (Å)")
    p.add_argument("--com-max-k", type=float, default=5e6, help="Max COM k (kJ/mol/nm^2)")
    p.add_argument("--com-scale", type=float, default=10.0, help="Scale factor to increase COM k when off-target")
    p.add_argument("--kcom-ramp-scale", type=float, default=5.0, help="Ramp scale for COM restraint")
    p.add_argument("--relax-pocket-margin-A", type=float, default=1.0, help="Pocket relaxation margin added to sphere radius (Å)")
    p.add_argument("--relax-pocket-scale", type=float, default=0.2, help="Scale factor for protein k inside relaxed pocket")
    # Docking
    p.add_argument("--dock-first", action="store_true", default=True, help="Dock at the first sphere (default)")
    p.add_argument("--no-dock-first", dest="dock_first", action="store_false", help="Disable first-sphere docking")
    p.add_argument("--vina-bin", default="vina", help="Path to vina executable")
    p.add_argument("--mgltools-root", default=None, help="Path to MGLTools root for prepare_* scripts")
    p.add_argument("--exhaustiveness", type=int, default=8)
    p.add_argument("--num-modes", type=int, default=1)
    p.add_argument("--box-size", type=float, default=None, help="Docking box edge length (Å)")
    p.add_argument("--keep-ligand-as-is", action="store_true", help="Do not protonate ligand before parametrization")
    # Merge mode
    p.add_argument("--bidirectional-merge", action="store_true", default=False, help="Run enter+exit and merge per-sphere by FF energy")
    p.add_argument("--single-direction", dest="bidirectional_merge", action="store_false", help="Disable bidirectional merge and run single direction only")
    # AmberTools prep
    p.add_argument("--ambertools-prep", action="store_true", help="Use antechamber/parmchk2/tleap to build complex")
    p.add_argument("--antechamber-bin", default="antechamber")
    p.add_argument("--parmchk2-bin", default="parmchk2")
    p.add_argument("--tleap-bin", default="tleap")
    p.add_argument("--ligand-charge", type=int, default=0)
    p.add_argument("--ligand-charge-method", default="bcc", choices=["bcc","gas"], help="Antechamber charge method (bcc requires sqm; use gas to avoid sqm)")
    p.add_argument("--ligand-resname", default="LIG")
    p.add_argument("--amber-protein-ready", action="store_true", default=False, help="Skip pdb4amber; protein PDB is already Amber-ready")
    # Rescoring
    vg = p.add_mutually_exclusive_group()
    vg.add_argument("--vina-rescore", dest="vina_rescore", action="store_true", default=True, help="Rescore frames with Vina (default)")
    vg.add_argument("--no-vina-rescore", dest="vina_rescore", action="store_false", help="Disable Vina rescoring")
    # Plot-only utility: rescore + plot using existing frames in --out
    p.add_argument("--plot-only", action="store_true", default=False, help="Only (re)compute Vina rescoring and write plot for existing frames in --out")

    args = p.parse_args()

    # Plot-only fast path
    if args.plot_only:
        frames_dir = os.path.join(args.out, "frames")
        if not os.path.isdir(frames_dir):
            raise SystemExit(f"--plot-only: frames directory not found: {frames_dir}")
        if args.vina_rescore:
            _run_vina_rescoring_and_plot(args, frames_dir)
        else:
            try:
                from .pipeline import _plot_energy_profile
                _plot_energy_profile(args.out)
            except Exception:
                pass
        return

    # Load spheres
    spheres = load_spheres(args.spheres)
    if not spheres:
        raise SystemExit("No spheres found in input file")
    orig_spheres = list(spheres)
    if args.direction == "exit" and not args.bidirectional_merge:
        spheres = list(reversed(spheres))

    # Pre-clean protein
    prep_root = os.path.join(args.out, "amber_prep")
    os.makedirs(prep_root, exist_ok=True)
    protein_clean = pre_clean_protein_pdb(args.protein, prep_root)

    # Optionally dock first pose (single-direction only)
    param_ligand_path = args.ligand
    ligand_path = args.ligand  # coordinates file; may be replaced by docked PDB
    translate_first_to_center = True
    timings: dict[str, float | int] = {}
    if args.dock_first and not args.bidirectional_merge:
        first = spheres[0]
        cx, cy, cz = first.x, first.y, first.z
        if args.box_size is not None:
            box = args.box_size
        else:
            box = max(2.0 * float(first.r) + (2.0 if args.direction == "enter" else 6.0), 8.0 if args.direction == "enter" else 20.0)
        mgl = args.mgltools_root or os.environ.get("MGLTOOLS_ROOT")
        vina_opts = VinaOptions(
            vina_bin=args.vina_bin,
            exhaustiveness=args.exhaustiveness,
            num_modes=args.num_modes,
            box_size=args.box_size,
            mgltools_root=mgl,
        )
        dock_dir = os.path.join(args.out, "docking")
        t0 = time.time()
        try:
            docked_pdb = run_vina_docking(
                protein_pdb=protein_clean,
                ligand_file=args.ligand,
                center_xyz_ang=(cx, cy, cz),
                box_size_ang=box,
                outdir=dock_dir,
                options=vina_opts,
            )
            ligand_path = docked_pdb
            timings["docking_s"] = time.time() - t0
            translate_first_to_center = False
        except Exception as e:
            raise SystemExit(f"Docking failed: {e}")

    # Protonate if requested (single-direction only) — apply to parameterization input
    if not args.keep_ligand_as_is and not args.bidirectional_merge:
        lig_prot = os.path.join(args.out, "ligand_protonated.pdb")
        try:
            param_ligand_path = protonate_ligand(param_ligand_path, lig_prot)
        except Exception as e:
            raise SystemExit(f"Ligand protonation failed: {e}")

    # Common run options
    run_opts = RunOptions(
        kcom=args.kcom,
        kpos_protein=args.kpos_protein,
        kpos_backbone=args.kpos_backbone,
        kpos_sidechain=args.kpos_sidechain,
        platform=args.platform,
        reparam_per_step=args.reparam_per_step,
        translate_first_to_center=translate_first_to_center,
        exiting=(args.direction == "exit"),
        active_center_nm=orig_spheres[-1].center_nm(),
        min_tolerance_kj_per_nm=args.min_tol_kj,
        min_max_iter=args.min_max_iter,
        pre_min_md_ps=args.pre_min_md_ps,
        com_target_nm=args.com_target_A * 0.1,
        com_max_k=args.com_max_k,
        com_scale=args.com_scale,
        kcom_ramp_scale=args.kcom_ramp_scale,
        relax_pocket_margin_nm=args.relax_pocket_margin_A * 0.1,
        relax_pocket_scale=args.relax_pocket_scale,
    )
    os.makedirs(args.out, exist_ok=True)

    # Bidirectional path
    if args.bidirectional_merge:
        _run_bidirectional_and_merge(args, orig_spheres)
        if args.vina_rescore:
            _run_vina_rescoring_and_plot(args, os.path.join(args.out, "frames"))
        return

    # Single direction path
    if args.ambertools_prep:
        prep_opts = AmberPrepOptions(
            antechamber_bin=args.antechamber_bin,
            parmchk2_bin=args.parmchk2_bin,
            tleap_bin=args.tleap_bin,
            ligand_charge=args.ligand_charge,
            ligand_resname=args.ligand_resname,
            charge_method=args.ligand_charge_method,
            skip_pdb4amber=args.amber_protein_ready,
        )
        amber_dir = os.path.join(args.out, "amber_prep")
        t0 = time.time()
        prmtop, inpcrd = prep_complex_with_ambertools(protein_clean, ligand_path, amber_dir, prep_opts)
        timings["amber_prep_s"] = time.time() - t0
        t1 = time.time()
        run_pipeline_amber(prmtop, inpcrd, spheres, args.out, run_opts, ligand_resname=args.ligand_resname)
        timings["minimization_total_s"] = time.time() - t1
    else:
        param_opts = ParamOptions(protein_ff=args.protein_ff, ligand_param=args.ligand_param)
        ff, ligand_mol, _ = build_forcefield_and_generators(param_opts, param_ligand_path)
        t1 = time.time()
        run_pipeline(args.protein, ligand_path, spheres, ff, ligand_mol, args.out, run_opts)
        timings["minimization_total_s"] = time.time() - t1

    timings["spheres"] = len(spheres)
    # Merge pipeline timings (if written)
    tp = os.path.join(args.out, "timings_pipeline.json")
    if os.path.isfile(tp):
        try:
            timings.update(json.load(open(tp)))
        except Exception:
            pass
    open(os.path.join(args.out, "timings.json"), "w").write(json.dumps(timings, indent=2))
    if args.vina_rescore:
        _run_vina_rescoring_and_plot(args, os.path.join(args.out, "frames"))


if __name__ == "__main__":
    main()
