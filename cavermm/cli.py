from __future__ import annotations

import argparse
import os

from .param import ParamOptions, build_forcefield_and_generators
from .spheres import load_spheres
from .pipeline import RunOptions, run_pipeline, run_pipeline_amber
from .docking import run_vina_docking, VinaOptions
from .amber_prep import prep_complex_with_ambertools, AmberPrepOptions
from .ligand_prep import protonate_ligand


def _merge_bidirectional(outdir: str, enter_dir: str, exit_dir: str, orig_spheres):
    import pandas as pd, numpy as np, os, shutil
    # Load energy profiles
    df_en = pd.read_csv(os.path.join(enter_dir, "energy_profile.csv"))
    df_ex = pd.read_csv(os.path.join(exit_dir, "energy_profile.csv"))
    n = len(orig_spheres)
    if len(df_en) != n or len(df_ex) != n:
        n = min(len(df_en), len(df_ex), n)
        df_en = df_en.iloc[:n]
        df_ex = df_ex.iloc[:n]
        orig_spheres = orig_spheres[:n]
    # Build merged per original index j
    rows = []
    frames_dir = os.path.join(outdir, "frames")
    if os.path.isdir(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)
    # Merge substrate trajectory
    traj_path = os.path.join(outdir, "substrate_traj.pdb")
    with open(traj_path, "w") as traj:
        for j in range(n):
            ex_step = n - 1 - j
            en_step = j
            e_en = float(df_en.iloc[en_step]["E_kJ_mol"]) if "E_kJ_mol" in df_en.columns else float('inf')
            e_ex = float(df_ex.iloc[ex_step]["E_kJ_mol"]) if "E_kJ_mol" in df_ex.columns else float('inf')
            use = "enter" if e_en <= e_ex else "exit"
            e_sel = e_en if use == "enter" else e_ex
            # Distance from active site in Å based on geometry
            act = np.array([orig_spheres[-1].x, orig_spheres[-1].y, orig_spheres[-1].z], dtype=float)
            here = np.array([orig_spheres[j].x, orig_spheres[j].y, orig_spheres[j].z], dtype=float)
            dist_A = float(np.linalg.norm(here - act))
            rows.append({
                "step": j,
                "distance_A": dist_A,
                "E_kJ_mol": e_sel,
                "E_kcal_mol": e_sel/4.184,
                "source": use,
            })
            # Copy chosen frame into merged frames and append to traj
            frame_sel = os.path.join(enter_dir if use=="enter" else exit_dir, "frames", f"step_{(en_step if use=='enter' else ex_step):04d}.pdb")
            shutil.copyfile(frame_sel, os.path.join(frames_dir, f"step_{j:04d}.pdb"))
            # Append ligand-only records to traj as model
            with open(frame_sel) as f:
                coords = []
                for line in f:
                    if (line.startswith("ATOM") or line.startswith("HETATM")) and line[17:20].strip() == "LIG":
                        traj.write(line)
                traj.write("TER\n")
    # Write merged energy profile with ΔE referenced to active site (step 0)
    dfm = pd.DataFrame(rows)
    dfm = dfm.sort_values("step").reset_index(drop=True)
    dfm["dE_kJ_mol"] = dfm["E_kJ_mol"] - float(dfm.iloc[0]["E_kJ_mol"]) if len(dfm) else 0.0
    dfm["dE_kcal_mol"] = dfm["dE_kJ_mol"]/4.184
    dfm.to_csv(os.path.join(outdir, "energy_profile.csv"), index=False)


def _run_bidirectional_and_merge(args, orig_spheres):
    import time, json, os
    # Rebuild spheres
    orig_s = load_spheres(args.spheres)
    s_enter = orig_s
    s_exit = list(reversed(orig_s))
    out_exit = os.path.join(args.out, "exit")
    out_enter = os.path.join(args.out, "enter")
    os.makedirs(out_exit, exist_ok=True)
    os.makedirs(out_enter, exist_ok=True)
    ligand_path = args.ligand
    translate_first_to_center = True
    timings = {}
    if args.dock_first:
        first = s_enter[0]
        cx, cy, cz = first.x, first.y, first.z
        if args.box_size is not None:
            box = args.box_size
        else:
            box = max(2.0 * float(first.r) + 6.0, 20.0)
        vina_opts = VinaOptions(
            vina_bin=args.vina_bin,
            obabel_bin=args.obabel_bin,
            exhaustiveness=args.exhaustiveness,
            num_modes=args.num_modes,
            box_size=args.box_size,
            mgltools_root=args.mgltools_root,
        )
        dock_dir = os.path.join(args.out, "docking")
        t0 = time.time()
        docked_pdb = run_vina_docking(
            protein_pdb=args.protein,
            ligand_file=args.ligand,
            center_xyz_ang=(cx, cy, cz),
            box_size_ang=box,
            outdir=dock_dir,
            options=vina_opts,
        )
        timings["docking_s"] = time.time() - t0
        ligand_path = docked_pdb
        translate_first_to_center = False

    if not args.keep_ligand_as_is:
        lig_prot = os.path.join(args.out, "ligand_protonated.pdb")
        ligand_path = protonate_ligand(ligand_path, lig_prot)

    run_opts_base = RunOptions(
        kcom=args.kcom,
        kpos_protein=args.kpos_protein,
        platform=args.platform,
        reparam_per_step=args.reparam_per_step,
        translate_first_to_center=translate_first_to_center,
        active_center_nm=orig_s[-1].center_nm(),
        min_tolerance_kj_per_nm=args.min_tol_kj,
        min_max_iter=args.min_max_iter,
        pre_min_md_ps=0.0,
        com_target_nm=args.com_target_A * 0.1,
        com_max_k=args.com_max_k,
        com_scale=args.com_scale,
        kcom_ramp_scale=args.kcom_ramp_scale,
    )

    if args.ambertools_prep:
        prep_opts = AmberPrepOptions(
            antechamber_bin=args.antechamber_bin,
            parmchk2_bin=args.parmchk2_bin,
            tleap_bin=args.tleap_bin,
            ligand_charge=args.ligand_charge,
            ligand_resname=args.ligand_resname,
        )
        amber_dir = os.path.join(args.out, "amber_prep")
        t0 = time.time()
        prmtop, inpcrd = prep_complex_with_ambertools(args.protein, ligand_path, amber_dir, prep_opts)
        timings["amber_prep_s"] = time.time() - t0
        t1 = time.time()
        ro = run_opts_base; ro.exiting = True
        run_pipeline_amber(prmtop, inpcrd, s_exit, out_exit, ro, ligand_resname=args.ligand_resname)
        ro = run_opts_base; ro.exiting = False
        run_pipeline_amber(prmtop, inpcrd, s_enter, out_enter, ro, ligand_resname=args.ligand_resname)
        timings["minimization_total_s"] = time.time() - t1
    else:
        param_opts = ParamOptions(protein_ff=args.protein_ff, ligand_param=args.ligand_param)
        ff, ligand_mol, _gens = build_forcefield_and_generators(param_opts, ligand_path)
        t1 = time.time()
        ro = run_opts_base; ro.exiting = True
        run_pipeline(args.protein, ligand_path, s_exit, ff, ligand_mol, out_exit, ro)
        ro = run_opts_base; ro.exiting = False
        run_pipeline(args.protein, ligand_path, s_enter, ff, ligand_mol, out_enter, ro)
        timings["minimization_total_s"] = time.time() - t1
    timings["spheres"] = len(orig_s)

    _merge_bidirectional(args.out, out_enter, out_exit, orig_s)
    with open(os.path.join(args.out, "timings.json"), "w") as f:
        json.dump(timings, f, indent=2)



def main():
    p = argparse.ArgumentParser(description="Caver-MM: sphere-by-sphere ligand minimization in tunnels")
    p.add_argument("--protein", required=True, help="Protein PDB path")
    p.add_argument("--ligand", required=True, help="Ligand file (PDB/SDF/MOL2)")
    p.add_argument("--spheres", required=True, help="CAVER spheres CSV or PDB (resname SPH)")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--protein-ff", default="amber14/protein.ff14SB.xml", help="Protein forcefield XML (OpenMM)")
    p.add_argument(
        "--ligand-param",
        default="gaff-am1bcc",
        choices=["gaff-am1bcc", "amoeba", "qmmm"],
        help="Ligand parameterization path",
    )
    p.add_argument("--kcom", type=float, default=2000.0, help="COM restraint k (kJ/mol/nm^2)")
    p.add_argument(
        "--kpos-protein",
        type=float,
        default=10000.0,
        help="Protein position restraint k (kJ/mol/nm^2)",
    )
    p.add_argument("--kpos-backbone", type=float, default=None, help="Backbone position restraint k (kJ/mol/nm^2); defaults to --kpos-protein")
    p.add_argument("--kpos-sidechain", type=float, default=None, help="Sidechain position restraint k (kJ/mol/nm^2); defaults to --kpos-protein")
    p.add_argument("--platform", default=None, help="OpenMM platform: CUDA|OpenCL|CPU")
    p.add_argument("--reparam-per-step", action="store_true", help="Rebuild ligand parameters every step")
    p.add_argument(
        "--direction",
        default="exit",
        choices=["exit", "enter"],
        help="Traversal direction: 'exit' = start at active site (last sphere) and move outward (default); 'enter' = start outside (first sphere) and move inward",
    )
    p.add_argument("--min-tol-kj", type=float, default=0.01, help="Energy minimization tolerance (kJ/mol/nm)")
    p.add_argument("--min-max-iter", type=int, default=20000, help="Maximum iterations for energy minimization")
    p.add_argument("--pre-min-md-ps", type=float, default=0.0, help="Run restrained MD for this many picoseconds before each minimization step (0 to disable)")
    p.add_argument("--com-target-A", type=float, default=0.1, help="Target max COM distance from sphere center (Angstrom)")
    p.add_argument("--com-max-k", type=float, default=5e6, help="Max COM restraint k (kJ/mol/nm^2) for tightening")
    p.add_argument("--com-scale", type=float, default=10.0, help="Scaling factor to increase COM k when off-target")
    p.add_argument("--kcom-ramp-scale", type=float, default=5.0, help="Ramp scale factor multiplying base k to steepen near sphere boundary")
    p.add_argument("--relax-pocket-margin-A", type=float, default=1.0, help="Pocket relaxation margin added to sphere radius (Angstrom)")
    p.add_argument("--relax-pocket-scale", type=float, default=0.2, help="Scale factor for protein k inside relaxed pocket (0–1)")
    # Docking options
    p.add_argument("--dock-first", action="store_true", help="Run Vina docking for the first sphere")
    p.add_argument("--vina-bin", default="vina", help="Path to vina executable")
    p.add_argument("--obabel-bin", default="obabel", help="Path to obabel executable")
    p.add_argument("--mgltools-root", default=None, help="Path to MGLTools root to use prepare_receptor4.py/prepare_ligand4.py")
    p.add_argument("--exhaustiveness", type=int, default=8, help="Vina exhaustiveness")
    p.add_argument("--num-modes", type=int, default=1, help="Vina number of poses")
    p.add_argument("--box-size", type=float, default=None, help="Docking box edge length (Angstrom)")
    # --rescore-vina removed to avoid mixing scoring sources in profiles
    p.add_argument("--keep-ligand-as-is", action="store_true", help="Do not protonate ligand before parametrization (not default)")
    # Bidirectional merge (enter + exit)
    p.add_argument("--bidirectional-merge", action="store_true", help="Run both enter and exit trajectories and select per-sphere the lower-energy frame; write merged outputs")
    # AmberTools preparation
    p.add_argument("--ambertools-prep", action="store_true", help="Use antechamber/parmchk2/tleap to parameterize and build complex")
    p.add_argument("--antechamber-bin", default="antechamber", help="Path to antechamber")
    p.add_argument("--parmchk2-bin", default="parmchk2", help="Path to parmchk2")
    p.add_argument("--tleap-bin", default="tleap", help="Path to tleap")
    p.add_argument("--ligand-charge", type=int, default=0, help="Ligand formal charge for antechamber")
    p.add_argument("--ligand-resname", default="LIG", help="Ligand residue name (Amber)")

    args = p.parse_args()

    spheres = load_spheres(args.spheres)
    if not spheres:
        raise SystemExit("No spheres found in input file")
    # Keep original order to identify active site center (assumed last sphere in file)
    orig_spheres = list(spheres)
    if args.bidirectional_merge:
        # We'll handle sphere order explicitly in merge mode
        pass
    if (not args.bidirectional_merge) and args.direction == "exit":
        # Start at active site (last) and move outward
        spheres = list(reversed(spheres))

    # Optionally dock first pose
    ligand_path = args.ligand
    translate_first_to_center = True
    timings = {}
    if args.dock_first:
        first = spheres[0]
        cx, cy, cz = first.x, first.y, first.z
        # Default box size: tighter for 'enter' to constrain to sphere
        if args.box_size is not None:
            box = args.box_size
        else:
            if args.direction == "enter":
                box = max(2.0 * float(first.r) + 2.0, 8.0)
            else:
                box = max(2.0 * float(first.r) + 6.0, 20.0)
        vina_opts = VinaOptions(
            vina_bin=args.vina_bin,
            obabel_bin=args.obabel_bin,
            exhaustiveness=args.exhaustiveness,
            num_modes=args.num_modes,
            box_size=args.box_size,
            mgltools_root=args.mgltools_root,
        )
        dock_dir = os.path.join(args.out, "docking")
        import time
        t0 = time.time()
        try:
            docked_pdb = run_vina_docking(
                protein_pdb=args.protein,
                ligand_file=args.ligand,
                center_xyz_ang=(cx, cy, cz),
                box_size_ang=box,
                outdir=dock_dir,
                options=vina_opts,
            )
        except Exception as e:
            raise SystemExit(f"Docking failed: {e}")
        timings["docking_s"] = time.time() - t0
        ligand_path = docked_pdb
        translate_first_to_center = False

    # Protonate ligand unless user asks to keep as-is
    if not args.keep_ligand_as_is:
        lig_prot = os.path.join(args.out, "ligand_protonated.pdb")
        try:
            ligand_path = protonate_ligand(ligand_path, lig_prot)
        except Exception as e:
            raise SystemExit(f"Ligand protonation failed: {e}")

    run_opts = RunOptions(
        kcom=args.kcom,
        kpos_protein=args.kpos_protein,
        kpos_backbone=args.kpos_backbone,
        kpos_sidechain=args.kpos_sidechain,
        platform=args.platform,
        reparam_per_step=args.reparam_per_step,
        translate_first_to_center=translate_first_to_center,
        exiting=(args.direction == "exit"),
        # Active site center is the last sphere in the original file order
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

    if args.bidirectional_merge:
        _run_bidirectional_and_merge(args, orig_spheres)
        return

    if args.ambertools_prep:
        # Prepare Amber complex using docked (or input) ligand
        prep_opts = AmberPrepOptions(
            antechamber_bin=args.antechamber_bin,
            parmchk2_bin=args.parmchk2_bin,
            tleap_bin=args.tleap_bin,
            ligand_charge=args.ligand_charge,
            ligand_resname=args.ligand_resname,
        )
        amber_dir = os.path.join(args.out, "amber_prep")
        import time, json
        t0 = time.time()
        prmtop, inpcrd = prep_complex_with_ambertools(
                protein_pdb=args.protein,
                ligand_pdb=ligand_path,
                outdir=amber_dir,
                options=prep_opts,
        )
        timings["amber_prep_s"] = time.time() - t0
        t1 = time.time()
        run_pipeline_amber(
                prmtop_path=prmtop,
                inpcrd_path=inpcrd,
                spheres=spheres,
                outdir=args.out,
                options=run_opts,
                ligand_resname=args.ligand_resname,
        )
        timings["minimization_total_s"] = time.time() - t1
        timings["spheres"] = len(spheres)
        with open(os.path.join(args.out, "timings.json"), "w") as f:
            json.dump(timings, f, indent=2)
    else:
        # Use OpenMM forcefield path
        param_opts = ParamOptions(
            protein_ff=args.protein_ff,
            ligand_param=args.ligand_param,
        )
        ff, ligand_mol, _gens = build_forcefield_and_generators(param_opts, ligand_path)
        import time, json
        t1 = time.time()
        run_pipeline(
                protein_path=args.protein,
                ligand_path=ligand_path,
                spheres=spheres,
                forcefield=ff,
                ligand_mol=ligand_mol,
                outdir=args.out,
                options=run_opts,
        )
        timings["minimization_total_s"] = time.time() - t1
        timings["spheres"] = len(spheres)
        with open(os.path.join(args.out, "timings.json"), "w") as f:
            json.dump(timings, f, indent=2)


if __name__ == "__main__":
    main()


def _run_bidirectional_and_merge(args, orig_spheres):
    import time, json, os
    # Rebuild spheres
    orig_s = load_spheres(args.spheres)
    s_enter = orig_s
    s_exit = list(reversed(orig_s))
    out_exit = os.path.join(args.out, "exit")
    out_enter = os.path.join(args.out, "enter")
    os.makedirs(out_exit, exist_ok=True)
    os.makedirs(out_enter, exist_ok=True)
    ligand_path = args.ligand
    translate_first_to_center = True
    timings = {}
    if args.dock_first:
        first = s_enter[0]
        cx, cy, cz = first.x, first.y, first.z
        if args.box_size is not None:
            box = args.box_size
        else:
            box = max(2.0 * float(first.r) + 6.0, 20.0)
        vina_opts = VinaOptions(
            vina_bin=args.vina_bin,
            obabel_bin=args.obabel_bin,
            exhaustiveness=args.exhaustiveness,
            num_modes=args.num_modes,
            box_size=args.box_size,
            mgltools_root=args.mgltools_root,
        )
        dock_dir = os.path.join(args.out, "docking")
        t0 = time.time()
        docked_pdb = run_vina_docking(
            protein_pdb=args.protein,
            ligand_file=args.ligand,
            center_xyz_ang=(cx, cy, cz),
            box_size_ang=box,
            outdir=dock_dir,
            options=vina_opts,
        )
        timings["docking_s"] = time.time() - t0
        ligand_path = docked_pdb
        translate_first_to_center = False

    if not args.keep_ligand_as_is:
        lig_prot = os.path.join(args.out, "ligand_protonated.pdb")
        ligand_path = protonate_ligand(ligand_path, lig_prot)

    run_opts_base = RunOptions(
        kcom=args.kcom,
        kpos_protein=args.kpos_protein,
        platform=args.platform,
        reparam_per_step=args.reparam_per_step,
        translate_first_to_center=translate_first_to_center,
        active_center_nm=orig_s[-1].center_nm(),
        min_tolerance_kj_per_nm=args.min_tol_kj,
        min_max_iter=args.min_max_iter,
        pre_min_md_ps=0.0,
        com_target_nm=args.com_target_A * 0.1,
        com_max_k=args.com_max_k,
        com_scale=args.com_scale,
        kcom_ramp_scale=args.kcom_ramp_scale,
    )

    if args.ambertools_prep:
        prep_opts = AmberPrepOptions(
            antechamber_bin=args.antechamber_bin,
            parmchk2_bin=args.parmchk2_bin,
            tleap_bin=args.tleap_bin,
            ligand_charge=args.ligand_charge,
            ligand_resname=args.ligand_resname,
        )
        amber_dir = os.path.join(args.out, "amber_prep")
        t0 = time.time()
        prmtop, inpcrd = prep_complex_with_ambertools(args.protein, ligand_path, amber_dir, prep_opts)
        timings["amber_prep_s"] = time.time() - t0
        t1 = time.time()
        ro = run_opts_base; ro.exiting = True
        run_pipeline_amber(prmtop, inpcrd, s_exit, out_exit, ro, ligand_resname=args.ligand_resname)
        ro = run_opts_base; ro.exiting = False
        run_pipeline_amber(prmtop, inpcrd, s_enter, out_enter, ro, ligand_resname=args.ligand_resname)
        timings["minimization_total_s"] = time.time() - t1
    else:
        param_opts = ParamOptions(protein_ff=args.protein_ff, ligand_param=args.ligand_param)
        ff, ligand_mol, _gens = build_forcefield_and_generators(param_opts, ligand_path)
        t1 = time.time()
        ro = run_opts_base; ro.exiting = True
        run_pipeline(args.protein, ligand_path, s_exit, ff, ligand_mol, out_exit, ro)
        ro = run_opts_base; ro.exiting = False
        run_pipeline(args.protein, ligand_path, s_enter, ff, ligand_mol, out_enter, ro)
        timings["minimization_total_s"] = time.time() - t1
    timings["spheres"] = len(orig_s)

    _merge_bidirectional(args.out, out_enter, out_exit, orig_s)
    with open(os.path.join(args.out, "timings.json"), "w") as f:
        json.dump(timings, f, indent=2)
