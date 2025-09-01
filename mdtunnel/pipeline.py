from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from openmm import unit
from openmm import LocalEnergyMinimizer
from openmm.app import PDBFile, Simulation, AmberPrmtopFile, AmberInpcrdFile, Topology
from openmm import Platform

from .spheres import Sphere
from .system_builder_amber import build_system_from_amber_with_restraints


@dataclass
class RunOptions:
    kcom: float = 2000.0
    kpos_protein: float = 10000.0
    kpos_backbone: float | None = None
    kpos_sidechain: float | None = None
    platform: Optional[str] = None  # CUDA|OpenCL|CPU
    reparam_per_step: bool = False
    translate_first_to_center: bool = True
    exiting: bool = True  # True: start at active site (last sphere) and move outward
    active_center_nm: Optional["np.ndarray"] = None  # Active site center in nm
    min_tolerance_kj_per_nm: float = 0.01  # minimization tolerance (kJ/mol/nm)
    com_target_nm: float = 0.01  # target COM distance (nm)
    com_max_k: float = 5e6  # maximum COM k allowed (kJ/mol/nm^2)
    com_scale: float = 10.0  # scale factor when off-target
    min_max_iter: int = 20000
    kcom_ramp_scale: float = 5.0
    pre_min_md_ps: float = 0.0
    relax_pocket_margin_nm: float = 0.1
    relax_pocket_scale: float = 0.2


def run_pipeline(
    protein_path: str,
    ligand_path: str,
    spheres: List[Sphere],
    forcefield: "openmm.app.ForceField",
    ligand_mol: Optional["openff.toolkit.topology.Molecule"],
    outdir: str,
    options: RunOptions,
):
    from .system_builder import build_system_with_restraints

    os.makedirs(outdir, exist_ok=True)
    frames_dir = os.path.join(outdir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Build initial system
    build = build_system_with_restraints(
        protein_path=protein_path,
        ligand_path=ligand_path,
        forcefield=forcefield,
        ligand_mol=ligand_mol,
        kpos_protein=options.kpos_protein,
        kpos_backbone=options.kpos_backbone,
        kpos_sidechain=options.kpos_sidechain,
        kcom=options.kcom,
    )

    # Prepare Simulation
    platform = Platform.getPlatformByName(options.platform) if options.platform else None
    integrator = _make_integrator()
    if platform is None:
        sim = Simulation(build.topology, build.system, integrator)
    else:
        sim = Simulation(build.topology, build.system, integrator, platform)
    sim.context.setPositions(build.positions)

    # Helper to set ghost center
    def set_ghost_center(center_nm: np.ndarray, r_nm: float):
        # Update per-particle parameters of ghost anchor (index 0 because only one particle added to that force)
        try:
            k_anchor = build.ghost_anchor_force.getParticleParameters(0)[3]
        except Exception:
            from openmm import unit as omm_unit
            k_anchor = 100000.0 * omm_unit.kilojoule_per_mole / omm_unit.nanometer**2
        build.ghost_anchor_force.setParticleParameters(
            0,
            build.ghost_index,
            [
                center_nm[0] * unit.nanometer,
                center_nm[1] * unit.nanometer,
                center_nm[2] * unit.nanometer,
                k_anchor,
            ],
        )
        build.ghost_anchor_force.updateParametersInContext(sim.context)
        # Update COM restraint current radius and ramp
        try:
            groups, params = build.com_restraint.getBondParameters(0)
            k_base = params[0]
            try:
                k_base_val = float(k_base.value_in_unit(unit.kilojoule_per_mole / unit.nanometer**2))
            except Exception:
                k_base_val = float(k_base)
            k_ramp = (k_base_val * options.kcom_ramp_scale) * unit.kilojoule_per_mole / unit.nanometer**2
            build.com_restraint.setBondParameters(0, groups, [k_base, k_ramp, r_nm * unit.nanometer])
            build.com_restraint.updateParametersInContext(sim.context)
        except Exception:
            pass

    energies = []
    com_dists_nm: List[float] = []
    ligand_pos_list_run: List[list] = []

    # Initial step: optionally translate to first sphere center
    first_center = spheres[0].center_nm()
    if options.translate_first_to_center:
        _translate_ligand_to_center(sim, build.ligand_atom_indices, first_center)
    set_ghost_center(first_center, spheres[0].r * 0.1)
    _update_protein_restraints_for_pocket(sim, build, first_center, spheres[0].r * 0.1, options)
    _update_protein_restraints_for_pocket(sim, build, first_center, spheres[0].r * 0.1, options)
    import time
    step_times = []
    t0 = time.time()
    # Optional restrained MD before minimization to smooth contacts
    if options.pre_min_md_ps and options.pre_min_md_ps > 0:
        nsteps = int(round(options.pre_min_md_ps / 0.001))  # timestep 0.001 ps
        if nsteps > 0:
            _run_safe_md(sim, nsteps)
    LocalEnergyMinimizer.minimize(
            sim.context, options.min_tolerance_kj_per_nm * unit.kilojoule_per_mole / unit.nanometer, options.min_max_iter
        )
    step_times.append(time.time() - t0)
    _write_frame(sim, os.path.join(frames_dir, f"step_0000.pdb"))
    e0 = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    energies.append((0, *first_center, spheres[0].r * 0.1, e0))
    com0 = _compute_ligand_com_nm(sim, build.ligand_atom_indices)
    com_dists_nm.append(float(np.linalg.norm(com0 - first_center)))
    _tighten_com_if_needed(sim, build, first_center, options)
    com0 = _compute_ligand_com_nm(sim, build.ligand_atom_indices)
    com_dists_nm[-1] = float(np.linalg.norm(com0 - first_center))
    ligand_pos_list_run.append(_get_ligand_positions(sim, build.ligand_atom_indices))

    for i, sph in enumerate(spheres[1:], start=1):
        center = sph.center_nm()
        # If re-parameterizing per step, rebuild system with updated ligand FF
        if options.reparam_per_step:
            state = sim.context.getState(getPositions=True)
            positions = state.getPositions(asNumpy=True)
            # Rebuild system with same FF (charges/templates recomputed via generators)
            build = build_system_with_restraints(
                protein_path=protein_path,
                ligand_path=ligand_path,
                forcefield=forcefield,
                ligand_mol=ligand_mol,
                kpos_protein=options.kpos_protein,
                kcom=options.kcom,
            )
            # Reinit simulation with new system and old positions
            if platform is None:
                sim = Simulation(build.topology, build.system, integrator)
            else:
                sim = Simulation(build.topology, build.system, integrator, platform)
            sim.context.setPositions(positions)

        # Move ligand COM to sphere center while keeping orientation
        _translate_ligand_to_center(sim, build.ligand_atom_indices, center)
        set_ghost_center(center, sph.r * 0.1)
        _update_protein_restraints_for_pocket(sim, build, center, sph.r * 0.1, options)
        _update_protein_restraints_for_pocket(sim, build, center, sph.r * 0.1, options)

        t0 = time.time()
        if options.pre_min_md_ps and options.pre_min_md_ps > 0:
            nsteps = int(round(options.pre_min_md_ps / 0.001))
            if nsteps > 0:
                _run_safe_md(sim, nsteps)
        LocalEnergyMinimizer.minimize(
            sim.context, options.min_tolerance_kj_per_nm * unit.kilojoule_per_mole / unit.nanometer, options.min_max_iter
        )
        step_times.append(time.time() - t0)
        _write_frame(sim, os.path.join(frames_dir, f"step_{i:04d}.pdb"))
        e = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        energies.append((i, *center, sph.r * 0.1, e))
        ligand_pos_list_run.append(_get_ligand_positions(sim, build.ligand_atom_indices))
        comi = _compute_ligand_com_nm(sim, build.ligand_atom_indices)
        com_dists_nm.append(float(np.linalg.norm(comi - center)))
        _tighten_com_if_needed(sim, build, center, options)
        comi = _compute_ligand_com_nm(sim, build.ligand_atom_indices)
        com_dists_nm[-1] = float(np.linalg.norm(comi - center))

    # Write energy profile
    df = pd.DataFrame(energies, columns=["step", "x_nm", "y_nm", "z_nm", "r_nm", "E_kJ_mol"])
    # Compute distance from active site and delta-E relative to active site energy
    if options.active_center_nm is not None:
        act = np.asarray(options.active_center_nm, dtype=float)
    else:
        # Fallback: active site is first if exiting else last in traversal
        act = np.array([energies[0][1], energies[0][2], energies[0][3]]) if options.exiting else np.array([energies[-1][1], energies[-1][2], energies[-1][3]])
    centers = df[["x_nm", "y_nm", "z_nm"]].to_numpy()
    dists = np.linalg.norm(centers - act[None, :], axis=1)
    ref_idx = 0 if options.exiting else (len(df) - 1)
    dEs = df["E_kJ_mol"].to_numpy() - float(df.iloc[ref_idx]["E_kJ_mol"])
    df["distance_nm"] = dists
    df["distance_A"] = dists * 10.0
    df["dE_kJ_mol"] = dEs
    # also output kcal/mol
    df["E_kcal_mol"] = df["E_kJ_mol"] / 4.184
    df["dE_kcal_mol"] = df["dE_kJ_mol"] / 4.184
    df.to_csv(os.path.join(outdir, "energy_profile.csv"), index=False)
    pd.DataFrame({"step": list(range(len(step_times))), "min_time_s": step_times}).to_csv(
        os.path.join(outdir, "timings_minimization.csv"), index=False
    )
    _plot_energy_profile(outdir)
    _write_ligand_trajectory(os.path.join(outdir, "substrate_traj.pdb"), build.topology, build.ligand_atom_indices, ligand_pos_list_run)
    # Write COM distance to sphere centers
    pd.DataFrame({
        "step": list(range(len(com_dists_nm))),
        "com_to_sphere_nm": com_dists_nm,
        "com_to_sphere_A": [d*10.0 for d in com_dists_nm],
    }).to_csv(os.path.join(outdir, "com_to_sphere.csv"), index=False)


def run_pipeline_amber(
    prmtop_path: str,
    inpcrd_path: str,
    spheres: List[Sphere],
    outdir: str,
    options: RunOptions,
    ligand_resname: str = "LIG",
):
    os.makedirs(outdir, exist_ok=True)
    frames_dir = os.path.join(outdir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    build = build_system_from_amber_with_restraints(
        prmtop_path=prmtop_path,
        inpcrd_path=inpcrd_path,
        ligand_resname=ligand_resname,
        kpos_protein=options.kpos_protein,
        kcom=options.kcom,
    )

    integrator = _make_integrator()
    platform = Platform.getPlatformByName(options.platform) if options.platform else None
    if platform is None:
        sim = Simulation(build.topology, build.system, integrator)
    else:
        sim = Simulation(build.topology, build.system, integrator, platform)
    sim.context.setPositions(build.positions)

    def set_ghost_center(center_nm: np.ndarray, r_nm: float):
        try:
            k_anchor = build.ghost_anchor_force.getParticleParameters(0)[3]
        except Exception:
            from openmm import unit as omm_unit
            k_anchor = 100000.0 * omm_unit.kilojoule_per_mole / omm_unit.nanometer**2
        build.ghost_anchor_force.setParticleParameters(
            0,
            build.ghost_index,
            [
                center_nm[0] * unit.nanometer,
                center_nm[1] * unit.nanometer,
                center_nm[2] * unit.nanometer,
                k_anchor,
            ],
        )
        build.ghost_anchor_force.updateParametersInContext(sim.context)

    energies = []
    com_dists_nm: List[float] = []
    ligand_pos_list_run: List[list] = []

    first_center = spheres[0].center_nm()
    # If ligand already docked, do not translate; default to keep orientation
    if options.translate_first_to_center:
        _translate_ligand_to_center(sim, build.ligand_atom_indices, first_center)
    set_ghost_center(first_center, spheres[0].r * 0.1)
    import time
    step_times = []
    t0 = time.time()
    if options.pre_min_md_ps and options.pre_min_md_ps > 0:
        nsteps = int(round(options.pre_min_md_ps / 0.001))
        if nsteps > 0:
            _run_safe_md(sim, nsteps)
    LocalEnergyMinimizer.minimize(sim.context, options.min_tolerance_kj_per_nm * unit.kilojoule_per_mole / unit.nanometer, options.min_max_iter)
    step_times.append(time.time() - t0)
    _write_frame(sim, os.path.join(frames_dir, f"step_0000.pdb"))
    e0 = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    energies.append((0, *first_center, spheres[0].r * 0.1, e0))
    com0 = _compute_ligand_com_nm(sim, build.ligand_atom_indices)
    com_dists_nm.append(float(np.linalg.norm(com0 - first_center)))
    _tighten_com_if_needed(sim, build, first_center, options)
    com0 = _compute_ligand_com_nm(sim, build.ligand_atom_indices)
    com_dists_nm[-1] = float(np.linalg.norm(com0 - first_center))
    com0 = _compute_ligand_com_nm(sim, build.ligand_atom_indices)
    com_dists_nm.append(float(np.linalg.norm(com0 - first_center)))
    ligand_pos_list_run.append(_get_ligand_positions(sim, build.ligand_atom_indices))

    for i, sph in enumerate(spheres[1:], start=1):
        center = sph.center_nm()
        _translate_ligand_to_center(sim, build.ligand_atom_indices, center)
        set_ghost_center(center, sph.r * 0.1)
        t0 = time.time()
        if options.pre_min_md_ps and options.pre_min_md_ps > 0:
            nsteps = int(round(options.pre_min_md_ps / 0.001))
            if nsteps > 0:
                _run_safe_md(sim, nsteps)
        LocalEnergyMinimizer.minimize(sim.context, options.min_tolerance_kj_per_nm * unit.kilojoule_per_mole / unit.nanometer, options.min_max_iter)
        step_times.append(time.time() - t0)
        _write_frame(sim, os.path.join(frames_dir, f"step_{i:04d}.pdb"))
        e = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        energies.append((i, *center, sph.r * 0.1, e))
        comi = _compute_ligand_com_nm(sim, build.ligand_atom_indices)
        com_dists_nm.append(float(np.linalg.norm(comi - center)))
        _tighten_com_if_needed(sim, build, center, options)
        comi = _compute_ligand_com_nm(sim, build.ligand_atom_indices)
        com_dists_nm[-1] = float(np.linalg.norm(comi - center))
        comi = _compute_ligand_com_nm(sim, build.ligand_atom_indices)
        com_dists_nm.append(float(np.linalg.norm(comi - center)))
        ligand_pos_list_run.append(_get_ligand_positions(sim, build.ligand_atom_indices))

    df = pd.DataFrame(energies, columns=["step", "x_nm", "y_nm", "z_nm", "r_nm", "E_kJ_mol"])
    # Compute distance from active site and delta-E relative to active site energy
    if options.active_center_nm is not None:
        act = np.asarray(options.active_center_nm, dtype=float)
    else:
        act = np.array([energies[0][1], energies[0][2], energies[0][3]]) if options.exiting else np.array([energies[-1][1], energies[-1][2], energies[-1][3]])
    centers = df[["x_nm", "y_nm", "z_nm"]].to_numpy()
    dists = np.linalg.norm(centers - act[None, :], axis=1)
    ref_idx = 0 if options.exiting else (len(df) - 1)
    dEs = df["E_kJ_mol"].to_numpy() - float(df.iloc[ref_idx]["E_kJ_mol"])
    df["distance_nm"] = dists
    df["distance_A"] = dists * 10.0
    df["dE_kJ_mol"] = dEs
    df["E_kcal_mol"] = df["E_kJ_mol"] / 4.184
    df["dE_kcal_mol"] = df["dE_kJ_mol"] / 4.184
    df.to_csv(os.path.join(outdir, "energy_profile.csv"), index=False)
    pd.DataFrame({"step": list(range(len(step_times))), "min_time_s": step_times}).to_csv(
        os.path.join(outdir, "timings_minimization.csv"), index=False
    )
    _plot_energy_profile(outdir)
    pd.DataFrame({
        "step": list(range(len(com_dists_nm))),
        "com_to_sphere_nm": com_dists_nm,
        "com_to_sphere_A": [d*10.0 for d in com_dists_nm],
    }).to_csv(os.path.join(outdir, "com_to_sphere.csv"), index=False)
    _write_ligand_trajectory(os.path.join(outdir, "substrate_traj.pdb"), build.topology, build.ligand_atom_indices, ligand_pos_list_run)


def _make_integrator():
    # Minimization only; integrator choice irrelevant but required by Simulation
    from openmm import LangevinIntegrator

    return LangevinIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 0.001 * unit.picoseconds)


def _write_frame(sim: Simulation, path: str):
    positions = sim.context.getState(getPositions=True).getPositions(asNumpy=True)
    # Some systems include extra particles (e.g., ghost); ensure positions match topology atoms
    n_atoms = sim.topology.getNumAtoms()
    positions_to_write = positions[:n_atoms, :]
    with open(path, "w") as f:
        PDBFile.writeFile(sim.topology, positions_to_write, f)


def _get_ligand_positions(sim: Simulation, ligand_indices: List[int]):
    positions = sim.context.getState(getPositions=True).getPositions()
    return [positions[i] for i in ligand_indices]


def _write_ligand_trajectory(path: str, full_top: Topology, ligand_indices: List[int], models: List[list]):
    # Build ligand-only topology (single residue)
    lig_top = Topology()
    chain = lig_top.addChain()
    idx_to_atom = {a.index: a for a in full_top.atoms()}
    resname = None
    for idx in ligand_indices:
        resname = idx_to_atom[idx].residue.name
        if resname:
            break
    if not resname:
        resname = "LIG"
    residue = lig_top.addResidue(resname, chain, id="1")
    for idx in ligand_indices:
        at = idx_to_atom[idx]
        lig_top.addAtom(at.name, at.element, residue)

    def _to_quantity(m):
        # Convert per-atom positions (Vec3*unit) to Quantity[N,3] in nm
        arr = np.array([[v[0].value_in_unit(unit.nanometer),
                         v[1].value_in_unit(unit.nanometer),
                         v[2].value_in_unit(unit.nanometer)] for v in m], dtype=float)
        return arr * unit.nanometer

    # Prepare static atom metadata from topology
    atoms = list(lig_top.atoms())
    residue = list(lig_top.residues())[0]
    chain_id = "A"
    resname = residue.name if residue.name else "LIG"

    def _write_model(fh, idx, pos_q):
        fh.write(f"MODEL     {idx}\n")
        for i, at in enumerate(atoms, start=1):
            x = pos_q[i-1, 0].value_in_unit(unit.angstrom)
            y = pos_q[i-1, 1].value_in_unit(unit.angstrom)
            z = pos_q[i-1, 2].value_in_unit(unit.angstrom)
            name = at.name[:4]
            elem = (at.element.symbol if at.element is not None else " ").rjust(2)
            fh.write(f"HETATM{i:5d} {name:<4s}{resname:>3s} {chain_id}{1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem}\n")
        fh.write(f"TER\nENDMDL\n")

    with open(path, "w") as f:
        if not models:
            return
        for idx, m in enumerate(models, start=1):
            _write_model(f, idx, _to_quantity(m))
        f.write("END\n")


def _compute_ligand_com_nm(sim: Simulation, ligand_indices: List[int]) -> np.ndarray:
    # Return ligand COM position in nm, mass-weighted if masses available
    state = sim.context.getState(getPositions=True)
    pos_q = state.getPositions(asNumpy=True)
    pos = pos_q.value_in_unit(unit.nanometer)
    masses = np.array([sim.system.getParticleMass(i).value_in_unit(unit.dalton) for i in ligand_indices])
    coords = pos[ligand_indices, :]
    if masses.sum() == 0:
        com = coords.mean(axis=0)
    else:
        com = (coords * masses[:, None]).sum(axis=0) / masses.sum()
    return np.asarray(com, dtype=float)


def _translate_ligand_to_center(sim: Simulation, ligand_indices: List[int], center_nm: np.ndarray):
    state = sim.context.getState(getPositions=True)
    pos_q = state.getPositions(asNumpy=True)
    pos = pos_q.value_in_unit(unit.nanometer)
    # Compute ligand COM (mass-weighted) in nm
    masses = np.array([sim.system.getParticleMass(i).value_in_unit(unit.dalton) for i in ligand_indices])
    coords = pos[ligand_indices, :]
    if masses.sum() == 0:
        com = coords.mean(axis=0)
    else:
        com = (coords * masses[:, None]).sum(axis=0) / masses.sum()
    shift = center_nm - com
    pos[ligand_indices, :] = coords + shift[None, :]
    sim.context.setPositions(pos * unit.nanometer)


def _plot_energy_profile(outdir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        df = pd.read_csv(os.path.join(outdir, "energy_profile.csv"))
        plt.figure(figsize=(6, 4))
        x = df.get("distance_A", (df.get("distance_nm", df["step"]) * 10.0)).to_numpy()
        y_ff = df.get("dE_kcal_mol", (df["E_kJ_mol"] / 4.184)).to_numpy()
        plt.plot(x, y_ff, marker="o")
        plt.xlabel("Distance from active site (Å)")
        plt.ylabel("Δ Potential energy (kcal/mol)")
        plt.title("Energy profile along tunnel")
        # single profile only (FF)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "energy_profile.png"), dpi=150)
        plt.close()
    except Exception:
        pass


def _run_safe_md(sim: Simulation, nsteps: int):
    # Run MD in chunks and revert if instability occurs
    from math import isnan
    # stash current positions
    pos_before = sim.context.getState(getPositions=True).getPositions(asNumpy=True)
    steps_left = nsteps
    chunk = 200
    while steps_left > 0:
        this = min(chunk, steps_left)
        sim.step(this)
        steps_left -= this
        pos = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        # check for NaNs or huge coords
        import numpy as np
        if not np.isfinite(pos).all() or np.max(np.abs(pos)) > 100.0:
            # revert and abort MD
            sim.context.setPositions(pos_before)
            break


def _update_protein_restraints_for_pocket(sim: Simulation, build, center_nm: np.ndarray, r_nm: float, opts: RunOptions):
    try:
        pos_rest = build.protein_pos_restraint
    except Exception:
        return
    # Build atom name map
    idx_to_name = {a.index: a.name for a in sim.topology.atoms()}
    from openmm import unit as omm_unit
    # Current positions for distance check
    pos_nm = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(omm_unit.nanometer)
    margin = float(opts.relax_pocket_margin_nm)
    k_bb = (opts.kpos_backbone if opts.kpos_backbone is not None else opts.kpos_protein)
    k_sc = (opts.kpos_sidechain if opts.kpos_sidechain is not None else opts.kpos_protein)
    k_bb_q = k_bb * omm_unit.kilojoule_per_mole / omm_unit.nanometer**2
    k_sc_q = k_sc * omm_unit.kilojoule_per_mole / omm_unit.nanometer**2
    k_bb_rel = (k_bb * opts.relax_pocket_scale) * omm_unit.kilojoule_per_mole / omm_unit.nanometer**2
    k_sc_rel = (k_sc * opts.relax_pocket_scale) * omm_unit.kilojoule_per_mole / omm_unit.nanometer**2
    # Target positions
    try:
        targets = build.protein_rest_targets_nm
    except Exception:
        targets = None
    N = sim.system.getNumParticles()
    bb_names = {"N", "CA", "C", "O", "OXT"}
    # Iterate over all particles; skip ligand and ghost
    lig = set(getattr(build, 'ligand_atom_indices', []))
    ghost = getattr(build, 'ghost_index', -1)
    for idx in range(N):
        if idx in lig or idx == ghost:
            continue
        name = idx_to_name.get(idx, "")
        is_bb = name in bb_names
        d = float(np.linalg.norm(pos_nm[idx] - center_nm))
        use_relax = (d <= (r_nm + margin))
        kq = (k_bb_rel if is_bb else k_sc_rel) if use_relax else (k_bb_q if is_bb else k_sc_q)
        # target x0,y0,z0
        if targets is not None:
            x0, y0, z0 = targets[idx]
        else:
            x0, y0, z0 = (pos_nm[idx][0]*omm_unit.nanometer, pos_nm[idx][1]*omm_unit.nanometer, pos_nm[idx][2]*omm_unit.nanometer)
        pos_rest.setParticleParameters(idx, [x0, y0, z0, kq])
    pos_rest.updateParametersInContext(sim.context)


def _tighten_com_if_needed(sim: Simulation, build, center_nm: np.ndarray, opts: RunOptions):
    # adaptively increase COM restraint k if COM is farther than target
    # read current bond settings
    try:
        groups, params = build.com_restraint.getBondParameters(0)
        # params: [k_base, k_ramp, r_nm]
        try:
            k_curr = float(params[0].value_in_unit(unit.kilojoule_per_mole / unit.nanometer**2))
        except Exception:
            k_curr = float(params[0])
    except Exception:
        return
    # compute current COM distance
    dist = np.linalg.norm(_compute_ligand_com_nm(sim, build.ligand_atom_indices) - center_nm)
    iters = 0
    while dist > opts.com_target_nm and k_curr < opts.com_max_k:
        # increase k and apply
        k_curr = min(k_curr * opts.com_scale, opts.com_max_k)
        kq = k_curr * unit.kilojoule_per_mole / unit.nanometer**2
        build.com_restraint.setBondParameters(0, groups, [kq, params[1], params[2]])
        build.com_restraint.updateParametersInContext(sim.context)
        # re-minimize
        LocalEnergyMinimizer.minimize(sim.context, opts.min_tolerance_kj_per_nm * unit.kilojoule_per_mole / unit.nanometer, 0)
        dist = np.linalg.norm(_compute_ligand_com_nm(sim, build.ligand_atom_indices) - center_nm)
        iters += 1
        if iters > 10:
            break
