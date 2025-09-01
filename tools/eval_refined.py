#!/usr/bin/env python3
from __future__ import annotations
import os, argparse
import numpy as np
import pandas as pd
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from openmm.app import Simulation, PDBFile
from openmm import Platform, unit, LocalEnergyMinimizer
from cavermm.system_builder_amber import build_system_from_amber_with_restraints

def load_xyz_ligand(pdb_path: str) -> unit.Quantity:
    # Return Nx3 Quantity in nm for LIG atoms
    xs=[]
    with open(pdb_path) as f:
        for line in f:
            if (line.startswith('ATOM') or line.startswith('HETATM')) and line[17:20].strip()=="LIG":
                x=float(line[30:38]); y=float(line[38:46]); z=float(line[46:54])
                xs.append([x*0.1, y*0.1, z*0.1])
    arr=np.array(xs, dtype=float)
    return arr*unit.nanometer

def set_ghost_and_ramp(sim: Simulation, build, center_nm: np.ndarray, r_nm: float, kcom_ramp_scale: float):
    # anchor ghost
    try:
        k_anchor = build.ghost_anchor_force.getParticleParameters(0)[3]
    except Exception:
        k_anchor = 100000.0*unit.kilojoule_per_mole/unit.nanometer**2
    build.ghost_anchor_force.setParticleParameters(0, build.ghost_index, [center_nm[0]*unit.nanometer, center_nm[1]*unit.nanometer, center_nm[2]*unit.nanometer, k_anchor])
    build.ghost_anchor_force.updateParametersInContext(sim.context)
    # ramp
    try:
        groups, params = build.com_restraint.getBondParameters(0)
        k_base = params[0]
        try:
            k_base_val = float(k_base.value_in_unit(unit.kilojoule_per_mole / unit.nanometer**2))
        except Exception:
            k_base_val = float(k_base)
        k_ramp = (k_base_val * kcom_ramp_scale) * unit.kilojoule_per_mole / unit.nanometer**2
        build.com_restraint.setBondParameters(0, groups, [k_base, k_ramp, r_nm*unit.nanometer])
        build.com_restraint.updateParametersInContext(sim.context)
    except Exception:
        pass

def update_pocket(sim: Simulation, build, center_nm: np.ndarray, r_nm: float, kpos_protein: float, kpos_backbone: float|None, kpos_sidechain: float|None, relax_margin_nm: float, relax_scale: float):
    pos_rest = build.protein_pos_restraint
    idx_to_name = {a.index: a.name for a in sim.topology.atoms()}
    pos_nm = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    k_bb = (kpos_backbone if kpos_backbone is not None else kpos_protein)
    k_sc = (kpos_sidechain if kpos_sidechain is not None else kpos_protein)
    k_bb_q = k_bb * unit.kilojoule_per_mole / unit.nanometer**2
    k_sc_q = k_sc * unit.kilojoule_per_mole / unit.nanometer**2
    k_bb_rel = (k_bb * relax_scale) * unit.kilojoule_per_mole / unit.nanometer**2
    k_sc_rel = (k_sc * relax_scale) * unit.kilojoule_per_mole / unit.nanometer**2
    bb_names = {"N","CA","C","O","OXT"}
    lig = set(getattr(build,'ligand_atom_indices', [])); ghost = getattr(build,'ghost_index', -1)
    N = sim.system.getNumParticles()
    for idx in range(N):
        if idx in lig or idx==ghost: continue
        name = idx_to_name.get(idx,"")
        is_bb = name in bb_names
        d = float(np.linalg.norm(pos_nm[idx]-center_nm))
        use_relax = (d <= (r_nm + relax_margin_nm))
        kq = (k_bb_rel if is_bb else k_sc_rel) if use_relax else (k_bb_q if is_bb else k_sc_q)
        # keep current target x0,y0,z0 as stored
        particle, params = pos_rest.getParticleParameters(idx)
        x0,y0,z0 = params[0], params[1], params[2]
        pos_rest.setParticleParameters(idx, particle, [x0,y0,z0,kq])
    pos_rest.updateParametersInContext(sim.context)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out', required=True, help='Base out directory with enter/ and exit/ and frames_refined')
    ap.add_argument('--kcom', type=float, default=1000.0)
    ap.add_argument('--kpos-protein', type=float, default=2000.0)
    ap.add_argument('--kpos-backbone', type=float, default=1500.0)
    ap.add_argument('--kpos-sidechain', type=float, default=500.0)
    ap.add_argument('--relax-pocket-margin-A', type=float, default=2.0)
    ap.add_argument('--relax-pocket-scale', type=float, default=0.1)
    ap.add_argument('--kcom-ramp-scale', type=float, default=2.0)
    args=ap.parse_args()

    out=args.out
    # Use exit energy_profile for centers
    ex = pd.read_csv(os.path.join(out,'exit','energy_profile.csv'))
    centers = ex[['x_nm','y_nm','z_nm']].to_numpy(dtype=float)
    radii = ex['r_nm'].to_numpy(dtype=float)
    # Build system
    prmtop=os.path.join(out,'amber_prep','complex.prmtop')
    inpcrd=os.path.join(out,'amber_prep','complex.inpcrd')
    build = build_system_from_amber_with_restraints(prmtop, inpcrd, ligand_resname='LIG', kpos_protein=args.kpos_protein, kcom=args.kcom)
    sim = Simulation(build.topology, build.system, (lambda: __import__('openmm').openmm.LangevinIntegrator(300*unit.kelvin,1.0/unit.picosecond,0.001*unit.picoseconds))())
    sim.context.setPositions(build.positions)

    # Evaluate energies for refined frames
    fd=os.path.join(out,'frames_refined')
    frames=sorted([os.path.join(fd,f) for f in os.listdir(fd) if f.startswith('step_')])
    vals=[]
    for j,f in enumerate(frames):
        # Overlay ligand refined coordinates onto current positions
        state=sim.context.getState(getPositions=True)
        posq=state.getPositions(asNumpy=True)
        lig_xyz = load_xyz_ligand(f)  # nm
        # If counts mismatch, skip
        if lig_xyz.shape[0] != len(build.ligand_atom_indices):
            continue
        for k,idx in enumerate(build.ligand_atom_indices):
            posq[idx] = lig_xyz[k]
        sim.context.setPositions(posq)
        center_nm = centers[j]
        r_nm = radii[j]
        set_ghost_and_ramp(sim, build, center_nm, r_nm, args.kcom_ramp_scale)
        update_pocket(sim, build, center_nm, r_nm, args.kpos_protein, args.kpos_backbone, args.kpos_sidechain, args.relax_pocket_margin_A*0.1, args.relax_pocket_scale)
        e = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        vals.append([j, e, e/4.184])
    df=pd.DataFrame(vals, columns=['step','E_kJ_mol','E_kcal_mol'])
    # dE vs step 0
    df['dE_kJ_mol']=df['E_kJ_mol']-df.iloc[0]['E_kJ_mol']
    df['dE_kcal_mol']=df['E_kcal_mol']-df.iloc[0]['E_kcal_mol']
    df.to_csv(os.path.join(out,'energy_profile_refined.csv'), index=False)
    # Quick report
    E=df['E_kcal_mol'].to_numpy(); de=np.abs(np.diff(E)) if len(E)>1 else [0.0]
    print('Refined energies recomputed: ΔE jumps max=%.3f mean=%.3f kcal/mol' % (max(de), float(np.mean(de))))

if __name__=='__main__':
    main()
