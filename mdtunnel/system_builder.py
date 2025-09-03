from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from openmm import unit


@dataclass
class SystemBuildResult:
    system: "openmm.System"
    topology: "openmm.app.Topology"
    positions: "openmm.unit.Quantity"
    ligand_atom_indices: List[int]
    ghost_index: int
    ghost_anchor_force: "openmm.CustomExternalForce"
    com_restraint: "openmm.CustomCentroidBondForce"
    protein_pos_restraint: "openmm.CustomExternalForce"
    protein_atom_indices: List[int]
    protein_atom_is_backbone: List[bool]
    protein_rest_targets_nm: "openmm.unit.Quantity"


def build_system_with_restraints(
    protein_path: str,
    ligand_path: str,
    forcefield: "openmm.app.ForceField",
    ligand_mol: Optional["openff.toolkit.topology.Molecule"],
    kpos_protein: float = 10000.0,  # kJ/mol/nm^2
    kpos_backbone: Optional[float] = None,
    kpos_sidechain: Optional[float] = None,
    kcom: float = 2000.0,  # kJ/mol/nm^2
    platform_name: Optional[str] = None,
) -> SystemBuildResult:
    from openmm import CustomCentroidBondForce, CustomExternalForce, System
    from openmm import unit
    from openmm.app import ForceField, Modeller, PDBFile, NoCutoff

    # Load protein PDB
    pdb_prot = PDBFile(protein_path)

    # Load ligand: allow PDB or SDF/MOL2 via RDKit/OpenFF conversion
    ligand_top, ligand_pos = _load_ligand_as_topology(ligand_path, ligand_mol)

    # Merge topologies and positions
    modeller = Modeller(pdb_prot.topology, pdb_prot.positions)
    modeller.add(ligand_top, ligand_pos)
    # Ensure standard protonation states/hydrogens for protein templates
    try:
        modeller.addHydrogens(forcefield)
    except Exception:
        pass

    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=NoCutoff,
        constraints=None,
        removeCMMotion=False,
    )

    # Identify ligand atoms (the last added chain)
    ligand_atom_indices = _get_ligand_atom_indices(modeller.topology)

    # Add ghost particle and COM restraint
    ghost_index = system.addParticle(12.0 * unit.dalton)

    # Two-group centroid bond: ligand COM to ghost anchor
    # COM restraint with ramp: 0.5*k_base*d^2 + 0.5*k_ramp*(d^4)/(r_nm^2)
    com_restraint = CustomCentroidBondForce(2, "0.5*k_base*distance(g1,g2)^2 + 0.5*k_ramp*(distance(g1,g2)*distance(g1,g2)*distance(g1,g2)*distance(g1,g2))/(r_nm*r_nm)")
    com_restraint.addPerBondParameter("k_base")
    com_restraint.addPerBondParameter("k_ramp")
    com_restraint.addPerBondParameter("r_nm")
    g1 = com_restraint.addGroup(ligand_atom_indices)
    g2 = com_restraint.addGroup([ghost_index])
    com_restraint.addBond(
        [g1, g2],
        [
            kcom * unit.kilojoule_per_mole / unit.nanometer**2,
            (kcom * 5.0) * unit.kilojoule_per_mole / unit.nanometer**2,  # default ramp 5x
            1.0 * unit.nanometer,
        ],
    )
    system.addForce(com_restraint)

    # Anchor ghost at desired point via strong harmonic restraint
    ghost_anchor = CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    ghost_anchor.addPerParticleParameter("x0")
    ghost_anchor.addPerParticleParameter("y0")
    ghost_anchor.addPerParticleParameter("z0")
    ghost_anchor.addPerParticleParameter("k")
    system.addForce(ghost_anchor)
    # Placeholder values; will be updated per-step
    ghost_anchor.addParticle(
        ghost_index,
        [
            0.0 * unit.nanometer,
            0.0 * unit.nanometer,
            0.0 * unit.nanometer,
            (100000.0 * unit.kilojoule_per_mole / unit.nanometer**2),
        ],
    )

    # Position restraints for protein atoms
    pos_rest = CustomExternalForce("0.5*kpos*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    pos_rest.addPerParticleParameter("x0")
    pos_rest.addPerParticleParameter("y0")
    pos_rest.addPerParticleParameter("z0")
    pos_rest.addPerParticleParameter("kpos")
    system.addForce(pos_rest)

    # Apply to protein atoms (all except ligand and ghost)
    prot_indices = [i for i in range(system.getNumParticles()) if i not in ligand_atom_indices and i != ghost_index]
    bb_names = {"N", "CA", "C", "O", "OXT"}
    is_backbone: List[bool] = []
    k_bb = (kpos_backbone if kpos_backbone is not None else kpos_protein) * unit.kilojoule_per_mole / unit.nanometer**2
    k_sc = (kpos_sidechain if kpos_sidechain is not None else kpos_protein) * unit.kilojoule_per_mole / unit.nanometer**2
    for idx in prot_indices:
        x, y, z = modeller.positions[idx]
        # map index to atom name in topology
        name = None
        for atom in modeller.topology.atoms():
            if atom.index == idx:
                name = atom.name
                break
        bb = bool(name in bb_names)
        is_backbone.append(bb)
        pos_rest.addParticle(idx, [x, y, z, (k_bb if bb else k_sc)])

    return SystemBuildResult(
        system=system,
        topology=modeller.topology,
        positions=modeller.positions,
        ligand_atom_indices=ligand_atom_indices,
        ghost_index=ghost_index,
        ghost_anchor_force=ghost_anchor,
        com_restraint=com_restraint,
        protein_pos_restraint=pos_rest,
        protein_atom_indices=prot_indices,
        protein_atom_is_backbone=is_backbone,
        protein_rest_targets_nm=np.array([[p.x, p.y, p.z] for p in modeller.positions]) * unit.nanometer,
    )


def _load_ligand_as_topology(ligand_path: str, ligand_mol=None):
    from openmm.app import PDBFile
    from openmm import unit
    # If ligand_mol is provided, always use it to define the topology (robust chemistry)
    if ligand_mol is not None:
        top = ligand_mol.to_topology().to_openmm()
        # Default positions from ligand_mol conformer
        if ligand_mol.n_conformers:
            coords = ligand_mol.conformers[0]
            try:
                pos_A = np.asarray(coords.value_in_unit(unit.angstrom), dtype=float)
            except Exception:
                try:
                    pos_A = np.asarray(getattr(coords, "magnitude", coords), dtype=float)
                except Exception:
                    pos_A = np.asarray(coords, dtype=float)
            positions = pos_A * 0.1 * unit.nanometer
        else:
            positions = np.zeros((top.getNumAtoms(), 3)) * unit.nanometer
        # If a PDB with docked coordinates is given, attempt to override positions
        try:
            if ligand_path.lower().endswith(".pdb"):
                pdb = PDBFile(ligand_path)
                # If atom counts match, override positions with docked PDB coords
                if pdb.topology.getNumAtoms() == top.getNumAtoms():
                    positions = pdb.positions
        except Exception:
            pass
        return top, positions

    # Fallback: build from files
    try:
        if ligand_path.lower().endswith(".pdb"):
            pdb = PDBFile(ligand_path)
            return pdb.topology, pdb.positions
    except Exception:
        pass
    try:
        from openff.toolkit.topology import Molecule
        ligand_mol = Molecule.from_file(ligand_path, allow_undefined_stereo=True)
    except Exception as e:  # pragma: no cover
        raise RuntimeError("RDKit/OpenFF required to read non-PDB ligands") from e
    top = ligand_mol.to_topology().to_openmm()
    if ligand_mol.n_conformers:
        coords = ligand_mol.conformers[0]
        try:
            pos_A = np.asarray(coords.value_in_unit(unit.angstrom), dtype=float)
        except Exception:
            try:
                pos_A = np.asarray(getattr(coords, "magnitude", coords), dtype=float)
            except Exception:
                pos_A = np.asarray(coords, dtype=float)
        positions = pos_A * 0.1 * unit.nanometer
    else:
        positions = np.zeros((top.getNumAtoms(), 3)) * unit.nanometer
    return top, positions


def _get_ligand_atom_indices(topology) -> List[int]:
    # Heuristic: atoms in the last chain belong to the ligand.
    chains = list(topology.chains())
    last_chain = chains[-1]
    atom_indices = [atom.index for atom in last_chain.atoms()]
    return atom_indices
