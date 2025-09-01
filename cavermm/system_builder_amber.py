from __future__ import annotations

from dataclasses import dataclass
from typing import List

from openmm import unit


@dataclass
class AmberBuildResult:
    system: "openmm.System"
    topology: "openmm.app.Topology"
    positions: "openmm.unit.Quantity"
    ligand_atom_indices: List[int]
    ghost_index: int
    ghost_anchor_force: "openmm.CustomExternalForce"
    com_restraint: "openmm.CustomCentroidBondForce"
    protein_pos_restraint: "openmm.CustomExternalForce"


def build_system_from_amber_with_restraints(
    prmtop_path: str,
    inpcrd_path: str,
    ligand_resname: str = "LIG",
    kpos_protein: float = 10000.0,
    kcom: float = 2000.0,
) -> AmberBuildResult:
    from openmm.app import AmberPrmtopFile, AmberInpcrdFile
    from openmm import CustomCentroidBondForce, CustomExternalForce

    prmtop = AmberPrmtopFile(prmtop_path)
    inpcrd = AmberInpcrdFile(inpcrd_path)
    system = prmtop.createSystem(removeCMMotion=False)
    topology = prmtop.topology
    positions = inpcrd.positions

    # Identify ligand atoms by residue name
    ligand_atom_indices: List[int] = []
    for atom in topology.atoms():
        if atom.residue.name.strip().upper() == ligand_resname.upper():
            ligand_atom_indices.append(atom.index)
    if not ligand_atom_indices:
        # Fallback to last residue
        last_res = list(topology.residues())[-1]
        ligand_atom_indices = [a.index for a in last_res.atoms()]

    # Add ghost particle and restraints
    ghost_index = system.addParticle(12.0 * unit.dalton)
    # Append a dummy position for the ghost so setPositions() matches particle count
    try:
        from openmm import Vec3
        from openmm import unit as omm_unit
        pos_list = list(positions)
        pos_list.append(Vec3(0.0, 0.0, 0.0) * omm_unit.nanometer)
        positions = pos_list  # Sequence of Quantities acceptable to setPositions
    except Exception:
        pass
    # Ensure NonbondedForce has a matching particle with zeroed parameters
    try:
        from openmm import NonbondedForce
        for f in (system.getForce(i) for i in range(system.getNumForces())):
            if isinstance(f, NonbondedForce):
                f.addParticle(0.0, 1.0, 0.0)  # charge, sigma (nm), epsilon (kJ/mol)
                break
    except Exception:
        pass

    # Two-group centroid bond: ligand COM to ghost anchor with ramped strength
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
            (kcom * 5.0) * unit.kilojoule_per_mole / unit.nanometer**2,
            1.0 * unit.nanometer,
        ],
    )
    system.addForce(com_restraint)

    ghost_anchor = CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    ghost_anchor.addPerParticleParameter("x0")
    ghost_anchor.addPerParticleParameter("y0")
    ghost_anchor.addPerParticleParameter("z0")
    ghost_anchor.addPerParticleParameter("k")
    system.addForce(ghost_anchor)
    ghost_anchor.addParticle(
        ghost_index,
        [
            0.0 * unit.nanometer,
            0.0 * unit.nanometer,
            0.0 * unit.nanometer,
            (100000.0 * unit.kilojoule_per_mole / unit.nanometer**2),
        ],
    )

    pos_rest = CustomExternalForce("0.5*kpos*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    pos_rest.addPerParticleParameter("x0")
    pos_rest.addPerParticleParameter("y0")
    pos_rest.addPerParticleParameter("z0")
    pos_rest.addPerParticleParameter("kpos")
    system.addForce(pos_rest)

    prot_indices = [i for i in range(system.getNumParticles()) if i not in ligand_atom_indices and i != ghost_index]
    # We need initial positions to tie restraints
    for idx in prot_indices:
        x, y, z = positions[idx]
        pos_rest.addParticle(
            idx,
            [x, y, z, (kpos_protein * unit.kilojoule_per_mole / unit.nanometer**2)],
        )

    return AmberBuildResult(
        system=system,
        topology=topology,
        positions=positions,
        ligand_atom_indices=ligand_atom_indices,
        ghost_index=ghost_index,
        ghost_anchor_force=ghost_anchor,
        com_restraint=com_restraint,
        protein_pos_restraint=pos_rest,
    )
