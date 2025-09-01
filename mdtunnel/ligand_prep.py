from __future__ import annotations

import os
from dataclasses import dataclass


class LigandPrepError(RuntimeError):
    pass


def _guess_format(path: str) -> str:
    low = path.lower()
    if low.endswith('.sdf'):
        return 'sdf'
    if low.endswith('.mol2'):
        return 'mol2'
    if low.endswith('.pdb'):
        return 'pdb'
    return 'pdb'


def protonate_ligand(infile: str, outfile: str) -> str:
    """Add explicit hydrogens to the ligand and write a PDB.

    - Preserves existing coordinates where possible.
    - Generates coordinates for missing atoms and optimizes lightly to place H.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except Exception as e:  # pragma: no cover
        raise LigandPrepError("RDKit is required for ligand protonation") from e

    fmt = _guess_format(infile)
    if fmt == 'sdf' or fmt == 'mol':
        mol = Chem.MolFromMolFile(infile, removeHs=False, sanitize=True)
    elif fmt == 'mol2':
        mol = Chem.MolFromMol2File(infile, removeHs=False, sanitize=True)
    elif fmt == 'pdb':
        mol = Chem.MolFromPDBFile(infile, removeHs=False, sanitize=True)
    else:
        mol = Chem.MolFromMolFile(infile, removeHs=False, sanitize=True)
    if mol is None:
        raise LigandPrepError(f"Failed to read ligand file: {infile}")

    # Add explicit hydrogens based on valence states
    molH = Chem.AddHs(mol, addCoords=True)

    # Ensure we have 3D coords; embed if missing
    if molH.GetNumConformers() == 0:
        AllChem.EmbedMolecule(molH, useRandomCoords=True)

    # Lightly optimize to position H without moving too much
    try:
        if AllChem.MMFFHasAllMoleculeParams(molH):
            AllChem.MMFFOptimizeMolecule(molH, maxIters=200)
        else:
            AllChem.UFFOptimizeMolecule(molH, maxIters=200)
    except Exception:
        pass

    # Write PDB
    outdir = os.path.dirname(os.path.abspath(outfile))
    if outdir and not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)
    Chem.MolToPDBFile(molH, outfile)
    return outfile

