from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class ParamOptions:
    protein_ff: str = "amber14/protein.ff14SB.xml"
    ligand_param: str = "gaff-am1bcc"  # or "amoeba" or "qmmm"
    amoeba: bool = False
    gbsa: Optional[str] = None  # e.g., "OBC2" (not wired by default)


def build_forcefield_and_generators(options: ParamOptions, ligand_file: str):
    """Prepare an OpenMM ForceField and (optionally) ligand template generators.

    Returns (forcefield, ligand_mol, generators_dict)
    """
    from openmm.app import ForceField

    ligand_mol = None
    generators = {}

    if options.ligand_param.lower().startswith("gaff"):
        # Build protein FF and GAFF generator
        try:
            from openff.toolkit.topology import Molecule
            from openmmforcefields.generators import GAFFTemplateGenerator
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "GAFF/AM1-BCC path requires openff-toolkit and openmmforcefields to be installed"
            ) from e

        ligand_mol = Molecule.from_file(ligand_file, allow_undefined_stereo=True)
        # Charges will be assigned by the generator as AM1-BCC by default.
        gaff = GAFFTemplateGenerator(molecules=[ligand_mol], forcefield="gaff-2.11")

        ff = ForceField(options.protein_ff)
        ff.registerTemplateGenerator(gaff.generator)
        generators["ligand"] = gaff
        return ff, ligand_mol, generators

    if options.ligand_param.lower() == "amoeba":
        # AMOEBA for protein and ligand.
        try:
            from openff.toolkit.topology import Molecule
            from openmmforcefields.generators import AmoebaTemplateGenerator  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "AMOEBA path requires openff-toolkit and openmmforcefields with AmoebaTemplateGenerator"
            ) from e

        ff = ForceField("amoeba2018.xml")
        ligand_mol = Molecule.from_file(ligand_file, allow_undefined_stereo=True)
        amoeba_gen = AmoebaTemplateGenerator(molecules=[ligand_mol])
        ff.registerTemplateGenerator(amoeba_gen.generator)
        generators["ligand"] = amoeba_gen
        return ff, ligand_mol, generators

    if options.ligand_param.lower() == "qmmm":
        # Expect user-supplied ligand template via ffxml/mol2; here we only return protein FF.
        ff = ForceField(options.protein_ff)
        return ff, None, generators

    raise ValueError(f"Unknown ligand_param: {options.ligand_param}")

