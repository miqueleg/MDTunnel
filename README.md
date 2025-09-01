## MDTunnel: Sphere-by-sphere ligand minimization in tunnels

Overview

- Inputs: protein structure, ligand, tunnel spheres (from CAVER3).
- Process: place ligand in first sphere (optionally docked pose), then for each sphere:
  - Translate ligand COM to sphere center while preserving orientation from previous step.
  - Apply a COM harmonic restraint to keep ligand near the sphere center (allows rotation).
  - Keep protein atoms restrained (effectively rigid) during minimization.
  - Minimize only the ligand degrees of freedom using OpenMM.
  - Record potential energy and write the minimized pose.
- Outputs: per-step PDB frames and an energy profile CSV.

Features

- Protein force fields: default Amber ff14SB (OpenMM XML).
- Ligand parameterization: default GAFF2 + AM1-BCC charges via OpenFF/openmmforcefields.
- Optional re-parameterization per step to emulate polarization effects.
- Optional AMOEBA path (requires AMOEBA templates for the ligand or openmmforcefields support).
- COM restraint implemented via CustomCentroidBondForce against a fixed ghost particle.

Status

This is a working scaffold designed to run with OpenMM, OpenFF Toolkit, RDKit, and openmmforcefields installed in your environment. Network access is not needed at runtime, but you must pre-install dependencies.

Install (editable)

- Create a virtualenv and install dependencies (example):

  pip install openmm openmmforcefields openff-toolkit rdkit-pypi numpy pandas

- Use the CLI from the repo root:

  python -m cavermm.cli --help

CLI usage

  python -m cavermm.cli \
    --protein protein.pdb \
    --ligand ligand.sdf \
    --spheres tunnel.csv \
    --out outdir \
    --protein-ff amber14/protein.ff14SB.xml \
    --ligand-param gaff-am1bcc \
    --kcom 2000 \
    --reparam-per-step

Dock first sphere with Vina

- Requires `vina` and `obabel` on PATH.
- The first pose is obtained by docking within a box centered at the first sphere.
- After docking, the ligand is minimized with the COM restraint at that sphere; subsequent spheres use the previous orientation and minimization only.

Example:

  python -m cavermm.cli \
    --protein protein.pdb \
    --ligand ligand.sdf \
    --spheres tunnel.csv \
    --out outdir \
    --dock-first \
    --exhaustiveness 16 \
    --num-modes 1 \
    --box-size 24

Spheres format

- CSV with columns: x,y,z,r (header optional), coordinates in Angstrom.
- Or a PDB with spheres as HETATM records (resname SPH). x,y,z from coordinates; radius read from B-factor if available (defaults to 1.5 Å).

Notes

- Initial “docking”: If you already have a pose in the active site, pass it as the ligand. Otherwise, the tool places the ligand COM at the first sphere center and minimizes under the COM restraint. You may later integrate an external docking step before running this pipeline.
- AMOEBA for ligands often requires template XML; this tool exposes a switch but expects you to provide valid ligand parameters if generic generation is not available in your environment.
- Re-parameterization per step is supported for GAFF+AM1-BCC by recomputing charges at each iteration. This is optional and off by default.
