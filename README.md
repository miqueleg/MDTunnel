<<<<<<< HEAD
# MDTunnel
=======
## MDTunnel: Sphere-by-sphere ligand minimization in tunnels
>>>>>>> 1c79bc7824cce3b9c1890b6b87ed6cc4a1e10524

Sphere‑by‑sphere ligand minimization along protein tunnels using OpenMM.

MDTunnel performs sphere‑by‑sphere ligand minimization along a protein tunnel path. Provide a protein, a ligand, and a sequence of tunnel spheres (e.g., from CAVER3), and it will generate minimized ligand poses and an energy profile. It can optionally dock the first pose with AutoDock Vina and supports an AmberTools‑based preparation path.

## Key Features

- Protein force field: Amber ff14SB (OpenMM XML) by default.
- Ligand parameterization: GAFF2 + AM1‑BCC via OpenFF + openmmforcefields.
- Optional AMOEBA path for protein/ligand (requires suitable templates/support).
- Optional re‑parameterization per step to emulate polarization effects.
- Center‑of‑mass restraint to keep the ligand near each sphere center.
- Bidirectional run and per‑sphere energy‑based merge.

## Installation

Requires Python 3.9+.

Option A: editable install from this repo

```
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

Option B: direct install (pyproject)

```
pip install -U pip
pip install .
```

Core runtime dependencies:

- openmm, openmmforcefields, openff-toolkit, rdkit-pypi, numpy, pandas

Optional external tools:

- AutoDock Vina (`vina`) and Open Babel (`obabel`) for `--dock-first`
- AmberTools (`antechamber`, `parmchk2`, `tleap`) for `--ambertools-prep`

## Quickstart

Basic minimization along spheres (GAFF2/AM1-BCC for ligand):

```
python -m mdtunnel.cli \
  --protein protein.pdb \
  --ligand ligand.sdf \
  --spheres tunnel.csv \
  --out outdir
```

Dock first sphere with Vina, then minimize:

```
python -m mdtunnel.cli \
  --protein protein.pdb \
  --ligand ligand.sdf \
  --spheres tunnel.csv \
  --out outdir \
  --dock-first \
  --exhaustiveness 16 \
  --num-modes 1 \
  --box-size 24
```

Bidirectional run with energy‑based merge:

```
python -m mdtunnel.cli \
  --protein protein.pdb \
  --ligand ligand.sdf \
  --spheres tunnel.csv \
  --out outdir \
  --bidirectional-merge
```

Get CLI help:

```
python -m mdtunnel.cli --help
```

## Inputs

- Protein: PDB file of the receptor.
- Ligand: SDF/MOL2/PDB of the small molecule; if not protonated, the tool will add hydrogens by default.
- Spheres: tunnel path as CSV or PDB.
  - CSV: columns `x,y,z,r` (header optional), coordinates in Å.
  - PDB: spheres as `HETATM` with `resname SPH`; radius read from B‑factor if present (defaults to 1.5 Å).

## Outputs

- `frames/step_XXXX.pdb`: ligand+protein frames per sphere.
- `substrate_traj.pdb`: concatenated ligand models across steps (if merge mode, merged trajectory).
- `energy_profile.csv`: per‑step energy in kJ/mol and kcal/mol; merged profile in bidirectional mode with ΔE relative to step 0.
- `timings.json`: wall‑clock timings and number of spheres.
- `docking/` and `amber_prep/`: created when using `--dock-first` and `--ambertools-prep`, respectively.

## Typical Options

- `--protein-ff`: OpenMM XML for protein (default `amber14/protein.ff14SB.xml`).
- `--ligand-param`: `gaff-am1bcc` (default) | `amoeba` | `qmmm`.
- `--kcom`: COM restraint k (kJ/mol/nm²), default 2000.
- `--kpos-protein`: protein position restraint k (kJ/mol/nm²).
- `--platform`: CUDA | OpenCL | CPU (delegated to OpenMM).
- `--reparam-per-step`: rebuild ligand params each step (GAFF path).
- `--dock-first`: run Vina docking at the first sphere.
- `--ambertools-prep`: build with AmberTools (requires external binaries).

## Notes

- If you already have a reasonable starting pose, you can skip docking. By default, the ligand COM is translated to the first sphere and minimized under a COM restraint that allows rotation.
- AMOEBA often requires specific ligand templates; enable only if your environment supports it.
- Re‑parameterization per step is optional and can be expensive.

## Development

- CLI entry point: `mdtunnel.cli:main` (`python -m mdtunnel.cli`).
- Core modules: `pipeline.py`, `system_builder.py`, `param.py`, `spheres.py`, `ligand_prep.py`.
- Utility scripts in `tools/` help sweep parameters and refine outputs.

## License

Please add a LICENSE file to clarify reuse. If you prefer, we can include a standard license (e.g., MIT/BSD-3-Clause/Apache-2.0).

## Citation

If you use this tool in academic work, please cite this repository and the underlying tools (OpenMM, OpenFF Toolkit, openmmforcefields, RDKit, AutoDock Vina, AmberTools).
