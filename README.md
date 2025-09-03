## MDTunnel: Sphere-by-sphere ligand minimization in tunnels

Sphere‑by‑sphere ligand minimization along protein tunnels using OpenMM. Provide a protein, a ligand, and a sequence of tunnel spheres (e.g., from CAVER3), and MDTunnel will generate minimized ligand poses and an energy profile. It can optionally dock the first pose with AutoDock Vina and supports an AmberTools‑based preparation path.

## Key Features

- Protein force field: Amber ff14SB (OpenMM XML) by default.
- Ligand parameterization: GAFF2 + AM1‑BCC via OpenFF + openmmforcefields.
- Optional AMOEBA path for protein/ligand (requires suitable templates/support).
- Optional re‑parameterization per step to emulate polarization effects.
- Center‑of‑mass restraint to keep the ligand near each sphere center.
- Bidirectional run and per‑sphere energy‑based merge (default).
- Optional post‑run Vina rescoring of poses (score‑only) and FF vs Vina comparison plot.

## Installation

Requires Python 3.9+.

Conda (recommended):

```
conda create -n OpenMM -c conda-forge python=3.11 openmm openmmforcefields openff-toolkit rdkit pandas numpy matplotlib
conda activate OpenMM
pip install -e .
```

Optional external tools:

- AutoDock Vina (`vina`) for docking and rescoring. MGLTools is required for PDBQT preparation; set `MGLTOOLS_ROOT` or pass `--mgltools-root`.
- AmberTools (`antechamber`, `parmchk2`, `tleap`, `pdb4amber`) for `--ambertools-prep`.

## Quickstart

Basic minimization along spheres (GAFF2/AM1-BCC for ligand). By default, the first pose is docked with Vina and a bidirectional run+merge is performed:

```
python -m mdtunnel.cli \
  --protein protein.pdb \
  --ligand ligand.sdf \
  --spheres tunnel.csv \
  --out outdir
```

Disable docking and/or bidirectional merge explicitly:

```
python -m mdtunnel.cli \
  --protein protein.pdb \
  --ligand ligand.sdf \
  --spheres tunnel.csv \
  --out outdir \
  --no-dock-first \
  --single-direction
```

Bidirectional run with energy‑based merge (default):

```
python -m mdtunnel.cli \
  --protein protein.pdb \
  --ligand ligand.sdf \
  --spheres tunnel.csv \
  --out outdir \
  --bidirectional-merge
```

Get CLI help: `python -m mdtunnel.cli --help`

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
- `vina_profile.csv`: Vina score‑only affinity (kcal/mol) for each frame when `--vina-rescore` is used.
- `energy_ff_vs_vina.png`: comparison plot with FF ΔE and Vina scores on two x‑axes (bottom: distance Å; top: step index).
- `timings.json`: wall‑clock timings and number of spheres.
- `docking/` and `amber_prep/`: created when using `--dock-first` and `--ambertools-prep`, respectively.

## Typical Options

- `--protein-ff`: OpenMM XML for protein (default `amber14/protein.ff14SB.xml`).
- `--ligand-param`: `gaff-am1bcc` (default) | `amoeba` | `qmmm`.
- `--kcom`: COM restraint k (kJ/mol/nm²), default 2000.
- `--kpos-protein`: protein position restraint k (kJ/mol/nm²).
- `--platform`: CUDA | OpenCL | CPU (delegated to OpenMM).
- `--reparam-per-step`: rebuild ligand params each step (GAFF path).
- `--dock-first`: run Vina docking at the first sphere (requires Vina and MGLTools).
- `--no-dock-first`: disable docking (overrides default `--dock-first`).
- `--ambertools-prep`: build with AmberTools (requires external binaries).
- `--bidirectional-merge` (default) | `--single-direction` to disable the merge.
- `--vina-rescore`: after generating frames, rescore poses with Vina (score‑only) and plot FF vs Vina.

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
