## MDTunnel: Sphere-by-sphere ligand minimization in tunnels

MDTunnel generates minimized ligand poses along a protein tunnel, one sphere at a time, and an energy profile. It can dock the initial pose with AutoDock Vina, build a protein–ligand complex with AmberTools, and rescore the minimized frames with Vina. Bidirectional runs are merged per step and lightly smoothed to reduce isolated spikes.

## Highlights

- OpenMM-based minimization with restraints per sphere; default protein FF: Amber ff14SB.
- Ligand parameters via OpenFF + openmmforcefields (GAFF2 + AM1‑BCC) or AmberTools path.
- Optional initial docking (Vina) and per-frame Vina rescoring (default on).
- Bidirectional enter/exit runs merged per step by lower FF energy; 3-point spike reduction for isolated outliers (>5 kcal/mol).
- Robust MGLTools-only PDBQT prep (no Open Babel): absolute paths, cwd-safe execution.
- Built-in protein PDB pre-cleaner: drops altLocs/duplicates; standardizes residue names (NSER→SER, MSE→MET with SE→SD), sets HIS tautomers per-residue from explicit Hs; runs before pdb4amber.

## Environment (MDTunnel)

Create a dedicated env with conda-forge and bioconda:
- conda create -n MDTunnel -c conda-forge -c bioconda \
  python=3.12 openmm ambertools autodock-vina mgltools \
  rdkit openff-toolkit openmmforcefields matplotlib pandas numpy

MGLTools note: prepare_[ligand|receptor]4.py are legacy Python 2 scripts. Pass `--mgltools-root` pointing to a working MGLTools install with `pythonsh`. If you have a system MGLTools (e.g., `~/Programs/Autodock4/mgltools_x86_64Linux2_1.5.7`), prefer that. The provided run script auto-detects it; otherwise it falls back to the env prefix.

## Quickstart

Reproducible script (recommended):
- bash tools/run_full.sh [OUT_DIR]
  - Defaults: `ENV=MDTunnel`, `PROTEIN=LinB_WT.pdb`, `LIGAND=DBE.sdf`, `SPHERES=CW_tunnel.dsd`, `OUT=runs/full_bidir_MDTunnel`.
  - Optional: set `MGLTOOLS_ROOT` to point at your MGLTools root.

Manual command:
- conda run -n MDTunnel python -m mdtunnel.cli \
  --protein LinB_WT.pdb \
  --ligand DBE.sdf \
  --spheres CW_tunnel.dsd \
  --out runs/full_bidir_MDTunnel \
  --bidirectional-merge \
  --dock-first \
  --ambertools-prep \
  --ligand-charge-method gas \
  --vina-rescore \
  --vina-bin vina \
  --mgltools-root "/home/$USER/Programs/Autodock4/mgltools_x86_64Linux2_1.5.7" \
  --box-size 22 --exhaustiveness 8 --num-modes 1

Get help: `python -m mdtunnel.cli --help`

## Inputs

- Protein: PDB file. The tool pre-cleans and then runs pdb4amber (unless `--amber-protein-ready`).
- Ligand: SDF/MOL2/PDB. For docking/prepare_ligand4, SDF is converted to MOL2 via AmberTools (`-c gas`), then used by MGLTools.
- Spheres: CSV or PDB of tunnel spheres.
  - CSV: columns `x,y,z,r` (Å). PDB: `HETATM`, `resname SPH`; radius from B‑factor if present.

## Outputs

- `frames/step_XXXX.pdb`: merged per-step frames (ligand+protein).
- `substrate_traj.pdb`: ligand-only concatenated across steps.
- `energy_profile.csv`: per-step FF energy; merged ΔE (kcal/mol) normalized to step 0; spike-reduced.
- `vina_profile.csv`: Vina score-only affinity (kcal/mol), normalized to the first step.
- `energy_ff_vs_vina.png`: dual-axis plot of FF ΔE (left, kcal/mol) vs distance (Å) and Vina (right, kcal/mol), both normalized to 0 at the docked pose.
- `energy_profile.png`: FF-only plot of merged ΔE vs distance.
- `timings.json`: timings summary.
- `docking/` and `amber_prep/`: created when `--dock-first` and `--ambertools-prep` are used.

## Common options

- `--protein-ff`: OpenMM XML (default `amber14/protein.ff14SB.xml`).
- `--ligand-param`: `gaff-am1bcc` (default) | `amoeba` | `qmmm`.
- `--kcom`, `--kpos-protein`, `--kpos-backbone`, `--kpos-sidechain`: restraint strengths (kJ/mol/nm²).
- `--platform`: CUDA | OpenCL | CPU.
- `--reparam-per-step`: rebuild ligand params each step (slower).
- `--dock-first` | `--no-dock-first`: enable/disable initial docking (default: enabled).
- `--vina-rescore` | `--no-vina-rescore`: rescoring on by default.
- `--ambertools-prep`: build complex with AmberTools (recommends `--ligand-charge-method gas`).
- `--mgltools-root`: path to MGLTools root (with `pythonsh` + `MGLToolsPckgs/AutoDockTools/Utilities24`).

## Notes

- No Open Babel: PDBQT prep uses MGLTools only; ligand bond orders supplied via AmberTools conversion to MOL2.
- Protein PDB pre-cleaner: standardizes nonstandard names, resolves altLocs/duplicates, assigns histidine tautomers per residue from explicit Hs, and removes lone N‑terminal H (per chain). pdb4amber then refines for Amber.
- Plots: PNGs are saved using a robust Agg backend; if a PNG save ever fails, the code will attempt an SVG and write a `plot_debug.txt` explaining the error.

## Development

- CLI entry point: `mdtunnel.cli:main` (`python -m mdtunnel.cli`).
- Core modules: `pipeline.py`, `system_builder.py`, `param.py`, `spheres.py`, `ligand_prep.py`, `docking.py`, `amber_prep.py`.
- Reproducible run script: `tools/run_full.sh` (uses conda run; configurable via env vars).

## License & Citation

- Please include an appropriate LICENSE if you plan to distribute.
- Cite this repository and underlying tools: OpenMM, OpenFF Toolkit, openmmforcefields, RDKit, AutoDock Vina, AmberTools, and MGLTools.
