Overview
- This repo implements a bidirectional, sphere-by-sphere minimization workflow with optional Vina docking + rescoring. It prepares receptor/ligand via MGLTools for Vina, builds an Amber complex (AmberTools), runs OpenMM minimizations with restraints per sphere, merges enter/exit profiles with spike smoothing, and plots FF vs Vina (normalized to the docked pose).

Conda Environment
- Create a clean env named MDTunnel with all required tools (conda-forge + bioconda):
  - conda create -n MDTunnel -c conda-forge -c bioconda \
    python=3.12 openmm ambertools autodock-vina mgltools \
    rdkit openff-toolkit openmmforcefields matplotlib pandas numpy

- Activate when running manually (the provided script uses conda run):
  - conda activate MDTunnel

MGLTools Notes
- MGLTools prepare_ligand4.py / prepare_receptor4.py are legacy Python 2 scripts. The code accepts a root path via --mgltools-root. Recommended:
  - If you have a system MGLTools install with pythonsh (e.g., ~/Programs/Autodock4/mgltools_x86_64Linux2_1.5.7), pass that as --mgltools-root for best compatibility.
  - Otherwise, conda’s mgltools package places the prepare_* scripts under $CONDA_PREFIX/MGLToolsPckgs/... and pythonsh under $CONDA_PREFIX/bin/pythonsh. The run script will auto-detect this if a system MGLTools is not found.

Running (Reproducible Script)
- Use the helper script to run a full bidirectional calculation with docking, Amber prep, and Vina rescoring. It writes normalized/merged CSVs and the FF vs Vina PNG.
  - tools/run_full.sh [OUT_DIR]
  - Environment variables (optional):
    - ENV=MDTunnel               # conda env name
    - PROTEIN=LinB_WT.pdb        # protein PDB
    - LIGAND=DBE.sdf             # ligand (SDF/PDB/MOL2)
    - SPHERES=CW_tunnel.dsd      # spheres CSV/PDB (resname SPH)
    - VINA_BIN=vina              # vina binary name/path
    - MGLTOOLS_ROOT=<path>       # MGLTools root (see notes above)

Outputs
- energy_ff_vs_vina.png: Dual-axis plot. Left: FF ΔE (kcal/mol) vs distance (Å). Right: Vina kcal/mol. Both normalized to zero at docked pose.
- energy_profile.csv: Merged FF profile (kcal/mol, spike-reduced) with distances (Å) and steps.
- vina_profile.csv: Vina scores per step (kcal/mol), normalized to first step.
- frames/step_*.pdb: Per-step frames after bidirectional merge.
- substrate_traj.pdb: Ligand-only trajectory concatenated through the merged path.

Key CLI Options (mdtunnel.cli)
- --dock-first / --no-dock-first: Enable/disable initial Vina docking at the first sphere.
- --vina-rescore: Re-score frames using Vina after minimizations.
- --mgltools-root: Root folder containing pythonsh and MGLToolsPckgs/AutoDockTools/Utilities24/prepare_*4.py.
- --ambertools-prep: Use AmberTools (antechamber/parmchk2/tleap) to build the complex.
- --ligand-charge-method gas: Prefer ‘gas’ to avoid sqm during antechamber (faster, robust).

Protein/Ligand Preparation Details
- Protein: The code pre-cleans PDBs before pdb4amber: drops altLocs/duplicates, standardizes residue names (e.g., NSER→SER; MSE→MET + SE→SD; HIS tautomer set by explicit hydrogens per residue), and removes the generic N-terminal “H” per chain (tleap adds correct Hs).
- Ligand: For SDF inputs, the docking helper converts to MOL2 via antechamber -c gas to ensure bond orders for MGLTools typing.

Troubleshooting
- Vina not found: set VINA_BIN or add the vina path in ENV. The run script passes this with --vina-bin.
- MGLTools Python errors: use a system MGLTools root with pythonsh (legacy Py2), or ensure conda’s mgltools is used via --mgltools-root "$CONDA_PREFIX".
- Plot PNG issues: The CLI uses a robust Agg backend and isolates from system LD_LIBRARY_PATH; PNGs should render under MDTunnel.

Quick Manual Command
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
