#!/usr/bin/env bash
set -euo pipefail

# Run the full MDTunnel workflow reproducibly using the MDTunnel conda env.
#
# Usage:
#   tools/run_full.sh [OUT_DIR]
#
# Optional environment variables:
#   ENV=MDTunnel                # Conda env name
#   PROTEIN=LinB_WT.pdb         # Protein PDB input
#   LIGAND=DBE.sdf              # Ligand file (SDF/PDB/MOL2)
#   SPHERES=CW_tunnel.dsd       # CAVER spheres CSV/PDB (resname SPH)
#   VINA_BIN=vina               # Path/name to AutoDock Vina binary
#   MGLTOOLS_ROOT=<path>        # Root of MGLTools (prepare_* scripts + pythonsh)
#
# Notes:
# - If MGLTOOLS_ROOT is not set, we try a system MGLTools under ~/Programs/Autodock4/...
#   and finally fall back to the MDTunnel env prefix.

OUT=${1:-runs/full_bidir_MDTunnel}
ENV_NAME=${ENV:-MDTunnel}
PROTEIN=${PROTEIN:-LinB_WT.pdb}
LIGAND=${LIGAND:-DBE.sdf}
SPHERES=${SPHERES:-CW_tunnel.dsd}
VINA_BIN=${VINA_BIN:-vina}

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda env '$ENV_NAME' not found. Create it first (see ABOUT_MDTUNNEL.md)." >&2
  exit 1
fi

# Detect MGLTools root
if [[ -n "${MGLTOOLS_ROOT:-}" ]]; then
  MGL_ROOT="$MGLTOOLS_ROOT"
elif [[ -d "/home/$USER/Programs/Autodock4/mgltools_x86_64Linux2_1.5.7" ]]; then
  MGL_ROOT="/home/$USER/Programs/Autodock4/mgltools_x86_64Linux2_1.5.7"
else
  MGL_ROOT="$(conda run -n "$ENV_NAME" bash -lc 'echo $CONDA_PREFIX')"
fi

mkdir -p "$OUT"
set -x
conda run -n "$ENV_NAME" python -m mdtunnel.cli \
  --protein "$PROTEIN" \
  --ligand "$LIGAND" \
  --spheres "$SPHERES" \
  --out "$OUT" \
  --bidirectional-merge \
  --dock-first \
  --ambertools-prep \
  --ligand-charge-method gas \
  --vina-rescore \
  --vina-bin "$VINA_BIN" \
  --mgltools-root "$MGL_ROOT" \
  --box-size 22 \
  --exhaustiveness 8 \
  --num-modes 1
set +x

echo "\nDone. Outputs:"
echo " - $OUT/energy_ff_vs_vina.png (FF vs Vina, normalized)"
echo " - $OUT/energy_profile.csv (FF profile, spike-reduced)"
echo " - $OUT/vina_profile.csv (Vina profile, normalized)"
echo " - $OUT/frames/step_*.pdb (merged bidirectional frames)"
