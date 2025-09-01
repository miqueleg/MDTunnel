from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class VinaOptions:
    vina_bin: str = "vina"
    obabel_bin: str = "obabel"
    exhaustiveness: int = 8
    num_modes: int = 1
    box_size: Optional[float] = None  # in Angstrom; if None, caller chooses
    mgltools_root: Optional[str] = None  # use MGLTools prepare_* if provided


class DockingError(RuntimeError):
    pass


def run_vina_docking(
    protein_pdb: str,
    ligand_file: str,
    center_xyz_ang: Tuple[float, float, float],
    box_size_ang: float,
    outdir: str,
    options: Optional[VinaOptions] = None,
) -> str:
    """Run AutoDock Vina docking for initial pose at the first sphere.

    Returns path to docked ligand PDB.
    Requires `vina` and `obabel` to be available in PATH, or specify in options.
    """
    options = options or VinaOptions()
    os.makedirs(outdir, exist_ok=True)

    def _check_exec(name, path):
        if shutil.which(path) is None:
            raise DockingError(f"Required executable '{name}' not found: {path}")

    _check_exec("vina", options.vina_bin)
    _check_exec("obabel", options.obabel_bin)

    receptor_pdbqt = os.path.join(outdir, "receptor.pdbqt")
    ligand_pdbqt = os.path.join(outdir, "ligand.pdbqt")
    out_pdbqt = os.path.join(outdir, "docked.pdbqt")
    out_pdb = os.path.join(outdir, "docked.pdb")
    log_path = os.path.join(outdir, "vina.log")

    if options.mgltools_root:
        # Use MGLTools prepare scripts
        pythonsh = os.path.join(options.mgltools_root, "bin", "pythonsh")
        prep_rec = os.path.join(options.mgltools_root, "MGLToolsPckgs", "AutoDockTools", "Utilities24", "prepare_receptor4.py")
        prep_lig = os.path.join(options.mgltools_root, "MGLToolsPckgs", "AutoDockTools", "Utilities24", "prepare_ligand4.py")
        if not os.path.isfile(pythonsh):
            raise DockingError(f"MGLTools pythonsh not found at {pythonsh}")
        if not os.path.isfile(prep_rec) or not os.path.isfile(prep_lig):
            raise DockingError("MGLTools prepare scripts not found under provided root")

        # Receptor
        if not os.path.isfile(receptor_pdbqt):
            cmd_rec = [pythonsh, prep_rec, "-r", protein_pdb, "-o", receptor_pdbqt, "-A", "hydrogens"]
            _run(cmd_rec, "MGLTools receptor preparation failed")
        # Ligand
        if not os.path.isfile(ligand_pdbqt):
            cmd_lig = [pythonsh, prep_lig, "-l", ligand_file, "-o", ligand_pdbqt, "-A", "hydrogens", "-C"]
            _run(cmd_lig, "MGLTools ligand preparation failed")
    else:
        # Prepare receptor/ligand PDBQT using Open Babel
        if not os.path.isfile(receptor_pdbqt):
            cmd_rec = [
                options.obabel_bin,
                "-i", "pdb", protein_pdb,
                "-o", "pdbqt", "-O", receptor_pdbqt,
                "-p", "7.4", "-h", "--partialcharge", "gasteiger",
            ]
            _run(cmd_rec, "Open Babel receptor conversion failed")

        if not os.path.isfile(ligand_pdbqt):
            in_fmt = _guess_format(ligand_file)
            cmd_lig = [
                options.obabel_bin,
                "-i", in_fmt, ligand_file,
                "-o", "pdbqt", "-O", ligand_pdbqt,
                "-p", "7.4", "-h", "--partialcharge", "gasteiger",
            ]
            _run(cmd_lig, "Open Babel ligand conversion failed")

    # Run Vina
    cx, cy, cz = center_xyz_ang
    size = float(box_size_ang)
    cmd_vina = [
        options.vina_bin,
        "--receptor", receptor_pdbqt,
        "--ligand", ligand_pdbqt,
        "--center_x", str(cx),
        "--center_y", str(cy),
        "--center_z", str(cz),
        "--size_x", str(size),
        "--size_y", str(size),
        "--size_z", str(size),
        "--exhaustiveness", str(int(options.exhaustiveness)),
        "--num_modes", str(int(options.num_modes)),
        "--out", out_pdbqt,
    ]
    proc = _run(cmd_vina, "Vina docking failed")
    try:
        with open(log_path, "w") as f:
            f.write(proc.stdout)
            f.write("\n==== STDERR ====\n")
            f.write(proc.stderr)
    except Exception:
        pass

    # Convert docked pose to PDB
    cmd_out = [
        options.obabel_bin,
        "-i", "pdbqt", out_pdbqt,
        "-o", "pdb", "-O", out_pdb,
    ]
    _run(cmd_out, "Open Babel PDB conversion failed")

    if not os.path.isfile(out_pdb):
        raise DockingError("Docking output PDB not found")
    return out_pdb


def _guess_format(path: str) -> str:
    low = path.lower()
    if low.endswith(".sdf"): return "sdf"
    if low.endswith(".mol2"): return "mol2"
    if low.endswith(".pdb"): return "pdb"
    if low.endswith(".pdbqt"): return "pdbqt"
    return "sdf"


def _run(cmd, err_msg):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise DockingError(f"{err_msg}: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return proc


## Trajectory rescoring with Vina intentionally removed to avoid mixed scoring in reports
