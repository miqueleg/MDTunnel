from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class VinaOptions:
    vina_bin: str = "vina"
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
    Requires `vina` and MGLTools prepare_* scripts (prepare_ligand4.py/prepare_receptor4.py).
    """
    options = options or VinaOptions()
    os.makedirs(outdir, exist_ok=True)

    def _check_exec(name, path):
        if shutil.which(path) is None:
            raise DockingError(f"Required executable '{name}' not found: {path}")

    _check_exec("vina", options.vina_bin)

    receptor_pdbqt = os.path.abspath(os.path.join(outdir, "receptor.pdbqt"))
    ligand_pdbqt = os.path.abspath(os.path.join(outdir, "ligand.pdbqt"))
    out_pdbqt = os.path.abspath(os.path.join(outdir, "docked.pdbqt"))
    out_pdb = os.path.abspath(os.path.join(outdir, "docked.pdb"))
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
            cmd_rec = [pythonsh, prep_rec, "-r", os.path.abspath(protein_pdb), "-o", os.path.basename(receptor_pdbqt), "-A", "hydrogens"]
            _run_cwd(cmd_rec, "MGLTools receptor preparation failed", cwd=os.path.dirname(receptor_pdbqt))
        _sanitize_pdbqt(receptor_pdbqt)
        # Ligand
        if not os.path.isfile(ligand_pdbqt):
            lig_in = _ensure_ligand_mgltools_compatible(ligand_file, outdir)
            lig_in_abs = os.path.abspath(lig_in)
            cmd_lig = [pythonsh, prep_lig, "-l", lig_in_abs, "-o", os.path.basename(ligand_pdbqt), "-A", "hydrogens", "-U", "nphs_lps", "-C"]
            _run_cwd(cmd_lig, "MGLTools ligand preparation failed", cwd=os.path.dirname(ligand_pdbqt))
    else:
        raise DockingError("MGLTools root must be provided for PDBQT preparation (Open Babel is not supported)")

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

    # Convert docked pose to PDB (simple PDBQT->PDB conversion)
    _pdbqt_to_pdb(out_pdbqt, out_pdb)

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

def _run_cwd(cmd, err_msg, cwd: Optional[str] = None):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=cwd)
    if proc.returncode != 0:
        raise DockingError(f"{err_msg}: {' '.join(cmd)} (cwd={cwd})\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return proc


def _ensure_pdbqt_receptor_and_ligand(
    protein_pdb: str,
    ligand_file: str,
    outdir: str,
    options: VinaOptions,
) -> Tuple[str, str]:
    """Prepare receptor and ligand PDBQT files using MGLTools (required)."""
    os.makedirs(outdir, exist_ok=True)
    receptor_pdbqt = os.path.abspath(os.path.join(outdir, "receptor.pdbqt"))
    ligand_pdbqt = os.path.abspath(os.path.join(outdir, "ligand.pdbqt"))

    def _check_exec(name, path):
        if shutil.which(path) is None:
            raise DockingError(f"Required executable '{name}' not found: {path}")

    _check_exec("vina", options.vina_bin)

    if options.mgltools_root:
        pythonsh = os.path.join(options.mgltools_root, "bin", "pythonsh")
        prep_rec = os.path.join(options.mgltools_root, "MGLToolsPckgs", "AutoDockTools", "Utilities24", "prepare_receptor4.py")
        prep_lig = os.path.join(options.mgltools_root, "MGLToolsPckgs", "AutoDockTools", "Utilities24", "prepare_ligand4.py")
        if not os.path.isfile(receptor_pdbqt):
            if not (os.path.isfile(pythonsh) and os.path.isfile(prep_rec)):
                raise DockingError("MGLTools receptor preparation tools not found")
            _run_cwd([pythonsh, prep_rec, "-r", os.path.abspath(protein_pdb), "-o", os.path.basename(receptor_pdbqt), "-A", "hydrogens"], "MGLTools receptor preparation failed", cwd=os.path.dirname(receptor_pdbqt))
        _sanitize_pdbqt(receptor_pdbqt)
        if not os.path.isfile(ligand_pdbqt):
            if not (os.path.isfile(pythonsh) and os.path.isfile(prep_lig)):
                raise DockingError("MGLTools ligand preparation tools not found")
            lig_in = _ensure_ligand_mgltools_compatible(ligand_file, outdir)
            lig_in_abs = os.path.abspath(lig_in)
            _run_cwd([pythonsh, prep_lig, "-l", lig_in_abs, "-o", os.path.basename(ligand_pdbqt), "-A", "hydrogens", "-U", "nphs_lps", "-C"], "MGLTools ligand preparation failed", cwd=os.path.dirname(ligand_pdbqt))
    else:
        raise DockingError("MGLTools root must be provided for PDBQT preparation (Open Babel is not supported)")
    return receptor_pdbqt, ligand_pdbqt


def _sanitize_pdbqt(path: str) -> None:
    """Remove torsion tree tags from PDBQT (receptor or ligand) in place.

    For score-only evaluations, Vina does not require torsion-tree blocks. Some
    builds can fail if these tags appear in receptor or ligand files prepared by
    third-party tools. This strips ROOT/ENDROOT/BRANCH/TORSDOF lines.
    """
    try:
        if not os.path.isfile(path):
            return
        with open(path, "r") as f:
            lines = f.readlines()
        bad_prefixes = ("ROOT", "ENDROOT", "BRANCH", "TORSDOF")
        changed = any(line.startswith(bad_prefixes) for line in lines)
        if not changed:
            return
        with open(path, "w") as f:
            for line in lines:
                if line.startswith(bad_prefixes):
                    continue
                f.write(line)
    except Exception:
        # Best-effort; leave as-is on error
        pass


def run_vina_score_only(
    protein_pdb: str,
    ligand_file: str,
    outdir: str,
    options: Optional[VinaOptions] = None,
) -> float:
    """Score a single ligand pose with Vina (no movement). Returns kcal/mol affinity.

    Uses --score_only; converts inputs to PDBQT if needed.
    """
    options = options or VinaOptions()
    os.makedirs(outdir, exist_ok=True)
    rec_pdbqt, lig_pdbqt = _ensure_pdbqt_receptor_and_ligand(protein_pdb, ligand_file, outdir, options)
    # Sanitize receptor only; keep ligand TORSDOF/BRANCH for Vina
    _sanitize_pdbqt(rec_pdbqt)
    # Derive a reasonable grid from ligand center to satisfy some Vina builds
    cx = cy = cz = 0.0
    size = 20.0
    try:
        if ligand_file.lower().endswith('.pdb') and os.path.isfile(ligand_file):
            xs = []
            ys = []
            zs = []
            with open(ligand_file, 'r') as f:
                for line in f:
                    if line.startswith('ATOM') or line.startswith('HETATM'):
                        try:
                            xs.append(float(line[30:38]))
                            ys.append(float(line[38:46]))
                            zs.append(float(line[46:54]))
                        except Exception:
                            pass
            if xs:
                cx = sum(xs)/len(xs)
                cy = sum(ys)/len(ys)
                cz = sum(zs)/len(zs)
                # size as 2x max extent + margin
                rx = (max(xs)-min(xs)) if len(xs)>1 else 10.0
                ry = (max(ys)-min(ys)) if len(ys)>1 else 10.0
                rz = (max(zs)-min(zs)) if len(zs)>1 else 10.0
                size = max(rx, ry, rz) + 10.0
    except Exception:
        pass
    cmd = [options.vina_bin, "--receptor", rec_pdbqt, "--ligand", lig_pdbqt, "--score_only",
           "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
           "--size_x", str(size), "--size_y", str(size), "--size_z", str(size)]
    proc = _run(cmd, "Vina scoring failed")
    # Parse affinity (kcal/mol) from stdout
    aff = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.lower().startswith("affinity:"):
            # Example: Affinity: -7.3 (kcal/mol)
            try:
                parts = line.split()
                aff = float(parts[1])
                break
            except Exception:
                pass
        if "Estimated Free Energy of Binding" in line:
            # Fallback parse: number immediately after the colon
            import re
            m = re.search(r":\s*([-+]?\d*\.\d+|[-+]?\d+)", line)
            if m:
                try:
                    aff = float(m.group(1))
                    break
                except Exception:
                    pass
    if aff is None:
        # Try stderr too
        for line in proc.stderr.splitlines():
            line = line.strip()
            if line.lower().startswith("affinity:"):
                try:
                    parts = line.split()
                    aff = float(parts[1])
                    break
                except Exception:
                    pass
    if aff is None:
        raise DockingError("Failed to parse Vina affinity from output")
    return float(aff)


def run_vina_rescore_frames(
    protein_pdb: str,
    frames_dir: str,
    outdir: str,
    options: Optional[VinaOptions] = None,
) -> str:
    """Score every frame_*.pdb in frames_dir with Vina and write vina_profile.csv.

    Returns path to the CSV written with columns: step, vina_kcal_mol
    """
    options = options or VinaOptions()
    os.makedirs(outdir, exist_ok=True)
    import re
    frame_paths = sorted([f for f in os.listdir(frames_dir) if re.match(r"step_\d+\.pdb$", f)])
    rows: List[Tuple[int, float]] = []
    tmp_dir = os.path.join(outdir, "vina_rescore")
    os.makedirs(tmp_dir, exist_ok=True)
    for fname in frame_paths:
        step = int(fname.split("_")[1].split(".")[0])
        frame_path = os.path.join(frames_dir, fname)
        # Extract ligand-only PDB from frame (assumes resname LIG)
        lig_path = os.path.join(tmp_dir, f"s{step:04d}", "ligand_only.pdb")
        os.makedirs(os.path.dirname(lig_path), exist_ok=True)
        try:
            with open(frame_path, 'r') as fin, open(lig_path, 'w') as fout:
                serial = 1
                for line in fin:
                    if (line.startswith('ATOM') or line.startswith('HETATM')) and line[17:20].strip() == 'LIG':
                        raw_name = line[12:16]
                        resn = 'LIG'
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        occ = line[54:60] if len(line) >= 60 else '  1.00'
                        bfac = line[60:66] if len(line) >= 66 else '  0.00'
                        elem_sym = (line[76:78].strip() if len(line) >= 78 else '').title()
                        if not elem_sym:
                            rn = raw_name.strip().upper()
                            elem_sym = 'Cl' if rn.startswith('CL') else ('Br' if rn.startswith('BR') else rn[0].title())
                        # Build safe atom name: element + index, avoid names like CE1 mis-read as cerium
                        name = (f"{elem_sym}{serial%1000:>3}" )[:4]
                        elem = elem_sym.rjust(2)
                        record = 'HETATM'
                        fout.write(f"{record:<6}{serial:5d} {name:<4} {resn:>3} L   1    {x:8.3f}{y:8.3f}{z:8.3f}{occ:>6}{bfac:>6}          {elem:>2}\n")
                        serial += 1
                fout.write('TER\n')
        except Exception as e:
            raise DockingError(f"Failed to extract ligand from frame {fname}: {e}")
        try:
            step_dir = os.path.join(tmp_dir, f"s{step:04d}")
            # Ensure fresh PDBQT generation each time
            try:
                for fn in ("receptor.pdbqt", "ligand.pdbqt"):
                    fp = os.path.join(step_dir, fn)
                    if os.path.exists(fp):
                        os.remove(fp)
            except Exception:
                pass
            aff = run_vina_score_only(protein_pdb, lig_path, step_dir, options)
        except Exception as e:
            raise DockingError(f"Vina scoring failed for frame {fname}: {e}")
        rows.append((step, float(aff)))
    # Write CSV
    import csv
    csv_path = os.path.join(outdir, "vina_profile.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "vina_kcal_mol"])
        for s, a in rows:
            w.writerow([s, a])
    return csv_path


def _pdbqt_to_pdb(in_path: str, out_path: str) -> None:
    """Convert PDBQT to PDB by copying ATOM/HETATM lines and stripping PDBQT extras.

    This preserves coordinates for downstream use; connectivity is not written.
    """
    try:
        with open(in_path, 'r') as fin, open(out_path, 'w') as fout:
            serial = 1
            for line in fin:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    # PDBQT is PDB-compatible in columns 1-66; element at 77-78
                    name = line[12:16]
                    resn = line[17:20]
                    chain = line[21]
                    resi = line[22:26]
                    x = line[30:38]
                    y = line[38:46]
                    z = line[46:54]
                    occ = line[54:60]
                    bfac = line[60:66]
                    elem = (line[76:78] if len(line) >= 78 else name.strip()[0]).rjust(2)
                    record = 'HETATM'
                    fout.write(f"{record:<6}{serial:5d} {name:<4}{resn:>4}{chain}{resi}   {x}{y}{z}{occ}{bfac}          {elem:>2}\n")
                    serial += 1
            fout.write('TER\n')
    except Exception as e:
        raise DockingError(f"Failed to convert PDBQT to PDB: {e}")


def _ensure_ligand_mgltools_compatible(ligand_file: str, outdir: str) -> str:
    """Ensure ligand is in a format MGLTools can parse (PDB or MOL2).

    If input is already PDB/MOL2/PDBQT, return as-is. Otherwise, attempt conversion to MOL2 via RDKit.
    """
    low = ligand_file.lower()
    if low.endswith(".mol2") or low.endswith('.pdbqt'):
        # Ensure the file exists under outdir so MGLTools (which can be picky
        # about cwd) can find it when we pass a basename.
        src = os.path.abspath(ligand_file)
        dst = os.path.abspath(os.path.join(outdir, os.path.basename(src)))
        if src != dst:
            try:
                os.makedirs(outdir, exist_ok=True)
                import shutil as _sh
                _sh.copyfile(src, dst)
                return dst
            except Exception:
                # Fall back to absolute path; caller uses absolute -l
                return src
        return dst
    if low.endswith('.pdb'):
        # Prefer MOL2 via antechamber to ensure bond orders for MGLTools
        try:
            out_mol2 = os.path.abspath(os.path.join(outdir, 'ligand_input.mol2'))
            cmd = ['antechamber', '-i', os.path.abspath(ligand_file), '-fi', 'pdb', '-o', out_mol2, '-fo', 'mol2', '-c', 'gas', '-s', '2']
            _run(cmd, 'Antechamber conversion failed')
            if not os.path.isfile(out_mol2):
                raise RuntimeError('Antechamber did not produce MOL2')
            return out_mol2
        except Exception as e2:
            # Fall back to original PDB
            return os.path.abspath(ligand_file)
    # Prefer Antechamber for SDF to ensure bond orders for MGLTools
    if low.endswith('.sdf'):
        try:
            out_mol2 = os.path.abspath(os.path.join(outdir, 'ligand_input.mol2'))
            cmd = ['antechamber', '-i', os.path.abspath(ligand_file), '-fi', 'sdf', '-o', out_mol2, '-fo', 'mol2', '-c', 'gas', '-s', '2']
            _run(cmd, 'Antechamber conversion failed')
            if not os.path.isfile(out_mol2):
                raise RuntimeError('Antechamber did not produce MOL2')
            return out_mol2
        except Exception:
            pass
    # Try RDKit conversion to PDB (robust reader for MGLTools)
    try:
        from rdkit import Chem
        mol = None
        if low.endswith('.sdf'):
            suppl = Chem.SDMolSupplier(ligand_file, removeHs=False, sanitize=True)
            mol = next((m for m in suppl if m is not None), None)
        elif low.endswith('.mol'):
            mol = Chem.MolFromMolFile(ligand_file, removeHs=False, sanitize=True)
        else:
            # Try a quick guess from file contents as SMILES
            try:
                with open(ligand_file, 'r') as fh:
                    smi = fh.read().strip()
                if smi:
                    mol = Chem.MolFromSmiles(smi)
            except Exception:
                mol = None
        if mol is None:
            raise RuntimeError("RDKit failed to read ligand")
        molH = Chem.AddHs(mol, addCoords=True)
        out_pdb = os.path.abspath(os.path.join(outdir, 'ligand_input.pdb'))
        # Ensure dir exists
        os.makedirs(outdir, exist_ok=True)
        if Chem.MolToPDBFile(molH, out_pdb) is None:
            # Some RDKit versions return None; check existence
            pass
        if not os.path.isfile(out_pdb):
            raise RuntimeError("RDKit failed to write PDB")
        return out_pdb
    except Exception as e:
        # Fallback: try AmberTools antechamber to generate MOL2
        try:
            fmt = _guess_format(ligand_file)
            out_mol2 = os.path.abspath(os.path.join(outdir, 'ligand_input.mol2'))
            cmd = ['antechamber', '-i', os.path.abspath(ligand_file), '-fi', fmt, '-o', out_mol2, '-fo', 'mol2', '-c', 'gas', '-s', '2']
            _run(cmd, 'Antechamber conversion failed')
            if not os.path.isfile(out_mol2):
                raise RuntimeError('Antechamber did not produce MOL2')
            return out_mol2
        except Exception as e2:
            raise DockingError(f"Failed to convert ligand to a MGLTools-compatible format: RDKit error: {e}; Antechamber error: {e2}")
