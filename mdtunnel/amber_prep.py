from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass


@dataclass
class AmberPrepOptions:
    antechamber_bin: str = "antechamber"
    parmchk2_bin: str = "parmchk2"
    tleap_bin: str = "tleap"
    pdb4amber_bin: str = "pdb4amber"
    ligand_charge: int = 0
    ligand_resname: str = "LIG"
    gaff_version: str = "gaff2"  # gaff2 recommended


class AmberPrepError(RuntimeError):
    pass


def prep_complex_with_ambertools(
    protein_pdb: str,
    ligand_pdb: str,
    outdir: str,
    options: AmberPrepOptions,
) -> tuple[str, str]:
    """Use antechamber+parmchk2+tleap to make an Amber complex for protein+ligand.

    Returns (prmtop, inpcrd) paths.
    - Protein: parametrized with ff14SB, hydrogens added by tleap.
    - Ligand: GAFF2 with AM1-BCC charges at specified integer charge.
    - No solvent or ions are added.
    """
    os.makedirs(outdir, exist_ok=True)

    def _which_or_err(name, exe):
        if shutil.which(exe) is None:
            raise AmberPrepError(f"Executable not found: {name} = {exe}")

    _which_or_err("antechamber", options.antechamber_bin)
    _which_or_err("parmchk2", options.parmchk2_bin)
    _which_or_err("tleap", options.tleap_bin)
    _which_or_err("pdb4amber", options.pdb4amber_bin)

    lig_mol2 = os.path.join(outdir, "lig.mol2")
    lig_frcmod = os.path.join(outdir, "lig.frcmod")
    leap_in = os.path.join(outdir, "leap.in")
    leap_log = os.path.join(outdir, "leap.log")
    prmtop = os.path.join(outdir, "complex.prmtop")
    inpcrd = os.path.join(outdir, "complex.inpcrd")

    # antechamber (add H, type atoms, AM1-BCC charges)
    cmd_ante = [
        options.antechamber_bin,
        "-i", ligand_pdb,
        "-fi", "pdb",
        "-o", lig_mol2,
        "-fo", "mol2",
        "-c", "bcc",
        "-s", "2",
        "-nc", str(int(options.ligand_charge)),
        "-rn", options.ligand_resname,
        "-at", options.gaff_version,
    ]
    _run(cmd_ante, "Antechamber failed")

    # parmchk2 to create frcmod for GAFF
    cmd_parm = [
        options.parmchk2_bin,
        "-i", lig_mol2,
        "-f", "mol2",
        "-o", lig_frcmod,
        "-s", options.gaff_version,
    ]
    _run(cmd_parm, "parmchk2 failed")

    # tleap input
    leaprc_gaff = "leaprc.gaff2" if options.gaff_version == "gaff2" else "leaprc.gaff"
    # Clean protein PDB for Amber with pdb4amber
    prot_clean = os.path.join(outdir, "protein_pdb4amber.pdb")
    _run([options.pdb4amber_bin, "-i", protein_pdb, "-o", prot_clean], "pdb4amber failed")
    # Work around NSER H naming: remove lone generic 'H' at N-terminus that can confuse LEaP
    prot_for_leap = os.path.join(outdir, "protein_pdb4amber_fixed.pdb")
    try:
        with open(prot_clean, "r") as fin, open(prot_for_leap, "w") as fout:
            for line in fin:
                if line.startswith(("ATOM", "HETATM")):
                    atom_name = line[12:16].strip()
                    resseq = line[22:26].strip()
                    if resseq == "1" and atom_name == "H":
                        # Skip problematic generic H at residue 1; LEaP will add correct Hs
                        continue
                fout.write(line)
    except Exception:
        # If anything goes wrong, fall back to the original cleaned PDB
        prot_for_leap = prot_clean
    protein_abs = os.path.abspath(prot_for_leap)
    leap_text = f"""
source leaprc.protein.ff14SB
source {leaprc_gaff}
LIG = loadmol2 {os.path.basename(lig_mol2)}
loadamberparams {os.path.basename(lig_frcmod)}
PROT = loadpdb {protein_abs}
complex = combine {{PROT LIG}}
saveamberparm complex {os.path.basename(prmtop)} {os.path.basename(inpcrd)}
quit
"""
    with open(leap_in, "w") as f:
        f.write(leap_text)

    # Run tleap in outdir so relative paths work
    _run([options.tleap_bin, "-f", os.path.basename(leap_in)], "tleap failed", cwd=outdir, capture_log=leap_log)

    if not (os.path.isfile(prmtop) and os.path.isfile(inpcrd)):
        raise AmberPrepError("tleap did not produce complex.prmtop/inpcrd")
    return prmtop, inpcrd


def _run(cmd, err, cwd=None, capture_log: str | None = None):
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if capture_log:
        with open(capture_log, "w") as f:
            f.write(proc.stdout)
            f.write("\n==== STDERR ====\n")
            f.write(proc.stderr)
    if proc.returncode != 0:
        raise AmberPrepError(f"{err}: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

