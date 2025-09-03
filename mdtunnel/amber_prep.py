from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass


def pre_clean_protein_pdb(in_pdb: str, outdir: str) -> str:
    """Pre-clean a protein PDB for docking/Amber.

    - Drop alternate locations (keep highest occupancy; prefer altLoc 'A' or blank).
    - Remove duplicate atoms within a residue.
    - Standardize common nonstandard residue names to Amber ff equivalents:
      * NSER->SER, NXXX->XXX; CXXX->XXX (terminal variants)
      * HSD->HID, HSE->HIE, HSP->HIP
      * CHIS/CHID/CHIE->HIS/HID/HIE
      * MSE->MET and rename atom SE->SD (element Se->S)
    - Keep only standard protein residues (ATOM records) and common caps (ACE,NME).
    Returns path to cleaned PDB.
    """
    import os as _os
    _os.makedirs(outdir, exist_ok=True)
    out_pdb = _os.path.join(outdir, "protein_preclean.pdb")

    aa_std = {
        "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
        "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
        "HID","HIE","HIP",
    }
    caps = {"ACE","NME"}

    def _std_resname(rn: str) -> str:
        r = rn.upper().strip()
        # Map all histidine variants to generic HIS so LEaP assigns H consistently
        if r in ("HIS","HID","HIE","HIP","HSD","HSE","HSP","CHIS","CHID","CHIE"):
            return "HIS"
        if len(r) == 4 and r[0] in ("N","C") and r[1:] in aa_std:
            return r[1:]
        if r == "MSE":
            return "MET"
        return r

    # Collect records grouped by residue and atom name
    residues = {}
    lines_by_res = {}
    with open(in_pdb, "r") as fin:
        for line in fin:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            resn = line[17:20]
            resn_std = _std_resname(resn)
            rec = line[:6]
            if rec.startswith("HETATM") and resn_std not in caps and resn_std not in aa_std:
                # Skip non-protein hetero residues
                continue
            chain = line[21]
            resi = line[22:26]
            icode = line[26]
            atname = line[12:16]
            altloc = line[16]
            # occupancy defaults to 0.0 if missing/malformed
            try:
                occ = float(line[54:60])
            except Exception:
                occ = 0.0
            key = (chain, resi, icode, resn_std)
            atom_key = atname.strip()
            # Build a sanitized line with standardized residue name and MSE->MET atom tweak
            line_mod = list(line)
            # Replace residue name
            resn_fmt = f"{resn_std:>3}"
            line_mod[17:20] = list(resn_fmt)
            # If MSE->MET, map atom SE->SD and element to S
            if resn.strip().upper() == "MSE":
                atom_raw = atname
                if atom_raw.strip().upper() == "SE":
                    # rename to SD; keep column alignment
                    new_atom = f" SD "
                    line_mod[12:16] = list(new_atom)
                    # element columns 77-78 to S
                    elem = " S"
                    if len(line_mod) >= 78:
                        line_mod[76:78] = list(elem)

            # Reconstruct possibly modified line
            line_new = "".join(line_mod)

            if key not in residues:
                residues[key] = {}
            if atom_key not in residues[key]:
                residues[key][atom_key] = []
            residues[key][atom_key].append((occ, altloc, line_new))

    # Choose best altLoc per atom and write out
    def _choose(recs):
        # Prefer highest occupancy, then altloc 'A', then blank, else first
        recs_sorted = sorted(recs, key=lambda t: (t[0], t[1] == 'A', t[1] == ' '), reverse=True)
        return recs_sorted[0][2]

    with open(out_pdb, "w") as fout:
        last_chain = None
        for key in sorted(residues, key=lambda k: (k[0], int(k[1].strip() or 0), k[2])):
            chain, resi, icode, resn_std = key
            if last_chain is not None and chain != last_chain:
                fout.write("TER\n")
            atom_map = residues[key]
            for atom_name in sorted(atom_map.keys()):
                fout.write(_choose(atom_map[atom_name]))
            last_chain = chain
        fout.write("TER\nEND\n")

    return out_pdb


@dataclass
class AmberPrepOptions:
    antechamber_bin: str = "antechamber"
    parmchk2_bin: str = "parmchk2"
    tleap_bin: str = "tleap"
    pdb4amber_bin: str = "pdb4amber"
    ligand_charge: int = 0
    ligand_resname: str = "LIG"
    gaff_version: str = "gaff2"  # gaff2 recommended
    charge_method: str = "bcc"   # bcc requires sqm; use 'gas' to avoid sqm
    skip_pdb4amber: bool = False  # if True, assume protein is Amber-ready


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
    have_pdb4amber = shutil.which(options.pdb4amber_bin) is not None

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
        "-c", options.charge_method,
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
    # Clean protein PDB for Amber with pdb4amber (best-effort)
    prot_clean = os.path.join(outdir, "protein_pdb4amber.pdb")
    if not options.skip_pdb4amber and have_pdb4amber:
        try:
            # Use flags to handle altlocs/occupancy quirks where possible
            _run([options.pdb4amber_bin, "-i", protein_pdb, "-o", prot_clean, "-y"], "pdb4amber failed")
        except Exception:
            prot_clean = protein_pdb
    else:
        prot_clean = protein_pdb
    # Work around terminal naming and standardize residue names again after pdb4amber
    prot_for_leap = os.path.join(outdir, "protein_pdb4amber_fixed.pdb")
    try:
        def _std_resname_after(rn: str) -> str:
            r = rn.upper().strip()
            # Keep generic HIS at this stage; we'll auto-resolve to HID/HIE/HIP by explicit H atoms below
            if r in ("HID","HIE","HIP","HSD","HSE","HSP","CHIS","CHID","CHIE"):
                return "HIS"
            if len(r) == 4 and r[0] in ("N","C"):
                return r[1:]
            if r == "MSE":
                return "MET"
            return r

        # First pass: load lines, standardize residue names and collect atoms per residue
        residues = {}
        lines = []
        first_res_by_chain: dict[str, int] = {}
        with open(prot_clean, "r") as fin:
            for line in fin:
                if line.startswith(("ATOM", "HETATM")):
                    atom_name = line[12:16].strip()
                    resseq = line[22:26].strip()
                    chain_id = line[21]
                    try:
                        resi_int = int(resseq)
                    except Exception:
                        resi_int = 10**9
                    # Track first residue index per chain
                    first = first_res_by_chain.get(chain_id)
                    if first is None or resi_int < first:
                        first_res_by_chain[chain_id] = resi_int
                    resn = line[17:20]
                    resn_std = _std_resname_after(resn)
                    line_mod = list(line)
                    line_mod[17:20] = list(f"{resn_std:>3}")
                    # Map MSE->MET atom SE->SD and element to S
                    if resn.strip().upper() == "MSE" and atom_name.upper() == "SE":
                        line_mod[12:16] = list(" SD ")
                        if len(line_mod) >= 78:
                            line_mod[76:78] = list(" S")
                    line = "".join(line_mod)
                    # Defer skipping generic H on first residue of each chain to second pass
                    lines.append(line)
                    key = (line[21], line[22:26], line[26])
                    residues.setdefault(key, {"resn": resn_std, "atoms": set()})
                    residues[key]["atoms"].add(atom_name.upper())
                else:
                    lines.append(line)

        # Decide per-residue histidine tautomer by explicit hydrogens
        resname_override = {}
        for key, data in residues.items():
            rn = data["resn"].upper()
            if rn == "HIS":
                atoms = data["atoms"]
                has_hd1 = "HD1" in atoms
                has_he2 = "HE2" in atoms
                new = "HIS"
                if has_hd1 and not has_he2:
                    new = "HID"
                elif has_he2 and not has_hd1:
                    new = "HIE"
                elif has_hd1 and has_he2:
                    new = "HIP"
                resname_override[key] = new

        # Second pass: write with overrides
        with open(prot_for_leap, "w") as fout:
            for line in lines:
                if line.startswith(("ATOM", "HETATM")):
                    chain_id = line[21]
                    resseq = line[22:26].strip()
                    atom_name = line[12:16].strip()
                    # Skip generic H on first residue of each chain
                    try:
                        if atom_name == 'H' and int(resseq) == first_res_by_chain.get(chain_id, -1):
                            continue
                    except Exception:
                        pass
                    key = (chain_id, line[22:26], line[26])
                    if key in resname_override:
                        new_rn = resname_override[key]
                        line_mod = list(line)
                        line_mod[17:20] = list(f"{new_rn:>3}")
                        line = "".join(line_mod)
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
