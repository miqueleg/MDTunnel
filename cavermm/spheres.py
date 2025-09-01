from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np


@dataclass
class Sphere:
    x: float
    y: float
    z: float
    r: float

    def center_nm(self) -> np.ndarray:
        # input is assumed Angstroms; convert to nm
        return np.array([self.x, self.y, self.z], dtype=float) * 0.1


def load_spheres(path: str) -> List[Sphere]:
    """Load spheres from CSV (x,y,z,r in Angstrom) or PDB with resname SPH.

    - CSV: header optional; comma or whitespace separated.
    - PDB: HETATM with resname SPH; radius read from B-factor (fallback 1.5 Å).
    """
    low = path.lower()
    if low.endswith(".pdb"):
        return _load_spheres_pdb(path)
    if low.endswith(".csv"):
        return _load_spheres_csv(path)
    if low.endswith(".dsd"):
        return _load_spheres_dsd(path)
    # Try CSV as fallback
    return _load_spheres_csv(path)


def _load_spheres_csv(path: str) -> List[Sphere]:
    import csv

    spheres: List[Sphere] = []
    with open(path, "r") as f:
        sample = f.read(2048)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=", \t")
        reader = csv.reader(f, dialect)
        rows = list(reader)

    # Drop header if present (non-numeric first token)
    start_idx = 0
    if rows and rows[0]:
        try:
            float(rows[0][0])
        except ValueError:
            start_idx = 1

    for row in rows[start_idx:]:
        if not row or len(row) < 3:
            continue
        vals = [float(x) for x in row[:4]]
        x, y, z = vals[:3]
        r = vals[3] if len(vals) > 3 else 1.5
        spheres.append(Sphere(x, y, z, r))
    return spheres


def _load_spheres_pdb(path: str) -> List[Sphere]:
    spheres: List[Sphere] = []
    with open(path, "r") as f:
        for line in f:
            if not (line.startswith("HETATM") or line.startswith("ATOM  ")):
                continue
            resname = line[17:20].strip()
            if resname.upper() != "SPH":
                continue
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            try:
                b = float(line[60:66])
            except ValueError:
                b = 1.5
            spheres.append(Sphere(x, y, z, b))
    return spheres


def _load_spheres_dsd(path: str) -> List[Sphere]:
    """Load CAVER .dsd spheres. Expected columns per line:
    x y z nx ny nz r  (Å)
    Normals are ignored; radius is the last column.
    """
    spheres: List[Sphere] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
                z = float(parts[2])
                r = float(parts[-1])
            except ValueError:
                continue
            spheres.append(Sphere(x, y, z, r))
    return spheres
