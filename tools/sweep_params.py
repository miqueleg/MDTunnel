#!/usr/bin/env python3
from __future__ import annotations

import csv
import itertools
import os
import subprocess
import time

PROTEIN = "LinB_WT.pdb"
LIGAND = "DBE.pdb"
SPHERES = "CW_tunnel.dsd"

def run(cmd: list[str]) -> tuple[int, str, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout, p.stderr

def parse_energy_profile(path: str) -> tuple[float, float]:
    # returns (max_jump_kcal, mean_jump_kcal)
    import pandas as pd
    df = pd.read_csv(path)
    # prefer E_kcal_mol if present otherwise convert
    if "E_kcal_mol" in df.columns:
        e = df["E_kcal_mol"].astype(float).to_numpy()
    else:
        e = df["E_kJ_mol"].astype(float).to_numpy() / 4.184
    import numpy as np
    if len(e) < 2:
        return 0.0, 0.0
    de = np.abs(np.diff(e))
    return float(de.max()), float(de.mean())

def parse_com(path: str) -> tuple[float, float]:
    # returns (max_com_A, mean_com_A)
    import pandas as pd
    df = pd.read_csv(path)
    a = df["com_to_sphere_A"].astype(float).to_numpy()
    import numpy as np
    return float(a.max()) if len(a) else 0.0, float(a.mean()) if len(a) else 0.0

def main():
    kcom_list = [1000, 2000]
    ramp_list = [2.0, 3.0, 5.0]
    targetA_list = [0.2, 0.5]
    min_tol_list = [0.005]
    max_iter_list = [30000]

    combos = list(itertools.product(kcom_list, ramp_list, targetA_list, min_tol_list, max_iter_list))
    out_root = "runs"
    os.makedirs(out_root, exist_ok=True)
    rows = []

    for i, (kcom, ramp, targA, tol, iters) in enumerate(combos, start=1):
        outdir = os.path.join(out_root, f"sweep_{i:02d}_k{kcom}_r{ramp}_t{targA}_tol{tol}_it{iters}")
        cmd = [
            "conda", "run", "-n", "OpenMM", "python", "-m", "mdtunnel.cli",
            "--protein", PROTEIN,
            "--ligand", LIGAND,
            "--spheres", SPHERES,
            "--out", outdir,
            "--ambertools-prep",
            "--ligand-charge", "0",
            "--ligand-resname", "LIG",
            "--kcom", str(kcom),
            "--kpos-protein", "10000",
            "--min-tol-kj", str(tol),
            "--min-max-iter", str(iters),
            "--kcom-ramp-scale", str(ramp),
            "--com-target-A", str(targA),
            "--com-max-k", "2000000",
            "--com-scale", "5",
        ]
        t0 = time.time()
        code, out, err = run(cmd)
        t = time.time() - t0
        max_jump = mean_jump = max_com = mean_com = float("nan")
        status = "ok" if code == 0 else f"err:{code}"
        try:
            epath = os.path.join(outdir, "energy_profile.csv")
            cpath = os.path.join(outdir, "com_to_sphere.csv")
            if os.path.isfile(epath):
                max_jump, mean_jump = parse_energy_profile(epath)
            if os.path.isfile(cpath):
                max_com, mean_com = parse_com(cpath)
        except Exception as ex:
            status += f" parse_fail:{ex}"
        rows.append({
            "outdir": outdir,
            "kcom": kcom,
            "ramp": ramp,
            "targetA": targA,
            "min_tol": tol,
            "max_iter": iters,
            "max_jump_kcal": round(max_jump, 3),
            "mean_jump_kcal": round(mean_jump, 3),
            "max_com_A": round(max_com, 3),
            "mean_com_A": round(mean_com, 3),
            "runtime_s": round(t, 2),
            "status": status,
        })
        print(f"[{i}/{len(combos)}] {status} out={outdir} jumps(max/mean)={max_jump:.3f}/{mean_jump:.3f} kcal, COM(max/mean)={max_com:.3f}/{mean_com:.3f} Å, {t:.1f}s")

    sumcsv = os.path.join(out_root, "sweep_summary.csv")
    with open(sumcsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote summary: {sumcsv}")

if __name__ == "__main__":
    main()
