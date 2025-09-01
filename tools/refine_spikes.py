#!/usr/bin/env python3
from __future__ import annotations
import os, argparse
import numpy as np
import pandas as pd

def load_ligand_xyz(pdb_path: str) -> np.ndarray:
    xs=[]
    with open(pdb_path) as f:
        for line in f:
            if (line.startswith('ATOM') or line.startswith('HETATM')) and line[17:20].strip()=='LIG':
                try:
                    x=float(line[30:38]); y=float(line[38:46]); z=float(line[46:54])
                except Exception:
                    parts=line.split()
                    x=float(parts[6]); y=float(parts[7]); z=float(parts[8])
                xs.append([x,y,z])
    return np.array(xs, dtype=float)

def kabsch(P: np.ndarray, Q: np.ndarray) -> float:
    # Compute RMSD after optimal superposition of P onto Q
    if P.shape != Q.shape or P.size==0:
        return np.inf
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    C = Pc.T @ Qc
    V,S,Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V@Wt))
    D = np.diag([1,1,d])
    U = V @ D @ Wt
    P2 = Pc @ U
    diff = P2 - Qc
    return float(np.sqrt((diff*diff).sum()/P.shape[0]))

def detect_spikes(E: np.ndarray, delta=10.0, eps=5.0, max_window=50) -> list[tuple[int,int]]:
    """Detect spike segments using cumulative rise-and-return logic.

    A spike starts at s if cumulative rise exceeds delta and later returns within eps of the start level.
    This is robust to broad peaks and plateaus and works near the trajectory end.
    """
    spikes=[]; n=len(E)
    i=0
    while i < n-2:
        base = E[i]
        cum = 0.0
        peak = 0.0
        end = None
        kmax = min(n-1, i+max_window)
        for k in range(i, kmax):
            if k+1 >= n: break
            step = E[k+1]-E[k]
            cum += step
            peak = max(peak, cum)
            # when we have risen sufficiently and returned near baseline, mark end
            if peak >= delta and abs((E[k+1]-base)) <= eps:
                end = k+1
                break
        if end is not None and end > i+1:
            spikes.append((i, end))
            i = end + 1
        else:
            i += 1
    return spikes

def refine(outdir: str, rmsd_thresh: float, delta: float, eps: float):
    enter_dir = os.path.join(outdir,'enter')
    exit_dir = os.path.join(outdir,'exit')
    if not (os.path.isdir(enter_dir) and os.path.isdir(exit_dir)):
        raise SystemExit('Expected enter/ and exit/ subdirs under '+outdir)
    ex = pd.read_csv(os.path.join(exit_dir,'energy_profile.csv'))
    en = pd.read_csv(os.path.join(enter_dir,'energy_profile.csv'))
    n = min(len(ex), len(en))
    E = (ex['E_kcal_mol'] if 'E_kcal_mol' in ex.columns else ex['E_kJ_mol']/4.184).to_numpy()
    spikes = detect_spikes(E[:n], delta=delta, eps=eps)
    frames_out = []
    for j in range(n):
        frames_out.append(os.path.join(exit_dir,'frames',f'step_{j:04d}.pdb'))
    # refine each spike region by replacing with enter frames if continuity ok
    for (s0,s2) in spikes:
        for j in range(s0+1, s2):
            en_idx = n-1-j
            f_en = os.path.join(enter_dir,'frames',f'step_{en_idx:04d}.pdb')
            f_prev = frames_out[j-1]
            # compare RMSD of enter vs exit wrt previous frame for continuity
            prev = load_ligand_xyz(f_prev)
            xyz_en = load_ligand_xyz(f_en)
            xyz_ex = load_ligand_xyz(frames_out[j])
            rmsd_en = kabsch(xyz_en, prev)
            rmsd_ex = kabsch(xyz_ex, prev)
            if rmsd_en < rmsd_ex and rmsd_en <= rmsd_thresh:
                frames_out[j] = f_en
    # Write merged frames and trajectory
    frames_dir = os.path.join(outdir,'frames_refined')
    os.makedirs(frames_dir, exist_ok=True)
    import shutil
    for j,f in enumerate(frames_out):
        shutil.copyfile(f, os.path.join(frames_dir,f'step_{j:04d}.pdb'))
    # Write ligand-only trajectory
    traj = os.path.join(outdir,'substrate_traj_refined.pdb')
    with open(traj,'w') as t:
        for j,f in enumerate(frames_out, start=1):
            t.write(f'MODEL     {j}\n')
            with open(f) as fh:
                for line in fh:
                    if (line.startswith('ATOM') or line.startswith('HETATM')) and line[17:20].strip()=='LIG':
                        t.write(line)
            t.write('TER\nENDMDL\n')
        t.write('END\n')
    print('Refinement done. Spikes:', spikes)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--out', required=True, help='Directory with enter/ and exit/ subfolders')
    ap.add_argument('--rmsd-thresh', type=float, default=2.0, help='Max RMSD (Å) allowed to accept enter frame for continuity')
    ap.add_argument('--spike-delta-kcal', type=float, default=10.0, help='Spike detection rise threshold (kcal/mol)')
    ap.add_argument('--return-eps-kcal', type=float, default=5.0, help='Return-to-baseline epsilon (kcal/mol)')
    args=ap.parse_args()
    refine(args.out, args.rmsd_thresh, args.spike_delta_kcal, args.return_eps_kcal)
