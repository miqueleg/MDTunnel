#!/usr/bin/env python3
from __future__ import annotations
import os, time, subprocess, itertools, csv

PROTEIN = "LinB_WT.pdb"
LIGAND = "DBE.pdb"
SPHERES = "CW_tunnel.dsd"

def run(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def metrics(outdir):
    import pandas as pd, numpy as np
    ep = pd.read_csv(os.path.join(outdir, 'energy_profile.csv'))
    E = (ep['E_kcal_mol'] if 'E_kcal_mol' in ep.columns else ep['E_kJ_mol']/4.184).to_numpy()
    de = abs(np.diff(E)) if len(E)>1 else [0.0]
    com = pd.read_csv(os.path.join(outdir, 'com_to_sphere.csv'))
    a = com['com_to_sphere_A'].to_numpy() if 'com_to_sphere_A' in com.columns else []
    return float(max(de)), float(sum(de)/len(de) if len(de) else 0.0), float(max(a) if len(a) else 0.0), float(sum(a)/len(a) if len(a) else 0.0)

def main():
    kcom = [1000]
    ramp = [2.0]
    comt = [0.9, 1.2]
    kpos_bb = [2000, 1000]
    kpos_sc = [1000, 500]
    marginA = [1.5, 2.5]
    scale = [0.1, 0.2]
    tol = [0.005]
    iters = [30000]
    combos = list(itertools.product(kcom, ramp, comt, kpos_bb, kpos_sc, marginA, scale, tol, iters))
    root = 'runs_pocket'
    os.makedirs(root, exist_ok=True)
    rows=[]
    for i,(kc,kr,ct,bb,sc,mg,sl,tl,it) in enumerate(combos, start=1):
        od = os.path.join(root, f"s{i:02d}_kc{kc}_kr{kr}_ct{ct}_bb{bb}_sc{sc}_mg{mg}_sl{sl}")
        cmd = [
            'conda','run','-n','OpenMM','python','-m','cavermm.cli',
            '--protein',PROTEIN,'--ligand',LIGAND,'--spheres',SPHERES,'--out',od,
            '--ambertools-prep','--ligand-charge','0','--ligand-resname','LIG',
            '--direction','exit',
            '--kcom',str(kc),'--kpos-protein','3000',
            '--kpos-backbone',str(bb),'--kpos-sidechain',str(sc),
            '--kcom-ramp-scale',str(kr),'--com-target-A',str(ct),'--com-max-k','2000000','--com-scale','5',
            '--relax-pocket-margin-A',str(mg),'--relax-pocket-scale',str(sl),
            '--min-tol-kj',str(tl),'--min-max-iter',str(it)
        ]
        t0=time.time(); p=run(cmd); dt=time.time()-t0
        status='ok' if p.returncode==0 else f'err:{p.returncode}'
        try:
            mj,mn,mc,mac = metrics(od)
        except Exception as e:
            mj=mn=mc=mac=float('nan'); status+=f' parse:{e}'
        rows.append({'outdir':od,'kcom':kc,'ramp':kr,'comtA':ct,'kpos_bb':bb,'kpos_sc':sc,'marginA':mg,'scale':sl,'max_jump_kcal':round(mj,3),'mean_jump_kcal':round(mn,3),'max_com_A':round(mc,3),'mean_com_A':round(mac,3),'runtime_s':round(dt,1),'status':status})
        print(f"[{i}/{len(combos)}] {status} jumps(max/mean)={mj:.2f}/{mn:.2f} kcal, COM(max/mean)={mc:.2f}/{mac:.2f} Å")
    with open(os.path.join(root,'summary.csv'),'w',newline='') as f:
        import csv
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print('Wrote', os.path.join(root,'summary.csv'))

if __name__=='__main__':
    main()

