#!/usr/bin/env python3


"""
contact_frequency.py

Key features:
IMPORTANT:
This script computes contact frequency of peptide to the membrane.

Dependencies:
  pip install MDAnalysis numpy pandas matplotlib tqdm


Usage:
    python contact_frequency.py --traj *.xtc --top *.tpr --ps_sel "resname POPS" --ps_head_atoms "C13 O13A O13B" --protein_sel "protein" --plot --stride 1

"""


import argparse, os, re
import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.lib import distances
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--traj', required=True)
    p.add_argument('--top', required=True)
    p.add_argument('--protein_sel', default='protein')
    p.add_argument('--ps_sel', default='resname POPS')
    p.add_argument('--ps_head_atoms', default=None)
    p.add_argument('--cutoff', type=float, default=0.45, help='nm')
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--outdir', default='contact_data')
    p.add_argument('--plot', action='store_true')
    return p.parse_args()


def build_ps_selection(ps_sel, ps_head_atoms):
    if not ps_head_atoms:
        return ps_sel
    atoms = re.split('[,\\s]+', ps_head_atoms.strip())
    atoms = [a for a in atoms if a]
    return f"({ps_sel}) and (name {' '.join(atoms)})"


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    ps_selection = build_ps_selection(args.ps_sel, args.ps_head_atoms)

    u = mda.Universe(args.top, args.traj)
    prot = u.select_atoms(args.protein_sel)
    ps = u.select_atoms(ps_selection)

    if len(prot.residues) == 0:
        raise SystemExit("Protein selection empty")
    if len(ps) == 0:
        raise SystemExit("PS selection empty")

    residues = prot.residues
    labels = [f"{r.resname}{r.resid}" for r in residues]
    resids = [r.resid for r in residues]

    nres = len(residues)
    nframes = int(np.ceil(len(u.trajectory) / args.stride))
    contact_bool = np.zeros((nres, nframes), dtype=bool)

    cutoff_A = args.cutoff * 10.0

    # global → local atom mapping
    g2l = {a.index: i for i, a in enumerate(prot.atoms)}
    res_to_atoms = [[g2l[a.index] for a in r.atoms] for r in residues]

    fidx = 0
    for ts in tqdm(u.trajectory[::args.stride], desc="Frames"):
        pair_idx, _ = distances.capped_distance(
            prot.positions, ps.positions,
            max_cutoff=cutoff_A, box=ts.dimensions
        )
        if pair_idx.size:
            contacting_atoms = set(pair_idx[:, 0])
            for i, atom_idxs in enumerate(res_to_atoms):
                if any(a in contacting_atoms for a in atom_idxs):
                    contact_bool[i, fidx] = True
        fidx += 1

    freq = contact_bool.sum(axis=1) / float(nframes)

    df = pd.DataFrame({
        "resid": resids,
        "resname": [r.resname for r in residues],
        "reslabel": labels,
        "contact_frequency": freq
    })

    csv_path = os.path.join(args.outdir, "per_residue_contact_frequency.csv")
    df.to_csv(csv_path, index=False)
    np.save(os.path.join(args.outdir, "contacts_bool.npy"), contact_bool)
    print("Saved:", csv_path)

    if args.plot:
        x = np.arange(nres)
        plt.figure(figsize=(max(6, nres * 0.35), 4))
        plt.bar(x, freq)
        plt.xticks(x, labels, rotation=90, fontsize=8)
        plt.ylabel("Contact frequency")
        plt.xlabel("Residue")
        plt.title("Per-residue PS contact frequency")
        plt.tight_layout()
        out = os.path.join(args.outdir, "contact_frequency_bar.png")
        plt.savefig(out, dpi=300)
        plt.close()
        print("Saved:", out)


if __name__ == "__main__":
    main()
