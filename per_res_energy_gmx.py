#!/usr/bin/env python3
"""
per_res_energy_gmx.py

A per-residue protein/peptide–PS interaction energy analysis.

Key features:
- Uses MDAnalysis atom charges directly (from .tpr)
- SIDE-CHAIN ONLY interaction energy (no backbone, no terminal artifacts)
- Coulomb + Lennard–Jones (12–6)

IMPORTANT:
This computes time-averaged nonbonded interaction energies,
NOT binding free energies.

Dependencies:
  pip install MDAnalysis numpy pandas matplotlib tqdm


Usage:
    python3 per_res_energy_gmx.py --traj *.xtc --top *.tpr --protein_sel "protein" --ps_sel "resname POPS" --ps_head_atoms "C13,O13A,O13B" --plot  --stride 1

"""

import argparse
import os
import re
import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.lib import distances
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COULOMB_CONST = 138.935456  # kJ mol^-1 nm e^-2


# ------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--traj", required=True, help="Trajectory file (xtc/trr)")
    p.add_argument("--top", required=True, help="Topology (tpr required)")
    p.add_argument("--protein_sel", default="protein",
                   help="Protein/peptide selection")
    p.add_argument("--ps_sel", default="resname POPS",
                   help="Base PS selection")
    p.add_argument("--ps_head_atoms", default=None,
                   help="Comma/space separated PS head atoms (e.g. P,O1,O2)")
    p.add_argument("--cutoff", type=float, default=1.2,
                   help="Cutoff distance (nm)")
    p.add_argument("--stride", type=int, default=1,
                   help="Frame stride")
    p.add_argument("--outdir", default="energy_data",)
    p.add_argument("--plot", action="store_true",
                   help="Generate per-residue energy barplot")
    return p.parse_args()


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def build_ps_selection(ps_sel, ps_head_atoms):
    if not ps_head_atoms:
        return ps_sel
    atoms = re.split("[,\\s]+", ps_head_atoms.strip())
    atoms = [a for a in atoms if a]
    return f"({ps_sel}) and (name {' '.join(atoms)})"


def lj_energy(r_nm, sigma_i, sigma_j, eps_i, eps_j):
    sigma = 0.5 * (sigma_i + sigma_j)
    eps = np.sqrt(max(eps_i, 0.0) * max(eps_j, 0.0))
    if sigma <= 0.0 or eps <= 0.0 or r_nm <= 1e-12:
        return 0.0
    x = sigma / r_nm
    x6 = x ** 6
    x12 = x6 * x6
    return 4.0 * eps * (x12 - x6)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    ps_selection = build_ps_selection(args.ps_sel, args.ps_head_atoms)

    # Load system
    u = mda.Universe(args.top, args.traj)

    # --- Autodetect MDAnalysis charges ---
    if not hasattr(u.atoms, "charges") or u.atoms.charges is None:
        raise RuntimeError("Charges not available. Use a .tpr topology.")
    charge_all = u.atoms.charges.copy()
    if np.allclose(charge_all, 0.0):
        raise RuntimeError("All charges are zero. Invalid topology.")

    print("INFO: Using MDAnalysis charges from topology.")

    # Selections
    protein_all = u.select_atoms(args.protein_sel)
    ps = u.select_atoms(ps_selection)

    if len(protein_all) == 0:
        raise SystemExit("Protein selection returned zero atoms.")
    if len(ps) == 0:
        raise SystemExit("PS selection returned zero atoms.")

    # ---- SIDE-CHAIN ONLY ----
    protein = protein_all.select_atoms("not backbone and not name OXT")
    if len(protein) == 0:
        raise SystemExit("Side-chain selection returned zero atoms.")

    residues = protein_all.residues
    nres = len(residues)

    resids = [r.resid for r in residues]
    resnames = [r.resname for r in residues]
    labels = [f"{r.resname}{r.resid}" for r in residues]

    # LJ parameters (disabled unless explicitly provided later)
    sigma_all = np.zeros(len(u.atoms))
    eps_all = np.zeros(len(u.atoms))

    # Atom indices
    prot_idx = protein.atoms.indices
    ps_idx = ps.atoms.indices

    prot_charges = charge_all[prot_idx]
    ps_charges = charge_all[ps_idx]

    prot_sigma = sigma_all[prot_idx]
    prot_eps = eps_all[prot_idx]
    ps_sigma = sigma_all[ps_idx]
    ps_eps = eps_all[ps_idx]

    # Map residue → side-chain atom indices
    g2l = {a.index: i for i, a in enumerate(protein.atoms)}

    res_to_atoms = []
    for res in residues:
        side_atoms = [a for a in res.atoms if a in protein.atoms]
        if side_atoms:
            res_to_atoms.append([g2l[a.index] for a in side_atoms])
        else:
            res_to_atoms.append([])  # e.g. Gly

    cutoff_A = args.cutoff * 10.0
    cutoff2 = cutoff_A * cutoff_A

    res_energy_sum = np.zeros(nres)
    res_energy_sq = np.zeros(nres)
    nframes = 0

    # --------------------------------------------------------
    # Trajectory loop
    # --------------------------------------------------------
    for ts in tqdm(u.trajectory[::args.stride], desc="Frames"):
        nframes += 1

        pp = protein.positions
        lp = ps.positions

        pair_idx, _ = distances.capped_distance(
            pp, lp, max_cutoff=cutoff_A, box=ts.dimensions
        )

        atom_energy = np.zeros(len(pp))

        if pair_idx.size:
            for p_local in np.unique(pair_idx[:, 0]):
                mask = pair_idx[:, 0] == p_local
                ps_locals = pair_idx[mask, 1]

                rij = lp[ps_locals] - pp[p_local]
                rij2 = np.sum(rij * rij, axis=1)
                sel = rij2 <= cutoff2
                if not np.any(sel):
                    continue

                r_nm = np.sqrt(rij2[sel]) * 0.1
                qi = prot_charges[p_local]
                qj = ps_charges[ps_locals[sel]]

                e_coul = COULOMB_CONST * qi * qj / r_nm

                lj_vals = np.array([
                    lj_energy(r_nm[k],
                              prot_sigma[p_local],
                              ps_sigma[ps_locals[sel][k]],
                              prot_eps[p_local],
                              ps_eps[ps_locals[sel][k]])
                    for k in range(len(r_nm))
                ])

                atom_energy[p_local] = np.sum(e_coul + lj_vals)

        for i, atom_list in enumerate(res_to_atoms):
            if atom_list:
                er = atom_energy[atom_list].sum()
            else:
                er = 0.0
            res_energy_sum[i] += er
            res_energy_sq[i] += er * er

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------
    avg_energy = res_energy_sum / nframes
    std_energy = np.sqrt(np.maximum(
        0.0, res_energy_sq / nframes - avg_energy ** 2
    ))

    df = pd.DataFrame({
        "resid": resids,
        "resname": resnames,
        "reslabel": labels,
        "avg_energy_kJmol": avg_energy,
        "std_kJmol": std_energy
    })

    out_csv = os.path.join(args.outdir, "per_residue_energy.csv")
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    if args.plot:
        x = np.arange(nres)
        plt.figure(figsize=(max(6, nres * 0.35), 4))
        plt.bar(x, avg_energy)
        plt.axhline(0, color="black", linewidth=0.8)
        plt.xticks(x, labels, rotation=90, fontsize=8)
        plt.ylabel("Avg interaction energy (kJ/mol)")
        plt.xlabel("Residue")
        plt.title("Per-residue PS interaction energy")
        plt.tight_layout()
        outfig = os.path.join(args.outdir, "energy_barplot.png")
        plt.savefig(outfig, dpi=300)
        plt.close()
        print("Saved:", outfig)

    print("Done. Frames processed:", nframes)


if __name__ == "__main__":
    main()
