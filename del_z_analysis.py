# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# © 2026 Surya Pratap Singh | suryabioinfo-at-gmail.com (-at- means @)
# Authored and developed by Surya Pratap Singh, PhD.
# This script comes with no warranty. You can use it freely.
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


import numpy as np
import pandas as pd
import glob
import os


# --------------------------------------------------------------------------------------------------------------------------------------------
#
# Delta-Z Analysis for Peptide–Membrane Systems
# This Python script performs batch Δz analysis between peptide COM and lipid headgroup COM from GROMACS .xvg files.
# It quantifies membrane penetration metrics (interface/core fractions, crossings) and classifies peptides as CPP-like or non-CPP.
#Results are automatically summarized into a CSV file for downstream analysis.
#
# --------------------------------------------------------------------------------------------------------------------------------------------

# ===================================================================
# Working directory (Change appropriately)
# ===================================================================
BASE_DIR = "peptides"        # folder containing peptide subfolders
OUT_CSV = "delta_z_summary.csv"

PHOSPHATE_WIDTH = 0.35       # nm (headgroup region)
CORE_THRESHOLD = 1.0         # nm (hydrophobic core entry)

# =================================================================================
# Load .xvg files 
# Use "gmx mindist" to compute these xvg files appropriately. 
# =================================================================================
def load_xvg(fname):
    t, z = [], []
    with open(fname) as f:
        for line in f:
            if line.startswith(('#', '@')):
                continue
            cols = line.split()
            t.append(float(cols[0]) / 1000.0)  # Converts time units ps to ns.
            z.append(float(cols[3]))           # The z-coordinate
    return np.array(t), np.array(z)

# =========================
# Analysis of each Peptide
# =========================
results = []

pep_dirs = sorted(glob.glob(os.path.join(BASE_DIR, "*")))

for pep_dir in pep_dirs:
    pep_name = os.path.basename(pep_dir)
    pep_file = os.path.join(pep_dir, "pep_com.xvg")
    po4_file = os.path.join(pep_dir, "hg_com.xvg")

    if not (os.path.isfile(pep_file) and os.path.isfile(po4_file)):
        print(f"Skipping {pep_name}: missing XVG files")
        continue

    t_pep, z_pep = load_xvg(pep_file)
    t_po4, z_po4 = load_xvg(po4_file)

    if len(z_pep) != len(z_po4):
        print(f"Skipping {pep_name}: frame mismatch")
        continue

    delta_z = z_pep - z_po4

    # Metrics
    max_abs_dz = np.max(np.abs(delta_z))
    frac_interface = np.mean(np.abs(delta_z) > PHOSPHATE_WIDTH)
    frac_core = np.mean(np.abs(delta_z) > CORE_THRESHOLD)

    # Crossing count
    sign = np.sign(delta_z)
    crossings = np.sum(np.diff(sign) != 0)

    # Classification
    if frac_core > 0.01:
        cpp_class = "CPP-like"
    else:
        cpp_class = "Non-CPP"

    results.append([
        pep_name,
        max_abs_dz,
        frac_interface,
        frac_core,
        crossings,
        cpp_class
    ])

# =============================================================================
# Final output writing (.csv)
# =============================================================================

df = pd.DataFrame(
    results,
    columns=[
        "Peptide",
        "Max_abs_DeltaZ_nm",
        "Fraction_Interface",
        "Fraction_Core",
        "Crossing_Count",
        "CPP_Classification"
    ]
)

df.to_csv(BASE_DIR + "/" + OUT_CSV, index=False)

#print(f"\n delz batch analysis complete")
print(f" Results saved to {OUT_CSV}\n")


