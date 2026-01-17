**About del_z_analysis.py**

*The Delta-Z Analysis for Peptide–Membrane Systems.*
This Python script performs batch Δz analysis between peptide COM and lipid headgroup COM from GROMACS .xvg files.
It quantifies membrane penetration metrics (interface/core fractions, crossings) and classifies peptides as CPP-like or non-CPP.
Results are automatically summarized into a CSV file for downstream analysis.



**About tauoff_analyzer.py**

*The tauoff Contact Lifetime Analysis for Peptide–Membrane Interactions*
This script computes tauoff-based binding kinetics from GROMACS contact .xvg files using event-duration analysis and bootstrapped confidence intervals.
It distinguishes reversible versus anchored contacts, generates per-peptide and global tauoff plots, and summarizes binding metrics in a CSV for comparative studies.






**Per-Residue Contact Frequency Analysis**

This script computes the per-residue contact frequency between a peptide/protein and membrane lipids (e.g., POPS) from GROMACS MD trajectories using MDAnalysis. Contacts are defined by a user-specified distance cutoff and summarized across the trajectory.


*Key features:*
Residue-level contact detection
Lipid headgroup–specific selections supported
CSV output and bar plot visualization.



**Per-Residue Protein–Membrane Interaction Energy Analysis**

This script calculates time-averaged per-residue nonbonded interaction energies (Coulomb + Lennard–Jones) between a peptide/protein and membrane lipids using GROMACS trajectories. Only side-chain atoms are considered to avoid backbone and terminal artifacts.


**Important**
*This method reports interaction energies, not binding free energies.*


*Key features:*
Uses atomic charges directly from GROMACS .tpr topology
Side-chain–only residue energy decomposition
Mean and standard deviation per residue
Energy bar plot

