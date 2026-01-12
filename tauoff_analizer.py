import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import csv

# =========================
# USER PARAMETERS
# =========================
DATA_DIR = "data/linear"
RESULT_DIR = "tau_results"
PLOT_DIR = "tau_plots"
SEQ_FILE = os.path.join(DATA_DIR, "linear.txt")

R_CUT = 0.35      # nm (Lys/Arg–POPS contact)
N_BOOT = 1000
CONF = 95

# =========================
# PREPARE TAU GRID
# =========================
# Only 200 ns
#
TAU_GRID = np.unique(np.concatenate([
    np.arange(2, 100, 2),
    np.arange(100, 201, 10)
]))

# Up to 1000 ns
# Uncomment below to use full 1000 ns grid
#

#TAU_GRID = np.unique(np.concatenate([
#    np.arange(2, 100, 2),
#    np.arange(100, 500, 10),
#    np.arange(500, 1001, 25)
#]))
PLATEAU_SLOPE = 0.02

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# =========================
# SEQUENCE & CHARGE
# =========================
CHARGE_MAP = {'K': 1, 'R': 1, 'D': -1, 'E': -1}

def load_sequences(fname):
    seqs = {}
    with open(fname) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                p, s = line.split()
                seqs[p] = s.upper()
    return seqs

def net_charge(seq):
    return sum(CHARGE_MAP.get(r, 0) for r in seq)

PEPTIDE_SEQS = load_sequences(SEQ_FILE)
charges = {p: net_charge(s) for p, s in PEPTIDE_SEQS.items()}

# =========================
# VALIDATION
# =========================
xvg_peptides = [
    os.path.basename(f).replace(".xvg", "")
    for f in glob.glob(f"{DATA_DIR}/*.xvg")
]

missing = [p for p in xvg_peptides if p not in PEPTIDE_SEQS]
if missing:
    raise ValueError("Missing sequences for:\n" + "\n".join(missing))

# =========================
# HELPERS
# =========================
def load_xvg(fname):
    t, d = [], []
    with open(fname) as f:
        for line in f:
            if line.startswith(('#', '@')):
                continue
            a, b = line.split()
            t.append(float(a) / 1000.0)
            d.append(float(b))
    return np.array(t), np.array(d)

def extract_events(t, contact):
    events, start = [], None
    for i in range(len(contact)):
        if contact[i] and start is None:
            start = t[i]
        elif not contact[i] and start is not None:
            events.append(t[i] - start)
            start = None
    if start is not None:
        events.append(t[-1] - start)
    return np.array(events)

def detect_tau_off(tau, ratio):
    slope = np.abs(np.gradient(ratio, tau))
    for t, s in zip(tau, slope):
        if s < PLATEAU_SLOPE:
            return t
    return tau[-1]

def bootstrap_ci(events, total_time, tau_off):
    if len(events) < 3:
        return np.nan, np.nan
    vals = []
    for _ in range(N_BOOT):
        res = np.random.choice(events, len(events), replace=True)
        rev = res[res < tau_off]
        vals.append(rev.sum() / total_time * 100)
    low = np.percentile(vals, (100 - CONF) / 2)
    high = np.percentile(vals, 100 - (100 - CONF) / 2)
    return low, high

# =========================
# ANALYSIS
# =========================
summary = []
all_curves = {}

# Generate unique colors (one per peptide)
color_map = plt.colormaps.get_cmap("tab20").resampled(len(xvg_peptides))
colors = {pep: color_map(i) for i, pep in enumerate(xvg_peptides)}

for fname in sorted(glob.glob(f"{DATA_DIR}/*.xvg")):
    pep = os.path.basename(fname).replace(".xvg", "")
    print(f"Processing {pep}")

    t, d = load_xvg(fname)
    contact = d < R_CUT
    events = extract_events(t, contact)
    total_time = t[-1] - t[0]

    ratios = []
    for tau in TAU_GRID:
        rev = events[events < tau]
        ratios.append(rev.sum() / total_time * 100)

    ratios = np.array(ratios)
    all_curves[pep] = ratios

    tau_off = detect_tau_off(TAU_GRID, ratios)

    rev = events[events < tau_off]
    anch = events[events >= tau_off]

    bind_ratio = rev.sum() / total_time * 100
    mean_tau = rev.mean() if len(rev) else 0
    anch_frac = len(anch) / len(events) if len(events) else 0
    ci_low, ci_high = bootstrap_ci(events, total_time, tau_off)

    summary.append([
        pep, PEPTIDE_SEQS[pep], charges[pep],
        tau_off, bind_ratio, ci_low, ci_high,
        mean_tau, anch_frac, len(events)
    ])

    # Per-peptide plot
    plt.figure()
    plt.plot(TAU_GRID, ratios, lw=2, color=colors[pep], label=pep)
    plt.axvline(tau_off, color='k', ls='--')
    plt.xlabel("τoff (ns)")
    plt.ylabel("Contact binding ratio (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{pep}_binding_ratio_vs_tauoff.png", dpi=300)
    plt.close()

# =========================
# GLOBAL PLOT 
# =========================
plt.figure(figsize=(15, 8))
for pep, ratios in all_curves.items():
    plt.plot(TAU_GRID, ratios, lw=1.8, color=colors[pep], label=pep)

plt.xlabel("τoff (ns)")
plt.ylabel("Contact binding ratio (%)")
plt.legend(
    fontsize=7,
    ncol=2,
    loc="upper right",
    bbox_to_anchor=(1.15, 1.0),
    frameon=False
)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/ALL_peptides_binding_ratio_vs_tauoff.png", dpi=300)
plt.close()

# =========================
# WRITE CSV
# =========================
with open(f"{RESULT_DIR}/contact_summary.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Peptide", "Sequence", "Net_charge",
        "Tau_off_ns", "Binding_ratio_percent",
        "CI_low", "CI_high",
        "Mean_reversible_tau_ns",
        "Anchoring_fraction",
        "Total_events"
    ])
    writer.writerows(summary)

print(" Tauoff analyses complete.")
