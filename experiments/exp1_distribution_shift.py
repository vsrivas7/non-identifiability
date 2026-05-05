# experiments/exp1_distribution_shift.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""
Experiment 1: Distribution Similarity under Hidden Corruption

Goal:
Show that two policies (clean vs corrupted) produce nearly identical
output distributions (low TV / MMD), despite very different risks.

This empirically supports non-identifiability.
"""

import numpy as np
from seed import set_seed
from core.generator import generate_dataset, VOCAB
from core.features import bow
from core.metrics import tv_distance, mmd_rbf

# ------------------------
# Setup
# ------------------------
set_seed(42)

N = 1000
DELTA = 0.1  # corruption shift

print("Generating datasets...")

# Clean policy (low corruption)
texts_clean, z_clean = generate_dataset(N, delta=0.0)

# Corrupted policy (higher corruption)
texts_corr, z_corr = generate_dataset(N, delta=DELTA)

# ------------------------
# Feature extraction
# ------------------------
print("Computing features...")

X_clean = bow(texts_clean, VOCAB)
X_corr = bow(texts_corr, VOCAB)

# ------------------------
# Metrics
# ------------------------
print("Computing distribution distances...")

tv = tv_distance(X_clean, X_corr)
mmd = mmd_rbf(X_clean[:500], X_corr[:500], gamma=5.0)

# Risk difference (ground truth)
risk_clean = z_clean.mean()
risk_corr = z_corr.mean()
risk_gap = abs(risk_clean - risk_corr)

# ------------------------
# Output
# ------------------------
print("\n=== Results ===")
print(f"TV distance:     {tv:.4f}")
print(f"MMD distance:    {mmd:.4f}")
print(f"Risk (clean):    {risk_clean:.4f}")
print(f"Risk (corrupt):  {risk_corr:.4f}")
print(f"Risk gap:        {risk_gap:.4f}")

# ------------------------
# Save results
# ------------------------
import os
os.makedirs("results", exist_ok=True)

with open("results/exp1_distribution_shift.txt", "w") as f:
    f.write("=== Experiment 1: Distribution Shift ===\n")
    f.write(f"TV distance: {tv:.6f}\n")
    f.write(f"MMD distance: {mmd:.6f}\n")
    f.write(f"Risk clean: {risk_clean:.6f}\n")
    f.write(f"Risk corrupt: {risk_corr:.6f}\n")
    f.write(f"Risk gap: {risk_gap:.6f}\n")

print("\nSaved results to results/exp1_distribution_shift.txt")