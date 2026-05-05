import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from seed import set_seed
set_seed()

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression

from core.generator import generate_dataset
from core.features import bow

import matplotlib.pyplot as plt

# ------------------------
# Generate data
# ------------------------
texts, z = generate_dataset(1200, delta=0.1)

# Logging feature (perfect proxy here)
log_feature = np.array(z).reshape(-1,1)

# Output-only features
X_out = bow(texts, [f"w{i}" for i in range(2000)])

# Two levels
levels = [
    X_out,
    np.hstack([X_out, log_feature])
]

mse_values = []

# ------------------------
# Run experiments
# ------------------------
for i, X in enumerate(levels):
    Xtr, Xte, ytr, yte = train_test_split(X, z, test_size=0.3)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:,1]

    mse = mean_squared_error(yte, p)
    mse_values.append(mse)

    print(f"Level {i+1} MSE: {mse:.4f}")

# ------------------------
# Save results (FIXED POSITION)
# ------------------------
os.makedirs("results", exist_ok=True)

with open("results/exp3_logging_ablation.txt", "w") as f:
    f.write(f"Output-only MSE: {mse_values[0]:.6f}\n")
    f.write(f"With logging MSE: {mse_values[1]:.6f}\n")

print("Saved results to results/exp3_logging_ablation.txt")

# ------------------------
# Save plot
# ------------------------
os.makedirs("plots", exist_ok=True)

plt.figure(figsize=(6,4))
plt.bar(["Output-only", "With Logging"], mse_values)
plt.title("Logging Improves Estimation (Exp3)")
plt.ylabel("MSE")
plt.tight_layout()
plt.savefig("plots/exp3_logging_ablation.png")
plt.close()