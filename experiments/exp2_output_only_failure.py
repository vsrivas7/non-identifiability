import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import train_test_split

from core.generator import generate_dataset
from core.features import bow
from core.estimators import train_logreg, predict
from seed import set_seed


set_seed()

# Generate
texts, z = generate_dataset(1200, delta=0.1)

# Features
from core.generator import VOCAB
X = bow(texts, VOCAB)

# Split
Xtr, Xte, ytr, yte = train_test_split(X, z, test_size=0.3, random_state=42)

# Train
clf = train_logreg(Xtr, ytr)
p = predict(clf, Xte)

# Metrics
auc = roc_auc_score(yte, p)
mse = mean_squared_error(yte, p)

print(f"Output-only AUROC: {auc:.3f}")
print(f"Output-only MSE:   {mse:.4f}")