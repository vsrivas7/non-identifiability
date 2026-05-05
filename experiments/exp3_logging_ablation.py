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





texts, z = generate_dataset(1200, delta=0.1)

# logging feature (simple proxy)
log_feature = np.array(z).reshape(-1,1)

X_out = bow(texts, [f"w{i}" for i in range(2000)])

levels = [
    X_out,
    np.hstack([X_out, log_feature])
]

for i, X in enumerate(levels):
    Xtr, Xte, ytr, yte = train_test_split(X, z, test_size=0.3)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:,1]

    mse = mean_squared_error(yte, p)
    print(f"Level {i+1} MSE: {mse:.4f}")