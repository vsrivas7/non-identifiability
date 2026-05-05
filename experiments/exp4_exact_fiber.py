
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# identical outputs
texts = ["constant output"] * 500
z = np.array([0]*250 + [1]*250)

# output-only features
X = np.array([[0]] * 500)

clf = LogisticRegression().fit(X, z)
p = clf.predict_proba(X)[:,1]

print("Exact Fiber AUROC:", roc_auc_score(z, p))