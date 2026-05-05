import numpy as np
from collections import Counter

def bow(texts, vocab):
    idx = {w:i for i,w in enumerate(vocab)}
    X = np.zeros((len(texts), len(vocab)))

    for i, t in enumerate(texts):
        counts = Counter(t.split())
        for w, c in counts.items():
            if w in idx:
                X[i, idx[w]] = c

    X /= (X.sum(axis=1, keepdims=True) + 1e-8)
    return X