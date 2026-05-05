import numpy as np
from scipy.spatial.distance import cdist

def tv_distance(X0, X1):
    p = X0.mean(axis=0)
    q = X1.mean(axis=0)
    return 0.5 * np.abs(p - q).sum()

def mmd_rbf(X, Y, gamma=1.0):
    Kxx = np.exp(-gamma * cdist(X, X, 'sqeuclidean'))
    Kyy = np.exp(-gamma * cdist(Y, Y, 'sqeuclidean'))
    Kxy = np.exp(-gamma * cdist(X, Y, 'sqeuclidean'))

    return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()