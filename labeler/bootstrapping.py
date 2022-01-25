import numpy as np


# Bootstrapping a 4D ndarray (n, c, h, w) by the first dim
def bootstrap(X: np.ndarray, size=30):
    n = X.shape[0]
    if size >= n:
        return X
    else:
        idx = np.random.choice(a=np.arange(n), size=size)
        return X[idx]


# Bootstrapping a 4D ndarray data and a 2D ndarray label
def bootstrap_xy(X: np.ndarray, Y: np.ndarray, size=30):
    if X.shape[0] != Y.shape[0]:
        raise ValueError
    n = X.shape[0]
    if size >= n:
        return X, Y
    else:
        idx = np.random.choice(a=np.arange(n), size=size)
        return X[idx], Y[idx]


# Bootstrapping on each class; guarantee that every class is selected and final numbers are balanced
def bootstrap_xy_balanced_class(X: np.ndarray, Y: np.ndarray, num_class=10, size_per_class=3):
    if X.shape[0] != Y.shape[0]:
        raise ValueError
    X_boot = []
    Y_boot = []
    for c in range(num_class):
        X_c = X[Y == c]
        Y_c = Y[Y == c]
        if X_c.shape[0] != Y_c.shape[0]:
            raise ValueError
        n = X_c.shape[0]
        if size_per_class >= n:
            X_boot.append(X_c)
            Y_boot.append(Y_c)
        else:
            idx = np.random.choice(a=np.arange(n), size=size_per_class)
            X_boot.append(X_c[idx])
            Y_boot.append(Y_c[idx])
    X_boot = np.concatenate(X_boot, axis=0)
    Y_boot = np.concatenate(Y_boot, axis=0)
    return X_boot, Y_boot