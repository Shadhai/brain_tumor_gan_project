import numpy as np
from config import PITUITARY_IDX, REAL_PITUITARY

def create_imbalance(X, y):
    X_list, y_list = [], []
    for cls in np.unique(y):
        mask = y == cls
        X_cls = X[mask]
        y_cls = y[mask]
        if cls == PITUITARY_IDX:
            X_cls = X_cls[:REAL_PITUITARY]
            y_cls = y_cls[:REAL_PITUITARY]
        X_list.append(X_cls)
        y_list.append(y_cls)
    X_new = np.concatenate(X_list, axis=0)
    y_new = np.concatenate(y_list, axis=0)
    idx = np.random.permutation(len(X_new))
    return X_new[idx], y_new[idx]