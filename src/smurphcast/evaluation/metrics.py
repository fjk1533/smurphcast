import numpy as np


def mape(y_true, y_pred, eps=1e-6):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps)))


def smape(y_true, y_pred, eps=1e-6):
    nom = np.abs(y_true - y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2 + eps
    return np.mean(nom / denom)


def coverage(y_true, lower, upper):
    """Interval coverage: share of true vals inside [lower, upper]."""
    inside = (y_true >= lower) & (y_true <= upper)
    return inside.mean()
