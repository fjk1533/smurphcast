"""
Common error / accuracy metrics used across SmurphCast.

All metrics accept *array‑like* arguments and return a float.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "mae",
    "mape",
    "smape",
    "pinball_loss",
    "coverage",
]



# --------------------------------------------------------------------------- #
# Point‑forecast metric
# --------------------------------------------------------------------------- #
def mae(y_true, y_pred) -> float:
    """
    **Mean Absolute Error**

    ```text
    MAE = mean( | y_true − y_pred | )
    ```

    Works with NumPy arrays, Pandas Series, Python lists.
    """
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    return float(np.mean(np.abs(y_true - y_pred)))


# --------------------------------------------------------------------------- #
# Probabilistic metrics
# --------------------------------------------------------------------------- #
def pinball_loss(y_true, y_pred, alpha: float) -> float:
    """
    Pinball / quantile loss for a single quantile forecast.

    Parameters
    ----------
    y_true : actual values
    y_pred : predicted *alpha*‑quantile
    alpha  : quantile level between 0 and 1
    """
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    delta = y_true - y_pred
    return float(np.mean(np.maximum(alpha * delta, (alpha - 1) * delta)))


def coverage(y_true, lower, upper) -> float:
    """
    Fraction of observations that fall inside a prediction interval.

    `lower`, `upper` are the bounds for each observation.
    """
    y_true = np.asarray(y_true, dtype="float64")
    lower = np.asarray(lower, dtype="float64")
    upper = np.asarray(upper, dtype="float64")
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))

# --------------------------------------------------------------------------- #
# Extra point‑forecast metrics requested by the tests
# --------------------------------------------------------------------------- #
def mape(y_true, y_pred, eps: float = 1e-9) -> float:
    """
    **Mean Absolute Percentage Error**

    ```text
    MAPE = mean( |y_t - ŷ_t| / (|y_t| + eps) )
    ```

    The small *eps* avoids division‑by‑zero if the target includes 0.
    """
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    pct_err = np.abs(y_true - y_pred) / (np.abs(y_true) + eps)
    return float(np.mean(pct_err))


def smape(y_true, y_pred, eps: float = 1e-9) -> float:
    r"""
    **Symmetric Mean Absolute Percentage Error**

    ```text
    SMAPE = mean( 2·|y_t - ŷ_t| / (|y_t| + |ŷ_t| + eps) )
    ```
    """
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom))
