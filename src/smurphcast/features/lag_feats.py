"""
Lag feature utilities
=====================

* `add_lag_features(df, lags)`    – plain lags (shift).
* `add_rolling_features(df, windows)` – mean, std over a rolling window.

All functions assume:
    • df has columns ["ds", "y_transformed"]   (after auto_transform)
"""

from __future__ import annotations
import pandas as pd


def add_lag_features(df: pd.DataFrame, *, lags: tuple[int, ...], y_col: str = "y_transformed") -> pd.DataFrame:
    out = df.copy()
    for L in lags:
        out[f"lag_{L}"] = out[y_col].shift(L)
    return out


def add_rolling_features(
    df: pd.DataFrame,
    *,
    windows: tuple[int, ...],
    y_col: str = "y_transformed",
    stats: tuple[str, ...] = ("mean", "std"),
) -> pd.DataFrame:
    """
    Adds rolling mean / std etc.  Example:

        add_rolling_features(df, windows=(4, 12), stats=("mean", "std"))
        → columns lag4_mean, lag4_std, lag12_mean, lag12_std …
    """
    out = df.copy()
    for w in windows:
        roll = out[y_col].rolling(window=w, min_periods=1)
        if "mean" in stats:
            out[f"roll{w}_mean"] = roll.mean()
        if "std" in stats:
            out[f"roll{w}_std"] = roll.std(ddof=0)
        if "min" in stats:
            out[f"roll{w}_min"] = roll.min()
        if "max" in stats:
            out[f"roll{w}_max"] = roll.max()
    return out
