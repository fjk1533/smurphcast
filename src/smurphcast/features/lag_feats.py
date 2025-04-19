"""
Lag & rolling feature utilities
===============================

* add_lag_features(df, lags)
* add_rolling_features(df, windows)

If `y_col` is not present (e.g. during future prediction), functions
return the input frame unchanged.
"""

from __future__ import annotations
import pandas as pd


def add_lag_features(
    df: pd.DataFrame,
    *,
    lags: tuple[int, ...],
    y_col: str = "y_transformed",
) -> pd.DataFrame:
    out = df.copy()
    if y_col not in out.columns:
        return out                        # prediction time: leave frame untouched

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
    out = df.copy()
    if y_col not in out.columns:
        return out                        # no target column yet → skip

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
