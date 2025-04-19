from __future__ import annotations
import pandas as pd


def validate_series(df: pd.DataFrame, y_col: str = "y", ds_col: str = "ds") -> pd.DataFrame:
    """
    Ensures:
      • date column exists & is datetime
      • y is numeric and within (0, 1)
      • sorted by date
      • duplicates dropped
    Returns a *copy* so callers don’t mutate their original frame.
    """
    if ds_col not in df.columns or y_col not in df.columns:
        raise KeyError(f"Expect columns '{ds_col}' and '{y_col}'")

    out = df[[ds_col, y_col]].copy()
    out[ds_col] = pd.to_datetime(out[ds_col], utc=False)
    out = out.drop_duplicates(ds_col).sort_values(ds_col).reset_index(drop=True)

    if (out[y_col] <= 0).any() or (out[y_col] >= 1).any():
        raise ValueError("y values must lie strictly within (0, 1) for percentage KPI modelling")

    return out
