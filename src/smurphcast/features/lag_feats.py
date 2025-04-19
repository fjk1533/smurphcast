import pandas as pd


def add_lag_features(df: pd.DataFrame, lags=(1, 7), y_col="y_transformed") -> pd.DataFrame:
    out = df.copy()
    for L in lags:
        out[f"lag_{L}"] = out[y_col].shift(L)
    return out
