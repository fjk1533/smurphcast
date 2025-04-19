import pandas as pd
import numpy as np


def make_time_features(df: pd.DataFrame, ds_col: str = "ds") -> pd.DataFrame:
    """
    Returns df with additional time‑based regressors:
      • sin/cos Fourier pairs for weekly & yearly
      • day‑of‑week one‑hots (Mon‑Sun)
    """
    df = df.copy()
    t = df[ds_col]

    # Fourier – weekly (period=7 days)
    week_sec = 7 * 24 * 3600
    ts_sec = t.view("int64") // 10 ** 9
    for k in (1, 2, 3):
        df[f"wk_sin{k}"] = np.sin(2 * np.pi * k * ts_sec / week_sec)
        df[f"wk_cos{k}"] = np.cos(2 * np.pi * k * ts_sec / week_sec)

    # Fourier – yearly (approx)
    yr_sec = 365.25 * 24 * 3600
    for k in (1, 2):
        df[f"yr_sin{k}"] = np.sin(2 * np.pi * k * ts_sec / yr_sec)
        df[f"yr_cos{k}"] = np.cos(2 * np.pi * k * ts_sec / yr_sec)

    # categorical DoW
    dow = t.dt.dayofweek
    dow_dummies = pd.get_dummies(dow, prefix="dow", drop_first=False)
    df = pd.concat([df, dow_dummies], axis=1)

    return df
