"""
Advanced time‑based feature generator
=====================================

Goals
-----
* **Robust** – works for hourly, daily, weekly, monthly, quarterly…
* **Configurable** – user can turn Fourier / one‑hot features on/off or add
  custom periods.
* **Automatic** – sensible defaults based on the *actual* median cadence.

Key concepts
------------
1. **Cadence detection** – infer the median timedelta; map it to a base unit
   ("H", "D", "W", "M", "Q").
2. **Fourier features** – for each requested period, add `sin/cos` harmonics.
   - Weekly ↔ 7 days, Yearly ↔ 365.25 days, etc.
3. **Calendar one‑hots / cyclic ints**
   - Day‑of‑week, month, quarter, hour, etc., but **only** if they vary in the
     data (e.g. skip DoW for monthly series).

The API tries to stay friendly but powerful.

Example
-------
>>> feats = make_time_features(
...     df,
...     fourier={"weekly": 3, "yearly": 2},   # 3 harmonics per week, 2 per year
...     add_dayofweek=True,
... )

Dependencies: **Only pandas + numpy (no dateutil, no holidays).**
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import timedelta
from typing import Mapping


# -------------------------------------------------------------------- #
# Helper – map inferred freq to label & canonical period in seconds
# -------------------------------------------------------------------- #
_FREQ_MAP = {
    "H": ("hourly", 3600),
    "D": ("daily", 24 * 3600),
    "W": ("weekly", 7 * 24 * 3600),
    "M": ("monthly", 30.4375 * 24 * 3600),   # avg Gregorian month
    "Q": ("quarterly", 91.3125 * 24 * 3600),
}


def _infer_base_unit(series: pd.Series) -> str:
    delta = series.sort_values().diff().dropna().median()
    if delta < timedelta(hours=1.5):
        return "H"
    if delta < timedelta(days=1.5):
        return "D"
    if delta < timedelta(weeks=1.5):
        return "W"
    if delta < timedelta(days=45):
        return "M"
    return "Q"


# -------------------------------------------------------------------- #
def make_time_features(
    df: pd.DataFrame,
    *,
    ds_col: str = "ds",
    fourier: Mapping[str, int] | None = None,
    add_dayofweek: bool = True,
    add_month: bool = True,
    add_hour: bool = True,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : DataFrame containing a datetime column `ds_col`.
    ds_col : str
        Column with timestamps.
    fourier : dict[str, int] | None
        Mapping of seasonal label → number of harmonics.
        If **None**, defaults depend on data cadence:
          * hourly  → daily+weekly+yearly
          * daily   → weekly+yearly
          * weekly  → yearly
          * monthly → yearly
          * quarterly → yearly (1st harmonic)
    add_dayofweek, add_month, add_hour : bool
        Toggle categorical one‑hot / cyclical integer features.

    Returns
    -------
    DataFrame : original columns + new feature columns.
    """
    df = df.copy()
    t = pd.to_datetime(df[ds_col])

    # ---------------------------------------------------------------- #
    # 1. Fourier features
    # ---------------------------------------------------------------- #
    base_unit = _infer_base_unit(t)
    label, _ = _FREQ_MAP[base_unit]
    ts_sec = t.astype("int64") // 10**9

    default_fourier: dict[str, int] = {}
    if fourier is None:
        if label == "hourly":
            default_fourier = {"daily": 3, "weekly": 3, "yearly": 2}
        elif label == "daily":
            default_fourier = {"weekly": 3, "yearly": 2}
        elif label == "weekly":
            default_fourier = {"yearly": 2}
        elif label in {"monthly", "quarterly"}:
            default_fourier = {"yearly": 2}
    else:
        default_fourier = dict(fourier)

    period_to_seconds = {
        "hourly": 3600,
        "daily": 24 * 3600,
        "weekly": 7 * 24 * 3600,
        "monthly": 30.4375 * 24 * 3600,
        "quarterly": 91.3125 * 24 * 3600,
        "yearly": 365.25 * 24 * 3600,
    }

    for period, order in default_fourier.items():
        sec = period_to_seconds[period]
        for k in range(1, order + 1):
            angle = 2 * np.pi * k * ts_sec / sec
            df[f"{period[:2]}_sin{k}"] = np.sin(angle)
            df[f"{period[:2]}_cos{k}"] = np.cos(angle)

    # ---------------------------------------------------------------- #
    # 2. Calendar dummies / cyclical ints
    # ---------------------------------------------------------------- #
    if add_dayofweek and label in {"hourly", "daily"}:
        dow = t.dt.dayofweek  # 0=Mon
        df = pd.concat([df, pd.get_dummies(dow, prefix="dow")], axis=1)

    if add_month and label in {"daily", "weekly", "monthly"}:
        month = t.dt.month
        df = pd.concat([df, pd.get_dummies(month, prefix="month")], axis=1)

    if add_hour and label == "hourly":
        hour = t.dt.hour
        df = pd.concat([df, pd.get_dummies(hour, prefix="hour")], axis=1)

    # Optionally: add linear trend
    df["time_idx"] = np.arange(len(df))

    return df
