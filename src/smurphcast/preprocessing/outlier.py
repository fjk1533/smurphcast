"""
Outlier detection / treatment for bounded‑percentage series
===========================================================

Keeps things **simple, robust, and lightweight**:

* default uses **rolling MAD** (median‑absolute‑deviation) because it is
  resistant to heavy‑tailed noise and does not rely on the mean.
* fallbacks: global MAD or IQR.
* two treatment modes:
    • "mask"  – just add a boolean `is_outlier` column (non‑destructive)
    • "winsor" – replace extreme values with nearest inlier
                 (soft clamp, preserves monotonicity of ranks)

The code never *drops* rows, so the time index stays aligned.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _mad(x: np.ndarray) -> float:
    """Median absolute deviation (scaled to ~σ for normal data)."""
    return np.median(np.abs(x - np.median(x))) * 1.4826


@dataclass
class OutlierHandler:
    y_col: str = "y"
    method: str = "rolling_mad"              # "rolling_mad" | "mad" | "iqr"
    window: int = 7                          # for rolling MAD
    thresh: float = 6.0                      # number of MADs / IQR‑units
    treat: str = "mask"                      # "mask" | "winsor"

    # -------------------------------------------------------------- #
    def detect(self, df: pd.DataFrame) -> pd.Series:
        """Return boolean mask of outliers."""
        s = df[self.y_col].astype(float)

        if self.method == "rolling_mad":
            med = s.rolling(self.window, center=True, min_periods=1).median()
            mad = (
                s.rolling(self.window, center=True, min_periods=1)
                .apply(_mad, raw=True)
                .replace(0, np.nan)
                .bfill()
                .ffill()
            )
            z = np.abs(s - med) / (mad + 1e-9)
            return z > self.thresh

        if self.method == "mad":
            med = np.median(s)
            mad = _mad(s) or 1e-9
            return np.abs(s - med) / mad > self.thresh

        if self.method == "iqr":
            q1, q3 = np.percentile(s, [25, 75])
            iqr = q3 - q1 or 1e-9
            lower, upper = q1 - self.thresh * iqr, q3 + self.thresh * iqr
            return (s < lower) | (s > upper)

        raise ValueError(f"Unknown method: {self.method}")

    # -------------------------------------------------------------- #
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a copy of *df* with:
          • `is_outlier` boolean column
          • (optional) `y_clean` treated series when treat == "winsor"
        """
        out = df.copy()
        mask = self.detect(out)
        out["is_outlier"] = mask

        if self.treat == "mask":
            return out

        if self.treat == "winsor":
            s = out[self.y_col]
            # nearest inlier clamp
            lower = s[~mask].min()
            upper = s[~mask].max()
            out["y_clean"] = s.clip(lower, upper)
            return out

        raise ValueError(f"Unknown treat mode: {self.treat}")
