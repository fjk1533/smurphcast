"""
AdditiveModel
=============

Fourier + calendar dummies → Ridge regression.
Lightweight, fully CPU‑friendly baseline.
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import Ridge

from ..features.time_feats import make_time_features

_EXCLUDE = {"ds", "y", "y_transformed"}


class AdditiveModel:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.reg = Ridge(alpha=self.alpha)
        self._feature_cols: list[str] | None = None

    # ------------------------------------------------------------------ #
    def _build_X(self, df: pd.DataFrame) -> pd.DataFrame:
        feats = make_time_features(df)
        cols = [c for c in feats.columns if c not in _EXCLUDE]
        return feats[cols]

    # ------------------------------------------------------------------ #
    def fit(self, df: pd.DataFrame, y_col: str = "y_transformed"):
        X = self._build_X(df)
        y = df[y_col].values
        self.reg.fit(X, y)
        self._feature_cols = list(X.columns)     # remember training schema
        return self

    # ------------------------------------------------------------------ #
    def predict(self, future_df: pd.DataFrame):
        if self._feature_cols is None:
            raise RuntimeError("Call .fit() before .predict().")

        Xf = self._build_X(future_df)
        # ensure same columns as during fit; fill unseen dummies with 0
        Xf = Xf.reindex(columns=self._feature_cols, fill_value=0)
        return self.reg.predict(Xf)
