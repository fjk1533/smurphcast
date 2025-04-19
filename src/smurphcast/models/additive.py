import pandas as pd
from sklearn.linear_model import Ridge
from ..features.time_feats import make_time_features


_EXCLUDE = {"ds", "y", "y_transformed"}      # never feed these as X


class AdditiveModel:
    """
    Fourier + DoW dummies → Ridge regression.
    Lightweight, interpretable baseline.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.reg = Ridge(alpha=self.alpha)
        self._feature_cols: list[str] | None = None

    # ------------------------------------------------------------------ #
    def _build_X(self, df: pd.DataFrame) -> pd.DataFrame:
        feats = make_time_features(df)
        cols_to_use = [c for c in feats.columns if c not in _EXCLUDE]
        return feats[cols_to_use]

    # ------------------------------------------------------------------ #
    def fit(self, df: pd.DataFrame, y_col: str = "y_transformed"):
        X = self._build_X(df)
        y = df[y_col].values
        self.reg.fit(X, y)
        self._feature_cols = list(X.columns)        # remember exact order
        return self

    # ------------------------------------------------------------------ #
    def predict(self, future_df: pd.DataFrame):
        Xf = self._build_X(future_df)[self._feature_cols]
        return self.reg.predict(Xf)
