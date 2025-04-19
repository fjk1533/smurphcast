from __future__ import annotations
import pandas as pd
import lightgbm as lgb
from ..features.time_feats import make_time_features
from ..features.lag_feats import add_lag_features, add_rolling_features

_EXCLUDE = {"ds", "y", "y_transformed"}


class QuantileGBMModel:
    """
    Three LightGBM models (lower / median / upper) for prediction intervals.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.2,
        num_leaves: int = 31,
        n_estimators: int = 300,
        lags: tuple[int, ...] = (1, 7, 14),
        rolls: tuple[int, ...] = (4, 12),
    ):
        self.alpha = alpha
        self.params = dict(
            metric="quantile",
            learning_rate=0.05,
            num_leaves=num_leaves,
            verbosity=-1,
        )
        self.n_estimators = n_estimators
        self.lags = lags
        self.rolls = rolls
        self.models: dict[str, lgb.Booster] = {}
        self._feature_cols: list[str] | None = None

    # --------------------------------------------- #
    def _build_X(self, df: pd.DataFrame) -> pd.DataFrame:
        feats = make_time_features(df)
        feats = add_lag_features(feats, lags=self.lags)
        feats = add_rolling_features(feats, windows=self.rolls)
        cols = [c for c in feats.columns if c not in _EXCLUDE]
        return feats[cols]

    # --------------------------------------------- #
    def fit(self, df: pd.DataFrame, y_col: str = "y_transformed"):
        X = self._build_X(df).dropna()
        y = df.loc[X.index, y_col]
        self._feature_cols = list(X.columns)

        quantiles = {
            "lower": self.alpha / 2,
            "median": 0.5,
            "upper": 1 - self.alpha / 2,
        }
        for name, q in quantiles.items():
            params = {**self.params, "objective": "quantile", "alpha": q}
            dset = lgb.Dataset(X, y)
            self.models[name] = lgb.train(params, dset, num_boost_round=self.n_estimators)
        return self

    # --------------------------------------------- #
    def _prep_future(self, future_df: pd.DataFrame) -> pd.DataFrame:
        Xf = self._build_X(future_df)
        Xf = Xf.reindex(columns=self._feature_cols, fill_value=0)
        return Xf.ffill().bfill()

    # --------------------------------------------- #
    def predict(self, future_df: pd.DataFrame):
        return self.models["median"].predict(self._prep_future(future_df))

    def predict_interval(self, future_df: pd.DataFrame) -> pd.DataFrame:
        Xf = self._prep_future(future_df)
        preds = {name: mdl.predict(Xf) for name, mdl in self.models.items()}
        return pd.DataFrame(preds, index=future_df["ds"])
