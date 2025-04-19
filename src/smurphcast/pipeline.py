"""
ForecastPipeline
================
End‑to‑end wrapper:

    raw df  →  validate  →  transform  →  feature/lag engineering
           →  fit chosen model  →  produce bounded forecasts
"""

from __future__ import annotations
import joblib, tempfile, os
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from .preprocessing.validator import validate_series
from .preprocessing.transform import auto_transform, TransformMeta
from .features.time_feats import make_time_features
from .models import (
    additive,
    beta_rnn,
    gbm,
    quantile_gbm,
    hybrid_esrnn,
    auto_selector,
)
# --------------------------------------------------------------------- #
AVAILABLE_MODELS: dict[str, type] = {
    "additive": additive.AdditiveModel,
    "gbm": gbm.GBMModel,
    "beta_rnn": beta_rnn.BetaRNNModel,
    "qgbm": quantile_gbm.QuantileGBMModel,
    "esrnn": hybrid_esrnn.HybridESRNNModel,
    "auto": auto_selector.AutoSelector,
}


# --------------------------------------------------------------------- #
@dataclass
class ForecastPipeline:
    """
    Thin orchestration layer – not a heavy AutoML engine (yet).

    Parameters
    ----------
    model_name :
        one of "additive", "gbm", "beta_rnn".
    """

    model_name: Literal["additive", "gbm", "beta_rnn"] = "additive"

    # populated after `.fit()`
    _train_df: pd.DataFrame | None = None
    _meta: TransformMeta | None = None
    _horizon: int | None = None
    _model = None

    # -------------------------------------------------- #
    def fit(self, df: pd.DataFrame, horizon: int, **model_kwargs):
        """
        Validate → transform → fit chosen model.

        Notes
        -----
        * Keeps a copy of the transformed training df so that
          `predict()` can infer frequency & build future dates.
        """
        df = validate_series(df)
        df, meta = auto_transform(df)      # adds ‘y_transformed’
        self._train_df = df
        self._meta = meta
        self._horizon = horizon

        model_cls = AVAILABLE_MODELS[self.model_name]
        self._model = model_cls(**model_kwargs)
        # many models still expect the original df (with ds & engineered cols)
        self._model.fit(df, y_col="y_transformed")
        return self

    # -------------------------------------------------- #
    # -------------------------------------------------- #
    def predict(self) -> pd.Series:
        """
        Produce `horizon` forecasts as a Pandas Series (index = future dates).
        """
        if any(v is None for v in (self._train_df, self._meta, self._model)):
            raise RuntimeError("Call .fit() before .predict().")

        last = self._train_df["ds"].iloc[-1]
        freq = pd.infer_freq(self._train_df["ds"])
        if freq is None:
            # fall back to the modal timedelta
            freq = self._train_df["ds"].diff().mode().iloc[0]

        # generate horizon + 1 points then drop the first (== last training ts)
        full_range = pd.date_range(start=last, periods=self._horizon + 1, freq=freq)
        future_dates = full_range[1:]

        future_df = pd.DataFrame({"ds": future_dates})
        preds_trans = self._model.predict(future_df)
        preds = self._meta.inverse_transform(preds_trans)
        return pd.Series(preds, index=future_dates, name="yhat")

    def predict_interval(self, level: float = 0.8) -> pd.DataFrame:
        """
        Return CI DataFrame (lower/median/upper) if the underlying
        model supports it (currently only qgbm).
        """
        if hasattr(self._model, "predict_interval"):
            last = self._train_df["ds"].iloc[-1]
            freq = pd.infer_freq(self._train_df["ds"]) or self._train_df["ds"].diff().mode().iloc[0]
            full_range = pd.date_range(start=last, periods=self._horizon + 1, freq=freq)
            future_df = pd.DataFrame({"ds": full_range[1:]})
            df_ci = self._model.predict_interval(future_df)
            # inverse‑transform all three columns
            for col in df_ci.columns:
                df_ci[col] = self._meta.inverse_transform(df_ci[col])
            return df_ci
        raise NotImplementedError("This model does not provide intervals.")

    # -------------------------------------------------- #
    #  PERSISTENCE
    # -------------------------------------------------- #
    def save(self, path: str | os.PathLike):
        """
        Persist the entire fitted pipeline with joblib (≈ pickling).
        """
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | os.PathLike) -> "ForecastPipeline":
        return joblib.load(path)