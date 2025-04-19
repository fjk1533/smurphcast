# ─────────────────────────────────────────────────────────────────────────
#  src/smurphcast/pipeline.py
# ─────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import os, joblib
from dataclasses import dataclass
from typing import Literal, Dict, Callable

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
_BASE_MODELS: Dict[str, type] = {
    "additive":   additive.AdditiveModel,
    "gbm":        gbm.GBMModel,
    "qgbm":       quantile_gbm.QuantileGBMModel,
    "beta_rnn":   beta_rnn.BetaRNNModel,
    "esrnn":      hybrid_esrnn.HybridESRNNModel,
}
_AVAILABLE_MODELS: Dict[str, type] = {**_BASE_MODELS, "auto": auto_selector.AutoSelector}
# --------------------------------------------------------------------- #


@dataclass
class ForecastPipeline:
    """
    End‑to‑end orchestrator.

    model_name:
        "additive" | "gbm" | "qgbm" | "beta_rnn" | "esrnn" | **"auto"**
    """
    model_name: Literal[
        "additive", "gbm", "qgbm", "beta_rnn", "esrnn", "auto"
    ] = "additive"

    # will be filled by .fit()
    _train_df: pd.DataFrame | None = None
    _meta: TransformMeta     | None = None
    _horizon: int            | None = None
    _model                   = None

    # ──────────────────────────────────────────────────────────────
    def _make_auto_selector(self, horizon: int, splits: int = 3):
        """Return an AutoSelector wired with tiny ForecastPipeline factories."""
        def _factory_for(name: str) -> Callable[[pd.DataFrame], "ForecastPipeline"]:
            def _f(df: pd.DataFrame):
                return ForecastPipeline(model_name=name).fit(df, horizon=horizon)
            return _f

        factories = {k: _factory_for(k) for k in _BASE_MODELS}
        return auto_selector.AutoSelector(
            base_factories=factories,
            horizon=horizon,
            splits=splits,
        )

    # ──────────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame, horizon: int, **model_kwargs):
        """Validate → transform → fit."""
        df = validate_series(df)
        df, meta = auto_transform(df)      # adds y_transformed
        self._train_df, self._meta, self._horizon = df, meta, horizon

        # choose the model
        if self.model_name == "auto":
            self._model = self._make_auto_selector(horizon)
            self._model.fit(df)                      # ← no y_col for AutoSelector
        else:
            model_cls = _AVAILABLE_MODELS[self.model_name]
            self._model = model_cls(**model_kwargs)
            self._model.fit(df, y_col="y_transformed")

        return self

    # ──────────────────────────────────────────────────────────────
    def predict(self) -> pd.Series:
        if any(v is None for v in (self._train_df, self._meta, self._model)):
            raise RuntimeError("Call .fit() before .predict().")

        last = self._train_df["ds"].iloc[-1]
        freq = pd.infer_freq(self._train_df["ds"]) or self._train_df["ds"].diff().mode().iloc[0]
        future_dates = pd.date_range(last, periods=self._horizon + 1, freq=freq, closed="right")
        future_df = pd.DataFrame({"ds": future_dates})

        y_hat_trans = self._model.predict(future_df)
        y_hat = self._meta.inverse_transform(y_hat_trans)
        return pd.Series(y_hat, index=future_dates, name="yhat")

    # predict_interval(), save(), load() unchanged …
