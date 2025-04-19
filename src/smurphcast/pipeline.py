# ─────────────────────────────────────────────────────────────────────────
#  src/smurphcast/pipeline.py
# ─────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import os
import joblib
from dataclasses import dataclass
from typing import Literal, Dict, Callable

import pandas as pd

from .preprocessing.validator import validate_series
from .preprocessing.transform import auto_transform, TransformMeta
from .features.time_feats import make_time_features   # kept for future use
from .models import (
    additive,
    beta_rnn,
    gbm,
    quantile_gbm,
    hybrid_esrnn,
    auto_selector,
)

# --------------------------------------------------------------------- #
#  Registry of concrete model classes
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
    High‑level orchestrator for SmurphCast models.

    Parameters
    ----------
    model_name : one of
        "additive" | "gbm" | "qgbm" | "beta_rnn" | "esrnn" | **"auto"**

    Typical usage
    -------------
    pipe = ForecastPipeline("auto").fit(df, horizon=12)
    yhat = pipe.predict()
    """
    model_name: Literal[
        "additive", "gbm", "qgbm", "beta_rnn", "esrnn", "auto"
    ] = "additive"

    # populated by .fit()
    _train_df: pd.DataFrame | None = None
    _meta: TransformMeta     | None = None
    _horizon: int            | None = None
    _model                   = None

    # ──────────────────────────────────────────────────────────────
    #  INTERNAL ‑ factory for AutoSelector
    # ──────────────────────────────────────────────────────────────
    def _make_auto_selector(self, horizon: int, splits: int = 3):
        """
        Build an AutoSelector *with* the required base‑model factories
        and horizon argument baked‑in.
        """
        def _factory_for(name: str) -> Callable[[pd.DataFrame], "ForecastPipeline"]:
            def _f(df: pd.DataFrame):
                # recurse into ForecastPipeline but with concrete model
                return ForecastPipeline(model_name=name).fit(df, horizon=horizon)
            return _f

        factories = {k: _factory_for(k) for k in _BASE_MODELS}
        return auto_selector.AutoSelector(
            base_factories=factories,
            horizon=horizon,
            splits=splits,
        )

    # ──────────────────────────────────────────────────────────────
    #  FIT
    # ──────────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame, horizon: int, **model_kwargs):
        """
        Validate > transform > fit the chosen model (or AutoSelector).
        """
        df = validate_series(df)
        df, meta = auto_transform(df)          # adds y_transformed column

        self._train_df, self._meta, self._horizon = df, meta, horizon

        # Choose concrete implementation
        if self.model_name == "auto":
            # build & fit AutoSelector (NO y_col kwarg!)
            self._model = self._make_auto_selector(horizon)
            self._model.fit(df)
        else:
            model_cls = _AVAILABLE_MODELS[self.model_name]
            self._model = model_cls(**model_kwargs)
            self._model.fit(df, y_col="y_transformed")

        return self

    # ──────────────────────────────────────────────────────────────
    #  PREDICT (point)
    # ──────────────────────────────────────────────────────────────
    def predict(self) -> pd.Series:
        if any(v is None for v in (self._train_df, self._meta, self._model)):
            raise RuntimeError("Call .fit() before .predict().")

        last = self._train_df["ds"].iloc[-1]
        freq = pd.infer_freq(self._train_df["ds"]) or self._train_df["ds"].diff().mode().iloc[0]
        future_dates = pd.date_range(last, periods=self._horizon + 1,
                                     freq=freq, closed="right")
        future_df = pd.DataFrame({"ds": future_dates})

        y_hat_trans = self._model.predict(future_df)
        y_hat = self._meta.inverse_transform(y_hat_trans)

        return pd.Series(y_hat, index=future_dates, name="yhat")

    # ──────────────────────────────────────────────────────────────
    #  PREDICT (interval)
    # ──────────────────────────────────────────────────────────────
    def predict_interval(self, level: float = 0.8) -> pd.DataFrame:
        """
        lower / median / upper – only supported for qgbm at present.
        """
        if not hasattr(self._model, "predict_interval"):
            raise NotImplementedError("This model does not provide intervals.")

        last = self._train_df["ds"].iloc[-1]
        freq = pd.infer_freq(self._train_df["ds"]) or self._train_df["ds"].diff().mode().iloc[0]
        future_dates = pd.date_range(last, periods=self._horizon + 1,
                                     freq=freq, closed="right")
        future_df = pd.DataFrame({"ds": future_dates})

        df_ci = self._model.predict_interval(future_df, level=level)
        for col in df_ci.columns:
            df_ci[col] = self._meta.inverse_transform(df_ci[col])
        return df_ci.set_index(future_dates)

    # ──────────────────────────────────────────────────────────────
    #  PERSISTENCE
    # ──────────────────────────────────────────────────────────────
    def save(self, path: str | os.PathLike):
        """Pickle the entire fitted pipeline (feature meta + model)."""
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | os.PathLike) -> "ForecastPipeline":
        return joblib.load(path)
