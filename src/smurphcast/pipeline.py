# ─────────────────────────────────────────────────────────────────────────
#  src/smurphcast/pipeline.py
# ─────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import os
import joblib
from dataclasses import dataclass
from typing import Dict, Callable, Literal

import pandas as pd

from .preprocessing.validator import validate_series
from .preprocessing.transform import auto_transform, TransformMeta
from .features.time_feats import make_time_features  # (still used by some models)
from .models import (
    additive,
    beta_rnn,
    gbm,
    quantile_gbm,
    hybrid_esrnn,
    auto_selector,
)

# --------------------------------------------------------------------- #
# 1) Base models that AutoSelector will consider
_BASE_MODELS: Dict[str, type] = {
    "additive":  additive.AdditiveModel,
    "gbm":       gbm.GBMModel,
    "qgbm":      quantile_gbm.QuantileGBMModel,
    "esrnn":     hybrid_esrnn.HybridESRNNModel,
    "beta_rnn":  beta_rnn.BetaRNNModel,          # ← included
}

# 2) All models that ForecastPipeline can instantiate directly
_AVAILABLE_MODELS: Dict[str, type] = {
    **_BASE_MODELS,
    "auto": auto_selector.AutoSelector,
}
# --------------------------------------------------------------------- #


@dataclass
class ForecastPipeline:
    """
    End‑to‑end orchestrator for SmurphCast models.

    Parameters
    ----------
    model_name :
        "additive" | "gbm" | "qgbm" | "esrnn" | "beta_rnn" | **"auto"**
    """
    model_name: Literal[
        "additive", "gbm", "qgbm", "esrnn", "beta_rnn", "auto"
    ] = "additive"

    # populated after `.fit()`
    _train_df: pd.DataFrame | None = None
    _meta: TransformMeta | None = None
    _horizon: int | None = None
    _model = None

    # ──────────────────────────────────────────────────────────────
    # AUTO‑SELECTOR HELPER
    # ──────────────────────────────────────────────────────────────
    def _make_auto_selector(self, horizon: int, splits: int = 3):
        """
        Build an AutoSelector instance with the required arguments.
        """

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
    # FIT
    # ──────────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame, horizon: int, **model_kwargs):
        """
        Validate → transform → fit the chosen model.
        """
        df = validate_series(df)
        df, meta = auto_transform(df)

        self._train_df, self._meta, self._horizon = df, meta, horizon

        if self.model_name == "auto":
            self._model = self._make_auto_selector(horizon)
        else:
            model_cls = _AVAILABLE_MODELS[self.model_name]
            self._model = model_cls(**model_kwargs)

        # most base models expect engineered features already in df
        y_col = "y_transformed" if "y_transformed" in df.columns else "y"
        self._model.fit(df, y_col=y_col)
        return self

    # ──────────────────────────────────────────────────────────────
    # PREDICT (point)
    # ──────────────────────────────────────────────────────────────
    def predict(self) -> pd.Series:
        if any(v is None for v in (self._train_df, self._meta, self._model)):
            raise RuntimeError("Call .fit() before .predict().")

        last = self._train_df["ds"].iloc[-1]
        freq = pd.infer_freq(self._train_df["ds"]) or self._train_df["ds"].diff().mode().iloc[0]
        future_dates = pd.date_range(last, periods=self._horizon + 1, freq=freq, closed="right")
        future_df = pd.DataFrame({"ds": future_dates})

        yhat_trans = self._model.predict(future_df)
        yhat = self._meta.inverse_transform(yhat_trans)
        return pd.Series(yhat, index=future_dates, name="yhat")

    # ──────────────────────────────────────────────────────────────
    # PREDICT (interval)
    # ──────────────────────────────────────────────────────────────
    def predict_interval(self, level: float = 0.8) -> pd.DataFrame:
        """
        Return CI DataFrame if the underlying model supports it.
        """
        if not hasattr(self._model, "predict_interval"):
            raise NotImplementedError("This model does not provide intervals.")

        last = self._train_df["ds"].iloc[-1]
        freq = pd.infer_freq(self._train_df["ds"]) or self._train_df["ds"].diff().mode().iloc[0]
        future_dates = pd.date_range(last, periods=self._horizon + 1, freq=freq, closed="right")
        future_df = pd.DataFrame({"ds": future_dates})

        df_ci = self._model.predict_interval(future_df, level=level)
        for col in df_ci.columns:
            df_ci[col] = self._meta.inverse_transform(df_ci[col])
        return df_ci

    # ──────────────────────────────────────────────────────────────
    # PERSISTENCE
    # ──────────────────────────────────────────────────────────────
    def save(self, path: str | os.PathLike):
        """Persist the entire fitted pipeline."""
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | os.PathLike) -> "ForecastPipeline":
        return joblib.load(path)
