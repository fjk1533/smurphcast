"""
AutoSelector
============

A lightweight AutoML / stacking wrapper around the core SmurphCast models.

• Trains every *base* model once.
• Evaluates MAE on a rolling back‑test.
• Creates two ensembles:
    1. Inverse‑MAE weighted average
    2. Non‑negative linear stack (weights learned on penultimate fold,
       validated on last fold).
• Chooses whichever candidate has the lowest MAE on the last fold.
• Exposes the winning model via a familiar `.predict()` API.

Parameters
----------
base_models         : list[str]
    Which core models to consider.  Keys must exist in ForecastPipeline.
horizon             : int
    Forecast horizon (passed through to each base model).
splits              : int, default 3
    Number of rolling folds.
stack_min_samples   : int, default 2
    If len(series) < stack_min_samples * horizon -> skip stacking.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from ..evaluation.backtest import rolling_backtest
from ..pipeline import ForecastPipeline


class AutoSelector:
    _EXCLUDE = {"auto"}  # avoid recursive call

    # ---- public ---------------------------------------------------- #
    def __init__(
        self,
        base_models: list[str] | None = None,
        splits: int = 3,
        stack_min_samples: int = 2,
    ):
        from ..pipeline import AVAILABLE_MODELS  # circular‑safe import

        all_models = [m for m in AVAILABLE_MODELS if m not in self._EXCLUDE]
        self.base_models = base_models or all_models
        self.splits = splits
        self.stack_min_samples = stack_min_samples

        # trained artefacts
        self._winner_key: str | None = None
        self._winner_pipe: ForecastPipeline | None = None
        self._future_pred: pd.Series | None = None
        self.weights_: dict[str, float] | None = None  # for ensembles

    # ---------------------------------------------------------------- #
    def fit(self, df: pd.DataFrame, horizon: int, **kwargs):
        maes, preds = self._fit_base_models(df, horizon, **kwargs)
        self._select_winner(df, horizon, maes, preds)
        return self

    # ---------------------------------------------------------------- #
    def predict(self) -> pd.Series:
        if self._future_pred is None:
            raise RuntimeError("Call .fit() first.")
        return self._future_pred.copy()

    # ---- helpers --------------------------------------------------- #
    def _fit_base_models(
        self, df: pd.DataFrame, horizon: int, **kwargs
    ) -> tuple[Dict[str, float], Dict[str, Dict[str, np.ndarray]]]:
        maes: Dict[str, float] = {}
        preds: Dict[str, Dict[str, np.ndarray]] = {}

        for name in self.base_models:
            # MAE via rolling back‑test
            res = rolling_backtest(
                fit_fn=lambda d, n=name: ForecastPipeline(model_name=n).fit(
                    d, horizon=horizon, **kwargs
                ),
                predict_fn=lambda m, h=horizon: m.predict(),
                df=df,
                horizon=horizon,
                splits=self.splits,
            )
            maes[name] = float(np.mean(res["mae_per_fold"]))

            # three fits for stacking
            pipe_full = ForecastPipeline(model_name=name).fit(df, horizon=horizon, **kwargs)
            pipe_train = ForecastPipeline(model_name=name).fit(
                df.iloc[: -2 * horizon], horizon=horizon, **kwargs
            )
            pipe_valid = ForecastPipeline(model_name=name).fit(
                df.iloc[: -horizon], horizon=horizon, **kwargs
            )

            preds[name] = {
                "future": pipe_full.predict().to_numpy(),
                "train": pipe_train.predict().to_numpy(),
                "valid": pipe_valid.predict().to_numpy(),
                "pipe": pipe_full,
            }

        return maes, preds

    # ---------------------------------------------------------------- #
    def _select_winner(
        self,
        df: pd.DataFrame,
        horizon: int,
        maes: Dict[str, float],
        preds: Dict[str, Dict[str, np.ndarray]],
    ):
        # inverse‑MAE blend
        w_inv = {m: 1 / maes[m] for m in preds}
        s = sum(w_inv.values())
        w_inv = {m: w_inv[m] / s for m in w_inv}
        inv_pred = sum(preds[m]["future"] * w_inv[m] for m in preds)
        inv_mae = float(
            np.mean(np.abs(df["y"].iloc[-horizon:].to_numpy() - inv_pred))
        )

        # stacking (if long enough)
        stack_pred, stack_mae = None, np.inf
        if len(df) >= self.stack_min_samples * horizon + 1:
            X_train = np.column_stack([preds[m]["train"] for m in preds])
            y_train = df["y"].iloc[-2 * horizon : -horizon].to_numpy()

            lr = LinearRegression(fit_intercept=False, positive=True).fit(X_train, y_train)
            coef = lr.coef_
            coef = coef / coef.sum() if coef.sum() > 0 else coef
            self.weights_ = {m: float(c) for m, c in zip(preds, coef)}

            X_valid = np.column_stack([preds[m]["valid"] for m in preds])
            y_valid = df["y"].iloc[-horizon:].to_numpy()
            stack_mae = float(np.mean(np.abs(y_valid - X_valid @ coef)))
            stack_pred = sum(preds[m]["future"] * coef[i] for i, m in enumerate(preds))

        # candidate pool
        candidates = {
            "inv_mae": (inv_pred, inv_mae),
            **{m: (preds[m]["future"], maes[m]) for m in preds},
        }
        if stack_pred is not None:
            candidates["stack"] = (stack_pred, stack_mae)

        winner_key = min(candidates, key=lambda k: candidates[k][1])
        self._winner_key = winner_key

        if winner_key in preds:  # single model
            self._winner_pipe = preds[winner_key]["pipe"]
            self._future_pred = candidates[winner_key][0]
        else:  # ensemble
            self._winner_pipe = None
            self._future_pred = candidates[winner_key][0]


# --------------------------------------------------------------------- #
# user‑facing helper: create pipeline with model_name="auto"
def register_auto_model():
    from ..pipeline import AVAILABLE_MODELS

    AVAILABLE_MODELS["auto"] = AutoSelector


register_auto_model()
