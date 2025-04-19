"""
AutoSelector – meta‑model that runs several base forecasters, evaluates
rolling‑CV MAE, combines them (inverse‑MAE blend + NN‑LS stack) and keeps
whichever candidate scores best on the final fold.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from ..evaluation.backtest import rolling_backtest
from ..evaluation.metrics import mae

__all__ = ["AutoSelector"]


@dataclass
class AutoSelector:
    base_factories: Dict[str, Callable[[pd.DataFrame], object]]
    horizon: int
    splits: int = 3

    # populated during .fit()
    _models: Dict[str, object] = field(init=False, default_factory=dict)
    _mae: Dict[str, float] = field(init=False, default_factory=dict)
    _best_key: str | None = field(init=False, default=None)
    _stack_coeffs: Dict[str, float] | None = field(init=False, default=None)
    _winner_series: pd.Series | None = field(init=False, default=None)

    # ──────────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame):
        """
        Run rolling‑CV for every base model, remember MAEs and keep a
        full‑data‑fit copy for prediction.
        """
        for name, factory in self.base_factories.items():
            res = rolling_backtest(
                fit_fn=factory,
                predict_fn=lambda m, *_: m.predict(),   # <── swallow “horizon”
                df=df,
                horizon=self.horizon,
                splits=self.splits,
            )
            self._mae[name] = float(np.mean(res["mae_per_fold"]))
            self._models[name] = factory(df)           # fit on entire series

        self._choose_winner(df)
        return self

    # ------------------------------------------------------------------
    def predict(self) -> pd.Series:
        if self._winner_series is None:
            raise RuntimeError("Call .fit() before .predict().")
        return self._winner_series.copy()

    # ------------------------------------------------------------------
    def _choose_winner(self, df: pd.DataFrame):
        """Blend/stack and pick the best candidate on the last fold."""
        topN = sorted(self._mae, key=self._mae.get)[:3]
        preds = {m: self._models[m].predict() for m in topN}

        inv_blend = _inverse_mae_blend(preds, {k: self._mae[k] for k in topN})
        stack_pred, coeffs = _nnls_stack(preds, df["y"].iloc[-self.horizon:].values)
        self._stack_coeffs = coeffs

        truth_last = df["y"].iloc[-self.horizon:].values
        candidates = {
            "inv": inv_blend,
            "best_single": preds[topN[0]],
            "stack": stack_pred,
        }
        self._best_key = min(candidates, key=lambda k: mae(truth_last,
                                                           candidates[k].values))
        self._winner_series = candidates[self._best_key]


# ─────────────────────────────────────────────────────────────────────────
# Helper functions (stand‑alone to avoid ForecastPipeline imports)
# ─────────────────────────────────────────────────────────────────────────
def _inverse_mae_blend(preds: Dict[str, pd.Series], maes: Dict[str, float]) -> pd.Series:
    wts = {k: 1 / maes[k] for k in preds}
    tot = sum(wts.values())
    wts = {k: v / tot for k, v in wts.items()}
    combo = sum(series * wts[k] for k, series in preds.items())
    return combo.rename("inv_blend")


def _nnls_stack(preds: Dict[str, pd.Series], y_true: np.ndarray):
    cols = list(preds)
    X = np.column_stack([preds[c].values for c in cols])
    lr = LinearRegression(positive=True, fit_intercept=False).fit(X, y_true)
    coeffs = {c: float(w) for c, w in zip(cols, lr.coef_)}
    yhat = X @ lr.coef_
    return pd.Series(yhat, index=preds[cols[0]].index, name="stack"), coeffs
