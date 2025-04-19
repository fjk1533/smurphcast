"""
AutoSelector – meta‑model that runs several base forecasters, evaluates
rolling‑CV MAE, then forms an inverse‑MAE blend *and* a non‑negative least
squares stack.  It returns whichever candidate scores best on the final fold.

The class is *self‑contained* – no import of ForecastPipeline – so it can be
imported early without circular‑import issues.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List

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

    # runtime attributes
    _models: Dict[str, object] = field(init=False, default_factory=dict)
    _mae: Dict[str, float] = field(init=False, default_factory=dict)
    _best_key: str | None = field(init=False, default=None)
    _stack_coeffs: Dict[str, float] | None = field(init=False, default=None)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fit(self, df: pd.DataFrame):
        """Run rolling‑CV on each base model, remember MAEs & fitted object."""
        for name, factory in self.base_factories.items():
            res = rolling_backtest(
                fit_fn=factory,
                predict_fn=lambda m: m.predict(),
                df=df,
                horizon=self.horizon,
                splits=self.splits,
            )
            self._mae[name] = float(np.mean(res["mae_per_fold"]))
            self._models[name] = factory(df)  # full‑data fit

        self._choose_winner(df)
        return self

    def predict(self) -> pd.Series:
        return self._winner_series

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _choose_winner(self, df: pd.DataFrame):
        topN = sorted(self._mae, key=self._mae.get)[:3]
        preds = {m: self._models[m].predict() for m in topN}

        # inverse‑MAE blend
        inv_preds = _inverse_mae_blend(preds, {k: self._mae[k] for k in topN})

        # non‑negative linear stack on last fold
        stack_preds, coeffs = _nnls_stack(preds, df["y"].iloc[-self.horizon :].values)
        self._stack_coeffs = coeffs

        # evaluate on last fold
        final_truth = df["y"].iloc[-self.horizon :].values
        candidates = {
            "inv": inv_preds,
            "best_single": preds[topN[0]],
            "stack": stack_preds,
        }
        pick = min(candidates, key=lambda k: mae(final_truth, candidates[k].values))
        self._winner_series = candidates[pick]
        self._best_key = pick


# ---------------------------------------------------------------------- #
# Utility blend helpers (stand‑alone to avoid importing ForecastPipeline)
# ---------------------------------------------------------------------- #
def _inverse_mae_blend(preds: Dict[str, pd.Series], maes: Dict[str, float]) -> pd.Series:
    wts = {k: 1 / maes[k] for k in preds}
    total = sum(wts.values())
    wts = {k: v / total for k, v in wts.items()}
    acc = sum(series * wts[k] for k, series in preds.items())
    return acc.rename("inv_blend")


def _nnls_stack(preds: Dict[str, pd.Series], y_true: np.ndarray):
    cols = list(preds)
    X = np.column_stack([preds[c].values for c in cols])
    lr = LinearRegression(positive=True, fit_intercept=False).fit(X, y_true)
    coeffs = {c: float(w) for c, w in zip(cols, lr.coef_)}
    yhat = (X @ lr.coef_).astype(float)
    return pd.Series(yhat, index=preds[cols[0]].index, name="stack"), coeffs
