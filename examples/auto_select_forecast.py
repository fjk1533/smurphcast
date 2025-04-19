#!/usr/bin/env python
"""
Auto‑select, stack, and forecast with SmurphCast.

Usage
-----
python examples/auto_select_forecast.py data.csv 3
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from smurphcast.pipeline import ForecastPipeline
from smurphcast.evaluation.metrics import pinball_loss, coverage
from smurphcast.evaluation.backtest import rolling_backtest

# ----------------------------------------------------------------- #
# Configuration                                                     #
# ----------------------------------------------------------------- #
MODELS = {
    "additive": {},
    "gbm": {},
    "qgbm": {},
    "esrnn": {"season_length": 12},
}
SPLITS = 3
ALPHA = 0.2  # -> 80 % PI
REPORT: dict = {}


# ----------------------------------------------------------------- #
def load_series(path: Path) -> pd.DataFrame:
    """Returns df with *ds* (datetime) & *y* (float)."""
    return pd.read_csv(path, parse_dates=["ds"])


def fit_factory(name: str, horizon: int, kwargs: dict) -> Callable[[pd.DataFrame], ForecastPipeline]:
    def _fit(df_: pd.DataFrame):
        return ForecastPipeline(model_name=name).fit(df_, horizon=horizon, **kwargs)

    return _fit


# ----------------------------------------------------------------- #
def evaluate_point_metrics(df: pd.DataFrame, horizon: int) -> Dict[str, float]:
    maes: Dict[str, float] = {}
    for name, kwargs in MODELS.items():
        res = rolling_backtest(
            fit_fn=fit_factory(name, horizon, kwargs),
            predict_fn=lambda m, h=horizon: m.predict(),
            df=df,
            horizon=horizon,
            splits=SPLITS,
        )
        mae = float(np.mean(res["mae_per_fold"]))
        REPORT[name] = {"mae_per_fold": res["mae_per_fold"]}
        maes[name] = mae
    return maes


def evaluate_prob_metrics(best: str, df: pd.DataFrame, horizon: int):
    if best != "qgbm":
        return
    pipe = fit_factory(best, horizon, MODELS[best])(df.iloc[:-horizon])
    ci = pipe.predict_interval(level=1 - ALPHA)
    truth = df["y"].iloc[-horizon:].to_numpy()
    REPORT[best].update(
        {
            "pinball": float(pinball_loss(truth, ci["median"].to_numpy(), 0.5)),
            "coverage": float(coverage(truth, ci["lower"], ci["upper"])),
        }
    )


# ----------------------------------------------------------------- #
def inv_mae_blend(preds: dict[str, np.ndarray], maes: dict[str, float]) -> np.ndarray:
    w = {m: 1 / maes[m] for m in preds}
    s = sum(w.values())
    w = {m: w[m] / s for m in w}
    REPORT["ensemble_weights"] = w
    out = sum(preds[m] * w[m] for m in preds)
    return out


def two_fold_stack(
    preds: dict[str, dict[str, np.ndarray]],
    y_train: np.ndarray,
    y_valid: np.ndarray,
) -> tuple[np.ndarray, dict, float]:
    """
    preds[model]['train'|'valid'|'future'] = np.ndarray(horizon,)
    """
    X_train = np.column_stack([preds[m]["train"] for m in preds])
    lr = LinearRegression(fit_intercept=False, positive=True).fit(X_train, y_train)
    coeffs = lr.coef_
    coeffs = coeffs / coeffs.sum() if coeffs.sum() > 0 else coeffs
    coeff_dict = {m: float(c) for m, c in zip(preds, coeffs)}
    REPORT["stack_coeffs"] = coeff_dict

    # validation MAE
    X_valid = np.column_stack([preds[m]["valid"] for m in preds])
    stack_valid = X_valid @ coeffs
    mae_valid = float(np.mean(np.abs(y_valid - stack_valid)))

    # blended future forecast
    stack_future = sum(preds[m]["future"] * coeffs[i] for i, m in enumerate(preds))
    return stack_future, coeff_dict, mae_valid


# ----------------------------------------------------------------- #
def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    path = Path(sys.argv[1])
    horizon = int(sys.argv[2])

    df = load_series(path)

    # 1 — back‑test single models
    maes = evaluate_point_metrics(df, horizon)
    best_single = min(maes, key=maes.get)
    evaluate_prob_metrics(best_single, df, horizon)

    print("MAE leaderboard:")
    for k, v in sorted(maes.items(), key=lambda kv: kv[1]):
        print(f"  {k:<8} {v:.6f}")
    print(f"🏆  Best single: {best_single}")

    # 2 — predictions for top‑3 models
    top3 = sorted(maes, key=maes.get)[:3]
    preds: dict[str, dict[str, np.ndarray]] = {}
    for m in top3:
        # full fit: forecast next horizon
        preds_full = fit_factory(m, horizon, MODELS[m])(df).predict().to_numpy()

        # train on data up to k‑2 fold
        preds_train = fit_factory(m, horizon, MODELS[m])(
            df.iloc[: -2 * horizon]
        ).predict().to_numpy()

        # train on data up to k‑1 fold
        preds_valid = fit_factory(m, horizon, MODELS[m])(
            df.iloc[: -horizon]
        ).predict().to_numpy()

        preds[m] = {"future": preds_full, "train": preds_train, "valid": preds_valid}

    # 3 — inverse‑MAE blend
    inv_blend = inv_mae_blend({m: preds[m]["future"] for m in preds}, maes)
    inv_mae_valid = float(
        np.mean(np.abs(df["y"].iloc[-horizon:].to_numpy() - inv_blend))
    )

    # 4 — stacking (requires ≥ 2 horizon samples)
    if len(df) >= 2 * horizon + 1:
        stack_pred, _, stack_mae = two_fold_stack(
            preds,
            y_train=df["y"].iloc[-2 * horizon : -horizon].to_numpy(),
            y_valid=df["y"].iloc[-horizon:].to_numpy(),
        )
    else:
        stack_pred, stack_mae = None, np.inf

    # candidate pool
    candidates = {
        best_single: (preds[best_single]["future"], maes[best_single]),
        "inv_mae": (inv_blend, inv_mae_valid),
    }
    if stack_pred is not None:
        candidates["stack"] = (stack_pred, stack_mae)

    print("🔎  Candidate MAEs on last fold:")
    for k, (_, v) in candidates.items():
        print(f"  {k:<8} {v:.6f}")

    winner_key = min(candidates, key=lambda k: candidates[k][1])
    print(f"🎯  Final winner: {winner_key}")

    # ---------------------------------------------------------------- #
    # Save artefacts
    # ---------------------------------------------------------------- #
    future_fc = candidates[winner_key][0]
    forecast_df = pd.DataFrame(
        {
            "ds": pd.date_range(
                df["ds"].iloc[-1] if "ds" in df else df.index[-1],
                periods=horizon + 1,
                freq=pd.infer_freq(df["ds"] if "ds" in df else df.index),
            )[1:],
            "yhat": future_fc,
        }
    )
    forecast_df.to_csv("best_forecast.csv", index=False)

    if winner_key in MODELS:
        best_pipe = fit_factory(winner_key, horizon, MODELS[winner_key])(df)
        best_pipe.save("best_model.pkl")
        suffix = " best_model.pkl"
    else:
        suffix = ""

    REPORT["final_winner"] = winner_key
    REPORT["candidate_mae"] = {k: v for k, (_, v) in candidates.items()}
    REPORT["mean_mae"] = maes
    with open("report.json", "w") as f:
        json.dump(REPORT, f, indent=2)

    print(f"📦  Saved: best_forecast.csv report.json{suffix}")


if __name__ == "__main__":
    main()
