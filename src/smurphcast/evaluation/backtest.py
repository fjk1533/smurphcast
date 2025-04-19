from __future__ import annotations
import pandas as pd
from sklearn.metrics import mean_absolute_error
from typing import Callable, List, Dict


def rolling_backtest(
    fit_fn: Callable[[pd.DataFrame], object],
    predict_fn: Callable[[object, int], pd.Series],
    df: pd.DataFrame,
    horizon: int,
    splits: int = 3,
    y_col: str = "y",
    return_forecasts: bool = False,
) -> Dict[str, object]:
    """
    Returns dict with 'mae_per_fold' and optionally 'forecasts' (list of Series).
    Each split trains on earlier part and tests on next *horizon* points.
    """
    errors: list[float] = []
    fcsts: list[pd.Series] = []

    n = len(df)
    for i in range(splits, 0, -1):
        train_end = n - i * horizon
        train_df = df.iloc[:train_end]
        test_df = df.iloc[train_end : train_end + horizon]

        model = fit_fn(train_df)
        preds = predict_fn(model, horizon)
        errors.append(mean_absolute_error(test_df[y_col].values, preds))

        if return_forecasts:
            preds.index = test_df.index           # align for convenience
            fcsts.append(preds)

    result = {"mae_per_fold": errors}
    if return_forecasts:
        result["forecasts"] = fcsts
    return result
