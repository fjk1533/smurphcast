import pandas as pd
from sklearn.metrics import mean_absolute_error
from typing import Callable, List


def rolling_backtest(
    fit_fn: Callable[[pd.DataFrame], object],
    predict_fn: Callable[[object, int], pd.Series],
    df: pd.DataFrame,
    horizon: int,
    splits: int = 3,
    y_col: str = "y",
) -> List[float]:
    """
    Simple rolling‑origin evaluation that returns MAE per split.
    """
    errors = []
    n = len(df)
    for i in range(splits, 0, -1):
        train_end = n - i * horizon
        train_df = df.iloc[:train_end]
        test_df = df.iloc[train_end : train_end + horizon]

        model = fit_fn(train_df)
        preds = predict_fn(model, horizon)
        errors.append(mean_absolute_error(test_df[y_col].values, preds))
    return errors
