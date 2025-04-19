# SmurphCast API Reference

## 1. `ForecastPipeline`

```python
ForecastPipeline(
    model_name: str = "additive",   # "additive" | "gbm" | "qgbm" | "esrnn" | "auto"
).fit(
    df: pd.DataFrame,               # columns: ds (datetime), y (float)
    horizon: int,                   # forecast steps
    **model_kwargs,
)
```

| Method | Returns | Notes |
|--------|---------|-------|
| `predict()` | Series | point forecast |
| `predict_interval(level)` | DF | lower / median / upper (only qgbm) |
| `save(path)` | — | dill‑serialised pipeline |
| `load(path)` (class) | object | restore saved pipeline |

## 2. Models

| Key | Class | Typical use |
|-----|-------|-------------|
| `additive` | `smurphcast.models.additive.AdditiveModel` | fast baseline w/ trend + seasonality |
| `gbm` | `...models.gbm.GBMModel` | tree‑based, exogenous lags / rolls |
| `qgbm` | `...models.quantile_gbm.QuantileGBMModel` | probabilistic forecast (pinball loss) |
| `esrnn` | `...models.hybrid_esrnn.HybridESRNN` | hybrid exponential‑smoothing + RNN |
| `auto` | `...models.auto_selector.AutoSelector` | back‑test, stack, pick best (default) |

Custom models can be inserted by adding a class that follows the same `fit` / `predict` contract and registering it in `smurphcast.pipeline.AVAILABLE_MODELS`.

## 3. CLI

```bash
smurphcast fit DATA.csv --horizon 4 --model auto [--save best.pkl]
smurphcast cv  DATA.csv --horizon 4 --splits 3 --model gbm
```

Run `smurphcast --help` for every sub‑command.

## 4. Utilities

- `smurphcast.features.time_feats.make_time_features`
- `smurphcast.features.lag_feats.add_lag_features`
- `smurphcast.evaluation.metrics` – mae, pinball_loss, coverage
- `smurphcast.evaluation.backtest.rolling_backtest`

## 5. Back-testing utility

```python
from smurphcast.evaluation.backtest import rolling_backtest

res = rolling_backtest(
    fit_fn=lambda d: ForecastPipeline("additive").fit(d, horizon=4),
    predict_fn=lambda m: m.predict(),
    df=df,
    horizon=4,
    splits=3,
)
print(res["mae_per_fold"])
```

## 6. Feature helpers

```python
from smurphcast.features import time_feats, lag_feats

df_feats = time_feats.make_time_features(df)  # Fourier, dummies, time_idx
df_feats = lag_feats.add_lag_features(df_feats, lags=(1,7), rolls=(7,))  # +lags
```

## 7. Metrics (`smurphcast.evaluation.metrics`)

| Function | Meaning |
|----------|---------|
| `mae(y_true, y_pred)` | Mean-absolute-error |
| `pinball_loss(y, q_hat, τ)` | Quantile loss (needed for probabilistic models) |
| `coverage(y, lo, hi)` | Proportion of `y` that falls in `lo, hi` |

## 8. Examples & sample data

- `examples/` – runnable scripts (`auto_select_forecast.py`) and notebooks
- `smurphcast/data/` – tiny CSVs shipped with the wheel for instant demos

Load a sample programmatically:

```python
from importlib.resources import files
import pandas as pd, smurphcast

path = files(smurphcast.data).joinpath("data_weekly.csv")
df = pd.read_csv(path, parse_dates=["ds"])
```