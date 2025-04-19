# SmurphCast 📈

> **SmurphCast** is a modular, CPU‑friendly forecasting toolkit built for  
> **small percentages** – think churn, click‑through, conversion and retention  
> rates.  
> It combines classic additive models, gradient‑boosted trees, RNN hybrids  
> and an *Auto* selector that stacks them all.

---

## Install

```bash
pip install smurphcast         # PyPI release

# Or from source:
git clone https://github.com/yourhandle/SmurphCast.git
cd SmurphCast
pip install -e .[dev]
```

## Quick‑start

```python
import pandas as pd
from smurphcast.pipeline import ForecastPipeline

df = pd.read_csv("examples/churn_example.csv", parse_dates=["ds"])
pipe = ForecastPipeline(model_name="auto").fit(df, horizon=3)
print(pipe.predict())
```

CLI one‑liner:

```bash
smurphcast fit examples/churn_example.csv --horizon 3 --model auto
```

## Key features

| Feature | Notes |
|---------|-------|
| Bound‑aware losses | Keep forecasts in 0, 1 or custom bounds. |
| Multiple seasonalities | Fourier terms, month‑dummies, automatic periodicity detection. |
| AutoSelector (meta‑model) | Rolling back‑tests, inverse‑MAE blend & non‑negative stacking. |
| Probability forecasts | Quantile GBM gives pinball‑loss‑optimised PIs. |
| Explainability | SHAP‑ready feature importances + residual diagnostics. |
| CPU‑only | All models run in < seconds on laptop‑scale data. |

See the [API reference](docs/api.md) for full details.

## Models

| Key | Class | Typical use |
|-----|-------|-------------|
| `additive` | `smurphcast.models.additive.AdditiveModel` | fast baseline w/ trend + seasonality |
| `gbm` | `...models.gbm.GBMModel` | tree‑based, exogenous lags / rolls |
| `qgbm` | `...models.quantile_gbm.QuantileGBMModel` | probabilistic forecast (pinball loss) |
| `esrnn` | `...models.hybrid_esrnn.HybridESRNN` | hybrid exponential‑smoothing + RNN |
| `auto` | `...models.auto_selector.AutoSelector` | back‑test, stack, pick best (default) |

Custom models can be inserted by adding a class that follows the same `fit` / `predict` contract and registering it in `smurphcast.pipeline.AVAILABLE_MODELS`.

## API Reference

### ForecastPipeline

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

### CLI

```bash
smurphcast fit DATA.csv --horizon 4 --model auto [--save best.pkl]
smurphcast cv  DATA.csv --horizon 4 --splits 3 --model gbm
```

Run `smurphcast --help` for every sub‑command.

### Utilities

- `smurphcast.features.time_feats.make_time_features`
- `smurphcast.features.lag_feats.add_lag_features`
- `smurphcast.evaluation.metrics` – mae, pinball_loss, coverage
- `smurphcast.evaluation.backtest.rolling_backtest`

## Data

SmurphCast ships with tiny, toy datasets so users can run examples without external downloads:

| File | Freq | Metric | Rows |
|------|------|--------|------|
| `data_weekly.csv` | W‑SAT| Synthetic weekly % series | 104 |
| `churn_example.csv` | M | Simulated churn rate | 36 |

Each CSV follows the canonical **Prophet** layout:

```text
ds,y
2023‑01‑01,0.0412
2023‑02‑01,0.0387
...
```

- `ds` – ISO‑8601 timestamp
- `y` – observed percentage in decimal form (0.0 – 1.0)

You can load them via:

```python
from importlib.resources import files
import pandas as pd, smurphcast

path = files(smurphcast.data).joinpath("data_weekly.csv")
df = pd.read_csv(path, parse_dates=["ds"])
```

Note – More realistic, larger examples live under `examples/`, outside the library wheel, so your production install stays minimal in size.

## Project roadmap

- richer holiday / event regressors
- Optuna integration for AutoSelector tuning
- Prophet‑style component plots

## Contributing

We welcome contributions! Please check the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines.

## License

MIT

© 2025 SmurphCast contributors