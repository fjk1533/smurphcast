# SmurphCast 📈 1.0.1

[![PyPI](https://img.shields.io/badge/pypi-v0.1.3-blue.svg)](https://pypi.org/project/smurphcast/)
[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Docs](https://img.shields.io/badge/docs-latest-green.svg)](https://github.com/yourhandle/SmurphCast)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**SmurphCast** is the first open-source forecasting library **designed explicitly for percentages**: churn, click-through, conversion, retention & rate-based KPIs. Lightweight ✅, explainable ✅, production-ready ✅—and it runs on a laptop CPU.

## Why another forecasting library?

Modern teams track **hundreds of tiny percentages**—they spike on Black Friday, dip during outages, and never exceed 100%.
Classic tools (ARIMA, Prophet, deep nets) either ignore those hard bounds or explode with gradient issues.

SmurphCast was born inside a growth-marketing team frustrated with:

* Unbounded predictions (> 100% CTR 😩)
* Brittle seasonality when data is bi-weekly, quarterly, or irregular
* Needing to babysit half a dozen libraries for every experiment

So we distilled the playbook that **actually worked** into one cohesive package.

## ✨ Key reasons you'll love SmurphCast

| Feature | SmurphCast advantage | What it means for you |
|---------|---------------------|----------------------|
| 🔒 Bound-aware losses | Bounded MSE / quantile pinball keep forecasts in [0, 1] | no more negative churn or > 100% conversion |
| 📅 Multiple seasonalities | Automatically detects weekly, monthly, yearly or n-period cycles | accurate retail & campaign spikes without manual fiddling |
| 🤝 Hybrid architecture | Additive (Fourier + dummies) • GBM • Quantile GBM • ES-RNN | pick the weapon that fits your data size + CPU budget |
| 🔄 AutoSelector | Back-tests every model, inverse-MAE blends, **non-negative stacking** | get "good enough" forecasts out-of-the-box—then fine-tune |
| 💬 Explainability | SHAP-ready importances, residual diagnostics, coverage metrics | show the C-suite **why** the forecast moved |
| 🚀 Zero-GPU | Pure NumPy / LightGBM / PyTorch-CPU | run in CI, serverless, or a Docker side-car |

## A brief history of the internal models

| Year | Model | Inspiration | What we kept / improved |
|------|-------|-------------|------------------------|
| 2017 | Prophet | Facebook's decomposable trend/seasonality | Fourier features & Laplace trend regularisation |
| 2018 | ES-RNN (M4 winner) | Uber's hybrid Holt-Winters + RNN | Our **HybridESRNN** shrinks to CPU-size & enforces bounds |
| 2020 | LightGBM CTR models | Ad-tech uses trees on lagged features | Wrapped as `GBMModel`, turnkey lags & rolling stats |
| 2022 | Quantile GBM | pinball-loss for PIs | Adds automatic 80% & 95% intervals |
| 2025 | AutoSelector | meta-learning & stacking competitions | Rolling CV, inverse-MAE weight blend, non-negative OLS stack |

The result: **four specialised forecasters + one meta-model** that systematically outperforms any single approach on marketing KPI datasets.

## Installation

```python
pip install smurphcast      # PyPI release

# Dev install
git clone https://github.com/yourhandle/smurphcast.git
cd smurphcast
pip install -e .[dev]       # tests, ruff, black, hatch

SmurphCast requires Python ≥ 3.9 and no GPU.
```

## 0-minute quick-start

```python
import pandas as pd
from smurphcast.pipeline import ForecastPipeline

df = pd.read_csv("examples/churn_example.csv", parse_dates=["ds"])

# Auto picks the best model & stacking weights
pipe = ForecastPipeline(model_name="auto").fit(df, horizon=3)

print(pipe.predict())         # point forecast
print(pipe.predict_interval(.9))  # 90% PI if supported
pipe.save("smurf.pkl")        # deploy anywhere (dill)
```

CLI:
```bash
smurphcast fit examples/churn_example.csv --horizon 3 --model auto --save best.pkl
```

## Architecture 🔧

```
Raw CSV  --> validator / outlier scrub
   |
   v      --> logit / log / Box-Cox transforms
Features  
   |
   v      --> Fourier + calendar dummies + lags + rolls
Base models     additive | gbm | qgbm | esrnn  (all CPU)
   |
   v      inverse-MAE blend  +   NNLS stacking
AutoSelect
```

Everything communicates via the ForecastPipeline interface, so you can slot in custom models or swap the feature generator without touching the rest.

## Documentation & API

* Full docs: https://smurphcast.readthedocs.io
* Quick cheatsheet: docs/api.md

## Sample data

The wheel ships with tiny toy CSVs (`smurphcast.data.*`) so you can run the docs offline. Larger examples stay in `examples/` to keep installs lean.

## Roadmap 🗺️

* Optuna integration for AutoSelector hyper-tuning
* Holiday / event regressor interface
* Group-by support for many related series (panel KPIs)
* Prophet-style component plots on every model

## Contributing

* Fork & create a feature branch
* Hatch run pytest - all tests must pass
* Follow ruff & black (pre-commit hooks included)
* Open a PR – descriptive title, before/after numbers if performance related

We happily accept new feature generators, models, and docs!

## License

Code released under the MIT License.
Sample data are synthetic and MIT-licensed as well.

© 2025 SmurphCast Contributors – built with 💻, 📊 and a bit of 💙 for tiny percentages.