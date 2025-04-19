# 📂 SmurphCast sample data

This sub-package ships **tiny, synthetic datasets** that install **inside the wheel** so new users can run examples instantly—even offline.

| File | Frequency | Metric (range) | Rows |
|------|-----------|----------------|------|
| `data_weekly.csv` | Weekly | Random % (0.02 - 0.06) | 104 |
| `churn_example.csv` | Monthly | Simulated churn rate | 36 |

All follow the familiar **Prophet** schema:

```csv
ds,y
2023-01-01,0.0412
2023-02-01,0.0387
...
```

- `ds` – ISO-8601 timestamp
- `y` – percentage in decimal form (0.0 – 1.0)

## Loading programmatically

```python
import pandas as pd
from importlib.resources import files
import smurphcast  # top-level package

path = files(smurphcast.data).joinpath("data_weekly.csv")
df = pd.read_csv(path, parse_dates=["ds"])
print(df.head())
```

The importlib `resources` API works the same whether SmurphCast is installed from PyPI or cloned locally.

## Why bundle data?

- Quick smoke-tests (`pip install smurphcast && python -m smurphcast ...`)
- User guides & documentation examples
- Keeps the main wheel small (≈ 10 kB combined)