import pandas as pd
import numpy as np
from smurphcast.pipeline import ForecastPipeline


def test_pipeline_smoke():
    # fake weekly percentage series
    dates = pd.date_range("2022-01-01", periods=100, freq="W-SAT")
    y = 0.02 + 0.01 * np.sin(np.arange(100) * 2 * np.pi / 13) + np.random.normal(0, 0.002, 100)
    df = pd.DataFrame({"ds": dates, "y": np.clip(y, 0.01, 0.04)})

    pipe = ForecastPipeline(model_name="additive").fit(df, horizon=4)
    fcst = pipe.predict()
    assert len(fcst) == 4
    assert (0 < fcst).all() and (fcst < 1).all()
