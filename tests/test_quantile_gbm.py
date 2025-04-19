import pandas as pd, numpy as np
from smurphcast.pipeline import ForecastPipeline

def test_qgbm_interval():
    dates = pd.date_range("2022-01-01", periods=90, freq="D")
    y = 0.04 + 0.01 * np.sin(np.arange(90) * 2 * np.pi / 30)
    df = pd.DataFrame({"ds": dates, "y": y})
    pipe = ForecastPipeline(model_name="qgbm").fit(df, horizon=5)
    ci = pipe.predict_interval()
    assert {"lower", "median", "upper"} <= set(ci.columns)
    assert len(ci) == 5
    # lower < median < upper
    assert ((ci["lower"] < ci["median"]) & (ci["median"] < ci["upper"])).all()
