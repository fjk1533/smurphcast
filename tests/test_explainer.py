import pandas as pd, numpy as np
from smurphcast.pipeline import ForecastPipeline
from smurphcast.explain.explainer import ForecastExplainer

def test_explainer_feature_importance():
    dates = pd.date_range("2020-01-01", periods=60, freq="D")
    y = 0.03 + 0.01 * np.sin(np.arange(60) * 2 * np.pi / 30)
    df = pd.DataFrame({"ds": dates, "y": y})
    pipe = ForecastPipeline(model_name="additive").fit(df, horizon=4)
    expl = ForecastExplainer(pipe._model)
    fi = expl.feature_importance()
    assert not fi.empty and fi.index[0] != ""  # has some feature names
