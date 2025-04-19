import pandas as pd
import numpy as np
from scipy.special import expit

from smurphcast.models.gbm import GBMModel
from smurphcast.preprocessing.transform import auto_transform


def test_gbm_fit_predict():
    dates = pd.date_range("2021-01-01", periods=120, freq="D")
    y = 0.03 + 0.005 * np.sin(np.arange(120) * 2 * np.pi / 30) + np.random.normal(0, 0.001, 120)
    df = pd.DataFrame({"ds": dates, "y": np.clip(y, 0.01, 0.06)})
    df, meta = auto_transform(df)

    m = GBMModel(lags=(1, 7), rolls=(7,))
    m.fit(df)
    preds_trans = m.predict(df.tail(7))
    preds = meta.inverse_transform(preds_trans)

    assert len(preds) == 7
    assert (preds > 0).all() and (preds < 1).all()
