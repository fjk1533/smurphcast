# tests/test_metrics.py
from smurphcast.evaluation.metrics import mape, smape, coverage
import numpy as np

def test_basic_metrics():
    y = np.array([0.1, 0.2, 0.3])
    yhat = np.array([0.1, 0.22, 0.28])
    assert mape(y, yhat) < 0.2
    assert smape(y, yhat) < 0.2
    assert coverage(y, y - 0.05, y + 0.05) == 1.0
