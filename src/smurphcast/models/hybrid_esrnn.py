"""
HybridESRNNModel
================

A minimal reproduction of the M4‑winning ES‑RNN idea:

1. Fit an additive **Exponential Smoothing** (ETS) model to capture
   level + seasonality (via statsmodels).
2. Train a small **LSTM** on the *residuals* to learn non‑linear patterns.
3. Forecast horizon = ETS forecast + LSTM residual forecast.

Design choices
--------------
* Seasonality period is user‑supplied (`season_length`) with default 7
  (weekly seasonality for daily data).  Set to 0 to disable seasonality.
* RNN uses a fixed look‑back window (`look_back`) and predicts one step ahead
  recursively.
* Training is kept short (`epochs=50`) so unit tests stay fast.

This is *not* a fully tuned production‑grade ES‑RNN, but provides a real,
working hybrid model that can be extended.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class _ResidualLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]              # only last step
        return self.fc(out)


class HybridESRNNModel:
    """
    Parameters
    ----------
    season_length : int
        Seasonal period for ETS (0 disables seasonality).
    look_back : int
        How many residual lags feed the LSTM.
    hidden_size : int
        LSTM hidden dimension.
    epochs : int
        Number of training epochs (kept small for speed).
    lr : float
        Learning rate.
    device : str
        "cpu" or "cuda".
    """

    def __init__(
        self,
        *,
        season_length: int = 7,
        look_back: int = 14,
        hidden_size: int = 32,
        epochs: int = 50,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        self.season_length = season_length
        self.look_back = look_back
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        self.device = device

        self.ets = None
        self.rnn: _ResidualLSTM | None = None
        self._residuals: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    def _build_rnn_dataset(self, residuals: np.ndarray):
        X, y = [], []
        for i in range(len(residuals) - self.look_back):
            X.append(residuals[i : i + self.look_back])
            y.append(residuals[i + self.look_back])
        X = np.expand_dims(np.array(X, dtype=np.float32), axis=-1)
        y = np.array(y, dtype=np.float32).reshape(-1, 1)
        return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))

    # ------------------------------------------------------------------ #
    def fit(self, df: pd.DataFrame, y_col: str = "y_transformed"):
        y = df[y_col].values.astype(float)

        # 1. Exponential Smoothing
        seasonal = "add" if self.season_length > 1 else None
        self.ets = ExponentialSmoothing(
            y,
            seasonal=seasonal,
            seasonal_periods=self.season_length or None,
            initialization_method="estimated",
        ).fit(optimized=True)

        fitted = self.ets.fittedvalues
        residuals = y - fitted
        self._residuals = residuals

        # 2. LSTM on residuals
        dataset = self._build_rnn_dataset(residuals)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        self.rnn = _ResidualLSTM(1, self.hidden_size).to(self.device)
        optim = torch.optim.Adam(self.rnn.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self.rnn.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred = self.rnn(xb)
                loss = loss_fn(pred, yb)
                optim.zero_grad()
                loss.backward()
                optim.step()
        return self

    # ------------------------------------------------------------------ #
    def _forecast_residuals(self, horizon: int) -> np.ndarray:
        """Recursive one‑step forecasting of residuals with LSTM."""
        history = self._residuals[-self.look_back :].tolist()
        preds = []
        self.rnn.eval()
        for _ in range(horizon):
            seq = torch.tensor(history[-self.look_back :], dtype=torch.float32).view(
                1, self.look_back, 1
            )
            with torch.no_grad():
                next_res = self.rnn(seq).item()
            preds.append(next_res)
            history.append(next_res)
        return np.array(preds)

    # ------------------------------------------------------------------ #
    def predict(self, future_df: pd.DataFrame):
        horizon = len(future_df)
        # ETS forecast
        ets_fcst = self.ets.forecast(horizon)
        # residual forecast
        res_fcst = self._forecast_residuals(horizon)
        return ets_fcst + res_fcst
