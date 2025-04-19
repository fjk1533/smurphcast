import torch
import torch.nn as nn
import pandas as pd
from ..features.time_feats import make_time_features


class _TinyRNN(nn.Module):
    def __init__(self, num_feats, hidden=32):
        super().__init__()
        self.rnn = nn.GRU(num_feats, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        y = self.head(out[:, -1, :])            # last step
        return torch.sigmoid(y)                 # bound (0,1)


class BetaRNNModel:
    """
    Very small GRU that consumes [lags + calendar feats] and returns bounded preds.
    For demo purposes only (1‑series, CPU, no batching).
    """

    def __init__(self, epochs=200, lr=1e-2):
        self.epochs = epochs
        self.lr = lr
        self.net = None
        self.num_feats = None

    # ----------------------------------------------------------- #
    def _prep(self, df, lags=(1, 7)):
        from ..features.lag_feats import add_lag_features

        df = add_lag_features(df, lags)
        df = df.dropna().reset_index(drop=True)
        feats = make_time_features(df).drop(columns=["ds"])
        self.num_feats = feats.shape[1]
        X = torch.tensor(feats.values, dtype=torch.float32)
        y = torch.tensor(df["y_transformed"].values, dtype=torch.float32).unsqueeze(1)
        return X, y

    # ----------------------------------------------------------- #
    def fit(self, df: pd.DataFrame):
        X, y = self._prep(df)
        self.net = _TinyRNN(self.num_feats)
        optimiser = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        # simple training loop
        for _ in range(self.epochs):
            optimiser.zero_grad()
            preds = self.net(X.unsqueeze(0))          # batch dim=1
            loss = loss_fn(preds, y)
            loss.backward()
            optimiser.step()
        return self

    # ----------------------------------------------------------- #
    def predict(self, future_df: pd.DataFrame):
        feats = make_time_features(future_df).drop(columns=["ds"])
        Xf = torch.tensor(feats.values, dtype=torch.float32)
        with torch.no_grad():
            preds = self.net(Xf.unsqueeze(0)).squeeze().numpy()
        return preds
