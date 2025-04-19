from dataclasses import dataclass
from .preprocessing.validator import validate_series
from .preprocessing.transform import auto_transform
from .features.time_feats import make_time_features
from .evaluation.backtest import rolling_backtest
from .models import additive, gbm, beta_rnn
from .features.time_feats import make_time_features   # needed for future_df creation

AVAILABLE_MODELS = {
    "additive": additive.AdditiveModel,
    "gbm": gbm.GBMModel,
    "beta_rnn": beta_rnn.BetaRNNModel,
}

@dataclass
class ForecastPipeline:
    model_name: str = "additive"

    def fit(self, df, horizon: int, **kwargs):
        df = validate_series(df)
        df, meta = auto_transform(df)
        self.meta_df = df
        X = make_time_features(df)

        # choose model
        model_cls = AVAILABLE_MODELS[self.model_name]
        self.model = model_cls(**kwargs)
        self.model.fit(X, y_col="y_transformed")

        self.meta = meta
        self.horizon = horizon
        return self

    def predict(self):
        # build future dates at same freq as training index
        last = self.meta_df["ds"].iloc[-1]
        freq = pd.infer_freq(self.meta_df["ds"])
        future_dates = pd.date_range(last, periods=self.horizon + 1, freq=freq, closed="right")
        future_df = pd.DataFrame({"ds": future_dates})

        preds_trans = self.model.predict(future_df)
        return self.meta.inverse_transform(preds_trans)
