from dataclasses import dataclass
from .preprocessing.validator import validate_series
from .preprocessing.transform import auto_transform
from .features.time_feats import make_time_features
from .evaluation.backtest import rolling_backtest
from .models import additive, gbm, beta_rnn

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
        X = make_time_features(df)

        # choose model
        model_cls = AVAILABLE_MODELS[self.model_name]
        self.model = model_cls(**kwargs)
        self.model.fit(X, y_col="y_transformed")

        self.meta = meta
        self.horizon = horizon
        return self

    def predict(self):
        future = self.meta.make_future_df(self.horizon)
        future_feats = make_time_features(future)
        preds = self.model.predict(future_feats)
        return self.meta.inverse_transform(preds)
