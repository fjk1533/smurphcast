import pandas as pd
import lightgbm as lgb
from ..features.time_feats import make_time_features
from ..preprocessing.transform import TransformMeta


class GBMModel:
    """
    LightGBM wrapper with reasonable defaults for small tabular sets.
    """

    def __init__(self, num_leaves=31, n_estimators=200):
        self.params = dict(
            objective="regression",
            metric="l2",
            num_leaves=num_leaves,
            learning_rate=0.05,
        )
        self.n_estimators = n_estimators
        self.model = None

    # ----------------------------------------------------------- #
    def fit(self, df: pd.DataFrame, y_col="y_transformed"):
        X = make_time_features(df)
        y = df[y_col]
        dataset = lgb.Dataset(X.drop(columns=["ds"]), y)
        self.model = lgb.train(self.params, dataset, num_boost_round=self.n_estimators)
        return self

    # ----------------------------------------------------------- #
    def predict(self, future_df: pd.DataFrame):
        Xf = make_time_features(future_df)
        return self.model.predict(Xf.drop(columns=["ds"]))
