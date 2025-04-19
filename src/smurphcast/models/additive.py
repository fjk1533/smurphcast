import pandas as pd
from sklearn.linear_model import Ridge
from ..features.time_feats import make_time_features
from ..preprocessing.transform import TransformMeta


class AdditiveModel:
    """
    Fourier + DoW dummies fed into a simple Ridge regression.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.reg = Ridge(alpha=self.alpha)
        self.meta: TransformMeta | None = None
        self.last_train_date = None

    # ----------------------------------------------------------- #
    def fit(self, df: pd.DataFrame, y_col="y_transformed"):
        X = make_time_features(df)
        y = df[y_col].values
        self.reg.fit(X.drop(columns=["ds"]), y)
        self.last_train_date = df["ds"].iloc[-1]
        return self

    # ----------------------------------------------------------- #
    def predict(self, future_df: pd.DataFrame):
        Xf = make_time_features(future_df)
        return self.reg.predict(Xf.drop(columns=["ds"]))
