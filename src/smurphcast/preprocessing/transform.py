import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class TransformMeta:
    """Stores info to invert transformations later."""
    y_col: str
    y_transformed: str

    def inverse_transform(self, y_arr):
        """sigmoid inverse (logit -> (0,1))."""
        import scipy.special as sc
        return sc.expit(y_arr)           # vectorised


def auto_transform(df: pd.DataFrame, y_col: str = "y") -> tuple[pd.DataFrame, TransformMeta]:
    """
    Applies logit transform so that linear/GBM/RNN models can operate on ℝ.
    Saves metadata so predictions can be mapped back.
    """
    df = df.copy()
    eps = 1e-6
    df["y_transformed"] = np.log(df[y_col] / (1 - df[y_col]))       # logit
    return df, TransformMeta(y_col=y_col, y_transformed="y_transformed")
