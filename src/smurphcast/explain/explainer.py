"""
Explainer
=========

Pulls native feature importances from additive (Ridge) or GBM models.
Also offers a crude 1‑step contribution breakdown useful for quick EDA.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ForecastExplainer:
    def __init__(self, fitted_model):
        self.model = fitted_model

    # ------------------------------------------------------------------ #
    def feature_importance(self) -> pd.Series:
        """
        Returns a pd.Series sorted descending by absolute importance.
        Works for:
          • AdditiveModel (coef_)
          • GBMModel      (LightGBM feature_importance)
        """
        if hasattr(self.model, "reg"):                     # Additive
            coef = self.model.reg.coef_
            names = self.model._feature_cols
        elif hasattr(self.model, "model"):                 # GBM
            gain = self.model.model.feature_importance(importance_type="gain")
            names = self.model.model.feature_name()
            coef = gain
        else:
            raise NotImplementedError("Model type unsupported for importance.")

        sr = pd.Series(coef, index=names).sort_values(key=np.abs, ascending=False)
        return sr

    # ------------------------------------------------------------------ #
    def contribution_plot(self, k: int = 10):
        """
        Bar‑chart of top‑k influential features.
        """
        fi = self.feature_importance().head(k)[::-1]   # small→big so plot is nice
        fi.plot(kind="barh")
        plt.title("Top feature importances")
        plt.tight_layout()
        plt.show()
