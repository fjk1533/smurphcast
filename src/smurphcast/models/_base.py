from abc import ABC, abstractmethod

class ForecastModel(ABC):
    """Common interface every model must expose."""

    @abstractmethod
    def fit(self, df, y_col: str = "y", **kwargs):
        ...

    @abstractmethod
    def predict(self, future_df, **kwargs):
        ...

    @property
    def requires_regular_index(self) -> bool:
        """Override if model insists on equally spaced timesteps."""
        return False
