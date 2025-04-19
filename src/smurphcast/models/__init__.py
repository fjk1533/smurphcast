"""
smurphcast.models
=================

Public model registry.

Any *core* model class should be imported here so library users can do:

>>> from smurphcast.models import GBMModel, QuantileGBMModel
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, List

# Core models
from .additive import AdditiveModel
from .gbm import GBMModel
from .quantile_gbm import QuantileGBMModel
from .beta_rnn import BetaRNNModel
from .hybrid_esrnn import HybridESRNNModel

__all__: List[str] = [
    "AdditiveModel",
    "GBMModel",
    "QuantileGBMModel",
    "BetaRNNModel",
    "HybridESRNNModel",
]

# Optional: extend with community / contrib models *if present*.
# We *never* fail if the extra package is missing or poorly formed.
if not TYPE_CHECKING:
    try:
        contrib = import_module("smurphcast_contrib")
        # Prefer contrib.__all__ if defined, else export every public attr
        extra = getattr(
            contrib,
            "__all__",
            [name for name in dir(contrib) if not name.startswith("_")],
        )
        __all__.extend(extra)
    except ModuleNotFoundError:
        # contrib package not installed – silently ignore
        pass
    except Exception as exc:  # malformed contrib; log and continue
        import warnings

        warnings.warn(f"⚠️  smurphcast_contrib ignored: {exc}")
