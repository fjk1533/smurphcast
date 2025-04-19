"""
smurphcast.preprocessing
========================

Convenience exports for common preprocessing helpers.
"""

from .validator import validate_series
from .transform import auto_transform
from .outlier import OutlierHandler

__all__ = [
    "validate_series",
    "auto_transform",
    "OutlierHandler",
]
