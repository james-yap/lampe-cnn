"""
Various utility functions for preprocessing and data normalization.
"""

import numpy as np


def minmax(arr: np.ndarray) -> np.ndarray:
    """
    Simple min-max normalization to [0.0, 1.0].
    Note: This is sensitive to hot pixels,
    so the _robust_minmax version is usually better for visualisation.
    """
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def robust_minmax(arr: np.ndarray, p_min=5.0, p_max=99.5) -> np.ndarray:
    """
    Normalizes using percentiles to ignore hot-pixel outliers.
    Args:
        arr: Input array to normalize.
        p_min: Lower percentile for robust min.
                Typically set to 10.0 or 5.0 to ignore low-end noise.
        p_max: Upper percentile for robust max.
                Typically set to 99.5 or 99.9 to ignore high-end outliers.
    Returns:
        Normalized array with values clipped to [0.0, 1.0].
    """
    lo = float(np.percentile(arr, p_min))
    hi = float(np.percentile(arr, p_max))
    if hi - lo < 1e-8:
        return np.zeros_like(arr)

    # Normalize and clip so values stay strictly between 0.0 and 1.0
    normalized = (arr - lo) / (hi - lo)
    return np.clip(normalized, 0.0, 1.0)
