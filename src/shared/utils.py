"""
Various utility functions for preprocessing and data normalization.
"""

from collections import Counter

import numpy as np

from architectures.class_balanced import CLASS_NAMES
from shared.mat_reader import MatReader


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


def report_class_distribution(
    mat_reader: MatReader, eff_fov_indices: list[int], train: bool
) -> None:
    """
    Prints a class distribution report for the given FOV indices.
    """
    class_counts: Counter[int] = Counter(
        int(mat_reader.class_labels[i]) for i in eff_fov_indices
    )
    total_fovs = len(eff_fov_indices)
    split_label = "train" if train else "val"
    print(f"  [{split_label}] Class distribution ({total_fovs} FOVs):")
    for cls_idx, _ in enumerate(CLASS_NAMES):
        count = class_counts.get(cls_idx, 0)
        pct = 100.0 * count / total_fovs if total_fovs > 0 else 0.0
        print(f"    {CLASS_NAMES[cls_idx]:8s}: {count:4d} FOVs ({pct:.1f}%)")
