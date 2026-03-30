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


def robust_minmax(arr: np.ndarray, p_min=5.0, p_max=99.5, axis=None) -> np.ndarray:
    """
    Normalizes using percentiles to ignore outliers, supporting axis-wise scaling.

    Args:
        arr: Input array to normalize.
        p_min: Lower percentile.
        p_max: Upper percentile.
        axis: Axis or axes along which to compute percentiles.
              e.g., (1, 2) for per-channel scaling on (C, H, W).
    """
    # 1. Compute percentiles. keepdims=True is crucial for broadcasting!
    lo = np.percentile(arr, p_min, axis=axis, keepdims=True)
    hi = np.percentile(arr, p_max, axis=axis, keepdims=True)

    diff = hi - lo

    # 2. Handle the case where the range is near zero to avoid DivisionByZero
    # We use np.where to keep the logic vectorized
    normalized = np.where(diff < 1e-8, 0.0, (arr - lo) / diff)

    # 3. Clip and ensure float32
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


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


def get_n_splits(mat_reader: MatReader, max_splits: int = 10) -> int:
    """
    Computes the smallest number of unique patients among all classes.
    Useful for determining the maximum number of splits in a patient-wise cross-validation.
    Args:
        mat_reader: Loaded MatReader instance containing patient IDs and class labels.
        max_splits: An upper limit on the number of splits to prevent excessive computation.
    Returns:
        The maximum number of splits that can be performed without repeating patients in the same split.
    """
    patients = {}  # class_name -> list of unique patient IDs
    min_patients = 9999
    for i, class_name in enumerate(CLASS_NAMES):
        class_indices = np.where(mat_reader.class_labels == i)
        patient_ids = mat_reader.patient_ids[class_indices]
        unique_patient_ids, _count = np.unique(patient_ids, return_counts=True)
        patients[class_name] = unique_patient_ids
        min_patients = min(min_patients, unique_patient_ids.shape[0])

    n_splits = min(min_patients, max_splits)
    return n_splits
