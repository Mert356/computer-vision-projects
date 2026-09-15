import numpy as np


# Calculate the percentage of the image covered by white mask pixels.
def coverage_percent(mask: np.ndarray) -> float:
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError("mask must be a non-empty 2D array.")

    return 100.0 * np.count_nonzero(mask) / mask.size


# Compare two masks using IoU; two empty foregrounds count as a perfect match.
def intersection_over_union(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    if prediction.shape != ground_truth.shape or prediction.ndim != 2 or prediction.size == 0:
        raise ValueError("Masks must be non-empty 2D arrays with matching shapes.")

    predicted_pixels = prediction > 0
    reference_pixels = ground_truth > 0

    intersection = np.count_nonzero(predicted_pixels & reference_pixels)
    union = np.count_nonzero(predicted_pixels | reference_pixels)

    return float(intersection / union) if union else 1.0
