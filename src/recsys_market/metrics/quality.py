from __future__ import annotations

import numpy as np


def mean_recommended_quality(slates: np.ndarray, qualities: np.ndarray) -> float:
    """Average quality of creators that actually appear in user slates.

    Args:
        slates:    (N, k) integer indices into creator dimension.
        qualities: (M,) per-creator quality values in [0, 1].

    Returns:
        Mean quality of recommended items, averaged over all (user, slot) pairs.
    """
    return float(qualities[slates].mean())


def quality_distribution(qualities: np.ndarray, bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Histogram of produced quality values.

    Args:
        qualities: (M,) per-creator quality values.
        bins: number of histogram bins.

    Returns:
        (counts, bin_edges) as returned by np.histogram.
    """
    return np.histogram(qualities, bins=bins, range=(0.0, 1.0))
