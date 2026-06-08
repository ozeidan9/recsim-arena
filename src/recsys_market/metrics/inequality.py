from __future__ import annotations

import numpy as np


def gini(exposures: np.ndarray) -> float:
    """Gini coefficient of creator exposure counts.

    0 = perfectly equal; 1 = one creator gets everything.

    Args:
        exposures: (M,) non-negative exposure counts.

    Returns:
        Gini coefficient ∈ [0, 1].
    """
    x = np.sort(exposures.astype(float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * (index * x).sum()) / (n * x.sum()) - (n + 1) / n)


def exposure_distribution(exposures: np.ndarray) -> np.ndarray:
    """Return cumulative exposure shares sorted ascending (Lorenz curve data).

    Args:
        exposures: (M,) non-negative exposure counts.

    Returns:
        cumulative_shares: (M,) sorted cumulative share ∈ [0, 1].
    """
    x = np.sort(exposures.astype(float))
    total = x.sum()
    if total == 0:
        return np.zeros_like(x)
    return np.cumsum(x) / total
