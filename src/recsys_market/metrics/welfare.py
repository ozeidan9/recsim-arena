from __future__ import annotations

import numpy as np


def user_welfare(
    clicks: np.ndarray,
    fatigue_delta: np.ndarray,
    lambda_fatigue: float = 0.5,
) -> np.ndarray:
    """Per-user welfare: clicks received minus fatigue penalty.

    Args:
        clicks:        (N,) click counts this round.
        fatigue_delta: (N,) fatigue increase this round (bait consumed).
        lambda_fatigue: weight on the fatigue penalty.

    Returns:
        welfare: (N,) per-user welfare values.
    """
    return clicks.astype(np.float32) - lambda_fatigue * fatigue_delta.astype(np.float32)


def total_welfare(
    clicks: np.ndarray,
    fatigue_delta: np.ndarray,
    lambda_fatigue: float = 0.5,
) -> float:
    """Sum of user welfare across all users."""
    return float(user_welfare(clicks, fatigue_delta, lambda_fatigue).sum())
