from __future__ import annotations

import numpy as np

from .base import Mechanism


class RandomMechanism(Mechanism):
    """Each user receives a uniformly random slate of creators (without replacement)."""

    def recommend(
        self,
        user_preferences: np.ndarray,
        creator_contents: np.ndarray,
        creator_quality: np.ndarray,
        creator_bait: np.ndarray,
        slate_size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        n_users = user_preferences.shape[0]
        n_creators = creator_contents.shape[0]
        slates = np.stack(
            [rng.choice(n_creators, size=slate_size, replace=False) for _ in range(n_users)]
        )
        return slates.astype(np.int32)


class PopularityMechanism(Mechanism):
    """Ranks creators by cumulative past exposure; top-k slate for all users.

    Exposure counts must be updated externally via `update_counts`.
    """

    def __init__(self) -> None:
        self._counts: np.ndarray | None = None

    def update_counts(self, exposures: np.ndarray) -> None:
        """Add new exposure counts (M,) to running totals."""
        if self._counts is None:
            self._counts = exposures.copy().astype(np.float64)
        else:
            self._counts += exposures

    def recommend(
        self,
        user_preferences: np.ndarray,
        creator_contents: np.ndarray,
        creator_quality: np.ndarray,
        creator_bait: np.ndarray,
        slate_size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        n_users = user_preferences.shape[0]
        n_creators = creator_contents.shape[0]

        if self._counts is None:
            counts = np.zeros(n_creators)
        else:
            counts = self._counts

        top_k = np.argsort(counts)[::-1][:slate_size]
        slates = np.tile(top_k, (n_users, 1))
        return slates.astype(np.int32)

    def reset(self) -> None:
        self._counts = None
