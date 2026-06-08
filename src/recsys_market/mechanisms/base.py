from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Mechanism(ABC):
    """Abstract recommender mechanism.

    All mechanisms take the current market state and return slates of creator
    indices — one slate per user. This is the sole interface the environment
    uses, making mechanisms fully swappable.
    """

    @abstractmethod
    def recommend(
        self,
        user_preferences: np.ndarray,  # (N, d)
        creator_contents: np.ndarray,  # (M, d)
        creator_quality: np.ndarray,   # (M,)
        creator_bait: np.ndarray,      # (M,)
        slate_size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return slate indices for each user.

        Returns:
            slates: (N, slate_size) integer indices into the creator dimension.
        """
        ...
