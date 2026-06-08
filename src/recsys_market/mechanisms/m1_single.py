from __future__ import annotations

import numpy as np

from .base import Mechanism


class SingleStageMechanism(Mechanism):
    """Single-stage relevance ranking: score = dot(user_pref, content) / temperature.

    This is the standard literature mechanism (dot-product softmax) and the
    baseline for H1. With deterministic=True it returns the exact top-k; with
    deterministic=False it samples proportional to softmax scores.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        bait_weight: float = 0.5,
        deterministic: bool = True,
    ) -> None:
        self.temperature = temperature
        self.bait_weight = bait_weight   # > 0: M1 rewards bait (conflates engagement w/ relevance)
        self.deterministic = deterministic

    def recommend(
        self,
        user_preferences: np.ndarray,  # (N, d)
        creator_contents: np.ndarray,  # (M, d)
        creator_quality: np.ndarray,   # (M,)  unused by M1
        creator_bait: np.ndarray,      # (M,)  M1 promotes bait (engagement proxy)
        slate_size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        # scores[i, j] = dot(pref_i, content_j) / temperature + bait_weight * bait_j
        scores = (user_preferences @ creator_contents.T) / self.temperature + self.bait_weight * creator_bait  # (N, M)

        if self.deterministic:
            # Top-k by score for each user
            slates = np.argsort(scores, axis=1)[:, ::-1][:, :slate_size]
        else:
            # Softmax sampling without replacement (Gumbel-max trick)
            n_users, n_creators = scores.shape
            log_probs = scores - scores.max(axis=1, keepdims=True)  # numerical stability
            gumbel = rng.gumbel(size=(n_users, n_creators))
            perturbed = log_probs + gumbel
            slates = np.argsort(perturbed, axis=1)[:, ::-1][:, :slate_size]

        return slates.astype(np.int32)
