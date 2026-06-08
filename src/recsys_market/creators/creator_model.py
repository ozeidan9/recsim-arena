from __future__ import annotations

import numpy as np


class CreatorPool:
    """M creators, each producing content with a quality and engagement-bait level.

    Action layout (flat float32 vector of length content_dim + 2):
        [0 : content_dim]   — raw content direction (l2-normalised internally)
        [content_dim]       — quality  ∈ [0, 1] (costly to produce)
        [content_dim + 1]   — bait     ∈ [0, 1] (cheap; boosts short-term clicks)
    """

    def __init__(
        self,
        n_creators: int,
        content_dim: int,
        quality_cost_scale: float = 0.5,
        seed: int = 0,
    ) -> None:
        self.n_creators = n_creators
        self.content_dim = content_dim
        self.quality_cost_scale = quality_cost_scale

        rng = np.random.default_rng(seed)
        init = rng.standard_normal((n_creators, content_dim)).astype(np.float32)
        self._init_content = init / np.linalg.norm(init, axis=1, keepdims=True)

    @property
    def action_dim(self) -> int:
        return self.content_dim + 2

    def action_to_components(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split flat action array into (content, quality, bait).

        Args:
            actions: (M, content_dim + 2) float32 in [-1, 1].

        Returns:
            content: (M, content_dim) l2-normalised.
            quality: (M,) clipped to [0, 1].
            bait:    (M,) clipped to [0, 1].
        """
        raw_content = actions[:, : self.content_dim]
        norms = np.linalg.norm(raw_content, axis=1, keepdims=True)
        # Avoid division by zero for near-zero content vectors
        content = raw_content / np.where(norms < 1e-8, 1.0, norms)

        # Actions are in [-1, 1]; map to [0, 1] for quality / bait
        quality = np.clip((actions[:, self.content_dim] + 1.0) / 2.0, 0.0, 1.0)
        bait = np.clip((actions[:, self.content_dim + 1] + 1.0) / 2.0, 0.0, 1.0)
        return content.astype(np.float32), quality.astype(np.float32), bait.astype(np.float32)

    def production_cost(self, quality: np.ndarray) -> np.ndarray:
        """Quadratic cost of quality; bait is free to produce.

        Args:
            quality: (M,) in [0, 1].

        Returns:
            cost: (M,) non-negative.
        """
        return (self.quality_cost_scale * quality**2).astype(np.float32)

    def initial_actions(self) -> np.ndarray:
        """Return sensible initial actions (normalised content, zero quality/bait)."""
        zero_qb = np.zeros((self.n_creators, 2), dtype=np.float32)
        return np.concatenate([self._init_content, zero_qb], axis=1)
