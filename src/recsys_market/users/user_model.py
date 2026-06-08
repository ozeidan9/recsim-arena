from __future__ import annotations

import numpy as np


class UserPool:
    """N users with latent preference vectors drawn from a mixture of Gaussians.

    Preferences are fixed across an episode; only fatigue state resets.
    """

    def __init__(
        self,
        n_users: int,
        n_populations: int,
        content_dim: int,
        seed: int = 0,
    ) -> None:
        self.n_users = n_users
        self.n_populations = n_populations
        self.content_dim = content_dim

        rng = np.random.default_rng(seed)
        # Place population centres on unit sphere, well-separated
        centres = rng.standard_normal((n_populations, content_dim))
        centres /= np.linalg.norm(centres, axis=1, keepdims=True)

        assignments = rng.integers(0, n_populations, size=n_users)
        noise = rng.standard_normal((n_users, content_dim)) * 0.3
        prefs = centres[assignments] + noise
        prefs /= np.linalg.norm(prefs, axis=1, keepdims=True)

        self._preferences = prefs.astype(np.float32)
        self._population_labels = assignments
        self.fatigue = np.zeros(n_users, dtype=np.float32)

    @property
    def preferences(self) -> np.ndarray:
        return self._preferences

    @property
    def population_labels(self) -> np.ndarray:
        return self._population_labels

    def update_fatigue(self, bait_consumed: np.ndarray, gamma: float = 0.95) -> None:
        """Decay existing fatigue, then add new bait-driven fatigue.

        Args:
            bait_consumed: (N,) per-user bait exposure this round.
            gamma: exponential decay factor.
        """
        self.fatigue = gamma * self.fatigue + bait_consumed.astype(np.float32)

    def reset(self) -> None:
        """Reset per-episode state (fatigue). Preferences are stable."""
        self.fatigue[:] = 0.0
