"""Two-tower encoder for candidate retrieval."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoTowerEncoder(nn.Module):
    """User and creator towers that project to the same embedding space.

    Both outputs are L2-normalised, so dot product = cosine similarity.
    This enables FAISS inner-product search for retrieval.
    """

    def __init__(self, content_dim: int, embed_dim: int = 32) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.user_tower = nn.Sequential(
            nn.Linear(content_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
        )
        self.creator_tower = nn.Sequential(
            nn.Linear(content_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def encode_users(self, user_prefs: torch.Tensor) -> torch.Tensor:
        """(N, d) → (N, embed_dim), L2-normalised."""
        return F.normalize(self.user_tower(user_prefs), dim=-1)

    def encode_creators(self, contents: torch.Tensor) -> torch.Tensor:
        """(M, d) → (M, embed_dim), L2-normalised."""
        return F.normalize(self.creator_tower(contents), dim=-1)
