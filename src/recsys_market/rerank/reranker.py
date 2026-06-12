"""Engagement reranker MLP."""
from __future__ import annotations

import torch
import torch.nn as nn


class EngagementReranker(nn.Module):
    """Predicts click probability from (user_emb, creator_emb, quality, bait).

    Trained via binary cross-entropy on click labels. Promotes content that
    historically got clicks — this creates the feedback loop that
    concentrates exposure under M2 (H4 hypothesis).

    Input: cat(user_emb [E], creator_emb [E], quality [1], bait [1]) = 2E+2
    Output: engagement logit (scalar, not sigmoid — use BCE with logits loss)
    """

    def __init__(self, embed_dim: int = 32, hidden_dim: int = 64) -> None:
        super().__init__()
        in_dim = embed_dim * 2 + 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(
        self, user_emb: torch.Tensor, creator_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        user_emb:     (*, E)
        creator_feat: (*, E+2)  — cat(creator_emb, quality, bait)
        → (*,) logit
        """
        x = torch.cat([user_emb, creator_feat], dim=-1)
        return self.net(x).squeeze(-1)
