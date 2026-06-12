"""Quality gate implementations: analytical, distilled, and LLM-backed.

Usage hierarchy (fastest → most accurate):
  1. SimpleLinearGate  — analytical sigmoid, no training, used as default
  2. DistilledGate     — small MLP trained on LLM scores, used in IPPO loops
  3. LLMQualityGate    — calls Anthropic API; use once for distillation labels

Typical workflow:
    llm = LLMQualityGate()
    distilled = llm.distill(n_samples=500)          # ~25 API calls of 20 items
    # Save distilled gate and use it in TwoStage+Gate experiments
"""
from __future__ import annotations

import json
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class SimpleLinearGate:
    """Analytical quality gate: score = sigmoid(alpha_q*quality + alpha_b*bait + bias).

    High quality → high score, high bait → low score. No training required.
    Used as the default gate in M3 experiments.
    """

    def __init__(
        self,
        alpha_quality: float = 3.0,
        alpha_bait: float = -3.0,
        bias: float = 0.0,
        threshold: float = 0.5,
    ) -> None:
        self.alpha_quality = alpha_quality
        self.alpha_bait = alpha_bait
        self.bias = bias
        self.threshold = threshold

    def score(self, quality: np.ndarray, bait: np.ndarray) -> np.ndarray:
        """Returns (M,) quality scores ∈ [0, 1]."""
        logit = self.alpha_quality * quality + self.alpha_bait * bait + self.bias
        return (1.0 / (1.0 + np.exp(-logit))).astype(np.float32)

    def passes(self, quality: np.ndarray, bait: np.ndarray) -> np.ndarray:
        """Returns (M,) boolean mask: True if item passes the gate."""
        return self.score(quality, bait) >= self.threshold


class DistilledGate(nn.Module):
    """Small MLP distilled from LLMQualityGate.

    Input: (quality, bait) → quality score ∈ [0, 1].
    Trained via MSE on LLM-generated labels using DistilledGate.fit().
    """

    def __init__(self, hidden_dim: int = 16, threshold: float = 0.5) -> None:
        super().__init__()
        self.threshold = threshold
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """(*, 2) → (*,) scores ∈ [0, 1]."""
        return self.net(features).squeeze(-1)

    def score(self, quality: np.ndarray, bait: np.ndarray) -> np.ndarray:
        """numpy interface for use in mechanisms."""
        with torch.no_grad():
            x = torch.tensor(
                np.stack([quality, bait], axis=-1), dtype=torch.float32
            )
            return self.forward(x).numpy()

    def passes(self, quality: np.ndarray, bait: np.ndarray) -> np.ndarray:
        return self.score(quality, bait) >= self.threshold

    @classmethod
    def fit(
        cls,
        quality: np.ndarray,
        bait: np.ndarray,
        target_scores: np.ndarray,
        n_epochs: int = 300,
        lr: float = 1e-3,
        hidden_dim: int = 16,
        threshold: float = 0.5,
    ) -> "DistilledGate":
        """Fit a DistilledGate to (quality, bait) → target_scores."""
        gate = cls(hidden_dim=hidden_dim, threshold=threshold)
        opt = Adam(gate.parameters(), lr=lr)
        x = torch.tensor(
            np.stack([quality, bait], axis=-1), dtype=torch.float32
        )
        y = torch.tensor(target_scores, dtype=torch.float32)
        for _ in range(n_epochs):
            pred = gate(x)
            loss = F.mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return gate

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.state_dict(), "threshold": self.threshold}, path)

    @classmethod
    def load(cls, path: str) -> "DistilledGate":
        data = torch.load(path, weights_only=True)
        gate = cls(threshold=data["threshold"])
        gate.load_state_dict(data["state_dict"])
        return gate


class LLMQualityGate:
    """Calls Anthropic API to score content quality.

    Designed for offline distillation: generate labels once, then train a
    DistilledGate for fast inference in IPPO training loops.

    The LLM scores based on quality and bait scalars — in a real deployment
    these would be replaced by content text signals.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        threshold: float = 0.5,
    ) -> None:
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.threshold = threshold

    def score_batch(self, quality: np.ndarray, bait: np.ndarray) -> np.ndarray:
        """Score M items via one API call. Returns (M,) float array ∈ [0, 1]."""
        prompt = self._build_prompt(quality, bait)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return self._parse_scores(response.content[0].text, len(quality))

    def _build_prompt(self, quality: np.ndarray, bait: np.ndarray) -> str:
        items = "\n".join(
            f"  Item {i + 1}: quality_effort={q:.3f}, sensationalism={b:.3f}"
            for i, (q, b) in enumerate(zip(quality, bait))
        )
        n = len(quality)
        return f"""You are a content quality auditor for a recommendation platform.

Rate each content item on an overall quality scale from 0.0 (very low) to 1.0 (very high).

Guidelines:
- quality_effort: creator's investment in substantive content (0=none, 1=maximum)
- sensationalism: creator's clickbait level (0=none, 1=maximum)
- High score (→1.0): high effort, low sensationalism
- Low score (→0.0): low effort, high sensationalism

Content items:
{items}

Respond with ONLY a JSON array of exactly {n} scores between 0.0 and 1.0:
[score1, score2, ...]"""

    def _parse_scores(self, text: str, n_items: int) -> np.ndarray:
        match = re.search(r"\[[\d.,\s]+\]", text)
        if match:
            try:
                scores = json.loads(match.group())
                arr = np.array(scores[:n_items], dtype=np.float32)
                if len(arr) == n_items:
                    return np.clip(arr, 0.0, 1.0)
            except (json.JSONDecodeError, ValueError):
                pass
        return np.full(n_items, 0.5, dtype=np.float32)

    def passes(self, quality: np.ndarray, bait: np.ndarray) -> np.ndarray:
        return self.score_batch(quality, bait) >= self.threshold

    def distill(
        self,
        n_samples: int = 500,
        chunk_size: int = 20,
        seed: int = 42,
        n_epochs: int = 300,
        threshold: float = 0.5,
    ) -> DistilledGate:
        """Query the LLM on random samples and fit a DistilledGate.

        Makes ceil(n_samples / chunk_size) API calls.
        """
        rng = np.random.default_rng(seed)
        quality = rng.uniform(0, 1, n_samples).astype(np.float32)
        bait = rng.uniform(0, 1, n_samples).astype(np.float32)

        scores: list[float] = []
        for i in range(0, n_samples, chunk_size):
            q_chunk = quality[i : i + chunk_size]
            b_chunk = bait[i : i + chunk_size]
            chunk_scores = self.score_batch(q_chunk, b_chunk)
            scores.extend(chunk_scores.tolist())

        return DistilledGate.fit(
            quality,
            bait,
            np.array(scores, dtype=np.float32),
            n_epochs=n_epochs,
            threshold=threshold,
        )
