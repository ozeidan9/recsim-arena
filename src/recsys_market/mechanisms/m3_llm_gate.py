"""M3: Two-stage pipeline + LLM quality gate.

M3 extends M2 (TwoStageMechanism) with a quality gate inserted between the
retrieval and reranking stages:

  Stage 1 (Retrieval):  Two-tower FAISS → top-R candidates
  Stage 2 (Quality gate): Filter candidates to those passing quality check
  Stage 3 (Reranking):  Engagement reranker on surviving candidates → top-k

The gate is an object with a `.score(quality, bait) → np.ndarray` method and a
`.threshold` attribute. Concrete implementations:
  - SimpleLinearGate  (analytical, no API cost, used by default)
  - DistilledGate     (MLP distilled from LLM, cheap inference)
  - LLMQualityGate    (live Anthropic API, expensive — use for distillation only)

H2 prediction: compared to M1, M3 should shift creator equilibrium toward
lower bait and higher quality — the gate removes the bait shortcut that M1
promotes but punishes under a quality-aware filter.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

from recsys_market.mechanisms.m2_two_stage import TwoStageMechanism, _retrieve_top_r


@runtime_checkable
class QualityGate(Protocol):
    """Duck-type interface for quality gates."""

    threshold: float

    def score(self, quality: np.ndarray, bait: np.ndarray) -> np.ndarray:
        """Return quality scores ∈ [0, 1] for M items."""
        ...


class LLMGateMechanism(TwoStageMechanism):
    """M3: two-stage retrieval + quality gate + engagement reranker.

    Parameters
    ----------
    gate:           Any object with `.score(quality, bait) → np.ndarray` and
                    a `.threshold` float (e.g. SimpleLinearGate, DistilledGate).
    gate_min_slots: Minimum number of gate-survivors used for reranking. If
                    fewer pass the gate, the top-scoring gate-rejects are added
                    until this many candidates are available. Defaults to
                    slate_size, ensuring the slate is always full.
    All other params are forwarded to TwoStageMechanism.
    """

    def __init__(
        self,
        *args: Any,
        gate: QualityGate,
        gate_min_slots: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.gate = gate
        self._gate_min_slots = gate_min_slots  # resolved to slate_size at call time

    def recommend(
        self,
        user_preferences: np.ndarray,
        creator_contents: np.ndarray,
        creator_quality: np.ndarray,
        creator_bait: np.ndarray,
        slate_size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        N = user_preferences.shape[0]
        M = creator_contents.shape[0]
        R = min(self.retrieval_size, M)
        min_slots = self._gate_min_slots if self._gate_min_slots is not None else slate_size

        # Score ALL creators once (cheap — quality/bait are just scalars)
        gate_scores = self.gate.score(creator_quality, creator_bait)  # (M,)

        with torch.no_grad():
            user_t = torch.FloatTensor(user_preferences)
            content_t = torch.FloatTensor(creator_contents)
            quality_t = torch.FloatTensor(creator_quality)
            bait_t = torch.FloatTensor(creator_bait)
            user_embs = self._encoder.encode_users(user_t)
            creator_embs = self._encoder.encode_creators(content_t)

        # Stage 1: inner-product retrieval
        candidates = _retrieve_top_r(
            user_embs.numpy(), creator_embs.numpy(), R
        )  # (N, R)

        slates = np.zeros((N, slate_size), dtype=np.int32)
        with torch.no_grad():
            for i in range(N):
                cand = candidates[i]  # (R,)

                # Stage 2: quality gate — split into passed / failed
                cand_gate = gate_scores[cand]
                passed_mask = cand_gate >= self.gate.threshold
                passed = cand[passed_mask]
                failed = cand[~passed_mask]

                # If too few pass, promote top gate-rejects (sorted by gate score)
                if len(passed) < min_slots:
                    n_extra = min_slots - len(passed)
                    extra_order = np.argsort(gate_scores[failed])[::-1][:n_extra]
                    passed = np.concatenate([passed, failed[extra_order]])

                # Stage 3: rerank survivors by engagement model
                u = user_embs[i : i + 1].expand(len(passed), -1)
                c = creator_embs[passed]
                q = quality_t[passed].unsqueeze(1)
                b = bait_t[passed].unsqueeze(1)
                c_feat = torch.cat([c, q, b], dim=1)
                scores = self._reranker(u, c_feat)

                k = min(slate_size, len(passed))
                top_local = scores.topk(k).indices.numpy()
                slates[i, :k] = passed[top_local]

        return slates
