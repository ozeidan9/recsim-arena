"""M2: Two-stage pipeline — FAISS retrieval + learned engagement reranker.

Architecture:
  Stage 1 (Retrieval): Two-tower encoder embeds users and creators into a
    shared L2-normalised space; FAISS inner-product search retrieves the top-R
    candidates for each user.

  Stage 2 (Reranking): An engagement MLP reranks the R candidates by predicted
    click probability, returning the top-k slate.

Online learning: the mechanism maintains a replay buffer of (user_pref,
creator_content, quality, bait, click=0/1) tuples. Every `update_every` calls
to `update()` it runs a few gradient steps on the encoder+reranker via BCE loss.

This creates the feedback loop central to H4: creators who attract early clicks
are promoted more heavily, generating more clicks — a rich-get-richer dynamic
that concentrates exposure beyond what M1 produces.

The env calls `mechanism.update()` after each round (see market_env.py).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from recsys_market.mechanisms.base import Mechanism
from recsys_market.rerank.reranker import EngagementReranker
from recsys_market.retrieval.encoder import TwoTowerEncoder


def _retrieve_top_r(
    user_embs: np.ndarray,  # (N, E) float32, L2-normalised
    creator_embs: np.ndarray,  # (M, E) float32, L2-normalised
    R: int,
) -> np.ndarray:
    """Return top-R candidate indices per user via inner-product similarity.

    Uses numpy for correctness and zero-crash behaviour on all platforms.
    For M >> 1000, swap this for a FAISS IndexFlatIP index.
    """
    sim = user_embs @ creator_embs.T  # (N, M)
    return np.argsort(sim, axis=1)[:, ::-1][:, :R].astype(np.int64)  # (N, R)


class TwoStageMechanism(Mechanism):
    """Two-stage retrieval + engagement reranker (M2).

    Parameters
    ----------
    content_dim:    Dimension of content/preference vectors (must match env).
    retrieval_size: Number of candidates retrieved in stage 1 (R).
                    With M=20 creators and R=10, half the creators compete per user.
    embed_dim:      Dimension of the shared embedding space.
    lr:             Learning rate for encoder + reranker joint optimiser.
    update_every:   Retrain every this many calls to `update()`.
    min_buffer:     Minimum replay buffer size before training starts.
    n_train_steps:  Gradient steps per training event.
    batch_size:     Minibatch size for each gradient step.
    buffer_maxlen:  Cap on replay buffer size.
    """

    def __init__(
        self,
        content_dim: int,
        retrieval_size: int = 10,
        embed_dim: int = 32,
        lr: float = 1e-3,
        update_every: int = 5,
        min_buffer: int = 200,
        n_train_steps: int = 5,
        batch_size: int = 256,
        buffer_maxlen: int = 10_000,
    ) -> None:
        self.content_dim = content_dim
        self.retrieval_size = retrieval_size
        self.update_every = update_every
        self.min_buffer = min_buffer
        self.n_train_steps = n_train_steps
        self.batch_size = batch_size
        self.buffer_maxlen = buffer_maxlen

        self._encoder = TwoTowerEncoder(content_dim, embed_dim)
        self._reranker = EngagementReranker(embed_dim, hidden_dim=64)
        self._opt = Adam(
            list(self._encoder.parameters()) + list(self._reranker.parameters()),
            lr=lr,
        )

        # Replay buffer: list of (user_pref, creator_content, quality, bait, label)
        self._buffer: list[tuple[np.ndarray, np.ndarray, float, float, float]] = []
        self._update_step: int = 0

    # ------------------------------------------------------------------
    # Mechanism interface
    # ------------------------------------------------------------------

    def recommend(
        self,
        user_preferences: np.ndarray,  # (N, d)
        creator_contents: np.ndarray,  # (M, d)
        creator_quality: np.ndarray,   # (M,)
        creator_bait: np.ndarray,      # (M,)
        slate_size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        N = user_preferences.shape[0]
        M = creator_contents.shape[0]
        R = min(self.retrieval_size, M)

        with torch.no_grad():
            user_t = torch.FloatTensor(user_preferences)
            content_t = torch.FloatTensor(creator_contents)
            quality_t = torch.FloatTensor(creator_quality)
            bait_t = torch.FloatTensor(creator_bait)

            user_embs = self._encoder.encode_users(user_t)          # (N, E)
            creator_embs = self._encoder.encode_creators(content_t) # (M, E)

        # Stage 1: inner-product retrieval (numpy, equivalent to FAISS IndexFlatIP)
        candidates = _retrieve_top_r(
            user_embs.numpy(), creator_embs.numpy(), R
        )  # (N, R)

        # Stage 2: rerank each user's candidates by engagement model
        slates = np.zeros((N, slate_size), dtype=np.int32)
        with torch.no_grad():
            for i in range(N):
                cand = candidates[i]               # (R,)
                u = user_embs[i : i + 1].expand(R, -1)  # (R, E)
                c = creator_embs[cand]             # (R, E)
                q = quality_t[cand].unsqueeze(1)   # (R, 1)
                b = bait_t[cand].unsqueeze(1)      # (R, 1)
                c_feat = torch.cat([c, q, b], dim=1)  # (R, E+2)
                scores = self._reranker(u, c_feat) # (R,)
                k = min(slate_size, R)
                top_local = scores.topk(k).indices.numpy()
                slates[i, :k] = cand[top_local]

        return slates

    # ------------------------------------------------------------------
    # Online learning
    # ------------------------------------------------------------------

    def update(
        self,
        slates: np.ndarray,             # (N, k)
        chosen_creators: np.ndarray,    # (N,) clicked creator index per user
        user_preferences: np.ndarray,   # (N, d)
        creator_contents: np.ndarray,   # (M, d)
        creator_quality: np.ndarray,    # (M,)
        creator_bait: np.ndarray,       # (M,)
    ) -> None:
        """Store click feedback and periodically retrain encoder+reranker."""
        N = user_preferences.shape[0]
        for i in range(N):
            for j in slates[i]:
                self._buffer.append((
                    user_preferences[i].copy(),
                    creator_contents[j].copy(),
                    float(creator_quality[j]),
                    float(creator_bait[j]),
                    1.0 if j == chosen_creators[i] else 0.0,
                ))

        if len(self._buffer) > self.buffer_maxlen:
            self._buffer = self._buffer[-self.buffer_maxlen :]

        self._update_step += 1
        if (
            self._update_step % self.update_every == 0
            and len(self._buffer) >= self.min_buffer
        ):
            self._train_step()

    def _train_step(self) -> None:
        buf = self._buffer
        n = len(buf)

        for _ in range(self.n_train_steps):
            idx = np.random.choice(n, min(self.batch_size, n), replace=False)

            user_prefs = torch.tensor(
                np.stack([buf[i][0] for i in idx]), dtype=torch.float32
            )
            creator_contents = torch.tensor(
                np.stack([buf[i][1] for i in idx]), dtype=torch.float32
            )
            qs = torch.tensor(
                [buf[i][2] for i in idx], dtype=torch.float32
            ).unsqueeze(1)
            bs = torch.tensor(
                [buf[i][3] for i in idx], dtype=torch.float32
            ).unsqueeze(1)
            labels = torch.tensor(
                [buf[i][4] for i in idx], dtype=torch.float32
            )

            # Re-encode using current encoder (buffer stores raw features)
            u_embs = self._encoder.encode_users(user_prefs)
            c_embs = self._encoder.encode_creators(creator_contents)
            c_feat = torch.cat([c_embs, qs, bs], dim=1)

            scores = self._reranker(u_embs, c_feat)
            loss = F.binary_cross_entropy_with_logits(scores, labels)

            self._opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self._encoder.parameters()) + list(self._reranker.parameters()),
                max_norm=1.0,
            )
            self._opt.step()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(
            {
                "encoder": self._encoder.state_dict(),
                "reranker": self._reranker.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        data = torch.load(path, weights_only=True)
        self._encoder.load_state_dict(data["encoder"])
        self._reranker.load_state_dict(data["reranker"])

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def buffer_size(self) -> int:
        return len(self._buffer)

    def reranker_loss(self, n_samples: int = 512) -> float:
        """Compute BCE loss on a random buffer sample (diagnostic)."""
        if len(self._buffer) < 10:
            return float("nan")
        buf = self._buffer
        idx = np.random.choice(len(buf), min(n_samples, len(buf)), replace=False)
        with torch.no_grad():
            u = torch.tensor(np.stack([buf[i][0] for i in idx]), dtype=torch.float32)
            c = torch.tensor(np.stack([buf[i][1] for i in idx]), dtype=torch.float32)
            qs = torch.tensor([buf[i][2] for i in idx], dtype=torch.float32).unsqueeze(1)
            bs = torch.tensor([buf[i][3] for i in idx], dtype=torch.float32).unsqueeze(1)
            labels = torch.tensor([buf[i][4] for i in idx], dtype=torch.float32)
            u_embs = self._encoder.encode_users(u)
            c_embs = self._encoder.encode_creators(c)
            c_feat = torch.cat([c_embs, qs, bs], dim=1)
            scores = self._reranker(u_embs, c_feat)
            return F.binary_cross_entropy_with_logits(scores, labels).item()
