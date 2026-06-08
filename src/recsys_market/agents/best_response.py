"""Gradient-ascent best-response dynamics baseline.

Each creator independently maximises their expected per-round payoff via
gradient ascent on a differentiable soft approximation of the click model.
This is a fast, analytical baseline that validates the IPPO findings are
not specific to the PPO learning algorithm.

Key difference from IPPO: no value function, no GAE, no clipping —
pure gradient ascent on the soft expected reward. Simpler and faster,
but sensitive to the soft approximation (uses softmax over all creators
rather than hard top-k).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from recsys_market.env.market_env import ContentMarketEnv


def _soft_expected_reward(
    action: torch.Tensor,    # (d+2,) for one creator
    others: torch.Tensor,    # (M-1, d+2) fixed other creators
    user_prefs: torch.Tensor,  # (N, d)
    fatigue: torch.Tensor,   # (N,)
    quality_cost_scale: float,
    alpha_quality: float,
    gamma_bait: float,
    beta_bait: float,
    bait_weight: float = 0.5,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Differentiable expected clicks minus cost for a single creator.

    Uses a single softmax over all creators (soft approximation combining the
    mechanism promotion and click-model attraction) so the objective is
    differentiable w.r.t. the creator's action.

    Logit = relevance/temp + bait_weight*bait (mechanism promotion)
                           + gamma_bait*bait  (short-term click attraction)
                           - beta_bait*bait*fatigue (fatigue penalty)
    """
    d = user_prefs.shape[1]

    # Parse own action
    raw_content = action[:d]
    norm = raw_content.norm().clamp(min=1e-8)
    content = raw_content / norm                           # (d,)
    quality = ((action[d] + 1.0) / 2.0).clamp(0.0, 1.0)
    bait = ((action[d + 1] + 1.0) / 2.0).clamp(0.0, 1.0)

    # Parse others
    others_content = others[:, :d]
    others_norms = others_content.norm(dim=1, keepdim=True).clamp(min=1e-8)
    others_content_n = others_content / others_norms      # (M-1, d)
    others_quality = ((others[:, d] + 1.0) / 2.0).clamp(0.0, 1.0)
    others_bait = ((others[:, d + 1] + 1.0) / 2.0).clamp(0.0, 1.0)

    # All contents: (M, d), all quality/bait: (M,)
    all_content = torch.cat([content.unsqueeze(0), others_content_n], dim=0)
    all_quality = torch.cat([quality.unsqueeze(0), others_quality], dim=0)
    all_bait = torch.cat([bait.unsqueeze(0), others_bait], dim=0)

    # Logits combining mechanism promotion and click-model terms: (N, M)
    relevance = user_prefs @ all_content.T              # (N, M)
    logits = (
        relevance / temperature
        + alpha_quality * all_quality
        + (bait_weight + gamma_bait) * all_bait
        - beta_bait * all_bait * fatigue.unsqueeze(1)
    )

    # Soft click probabilities: (N, M)
    click_probs = F.softmax(logits, dim=1)

    # Expected clicks for creator 0 (index 0 = self)
    expected_clicks = click_probs[:, 0].sum()

    # Cost
    cost = quality_cost_scale * quality**2

    return expected_clicks - cost


class GradientAscentDynamics:
    """Gradient-ascent best-response dynamics for all creators simultaneously.

    Each round of GRD:
    1. Each creator computes ∇_action E[clicks - cost] w.r.t. their own action,
       treating all other creators' actions as fixed.
    2. Update own action via gradient ascent.
    3. Clip actions to [-1, 1].

    Multiple GRD rounds ≈ one episode. Tracks the same metrics as IPPO for
    side-by-side comparison.
    """

    def __init__(
        self,
        env: ContentMarketEnv,
        lr: float = 0.05,
        n_steps_per_round: int = 5,
        temperature: float = 1.0,
        bait_weight: float = 0.5,
    ) -> None:
        self.env = env
        self.lr = lr
        self.n_steps = n_steps_per_round
        self.temperature = temperature
        self.bait_weight = bait_weight

        d = env.content_dim
        # Initialise with env's initial random content
        init = env._creator_pool.initial_actions()  # (M, d+2)
        self.actions = torch.FloatTensor(init)       # (M, d+2), requires_grad per-update

    def _update_all(self) -> None:
        """One round of simultaneous gradient-ascent updates for all creators."""
        user_prefs = torch.FloatTensor(self.env._user_pool.preferences)
        fatigue = torch.FloatTensor(self.env._user_pool.fatigue)
        M = self.env.n_creators

        new_actions = self.actions.clone()

        for j in range(M):
            others_idx = [i for i in range(M) if i != j]
            others = self.actions[others_idx].detach()

            action_j = new_actions[j].detach().requires_grad_(True)

            for _ in range(self.n_steps):
                reward = _soft_expected_reward(
                    action_j, others, user_prefs, fatigue,
                    quality_cost_scale=self.env._creator_pool.quality_cost_scale,
                    alpha_quality=self.env.alpha_quality,
                    gamma_bait=self.env.gamma_bait,
                    beta_bait=self.env.beta_bait,
                    bait_weight=self.bait_weight,
                    temperature=self.temperature,
                )
                reward.backward()
                with torch.no_grad():
                    action_j = (action_j + self.lr * action_j.grad).clamp(-1.0, 1.0)
                action_j = action_j.detach().requires_grad_(True)

            new_actions[j] = action_j.detach()

        self.actions = new_actions

    def _current_market_metrics(self) -> dict[str, Any]:
        from recsys_market.metrics.diversity import content_entropy
        from recsys_market.metrics.inequality import gini
        from recsys_market.mechanisms.m1_single import SingleStageMechanism

        actions_np = self.actions.numpy()
        contents, quality, bait = self.env._creator_pool.action_to_components(actions_np)

        # Fake exposure via soft click probabilities for Gini
        user_prefs = torch.FloatTensor(self.env._user_pool.preferences)
        fatigue = torch.FloatTensor(self.env._user_pool.fatigue)
        contents_t = torch.FloatTensor(contents)
        relevance = user_prefs @ contents_t.T
        probs = F.softmax(relevance, dim=1).numpy()
        exposure = probs.sum(axis=0)

        return {
            "diversity_entropy": float(content_entropy(contents, n_clusters=8)),
            "gini": float(gini(exposure)),
            "mean_quality": float(quality.mean()),
            "mean_bait": float(bait.mean()),
        }

    def run(
        self,
        n_rounds: int,
        eval_every: int = 10,
        verbose: bool = True,
    ) -> list[dict[str, Any]]:
        """Run GRD for n_rounds and return metric log."""
        # Reset env to initialise user pool
        self.env.reset()
        log: list[dict[str, Any]] = []

        for r in range(n_rounds):
            self._update_all()

            # Simulate one env round to update fatigue
            actions_dict = {
                f"creator_{i}": self.actions[i].numpy()
                for i in range(self.env.n_creators)
            }
            if self.env.agents:
                self.env.step(actions_dict)

            if r % eval_every == 0 or r == n_rounds - 1:
                metrics = self._current_market_metrics()
                metrics["round"] = r
                log.append(metrics)
                if verbose:
                    print(
                        f"grd {r:4d}  "
                        f"entropy={metrics['diversity_entropy']:.3f}  "
                        f"quality={metrics['mean_quality']:.3f}  "
                        f"bait={metrics['mean_bait']:.3f}  "
                        f"gini={metrics['gini']:.3f}"
                    )

        return log
