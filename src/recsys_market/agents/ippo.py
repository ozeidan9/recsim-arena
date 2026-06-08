"""Independent PPO (IPPO): one actor-critic per creator, trained independently.

Each creator is an independent PPO learner. They share no weights and do not
coordinate — identical to how independent learning works in practice (creators
can't observe each other's policies, only the market outcomes they receive).

Architecture: 2-layer MLP → mean + log_std (state-independent) of a
TanhNormal, which squashes actions to [-1, 1]. Pre-tanh residuals (z) are
stored in the rollout buffer so log-prob can be recomputed exactly during the
PPO update without numerical approximation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

if TYPE_CHECKING:
    from recsys_market.env.market_env import ContentMarketEnv


# ─── GAE ─────────────────────────────────────────────────────────────────────


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalised Advantage Estimation.

    Returns:
        advantages: (T,) zero-mean advantage estimates.
        returns:    (T,) value targets (advantages + values).
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        # Bootstrap from next value unless terminal
        if dones[t] or t == T - 1:
            next_v = 0.0
        else:
            next_v = float(values[t + 1])
        delta = rewards[t] + gamma * next_v * (1.0 - dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * last_gae
    return advantages, advantages + values


# ─── Rollout buffer ───────────────────────────────────────────────────────────


@dataclass
class RolloutBuffer:
    obs: list[np.ndarray] = field(default_factory=list)
    z: list[np.ndarray] = field(default_factory=list)       # pre-tanh action
    log_probs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[float] = field(default_factory=list)

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {
            "obs": np.array(self.obs, dtype=np.float32),
            "z": np.array(self.z, dtype=np.float32),
            "log_probs": np.array(self.log_probs, dtype=np.float32),
            "values": np.array(self.values, dtype=np.float32),
            "rewards": np.array(self.rewards, dtype=np.float32),
            "dones": np.array(self.dones, dtype=np.float32),
        }


# ─── Actor-critic ─────────────────────────────────────────────────────────────


class ActorCritic(nn.Module):
    """2-layer MLP actor-critic with TanhNormal policy and state-independent log_std."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        # State-independent log_std parameter; clamped to [-4, 0] → std in [e^-4, 1]
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim) - 0.5)
        self.critic_head = nn.Linear(hidden_dim, 1)

        # Orthogonal initialisation — standard for PPO
        for layer in [self.shared[0], self.shared[2], self.critic_head]:
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.shared(obs)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f = self._features(obs)
        mean = self.actor_mean(f)
        log_std = self.actor_log_std.expand_as(mean).clamp(-4.0, 0.0)
        value = self.critic_head(f).squeeze(-1)
        return mean, log_std, value

    @torch.no_grad()
    def get_action(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action and return (action, z, log_prob, value)."""
        mean, log_std, value = self.forward(obs)
        std = log_std.exp()
        z = mean + std * torch.randn_like(mean)
        action = torch.tanh(z)
        log_prob = self._tanh_normal_log_prob(mean, std, z)
        return action, z, log_prob, value

    def evaluate_actions(
        self, obs: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recompute log_prob, entropy, value for stored (obs, z) during PPO update."""
        mean, log_std, value = self.forward(obs)
        std = log_std.exp()
        log_prob = self._tanh_normal_log_prob(mean, std, z)
        entropy = Normal(mean, std).entropy().sum(-1)
        return log_prob, entropy, value

    @staticmethod
    def _tanh_normal_log_prob(
        mean: torch.Tensor, std: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """log p(tanh(z)) = log p_N(z) - sum log(1 - tanh(z)^2)."""
        normal_lp = Normal(mean, std).log_prob(z).sum(-1)
        # Numerically stable: log(1-tanh^2(z)) = log4 - 2z - 2*softplus(-2z)
        correction = (2.0 * (math.log(2) - z - F.softplus(-2.0 * z))).sum(-1)
        return normal_lp - correction


# ─── Trainer ──────────────────────────────────────────────────────────────────


class IPPOTrainer:
    """Independent PPO trainer for the content-market environment."""

    def __init__(
        self,
        env: ContentMarketEnv,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 64,
        hidden_dim: int = 64,
        device: str = "cpu",
    ) -> None:
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = torch.device(device)

        obs_dim = env.content_dim + 5
        action_dim = env.content_dim + 2

        self.policies: dict[str, ActorCritic] = {
            agent: ActorCritic(obs_dim, action_dim, hidden_dim).to(self.device)
            for agent in env.possible_agents
        }
        self.optimizers: dict[str, torch.optim.Adam] = {
            agent: torch.optim.Adam(policy.parameters(), lr=lr, eps=1e-5)
            for agent, policy in self.policies.items()
        }

    # ── data collection ───────────────────────────────────────────────────────

    def collect_episode(self) -> dict[str, RolloutBuffer]:
        obs_dict, _ = self.env.reset()
        buffers: dict[str, RolloutBuffer] = {
            agent: RolloutBuffer() for agent in self.env.possible_agents
        }

        while self.env.agents:
            step_actions: dict[str, np.ndarray] = {}
            step_z: dict[str, np.ndarray] = {}
            step_lp: dict[str, float] = {}
            step_v: dict[str, float] = {}

            for agent in self.env.possible_agents:
                buffers[agent].obs.append(obs_dict[agent].copy())

                obs_t = torch.FloatTensor(obs_dict[agent]).unsqueeze(0).to(self.device)
                action, z, log_prob, value = self.policies[agent].get_action(obs_t)

                step_actions[agent] = action.squeeze(0).cpu().numpy()
                step_z[agent] = z.squeeze(0).cpu().numpy()
                step_lp[agent] = log_prob.item()
                step_v[agent] = value.item()

            obs_dict, rewards, terminations, _, _ = self.env.step(step_actions)

            for agent in self.env.possible_agents:
                buf = buffers[agent]
                buf.z.append(step_z[agent])
                buf.log_probs.append(step_lp[agent])
                buf.values.append(step_v[agent])
                buf.rewards.append(float(rewards.get(agent, 0.0)))
                buf.dones.append(float(terminations.get(agent, False)))

        return buffers

    # ── PPO update ────────────────────────────────────────────────────────────

    def update(
        self, buffers: dict[str, RolloutBuffer]
    ) -> dict[str, dict[str, float]]:
        all_losses: dict[str, dict[str, float]] = {}
        rng = np.random.default_rng()

        for agent, buf in buffers.items():
            arrays = buf.to_arrays()
            obs_np = arrays["obs"]
            z_np = arrays["z"]
            old_lp_np = arrays["log_probs"]
            rewards_np = arrays["rewards"]
            values_np = arrays["values"]
            dones_np = arrays["dones"]

            advantages, returns = compute_gae(
                rewards_np, values_np, dones_np, self.gamma, self.gae_lambda
            )
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            obs_t = torch.FloatTensor(obs_np).to(self.device)
            z_t = torch.FloatTensor(z_np).to(self.device)
            old_lp_t = torch.FloatTensor(old_lp_np).to(self.device)
            adv_t = torch.FloatTensor(advantages).to(self.device)
            ret_t = torch.FloatTensor(returns).to(self.device)

            T = len(rewards_np)
            last_pl = last_vl = last_ent = 0.0

            for _ in range(self.n_epochs):
                idx = rng.permutation(T)
                for start in range(0, T, self.batch_size):
                    b = idx[start : start + self.batch_size]

                    new_lp, entropy, new_v = self.policies[agent].evaluate_actions(
                        obs_t[b], z_t[b]
                    )
                    ratio = (new_lp - old_lp_t[b]).exp()
                    adv_b = adv_t[b]

                    policy_loss = -torch.min(
                        ratio * adv_b,
                        torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef) * adv_b,
                    ).mean()
                    value_loss = F.mse_loss(new_v, ret_t[b])
                    entropy_loss = -entropy.mean()

                    loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                    self.optimizers[agent].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.policies[agent].parameters(), self.max_grad_norm
                    )
                    self.optimizers[agent].step()

                    last_pl = policy_loss.item()
                    last_vl = value_loss.item()
                    last_ent = entropy.mean().item()

            all_losses[agent] = {
                "policy_loss": last_pl,
                "value_loss": last_vl,
                "entropy": last_ent,
            }

        return all_losses

    # ── episode metrics ───────────────────────────────────────────────────────

    def _episode_metrics(self, buffers: dict[str, RolloutBuffer]) -> dict[str, Any]:
        """Compute market-health metrics from the just-completed episode."""
        from recsys_market.metrics.diversity import content_entropy, intra_list_diversity
        from recsys_market.metrics.inequality import gini
        from recsys_market.mechanisms.m1_single import SingleStageMechanism

        contents = self.env.current_contents          # (M, d)
        quality = self.env.current_quality            # (M,)
        bait = self.env.current_bait                  # (M,)
        exposure = self.env.cumulative_exposure       # (M,)

        # Slates for ILD (use current M1 ranking of current content)
        _mech = SingleStageMechanism()
        _rng = np.random.default_rng(0)
        slates = _mech.recommend(
            self.env._user_pool.preferences, contents, quality, bait,
            slate_size=self.env.slate_size, rng=_rng,
        )

        mean_ep_reward = np.mean(
            [np.mean(buf.rewards) for buf in buffers.values()]
        )
        total_ep_reward = np.mean(
            [np.sum(buf.rewards) for buf in buffers.values()]
        )

        return {
            "diversity_entropy": float(content_entropy(contents, n_clusters=8)),
            "gini": float(gini(exposure)),
            "mean_quality": float(quality.mean()),
            "mean_bait": float(bait.mean()),
            "ild": float(intra_list_diversity(slates, contents)),
            "mean_ep_reward": float(mean_ep_reward),
            "total_ep_reward": float(total_ep_reward),
        }

    # ── train loop ────────────────────────────────────────────────────────────

    def train(
        self,
        n_episodes: int,
        eval_every: int = 10,
        verbose: bool = True,
    ) -> list[dict[str, Any]]:
        """Run training and return episode-level metrics list."""
        log: list[dict[str, Any]] = []

        for ep in range(n_episodes):
            buffers = self.collect_episode()
            losses = self.update(buffers)

            if ep % eval_every == 0 or ep == n_episodes - 1:
                metrics = self._episode_metrics(buffers)
                mean_losses = {
                    k: float(np.mean([v[k] for v in losses.values()]))
                    for k in next(iter(losses.values()))
                }
                metrics.update(mean_losses)
                metrics["episode"] = ep
                log.append(metrics)

                if verbose:
                    print(
                        f"ep {ep:4d}  "
                        f"entropy={metrics['diversity_entropy']:.3f}  "
                        f"quality={metrics['mean_quality']:.3f}  "
                        f"bait={metrics['mean_bait']:.3f}  "
                        f"gini={metrics['gini']:.3f}  "
                        f"reward={metrics['total_ep_reward']:.1f}"
                    )

        return log

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        torch.save(
            {agent: policy.state_dict() for agent, policy in self.policies.items()},
            path,
        )

    def load(self, path: str) -> None:
        state_dicts = torch.load(path, map_location=self.device)
        for agent, sd in state_dicts.items():
            self.policies[agent].load_state_dict(sd)
