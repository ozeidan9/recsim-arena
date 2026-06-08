from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium.spaces import Box
from pettingzoo import ParallelEnv

from recsys_market.creators.creator_model import CreatorPool
from recsys_market.mechanisms.base import Mechanism
from recsys_market.users.user_model import UserPool


class ContentMarketEnv(ParallelEnv):
    """Two-sided content market as a PettingZoo ParallelEnv.

    Agents are creators who simultaneously choose content vectors, quality, and
    bait each round. The recommender mechanism presents slates to users, who
    click according to a logit choice model. Creator rewards are clicks minus
    production cost.

    Observation per creator (float32, shape (content_dim + 5,)):
        [0:d]   own content vector (l2-normalised)
        [d]     own quality this round
        [d+1]   own bait this round
        [d+2]   own exposure fraction last round (clicks / n_users)
        [d+3]   market diversity (content entropy, normalised to [0,1])
        [d+4]   mean similarity to nearest competitor

    Action per creator (float32, shape (content_dim + 2,)):
        [0:d]   content direction (l2-normalised internally)
        [d]     quality ∈ [-1,1] → mapped to [0,1]
        [d+1]   bait   ∈ [-1,1] → mapped to [0,1]
    """

    metadata = {"render_modes": [], "name": "content_market_v0"}

    def __init__(
        self,
        n_users: int = 50,
        n_populations: int = 5,
        n_creators: int = 20,
        content_dim: int = 16,
        n_rounds: int = 200,
        slate_size: int = 5,
        quality_cost_scale: float = 0.5,
        fatigue_gamma: float = 0.95,
        alpha_quality: float = 0.3,
        gamma_bait: float = 0.5,
        beta_bait: float = 0.5,
        mechanism: Mechanism | None = None,
        seed: int = 0,
    ) -> None:
        self.n_users = n_users
        self.n_creators = n_creators
        self.content_dim = content_dim
        self.n_rounds = n_rounds
        self.slate_size = slate_size
        self.alpha_quality = alpha_quality
        self.gamma_bait = gamma_bait   # short-term click attraction of bait
        self.beta_bait = beta_bait     # fatigue-weighted bait penalty

        self.possible_agents = [f"creator_{i}" for i in range(n_creators)]

        self._user_pool = UserPool(
            n_users=n_users,
            n_populations=n_populations,
            content_dim=content_dim,
            seed=seed,
        )
        self._creator_pool = CreatorPool(
            n_creators=n_creators,
            content_dim=content_dim,
            quality_cost_scale=quality_cost_scale,
            seed=seed + 1,
        )

        from recsys_market.mechanisms.m1_single import SingleStageMechanism
        self._mechanism: Mechanism = mechanism if mechanism is not None else SingleStageMechanism()

        self._fatigue_gamma = fatigue_gamma
        self._rng = np.random.default_rng(seed)

        # Runtime state — initialised in reset()
        self._round: int = 0
        self._last_contents: np.ndarray = np.zeros((n_creators, content_dim), dtype=np.float32)
        self._last_quality: np.ndarray = np.zeros(n_creators, dtype=np.float32)
        self._last_bait: np.ndarray = np.zeros(n_creators, dtype=np.float32)
        self._last_exposure_frac: np.ndarray = np.zeros(n_creators, dtype=np.float32)
        self._cumulative_exposure: np.ndarray = np.zeros(n_creators, dtype=np.float32)

        # Set after reset() — public attributes used by wrappers/trainers
        self.agents: list[str] = []

    # ------------------------------------------------------------------
    # PettingZoo API
    # ------------------------------------------------------------------

    def observation_space(self, agent: str) -> Box:
        d = self.content_dim
        return Box(low=-np.inf, high=np.inf, shape=(d + 5,), dtype=np.float32)

    def action_space(self, agent: str) -> Box:
        d = self.content_dim
        return Box(low=-1.0, high=1.0, shape=(d + 2,), dtype=np.float32)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._user_pool.reset()
        self._round = 0
        self._last_exposure_frac = np.zeros(self.n_creators, dtype=np.float32)
        self._cumulative_exposure = np.zeros(self.n_creators, dtype=np.float32)

        init_actions = self._creator_pool.initial_actions()
        contents, quality, bait = self._creator_pool.action_to_components(init_actions)
        self._last_contents = contents
        self._last_quality = quality
        self._last_bait = bait

        self.agents = list(self.possible_agents)
        obs = self._compute_observations()
        infos = {agent: {} for agent in self.agents}
        return obs, infos

    def step(
        self, actions: dict[str, np.ndarray]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        # Stack per-agent actions into (M, d+2)
        action_matrix = np.stack(
            [actions[agent] for agent in self.possible_agents], axis=0
        ).astype(np.float32)

        contents, quality, bait = self._creator_pool.action_to_components(action_matrix)
        self._last_contents = contents
        self._last_quality = quality
        self._last_bait = bait

        # Recommend slates: (N, slate_size)
        slates = self._mechanism.recommend(
            user_preferences=self._user_pool.preferences,
            creator_contents=contents,
            creator_quality=quality,
            creator_bait=bait,
            slate_size=self.slate_size,
            rng=self._rng,
        )

        # Click probabilities via logit choice model
        clicks, bait_consumed = self._compute_clicks(slates, contents, quality, bait)

        # Creator rewards
        exposure_counts = clicks  # (M,) total clicks received
        costs = self._creator_pool.production_cost(quality)
        rewards_arr = exposure_counts.astype(np.float32) - costs

        # Update state
        self._last_exposure_frac = exposure_counts / max(self.n_users, 1)
        self._cumulative_exposure += exposure_counts
        self._user_pool.update_fatigue(bait_consumed, gamma=self._fatigue_gamma)
        self._round += 1

        done = self._round >= self.n_rounds
        terminations = {agent: done for agent in self.possible_agents}
        truncations = {agent: False for agent in self.possible_agents}

        obs = self._compute_observations()
        rewards = {
            agent: float(rewards_arr[i])
            for i, agent in enumerate(self.possible_agents)
        }
        infos = {
            agent: {
                "exposure": float(exposure_counts[i]),
                "quality": float(quality[i]),
                "bait": float(bait[i]),
                "cost": float(costs[i]),
            }
            for i, agent in enumerate(self.possible_agents)
        }

        if done:
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_clicks(
        self,
        slates: np.ndarray,      # (N, k)
        contents: np.ndarray,    # (M, d)
        quality: np.ndarray,     # (M,)
        bait: np.ndarray,        # (M,)
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample clicks from the logit choice model.

        Returns:
            clicks:        (M,) integer click counts per creator.
            bait_consumed: (N,) bait exposure per user (for fatigue update).
        """
        n_users, k = slates.shape
        user_prefs = self._user_pool.preferences  # (N, d)
        fatigue = self._user_pool.fatigue          # (N,)

        # Gather slate contents, quality, bait for each user: all (N, k)
        slate_contents = contents[slates]          # (N, k, d)
        slate_quality = quality[slates]            # (N, k)
        slate_bait = bait[slates]                  # (N, k)

        # Relevance: dot(user_pref, content_j) — (N, k)
        relevance = np.einsum("nd,nkd->nk", user_prefs, slate_contents)

        # Logit: relevance + alpha*quality + gamma*bait - beta*bait*fatigue
        # gamma_bait: short-term click attraction; beta_bait: fatigue-weighted penalty.
        # Net bait effect = (gamma - beta*fatigue) * bait, positive when fatigue < gamma/beta.
        fatigue_col = fatigue[:, np.newaxis]       # (N, 1)
        logits = (
            relevance
            + self.alpha_quality * slate_quality
            + self.gamma_bait * slate_bait
            - self.beta_bait * slate_bait * fatigue_col
        )

        # Softmax over slate
        logits -= logits.max(axis=1, keepdims=True)  # numerical stability
        exp_l = np.exp(logits)
        probs = exp_l / exp_l.sum(axis=1, keepdims=True)  # (N, k)

        # Sample one click per user
        chosen = np.array(
            [self._rng.choice(k, p=probs[i]) for i in range(n_users)]
        )  # (N,)

        # Map back to creator indices
        chosen_creators = slates[np.arange(n_users), chosen]  # (N,)

        # Accumulate clicks per creator
        clicks = np.bincount(chosen_creators, minlength=self.n_creators).astype(np.float32)

        # Bait consumed per user = bait level of chosen creator
        bait_consumed = slate_bait[np.arange(n_users), chosen]  # (N,)

        return clicks, bait_consumed

    def _compute_observations(self) -> dict[str, np.ndarray]:
        """Build per-creator observation vectors."""
        from recsys_market.metrics.diversity import content_entropy

        # Market diversity (normalised)
        entropy = content_entropy(self._last_contents, n_clusters=8)
        max_entropy = float(np.log(8))
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        # Mean similarity to nearest competitor for each creator
        sims = self._last_contents @ self._last_contents.T  # (M, M)
        np.fill_diagonal(sims, -np.inf)
        nearest_sim = sims.max(axis=1)                       # (M,)

        obs: dict[str, np.ndarray] = {}
        for i, agent in enumerate(self.possible_agents):
            vec = np.concatenate([
                self._last_contents[i],                    # (d,)
                [self._last_quality[i]],                   # (1,)
                [self._last_bait[i]],                      # (1,)
                [self._last_exposure_frac[i]],             # (1,)
                [norm_entropy],                            # (1,)
                [float(nearest_sim[i])],                   # (1,)
            ]).astype(np.float32)
            obs[agent] = vec
        return obs

    # ------------------------------------------------------------------
    # Convenience accessors (used by metrics and analysis scripts)
    # ------------------------------------------------------------------

    @property
    def current_contents(self) -> np.ndarray:
        return self._last_contents

    @property
    def current_quality(self) -> np.ndarray:
        return self._last_quality

    @property
    def current_bait(self) -> np.ndarray:
        return self._last_bait

    @property
    def cumulative_exposure(self) -> np.ndarray:
        return self._cumulative_exposure

    @property
    def round(self) -> int:
        return self._round
