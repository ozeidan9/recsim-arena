import numpy as np
import pytest

from recsys_market.env.market_env import ContentMarketEnv
from recsys_market.mechanisms.m0_random import RandomMechanism
from recsys_market.mechanisms.m1_single import SingleStageMechanism


@pytest.fixture
def env():
    return ContentMarketEnv(
        n_users=10,
        n_populations=3,
        n_creators=5,
        content_dim=4,
        n_rounds=10,
        slate_size=3,
        seed=42,
    )


def random_actions(env, rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {
        agent: rng.uniform(-1, 1, env.action_space(agent).shape).astype(np.float32)
        for agent in env.possible_agents
    }


def test_obs_shape_after_reset(env):
    obs, _ = env.reset(seed=0)
    expected_dim = env.content_dim + 5
    for agent, o in obs.items():
        assert o.shape == (expected_dim,), f"Bad obs shape for {agent}: {o.shape}"


def test_agents_list_populated(env):
    env.reset(seed=0)
    assert len(env.agents) == env.n_creators


def test_action_roundtrip_normalised(env):
    """Content part of any action is l2-normalised inside the creator pool."""
    rng = np.random.default_rng(7)
    raw = rng.uniform(-1, 1, (env.n_creators, env.content_dim + 2)).astype(np.float32)
    content, quality, bait = env._creator_pool.action_to_components(raw)
    norms = np.linalg.norm(content, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_reward_non_negative_random_policy(env):
    """Rewards can be negative (cost > clicks) but should be finite."""
    env.reset(seed=0)
    rng = np.random.default_rng(1)
    actions = random_actions(env, rng)
    _, rewards, _, _, _ = env.step(actions)
    for r in rewards.values():
        assert np.isfinite(r), f"Non-finite reward: {r}"


def test_termination_at_round_T(env):
    env.reset(seed=0)
    rng = np.random.default_rng(2)
    terminated = False
    for step in range(env.n_rounds + 5):  # intentionally over-run
        if not env.agents:
            break
        actions = random_actions(env, rng)
        _, _, terminations, _, _ = env.step(actions)
        if all(terminations.values()):
            terminated = True
            break
    assert terminated, "Env never terminated"
    assert env.round == env.n_rounds


def test_reproducibility(env):
    """Same seed → same episode trajectory."""
    def run_episode(seed):
        e = ContentMarketEnv(n_users=10, n_populations=3, n_creators=5, content_dim=4, n_rounds=5, slate_size=3, seed=0)
        obs, _ = e.reset(seed=seed)
        rng = np.random.default_rng(seed)
        rewards_trace = []
        for _ in range(5):
            if not e.agents:
                break
            actions = random_actions(e, rng)
            _, rewards, _, _, _ = e.step(actions)
            rewards_trace.append(list(rewards.values()))
        return rewards_trace

    trace1 = run_episode(99)
    trace2 = run_episode(99)
    assert trace1 == trace2, "Non-deterministic episodes with same seed"


def test_obs_dtype(env):
    obs, _ = env.reset(seed=0)
    for o in obs.values():
        assert o.dtype == np.float32


def test_step_returns_all_agents(env):
    env.reset(seed=0)
    rng = np.random.default_rng(3)
    actions = random_actions(env, rng)
    obs, rewards, terminations, truncations, infos = env.step(actions)
    assert set(obs.keys()) == set(env.possible_agents)
    assert set(rewards.keys()) == set(env.possible_agents)


def test_m0_mechanism_works(env):
    """Env should work with RandomMechanism too."""
    env_m0 = ContentMarketEnv(
        n_users=10, n_populations=3, n_creators=5, content_dim=4,
        n_rounds=5, slate_size=3, mechanism=RandomMechanism(), seed=0,
    )
    obs, _ = env_m0.reset(seed=0)
    rng = np.random.default_rng(0)
    actions = random_actions(env_m0, rng)
    _, rewards, _, _, _ = env_m0.step(actions)
    assert all(np.isfinite(r) for r in rewards.values())


def test_cumulative_exposure_non_negative(env):
    env.reset(seed=0)
    rng = np.random.default_rng(4)
    for _ in range(5):
        if not env.agents:
            break
        env.step(random_actions(env, rng))
    assert (env.cumulative_exposure >= 0).all()
