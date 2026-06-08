import numpy as np
import pytest
import torch

from recsys_market.agents.ippo import ActorCritic, IPPOTrainer, RolloutBuffer, compute_gae
from recsys_market.env.market_env import ContentMarketEnv
from recsys_market.mechanisms.m1_single import SingleStageMechanism


# ─── Fixtures ─────────────────────────────────────────────────────────────────

OBS_DIM = 21   # content_dim=16 + 5
ACTION_DIM = 18  # content_dim=16 + 2


@pytest.fixture
def small_env():
    return ContentMarketEnv(
        n_users=10,
        n_populations=3,
        n_creators=5,
        content_dim=4,
        n_rounds=10,
        slate_size=3,
        seed=0,
    )


@pytest.fixture
def actor_critic():
    return ActorCritic(obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden_dim=32)


# ─── compute_gae ──────────────────────────────────────────────────────────────

def test_gae_zero_rewards_zero_advantages():
    """With zero rewards and accurate value estimates, advantages are zero."""
    T = 10
    values = np.zeros(T, dtype=np.float32)
    rewards = np.zeros(T, dtype=np.float32)
    dones = np.zeros(T, dtype=np.float32)
    dones[-1] = 1.0
    advantages, returns = compute_gae(rewards, values, dones)
    np.testing.assert_allclose(advantages, 0.0, atol=1e-6)


def test_gae_positive_reward_positive_advantage():
    rewards = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    values = np.zeros(3, dtype=np.float32)
    dones = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    advantages, _ = compute_gae(rewards, values, dones)
    assert advantages[0] > 0


def test_gae_returns_shape():
    T = 20
    rewards = np.random.randn(T).astype(np.float32)
    values = np.random.randn(T).astype(np.float32)
    dones = np.zeros(T, dtype=np.float32)
    dones[-1] = 1.0
    adv, ret = compute_gae(rewards, values, dones)
    assert adv.shape == (T,)
    assert ret.shape == (T,)


def test_gae_returns_equals_adv_plus_values():
    T = 10
    rewards = np.ones(T, dtype=np.float32)
    values = np.ones(T, dtype=np.float32) * 0.5
    dones = np.zeros(T, dtype=np.float32)
    dones[-1] = 1.0
    adv, ret = compute_gae(rewards, values, dones)
    np.testing.assert_allclose(ret, adv + values, atol=1e-5)


# ─── ActorCritic ──────────────────────────────────────────────────────────────

def test_actor_critic_output_shapes(actor_critic):
    obs = torch.randn(4, OBS_DIM)
    mean, log_std, value = actor_critic(obs)
    assert mean.shape == (4, ACTION_DIM)
    assert log_std.shape == (4, ACTION_DIM)
    assert value.shape == (4,)


def test_get_action_in_range(actor_critic):
    obs = torch.randn(1, OBS_DIM)
    action, z, log_prob, value = actor_critic.get_action(obs)
    assert action.shape == (1, ACTION_DIM)
    assert (action.abs() <= 1.0 + 1e-6).all(), "Action outside [-1, 1]"


def test_log_prob_finite(actor_critic):
    obs = torch.randn(8, OBS_DIM)
    action, z, log_prob, value = actor_critic.get_action(obs)
    assert torch.isfinite(log_prob).all()


def test_evaluate_actions_matches_get_action(actor_critic):
    obs = torch.randn(4, OBS_DIM)
    action, z, lp_collect, value_collect = actor_critic.get_action(obs)
    lp_eval, entropy, value_eval = actor_critic.evaluate_actions(obs, z)
    # Log-probs from get_action and evaluate_actions must match exactly
    torch.testing.assert_close(lp_collect, lp_eval, atol=1e-5, rtol=1e-5)


def test_evaluate_actions_entropy_positive(actor_critic):
    obs = torch.randn(4, OBS_DIM)
    _, z, _, _ = actor_critic.get_action(obs)
    _, entropy, _ = actor_critic.evaluate_actions(obs, z)
    assert (entropy > 0).all()


def test_actor_critic_gradient_flows(actor_critic):
    obs = torch.randn(4, OBS_DIM)
    z = torch.randn(4, ACTION_DIM)
    lp, entropy, value = actor_critic.evaluate_actions(obs, z)
    loss = -lp.mean() - entropy.mean() + value.pow(2).mean()
    loss.backward()
    for name, param in actor_critic.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


# ─── RolloutBuffer ────────────────────────────────────────────────────────────

def test_rollout_buffer_to_arrays():
    buf = RolloutBuffer()
    d = 4
    for _ in range(5):
        buf.obs.append(np.zeros(d + 5))
        buf.z.append(np.zeros(d + 2))
        buf.log_probs.append(-1.0)
        buf.values.append(0.5)
        buf.rewards.append(1.0)
        buf.dones.append(0.0)
    arrays = buf.to_arrays()
    assert arrays["obs"].shape == (5, d + 5)
    assert arrays["rewards"].shape == (5,)


# ─── IPPOTrainer ─────────────────────────────────────────────────────────────

def test_ippo_trainer_one_episode(small_env):
    trainer = IPPOTrainer(small_env, lr=1e-3, n_epochs=1, batch_size=16, hidden_dim=16)
    buffers = trainer.collect_episode()
    assert set(buffers.keys()) == set(small_env.possible_agents)
    # Each buffer should have exactly n_rounds entries
    for buf in buffers.values():
        assert len(buf.rewards) == small_env.n_rounds


def test_ippo_trainer_update_no_crash(small_env):
    trainer = IPPOTrainer(small_env, lr=1e-3, n_epochs=2, batch_size=8, hidden_dim=16)
    buffers = trainer.collect_episode()
    losses = trainer.update(buffers)
    assert set(losses.keys()) == set(small_env.possible_agents)
    for agent_losses in losses.values():
        assert np.isfinite(agent_losses["policy_loss"])
        assert np.isfinite(agent_losses["value_loss"])


def test_ippo_trainer_rewards_finite(small_env):
    trainer = IPPOTrainer(small_env, lr=1e-3, n_epochs=1, batch_size=16, hidden_dim=16)
    buffers = trainer.collect_episode()
    for buf in buffers.values():
        assert all(np.isfinite(r) for r in buf.rewards)


def test_ippo_train_returns_log(small_env):
    trainer = IPPOTrainer(small_env, lr=1e-3, n_epochs=1, batch_size=16, hidden_dim=16)
    log = trainer.train(n_episodes=3, eval_every=1, verbose=False)
    assert len(log) == 3
    assert "diversity_entropy" in log[0]
    assert "gini" in log[0]


def test_ippo_policy_updates_over_time(small_env):
    """Policy weights should change after training steps."""
    trainer = IPPOTrainer(small_env, lr=1e-3, n_epochs=4, batch_size=8, hidden_dim=16)
    agent = small_env.possible_agents[0]

    before = trainer.policies[agent].actor_mean.weight.data.clone()
    buffers = trainer.collect_episode()
    trainer.update(buffers)
    after = trainer.policies[agent].actor_mean.weight.data

    assert not torch.allclose(before, after), "Policy weights did not change after update"


def test_ippo_save_load(small_env, tmp_path):
    trainer = IPPOTrainer(small_env, lr=1e-3, n_epochs=1, batch_size=8, hidden_dim=16)
    path = str(tmp_path / "ckpt.pt")
    trainer.save(path)

    trainer2 = IPPOTrainer(small_env, lr=1e-3, n_epochs=1, batch_size=8, hidden_dim=16)
    trainer2.load(path)

    agent = small_env.possible_agents[0]
    for p1, p2 in zip(
        trainer.policies[agent].parameters(),
        trainer2.policies[agent].parameters(),
    ):
        torch.testing.assert_close(p1, p2)
