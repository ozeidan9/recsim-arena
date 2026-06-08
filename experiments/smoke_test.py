"""Quick smoke test: run one episode with random agents, print key metrics.

Usage:
    python experiments/smoke_test.py                      # M1, seed=42
    python experiments/smoke_test.py --mechanism m0       # Random mechanism
    python experiments/smoke_test.py --seed 7
"""

import argparse
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanism", choices=["m0", "m1"], default="m1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-rounds", type=int, default=10)
    args = parser.parse_args()

    from recsys_market.env.market_env import ContentMarketEnv
    from recsys_market.mechanisms.m0_random import RandomMechanism
    from recsys_market.mechanisms.m1_single import SingleStageMechanism
    import recsys_market.metrics as M

    mechanism = SingleStageMechanism() if args.mechanism == "m1" else RandomMechanism()
    mech_name = f"SingleStage(temp=1.0)" if args.mechanism == "m1" else "Random"

    env = ContentMarketEnv(
        n_users=50,
        n_populations=5,
        n_creators=20,
        content_dim=16,
        n_rounds=args.n_rounds,
        slate_size=5,
        quality_cost_scale=0.5,
        fatigue_gamma=0.95,
        alpha_quality=0.3,
        gamma_bait=0.5,
        beta_bait=0.5,
        mechanism=mechanism,
        seed=args.seed,
    )

    obs, _ = env.reset(seed=args.seed)
    rng = np.random.default_rng(args.seed)

    print(f"\n{'='*55}")
    print(f"Smoke test: {mech_name}  seed={args.seed}")
    print(f"Observation shape : {next(iter(obs.values())).shape}")
    print(f"Action shape      : {env.action_space(env.agents[0]).shape}")
    print(f"{'='*55}")

    total_rewards = np.zeros(env.n_creators)

    for step in range(args.n_rounds):
        if not env.agents:
            break
        actions = {
            agent: rng.uniform(-1, 1, env.action_space(agent).shape).astype(np.float32)
            for agent in env.agents
        }
        obs, rewards, terminations, _, infos = env.step(actions)
        rewards_arr = np.array([rewards.get(f"creator_{i}", 0.0) for i in range(env.n_creators)])
        total_rewards += rewards_arr

    contents = env.current_contents
    quality = env.current_quality
    bait = env.current_bait
    exposure = env.cumulative_exposure

    # Build a dummy slate for quality reporting (last round's top-1 per user)
    from recsys_market.mechanisms.m1_single import SingleStageMechanism as S1
    dummy_slates = S1().recommend(
        env._user_pool.preferences, contents, quality, bait,
        slate_size=5, rng=rng,
    )

    diversity = M.content_entropy(contents, n_clusters=8)
    g = M.gini(exposure)
    mqual = M.mean_recommended_quality(dummy_slates, quality)
    ild = M.intra_list_diversity(dummy_slates, contents)
    cov = M.coverage(contents, env._user_pool.preferences)

    print(f"\nAfter {args.n_rounds} rounds (random policy):")
    print(f"  Content entropy (diversity) : {diversity:.4f}  (max: {np.log(8):.4f})")
    print(f"  Gini (exposure inequality)  : {g:.4f}")
    print(f"  Mean recommended quality    : {mqual:.4f}")
    print(f"  Intra-list diversity (ILD)  : {ild:.4f}")
    print(f"  User coverage               : {cov:.4f}")
    print(f"  Mean reward per creator     : {total_rewards.mean():.4f}")
    print(f"  Reward range                : [{total_rewards.min():.4f}, {total_rewards.max():.4f}]")

    assert all(np.isfinite(total_rewards)), "Non-finite rewards detected!"
    print(f"\n✓ Smoke test passed.")


if __name__ == "__main__":
    main()
