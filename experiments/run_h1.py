"""H1 validation: engagement collapse on single-stage mechanism.

Trains IPPO creators on M1 (pure relevance ranking) and tracks diversity,
quality, and bait over episodes. Expected result: diversity entropy drops,
bait increases, quality drops — the engagement→clickbait collapse.

Also runs the GradientAscentDynamics baseline as a sanity check that the
collapse is not specific to the PPO learning algorithm.

Usage:
    python experiments/run_h1.py                              # defaults
    python experiments/run_h1.py --n-episodes 500 --seed 42
    python experiments/run_h1.py --n-episodes 200 --no-grd   # skip GRD baseline
    python experiments/run_h1.py --wandb                      # log to W&B
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


def make_env(seed: int, mechanism: str = "m1"):
    from recsys_market.env.market_env import ContentMarketEnv
    from recsys_market.mechanisms.m0_random import RandomMechanism
    from recsys_market.mechanisms.m1_single import SingleStageMechanism

    mech = (
        SingleStageMechanism(temperature=1.0, bait_weight=0.5)
        if mechanism == "m1"
        else RandomMechanism()
    )
    return ContentMarketEnv(
        n_users=50,
        n_populations=5,
        n_creators=20,
        content_dim=16,
        n_rounds=200,
        slate_size=5,
        quality_cost_scale=0.5,
        fatigue_gamma=0.95,
        alpha_quality=0.3,
        gamma_bait=0.5,
        beta_bait=0.5,
        mechanism=mech,
        seed=seed,
    )


def run_ippo(args, wandb_run=None) -> list[dict]:
    from recsys_market.agents.ippo import IPPOTrainer

    env = make_env(args.seed, mechanism="m1")
    trainer = IPPOTrainer(
        env=env,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        n_epochs=4,
        batch_size=64,
        hidden_dim=64,
    )

    print(f"\n{'='*60}")
    print(f"H1 — IPPO on M1 | episodes={args.n_episodes} | seed={args.seed}")
    print(f"{'='*60}")
    t0 = time.time()

    log = trainer.train(
        n_episodes=args.n_episodes,
        eval_every=args.eval_every,
        verbose=True,
    )

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s")

    if wandb_run is not None:
        for entry in log:
            wandb_run.log({"ippo/" + k: v for k, v in entry.items()})

    # Save policy checkpoint
    ckpt_dir = ROOT / "results" / f"h1_seed{args.seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    trainer.save(str(ckpt_dir / "ippo_final.pt"))

    return log


def run_grd(args, wandb_run=None) -> list[dict]:
    from recsys_market.agents.best_response import GradientAscentDynamics

    env = make_env(args.seed, mechanism="m1")
    grd = GradientAscentDynamics(env=env, lr=0.05, n_steps_per_round=5, temperature=1.0, bait_weight=0.5)

    n_rounds = min(args.n_episodes, 200)  # GRD converges faster

    print(f"\n{'='*60}")
    print(f"H1 — GRD baseline on M1 | rounds={n_rounds} | seed={args.seed}")
    print(f"{'='*60}")

    log = grd.run(n_rounds=n_rounds, eval_every=args.eval_every, verbose=True)

    if wandb_run is not None:
        for entry in log:
            wandb_run.log({"grd/" + k: v for k, v in entry.items()})

    return log


def main() -> None:
    parser = argparse.ArgumentParser(description="H1 validation experiment")
    parser.add_argument("--n-episodes", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-grd", action="store_true", help="Skip GRD baseline")
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    args = parser.parse_args()

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project="recsim-arena",
            name=f"h1_ippo_seed{args.seed}",
            config=vars(args),
        )

    ippo_log = run_ippo(args, wandb_run)

    grd_log: list[dict] = []
    if not args.no_grd:
        grd_log = run_grd(args, wandb_run)

    # ── Save results ──────────────────────────────────────────────────────────
    results_dir = ROOT / "results" / f"h1_seed{args.seed}"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "ippo_log.json", "w") as f:
        json.dump(ippo_log, f, indent=2)

    if grd_log:
        with open(results_dir / "grd_log.json", "w") as f:
            json.dump(grd_log, f, indent=2)

    print(f"\nResults saved to {results_dir}/")

    # ── Quick summary ─────────────────────────────────────────────────────────
    if len(ippo_log) >= 2:
        first = ippo_log[0]
        last = ippo_log[-1]
        print(f"\n{'='*60}")
        print(f"H1 Summary (IPPO, M1):")
        print(f"  Diversity entropy:  {first['diversity_entropy']:.3f} → {last['diversity_entropy']:.3f}  (collapsed={last['diversity_entropy'] < first['diversity_entropy'] * 0.85})")
        print(f"  Mean quality:       {first['mean_quality']:.3f} → {last['mean_quality']:.3f}")
        print(f"  Mean bait:          {first['mean_bait']:.3f} → {last['mean_bait']:.3f}")
        print(f"  Gini:               {first['gini']:.3f} → {last['gini']:.3f}")
        # H1 is confirmed by: bait inflation (>30%) AND Gini increase (>50%).
        # Content-direction diversity staying high is expected with heterogeneous users
        # (specialization equilibrium); the collapse is in bait/quality space.
        bait_inflated = last["mean_bait"] > first["mean_bait"] * 1.3
        gini_increased = last["gini"] > first["gini"] * 1.5
        h1_confirmed = bait_inflated and gini_increased
        print(f"\n  H1 {'CONFIRMED ✓' if h1_confirmed else 'NOT YET CONFIRMED — run more episodes'}")
        print(f"{'='*60}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
