"""H4 validation: two-stage pipeline concentrates exposure more than single-stage.

Compares Gini coefficient trajectories under:
  M1  (H1 baseline): single-stage dot-product ranking
  M2  (H4 test):     two-stage FAISS retrieval + learned engagement reranker

H4 prediction: the reranker's feedback loop (popular creators attract more data
→ better predictions → more promotion → more clicks) concentrates exposure
beyond what static M1 ranking achieves.

Confirmation criterion (H4 confirmed):
  - Final Gini under M2 > final Gini under M1 * 1.3  (≥30% more concentrated)

M1 baseline is loaded from results/h1_seed{seed}/ippo_log.json if present.

Usage:
    python experiments/run_h4.py
    python experiments/run_h4.py --n-episodes 300 --seed 42
    python experiments/run_h4.py --retrieval-size 8    # tighter retrieval bottleneck
    python experiments/run_h4.py --wandb
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


def make_env(seed: int, mechanism_name: str = "m2", retrieval_size: int = 10):
    from recsys_market.env.market_env import ContentMarketEnv
    from recsys_market.mechanisms.m1_single import SingleStageMechanism
    from recsys_market.mechanisms.m2_two_stage import TwoStageMechanism

    content_dim = 16

    if mechanism_name == "m1":
        mech = SingleStageMechanism(temperature=1.0, bait_weight=0.5)
    else:
        mech = TwoStageMechanism(
            content_dim=content_dim,
            retrieval_size=retrieval_size,
        )

    return ContentMarketEnv(
        n_users=50,
        n_populations=5,
        n_creators=20,
        content_dim=content_dim,
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


def run_ippo(args, mechanism_name: str, wandb_run=None) -> list[dict]:
    from recsys_market.agents.ippo import IPPOTrainer

    env = make_env(args.seed, mechanism_name, args.retrieval_size)
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

    label = mechanism_name.upper()
    print(f"\n{'='*60}")
    print(f"H4 — IPPO on {label} | episodes={args.n_episodes} | seed={args.seed}")
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
            wandb_run.log({f"{mechanism_name}/" + k: v for k, v in entry.items()})

    ckpt_dir = ROOT / "results" / f"h4_{mechanism_name}_seed{args.seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    trainer.save(str(ckpt_dir / "ippo_final.pt"))

    return log


def load_m1_baseline(seed: int) -> dict | None:
    path = ROOT / "results" / f"h1_seed{seed}" / "ippo_log.json"
    if not path.exists():
        return None
    with open(path) as f:
        log = json.load(f)
    return log[-1] if log else None


def main() -> None:
    parser = argparse.ArgumentParser(description="H4 validation experiment")
    parser.add_argument("--n-episodes", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retrieval-size", type=int, default=10,
                        help="R: number of candidates in retrieval stage (default 10 of 20)")
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    args = parser.parse_args()

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project="recsim-arena",
            name=f"h4_m2_seed{args.seed}",
            config=vars(args),
        )

    m2_log = run_ippo(args, "m2", wandb_run)

    m1_last = load_m1_baseline(args.seed)

    # ── Save results ─────────────────────────────────────────────────────────
    results_dir = ROOT / "results" / f"h4_m2_seed{args.seed}"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "ippo_log.json", "w") as f:
        json.dump(m2_log, f, indent=2)

    print(f"\nResults saved to {results_dir}/")

    # ── Summary ───────────────────────────────────────────────────────────────
    if len(m2_log) >= 2:
        first = m2_log[0]
        last = m2_log[-1]
        print(f"\n{'='*60}")
        print(f"H4 Summary (IPPO, M2 retrieval_size={args.retrieval_size}):")
        print(f"  Diversity entropy : {first['diversity_entropy']:.3f} → {last['diversity_entropy']:.3f}")
        print(f"  Mean quality      : {first['mean_quality']:.3f} → {last['mean_quality']:.3f}")
        print(f"  Mean bait         : {first['mean_bait']:.3f} → {last['mean_bait']:.3f}")
        print(f"  Gini              : {first['gini']:.3f} → {last['gini']:.3f}")

        if m1_last is not None:
            gini_concentrated = last["gini"] > m1_last["gini"] * 1.3
            h4_confirmed = gini_concentrated
            print(f"\n  M1 baseline (H1): Gini={m1_last['gini']:.3f}")
            print(f"  M2 endpoint      : Gini={last['gini']:.3f}")
            print(f"\n  Gini ≥130% of M1 : {gini_concentrated}")
            print(f"\n  H4 {'CONFIRMED ✓' if h4_confirmed else 'NOT YET CONFIRMED — run more episodes'}")
        else:
            print(f"\n  (Run experiments/run_h1.py first to get M1 baseline for comparison)")
        print(f"{'='*60}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
