"""H2 validation: LLM quality gate shifts equilibrium toward quality/diversity.

Compares IPPO creator equilibria under:
  M1  (H1 baseline): single-stage, promotes bait
  M3  (H2 test):     two-stage + quality gate, penalises bait

H2 prediction: at equilibrium, M3 shows lower bait AND higher quality than M1.
The gate removes the bait shortcut that M1 exploits, forcing creators toward
substantive quality as the route to exposure.

Confirmation criterion (H2 confirmed):
  - Final mean bait under M3 < M1 endpoint * 0.85  (≥15% reduction)
  - AND final mean quality under M3 > M1 endpoint * 1.05  (≥5% improvement)

M1 baseline is loaded from results/h1_seed{seed}/ippo_log.json if present.

Usage:
    python experiments/run_h2.py
    python experiments/run_h2.py --n-episodes 300 --seed 42
    python experiments/run_h2.py --wandb
    python experiments/run_h2.py --gate-alpha-quality 4.0 --gate-alpha-bait -4.0
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


def make_env(seed: int, mechanism_name: str = "m3"):
    from recsys_market.env.market_env import ContentMarketEnv
    from recsys_market.llm_gate.gate import SimpleLinearGate
    from recsys_market.mechanisms.m1_single import SingleStageMechanism
    from recsys_market.mechanisms.m2_two_stage import TwoStageMechanism
    from recsys_market.mechanisms.m3_llm_gate import LLMGateMechanism

    content_dim = 16

    if mechanism_name == "m1":
        mech = SingleStageMechanism(temperature=1.0, bait_weight=0.5)
    elif mechanism_name == "m2":
        mech = TwoStageMechanism(content_dim=content_dim, retrieval_size=10)
    else:  # m3
        gate = SimpleLinearGate(alpha_quality=3.0, alpha_bait=-3.0, threshold=0.5)
        mech = LLMGateMechanism(
            content_dim=content_dim,
            retrieval_size=10,
            gate=gate,
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

    env = make_env(args.seed, mechanism_name)
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
    print(f"H2 — IPPO on {label} | episodes={args.n_episodes} | seed={args.seed}")
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

    ckpt_dir = ROOT / "results" / f"h2_{mechanism_name}_seed{args.seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    trainer.save(str(ckpt_dir / "ippo_final.pt"))

    return log


def load_m1_baseline(seed: int) -> dict | None:
    """Load last entry from H1 IPPO log if it exists."""
    path = ROOT / "results" / f"h1_seed{seed}" / "ippo_log.json"
    if not path.exists():
        return None
    with open(path) as f:
        log = json.load(f)
    return log[-1] if log else None


def main() -> None:
    parser = argparse.ArgumentParser(description="H2 validation experiment")
    parser.add_argument("--n-episodes", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mechanism", choices=["m3", "m2", "m1"], default="m3",
                        help="Mechanism to compare against M1 baseline")
    parser.add_argument("--gate-alpha-quality", type=float, default=3.0)
    parser.add_argument("--gate-alpha-bait", type=float, default=-3.0)
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    args = parser.parse_args()

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project="recsim-arena",
            name=f"h2_{args.mechanism}_seed{args.seed}",
            config=vars(args),
        )

    # Run the target mechanism
    target_log = run_ippo(args, args.mechanism, wandb_run)

    # Load M1 baseline (from H1 run, if available)
    m1_last = load_m1_baseline(args.seed)

    # ── Save results ─────────────────────────────────────────────────────────
    results_dir = ROOT / "results" / f"h2_{args.mechanism}_seed{args.seed}"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "ippo_log.json", "w") as f:
        json.dump(target_log, f, indent=2)

    from run_meta import ENV_CONFIG, IPPO_CONFIG, save_run_config
    if args.mechanism == "m3":
        mech_params = {
            "retrieval_size": 10,
            "gate": "SimpleLinearGate",
            "gate_alpha_quality": args.gate_alpha_quality,
            "gate_alpha_bait": args.gate_alpha_bait,
            "gate_threshold": 0.5,
        }
    elif args.mechanism == "m2":
        mech_params = {"retrieval_size": 10}
    else:
        mech_params = {"temperature": 1.0, "bait_weight": 0.5}
    save_run_config(results_dir, {
        "experiment": "h2",
        "mechanism": args.mechanism,
        "mechanism_params": mech_params,
        "seed": args.seed,
        "n_episodes": args.n_episodes,
        "eval_every": args.eval_every,
        "env": ENV_CONFIG,
        "ippo": IPPO_CONFIG,
    })

    print(f"\nResults saved to {results_dir}/")

    # ── Summary ───────────────────────────────────────────────────────────────
    if len(target_log) >= 2:
        first = target_log[0]
        last = target_log[-1]
        print(f"\n{'='*60}")
        print(f"H2 Summary (IPPO, {args.mechanism.upper()}):")
        print(f"  Diversity entropy : {first['diversity_entropy']:.3f} → {last['diversity_entropy']:.3f}")
        print(f"  Mean quality      : {first['mean_quality']:.3f} → {last['mean_quality']:.3f}")
        print(f"  Mean bait         : {first['mean_bait']:.3f} → {last['mean_bait']:.3f}")
        print(f"  Gini              : {first['gini']:.3f} → {last['gini']:.3f}")

        if m1_last is not None:
            bait_reduction = last["mean_bait"] < m1_last["mean_bait"] * 0.85
            quality_improvement = last["mean_quality"] > m1_last["mean_quality"] * 1.05
            h2_confirmed = bait_reduction and quality_improvement
            print(f"\n  M1 baseline (H1): bait={m1_last['mean_bait']:.3f}  quality={m1_last['mean_quality']:.3f}")
            print(f"  {args.mechanism.upper()} endpoint  : bait={last['mean_bait']:.3f}  quality={last['mean_quality']:.3f}")
            print(f"\n  Bait reduction ≥15%  : {bait_reduction}")
            print(f"  Quality gain   ≥5%   : {quality_improvement}")
            print(f"\n  H2 {'CONFIRMED ✓' if h2_confirmed else 'NOT YET CONFIRMED — run more episodes'}")
        else:
            print(f"\n  (Run experiments/run_h1.py first to get M1 baseline for comparison)")
        print(f"{'='*60}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
