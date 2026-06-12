"""H3 validation: ranking temperature governs creator specialization.

Sweeps the SingleStageMechanism (M1) temperature and trains IPPO creators
at each setting, tracking the final diversity entropy of exposure.

H3 prediction: lower temperature → sharper (more winner-take-all) ranking →
creators concentrate → lower diversity entropy. Higher temperature → flatter
ranking → more exploration → higher diversity entropy. Diversity entropy
should increase (roughly monotonically) with temperature.

Each temperature is its own M1 run with bait_weight held fixed at 0.5, so the
only thing that varies across the sweep is the temperature. Logs for each
temperature are saved to results/h3_temp{t}_seed{seed}/ippo_log.json so the
analysis script (analysis/plot_h3.py) can read final entropy per temperature.

Usage:
    python experiments/run_h3.py                                   # defaults
    python experiments/run_h3.py --seed 42 --n-episodes 200
    python experiments/run_h3.py --temperatures 0.1 1.0 10.0
    python experiments/run_h3.py --wandb
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

DEFAULT_TEMPERATURES = [0.1, 0.3, 1.0, 3.0, 10.0]


def temp_tag(t: float) -> str:
    """Stable, filesystem-friendly tag for a temperature value (matches CLI)."""
    return f"{t:g}"


def make_env(seed: int, temperature: float):
    from recsys_market.env.market_env import ContentMarketEnv
    from recsys_market.mechanisms.m1_single import SingleStageMechanism

    mech = SingleStageMechanism(temperature=temperature, bait_weight=0.5)
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


def run_ippo(args, temperature: float, wandb_run=None) -> list[dict]:
    from recsys_market.agents.ippo import IPPOTrainer

    env = make_env(args.seed, temperature)
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
    print(f"H3 — IPPO on M1 | temp={temperature:g} | episodes={args.n_episodes} | seed={args.seed}")
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
            wandb_run.log({f"temp{temp_tag(temperature)}/" + k: v for k, v in entry.items()})

    ckpt_dir = ROOT / "results" / f"h3_temp{temp_tag(temperature)}_seed{args.seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    trainer.save(str(ckpt_dir / "ippo_final.pt"))

    return log


def main() -> None:
    parser = argparse.ArgumentParser(description="H3 temperature-sweep experiment")
    parser.add_argument("--n-episodes", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=DEFAULT_TEMPERATURES,
        help="Temperature values to sweep over",
    )
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    args = parser.parse_args()

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project="recsim-arena",
            name=f"h3_tempsweep_seed{args.seed}",
            config=vars(args),
        )

    summary: list[tuple[float, float]] = []  # (temperature, final diversity_entropy)

    for temperature in args.temperatures:
        log = run_ippo(args, temperature, wandb_run)

        results_dir = ROOT / "results" / f"h3_temp{temp_tag(temperature)}_seed{args.seed}"
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / "ippo_log.json", "w") as f:
            json.dump(log, f, indent=2)

        from run_meta import ENV_CONFIG, IPPO_CONFIG, save_run_config
        save_run_config(results_dir, {
            "experiment": "h3",
            "mechanism": "m1",
            "mechanism_params": {"temperature": temperature, "bait_weight": 0.5},
            "seed": args.seed,
            "n_episodes": args.n_episodes,
            "eval_every": args.eval_every,
            "env": ENV_CONFIG,
            "ippo": IPPO_CONFIG,
        })
        print(f"Results saved to {results_dir}/")

        final_entropy = log[-1]["diversity_entropy"] if log else float("nan")
        summary.append((temperature, final_entropy))

    # ── Sweep summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"H3 Summary (IPPO on M1, final diversity entropy by temperature):")
    print(f"{'='*60}")
    print(f"  {'temperature':>12} | {'final entropy':>14}")
    print(f"  {'-'*12} | {'-'*14}")
    for temperature, entropy in summary:
        print(f"  {temperature:>12g} | {entropy:>14.4f}")

    # Monotonicity check: does diversity entropy increase with temperature?
    entropies = [e for _, e in summary]
    if len(entropies) >= 2:
        diffs = np.diff(entropies)
        monotonic = bool(np.all(diffs >= 0))
        increasing_overall = entropies[-1] > entropies[0]
        print(f"\n  Monotonically increasing : {monotonic}")
        print(f"  Net increase (hi vs lo)  : {increasing_overall} "
              f"({entropies[0]:.4f} → {entropies[-1]:.4f})")
        print(f"  H3 direction {'CONFIRMED ✓' if increasing_overall else 'NOT in predicted direction'}")
    print(f"{'='*60}")

    # Save a machine-readable sweep summary alongside the per-temp logs.
    summary_path = ROOT / "results" / f"h3_summary_seed{args.seed}.json"
    with open(summary_path, "w") as f:
        json.dump(
            [{"temperature": t, "final_diversity_entropy": e} for t, e in summary],
            f,
            indent=2,
        )
    print(f"\nSweep summary saved to {summary_path}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
