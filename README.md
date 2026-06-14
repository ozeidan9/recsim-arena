# recsim-arena

A small multi-agent simulator for studying **how recommender mechanism design shapes
creator equilibria** — bait vs. quality, content diversity, and exposure inequality.
Creators are independent PPO learners (IPPO) competing for exposure in front of a
clustered user population; the recommender is a swappable *mechanism*. We pre-register
four hypotheses and test which mechanism choices drive creators toward clickbait /
winner-take-all collapse versus quality and diversity.

📄 **Write-up:** see [`paper/recsim-arena.pdf`](paper/recsim-arena.pdf).

## Setup

Population: 50 users / 5 preference clusters / 20 creators / `content_dim=16` /
200 rounds per episode / `slate_size=5`. Click model:

```
logit = (u·c)/temp + alpha_quality*q − beta_bait*bait*fatigue
```

where bait accrues user fatigue and quality has a persistent positive effect.

### Mechanisms

| ID | Class | Description |
|----|-------|-------------|
| **M1** | `SingleStageMechanism` | Relevance + `bait_weight`, deterministic top-k. Baseline. |
| **M2** | `TwoStageMechanism` | numpy two-tower retrieval → online-trained engagement reranker MLP. |
| **M3** | `LLMGateMechanism` | M2 + a quality gate inserted before reranking. |

> **Note on M3:** the reported results use `SimpleLinearGate` (`alpha_quality=3`,
> `alpha_bait=-3`), an analytical linear proxy for an LLM judge. A live Anthropic gate
> (`LLMQualityGate`) and a `DistilledGate` exist in code but were **not** run for these
> results. M3 should be described as a quality gate / linear LLM-judge stand-in, never as
> live-LLM results.

## Hypotheses & headline results

Pre-registered hypotheses are in [`HYPOTHESES.md`](HYPOTHESES.md). Headline findings
(recompute from the JSON logs before citing):

- **H1 — Engagement collapse (IPPO on M1).** Engagement ranking drives a
  clickbait / winner-take-all collapse. Over **5 seeds [42, 0, 7, 13, 99]** the Gini
  collapse is **robust** (5/5 pass >+50%; mean +263%, sd 59). Bait inflation is
  **seed-fragile** (only seed 42 passes). Report as two effects of differing robustness.
- **H2 — Quality gate (M3 vs M1).** The gate shifts equilibrium toward quality.
  **Confirmed 5/5 seeds** — strongest, most robust result.
- **H3 — Temperature/exploration (M1 sweep, seed 42).** Recommender temperature governs
  specialization. Signal is **weak in diversity entropy but strong in bait** (final bait
  0.12 → ~0.99 as temperature rises). Report via bait.
- **H4 — Pipeline depth (M2 vs M1, seed 42).** Two-stage reranking concentrates exposure
  (Gini 0.187 vs 0.092). Single-seed.

Only H1 and H2 have multi-seed confidence intervals (5 seeds). H3 and H4 are single-seed
(42): no error bars.

## Repository layout

```
src/recsys_market/   env, mechanisms, agents (IPPO), retrieval, rerank, llm_gate, metrics
experiments/         run_h1.py ... run_h4.py, smoke_test.py
analysis/            plot_h1.py ... plot_h4.py
configs/             Hydra configs (env / mechanism / agent / experiment)
results/             per-run ippo_log.json + config.json (env/IPPO/mechanism params, git commit, seed)
paper/figures/       generated PDF + PNG figures
tests/               pytest invariants (one per env mechanic)
```

Each `results/<run>/config.json` records env / IPPO / mechanism params plus the git commit
and seed for reproducibility.

## Reproducing

Uses `uv` and Python 3.11+. Install deps with `uv sync`, then:

```bash
.venv/bin/python experiments/run_h1.py --seed <s> --n-episodes 300 --no-grd
.venv/bin/python experiments/run_h2.py --seed <s> --n-episodes 300        # M3
.venv/bin/python experiments/run_h4.py --seed 42 --n-episodes 300         # M2
.venv/bin/python experiments/run_h3.py --seed 42 --n-episodes 200         # temp sweep
.venv/bin/python -m pytest tests/ -q                                      # invariants
```

Regenerate figures with the matching `analysis/plot_h*.py` script.

## Limitations

1. **M3 is a linear proxy**, not a live LLM (see note above).
2. **H3/H4 are single-seed** (42) — no significance claims.
3. **Small scale** (20 creators, 50 users); retrieval uses numpy inner product rather than
   FAISS (intentional: PyTorch+FAISS OpenMP crash on macOS; equivalent at this scale).

## Conventions

Every experiment is a Hydra config + a W&B run with a fixed seed; every new env mechanic
gets a pytest invariant in `tests/`. See [`CLAUDE.md`](CLAUDE.md) for project constraints.
