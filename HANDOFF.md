# recsim-arena — Handoff for the write-up / analysis agent

The experiments are **done and committed** (`main` @ the commit that adds this
file). Your job: analyze the saved results and draft the paper's
results/analysis sections. **Do NOT re-run training unless explicitly asked** —
the data is final. Use `.venv/bin/python` (Python 3.14).

## What the project studies
How recommender mechanism design shapes creator equilibria — bait vs. quality,
content diversity, and exposure inequality. 50 users / 5 preference clusters /
20 creators / `content_dim=16` / 200 rounds per episode / `slate_size=5`.
Creators are independent PPO learners (IPPO). Click model:

```
logit = (u·c)/temp + alpha_quality*q - beta_bait*bait*fatigue
```

where bait accrues user fatigue and quality has a persistent positive effect.

## Mechanisms
- **M1** `SingleStageMechanism` — relevance + `bait_weight`, deterministic top-k. Baseline.
- **M2** `TwoStageMechanism` — numpy two-tower retrieval → online-trained engagement reranker MLP.
- **M3** `LLMGateMechanism` — M2 + a quality gate inserted before reranking.

## Hypotheses & results
Numbers below are the headline; **recompute from the JSON logs, don't trust this summary blindly.**

- **H1** (IPPO on M1): engagement ranking → clickbait / winner-take-all collapse.
  **5 seeds [42, 0, 7, 13, 99]:** Gini collapse **ROBUST** (5/5 pass >+50%; mean **+263%**, sd 59).
  Bait inflation **SEED-FRAGILE** (only seed 42 passes >+30% at +44%; seeds 7/13/99 bait *decreased*).
  Report H1 as **two effects of different robustness**; do not present bait inflation as a general result.
- **H2** (M3 vs M1): quality gate shifts equilibrium toward quality.
  **CONFIRMED 5/5 seeds.** M3 final bait ~0.03–0.09 vs M1 ~0.46–0.67; M3 quality ~0.87–0.97 vs M1 ~0.40–0.56.
  This is the strongest, most robust result.
- **H4** (M2 vs M1): two-stage reranker feedback loop concentrates exposure.
  **(seed 42 only) CONFIRMED**, M2 Gini 0.187 vs M1 0.092 (203%).
- **H3** (M1 temperature sweep [0.1, 0.3, 1, 3, 10], seed 42 only): temperature governs specialization.
  Predicted signal is **WEAK in `diversity_entropy`** (all ~1.84–2.0, non-monotonic) but **STRONG in bait** —
  final bait 0.12 (temp 0.1) → 0.82 (temp 1) → ~0.99 (temp 3, 10).
  Report H3 **via bait**: sharp ranking forces relevance-matching/specialization (low bait); flat ranking
  lets the bait term dominate.

## Critical honesty constraints (do not overclaim)
1. **M3 used `SimpleLinearGate`** (`alpha_quality=3`, `alpha_bait=-3`), an **analytical linear proxy**.
   No live LLM and no distillation were run. `LLMQualityGate` (live Anthropic) and `DistilledGate` exist
   in code but were never executed. Describe M3 as a quality gate / linear LLM-judge stand-in —
   **never as live-LLM results.**
2. **Only H1 and H2 have multi-seed CI** (5 seeds). **H4 and H3 are single-seed (42):** no error bars,
   no significance claims for them.
3. **Scale is small** (20 creators, 50 users). Retrieval uses **numpy inner product, not FAISS**
   (intentional: PyTorch+FAISS OpenMP crash on macOS; mathematically equivalent at this scale).

## Where everything is
- `results/h1_seed{42,0,7,13,99}/ippo_log.json` — per-episode metrics. Keys:
  `diversity_entropy, mean_quality, mean_bait, gini, ild, mean_ep_reward, total_ep_reward,
  policy_loss, value_loss, entropy, episode`.
- `results/h2_m3_seed{42,0,7,13,99}/ippo_log.json`
- `results/h4_m2_seed42/ippo_log.json`
- `results/h3_temp{0.1,0.3,1,3,10}_seed42/ippo_log.json` + `results/h3_summary_seed42.json`
- Every run dir also has `config.json` (env / IPPO / mechanism params + git commit + seed) —
  cite these for reproducibility.
- Figures (PDF + PNG) in `paper/figures/`: `h1_collapse_seed42_0_7_13_99`,
  `h2_gate_seed42_0_7_13_99`, `h4_gini_seed42`, `h3_temp_seed42` (2 panels: entropy + bait).
- Code: `experiments/run_h{1,2,3,4}.py`, `analysis/plot_h{1,2,3,4}.py`,
  `src/recsys_market/` (env, mechanisms, agents, metrics). `CLAUDE.md` has project constraints.

## Deliverables
1. A **Results** section per hypothesis with the numbers above, reading directly from the JSON logs.
2. A **Limitations** section covering the three honesty constraints.
3. A short **Future work** noting: run the live `LLMQualityGate` + `DistilledGate` path for M3;
   add multi-seed CI for H3/H4; scale up creators/users.

**Verify any number you cite by loading the corresponding `ippo_log.json`.**

## Reproducing (if ever needed)
```
.venv/bin/python experiments/run_h1.py --seed <s> --n-episodes 300 --no-grd
.venv/bin/python experiments/run_h2.py --seed <s> --n-episodes 300          # M3
.venv/bin/python experiments/run_h4.py --seed 42 --n-episodes 300           # M2
.venv/bin/python experiments/run_h3.py --seed 42 --n-episodes 200           # temp sweep
.venv/bin/python -m pytest tests/ -q                                        # 77/77
```
