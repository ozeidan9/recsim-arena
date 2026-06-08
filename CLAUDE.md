# recsim-arena

Stack: Python 3.11+, uv, PyTorch, PettingZoo, Gymnasium, FAISS, Hydra, W&B, Anthropic API.
Every experiment is a Hydra config + a W&B run with a seed.
Every new env mechanic gets a pytest invariant in tests/.
Don't modify configs/ without asking me first.
Reproduce H1 (engagement→clickbait collapse on M1 single-stage) before trusting any M3 results.
Keep all experiments reproducible: fixed seeds, logged to W&B, config saved alongside results.
