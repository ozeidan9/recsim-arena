"""Shared run-provenance helper.

CLAUDE.md requires every experiment to save its config alongside results so the
run is reproducible (fixed seed + exact env/agent hyperparameters + code
version). This writes a `config.json` into the run's results directory capturing
the full configuration plus git commit, timestamp, and interpreter version.

The canonical env / IPPO hyperparameters are centralised here because they are
identical across H1–H4; experiment scripts override only what differs
(mechanism, mechanism_params, seed, n_episodes).
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Canonical environment configuration shared by H1–H4 (see each run_h*.make_env).
ENV_CONFIG: dict[str, Any] = {
    "n_users": 50,
    "n_populations": 5,
    "n_creators": 20,
    "content_dim": 16,
    "n_rounds": 200,
    "slate_size": 5,
    "quality_cost_scale": 0.5,
    "fatigue_gamma": 0.95,
    "alpha_quality": 0.3,
    "gamma_bait": 0.5,
    "beta_bait": 0.5,
}

# Canonical IPPO hyperparameters shared by H1–H4 (see each run_h*.run_ippo).
IPPO_CONFIG: dict[str, Any] = {
    "lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_coef": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "n_epochs": 4,
    "batch_size": 64,
    "hidden_dim": 64,
}


def _git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return None


def save_run_config(results_dir: str | Path, config: dict[str, Any]) -> Path:
    """Write config.json (with provenance) into results_dir; return its path."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        **config,
        "_provenance": {
            "git_commit": _git_commit(),
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version.split()[0],
        },
    }
    path = results_dir / "config.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path
