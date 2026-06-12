"""Generate the H4 figure: two-stage pipeline concentrates exposure.

Plots the Gini coefficient (exposure inequality) over training episodes under
M1 (single-stage, H1 baseline) and M2 (two-stage retrieval + engagement
reranker). A single panel saved to paper/figures/.

  M1 logs: results/h1_seed<seed>/ippo_log.json
  M2 logs: results/h4_m2_seed<seed>/ippo_log.json

Usage:
    python analysis/plot_h4.py                        # seed=42
    python analysis/plot_h4.py --seed 0
    python analysis/plot_h4.py --seeds 42 0 7 13 99   # multi-seed CI bands
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

FIGURE_DIR = ROOT / "paper" / "figures"


def load_log(results_dir: str, seed: int, algo: str = "ippo") -> list[dict[str, Any]]:
    path = ROOT / "results" / results_dir.format(seed=seed) / f"{algo}_log.json"
    if not path.exists():
        raise FileNotFoundError(f"No log found at {path}. Run the corresponding experiment first.")
    with open(path) as f:
        return json.load(f)


def _extract_series(
    logs: list[list[dict]], key: str, x_key: str = "episode"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align multiple seed logs and return (x, mean, std)."""
    min_len = min(len(log) for log in logs)
    xs = np.array([entry[x_key] for entry in logs[0][:min_len]])
    ys = np.array([[entry[key] for entry in log[:min_len]] for log in logs])
    return xs, ys.mean(axis=0), ys.std(axis=0)


def plot_h4(seeds: list[int], out_dir: Path = FIGURE_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    m1_logs = [load_log("h1_seed{seed}", s) for s in seeds]
    m2_logs = [load_log("h4_m2_seed{seed}", s) for s in seeds]

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(
        "H4: Two-stage pipeline concentrates exposure (M2 vs M1)",
        fontsize=13,
        fontweight="bold",
    )

    series = [
        ("M1 (single-stage)", m1_logs, "#1f77b4"),
        ("M2 (two-stage)", m2_logs, "#d62728"),
    ]

    for label, logs, color in series:
        xs, mean, std = _extract_series(logs, "gini")
        ax.plot(xs, mean, color=color, label=label, linewidth=2)
        if len(seeds) > 1:
            ax.fill_between(xs, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel("Training episode")
    ax.set_ylabel("Gini coefficient")
    ax.set_title("Exposure inequality across creators", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()

    seed_str = "_".join(str(s) for s in seeds)
    pdf_path = out_dir / f"h4_gini_seed{seed_str}.pdf"
    png_path = out_dir / f"h4_gini_seed{seed_str}.png"
    fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", help="Multiple seeds for CI bands")
    args = parser.parse_args()

    seeds = args.seeds if args.seeds else [args.seed]
    plot_h4(seeds)


if __name__ == "__main__":
    main()
