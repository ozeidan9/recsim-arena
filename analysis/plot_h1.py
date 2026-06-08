"""Generate the H1 figure: engagement collapse on single-stage mechanism.

Reads results from results/h1_seed<seed>/ippo_log.json (and optionally
grd_log.json) and produces a 2×2 panel figure saved to paper/figures/.

Usage:
    python analysis/plot_h1.py                        # seed=42
    python analysis/plot_h1.py --seed 0
    python analysis/plot_h1.py --seeds 42 0 7         # multi-seed average
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


def load_log(seed: int, algo: str = "ippo") -> list[dict[str, Any]]:
    path = ROOT / "results" / f"h1_seed{seed}" / f"{algo}_log.json"
    if not path.exists():
        raise FileNotFoundError(f"No log found at {path}. Run experiments/run_h1.py first.")
    with open(path) as f:
        return json.load(f)


def _extract_series(
    logs: list[list[dict]], key: str, x_key: str = "episode"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align multiple seed logs and return (x, mean, std)."""
    # Use the shortest run's x-axis
    min_len = min(len(log) for log in logs)
    xs = np.array([entry[x_key] for entry in logs[0][:min_len]])
    ys = np.array([[entry[key] for entry in log[:min_len]] for log in logs])
    return xs, ys.mean(axis=0), ys.std(axis=0)


def plot_h1(seeds: list[int], out_dir: Path = FIGURE_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load IPPO logs
    ippo_logs = [load_log(s, "ippo") for s in seeds]

    # Load GRD logs if available
    grd_logs = []
    for s in seeds:
        try:
            grd_logs.append(load_log(s, "grd"))
        except FileNotFoundError:
            pass
    has_grd = len(grd_logs) == len(seeds)

    metrics = [
        ("diversity_entropy", "Diversity entropy", r"$H$ (nats)"),
        ("mean_quality", "Mean quality", "Quality"),
        ("mean_bait", "Mean bait", "Bait level"),
        ("gini", "Gini (exposure inequality)", "Gini coefficient"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(
        "H1: Engagement collapse under single-stage relevance ranking (M1)",
        fontsize=13,
        fontweight="bold",
    )

    palette = {"ippo": "#1f77b4", "grd": "#d62728"}

    for ax, (key, title, ylabel) in zip(axes.flat, metrics):
        xs, mean, std = _extract_series(ippo_logs, key)
        ax.plot(xs, mean, color=palette["ippo"], label="IPPO", linewidth=2)
        if len(seeds) > 1:
            ax.fill_between(xs, mean - std, mean + std, alpha=0.2, color=palette["ippo"])

        if has_grd:
            gxs, gmean, gstd = _extract_series(grd_logs, key, x_key="round")
            ax.plot(gxs, gmean, color=palette["grd"], label="GRD baseline",
                    linewidth=1.5, linestyle="--")
            if len(seeds) > 1:
                ax.fill_between(gxs, gmean - gstd, gmean + gstd, alpha=0.15, color=palette["grd"])

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Training episode")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if has_grd:
            ax.legend(fontsize=9)

    plt.tight_layout()

    seed_str = "_".join(str(s) for s in seeds)
    pdf_path = out_dir / f"h1_collapse_seed{seed_str}.pdf"
    png_path = out_dir / f"h1_collapse_seed{seed_str}.png"
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
    plot_h1(seeds)


if __name__ == "__main__":
    main()
