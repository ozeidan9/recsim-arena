"""Generate the H3 figure: ranking temperature governs specialization.

Plots final diversity entropy (y) against the M1 ranking temperature (x, log
scale), one point per temperature from the H3 sweep. A single panel saved to
paper/figures/.

  Per-temperature logs: results/h3_temp<t>_seed<seed>/ippo_log.json

H3 prediction: diversity entropy rises with temperature (sharper ranking →
concentration → lower entropy; flatter ranking → exploration → higher entropy).

Usage:
    python analysis/plot_h3.py                                  # seed=42
    python analysis/plot_h3.py --seed 0
    python analysis/plot_h3.py --temperatures 0.1 0.3 1.0 3.0 10.0
    python analysis/plot_h3.py --seeds 42 0 7 13 99             # multi-seed CI
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

DEFAULT_TEMPERATURES = [0.1, 0.3, 1.0, 3.0, 10.0]


def temp_tag(t: float) -> str:
    """Must match experiments/run_h3.py temp_tag for path consistency."""
    return f"{t:g}"


def load_final_metric(temperature: float, seed: int, key: str) -> float:
    path = ROOT / "results" / f"h3_temp{temp_tag(temperature)}_seed{seed}" / "ippo_log.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No log found at {path}. Run experiments/run_h3.py first."
        )
    with open(path) as f:
        log: list[dict[str, Any]] = json.load(f)
    return log[-1][key]


def _agg(temperatures: list[float], seeds: list[int], key: str) -> tuple[np.ndarray, np.ndarray]:
    means, stds = [], []
    for t in temperatures:
        vals = np.array([load_final_metric(t, s, key) for s in seeds])
        means.append(vals.mean())
        stds.append(vals.std())
    return np.array(means), np.array(stds)


def plot_h3(temperatures: list[float], seeds: list[int], out_dir: Path = FIGURE_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # The specialization signal is weak in content-direction entropy but strong
    # in bait, so plot both: entropy (left, as pre-registered) and bait (right).
    panels = [
        ("diversity_entropy", r"Final diversity entropy $H$ (nats)",
         "Content-direction entropy (weak signal)", "#9467bd"),
        ("mean_bait", "Final mean bait",
         "Bait equilibrium (strong signal)", "#d62728"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "H3: Temperature governs specialization",
        fontsize=13,
        fontweight="bold",
    )

    for ax, (key, ylabel, title, color) in zip(axes, panels):
        means_arr, stds_arr = _agg(temperatures, seeds, key)
        if len(seeds) > 1:
            ax.errorbar(
                temperatures, means_arr, yerr=stds_arr,
                marker="o", markersize=7, linewidth=2, capsize=4, color=color,
            )
        else:
            ax.plot(
                temperatures, means_arr,
                marker="o", markersize=8, linewidth=2, color=color,
            )
        ax.set_xscale("log")
        ax.set_xlabel("Ranking temperature (log scale)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()

    seed_str = "_".join(str(s) for s in seeds)
    pdf_path = out_dir / f"h3_temp_seed{seed_str}.pdf"
    png_path = out_dir / f"h3_temp_seed{seed_str}.png"
    fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", help="Multiple seeds for CI bands")
    parser.add_argument(
        "--temperatures", type=float, nargs="+", default=DEFAULT_TEMPERATURES,
        help="Temperature values to plot (must match the sweep that was run)",
    )
    args = parser.parse_args()

    seeds = args.seeds if args.seeds else [args.seed]
    plot_h3(args.temperatures, seeds)


if __name__ == "__main__":
    main()
