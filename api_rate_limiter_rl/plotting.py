from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_training_curves(history: Dict[str, List[float]], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("episode_reward", "Episode Reward", "#0B6E4F"),
        ("episode_latency", "Average Latency", "#C75146"),
        ("episode_throughput", "Throughput", "#1F77B4"),
        ("episode_drop_rate", "Drop Rate", "#8E5C2A"),
    ]
    if "episode_served_cost" in history:
        metrics.append(("episode_served_cost", "Processed Cost", "#6C5CE7"))
    if "episode_budget_utilization" in history:
        metrics.append(("episode_budget_utilization", "Budget Utilization", "#FF7F11"))

    ncols = 2
    nrows = (len(metrics) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    axes = axes.ravel()
    for ax, (key, title, color) in zip(axes, metrics):
        ax.plot(history[key], color=color, linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.grid(True, alpha=0.25)

    for ax in axes[len(metrics) :]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=160)
    fig.savefig(output_dir / "training_curves.svg")
    plt.close(fig)


def plot_comparison(results: Dict[str, Dict[str, float]], metric: str, output_dir: Path, title: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    names = list(results.keys())
    values = [results[name][metric] for name in names]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(names, values, color=["#2E86AB", "#F6AE2D", "#A23B72", "#3D5A80"])
    ax.set_title(title)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=18)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / f"{metric}.png", dpi=160)
    fig.savefig(output_dir / f"{metric}.svg")
    plt.close(fig)
