from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_training_curves(history: Dict[str, List[float]], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    axes[0].plot(history["episode_reward"], color="#0B6E4F", linewidth=1.5)
    axes[0].set_title("Episode Reward")
    axes[0].set_xlabel("Episode")

    axes[1].plot(history["episode_latency"], color="#C75146", linewidth=1.5)
    axes[1].set_title("Average Latency")
    axes[1].set_xlabel("Episode")

    axes[2].plot(history["episode_throughput"], color="#1F77B4", linewidth=1.5)
    axes[2].set_title("Throughput")
    axes[2].set_xlabel("Episode")

    axes[3].plot(history["episode_drop_rate"], color="#8E5C2A", linewidth=1.5)
    axes[3].set_title("Drop Rate")
    axes[3].set_xlabel("Episode")

    for ax in axes:
        ax.grid(True, alpha=0.25)

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

