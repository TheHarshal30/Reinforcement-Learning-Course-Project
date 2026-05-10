import argparse
from pathlib import Path

from api_rate_limiter_rl.experiments import (
    build_env_config,
    build_ppo_config,
    quick_screen_variant,
)
from api_rate_limiter_rl.utils import ensure_dir, write_json


RUNS = [
    {
        "name": "baseline",
        "env_overrides": {},
        "ppo_overrides": {},
    },
    {
        "name": "hp_penalty_1_0",
        "env_overrides": {"reward_high_drop_weight": 1.0},
        "ppo_overrides": {},
    },
    {
        "name": "hp_penalty_1_5",
        "env_overrides": {"reward_high_drop_weight": 1.5},
        "ppo_overrides": {},
    },
    {
        "name": "throughput_2_0",
        "env_overrides": {"reward_throughput_weight": 2.0},
        "ppo_overrides": {},
    },
    {
        "name": "hidden_32",
        "env_overrides": {},
        "ppo_overrides": {"hidden_size": 32},
    },
    {
        "name": "hidden_64",
        "env_overrides": {},
        "ppo_overrides": {"hidden_size": 64},
    },
    {
        "name": "queue_trend",
        "env_overrides": {"include_queue_trend": True},
        "ppo_overrides": {},
    },
    {
        "name": "queue_arrival_trend",
        "env_overrides": {
            "include_queue_trend": True,
            "include_arrival_trend": True,
        },
        "ppo_overrides": {},
    },
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="screening")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-episodes", type=int, default=30)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--episode-length", type=int, default=100)
    parser.add_argument("--eval-scenario", default="bursty")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--runs", nargs="*", default=None)
    return parser.parse_args()


def select_runs(run_names):
    if not run_names:
        return RUNS
    requested = set(run_names)
    return [run for run in RUNS if run["name"] in requested]


def build_summary_markdown(results):
    lines = [
        "# Quick Screen Summary",
        "",
        "| run | episodes | eval scenario | train reward (recent) | eval reward | throughput | latency | drop rate | hp drop | served cost | utilization | seconds |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        lines.append(
            "| {name} | {train_episodes} | {eval_scenario} | {recent_training_reward:.3f} | "
            "{eval_reward_mean:.3f} | {eval_throughput_mean:.3f} | {eval_latency_mean:.3f} | "
            "{eval_drop_rate_mean:.3f} | {eval_high_priority_drop_mean:.3f} | "
            "{eval_served_cost_mean:.3f} | {eval_budget_utilization_mean:.3f} | {elapsed_seconds:.1f} |".format(
                **result
            )
        )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    selected_runs = select_runs(args.runs)
    output_dir = ensure_dir(Path(args.output_dir))
    results = []

    for run in selected_runs:
        run_name = run["name"]
        env_cfg = build_env_config(
            episode_length=args.episode_length,
            **run["env_overrides"],
        )
        ppo_cfg = build_ppo_config(
            train_episodes=args.train_episodes,
            eval_episodes=args.eval_episodes,
            device=args.device,
            early_stop_window=10,
            early_stop_patience=2,
            early_stop_min_delta=0.05,
            **run["ppo_overrides"],
        )
        print(f"[screen] start {run_name}", flush=True)
        summary, training_history, evaluation = quick_screen_variant(
            run_name,
            env_cfg,
            ppo_cfg,
            eval_scenario_name=args.eval_scenario,
            eval_episodes=args.eval_episodes,
            log_every=args.log_every,
        )
        run_dir = ensure_dir(output_dir / run_name)
        write_json(run_dir / "summary.json", summary)
        write_json(run_dir / "training_history.json", training_history)
        write_json(run_dir / "evaluation.json", evaluation)
        results.append(summary)
        print(
            f"[screen] done {run_name} "
            f"eval_reward={summary['eval_reward_mean']:.3f} "
            f"throughput={summary['eval_throughput_mean']:.3f} "
            f"latency={summary['eval_latency_mean']:.3f} "
            f"drop={summary['eval_drop_rate_mean']:.3f}",
            flush=True,
        )

    write_json(output_dir / "quick_screen_summary.json", results)
    (output_dir / "quick_screen_summary.md").write_text(build_summary_markdown(results), encoding="utf-8")
    print(output_dir / "quick_screen_summary.md")


if __name__ == "__main__":
    main()
