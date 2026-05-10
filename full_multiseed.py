import argparse
from pathlib import Path
from time import perf_counter

from api_rate_limiter_rl.experiments import build_env_config, build_ppo_config, run_experiments
from api_rate_limiter_rl.utils import ensure_dir, mean, stdev, write_json


RUNS = [
    {
        "name": "baseline",
        "env_overrides": {},
        "ppo_overrides": {},
    },
    {
        "name": "throughput_2_0",
        "env_overrides": {"reward_throughput_weight": 2.0},
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
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="full_runs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-episodes", type=int, default=260)
    parser.add_argument("--eval-episodes", type=int, default=24)
    parser.add_argument("--episode-length", type=int, default=180)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--runs", nargs="*", default=None)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--early-stop-window", type=int, default=0)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.05)
    return parser.parse_args()


def select_runs(run_names):
    if not run_names:
        return RUNS
    requested = set(run_names)
    return [run for run in RUNS if run["name"] in requested]


def aggregate_variant(variant_name, per_seed_rows):
    scenario_names = sorted(per_seed_rows[0]["scenario_metrics"])
    scenario_metrics = {}
    for scenario_name in scenario_names:
        scenario_metrics[scenario_name] = {}
        metric_names = sorted(per_seed_rows[0]["scenario_metrics"][scenario_name])
        for metric_name in metric_names:
            values = [row["scenario_metrics"][scenario_name][metric_name] for row in per_seed_rows]
            scenario_metrics[scenario_name][metric_name] = {
                "mean": mean(values),
                "stdev": stdev(values),
            }
    return {
        "name": variant_name,
        "seeds": [row["seed"] for row in per_seed_rows],
        "train_reward_mean": mean(row["final_train_reward"] for row in per_seed_rows),
        "train_reward_stdev": stdev(row["final_train_reward"] for row in per_seed_rows),
        "elapsed_seconds_mean": mean(row["elapsed_seconds"] for row in per_seed_rows),
        "scenario_metrics": scenario_metrics,
    }


def build_summary_markdown(aggregated_rows):
    lines = [
        "# Full Multi-Seed Summary",
        "",
        "| run | seeds | train reward mean | avg seconds | bursty reward | high_load reward | oscillating reward |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregated_rows:
        lines.append(
            "| {name} | {seed_count} | {train_reward_mean:.3f} | {elapsed_seconds_mean:.1f} | "
            "{bursty_reward:.3f} | {high_load_reward:.3f} | {oscillating_reward:.3f} |".format(
                name=row["name"],
                seed_count=",".join(str(seed) for seed in row["seeds"]),
                train_reward_mean=row["train_reward_mean"],
                elapsed_seconds_mean=row["elapsed_seconds_mean"],
                bursty_reward=row["scenario_metrics"]["bursty"]["ppo_total_reward_mean"]["mean"],
                high_load_reward=row["scenario_metrics"]["high_load"]["ppo_total_reward_mean"]["mean"],
                oscillating_reward=row["scenario_metrics"]["oscillating"]["ppo_total_reward_mean"]["mean"],
            )
        )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_dir))
    selected_runs = select_runs(args.runs)
    per_seed_rows = []
    aggregated_rows = []

    for run in selected_runs:
        variant_rows = []
        for seed in args.seeds:
            run_name = run["name"]
            run_dir = ensure_dir(output_dir / run_name / f"seed_{seed}")
            env_cfg = build_env_config(
                episode_length=args.episode_length,
                seed=seed,
                **run["env_overrides"],
            )
            ppo_cfg = build_ppo_config(
                train_episodes=args.train_episodes,
                eval_episodes=args.eval_episodes,
                device=args.device,
                seed=seed,
                early_stop_window=args.early_stop_window,
                early_stop_patience=args.early_stop_patience,
                early_stop_min_delta=args.early_stop_min_delta,
                **run["ppo_overrides"],
            )
            print(f"[full] start run={run_name} seed={seed}", flush=True)
            started_at = perf_counter()
            result = run_experiments(
                output_dir=str(run_dir),
                env_cfg=env_cfg,
                ppo_cfg=ppo_cfg,
                log_every=args.log_every,
            )
            training_history = result["training_history"]
            evaluation_results = result["evaluation_results"]
            row = {
                "name": run_name,
                "seed": seed,
                "elapsed_seconds": perf_counter() - started_at,
                "final_train_reward": mean(training_history["episode_reward"][-20:]),
                "scenario_metrics": {
                    scenario_name: {
                        "ppo_total_reward_mean": scenario_results["ppo"]["total_reward"]["mean"],
                        "ppo_latency_mean": scenario_results["ppo"]["latency"]["mean"],
                        "ppo_drop_rate_mean": scenario_results["ppo"]["drop_rate"]["mean"],
                        "ppo_throughput_mean": scenario_results["ppo"]["throughput"]["mean"],
                        "ppo_high_priority_success_mean": scenario_results["ppo"]["high_priority_success_rate"]["mean"],
                    }
                    for scenario_name, scenario_results in evaluation_results.items()
                },
            }
            write_json(run_dir / "seed_summary.json", row)
            per_seed_rows.append(row)
            variant_rows.append(row)
            print(
                f"[full] done run={run_name} seed={seed} "
                f"final_train_reward={row['final_train_reward']:.3f}",
                flush=True,
            )
        aggregated = aggregate_variant(run["name"], variant_rows)
        aggregated_rows.append(aggregated)

    write_json(output_dir / "per_seed_summary.json", per_seed_rows)
    write_json(output_dir / "aggregate_summary.json", aggregated_rows)
    (output_dir / "aggregate_summary.md").write_text(build_summary_markdown(aggregated_rows), encoding="utf-8")
    print(output_dir / "aggregate_summary.md")


if __name__ == "__main__":
    main()
