import argparse
import json
from pathlib import Path

from api_rate_limiter_rl.config import default_scenarios
from api_rate_limiter_rl.experiments import build_env_config, evaluate_policy, make_baselines
from api_rate_limiter_rl.env import RateLimitEnv
from api_rate_limiter_rl.ppo import PPOAgent
from api_rate_limiter_rl.utils import ensure_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-json", default="azure_trace_artifacts/azure_cost_profile.json")
    parser.add_argument("--output-dir", default="azure_transfer_eval")
    parser.add_argument("--eval-episodes", type=int, default=24)
    parser.add_argument("--episode-length", type=int, default=180)
    parser.add_argument(
        "--cost-only-checkpoint",
        default="cost_aware_finetune_run/base_checkpoint.pt",
    )
    parser.add_argument(
        "--warm-start-checkpoint",
        default="warm_start_reg_final/warm_started_ppo.pt",
    )
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def load_profile(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_azure_env_config(profile, episode_length: int, seed: int):
    clip_max = float(profile["recommended_max_expected_service_cost"])
    queue_cost = float(profile["recommended_max_expected_queue_cost"])
    return build_env_config(
        episode_length=episode_length,
        reward_high_drop_weight=1.5,
        use_request_costs=True,
        use_azure_cost=True,
        azure_cost_values=tuple(int(v) for v in profile["azure_cost_values"]),
        azure_cost_probabilities=tuple(float(v) for v in profile["azure_cost_probabilities"]),
        service_budget_per_step=24.0,
        max_expected_queue_cost=queue_cost,
        max_expected_service_cost=clip_max,
        reward_throughput_weight=0.0,
        reward_cost_processed_weight=1.5,
        reward_budget_utilization_weight=0.0,
        include_rejected_cost=True,
        seed=seed,
    )


def load_agent(checkpoint_path: Path, env_cfg, seed: int):
    warmup_env = RateLimitEnv(default_scenarios()["high_load"], env_cfg, seed=seed)
    agent = PPOAgent(
        state_dim=len(warmup_env.reset(seed)),
        action_dim=3,
        hidden_dim=48,
        actor_lr=0.001,
        critic_lr=0.0015,
        seed=seed,
        device="cpu",
    )
    agent.load(checkpoint_path)
    return agent


def main():
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_dir))
    profile = load_profile(Path(args.profile_json))
    env_cfg = build_azure_env_config(profile, args.episode_length, args.seed)

    cost_only_agent = load_agent(Path(args.cost_only_checkpoint), env_cfg, args.seed)
    warm_start_agent = load_agent(Path(args.warm_start_checkpoint), env_cfg, args.seed)
    baselines = make_baselines(env_cfg)

    scenarios = {name: cfg for name, cfg in default_scenarios().items() if name != "training_mix"}
    policies = {
        "ppo_cost_only": cost_only_agent,
        "ppo_warm_start_reg_l010": warm_start_agent,
        "cost_aware_priority": baselines["cost_aware_priority"],
        "priority_heuristic": baselines["priority_heuristic"],
    }

    results = {}
    for scenario_name, scenario in scenarios.items():
        results[scenario_name] = {
            policy_name: evaluate_policy(policy, scenario, env_cfg, episodes=args.eval_episodes, seed=args.seed)
            for policy_name, policy in policies.items()
        }

    summary_lines = ["# Azure Cost Transfer Evaluation", ""]
    for scenario_name, scenario_results in results.items():
        summary_lines.append(f"## {scenario_name}")
        for policy_name, metrics in scenario_results.items():
            summary_lines.append(
                f"- {policy_name}: reward={metrics['total_reward']['mean']:.3f}, "
                f"throughput={metrics['throughput']['mean']:.3f}, "
                f"latency={metrics['latency']['mean']:.3f}, "
                f"drop={metrics['drop_rate']['mean']:.3f}, "
                f"hp_success={metrics['high_priority_success_rate']['mean']:.3f}, "
                f"served_cost={metrics['served_cost']['mean']:.3f}, "
                f"budget_utilization={metrics['budget_utilization']['mean']:.3f}"
            )
        summary_lines.append("")

    with (output_dir / "azure_transfer_results.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "profile": profile,
                "env_config": {
                    "use_azure_cost": env_cfg.use_azure_cost,
                    "service_budget_per_step": env_cfg.service_budget_per_step,
                    "max_expected_service_cost": env_cfg.max_expected_service_cost,
                    "max_expected_queue_cost": env_cfg.max_expected_queue_cost,
                },
                "results": results,
            },
            handle,
            indent=2,
        )
    (output_dir / "SUMMARY.md").write_text("\n".join(summary_lines).strip() + "\n", encoding="utf-8")
    print(output_dir / "azure_transfer_results.json")


if __name__ == "__main__":
    main()
