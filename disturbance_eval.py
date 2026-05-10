import argparse
import json
from dataclasses import asdict
from pathlib import Path

from api_rate_limiter_rl.config import DisturbanceConfig, default_scenarios
from api_rate_limiter_rl.experiments import build_env_config, evaluate_policy, make_baselines
from api_rate_limiter_rl.env import RateLimitEnv
from api_rate_limiter_rl.ppo import PPOAgent
from api_rate_limiter_rl.utils import ensure_dir, mean, stdev


PHASES = {
    "pre": (0, 39),
    "shock": (40, 79),
    "post": (80, 119),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="disturbance_eval")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--episode-length", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cost-only-checkpoint", default="cost_aware_finetune_run/base_checkpoint.pt")
    parser.add_argument("--warm-start-checkpoint", default="warm_start_reg_final/warm_started_ppo.pt")
    return parser.parse_args()


def cost_only_env_config(args, disturbance=None):
    return build_env_config(
        episode_length=args.episode_length,
        reward_high_drop_weight=1.5,
        use_request_costs=True,
        service_budget_per_step=24.0,
        max_expected_queue_cost=360.0,
        max_expected_service_cost=5.0,
        reward_throughput_weight=0.0,
        reward_cost_processed_weight=1.5,
        reward_budget_utilization_weight=0.0,
        include_rejected_cost=True,
        disturbance=disturbance,
        seed=args.seed,
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


def phase_name_for_step(step: int):
    for name, (start, end) in PHASES.items():
        if start <= step <= end:
            return name
    return "post"


def summarize_info_list(info_list):
    if not info_list:
        return {
            "reward": 0.0,
            "throughput": 0.0,
            "latency": 0.0,
            "drop_rate": 0.0,
            "high_priority_success_rate": 0.0,
            "served_cost": 0.0,
            "budget_utilization": 0.0,
        }
    return {
        "reward": mean(item["reward"] for item in info_list),
        "throughput": mean(item["served"] for item in info_list),
        "latency": mean(item["average_latency"] for item in info_list),
        "drop_rate": mean(item["drop_rate"] for item in info_list),
        "high_priority_success_rate": mean(item["high_priority_success_rate"] for item in info_list),
        "served_cost": mean(item.get("served_cost", 0.0) for item in info_list),
        "budget_utilization": mean(item.get("budget_utilization", 0.0) for item in info_list),
    }


def evaluate_episode(policy, scenario, env_cfg, seed: int):
    env = RateLimitEnv(scenario, env_cfg, seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()
    state = env.reset(seed)
    done = False
    phase_infos = {name: [] for name in PHASES}

    while not done:
        if isinstance(policy, PPOAgent):
            action, _, _, _ = policy.act(state, deterministic=True)
        else:
            _, desired_limit = policy.act(state, env.last_info)
            current_limit = int(env.rate_limit)
            if desired_limit > current_limit:
                action = RateLimitEnv.action_increase
            elif desired_limit < current_limit:
                action = RateLimitEnv.action_decrease
            else:
                action = RateLimitEnv.action_hold
        action = env.maybe_corrupt_action(action)
        next_state, _, done, info = env.step(action)
        if not isinstance(policy, PPOAgent) and hasattr(policy, "observe"):
            policy.observe(info)
        phase_infos[phase_name_for_step(env.time_step - 1)].append(info)
        state = next_state

    phase_metrics = {phase: summarize_info_list(items) for phase, items in phase_infos.items()}
    shock = phase_metrics["shock"]
    pre = phase_metrics["pre"]
    post = phase_metrics["post"]
    collapse = 1 if (shock["latency"] > 5.0 or shock["drop_rate"] > 0.5) else 0
    recovery_checks = []
    for key in ("reward", "throughput", "latency", "drop_rate", "high_priority_success_rate", "budget_utilization"):
        pre_value = pre[key]
        post_value = post[key]
        if key in ("latency", "drop_rate"):
            threshold = abs(pre_value) * 1.10
            recovery_checks.append(post_value <= threshold if pre_value > 0 else post_value <= 0.1)
        else:
            tolerance = max(0.1, abs(pre_value) * 0.10)
            recovery_checks.append(abs(post_value - pre_value) <= tolerance)
    recovered = 1 if all(recovery_checks) else 0
    return {
        "phases": phase_metrics,
        "collapse": collapse,
        "recovered": recovered,
    }


def aggregate_episodes(episodes):
    result = {"phases": {}}
    for phase in PHASES:
        keys = episodes[0]["phases"][phase].keys()
        result["phases"][phase] = {
            key: {
                "mean": mean(ep["phases"][phase][key] for ep in episodes),
                "stdev": stdev(ep["phases"][phase][key] for ep in episodes),
            }
            for key in keys
        }
    collapse_rate = mean(ep["collapse"] for ep in episodes)
    recovery_rate = mean(ep["recovered"] for ep in episodes)
    normalized_post_reward = max(0.0, result["phases"]["post"]["reward"]["mean"] / max(1.0, abs(result["phases"]["pre"]["reward"]["mean"])))
    robustness_score = 0.5 * (1.0 - collapse_rate) + 0.3 * recovery_rate + 0.2 * normalized_post_reward
    result["collapse_rate"] = collapse_rate
    result["recovery_rate"] = recovery_rate
    result["robustness_score"] = robustness_score
    return result


def disturbance_suite():
    return {
        "demand_surge": DisturbanceConfig(type="demand_surge", start_step=40, end_step=79, intensity=3.0),
        "budget_cut": DisturbanceConfig(type="budget_cut", start_step=40, end_step=79, intensity=8.0),
        "cost_shock": DisturbanceConfig(type="cost_shock", start_step=40, end_step=79, intensity=1.0, cost_range=(8.0, 12.0)),
        "action_corruption": DisturbanceConfig(type="action_corruption", start_step=40, end_step=79, intensity=0.30),
    }


def main():
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_dir))
    scenarios = {name: cfg for name, cfg in default_scenarios().items() if name != "training_mix"}
    disturbances = disturbance_suite()
    base_env_cfg = cost_only_env_config(args)
    cost_only_agent = load_agent(Path(args.cost_only_checkpoint), base_env_cfg, args.seed)
    warm_start_agent = load_agent(Path(args.warm_start_checkpoint), base_env_cfg, args.seed)
    baselines = make_baselines(base_env_cfg)
    policies = {
        "ppo_cost_only": cost_only_agent,
        "ppo_warm_start_reg_l010": warm_start_agent,
        "cost_aware_priority": baselines["cost_aware_priority"],
    }

    all_results = {}
    summary_lines = ["# Disturbance Evaluation", ""]

    for disturbance_name, disturbance in disturbances.items():
        summary_lines.append(f"## {disturbance_name}")
        all_results[disturbance_name] = {
            "disturbance": asdict(disturbance),
            "scenarios": {},
        }
        for scenario_name, scenario in scenarios.items():
            scenario_results = {}
            disturbed_env_cfg = cost_only_env_config(args, disturbance=disturbance)
            for policy_name, policy in policies.items():
                episodes = [
                    evaluate_episode(policy, scenario, disturbed_env_cfg, seed=args.seed + idx)
                    for idx in range(args.episodes)
                ]
                scenario_results[policy_name] = aggregate_episodes(episodes)
            all_results[disturbance_name]["scenarios"][scenario_name] = scenario_results

        for policy_name in policies:
            rewards = []
            collapses = []
            recoveries = []
            for scenario_name in scenarios:
                metrics = all_results[disturbance_name]["scenarios"][scenario_name][policy_name]
                rewards.append(metrics["phases"]["shock"]["reward"]["mean"])
                collapses.append(metrics["collapse_rate"])
                recoveries.append(metrics["recovery_rate"])
            summary_lines.append(
                f"- {policy_name}: shock_reward={mean(rewards):.3f}, collapse_rate={mean(collapses):.3f}, recovery_rate={mean(recoveries):.3f}"
            )
        summary_lines.append("")

    with (output_dir / "disturbance_eval_results.json").open("w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2)
    (output_dir / "SUMMARY.md").write_text("\n".join(summary_lines).strip() + "\n", encoding="utf-8")
    print(output_dir / "disturbance_eval_results.json")


if __name__ == "__main__":
    main()
