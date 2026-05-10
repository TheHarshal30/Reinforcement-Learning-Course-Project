import argparse
from pathlib import Path

from api_rate_limiter_rl.config import cost_shift_scenarios, default_scenarios
from api_rate_limiter_rl.experiments import (
    build_env_config,
    build_ppo_config,
    evaluate_policy,
    make_baselines,
    train_ppo,
)
from api_rate_limiter_rl.utils import ensure_dir, write_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="cost_aware_finetune")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--base-episodes", type=int, default=260)
    parser.add_argument("--finetune-episodes", type=int, default=30)
    parser.add_argument("--eval-episodes", type=int, default=24)
    parser.add_argument("--episode-length", type=int, default=180)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--finetune-actor-lr", type=float, default=0.0075)
    parser.add_argument("--finetune-critic-lr", type=float, default=0.0100)
    return parser.parse_args()


def cost_only_env_config(args):
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
    )


def fixed_high_load_training_scenario(_rng):
    scenario = default_scenarios()["high_load"]
    return type(scenario)(
        name="train_high_load_specialized",
        base_rate=scenario.base_rate,
        oscillation_amplitude=scenario.oscillation_amplitude,
        oscillation_period=scenario.oscillation_period,
        burst_interval=scenario.burst_interval,
        burst_length=scenario.burst_length,
        burst_multiplier=scenario.burst_multiplier,
        priority_probs=scenario.priority_probs,
        cost_ranges_by_priority=scenario.cost_ranges_by_priority,
    )


def evaluate_bundle(agent, env_cfg, eval_episodes):
    scenarios = {
        "high_load": default_scenarios()["high_load"],
        "bursty": default_scenarios()["bursty"],
        "oscillating": default_scenarios()["oscillating"],
        "expensive_attack": cost_shift_scenarios()["expensive_attack"],
    }
    baselines = make_baselines(env_cfg)
    policies = {"ppo": agent, **baselines}
    results = {}
    for scenario_name, scenario in scenarios.items():
        results[scenario_name] = {
            policy_name: evaluate_policy(policy, scenario, env_cfg, episodes=eval_episodes, seed=7)
            for policy_name, policy in policies.items()
        }
    return results


def main():
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_dir))
    env_cfg = cost_only_env_config(args)

    base_cfg = build_ppo_config(
        train_episodes=args.base_episodes,
        eval_episodes=args.eval_episodes,
        device=args.device,
        validation_eval_every=20,
        validation_eval_episodes=3,
    )
    fine_cfg = build_ppo_config(
        train_episodes=args.finetune_episodes,
        eval_episodes=args.eval_episodes,
        device=args.device,
        actor_lr=args.finetune_actor_lr,
        critic_lr=args.finetune_critic_lr,
        validation_eval_every=10,
        validation_eval_episodes=3,
    )

    print("[finetune] base cost-only training", flush=True)
    agent, base_history = train_ppo(env_cfg, base_cfg, log_every=args.log_every)
    agent.save(output_dir / "base_checkpoint.pt")
    write_json(output_dir / "base_training_history.json", base_history)
    base_eval = evaluate_bundle(agent, env_cfg, args.eval_episodes)
    write_json(output_dir / "base_evaluation.json", base_eval)

    print("[finetune] high-load specialization", flush=True)
    agent, finetune_history = train_ppo(
        env_cfg,
        fine_cfg,
        log_every=args.log_every,
        agent=agent,
        scenario_factory=fixed_high_load_training_scenario,
        validation_scenario=default_scenarios()["high_load"],
    )
    agent.save(output_dir / "finetuned_checkpoint.pt")
    write_json(output_dir / "finetune_training_history.json", finetune_history)
    finetuned_eval = evaluate_bundle(agent, env_cfg, args.eval_episodes)
    write_json(output_dir / "finetuned_evaluation.json", finetuned_eval)
    print(output_dir)


if __name__ == "__main__":
    main()
