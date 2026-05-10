import argparse
from pathlib import Path

from api_rate_limiter_rl.config import cost_shift_scenarios, default_scenarios
from api_rate_limiter_rl.experiments import build_env_config, build_ppo_config, quick_screen_variant, run_experiments
from api_rate_limiter_rl.utils import ensure_dir, write_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="cost_aware_runs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=["screen", "full"], default="screen")
    parser.add_argument("--train-episodes", type=int, default=30)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--episode-length", type=int, default=180)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def blind_env_config(args):
    return build_env_config(
        episode_length=args.episode_length,
        reward_high_drop_weight=1.5,
        use_request_costs=False,
    )


def aware_env_config(args):
    return build_env_config(
        episode_length=args.episode_length,
        reward_high_drop_weight=1.5,
        use_request_costs=True,
        service_budget_per_step=24.0,
        max_expected_queue_cost=360.0,
        max_expected_service_cost=5.0,
        reward_cost_processed_weight=0.2,
    )


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


def screen_variants(args):
    return {
        "cost_blind_hp15": blind_env_config(args),
        "current_fix": aware_env_config(args),
        "cost_only": cost_only_env_config(args),
        "cost_only_util": build_env_config(
            episode_length=args.episode_length,
            reward_high_drop_weight=1.5,
            use_request_costs=True,
            service_budget_per_step=24.0,
            max_expected_queue_cost=360.0,
            max_expected_service_cost=5.0,
            reward_throughput_weight=0.0,
            reward_cost_processed_weight=1.0,
            reward_budget_utilization_weight=0.5,
            include_rejected_cost=True,
        ),
        "cost_only_high_util": build_env_config(
            episode_length=args.episode_length,
            reward_high_drop_weight=1.5,
            use_request_costs=True,
            service_budget_per_step=24.0,
            max_expected_queue_cost=360.0,
            max_expected_service_cost=5.0,
            reward_throughput_weight=0.0,
            reward_cost_processed_weight=0.5,
            reward_budget_utilization_weight=1.0,
            include_rejected_cost=True,
        ),
    }


def main():
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_dir))
    ppo_cfg = build_ppo_config(
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        device=args.device,
        validation_eval_every=10 if args.mode == "screen" else 20,
        validation_eval_episodes=2 if args.mode == "screen" else 3,
    )

    if args.mode == "screen":
        results = []
        configs = screen_variants(args)
        for name, env_cfg in configs.items():
            print(f"[compare] start {name}", flush=True)
            summary, training_history, evaluation = quick_screen_variant(
                name,
                env_cfg,
                ppo_cfg,
                eval_scenario_name="high_load",
                eval_episodes=args.eval_episodes,
                log_every=args.log_every,
            )
            run_dir = ensure_dir(output_dir / name)
            write_json(run_dir / "summary.json", summary)
            write_json(run_dir / "training_history.json", training_history)
            write_json(run_dir / "evaluation.json", evaluation)
            results.append(summary)
        write_json(output_dir / "screen_summary.json", results)
        print(output_dir / "screen_summary.json")
        return

    blind_dir = ensure_dir(output_dir / "cost_blind_hp15")
    aware_dir = ensure_dir(output_dir / "cost_aware_cost_only")

    print("[compare] full cost-blind run", flush=True)
    run_experiments(
        output_dir=str(blind_dir),
        env_cfg=blind_env_config(args),
        ppo_cfg=ppo_cfg,
        log_every=args.log_every,
    )
    print("[compare] full cost-aware run", flush=True)
    standard_and_ood = {
        **{name: cfg for name, cfg in default_scenarios().items() if name != "training_mix"},
        **cost_shift_scenarios(),
    }
    run_experiments(
        output_dir=str(aware_dir),
        env_cfg=cost_only_env_config(args),
        ppo_cfg=ppo_cfg,
        log_every=args.log_every,
        scenario_set=standard_and_ood,
    )
    print(output_dir)


if __name__ == "__main__":
    main()
