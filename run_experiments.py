import argparse

from api_rate_limiter_rl.experiments import build_ppo_config, run_experiments


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-episodes", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=0)
    args = parser.parse_args()

    ppo_overrides = {}
    if args.train_episodes is not None:
        ppo_overrides["train_episodes"] = args.train_episodes
    if args.eval_episodes is not None:
        ppo_overrides["eval_episodes"] = args.eval_episodes
    ppo_cfg = build_ppo_config(**ppo_overrides) if ppo_overrides else None

    result = run_experiments(ppo_cfg=ppo_cfg, log_every=args.log_every)
    print(result["report_path"])
