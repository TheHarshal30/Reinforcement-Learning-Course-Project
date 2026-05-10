import argparse
from dataclasses import asdict
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F

from api_rate_limiter_rl.config import default_scenarios
from api_rate_limiter_rl.env import RateLimitEnv
from api_rate_limiter_rl.experiments import (
    build_env_config,
    build_ppo_config,
    evaluate_policy,
    make_baselines,
    run_experiments,
    summarize_episode,
    train_ppo,
)
from api_rate_limiter_rl.ppo import PPOAgent
from api_rate_limiter_rl.utils import ensure_dir, mean, write_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="warm_start_runs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--teacher", choices=["cost_aware_priority", "priority_heuristic"], default="cost_aware_priority")
    parser.add_argument("--demo-episodes", type=int, default=300)
    parser.add_argument("--bc-epochs", type=int, default=120)
    parser.add_argument("--bc-batch-size", type=int, default=128)
    parser.add_argument("--bc-lr", type=float, default=0.001)
    parser.add_argument("--bc-critic-weight", type=float, default=0.5)
    parser.add_argument("--ppo-train-episodes", type=int, default=260)
    parser.add_argument("--ppo-eval-episodes", type=int, default=24)
    parser.add_argument("--ppo-actor-lr", type=float, default=0.001)
    parser.add_argument("--ppo-critic-lr", type=float, default=0.0015)
    parser.add_argument("--behavior-lambda", type=float, default=0.1)
    parser.add_argument("--episode-length", type=int, default=180)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
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


def sample_training_scenario(rng: random.Random):
    scenarios = default_scenarios()
    choices = [
        scenarios["low_load"],
        scenarios["high_load"],
        scenarios["high_load"],
        scenarios["bursty"],
        scenarios["bursty"],
        scenarios["oscillating"],
        scenarios["oscillating"],
    ]
    return rng.choice(choices)


def monte_carlo_returns(rewards, gamma: float):
    returns = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for idx in reversed(range(len(rewards))):
        running = rewards[idx] + gamma * running
        returns[idx] = running
    return returns


def teacher_action_to_discrete(policy, env, state):
    _, desired_limit = policy.act(state, env.last_info)
    current_limit = int(env.rate_limit)
    if desired_limit > current_limit:
        return RateLimitEnv.action_increase
    if desired_limit < current_limit:
        return RateLimitEnv.action_decrease
    return RateLimitEnv.action_hold


def collect_demonstrations(policy, env_cfg, episodes: int, gamma: float, seed: int):
    rng = random.Random(seed)
    states = []
    actions = []
    returns = []
    episode_summaries = []
    for episode in range(episodes):
        scenario = sample_training_scenario(rng)
        env = RateLimitEnv(scenario, env_cfg, seed=seed + episode)
        if hasattr(policy, "reset"):
            policy.reset()
        state = env.reset(seed + episode)
        done = False
        ep_states = []
        ep_actions = []
        ep_rewards = []
        infos = []
        while not done:
            action = teacher_action_to_discrete(policy, env, state)
            next_state, reward, done, info = env.step(action)
            if hasattr(policy, "observe"):
                policy.observe(info)
            ep_states.append(state)
            ep_actions.append(action)
            ep_rewards.append(reward)
            infos.append(info)
            state = next_state
        ep_returns = monte_carlo_returns(ep_rewards, gamma)
        states.extend(ep_states)
        actions.extend(ep_actions)
        returns.extend(ep_returns.tolist())
        summary = summarize_episode(infos)
        summary["scenario"] = scenario.name
        summary["total_reward"] = float(sum(ep_rewards))
        episode_summaries.append(summary)
    return {
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.int64),
        "returns": np.asarray(returns, dtype=np.float32),
        "episode_summaries": episode_summaries,
    }


def pretrain_bc(agent: PPOAgent, dataset, epochs: int, batch_size: int, lr: float, critic_weight: float, seed: int):
    rng = np.random.default_rng(seed)
    states = torch.tensor(dataset["states"], dtype=torch.float32, device=agent.device)
    actions = torch.tensor(dataset["actions"], dtype=torch.int64, device=agent.device)
    returns = torch.tensor(dataset["returns"], dtype=torch.float32, device=agent.device)

    total = len(states)
    split = int(0.8 * total)
    indices = rng.permutation(total)
    train_idx = torch.tensor(indices[:split], dtype=torch.int64, device=agent.device)
    val_idx = torch.tensor(indices[split:], dtype=torch.int64, device=agent.device)

    optimizer = torch.optim.Adam(agent.model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best = None
    best_val = float("inf")

    for epoch in range(epochs):
        perm = train_idx[torch.randperm(len(train_idx), device=agent.device)]
        agent.model.train()
        batch_losses = []
        for start in range(0, len(perm), batch_size):
            mb = perm[start : start + batch_size]
            logits, values = agent.model(states[mb])
            ce_loss = F.cross_entropy(logits, actions[mb])
            mse_loss = F.mse_loss(values, returns[mb])
            loss = ce_loss + critic_weight * mse_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.model.parameters(), 0.5)
            optimizer.step()
            batch_losses.append(float(loss.item()))

        agent.model.eval()
        with torch.no_grad():
            val_logits, val_values = agent.model(states[val_idx])
            val_ce = F.cross_entropy(val_logits, actions[val_idx])
            val_mse = F.mse_loss(val_values, returns[val_idx])
            val_loss = float((val_ce + critic_weight * val_mse).item())
            val_pred = torch.argmax(val_logits, dim=-1)
            val_acc = float((val_pred == actions[val_idx]).float().mean().item())
        history["train_loss"].append(mean(batch_losses))
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if val_loss < best_val:
            best_val = val_loss
            best = agent.checkpoint()
        if (epoch + 1) % 20 == 0 or epoch == 0 or epoch + 1 == epochs:
            print(
                f"[bc] epoch {epoch + 1}/{epochs} train_loss={history['train_loss'][-1]:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}",
                flush=True,
            )

    if best is not None:
        agent.restore(best)
    return history


def evaluate_bundle(agent, teacher, env_cfg, eval_episodes: int, seed: int):
    scenarios = {
        name: cfg for name, cfg in default_scenarios().items() if name != "training_mix"
    }
    baselines = make_baselines(env_cfg)
    selected = {
        "teacher": teacher,
        "ppo": agent,
        "cost_aware_priority": baselines["cost_aware_priority"],
        "priority_heuristic": baselines["priority_heuristic"],
        "token_bucket": baselines["token_bucket"],
        "fixed": baselines["fixed"],
    }
    results = {}
    for scenario_name, scenario in scenarios.items():
        results[scenario_name] = {
            name: evaluate_policy(policy, scenario, env_cfg, episodes=eval_episodes, seed=seed)
            for name, policy in selected.items()
        }
    return results


def main():
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_dir))
    env_cfg = cost_only_env_config(args)
    ppo_cfg = build_ppo_config(
        train_episodes=args.ppo_train_episodes,
        eval_episodes=args.ppo_eval_episodes,
        actor_lr=args.ppo_actor_lr,
        critic_lr=args.ppo_critic_lr,
        device=args.device,
        validation_eval_every=20,
        validation_eval_episodes=3,
        seed=args.seed,
    )

    warmup_env = RateLimitEnv(default_scenarios()["high_load"], env_cfg, seed=args.seed)
    agent = PPOAgent(
        state_dim=len(warmup_env.reset()),
        action_dim=3,
        hidden_dim=ppo_cfg.hidden_size,
        actor_lr=ppo_cfg.actor_lr,
        critic_lr=ppo_cfg.critic_lr,
        clip_eps=ppo_cfg.clip_eps,
        gamma=ppo_cfg.gamma,
        lam=ppo_cfg.lam,
        update_epochs=ppo_cfg.update_epochs,
        minibatch_size=ppo_cfg.minibatch_size,
        seed=ppo_cfg.seed,
        device=None if ppo_cfg.device == "auto" else ppo_cfg.device,
    )
    teacher = make_baselines(env_cfg)[args.teacher]

    print(f"[warm-start] collect demos teacher={args.teacher}", flush=True)
    dataset = collect_demonstrations(teacher, env_cfg, args.demo_episodes, ppo_cfg.gamma, args.seed)
    write_json(
        output_dir / "demo_summary.json",
        {
            "teacher": args.teacher,
            "num_samples": int(len(dataset["states"])),
            "demo_episodes": args.demo_episodes,
            "env_config": asdict(env_cfg),
            "ppo_config": asdict(ppo_cfg),
            "mean_demo_reward": mean(item["total_reward"] for item in dataset["episode_summaries"]),
            "mean_demo_throughput": mean(item["throughput"] for item in dataset["episode_summaries"]),
        },
    )
    np.save(output_dir / "demo_states.npy", dataset["states"])
    np.save(output_dir / "demo_actions.npy", dataset["actions"])
    np.save(output_dir / "demo_returns.npy", dataset["returns"])

    print("[warm-start] behavior cloning", flush=True)
    bc_history = pretrain_bc(
        agent,
        dataset,
        epochs=args.bc_epochs,
        batch_size=args.bc_batch_size,
        lr=args.bc_lr,
        critic_weight=args.bc_critic_weight,
        seed=args.seed,
    )
    write_json(output_dir / "bc_history.json", bc_history)
    agent.save(output_dir / "bc_pretrained_actor.pt")

    print("[warm-start] evaluate cloned policy", flush=True)
    cloned_eval = evaluate_bundle(agent, teacher, env_cfg, args.ppo_eval_episodes, args.seed)
    write_json(output_dir / "cloned_evaluation.json", cloned_eval)

    print("[warm-start] PPO fine-tune", flush=True)
    agent, ppo_history = train_ppo(
        env_cfg,
        ppo_cfg,
        log_every=args.log_every,
        agent=agent,
        behavior_clone_dataset={
            "states": dataset["states"],
            "actions": dataset["actions"],
        },
        behavior_clone_weight=args.behavior_lambda,
    )
    agent.save(output_dir / "warm_started_ppo.pt")
    write_json(output_dir / "ppo_history.json", ppo_history)

    print("[warm-start] evaluate fine-tuned policy", flush=True)
    finetuned_eval = evaluate_bundle(agent, teacher, env_cfg, args.ppo_eval_episodes, args.seed)
    write_json(output_dir / "finetuned_evaluation.json", finetuned_eval)
    write_json(
        output_dir / "warm_start_config.json",
        {
            "teacher": args.teacher,
            "behavior_lambda": args.behavior_lambda,
            "demo_episodes": args.demo_episodes,
            "bc_epochs": args.bc_epochs,
            "ppo_train_episodes": args.ppo_train_episodes,
        },
    )

    print(output_dir)


if __name__ == "__main__":
    main()
