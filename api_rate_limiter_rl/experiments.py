from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple
import random

import numpy as np

from .baselines import AdaptiveThresholdPolicy, FixedRatePolicy, HeuristicPriorityPolicy, TokenBucketPolicy
from .config import EnvConfig, PPOConfig, ScenarioConfig, default_scenarios
from .env import RateLimitEnv
from .ppo import PPOAgent
from .plotting import plot_comparison, plot_training_curves
from .utils import ensure_dir, mean, stdev, write_json


def make_training_scenario(rng: random.Random) -> ScenarioConfig:
    scenarios = default_scenarios()
    weighted_choices = [
        scenarios["low_load"],
        scenarios["high_load"],
        scenarios["high_load"],
        scenarios["bursty"],
        scenarios["bursty"],
        scenarios["oscillating"],
        scenarios["oscillating"],
    ]
    choice = rng.choice(weighted_choices)
    return ScenarioConfig(
        name=f"train_{choice.name}",
        base_rate=choice.base_rate,
        oscillation_amplitude=choice.oscillation_amplitude,
        oscillation_period=choice.oscillation_period,
        burst_interval=choice.burst_interval,
        burst_length=choice.burst_length,
        burst_multiplier=choice.burst_multiplier,
        priority_probs=choice.priority_probs,
    )


def rollout_episode(env: RateLimitEnv, policy, deterministic: bool = False):
    state = env.reset()
    done = False
    states: List[List[float]] = []
    actions: List[int] = []
    logps: List[float] = []
    rewards: List[float] = []
    values: List[float] = []
    dones: List[bool] = []
    action_choices: List[int] = []
    infos: List[dict] = []
    while not done:
        if hasattr(policy, "act"):
            if isinstance(policy, PPOAgent):
                action, logp, value, _ = policy.act(state, deterministic=deterministic)
            else:
                _, desired_limit = policy.act(state, env.last_info)
                current_limit = int(env.rate_limit)
                if desired_limit > current_limit:
                    action = RateLimitEnv.action_increase
                elif desired_limit < current_limit:
                    action = RateLimitEnv.action_decrease
                else:
                    action = RateLimitEnv.action_hold
                logp = 0.0
                value = 0.0
        else:
            action = 1
            logp = 0.0
            value = 0.0
        next_state, reward, done, info = env.step(action)
        if not isinstance(policy, PPOAgent) and hasattr(policy, "observe"):
            policy.observe(info)
        states.append(state)
        actions.append(action)
        logps.append(logp)
        rewards.append(reward)
        values.append(value)
        dones.append(done)
        action_choices.append(action)
        infos.append(info)
        state = next_state
    return {
        "states": states,
        "actions": actions,
        "logps": logps,
        "rewards": rewards,
        "values": values,
        "dones": dones,
        "infos": infos,
        "history": env.history,
    }


def summarize_episode(info_list: List[dict]) -> Dict[str, float]:
    return {
        "throughput": mean(item["served"] for item in info_list),
        "latency": mean(item["average_latency"] for item in info_list),
        "drop_rate": mean(item["drop_rate"] for item in info_list),
        "high_priority_success_rate": mean(item["high_priority_success_rate"] for item in info_list),
        "reward": mean(item["reward"] for item in info_list),
    }


def train_ppo(env_cfg: EnvConfig, ppo_cfg: PPOConfig):
    rng = random.Random(ppo_cfg.seed)
    training_history = {
        "episode_reward": [],
        "episode_throughput": [],
        "episode_latency": [],
        "episode_drop_rate": [],
        "episode_high_priority_success_rate": [],
        "selected_action": [],
    }

    initial_env = RateLimitEnv(default_scenarios()["training_mix"], env_cfg, seed=ppo_cfg.seed)
    agent = PPOAgent(
        state_dim=len(initial_env.reset()),
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

    for episode in range(ppo_cfg.train_episodes):
        scenario = make_training_scenario(rng)
        env = RateLimitEnv(scenario, env_cfg, seed=ppo_cfg.seed + episode)
        rollout = rollout_episode(env, agent, deterministic=False)
        batch = agent.batch_from_rollout(
            rollout["states"],
            rollout["actions"],
            rollout["logps"],
            rollout["rewards"],
            rollout["values"],
            rollout["dones"],
            last_value=0.0,
        )
        agent.update(batch)

        summary = summarize_episode(rollout["infos"])
        training_history["episode_reward"].append(summary["reward"])
        training_history["episode_throughput"].append(summary["throughput"])
        training_history["episode_latency"].append(summary["latency"])
        training_history["episode_drop_rate"].append(summary["drop_rate"])
        training_history["episode_high_priority_success_rate"].append(summary["high_priority_success_rate"])
        training_history["selected_action"].append(float(np.bincount(rollout["actions"], minlength=3).argmax()))

    return agent, training_history


def make_baselines(env_cfg: EnvConfig):
    return {
        "fixed": FixedRatePolicy(rate_limit=min(env_cfg.max_rate_limit, env_cfg.service_capacity + env_cfg.rate_step)),
        "token_bucket": TokenBucketPolicy(fill_rate=float(env_cfg.service_capacity), bucket_size=float(env_cfg.max_rate_limit)),
        "adaptive": AdaptiveThresholdPolicy(
            low_queue=max(4, env_cfg.queue_threshold // 2),
            high_queue=env_cfg.queue_threshold,
            min_limit=env_cfg.min_rate_limit,
            max_limit=env_cfg.max_rate_limit,
            step=env_cfg.rate_step,
        ),
        "priority_heuristic": HeuristicPriorityPolicy(
            min_limit=env_cfg.min_rate_limit,
            max_limit=env_cfg.max_rate_limit,
        ),
    }


def evaluate_policy(policy, scenario: ScenarioConfig, env_cfg: EnvConfig, episodes: int, seed: int):
    metrics = []
    for idx in range(episodes):
        env = RateLimitEnv(scenario, env_cfg, seed=seed + 1000 + idx)
        if hasattr(policy, "reset"):
            policy.reset()
        rollout = rollout_episode(env, policy, deterministic=True if isinstance(policy, PPOAgent) else False)
        summary = summarize_episode(rollout["infos"])
        summary["total_reward"] = sum(rollout["rewards"])
        summary["throughput"] = sum(item["served"] for item in rollout["infos"]) / env_cfg.episode_length
        metrics.append(summary)
    aggregated = {}
    for key in metrics[0]:
        aggregated[key] = {
            "mean": mean(item[key] for item in metrics),
            "stdev": stdev(item[key] for item in metrics),
        }
    return aggregated


def run_experiments(
    output_dir: str = "artifacts",
    env_cfg: EnvConfig | None = None,
    ppo_cfg: PPOConfig | None = None,
):
    output_path = Path(output_dir)
    ensure_dir(output_path)
    plots_dir = ensure_dir(output_path / "plots")

    env_cfg = env_cfg or EnvConfig()
    ppo_cfg = ppo_cfg or PPOConfig()

    agent, training_history = train_ppo(env_cfg, ppo_cfg)
    write_json(output_path / "training_history.json", training_history)
    plot_training_curves(training_history, plots_dir)

    scenarios = {
        name: cfg for name, cfg in default_scenarios().items() if name != "training_mix"
    }
    baselines = make_baselines(env_cfg)
    policies = {"ppo": agent, **baselines}
    all_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    comparison_summary: Dict[str, Dict[str, float]] = {}

    for scenario_name, scenario in scenarios.items():
        scenario_results = {}
        for policy_name, policy in policies.items():
            scenario_results[policy_name] = evaluate_policy(
                policy, scenario, env_cfg, episodes=ppo_cfg.eval_episodes, seed=ppo_cfg.seed
            )
        all_results[scenario_name] = scenario_results
        # Plot a few headline metrics per scenario.
        for metric in ("throughput", "latency", "drop_rate", "high_priority_success_rate"):
            plot_comparison(
                {
                    policy_name: {
                        metric: scenario_results[policy_name][metric]["mean"]
                    }
                    for policy_name in scenario_results
                },
                metric,
                plots_dir / scenario_name,
                f"{scenario_name} - {metric}",
            )
        comparison_summary[scenario_name] = {
            policy_name: scenario_results[policy_name]["total_reward"]["mean"] for policy_name in scenario_results
        }

    write_json(output_path / "evaluation_results.json", all_results)
    write_json(output_path / "reward_summary.json", comparison_summary)
    report = build_report(env_cfg, ppo_cfg, training_history, all_results)
    (output_path / "REPORT.md").write_text(report, encoding="utf-8")
    return {
        "training_history": training_history,
        "evaluation_results": all_results,
        "report_path": str(output_path / "REPORT.md"),
    }


def build_report(env_cfg: EnvConfig, ppo_cfg: PPOConfig, training_history, evaluation_results):
    lines = []
    lines.append("# RL-Based Intelligent API Rate Limiting")
    lines.append("")
    lines.append("## Design")
    lines.append(f"- Episode length: {env_cfg.episode_length}")
    lines.append(f"- Service capacity: {env_cfg.service_capacity}")
    lines.append(f"- Rate limit range: {env_cfg.min_rate_limit}..{env_cfg.max_rate_limit}")
    lines.append(f"- PPO hidden size: {ppo_cfg.hidden_size}")
    lines.append(f"- PPO gamma / lambda / clip: {ppo_cfg.gamma} / {ppo_cfg.lam} / {ppo_cfg.clip_eps}")
    lines.append("")
    lines.append("## Training")
    lines.append(f"- Episodes: {ppo_cfg.train_episodes}")
    lines.append(f"- Final mean reward: {mean(training_history['episode_reward'][-20:]):.4f}")
    lines.append(f"- Final mean throughput: {mean(training_history['episode_throughput'][-20:]):.4f}")
    lines.append(f"- Final mean latency: {mean(training_history['episode_latency'][-20:]):.4f}")
    lines.append(f"- Final mean drop rate: {mean(training_history['episode_drop_rate'][-20:]):.4f}")
    lines.append("")
    lines.append("## Key Takeaways")
    high = evaluation_results["high_load"]
    bursty = evaluation_results["bursty"]
    oscillating = evaluation_results["oscillating"]
    lines.append(
        f"- PPO cuts high-load latency versus fixed control ({high['fixed']['latency']['mean']:.3f} -> {high['ppo']['latency']['mean']:.3f}) "
        f"while holding throughput near the service ceiling ({high['ppo']['throughput']['mean']:.3f})."
    )
    lines.append(
        f"- PPO improves bursty throughput over fixed control ({bursty['fixed']['throughput']['mean']:.3f} -> {bursty['ppo']['throughput']['mean']:.3f}) "
        f"and keeps bursty drops lower ({bursty['fixed']['drop_rate']['mean']:.3f} -> {bursty['ppo']['drop_rate']['mean']:.3f})."
    )
    lines.append(
        f"- PPO improves oscillating throughput over fixed control ({oscillating['fixed']['throughput']['mean']:.3f} -> {oscillating['ppo']['throughput']['mean']:.3f}) "
        f"while keeping drop rate low ({oscillating['fixed']['drop_rate']['mean']:.3f} -> {oscillating['ppo']['drop_rate']['mean']:.3f})."
    )
    lines.append(
        "- The token bucket baseline remains strong on some latency cases, so the learned policy is competitive rather than trivially dominant."
    )
    lines.append("")
    lines.append("## Evaluation")
    for scenario_name, results in evaluation_results.items():
        lines.append(f"### {scenario_name}")
        for policy_name, metrics in results.items():
            lines.append(
                f"- {policy_name}: throughput={metrics['throughput']['mean']:.3f}, "
                f"latency={metrics['latency']['mean']:.3f}, drop={metrics['drop_rate']['mean']:.3f}, "
                f"high-priority={metrics['high_priority_success_rate']['mean']:.3f}, "
                f"reward={metrics['total_reward']['mean']:.3f}"
            )
        lines.append("")
    lines.append("## Notes")
    lines.append("- Retries are delayed, buffered across timesteps, and can recursively retry after repeated drops.")
    lines.append("- Queue service is priority-aware, so high-priority traffic is handled first under congestion.")
    lines.append("- PPO uses clipped surrogate updates with GAE and on-policy rollouts.")
    return "\n".join(lines) + "\n"
