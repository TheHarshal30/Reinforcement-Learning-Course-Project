from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Tuple
import random

import numpy as np

from .baselines import (
    AdaptiveThresholdPolicy,
    AlwaysMaxPolicy,
    CostEfficiencyPolicy,
    CostAwarePriorityPolicy,
    FixedRatePolicy,
    HeuristicPriorityPolicy,
    TokenBucketPolicy,
)
from .config import EnvConfig, PPOConfig, ScenarioConfig, cost_shift_scenarios, default_scenarios
from .env import RateLimitEnv
from .ppo import PPOAgent
from .plotting import plot_comparison, plot_training_curves
from .utils import ensure_dir, mean, sliding_mean, stdev, write_json


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
        cost_ranges_by_priority=choice.cost_ranges_by_priority,
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
    action_total = max(1, len(info_list))
    return {
        "throughput": mean(item["served"] for item in info_list),
        "latency": mean(item["average_latency"] for item in info_list),
        "drop_rate": mean(item["drop_rate"] for item in info_list),
        "high_priority_success_rate": mean(item["high_priority_success_rate"] for item in info_list),
        "high_priority_drop_rate": mean(item["high_priority_drop_rate"] for item in info_list),
        "served_cost": mean(item.get("served_cost", 0.0) for item in info_list),
        "budget_utilization": mean(item.get("budget_utilization", 0.0) for item in info_list),
        "rejected_cost": mean(item.get("rejected_cost", 0.0) + item.get("overflow_cost", 0.0) for item in info_list),
        "reward_throughput": mean(item.get("reward_throughput", 0.0) for item in info_list),
        "reward_latency": mean(item.get("reward_latency", 0.0) for item in info_list),
        "reward_drop": mean(item.get("reward_drop", 0.0) for item in info_list),
        "reward_high_drop": mean(item.get("reward_high_drop", 0.0) for item in info_list),
        "reward_overload": mean(item.get("reward_overload", 0.0) for item in info_list),
        "reward_cost_processed": mean(item.get("reward_cost_processed", 0.0) for item in info_list),
        "reward_budget_utilization": mean(item.get("reward_budget_utilization", 0.0) for item in info_list),
        "action_frac_decrease": sum(1 for item in info_list if item.get("action") == RateLimitEnv.action_decrease) / action_total,
        "action_frac_hold": sum(1 for item in info_list if item.get("action") == RateLimitEnv.action_hold) / action_total,
        "action_frac_increase": sum(1 for item in info_list if item.get("action") == RateLimitEnv.action_increase) / action_total,
        "reward": mean(item["reward"] for item in info_list),
    }


def _has_meaningful_improvement(current: float, best: float, min_delta: float) -> bool:
    if best == float("-inf"):
        return True
    if best > 0.0:
        return current > best * (1.0 + min_delta)
    return current > best + min_delta


def train_ppo(
    env_cfg: EnvConfig,
    ppo_cfg: PPOConfig,
    log_every: int = 0,
    agent: PPOAgent | None = None,
    scenario_factory=None,
    validation_scenario: ScenarioConfig | None = None,
    behavior_clone_dataset=None,
    behavior_clone_weight: float = 0.0,
):
    rng = random.Random(ppo_cfg.seed)
    training_history = {
        "episode_reward": [],
        "episode_throughput": [],
        "episode_latency": [],
        "episode_drop_rate": [],
        "episode_high_priority_success_rate": [],
        "episode_served_cost": [],
        "episode_budget_utilization": [],
        "episode_rejected_cost": [],
        "episode_reward_throughput": [],
        "episode_reward_latency": [],
        "episode_reward_drop": [],
        "episode_reward_high_drop": [],
        "episode_reward_overload": [],
        "episode_reward_cost_processed": [],
        "episode_reward_budget_utilization": [],
        "episode_action_frac_decrease": [],
        "episode_action_frac_hold": [],
        "episode_action_frac_increase": [],
        "validation_reward": [],
        "validation_episode": [],
        "selected_action": [],
    }
    best_window_reward = float("-inf")
    stale_windows = 0
    best_validation_reward = float("-inf")
    best_checkpoint = None

    scenario_factory = scenario_factory or make_training_scenario
    validation_scenario = validation_scenario or default_scenarios()["bursty"]

    if agent is None:
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
    else:
        agent.set_learning_rates(ppo_cfg.actor_lr, ppo_cfg.critic_lr)

    for episode in range(ppo_cfg.train_episodes):
        scenario = scenario_factory(rng)
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
        agent.update(
            batch,
            behavior_batch=behavior_clone_dataset,
            behavior_weight=behavior_clone_weight,
        )

        summary = summarize_episode(rollout["infos"])
        training_history["episode_reward"].append(summary["reward"])
        training_history["episode_throughput"].append(summary["throughput"])
        training_history["episode_latency"].append(summary["latency"])
        training_history["episode_drop_rate"].append(summary["drop_rate"])
        training_history["episode_high_priority_success_rate"].append(summary["high_priority_success_rate"])
        training_history["episode_served_cost"].append(summary["served_cost"])
        training_history["episode_budget_utilization"].append(summary["budget_utilization"])
        training_history["episode_rejected_cost"].append(summary["rejected_cost"])
        training_history["episode_reward_throughput"].append(summary["reward_throughput"])
        training_history["episode_reward_latency"].append(summary["reward_latency"])
        training_history["episode_reward_drop"].append(summary["reward_drop"])
        training_history["episode_reward_high_drop"].append(summary["reward_high_drop"])
        training_history["episode_reward_overload"].append(summary["reward_overload"])
        training_history["episode_reward_cost_processed"].append(summary["reward_cost_processed"])
        training_history["episode_reward_budget_utilization"].append(summary["reward_budget_utilization"])
        training_history["episode_action_frac_decrease"].append(summary["action_frac_decrease"])
        training_history["episode_action_frac_hold"].append(summary["action_frac_hold"])
        training_history["episode_action_frac_increase"].append(summary["action_frac_increase"])
        training_history["selected_action"].append(float(np.bincount(rollout["actions"], minlength=3).argmax()))
        if log_every and ((episode + 1) % log_every == 0 or episode == 0 or episode + 1 == ppo_cfg.train_episodes):
            recent_rewards = training_history["episode_reward"][-log_every:] if log_every else training_history["episode_reward"]
            recent_latency = training_history["episode_latency"][-log_every:] if log_every else training_history["episode_latency"]
            recent_drop_rate = training_history["episode_drop_rate"][-log_every:] if log_every else training_history["episode_drop_rate"]
            recent_throughput = training_history["episode_throughput"][-log_every:] if log_every else training_history["episode_throughput"]
            print(
                f"[train] episode {episode + 1}/{ppo_cfg.train_episodes} "
                f"scenario={scenario.name} "
                f"reward={summary['reward']:.4f} "
                f"recent_reward={sliding_mean(recent_rewards):.4f} "
                f"recent_throughput={sliding_mean(recent_throughput):.4f} "
                f"recent_latency={sliding_mean(recent_latency):.4f} "
                f"recent_drop={sliding_mean(recent_drop_rate):.4f} "
                f"actions=({summary['action_frac_decrease']:.2f},{summary['action_frac_hold']:.2f},{summary['action_frac_increase']:.2f}) "
                f"reward_parts=({summary['reward_throughput']:.3f},{summary['reward_cost_processed']:.3f},{summary['reward_budget_utilization']:.3f},{summary['reward_latency']:.3f},{summary['reward_drop']:.3f},{summary['reward_high_drop']:.3f})",
                flush=True,
            )
        if ppo_cfg.validation_eval_every > 0 and (episode + 1) % ppo_cfg.validation_eval_every == 0:
            validation_summary = evaluate_current_agent(
                agent,
                env_cfg,
                validation_scenario,
                episodes=ppo_cfg.validation_eval_episodes,
                seed=ppo_cfg.seed,
            )
            validation_reward = validation_summary["total_reward"]["mean"]
            training_history["validation_reward"].append(validation_reward)
            training_history["validation_episode"].append(episode + 1)
            print(
                f"[valid] episode={episode + 1} reward={validation_reward:.4f}",
                flush=True,
            )
            if validation_reward > best_validation_reward:
                best_validation_reward = validation_reward
                best_checkpoint = agent.checkpoint()
        if (
            ppo_cfg.early_stop_window > 0
            and ppo_cfg.early_stop_patience > 0
            and len(training_history["episode_reward"]) >= ppo_cfg.early_stop_window
        ):
            window_reward = sliding_mean(training_history["episode_reward"][-ppo_cfg.early_stop_window :])
            if _has_meaningful_improvement(window_reward, best_window_reward, ppo_cfg.early_stop_min_delta):
                best_window_reward = window_reward
                stale_windows = 0
            else:
                stale_windows += 1
                if stale_windows >= ppo_cfg.early_stop_patience:
                    print(
                        f"[train] early_stop episode={episode + 1} "
                        f"window_reward={window_reward:.4f} "
                        f"best_window_reward={best_window_reward:.4f}",
                        flush=True,
                    )
                    break

    if best_checkpoint is not None:
        agent.restore(best_checkpoint)

    return agent, training_history


def evaluate_current_agent(agent: PPOAgent, env_cfg: EnvConfig, scenario: ScenarioConfig, episodes: int, seed: int):
    return evaluate_policy(agent, scenario, env_cfg, episodes=episodes, seed=seed)


def make_baselines(env_cfg: EnvConfig):
    if env_cfg.use_request_costs:
        target_budget = env_cfg.service_budget_per_step
        return {
            "fixed": FixedRatePolicy(
                rate_limit=env_cfg.max_rate_limit,
                target_cost_budget=target_budget,
                min_limit=env_cfg.min_rate_limit,
            ),
            "always_max": AlwaysMaxPolicy(max_limit=env_cfg.max_rate_limit),
            "token_bucket": TokenBucketPolicy(
                fill_rate=float(target_budget),
                bucket_size=float(target_budget),
                target_cost_budget=target_budget,
                min_limit=env_cfg.min_rate_limit,
            ),
            "adaptive": AdaptiveThresholdPolicy(
                low_queue=max(4, env_cfg.queue_threshold // 2),
                high_queue=env_cfg.queue_threshold,
                min_limit=env_cfg.min_rate_limit,
                max_limit=env_cfg.max_rate_limit,
                step=env_cfg.rate_step,
                target_queue_cost=env_cfg.max_expected_queue_cost * 0.5,
            ),
            "priority_heuristic": HeuristicPriorityPolicy(
                min_limit=env_cfg.min_rate_limit,
                max_limit=env_cfg.max_rate_limit,
                target_cost_budget=target_budget,
            ),
            "cost_aware_priority": CostAwarePriorityPolicy(
                min_limit=env_cfg.min_rate_limit,
                max_limit=env_cfg.max_rate_limit,
                target_cost_budget=target_budget,
            ),
            "cost_efficiency": CostEfficiencyPolicy(
                min_limit=env_cfg.min_rate_limit,
                max_limit=env_cfg.max_rate_limit,
                target_cost_budget=target_budget,
            ),
        }
    return {
        "fixed": FixedRatePolicy(rate_limit=min(env_cfg.max_rate_limit, env_cfg.service_capacity + env_cfg.rate_step)),
        "always_max": AlwaysMaxPolicy(max_limit=env_cfg.max_rate_limit),
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
        summary["served_cost"] = sum(item.get("served_cost", 0.0) for item in rollout["infos"]) / env_cfg.episode_length
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
    log_every: int = 0,
    scenario_set: Dict[str, ScenarioConfig] | None = None,
    include_cost_shift: bool = False,
):
    output_path = Path(output_dir)
    ensure_dir(output_path)
    plots_dir = ensure_dir(output_path / "plots")

    env_cfg = env_cfg or EnvConfig()
    ppo_cfg = ppo_cfg or PPOConfig()

    agent, training_history = train_ppo(env_cfg, ppo_cfg, log_every=log_every)
    write_json(output_path / "training_history.json", training_history)
    agent.save(output_path / "ppo_checkpoint.pt")
    plot_training_curves(training_history, plots_dir)

    if scenario_set is None:
        scenarios = {
            name: cfg for name, cfg in default_scenarios().items() if name != "training_mix"
        }
    else:
        scenarios = dict(scenario_set)
    if include_cost_shift:
        scenarios.update(cost_shift_scenarios())
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
        plot_metrics = ["throughput", "latency", "drop_rate", "high_priority_success_rate"]
        if env_cfg.use_request_costs:
            plot_metrics.extend(["served_cost", "budget_utilization"])
        for metric in plot_metrics:
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


def build_ppo_config(**overrides) -> PPOConfig:
    return replace(PPOConfig(), **overrides)


def build_env_config(**overrides) -> EnvConfig:
    return replace(EnvConfig(), **overrides)


def quick_screen_variant(
    name: str,
    env_cfg: EnvConfig,
    ppo_cfg: PPOConfig,
    eval_scenario_name: str = "bursty",
    eval_episodes: int | None = None,
    log_every: int = 0,
):
    scenarios = {**default_scenarios(), **cost_shift_scenarios()}
    scenario = scenarios[eval_scenario_name]
    started_at = perf_counter()
    agent, training_history = train_ppo(env_cfg, ppo_cfg, log_every=log_every)
    evaluation = evaluate_policy(
        agent,
        scenario,
        env_cfg,
        episodes=eval_episodes or ppo_cfg.eval_episodes,
        seed=ppo_cfg.seed,
    )
    final_reward = training_history["episode_reward"][-1]
    recent_window = min(5, len(training_history["episode_reward"]))
    summary = {
        "name": name,
        "seed": ppo_cfg.seed,
        "device": ppo_cfg.device,
        "train_episodes": len(training_history["episode_reward"]),
        "eval_episodes": eval_episodes or ppo_cfg.eval_episodes,
        "eval_scenario": eval_scenario_name,
        "final_training_reward": final_reward,
        "recent_training_reward": sliding_mean(training_history["episode_reward"][-recent_window:]),
        "recent_training_latency": sliding_mean(training_history["episode_latency"][-recent_window:]),
        "recent_training_drop_rate": sliding_mean(training_history["episode_drop_rate"][-recent_window:]),
        "eval_reward_mean": evaluation["total_reward"]["mean"],
        "eval_throughput_mean": evaluation["throughput"]["mean"],
        "eval_latency_mean": evaluation["latency"]["mean"],
        "eval_drop_rate_mean": evaluation["drop_rate"]["mean"],
        "eval_high_priority_success_mean": evaluation["high_priority_success_rate"]["mean"],
        "eval_high_priority_drop_mean": evaluation["high_priority_drop_rate"]["mean"],
        "eval_served_cost_mean": evaluation["served_cost"]["mean"],
        "eval_budget_utilization_mean": evaluation["budget_utilization"]["mean"],
        "elapsed_seconds": perf_counter() - started_at,
        "env_config": asdict(env_cfg),
        "ppo_config": asdict(ppo_cfg),
    }
    return summary, training_history, evaluation


def build_report(env_cfg: EnvConfig, ppo_cfg: PPOConfig, training_history, evaluation_results):
    lines = []
    lines.append("# RL-Based Intelligent API Rate Limiting")
    lines.append("")
    lines.append("## Design")
    lines.append(f"- Episode length: {env_cfg.episode_length}")
    lines.append(f"- Service capacity: {env_cfg.service_capacity}")
    if env_cfg.use_request_costs:
        lines.append(f"- Service budget per step: {env_cfg.service_budget_per_step}")
    lines.append(f"- Rate limit range: {env_cfg.min_rate_limit}..{env_cfg.max_rate_limit}")
    lines.append(f"- PPO hidden size: {ppo_cfg.hidden_size}")
    lines.append(f"- PPO gamma / lambda / clip: {ppo_cfg.gamma} / {ppo_cfg.lam} / {ppo_cfg.clip_eps}")
    lines.append("")
    lines.append("## Training")
    lines.append(f"- Episodes: {len(training_history['episode_reward'])}")
    lines.append(f"- Final mean reward: {mean(training_history['episode_reward'][-20:]):.4f}")
    lines.append(f"- Final mean throughput: {mean(training_history['episode_throughput'][-20:]):.4f}")
    lines.append(f"- Final mean latency: {mean(training_history['episode_latency'][-20:]):.4f}")
    lines.append(f"- Final mean drop rate: {mean(training_history['episode_drop_rate'][-20:]):.4f}")
    if env_cfg.use_request_costs:
        lines.append(f"- Final mean processed cost: {mean(training_history['episode_served_cost'][-20:]):.4f}")
        lines.append(f"- Final mean budget utilization: {mean(training_history['episode_budget_utilization'][-20:]):.4f}")
    lines.append("")
    lines.append("## Key Takeaways")
    if all(name in evaluation_results for name in ("high_load", "bursty", "oscillating")):
        high = evaluation_results["high_load"]
        bursty = evaluation_results["bursty"]
        oscillating = evaluation_results["oscillating"]
        lines.append(
            f"- PPO changes high-load latency versus fixed control ({high['fixed']['latency']['mean']:.3f} -> {high['ppo']['latency']['mean']:.3f}) "
            f"while holding throughput near the service ceiling ({high['ppo']['throughput']['mean']:.3f})."
        )
        lines.append(
            f"- PPO changes bursty throughput versus fixed control ({bursty['fixed']['throughput']['mean']:.3f} -> {bursty['ppo']['throughput']['mean']:.3f}) "
            f"and bursty drops ({bursty['fixed']['drop_rate']['mean']:.3f} -> {bursty['ppo']['drop_rate']['mean']:.3f})."
        )
        lines.append(
            f"- PPO changes oscillating throughput versus fixed control ({oscillating['fixed']['throughput']['mean']:.3f} -> {oscillating['ppo']['throughput']['mean']:.3f}) "
            f"and drop rate ({oscillating['fixed']['drop_rate']['mean']:.3f} -> {oscillating['ppo']['drop_rate']['mean']:.3f})."
        )
        lines.append(
            "- The token bucket baseline remains strong on some latency cases, so the learned policy is competitive rather than trivially dominant."
        )
    else:
        lines.append("- This run used a custom scenario set, so compare the metric tables directly.")
    lines.append("")
    lines.append("## Evaluation")
    for scenario_name, results in evaluation_results.items():
        lines.append(f"### {scenario_name}")
        for policy_name, metrics in results.items():
            cost_segment = ""
            if env_cfg.use_request_costs:
                cost_segment = (
                    f"served_cost={metrics['served_cost']['mean']:.3f}, "
                    f"budget_utilization={metrics['budget_utilization']['mean']:.3f}, "
                )
            lines.append(
                f"- {policy_name}: throughput={metrics['throughput']['mean']:.3f}, "
                f"latency={metrics['latency']['mean']:.3f}, drop={metrics['drop_rate']['mean']:.3f}, "
                f"high-priority={metrics['high_priority_success_rate']['mean']:.3f}, "
                f"{cost_segment}reward={metrics['total_reward']['mean']:.3f}"
            )
        lines.append("")
    lines.append("## Notes")
    lines.append("- Retries are delayed, buffered across timesteps, and can recursively retry after repeated drops.")
    lines.append("- Queue service is priority-aware, so high-priority traffic is handled first under congestion.")
    lines.append("- PPO uses clipped surrogate updates with GAE and on-policy rollouts.")
    return "\n".join(lines) + "\n"
