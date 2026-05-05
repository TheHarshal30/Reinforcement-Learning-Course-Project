from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    base_rate: float
    oscillation_amplitude: float = 0.0
    oscillation_period: int = 0
    burst_interval: int = 0
    burst_length: int = 0
    burst_multiplier: float = 1.0
    priority_probs: Tuple[float, float, float] = (0.55, 0.30, 0.15)


@dataclass
class EnvConfig:
    episode_length: int = 180
    service_capacity: int = 12
    max_rate_limit: int = 18
    min_rate_limit: int = 2
    rate_step: int = 2
    queue_threshold: int = 24
    queue_hard_cap: int = 72
    retry_max_delay: int = 5
    max_expected_arrivals: int = 28
    retry_probability_by_priority: Tuple[float, float, float] = (0.12, 0.22, 0.45)
    reward_throughput_weight: float = 1.1
    reward_latency_weight: float = 1.7
    reward_drop_weight: float = 1.2
    reward_high_drop_weight: float = 3.1
    reward_overload_weight: float = 1.0
    state_drop_window: int = 5
    seed: int = 7


@dataclass
class PPOConfig:
    hidden_size: int = 48
    actor_lr: float = 0.005
    critic_lr: float = 0.008
    device: str = "auto"
    gamma: float = 0.97
    lam: float = 0.93
    clip_eps: float = 0.2
    update_epochs: int = 8
    minibatch_size: int = 128
    train_episodes: int = 260
    eval_episodes: int = 24
    seed: int = 7


def default_scenarios() -> Dict[str, ScenarioConfig]:
    return {
        "low_load": ScenarioConfig(name="low_load", base_rate=4.5),
        "high_load": ScenarioConfig(name="high_load", base_rate=15.0),
        "bursty": ScenarioConfig(
            name="bursty",
            base_rate=7.5,
            burst_interval=22,
            burst_length=6,
            burst_multiplier=2.6,
        ),
        "oscillating": ScenarioConfig(
            name="oscillating",
            base_rate=8.5,
            oscillation_amplitude=5.0,
            oscillation_period=42,
        ),
        "training_mix": ScenarioConfig(name="training_mix", base_rate=8.0),
    }
