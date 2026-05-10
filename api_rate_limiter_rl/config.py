from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class DisturbanceConfig:
    type: str
    start_step: int
    end_step: int
    intensity: float = 1.0
    cost_range: Tuple[float, float] = (8.0, 12.0)


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
    cost_ranges_by_priority: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
        (1.0, 5.0),
        (1.0, 5.0),
        (1.0, 5.0),
    )


@dataclass
class EnvConfig:
    episode_length: int = 180
    service_capacity: int = 12
    service_budget_per_step: float = 18.0
    max_rate_limit: int = 18
    min_rate_limit: int = 2
    rate_step: int = 2
    queue_threshold: int = 24
    queue_hard_cap: int = 72
    retry_max_delay: int = 5
    max_expected_arrivals: int = 28
    max_expected_queue_cost: float = 360.0
    max_expected_service_cost: float = 5.0
    retry_probability_by_priority: Tuple[float, float, float] = (0.12, 0.22, 0.45)
    reward_throughput_weight: float = 1.1
    reward_latency_weight: float = 1.7
    reward_drop_weight: float = 1.2
    reward_high_drop_weight: float = 3.1
    reward_overload_weight: float = 1.0
    reward_cost_processed_weight: float = 0.0
    reward_budget_utilization_weight: float = 0.0
    state_drop_window: int = 5
    use_request_costs: bool = False
    use_azure_cost: bool = False
    azure_cost_values: Tuple[int, ...] = ()
    azure_cost_probabilities: Tuple[float, ...] = ()
    include_queue_trend: bool = False
    include_arrival_trend: bool = False
    include_time_phase: bool = False
    include_rejected_cost: bool = False
    disturbance: DisturbanceConfig | None = None
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
    early_stop_window: int = 0
    early_stop_patience: int = 0
    early_stop_min_delta: float = 0.05
    validation_eval_every: int = 0
    validation_eval_episodes: int = 3
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


def cost_shift_scenarios() -> Dict[str, ScenarioConfig]:
    return {
        "expensive_attack": ScenarioConfig(
            name="expensive_attack",
            base_rate=10.0,
            burst_interval=40,
            burst_length=50,
            burst_multiplier=2.0,
            priority_probs=(0.15, 0.15, 0.70),
            cost_ranges_by_priority=((1.0, 3.0), (2.0, 4.0), (10.0, 15.0)),
        ),
        "cheap_flood": ScenarioConfig(
            name="cheap_flood",
            base_rate=16.0,
            oscillation_amplitude=3.0,
            oscillation_period=30,
            priority_probs=(0.80, 0.15, 0.05),
            cost_ranges_by_priority=((0.5, 1.0), (1.5, 3.0), (3.0, 5.0)),
        ),
        "mixed_shift": ScenarioConfig(
            name="mixed_shift",
            base_rate=11.5,
            oscillation_amplitude=4.0,
            oscillation_period=21,
            burst_interval=17,
            burst_length=4,
            burst_multiplier=1.8,
            priority_probs=(0.45, 0.30, 0.25),
            cost_ranges_by_priority=((0.5, 2.0), (3.0, 6.0), (7.0, 12.0)),
        ),
    }
