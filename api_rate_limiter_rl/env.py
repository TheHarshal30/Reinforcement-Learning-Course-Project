from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple
import random

from .config import EnvConfig, ScenarioConfig
from .utils import clamp, sample_categorical, sample_poisson, sliding_mean


PRIORITY_LOW = 0
PRIORITY_MED = 1
PRIORITY_HIGH = 2


@dataclass
class Request:
    created_step: int
    priority: int
    retry_count: int = 0


class RateLimitEnv:
    """Discrete-time simulator with retries, queueing, and priority-aware service."""

    action_decrease = 0
    action_hold = 1
    action_increase = 2

    def __init__(self, scenario: ScenarioConfig, config: EnvConfig, seed: int | None = None):
        self.scenario = scenario
        self.config = config
        self.seed = config.seed if seed is None else seed
        self.rng = random.Random(self.seed)
        self.reset(self.seed)

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng.seed(seed)
        self.time_step = 0
        self.rate_limit = self.config.service_capacity
        self.queue: Deque[Request] = deque()
        self.retry_buffer: Dict[int, List[Request]] = defaultdict(list)
        self.recent_drop_rates = deque(maxlen=self.config.state_drop_window)
        self.history = {
            "reward": [],
            "throughput": [],
            "latency": [],
            "drop_rate": [],
            "action": [],
        }
        self.last_info = {}
        return self._state(
            incoming_rate=self.scenario.base_rate,
            arrivals_by_priority=(0, 0, 0),
            retry_count=0,
        )

    def _traffic_rate(self, step: int) -> float:
        rate = self.scenario.base_rate
        if self.scenario.oscillation_period:
            import math

            phase = (2.0 * math.pi * step) / self.scenario.oscillation_period
            rate += self.scenario.oscillation_amplitude * (0.5 + 0.5 * math.sin(phase))
        if self.scenario.burst_interval and step % self.scenario.burst_interval < self.scenario.burst_length:
            rate *= self.scenario.burst_multiplier
        return max(0.1, rate)

    def _sample_priorities(self, count: int) -> List[int]:
        probs = self.scenario.priority_probs
        return [sample_categorical(probs, self.rng) for _ in range(count)]

    def _scheduled_retries(self) -> List[Request]:
        retry_requests = self.retry_buffer.pop(self.time_step, [])
        return list(retry_requests)

    def _state(self, incoming_rate: float, arrivals_by_priority: Tuple[int, int, int], retry_count: int):
        queue_counts = [0, 0, 0]
        for request in self.queue:
            queue_counts[request.priority] += 1
        queue_total = len(self.queue)
        arrival_total = sum(arrivals_by_priority)
        drop_rate = sliding_mean(self.recent_drop_rates)
        buffered_retries = sum(len(bucket) for bucket in self.retry_buffer.values())
        retry_rate = (retry_count + buffered_retries) / max(1.0, self.config.max_expected_arrivals)
        incoming_dist = [c / max(1, arrival_total) for c in arrivals_by_priority]
        queue_dist = [c / max(1, queue_total) for c in queue_counts]
        return [
            incoming_rate / max(1.0, self.config.max_expected_arrivals),
            queue_total / max(1.0, self.config.queue_hard_cap),
            drop_rate,
            retry_rate,
            self.rate_limit / max(1.0, self.config.max_rate_limit),
            *incoming_dist,
            *queue_dist,
        ]

    def step(self, action: int):
        if action == self.action_decrease:
            self.rate_limit = clamp(
                self.rate_limit - self.config.rate_step,
                self.config.min_rate_limit,
                self.config.max_rate_limit,
            )
        elif action == self.action_increase:
            self.rate_limit = clamp(
                self.rate_limit + self.config.rate_step,
                self.config.min_rate_limit,
                self.config.max_rate_limit,
            )
        else:
            self.rate_limit = clamp(
                self.rate_limit,
                self.config.min_rate_limit,
                self.config.max_rate_limit,
            )

        incoming_rate = self._traffic_rate(self.time_step)
        incoming_count = sample_poisson(incoming_rate, self.rng)
        arrivals = [Request(created_step=self.time_step, priority=p) for p in self._sample_priorities(incoming_count)]

        retry_arrivals = self._scheduled_retries()
        arrivals.extend(retry_arrivals)
        arrivals_by_priority = [0, 0, 0]
        for req in arrivals:
            arrivals_by_priority[req.priority] += 1

        arrivals.sort(key=lambda req: (-req.priority, req.created_step, req.retry_count))
        accepted = arrivals[: int(self.rate_limit)]
        rejected = arrivals[int(self.rate_limit) :]

        for req in rejected:
            delay = 1 + self.rng.randint(0, self.config.retry_max_delay - 1)
            if self.rng.random() < self.config.retry_probability_by_priority[req.priority]:
                scheduled_step = self.time_step + delay
                self.retry_buffer[scheduled_step].append(
                    Request(
                        created_step=req.created_step,
                        priority=req.priority,
                        retry_count=req.retry_count + 1,
                    )
                )

        for req in accepted:
            self.queue.append(req)

        # Hard-cap overflow drops keep simulator bounded.
        overflow = []
        while len(self.queue) > self.config.queue_hard_cap:
            overflow.append(self.queue.pop())

        service_budget = min(self.config.service_capacity, len(self.queue))
        service_candidates = sorted(
            list(self.queue),
            key=lambda req: (-req.priority, req.created_step, req.retry_count),
        )
        served = service_candidates[:service_budget]
        served_set = set(id(req) for req in served)
        self.queue = deque(req for req in self.queue if id(req) not in served_set)

        current_queue_pressure = len(self.queue) / max(1.0, self.config.queue_threshold)
        latency_values = [
            (self.time_step - req.created_step + 1) + 0.15 * current_queue_pressure
            for req in served
        ]
        average_latency = sum(latency_values) / len(latency_values) if latency_values else 0.0

        total_dropped = len(rejected) + len(overflow)
        high_priority_dropped = sum(1 for req in rejected + overflow if req.priority == PRIORITY_HIGH)
        total_arrivals = max(1, len(arrivals))
        drop_rate = total_dropped / total_arrivals
        self.recent_drop_rates.append(drop_rate)

        throughput = len(served)
        overload = max(0.0, len(self.queue) - self.config.queue_threshold) / max(
            1.0, self.config.queue_threshold
        )
        high_total = sum(1 for req in arrivals if req.priority == PRIORITY_HIGH)
        high_success = sum(1 for req in served if req.priority == PRIORITY_HIGH)
        high_success_rate = high_success / max(1, high_total)

        reward = (
            self.config.reward_throughput_weight * (throughput / max(1.0, self.config.service_capacity))
            - self.config.reward_latency_weight * (average_latency / 18.0)
            - self.config.reward_drop_weight * drop_rate
            - self.config.reward_high_drop_weight * (high_priority_dropped / max(1, max(high_total, 1)))
            - self.config.reward_overload_weight * overload
        )

        self.history["reward"].append(reward)
        self.history["throughput"].append(throughput)
        self.history["latency"].append(average_latency)
        self.history["drop_rate"].append(drop_rate)
        self.history["action"].append(action)

        self.last_info = {
            "incoming_rate": incoming_rate,
            "incoming_count": incoming_count,
            "accepted": len(accepted),
            "rejected": len(rejected),
            "served": len(served),
            "queue_length": len(self.queue),
            "average_latency": average_latency,
            "drop_rate": drop_rate,
            "high_priority_success_rate": high_success_rate,
            "high_priority_dropped": high_priority_dropped,
            "total_arrivals": len(arrivals),
            "retry_count": len(retry_arrivals),
            "overflow_dropped": len(overflow),
            "rate_limit": self.rate_limit,
            "reward": reward,
        }

        self.time_step += 1
        done = self.time_step >= self.config.episode_length
        next_state = self._state(
            incoming_rate=incoming_rate,
            arrivals_by_priority=tuple(arrivals_by_priority),
            retry_count=len(retry_arrivals),
        )
        return next_state, reward, done, self.last_info
