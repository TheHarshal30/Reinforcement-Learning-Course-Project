from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple
import math
import random

from .config import DisturbanceConfig, EnvConfig, ScenarioConfig
from .utils import clamp, sample_categorical, sample_poisson, sliding_mean


PRIORITY_LOW = 0
PRIORITY_MED = 1
PRIORITY_HIGH = 2


@dataclass
class Request:
    created_step: int
    priority: int
    service_cost: float = 1.0
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
        self.prev_queue_total = 0
        self.prev_incoming_rate = self.scenario.base_rate
        self.prev_queue_cost = 0.0
        self.prev_rejected_cost = 0.0
        self.last_info = {}
        return self._state(
            incoming_rate=self.scenario.base_rate,
            arrivals_by_priority=(0, 0, 0),
            arrivals_cost_by_priority=(0.0, 0.0, 0.0),
            retry_count=0,
        )

    def _traffic_rate(self, step: int) -> float:
        rate = self.scenario.base_rate
        if self.scenario.oscillation_period:
            phase = (2.0 * math.pi * step) / self.scenario.oscillation_period
            rate += self.scenario.oscillation_amplitude * (0.5 + 0.5 * math.sin(phase))
        if self.scenario.burst_interval and step % self.scenario.burst_interval < self.scenario.burst_length:
            rate *= self.scenario.burst_multiplier
        return max(0.1, rate)

    def _active_disturbance(self) -> DisturbanceConfig | None:
        disturbance = self.config.disturbance
        if disturbance is None:
            return None
        if disturbance.start_step <= self.time_step <= disturbance.end_step:
            return disturbance
        return None

    def _service_budget_for_step(self) -> float:
        disturbance = self._active_disturbance()
        if disturbance is not None and disturbance.type == "budget_cut":
            return max(1.0, disturbance.intensity)
        return self.config.service_budget_per_step

    def maybe_corrupt_action(self, action: int) -> int:
        disturbance = self._active_disturbance()
        if disturbance is None or disturbance.type != "action_corruption":
            return action
        if self.rng.random() < disturbance.intensity:
            return self.rng.choice([self.action_decrease, self.action_hold, self.action_increase])
        return action

    def _sample_priorities(self, count: int) -> List[int]:
        probs = self.scenario.priority_probs
        return [sample_categorical(probs, self.rng) for _ in range(count)]

    def _scheduled_retries(self) -> List[Request]:
        retry_requests = self.retry_buffer.pop(self.time_step, [])
        return list(retry_requests)

    def _sample_request_cost(self, priority: int) -> float:
        disturbance = self._active_disturbance()
        if disturbance is not None and disturbance.type == "cost_shock":
            low, high = disturbance.cost_range
            return self.rng.uniform(low, high)
        if (
            self.config.use_azure_cost
            and self.config.azure_cost_values
            and self.config.azure_cost_probabilities
        ):
            idx = sample_categorical(self.config.azure_cost_probabilities, self.rng)
            return float(self.config.azure_cost_values[idx])
        low, high = self.scenario.cost_ranges_by_priority[priority]
        return self.rng.uniform(low, high)

    def _queue_cost_breakdown(self):
        queue_costs = [0.0, 0.0, 0.0]
        for request in self.queue:
            queue_costs[request.priority] += request.service_cost
        return queue_costs

    def _state(
        self,
        incoming_rate: float,
        arrivals_by_priority: Tuple[int, int, int],
        arrivals_cost_by_priority: Tuple[float, float, float],
        retry_count: int,
    ):
        queue_counts = [0, 0, 0]
        for request in self.queue:
            queue_counts[request.priority] += 1
        queue_costs = self._queue_cost_breakdown()
        queue_total = len(self.queue)
        queue_total_cost = sum(queue_costs)
        arrival_total = sum(arrivals_by_priority)
        arrival_total_cost = sum(arrivals_cost_by_priority)
        drop_rate = sliding_mean(self.recent_drop_rates)
        buffered_retries = sum(len(bucket) for bucket in self.retry_buffer.values())
        retry_rate = (retry_count + buffered_retries) / max(1.0, self.config.max_expected_arrivals)
        incoming_dist = [c / max(1, arrival_total) for c in arrivals_by_priority]
        queue_dist = [c / max(1, queue_total) for c in queue_counts]
        state = [
            incoming_rate / max(1.0, self.config.max_expected_arrivals),
            queue_total / max(1.0, self.config.queue_hard_cap),
            drop_rate,
            retry_rate,
            self.rate_limit / max(1.0, self.config.max_rate_limit),
            *incoming_dist,
            *queue_dist,
        ]
        if self.config.use_request_costs:
            mean_queue_cost = queue_total_cost / max(1, queue_total)
            state.extend(
                [
                    queue_total_cost / max(1.0, self.config.max_expected_queue_cost),
                    mean_queue_cost / max(1.0, self.config.max_expected_service_cost),
                    queue_costs[PRIORITY_HIGH] / max(1.0, self.config.max_expected_queue_cost),
                    queue_costs[PRIORITY_MED] / max(1.0, self.config.max_expected_queue_cost),
                    queue_costs[PRIORITY_LOW] / max(1.0, self.config.max_expected_queue_cost),
                    arrival_total_cost / max(
                        1.0, self.config.max_expected_arrivals * self.config.max_expected_service_cost
                    ),
                ]
            )
        if self.config.include_queue_trend:
            queue_trend = (queue_total - self.prev_queue_total) / max(1.0, self.config.queue_hard_cap)
            state.append(queue_trend)
            if self.config.use_request_costs:
                queue_cost_trend = (queue_total_cost - self.prev_queue_cost) / max(
                    1.0, self.config.max_expected_queue_cost
                )
                state.append(queue_cost_trend)
        if self.config.include_arrival_trend:
            arrival_trend = (incoming_rate - self.prev_incoming_rate) / max(1.0, self.config.max_expected_arrivals)
            state.append(arrival_trend)
        if self.config.include_rejected_cost:
            state.append(
                self.prev_rejected_cost
                / max(1.0, self.config.max_expected_arrivals * self.config.max_expected_service_cost)
            )
        if self.config.include_time_phase:
            phase = (2.0 * math.pi * self.time_step) / max(1, self.config.episode_length)
            state.extend([math.sin(phase), math.cos(phase)])
        return state

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

        disturbance = self._active_disturbance()
        incoming_rate = self._traffic_rate(self.time_step)
        incoming_count = sample_poisson(incoming_rate, self.rng)
        arrivals = [
            Request(
                created_step=self.time_step,
                priority=p,
                service_cost=self._sample_request_cost(p) if self.config.use_request_costs else 1.0,
            )
            for p in self._sample_priorities(incoming_count)
        ]
        if disturbance is not None and disturbance.type == "demand_surge":
            extra_high = sample_poisson(
                incoming_rate * max(0.0, disturbance.intensity - 1.0) * self.scenario.priority_probs[PRIORITY_HIGH],
                self.rng,
            )
            arrivals.extend(
                [
                    Request(
                        created_step=self.time_step,
                        priority=PRIORITY_HIGH,
                        service_cost=self._sample_request_cost(PRIORITY_HIGH) if self.config.use_request_costs else 1.0,
                    )
                    for _ in range(extra_high)
                ]
            )

        retry_arrivals = self._scheduled_retries()
        arrivals.extend(retry_arrivals)
        arrivals_by_priority = [0, 0, 0]
        arrivals_cost_by_priority = [0.0, 0.0, 0.0]
        for req in arrivals:
            arrivals_by_priority[req.priority] += 1
            arrivals_cost_by_priority[req.priority] += req.service_cost
        arrival_total_cost = sum(arrivals_cost_by_priority)

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
                        service_cost=req.service_cost,
                        retry_count=req.retry_count + 1,
                    )
                )

        for req in accepted:
            self.queue.append(req)

        # Hard-cap overflow drops keep simulator bounded.
        overflow = []
        while len(self.queue) > self.config.queue_hard_cap:
            overflow.append(self.queue.pop())

        service_candidates = sorted(
            list(self.queue),
            key=lambda req: (-req.priority, req.created_step, req.retry_count),
        )
        service_budget_now = self._service_budget_for_step()
        if self.config.use_request_costs:
            remaining_budget = service_budget_now
            served = []
            for req in service_candidates:
                if req.service_cost <= remaining_budget:
                    served.append(req)
                    remaining_budget -= req.service_cost
        else:
            service_budget = min(self.config.service_capacity, len(self.queue))
            served = service_candidates[:service_budget]
        served_set = set(id(req) for req in served)
        self.queue = deque(req for req in self.queue if id(req) not in served_set)
        served_total_cost = sum(req.service_cost for req in served)
        queue_total_cost = sum(req.service_cost for req in self.queue)

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
        rejected_total_cost = sum(req.service_cost for req in rejected)
        overflow_total_cost = sum(req.service_cost for req in overflow)

        throughput = len(served)
        budget_utilization = (
            served_total_cost / max(1.0, service_budget_now)
            if self.config.use_request_costs
            else throughput / max(1.0, self.config.service_capacity)
        )
        overload = max(0.0, len(self.queue) - self.config.queue_threshold) / max(
            1.0, self.config.queue_threshold
        )
        high_total = sum(1 for req in arrivals if req.priority == PRIORITY_HIGH)
        high_success = sum(1 for req in served if req.priority == PRIORITY_HIGH)
        high_success_rate = high_success / max(1, high_total)
        high_priority_drop_rate = high_priority_dropped / max(1, high_total)

        reward_throughput = self.config.reward_throughput_weight * (throughput / max(1.0, self.config.service_capacity))
        reward_latency = -self.config.reward_latency_weight * (average_latency / 18.0)
        reward_drop = -self.config.reward_drop_weight * drop_rate
        reward_high_drop = -self.config.reward_high_drop_weight * (high_priority_dropped / max(1, max(high_total, 1)))
        reward_overload = -self.config.reward_overload_weight * overload
        reward_cost_processed = (
            self.config.reward_cost_processed_weight
            * (served_total_cost / max(1.0, service_budget_now))
        )
        reward_budget_utilization = self.config.reward_budget_utilization_weight * budget_utilization
        reward = (
            reward_throughput
            + reward_latency
            + reward_drop
            + reward_high_drop
            + reward_overload
            + reward_cost_processed
            + reward_budget_utilization
        )

        self.history["reward"].append(reward)
        self.history["throughput"].append(throughput)
        self.history["latency"].append(average_latency)
        self.history["drop_rate"].append(drop_rate)
        self.history["action"].append(action)
        if self.config.use_request_costs:
            self.history.setdefault("processed_cost", []).append(served_total_cost)
            self.history.setdefault("budget_utilization", []).append(budget_utilization)

        self.last_info = {
            "incoming_rate": incoming_rate,
            "incoming_count": incoming_count,
            "accepted": len(accepted),
            "accepted_cost": sum(req.service_cost for req in accepted),
            "rejected": len(rejected),
            "rejected_cost": rejected_total_cost,
            "overflow_cost": overflow_total_cost,
            "served": len(served),
            "served_cost": served_total_cost,
            "budget_utilization": budget_utilization,
            "queue_length": len(self.queue),
            "queue_total_cost": queue_total_cost,
            "mean_queue_cost": queue_total_cost / max(1, len(self.queue)),
            "mean_arrival_cost": arrival_total_cost / max(1, len(arrivals)),
            "disturbance_type": disturbance.type if disturbance is not None else "none",
            "disturbance_active": disturbance is not None,
            "service_budget_now": service_budget_now,
            "average_latency": average_latency,
            "drop_rate": drop_rate,
            "high_priority_success_rate": high_success_rate,
            "high_priority_drop_rate": high_priority_drop_rate,
            "high_priority_dropped": high_priority_dropped,
            "total_arrivals": len(arrivals),
            "retry_count": len(retry_arrivals),
            "overflow_dropped": len(overflow),
            "rate_limit": self.rate_limit,
            "action": action,
            "reward": reward,
            "reward_throughput": reward_throughput,
            "reward_latency": reward_latency,
            "reward_drop": reward_drop,
            "reward_high_drop": reward_high_drop,
            "reward_overload": reward_overload,
            "reward_cost_processed": reward_cost_processed,
            "reward_budget_utilization": reward_budget_utilization,
        }

        self.time_step += 1
        done = self.time_step >= self.config.episode_length
        next_state = self._state(
            incoming_rate=incoming_rate,
            arrivals_by_priority=tuple(arrivals_by_priority),
            arrivals_cost_by_priority=tuple(arrivals_cost_by_priority),
            retry_count=len(retry_arrivals),
        )
        self.prev_queue_total = len(self.queue)
        self.prev_incoming_rate = incoming_rate
        self.prev_queue_cost = queue_total_cost
        self.prev_rejected_cost = rejected_total_cost + overflow_total_cost
        return next_state, reward, done, self.last_info
