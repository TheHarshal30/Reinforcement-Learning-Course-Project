from __future__ import annotations

from dataclasses import dataclass
import random


class BasePolicy:
    def reset(self):
        pass

    def act(self, state, info=None):
        raise NotImplementedError

    def observe(self, info):
        pass


def _estimated_request_cost(info, fallback: float = 1.0) -> float:
    if not info:
        return fallback
    candidates = [
        info.get("mean_arrival_cost", 0.0),
        info.get("mean_queue_cost", 0.0),
    ]
    for value in candidates:
        if value and value > 0.0:
            return value
    return fallback


def _limit_from_budget(budget: float, info, min_limit: int, max_limit: int, fallback_cost: float = 1.0) -> int:
    est_cost = max(0.25, _estimated_request_cost(info, fallback=fallback_cost))
    desired_limit = int(round(budget / est_cost))
    return max(min_limit, min(max_limit, desired_limit))


@dataclass
class FixedRatePolicy(BasePolicy):
    rate_limit: int
    target_cost_budget: float | None = None
    min_limit: int = 1

    def act(self, state, info=None):
        if self.target_cost_budget is not None:
            return 1, _limit_from_budget(self.target_cost_budget, info, self.min_limit, self.rate_limit)
        return 1, self.rate_limit


@dataclass
class AlwaysMaxPolicy(BasePolicy):
    max_limit: int

    def act(self, state, info=None):
        return 1, self.max_limit


@dataclass
class TokenBucketPolicy(BasePolicy):
    fill_rate: float
    bucket_size: float
    target_cost_budget: float | None = None
    min_limit: int = 1

    def __post_init__(self):
        self.tokens = self.bucket_size
        self.last_limit = int(self.fill_rate)

    def reset(self):
        self.tokens = self.bucket_size
        self.last_limit = int(self.fill_rate)

    def act(self, state, info=None):
        if self.target_cost_budget is not None:
            budget = min(self.target_cost_budget, self.tokens)
            limit = _limit_from_budget(budget, info, self.min_limit, int(max(self.bucket_size, 1)))
        else:
            limit = max(1, int(self.tokens))
        self.last_limit = limit
        return 1, limit

    def observe(self, info):
        spent = info.get("accepted_cost", 0.0) if self.target_cost_budget is not None else info.get("accepted", 0)
        self.tokens = min(self.bucket_size, self.tokens + self.fill_rate - spent)


@dataclass
class AdaptiveThresholdPolicy(BasePolicy):
    low_queue: int
    high_queue: int
    min_limit: int
    max_limit: int
    step: int
    target_queue_cost: float | None = None

    def __post_init__(self):
        self.limit = (self.min_limit + self.max_limit) // 2

    def reset(self):
        self.limit = (self.min_limit + self.max_limit) // 2

    def act(self, state, info=None):
        queue_length = 0.0 if not state else state[1] * 72.0
        queue_pressure = info.get("queue_total_cost", queue_length) if info and self.target_queue_cost is not None else queue_length
        high_threshold = self.target_queue_cost if self.target_queue_cost is not None else self.high_queue
        low_threshold = max(1.0, high_threshold / 2.0) if self.target_queue_cost is not None else self.low_queue
        if queue_pressure > high_threshold:
            self.limit = min(self.max_limit, self.limit + self.step)
        elif queue_pressure < low_threshold:
            self.limit = max(self.min_limit, self.limit - self.step)
        return 1, self.limit

    def observe(self, info):
        queue_length = info.get("queue_total_cost", 0.0) if self.target_queue_cost is not None else info.get("queue_length", 0)
        high_threshold = self.target_queue_cost if self.target_queue_cost is not None else self.high_queue
        low_threshold = max(1.0, high_threshold / 2.0) if self.target_queue_cost is not None else self.low_queue
        if queue_length > high_threshold:
            self.limit = min(self.max_limit, self.limit + self.step)
        elif queue_length < low_threshold:
            self.limit = max(self.min_limit, self.limit - self.step)


@dataclass
class HeuristicPriorityPolicy(BasePolicy):
    min_limit: int
    max_limit: int
    target_cost_budget: float | None = None

    def reset(self):
        pass

    def act(self, state, info=None):
        queue_norm = state[1] if state else 0.0
        retry_norm = state[3] if state else 0.0
        limit = self.min_limit + int((1.0 - queue_norm) * (self.max_limit - self.min_limit))
        if retry_norm > 0.1:
            limit = min(self.max_limit, limit + 2)
        if self.target_cost_budget is not None:
            limit = _limit_from_budget(self.target_cost_budget * (1.05 - queue_norm), info, self.min_limit, self.max_limit)
        return 1, max(self.min_limit, min(self.max_limit, limit))


@dataclass
class CostAwarePriorityPolicy(BasePolicy):
    min_limit: int
    max_limit: int
    target_cost_budget: float

    def act(self, state, info=None):
        queue_total_cost = info.get("queue_total_cost", 0.0) if info else 0.0
        high_queue_cost = queue_total_cost
        if info and "mean_queue_cost" in info and info.get("queue_length", 0) > 0:
            high_queue_cost = min(queue_total_cost, info["mean_queue_cost"] * max(1, info.get("queue_length", 0)))
        budget = self.target_cost_budget
        if high_queue_cost > self.target_cost_budget:
            budget *= 0.8
        return 1, _limit_from_budget(budget, info, self.min_limit, self.max_limit)


@dataclass
class CostEfficiencyPolicy(BasePolicy):
    min_limit: int
    max_limit: int
    target_cost_budget: float

    def act(self, state, info=None):
        mean_cost = _estimated_request_cost(info, fallback=1.0)
        utility_scale = 1.0
        if info:
            high_success = info.get("high_priority_success_rate", 0.5)
            utility_scale += 0.5 * high_success
        desired_budget = self.target_cost_budget * utility_scale
        return 1, _limit_from_budget(desired_budget, info, self.min_limit, self.max_limit, fallback_cost=mean_cost)
