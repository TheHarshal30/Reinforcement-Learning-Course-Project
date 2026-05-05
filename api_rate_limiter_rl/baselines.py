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


@dataclass
class FixedRatePolicy(BasePolicy):
    rate_limit: int

    def act(self, state, info=None):
        return 1, self.rate_limit


@dataclass
class TokenBucketPolicy(BasePolicy):
    fill_rate: float
    bucket_size: float

    def __post_init__(self):
        self.tokens = self.bucket_size
        self.last_limit = int(self.fill_rate)

    def reset(self):
        self.tokens = self.bucket_size
        self.last_limit = int(self.fill_rate)

    def act(self, state, info=None):
        limit = max(1, int(self.tokens))
        self.last_limit = limit
        return 1, limit

    def observe(self, info):
        admitted = info.get("accepted", 0)
        self.tokens = min(self.bucket_size, self.tokens + self.fill_rate - admitted)


@dataclass
class AdaptiveThresholdPolicy(BasePolicy):
    low_queue: int
    high_queue: int
    min_limit: int
    max_limit: int
    step: int

    def __post_init__(self):
        self.limit = (self.min_limit + self.max_limit) // 2

    def reset(self):
        self.limit = (self.min_limit + self.max_limit) // 2

    def act(self, state, info=None):
        queue_length = 0.0 if not state else state[1] * 72.0
        if queue_length > self.high_queue:
            self.limit = min(self.max_limit, self.limit + self.step)
        elif queue_length < self.low_queue:
            self.limit = max(self.min_limit, self.limit - self.step)
        return 1, self.limit

    def observe(self, info):
        queue_length = info.get("queue_length", 0)
        if queue_length > self.high_queue:
            self.limit = min(self.max_limit, self.limit + self.step)
        elif queue_length < self.low_queue:
            self.limit = max(self.min_limit, self.limit - self.step)


@dataclass
class HeuristicPriorityPolicy(BasePolicy):
    min_limit: int
    max_limit: int

    def reset(self):
        pass

    def act(self, state, info=None):
        queue_norm = state[1] if state else 0.0
        retry_norm = state[3] if state else 0.0
        limit = self.min_limit + int((1.0 - queue_norm) * (self.max_limit - self.min_limit))
        if retry_norm > 0.1:
            limit = min(self.max_limit, limit + 2)
        return 1, max(self.min_limit, min(self.max_limit, limit))

