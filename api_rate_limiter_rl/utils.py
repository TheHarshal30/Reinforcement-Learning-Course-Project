from __future__ import annotations

import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Sequence


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def softmax(logits: Sequence[float]) -> List[float]:
    max_logit = max(logits)
    exps = [math.exp(x - max_logit) for x in logits]
    denom = sum(exps)
    return [x / denom for x in exps]


def sample_categorical(probs: Sequence[float], rng: random.Random) -> int:
    r = rng.random()
    total = 0.0
    for idx, prob in enumerate(probs):
        total += prob
        if r <= total:
            return idx
    return len(probs) - 1


def sample_poisson(lam: float, rng: random.Random) -> int:
    if lam <= 0:
        return 0
    if lam < 30.0:
        limit = math.exp(-lam)
        k = 0
        prod = 1.0
        while prod > limit:
            k += 1
            prod *= rng.random()
        return k - 1
    # Gaussian approximation for larger rates.
    value = int(round(rng.gauss(lam, math.sqrt(lam))))
    return max(0, value)


def sliding_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def stdev(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))

