# Cost-Aware Reproducibility

This project now supports two environment modes:

- count-based mode: original request-count service capacity
- cost-aware mode: per-request service cost plus per-step service budget

## Files Added or Extended

- [api_rate_limiter_rl/config.py](/Users/harshalrudra/Documents/Codex/2026-05-09/https-github-com-theharshal30-reinforcement-learning/Reinforcement-Learning-Course-Project/api_rate_limiter_rl/config.py)
- [api_rate_limiter_rl/env.py](/Users/harshalrudra/Documents/Codex/2026-05-09/https-github-com-theharshal30-reinforcement-learning/Reinforcement-Learning-Course-Project/api_rate_limiter_rl/env.py)
- [api_rate_limiter_rl/baselines.py](/Users/harshalrudra/Documents/Codex/2026-05-09/https-github-com-theharshal30-reinforcement-learning/Reinforcement-Learning-Course-Project/api_rate_limiter_rl/baselines.py)
- [api_rate_limiter_rl/experiments.py](/Users/harshalrudra/Documents/Codex/2026-05-09/https-github-com-theharshal30-reinforcement-learning/Reinforcement-Learning-Course-Project/api_rate_limiter_rl/experiments.py)
- [api_rate_limiter_rl/plotting.py](/Users/harshalrudra/Documents/Codex/2026-05-09/https-github-com-theharshal30-reinforcement-learning/Reinforcement-Learning-Course-Project/api_rate_limiter_rl/plotting.py)
- [cost_aware_compare.py](/Users/harshalrudra/Documents/Codex/2026-05-09/https-github-com-theharshal30-reinforcement-learning/Reinforcement-Learning-Course-Project/cost_aware_compare.py)

## Cost-Aware Behavior

When `EnvConfig.use_request_costs=True`:

- each request is assigned `service_cost`
- queue service uses `service_budget_per_step`
- state vector includes queue-cost features
- evaluation records:
  - served cost
  - budget utilization
- OOD cost-shift scenarios are available:
  - `expensive_attack`
  - `cheap_flood`
  - `mixed_shift`

## Environment Setup

```bash
cd /Users/harshalrudra/Documents/Codex/2026-05-09/https-github-com-theharshal30-reinforcement-learning/Reinforcement-Learning-Course-Project
source ../.miniforge3/bin/activate
conda activate rl-rate-limit
```

## Smoke Test

```bash
python cost_aware_compare.py --mode screen --train-episodes 2 --eval-episodes 1 --episode-length 60 --log-every 1 --output-dir cost_aware_smoke
```

This runs:

- cost-blind PPO with `reward_high_drop_weight=1.5`
- cost-aware PPO with the same PPO hyperparameters and cost-aware environment

## Quick Screening

```bash
python cost_aware_compare.py --mode screen --train-episodes 30 --eval-episodes 3 --episode-length 180 --log-every 10 --output-dir cost_aware_screen
```

Outputs:

- `cost_aware_screen/cost_blind_hp15/`
- `cost_aware_screen/cost_aware_hp15/`
- `cost_aware_screen/screen_summary.json`

## Full Compare Run

```bash
python cost_aware_compare.py --mode full --train-episodes 260 --eval-episodes 24 --episode-length 180 --log-every 25 --output-dir cost_aware_full
```

This runs:

- count-based PPO with `reward_high_drop_weight=1.5`
- cost-aware PPO with the same PPO settings

The cost-aware run evaluates on:

- standard scenarios:
  - `low_load`
  - `high_load`
  - `bursty`
  - `oscillating`
- OOD cost-shift scenarios:
  - `expensive_attack`
  - `cheap_flood`
  - `mixed_shift`

## Suggested tmux Command

```bash
mkdir -p cost_aware_full
tmux new-session -d -s rl_cost_aware "cd /Users/harshalrudra/Documents/Codex/2026-05-09/https-github-com-theharshal30-reinforcement-learning/Reinforcement-Learning-Course-Project && source ../.miniforge3/bin/activate && conda activate rl-rate-limit && PYTHONUNBUFFERED=1 python cost_aware_compare.py --mode full --train-episodes 260 --eval-episodes 24 --episode-length 180 --log-every 25 --output-dir cost_aware_full | tee cost_aware_full/run.log"
```

Attach:

```bash
tmux attach -t rl_cost_aware
```

## Metrics to Report

For each policy and scenario:

- throughput
- latency
- drop rate
- high-priority success rate
- high-priority drop rate
- served cost
- budget utilization
- total reward

## Key Config Knobs

Useful `EnvConfig` fields:

- `use_request_costs`
- `service_budget_per_step`
- `max_expected_queue_cost`
- `max_expected_service_cost`
- `reward_high_drop_weight`
- `reward_cost_processed_weight`

Useful `ScenarioConfig` field:

- `cost_ranges_by_priority`

## Backward Compatibility

If `use_request_costs=False`, the original count-based environment remains active.

That means the existing PPO pipeline, count-based reports, and legacy experiment scripts still work.
