# RL-Based Intelligent API Rate Limiting

Standalone simulator + PPO training pipeline for adaptive API rate limiting.

## What it includes

- Discrete-time traffic simulator with Poisson arrivals, bursts, oscillations, and retries
- Priority-aware queueing and service
- PPO actor-critic agent with GAE and clipped surrogate updates
- Baselines: fixed threshold, token bucket, adaptive threshold
- Evaluation across low-load, high-load, bursty, and oscillating traffic
- Training and comparison plots in SVG and PNG

## Run

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
# Install PyTorch for your local accelerator/runtime.
# Apple Silicon example: pip install torch
# NVIDIA CUDA example: pip install torch --index-url https://download.pytorch.org/whl/cu121
.venv/bin/python run_experiments.py
```

## Outputs

- `artifacts/REPORT.md`
- `artifacts/evaluation_results.json`
- `artifacts/training_history.json`
- `artifacts/plots/`

## Notes

- PPO uses only the simulator environment.
- Retry traffic is buffered across timesteps and rescheduled after delayed drops.
- Service is priority-aware: higher priority requests are processed first when the queue builds.
- `PPOConfig.device = "auto"` uses CUDA when available, else Apple `mps` when available, else CPU.
