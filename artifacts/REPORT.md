# RL-Based Intelligent API Rate Limiting

## Design
- Episode length: 180
- Service capacity: 12
- Rate limit range: 2..18
- PPO hidden size: 48
- PPO gamma / lambda / clip: 0.97 / 0.93 / 0.2

## Training
- Episodes: 260
- Final mean reward: 0.6033
- Final mean throughput: 9.3672
- Final mean latency: 1.6035
- Final mean drop rate: 0.0830

## Key Takeaways
- PPO cuts high-load latency versus fixed control (5.351 -> 2.934) while holding throughput near the service ceiling (11.990).
- PPO improves bursty throughput over fixed control (9.416 -> 9.803) and keeps bursty drops lower (0.076 -> 0.055).
- PPO improves oscillating throughput over fixed control (10.521 -> 10.671) while keeping drop rate low (0.030 -> 0.022).
- The token bucket baseline remains strong on some latency cases, so the learned policy is competitive rather than trivially dominant.

## Evaluation
### bursty
- adaptive: throughput=2.110, latency=1.845, drop=0.783, high-priority=0.670, reward=-237.851
- fixed: throughput=9.416, latency=1.273, drop=0.076, high-priority=0.748, reward=117.359
- ppo: throughput=9.803, latency=1.580, drop=0.055, high-priority=0.743, reward=122.680
- priority_heuristic: throughput=9.639, latency=1.504, drop=0.057, high-priority=0.744, reward=121.205
- token_bucket: throughput=9.187, latency=1.281, drop=0.088, high-priority=0.747, reward=110.846

### high_load
- adaptive: throughput=2.111, latency=2.107, drop=0.871, high-priority=0.694, reward=-321.328
- fixed: throughput=11.994, latency=5.351, drop=0.161, high-priority=0.893, reward=-132.902
- ppo: throughput=11.990, latency=2.934, drop=0.177, high-priority=0.896, reward=101.310
- priority_heuristic: throughput=11.998, latency=3.166, drop=0.175, high-priority=0.890, reward=95.967
- token_bucket: throughput=11.976, latency=1.584, drop=0.184, high-priority=0.896, reward=131.033

### low_load
- adaptive: throughput=1.999, latency=1.352, drop=0.505, high-priority=0.492, reward=-106.270
- fixed: throughput=4.513, latency=0.988, drop=0.000, high-priority=0.484, reward=57.661
- ppo: throughput=4.508, latency=0.987, drop=0.000, high-priority=0.482, reward=57.598
- priority_heuristic: throughput=4.508, latency=0.987, drop=0.000, high-priority=0.482, reward=57.598
- token_bucket: throughput=4.508, latency=0.987, drop=0.000, high-priority=0.482, reward=57.598

### oscillating
- adaptive: throughput=2.104, latency=1.809, drop=0.814, high-priority=0.710, reward=-242.472
- fixed: throughput=10.521, latency=1.259, drop=0.030, high-priority=0.807, reward=145.811
- ppo: throughput=10.671, latency=1.367, drop=0.022, high-priority=0.793, reward=148.032
- priority_heuristic: throughput=10.728, latency=1.457, drop=0.015, high-priority=0.799, reward=149.013
- token_bucket: throughput=10.555, latency=1.226, drop=0.034, high-priority=0.802, reward=145.957

## Notes
- Retries are delayed, buffered across timesteps, and can recursively retry after repeated drops.
- Queue service is priority-aware, so high-priority traffic is handled first under congestion.
- PPO uses clipped surrogate updates with GAE and on-policy rollouts.
