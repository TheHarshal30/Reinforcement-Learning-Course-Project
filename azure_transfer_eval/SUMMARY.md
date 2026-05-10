# Azure Cost Transfer Evaluation

## low_load
- ppo_cost_only: reward=16.886, throughput=2.796, latency=7.397, drop=0.303, hp_success=0.496, served_cost=18.700, budget_utilization=0.779
- ppo_warm_start_reg_l010: reward=22.214, throughput=2.828, latency=7.186, drop=0.299, hp_success=0.505, served_cost=18.804, budget_utilization=0.784
- cost_aware_priority: reward=66.880, throughput=2.769, latency=4.355, drop=0.304, hp_success=0.518, served_cost=18.597, budget_utilization=0.775
- priority_heuristic: reward=49.490, throughput=2.801, latency=5.586, drop=0.289, hp_success=0.503, served_cost=18.726, budget_utilization=0.780

## high_load
- ppo_cost_only: reward=-208.662, throughput=2.926, latency=7.787, drop=0.806, hp_success=0.839, served_cost=19.140, budget_utilization=0.797
- ppo_warm_start_reg_l010: reward=-90.955, throughput=2.729, latency=4.612, drop=0.828, hp_success=0.811, served_cost=18.561, budget_utilization=0.773
- cost_aware_priority: reward=-118.822, throughput=2.910, latency=5.477, drop=0.807, hp_success=0.857, served_cost=19.074, budget_utilization=0.795
- priority_heuristic: reward=-104.532, throughput=2.872, latency=5.830, drop=0.815, hp_success=0.850, served_cost=18.902, budget_utilization=0.788

## bursty
- ppo_cost_only: reward=-124.311, throughput=2.865, latency=6.890, drop=0.693, hp_success=0.731, served_cost=19.068, budget_utilization=0.795
- ppo_warm_start_reg_l010: reward=-62.056, throughput=2.784, latency=5.463, drop=0.696, hp_success=0.721, served_cost=18.722, budget_utilization=0.780
- cost_aware_priority: reward=-78.989, throughput=2.985, latency=5.372, drop=0.670, hp_success=0.774, served_cost=19.248, budget_utilization=0.802
- priority_heuristic: reward=-53.194, throughput=2.903, latency=5.140, drop=0.681, hp_success=0.750, served_cost=19.011, budget_utilization=0.792

## oscillating
- ppo_cost_only: reward=-112.369, throughput=2.878, latency=6.125, drop=0.730, hp_success=0.794, served_cost=18.966, budget_utilization=0.790
- ppo_warm_start_reg_l010: reward=-50.857, throughput=2.736, latency=4.453, drop=0.748, hp_success=0.789, served_cost=18.564, budget_utilization=0.774
- cost_aware_priority: reward=-68.661, throughput=2.946, latency=5.024, drop=0.712, hp_success=0.808, served_cost=19.130, budget_utilization=0.797
- priority_heuristic: reward=-61.186, throughput=2.845, latency=5.004, drop=0.729, hp_success=0.814, served_cost=18.813, budget_utilization=0.784
