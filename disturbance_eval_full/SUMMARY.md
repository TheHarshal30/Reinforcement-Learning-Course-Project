# Disturbance Evaluation

## demand_surge
- ppo_cost_only: shock_reward=0.495, collapse_rate=0.250, recovery_rate=0.138
- ppo_warm_start_reg_l010: shock_reward=0.599, collapse_rate=0.250, recovery_rate=0.312
- cost_aware_priority: shock_reward=0.653, collapse_rate=0.263, recovery_rate=0.238

## budget_cut
- ppo_cost_only: shock_reward=-0.441, collapse_rate=0.900, recovery_rate=0.000
- ppo_warm_start_reg_l010: shock_reward=-0.118, collapse_rate=0.938, recovery_rate=0.000
- cost_aware_priority: shock_reward=-0.971, collapse_rate=0.887, recovery_rate=0.000

## cost_shock
- ppo_cost_only: shock_reward=-0.318, collapse_rate=0.875, recovery_rate=0.000
- ppo_warm_start_reg_l010: shock_reward=-0.299, collapse_rate=0.838, recovery_rate=0.000
- cost_aware_priority: shock_reward=-0.152, collapse_rate=0.787, recovery_rate=0.000

## action_corruption
- ppo_cost_only: shock_reward=0.685, collapse_rate=0.138, recovery_rate=0.275
- ppo_warm_start_reg_l010: shock_reward=0.682, collapse_rate=0.225, recovery_rate=0.312
- cost_aware_priority: shock_reward=0.755, collapse_rate=0.075, recovery_rate=0.425
