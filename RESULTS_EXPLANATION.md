# Results Explanation

This document explains every result generated during this working session, what each run was trying to show, and which conclusions are safe to use.

## 1. Initial Sample Run

This was the first full default run after the environment was created.

Configuration:

- default reward weights
- default PPO config
- default training episodes: `260`
- default evaluation episodes: `24`

Reported result at that time:

- final mean reward: `0.4734`
- final mean throughput: `9.4200`
- final mean latency: `2.0109`
- final mean drop rate: `0.0841`

Interpretation:

- this was a valid baseline-style run
- it showed PPO could train and produce sensible outputs
- it was only a single seed, so useful as a demo but weak as evidence

## 2. Mac M5 / MPS Run

After patching device selection to support Apple `mps`, another full run was executed.

Configuration:

- same default training structure
- device auto-resolved to `mps`
- default training episodes: `260`
- default evaluation episodes: `24`

Saved file:

- [artifacts/REPORT.md](/Users/harshalrudra/Documents/Codex/2026-05-09/https-github-com-theharshal30-reinforcement-learning/Reinforcement-Learning-Course-Project/artifacts/REPORT.md)

Reported result:

- final mean reward: `0.1412`
- final mean throughput: `9.6103`
- final mean latency: `2.8188`
- final mean drop rate: `0.0564`

Key scenario outcomes from that report:

- `low_load`: PPO roughly matched simple baselines
- `high_load`: PPO throughput was high, but latency was worse than fixed and much worse than token bucket
- `bursty`: PPO improved throughput and reduced drops versus fixed control
- `oscillating`: PPO improved throughput and reduced drops, but not enough to dominate the strongest heuristics

Important caveat:

The generated takeaway sentence in the report says PPO "cuts high-load latency" versus fixed control, but the numbers show the opposite:

- fixed latency: `5.351`
- PPO latency: `5.805`

So the report text is partly misleading. The metric table is the reliable part.

Interpretation:

- PPO was competitive on dynamic traffic
- PPO was weak on sustained high load
- token bucket remained a strong baseline

## 3. Quick Screening Harness

To avoid long runs on slow hardware, a fast screening pipeline was added.

Main file:

- [quick_screen.py](/Users/harshalrudra/Documents/Codex/2026-05-09/https-github-com-theharshal30-reinforcement-learning/Reinforcement-Learning-Course-Project/quick_screen.py)

Summary file:

- [screening/quick_screen_summary.md](/Users/harshalrudra/Documents/Codex/2026-05-09/https-github-com-theharshal30-reinforcement-learning/Reinforcement-Learning-Course-Project/screening/quick_screen_summary.md)

Screening setup:

- training episodes nominally `30`
- evaluation scenario: `bursty`
- evaluation episodes: `3`
- episode length: `100`
- early stopping enabled, so most runs stopped around `12-15` episodes
- device: `cpu`

### Quick Screening Results

`baseline`

- episodes run: `15`
- recent train reward: `0.254`
- eval reward: `71.278`
- throughput: `10.363`
- latency: `1.842`
- drop rate: `0.032`

Interpretation:

- reference point for the short-screen protocol

`hp_penalty_1_0`

- episodes run: `15`
- recent train reward: `0.248`
- eval reward: `58.503`
- throughput: `9.017`
- latency: `1.085`
- drop rate: `0.116`

Interpretation:

- reducing the high-priority drop penalty changed behavior
- latency improved
- drops got much worse
- overall reward got worse
- this supports the claim that reward weights materially shape policy behavior

`throughput_2_0`

- episodes run: `15`
- recent train reward: `0.875`
- eval reward: `149.003`
- throughput: `10.363`
- latency: `1.842`
- drop rate: `0.032`

Interpretation:

- strongest short-screen result
- increasing throughput reward strongly helped under this quick protocol
- this became the most promising candidate for further testing

`hidden_32`

- episodes run: `15`
- recent train reward: `0.261`
- eval reward: `71.278`

Interpretation:

- no meaningful gain over baseline in the short screen

`hidden_64`

- episodes run: `13`
- recent train reward: `0.288`
- eval reward: `71.278`

Interpretation:

- no meaningful gain over baseline in the short screen

`queue_trend`

- episodes run: `14`
- recent train reward: `0.090`
- eval reward: `71.278`

Interpretation:

- adding queue trend alone did not help in the short screen

`queue_arrival_trend`

- episodes run: `12`
- recent train reward: `-1.121`
- eval reward: `-126.663`
- throughput: `2.200`
- drop rate: `0.772`
- high-priority drop rate: `0.120`

Interpretation:

- this feature combination was clearly harmful in the quick-screen setting
- it destabilized learning badly

### Safe Conclusions From Quick Screen

- increasing throughput reward to `2.0` looked promising
- reducing high-priority penalty made the agent more willing to drop traffic
- queue trend did not show clear value
- queue + arrival trend was a bad variant

### Limits of Quick Screen

- only one evaluation scenario
- only a few evaluation episodes
- heavy use of early stopping
- results are for ranking variants, not for final claims

## 4. Full Multi-Seed Runner With Early Stopping

A larger runner was added for repeated seeded runs.

Main file:

- [full_multiseed.py](/Users/harshalrudra/Documents/Codex/2026-05-09/https-github-com-theharshal30-reinforcement-learning/Reinforcement-Learning-Course-Project/full_multiseed.py)

Saved results:

- [full_runs/aggregate_summary.md](/Users/harshalrudra/Documents/Codex/2026-05-09/https-github-com-theharshal30-reinforcement-learning/Reinforcement-Learning-Course-Project/full_runs/aggregate_summary.md)
- [full_runs/aggregate_summary.json](/Users/harshalrudra/Documents/Codex/2026-05-09/https-github-com-theharshal30-reinforcement-learning/Reinforcement-Learning-Course-Project/full_runs/aggregate_summary.json)
- [full_runs/per_seed_summary.json](/Users/harshalrudra/Documents/Codex/2026-05-09/https-github-com-theharshal30-reinforcement-learning/Reinforcement-Learning-Course-Project/full_runs/per_seed_summary.json)

Variants run:

- `baseline`
- `throughput_2_0`
- `queue_trend`

Seeds:

- `0`
- `1`
- `2`

Important run condition:

- early stopping was enabled
- most runs stopped near `50` episodes, not `260`

### Early-Stopped Multi-Seed Summary

`baseline`

- train reward mean: `0.183`
- bursty reward: `110.735`
- high_load reward: `27.340`
- oscillating reward: `137.476`

`throughput_2_0`

- train reward mean: `0.873`
- bursty reward: `248.735`
- high_load reward: `86.348`
- oscillating reward: `285.751`

`queue_trend`

- train reward mean: `0.171`
- bursty reward: `110.735`
- high_load reward: `27.340`
- oscillating reward: `137.476`

Interpretation:

- `throughput_2_0` clearly dominated in this early-stopped protocol
- `queue_trend` again failed to improve over baseline
- this reinforced the quick-screen result that throughput reward was the most promising change

Important limitation:

- because runs stopped around 50 episodes, this was still closer to screening than to full training

## 5. Later Baseline vs Throughput Compare

Another compare run was executed and saved under:

- [full_runs/aggregate_summary.md](/Users/harshalrudra/Documents/Codex/2026-05-09/https-github-com-theharshal30-reinforcement-learning/Reinforcement-Learning-Course-Project/full_runs/aggregate_summary.md)

Saved summary:

- `baseline`: train reward mean `0.766`, bursty `214.141`, high_load `168.770`, oscillating `248.035`
- `throughput_2_0`: train reward mean `0.766`, bursty `214.141`, high_load `168.770`, oscillating `248.035`

Interpretation:

- these exact matches are a red flag
- this run is not a valid ablation

Reason:

- the global default `reward_throughput_weight` was changed from `1.1` to `2.0`
- after that change, the so-called `baseline` and `throughput_2_0` runs both used the same throughput reward
- therefore they were effectively the same configuration

Conclusion:

- these files are saved correctly
- but this specific baseline-vs-throughput comparison must not be cited as evidence

## 6. Current State of the Evidence

Valid findings so far:

- PPO can learn useful behavior in this simulator
- PPO is better on bursty and oscillating traffic than on sustained high load
- reward design matters a lot
- lowering high-priority drop penalty hurts traffic protection
- increasing throughput reward to `2.0` looks promising in both quick screening and early-stopped multi-seed testing
- queue trend alone does not show a convincing benefit
- queue + arrival trend is harmful in the current implementation

Findings that are not yet rigorous enough:

- any claim that `throughput_2_0` is definitively superior after full-length training
- any claim based on the later contaminated baseline-vs-throughput compare
- any claim of statistical significance beyond rough directional evidence

## 7. Best Files to Use

For a quick narrative:

- [artifacts/REPORT.md](/Users/harshalrudra/Documents/Codex/2026-05-09/https-github-com-theharshal30-reinforcement-learning/Reinforcement-Learning-Course-Project/artifacts/REPORT.md)
- [screening/quick_screen_summary.md](/Users/harshalrudra/Documents/Codex/2026-05-09/https-github-com-theharshal30-reinforcement-learning/Reinforcement-Learning-Course-Project/screening/quick_screen_summary.md)
- [full_runs/aggregate_summary.json](/Users/harshalrudra/Documents/Codex/2026-05-09/https-github-com-theharshal30-reinforcement-learning/Reinforcement-Learning-Course-Project/full_runs/aggregate_summary.json)

For deeper inspection:

- per-seed reports in [full_runs/baseline](/Users/harshalrudra/Documents/Codex/2026-05-09/https-github-com-theharshal30-reinforcement-learning/Reinforcement-Learning-Course-Project/full_runs/baseline)
- per-seed reports in [full_runs/throughput_2_0](/Users/harshalrudra/Documents/Codex/2026-05-09/https-github-com-theharshal30-reinforcement-learning/Reinforcement-Learning-Course-Project/full_runs/throughput_2_0)
- per-seed reports in [full_runs/queue_trend](/Users/harshalrudra/Documents/Codex/2026-05-09/https-github-com-theharshal30-reinforcement-learning/Reinforcement-Learning-Course-Project/full_runs/queue_trend)

## 8. Recommended Next Step

To produce a clean final comparison:

1. restore baseline throughput reward to `1.1`
2. keep `throughput_2_0` as an explicit variant override only
3. rerun `baseline` vs `throughput_2_0`
4. disable early stopping for the final report run
5. keep `3` seeds

That will produce a valid ablation with a proper baseline.

## 9. Clean Final Ablation Results

A corrected full comparison was run after restoring the true baseline reward and making ablation variants explicit.

Saved files:

- [full_runs/aggregate_summary.md](/Users/harshalrudra/Documents/Codex/2026-05-09/https-github-com-theharshal30-reinforcement-learning/Reinforcement-Learning-Course-Project/full_runs/aggregate_summary.md)
- [full_runs/aggregate_summary.json](/Users/harshalrudra/Documents/Codex/2026-05-09/https-github-com-theharshal30-reinforcement-learning/Reinforcement-Learning-Course-Project/full_runs/aggregate_summary.json)

Variants:

- `baseline`
- `throughput_2_0`
- `hp_penalty_1_0`

Seeds:

- `0`
- `1`
- `2`

### Aggregate Reward Summary

`baseline`

- train reward mean: `-0.332`
- bursty reward: `-3.011`
- high_load reward: `-120.824`
- oscillating reward: `12.984`

Interpretation:

- the original reward design is too punitive
- the learned policy becomes overly conservative
- throughput and reward collapse, especially in high load

`throughput_2_0`

- train reward mean: `0.766`
- bursty reward: `214.141`
- high_load reward: `168.770`
- oscillating reward: `248.035`

Interpretation:

- highest total reward among the clean ablations
- very strong all-scenario performance in scalar reward terms
- higher variance than the high-priority penalty variants

`hp_penalty_1_0`

- train reward mean: `0.292`
- bursty reward: `103.772`
- high_load reward: `128.383`
- oscillating reward: `132.273`

Interpretation:

- much better than baseline
- lower total reward than `throughput_2_0`
- much more stable across seeds

### Component Metric Interpretation

For `high_load`, the saved metrics show:

`baseline`

- throughput: `8.587`
- latency: `3.004`
- drop rate: `0.411`
- high-priority success: `0.830`

`throughput_2_0`

- throughput: `11.388`
- latency: `2.703`
- drop rate: `0.214`
- high-priority success: `0.894`

`hp_penalty_1_0`

- throughput: `11.654`
- latency: `1.111`
- drop rate: `0.208`
- high-priority success: `0.898`

Interpretation:

- `throughput_2_0` does not merely open the floodgates
- it is much better than baseline on every major high-load operational metric
- `hp_penalty_1_0` is even more attractive from an operational-quality perspective
- `throughput_2_0` wins the reward objective
- `hp_penalty_1_0` wins on balanced system behavior

## 10. Always-Max Sanity Check

An `always_max` dummy policy was added. It always sets the rate limit to the environment maximum (`18`) and never adapts.

Purpose:

- test whether the best learned policy is just a trivial "always admit everything" strategy

Results:

- `low_load`: throughput `4.389`, latency `0.977`, drop `0.000`, HP success `0.469`, reward `55.806`
- `high_load`: throughput `12.000`, latency `5.840`, drop `0.153`, HP success `0.881`, reward `-176.735`
- `bursty`: throughput `10.089`, latency `1.758`, drop `0.032`, HP success `0.759`, reward `125.752`
- `oscillating`: throughput `10.667`, latency `1.583`, drop `0.004`, HP success `0.787`, reward `145.123`

Interpretation:

- `throughput_2_0` is not equivalent to always-max
- especially in `high_load`, always-max has much worse latency and much worse reward than learned `throughput_2_0`
- this strengthens the claim that the PPO policy learned nontrivial control behavior

## 11. Intermediate High-Priority Penalty Check (`1.5`)

A quick high-load screening run was executed for an intermediate reward setting:

- `reward_high_drop_weight = 1.5`

Saved file:

- [screening_hp_1_5/quick_screen_summary.md](/Users/harshalrudra/Documents/Codex/2026-05-09/https-github-com-theharshal30-reinforcement-learning/Reinforcement-Learning-Course-Project/screening_hp_1_5/quick_screen_summary.md)

Quick-screen result:

- reward: `70.236`
- throughput: `11.620`
- latency: `1.110`
- drop rate: `0.215`
- high-priority success: `0.893`

Interpretation:

- this setting produced a balanced result
- it kept throughput and high-priority success high
- latency remained low
- it looked more "operationally safe" than `throughput_2_0`

## 12. Long Run for `hp_penalty_1_5`

Because `hp_penalty_1_5` looked promising, it was trained longer:

- `500` training episodes
- `2` seeds
- no early stopping

Saved files:

- [full_runs_hp15_long/aggregate_summary.md](/Users/harshalrudra/Documents/Codex/2026-05-09/https-github-com-theharshal30-reinforcement-learning/Reinforcement-Learning-Course-Project/full_runs_hp15_long/aggregate_summary.md)
- [full_runs_hp15_long/aggregate_summary.json](/Users/harshalrudra/Documents/Codex/2026-05-09/https-github-com-theharshal30-reinforcement-learning/Reinforcement-Learning-Course-Project/full_runs_hp15_long/aggregate_summary.json)

Summary:

- train reward mean: `0.588`
- bursty reward: `103.764`
- high_load reward: `128.322`
- oscillating reward: `132.260`

Component metrics:

`low_load`

- throughput: `4.507`
- latency: `0.985`
- drop: `0.00022`
- high-priority success: `0.482`
- reward: `57.580`

`high_load`

- throughput: `11.654`
- latency: `1.111`
- drop: `0.2087`
- high-priority success: `0.8979`
- reward: `128.322`

`bursty`

- throughput: `8.871`
- latency: `1.083`
- drop: `0.1121`
- high-priority success: `0.7513`
- reward: `103.764`

`oscillating`

- throughput: `10.043`
- latency: `1.039`
- drop: `0.0731`
- high-priority success: `0.8071`
- reward: `132.260`

Interpretation:

- the longer run did not materially change the policy behavior
- the policy is extremely stable across seeds
- `hp_penalty_1_5` remains a strong "balanced operations" candidate
- it still does not beat `throughput_2_0` on scalar reward

## 13. Final Recommendation

There are now two defensible final narratives, depending on project emphasis.

### If you want the strongest reward-maximization result

Choose `throughput_2_0`.

Reason:

- highest total reward across clean ablations
- much better than baseline
- clearly learned something better than the always-max dummy controller

Best framing:

- reward design strongly shapes learned policy
- boosting throughput weight gave the best scalar objective value

### If you want the strongest systems/operations story

Choose `hp_penalty_1_5` or `hp_penalty_1_0`, with preference for `hp_penalty_1_5`.

Reason:

- better balance across throughput, latency, drop rate, and high-priority success
- very stable across seeds
- easier to defend as a realistic policy choice

Best framing:

- the original reward over-penalized high-priority drops and caused over-conservative behavior
- rebalancing that penalty produced a policy that degrades gracefully under load

### Recommended overall presentation choice

For a course project, the strongest combined story is:

1. show baseline failure under the original reward
2. show `throughput_2_0` as the reward-maximizing fix
3. show `hp_penalty_1_5` as the more operationally balanced fix
4. explain that "best reward" and "best system behavior" are not always the same

That gives you both technical depth and a strong discussion section.
