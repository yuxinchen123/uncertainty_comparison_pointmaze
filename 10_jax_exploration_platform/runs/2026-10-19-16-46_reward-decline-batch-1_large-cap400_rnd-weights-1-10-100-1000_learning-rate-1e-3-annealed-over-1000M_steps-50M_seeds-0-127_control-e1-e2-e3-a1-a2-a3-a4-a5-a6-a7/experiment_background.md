# Reward-decline study, batch 1: the control and ten ideas

## 1. Purpose

Test ten candidate fixes for the rise and fall of extrinsic reward under RND training, each as one
change against a shared control, on the setting the map screening chose: the large maze at
learning rate 1e-3 for 50M steps (every small map failed the screening's first rule, because there
a policy with no bonus reaches the goal in 44-64% of copies).

| id | unit | the one change against the control |
|---|---|---|
| control | `control` | none |
| E1 | `e1_terminate-at-goal` | the episode ends on the first step inside the goal radius (+1 once) |
| E2 | `e2_terminate-at-goal_reward-shift-minus-1` | ends at the goal; reward 0 there and -1 on every other step |
| E3 | `e3_terminate-at-goal_goal-reward-remaining-steps` | ends at the goal and pays cap - step + 1 at once |
| A1 | `a1_intrinsic-weight-linear-decay` | intrinsic weight x (1 - (k - 1)/N) at iteration k of N |
| A2 | `a2_intrinsic-weight-goal-rate-gate-0.05` | intrinsic weight x (1 - running goal-reaching rate), step 0.05 |
| A3 | `a3_clipped-ppo_epochs-4_minibatches-4` | clipped PPO: 4 epochs x 4 minibatches, clip 0.2 |
| A4 | `a4_entropy-coefficient-0.01` | entropy bonus 0.01 |
| A5 | `a5_normalize-extrinsic-reward` | task reward / max(1, running std of its discounted sum) |
| A6 | `a6_extrinsic-discount-0.99` | extrinsic discount 0.99 instead of 0.999 |
| A7 | `a7_gradient-clip-per-network` | actor, critic and predictor gradients clipped separately |

"Fixed" (decided with the user 2026-10-19): the final-10% mean episode return is at least 90% of the
idea's own peak, and the idea still learns (algorithm ideas: final above the control's peak minus one
standard error; environment ideas: final share of episodes reaching the goal at least the control's
peak share).

## 2. Key hyperparameters

- Large PointMaze, start (7,1), goal (1,10), continuing task with a 400-step cap unless the idea
  ends episodes at the goal; no reset noise.
- RND, intrinsic weights 1, 10, 100, 1000 packed into one 512-copy program per unit (128 paired
  copies each, seeds 0-127; every unit has the same copy count, so copy k meets the same policy
  noise and environments in every idea).
- Learning rate 1e-3, annealed linearly over 1000M steps (1,953,125 iterations), so over these 50M
  steps it stays above 97.5% of 1e-3, as in the 1000M-step runs where the fall was first seen.
  Annealing over the run's own 50M would hide part of the fall (the corrected 100M run climbed back
  while its rate went to zero).
- Full-batch updates (one gradient step per 128-step x 4-environment rollout) except A3; extrinsic
  discount 0.999 and intrinsic 0.99 except A6; extrinsic weight 2; gradient-norm clip 0.5.
- 50M steps per copy (97,657 iterations); records every 250 iterations with `--diagnostics` and
  coverage.

## 3. Code and config changes

The study's switches (all off by default; tested in `tests/agents/test_reward_decline_switches.py`):
`--terminate-at-goal`, `--reward-shift`, `--goal-reward`, `--intrinsic-weight-schedule`,
`--success-gate-rate`, `--update-style epoch_minibatch`, `--ent-coef`,
`--normalize-extrinsic-reward`, `--gamma-ext`, `--clip-per-network`, `--lr-anneal-iterations`.
Queue built by `code/build_queue.py`; workers `code/worker.sh` (shared compilation cache).
Summaries by `scripts/summarize_episode_records.py`; a terminating task with the -1 shift is read
on the lump-sum scale (return + cap).

## 4. Git state

Recorded at submission below.
