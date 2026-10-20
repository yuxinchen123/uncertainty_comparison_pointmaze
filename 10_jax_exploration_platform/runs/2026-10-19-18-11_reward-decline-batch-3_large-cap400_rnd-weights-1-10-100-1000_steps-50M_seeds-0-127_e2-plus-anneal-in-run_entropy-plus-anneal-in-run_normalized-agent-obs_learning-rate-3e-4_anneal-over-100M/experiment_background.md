# Reward-decline study, batch 3: annealing within the run, combinations, and the agent's input scale

## 1. Purpose

Batch 1 found that annealing the learning rate to zero within the run (A8) keeps the peak and ends
twice as high as the control's peak, that ending the episode at the goal (E1, E2) holds but learns
less, that an entropy bonus (A4) reaches the highest level but falls, and that the falls come with a
slowing ball (speed 5 -> 2-3 m/s), not with the shrinking log standard deviation that every run
shows. Batch 3 combines the holders with the high-level ideas and probes the learning rate and the
agent's input scale.

| id | unit | change against batch 1's control |
|---|---|---|
| C1 | `c1_terminate-at-goal_reward-shift-minus-1_anneal-in-run` | E2 with the rate annealed over the run's own 50M steps (A8) |
| C2 | `c2_entropy-coefficient-0.01_anneal-in-run` | A4 with A8 |
| C3 | `c3_normalized-agent-observation` | actor and critic see the observation whitened by running per-copy statistics (clipped to +-5) |
| C4 | `c4_learning-rate-3e-4` | learning rate 3e-4 instead of 1e-3, same 1000M anneal |
| C5 | `c5_anneal-over-100M` | the rate anneals over 100M steps (from 1e-3 to 5e-4 within these 50M) |

The control is batch 1's: 512 copies, seeds 0-127, environments and policy noise paired.

## 2. Key hyperparameters

As batch 1 except the change per idea: large PointMaze, cap 400; RND with intrinsic weights 1, 10,
100, 1000 in one 512-copy program; learning rate 1e-3 annealed over 1000M steps; full-batch updates;
50M steps per copy; records every 250 iterations with diagnostics and coverage.

## 3. Code and config changes

New switch `--normalize-agent-obs` (agents/ppo/config.py `normalize_agent_obs`; train_step.py feeds
the actor and the critic `whiten(obs, agent obs_rms)`, the statistics updated after each rollout).
C1, C2 and C5 override `--lr-anneal-iterations`; C4 overrides `--parameter-settings`. Workers from
both accounts.

## 4. Git state

Recorded at submission below.
