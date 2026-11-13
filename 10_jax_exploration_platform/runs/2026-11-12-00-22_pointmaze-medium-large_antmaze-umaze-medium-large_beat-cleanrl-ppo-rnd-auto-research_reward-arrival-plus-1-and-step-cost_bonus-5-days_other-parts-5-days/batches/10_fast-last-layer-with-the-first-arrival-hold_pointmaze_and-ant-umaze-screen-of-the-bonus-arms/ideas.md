# Batch 10: the fast last layer together with the first-arrival hold on PointMaze, and the first AntMaze U-maze screen of the bonus arms

Written 2026-11-13 02:57 PT, before any run of the batch. Bonus-only changes (program.md, days 1 to 5). Every option exists in
the frozen code of commit bed45842, so there are no code changes.

## What the earlier runs showed

1. **Two arms lead, on different metrics and at different configurations** (256 validation seeds,
   `analysis/standing.md`):
   - the divisor held from each policy's first arrival is ahead of CleanRL PPO-RND on the whole-curve mean in the
     three PointMaze cases read so far (about +0.009 on the two cases with reward 1 on arrival, +0.0065 on Medium
     with the step cost), at intrinsic weights 1 to 4 and entropy 0.01;
   - the fast last layer inside the rollout (step 0.1) reaches the goal sooner in three PointMaze cases (−63 and
     −61 thousand steps on the two Large cases, −5.0 on Medium with reward 1 on arrival), at weight 16 and entropy
     0.03 mostly.
2. **At the configurations that find the goal soonest, nobody keeps the goal.** On PointMaze Large with reward 1
   on arrival the whole-curve mean is 0.001 for the fast last layer (learning rate 6e-3, entropy 0.03, weight 16),
   0.005 for RND's own best configuration for discovery, and 0.001 for the three-part arm, whose divisor was held
   after 1/8 of the run.
3. **The fast last layer also costs return at a moderate weight.** PointMaze Medium with reward 1 on arrival,
   learning rate 1e-2, entropy 0.01, weight 4: whole-curve mean 0.379 and final level 0.51 for the fast last layer,
   0.872 and 0.95 for RND at the same configuration, while the first-goal times are 13.8 and 17.6 thousand steps.
4. **On the ant U-maze RND finds the goal and does not keep it** (16 development seeds, 16,777,216 steps, the
   rollout-shape check of the AntMaze baseline folder): every policy arrives, after 0.3 to 2.6 million steps, yet
   the whole-curve mean is 0.04 to 0.05 and the final level 0.05 to 0.065, nearly the same for intrinsic weights
   from 1 to 64.

## Arms

| arm | change against CleanRL PPO-RND | where it runs |
|---|---|---|
| fast last layer with the first-arrival hold | last layer of the predictor refreshed at every step of the rollout (step 0.1); divisor held from each policy's first arrival | four PointMaze cases, first validation block (seeds 10000 to 10255), 2 configurations per case |
| divisor held from the first arrival | divisor held from each policy's first arrival | ant U-maze, 16 development seeds |
| fast last layer | last layer refreshed at every step of the rollout (step 0.1) | ant U-maze, 16 development seeds |
| fast last layer with the first-arrival hold | both | ant U-maze, 16 development seeds |

**PointMaze configurations** (learning rate, entropy, intrinsic weight). In each case the first one is paired with
the first-arrival hold alone on the same seeds, and the second with the fast last layer alone on the same seeds:

1. Medium, reward 1 on arrival: (6e-3, 0.01, 4) and (1e-2, 0.01, 4).
2. Medium, step cost: (6e-3, 0.01, 2) and (1e-2, 0.01, 4).
3. Large, reward 1 on arrival: (2e-3, 0.01, 2) and (2e-3, 0.01, 16).
4. Large, step cost: (2e-3, 0.01, 8) and (4.5e-3, 0.01, 8).

**Ant U-maze configurations:** the control is the RND unit of the rollout-shape check at the same configuration
and the same seeds 0 to 15; 16,777,216 steps per policy; entropy 0.001; both environment counts that tie in the
check.

1. 128 environments per policy: learning rates 1e-2 and 3e-3, intrinsic weights 1, 4, 16, 64.
2. 16 environments per policy: learning rates 3e-4 and 1e-2, intrinsic weights 0.0625, 1, 4, 16.

The ant units run in the AntMaze baseline folder under the phase `bonus-arms-screen`, because that folder holds
the ant's server settings (the interpreter with MuJoCo Warp, the job shape); this batch folder holds their
description and their summary.

## Predictions

1. **PointMaze, discovery unchanged by the hold.** At the same configuration and seeds the combined arm's
   first-goal time equals the fast last layer's alone within its interval, because the two are the same program
   until a policy first arrives (13.8, 11.7, 77.7 and 76.4 thousand steps at the four second configurations).
2. **PointMaze, the hold gives back part of what the fast last layer costs.** At the four second configurations
   the whole-curve mean is above the fast last layer's alone with the interval above zero (0.379 on Medium with
   reward 1 on arrival, −0.586 on Medium with the step cost, 0.036 on Large with reward 1 on arrival), and still
   behind RND's best in the two Medium cases.
3. **PointMaze, at the hold's own best configurations.** Adding the fast last layer lowers the first-goal time
   against the hold alone on both Large cases (interval below zero) and does not raise the whole-curve mean: level
   on Large (within 0.005), behind by more than 0.01 on Medium with reward 1 on arrival, as the three-part arm
   was (−0.029). So no configuration of this batch is ahead of RND's best on both metrics in a Medium case.
4. **Ant, the first-arrival hold.** Its whole-curve mean is above RND's at the same configuration in at least 6
   of the 8 configurations at each environment count, its best is at least 0.06 (RND's best: 0.0487), and its
   final level rises more than its whole-curve mean (at least 0.09 against 0.065). Reason: RND's divisor keeps the
   intrinsic reward at the same scale for the whole run, which pays the ant for throwing itself about; held, the
   bonus falls as the predictor learns. I am least sure of this one: the low return may come from the learner
   needing more steps, whatever the bonus does.
5. **Ant, the fast last layer.** Its first-goal time is below RND's at the same configuration in at least 6 of 8
   configurations at each count, and its whole-curve mean is not ahead.
6. **Ant, both together.** First-goal time as the fast last layer's; whole-curve mean between the two single arms.
7. **Ant, the physics guard.** Respawns by the fault guard stay as rare as in the check (32 and 43 in about 13
   billion steps per environment count).

## What decides the next step

- An arm ahead of RND on the ant by more than twice its standard error at its best configuration goes to the
  AntMaze development sweep with RND, once the environments per policy and the budget are fixed.
- If prediction 3 holds, the next code change is the fast last layer that stops at a policy's first arrival, so
  that the goal is found as soon as with the fast last layer and kept as with the hold.
