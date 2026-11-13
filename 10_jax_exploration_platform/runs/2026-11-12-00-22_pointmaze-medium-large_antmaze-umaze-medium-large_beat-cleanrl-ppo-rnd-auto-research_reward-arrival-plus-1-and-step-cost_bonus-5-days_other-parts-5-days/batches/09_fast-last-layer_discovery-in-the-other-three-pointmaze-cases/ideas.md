# Batch 9: the fast last layer for discovery in the other three PointMaze cases

Written 2026-11-13 01:07 PT, before any run of the batch. Bonus-only (program.md, days 1 to 5); the option exists
in the frozen code of batch 2 (commit d7054945). No new idea: the batch asks whether the discovery result of
PointMaze Large with reward 1 on arrival holds in the other cases. The measure is the secondary metric, the mean
capped first-goal time, for which every method and RND use their own configurations.

## What the earlier runs showed

1. **PointMaze Large, reward 1 on arrival, 256 validation seeds:** the fast last layer inside the rollout (step
   0.1) first reaches the goal after 55.8 ± 2.2 thousand steps at its best of three configurations (learning rate
   6e-3, entropy 0.03, weight 16), against 119 ± 6 for RND's best of three: −63 [−75, −52].
2. **RND's own best configurations for this measure**, confirmed on 1,024 fresh seeds: PointMaze Medium with reward
   1 on arrival 17.4 ± 0.5 thousand steps (learning rate 2e-2, entropy 0.01, weight 4); Medium with the step cost
   14.1 ± 0.3 (1e-2, 0.01, weight 4); Large with the step cost 131 ± 3 (4.5e-3, 0.01, weight 8).
3. On the two Medium cases the goal is first reached within about 1.5% of the run, so there is little room: RND's
   best is already 14 to 17 thousand steps.

## What is run (first validation block, phase `discovery-validation`, 256 seeds)

| case | fast last layer (step 0.1) | CleanRL PPO-RND |
|---|---|---|
| PointMaze Medium, reward 1 on arrival | learning rates 1e-2 and 2e-2, entropy 0.01 and 0.03, weights 4 and 16 (8 configurations) | (2e-2, 0.01, 4), (1e-2, 0.01, 4), (2e-2, 0.1, 4) |
| PointMaze Medium, step cost | the same 8 configurations | (1e-2, 0.01, 4), (2e-2, 0.01, 4), (1e-2, 0.03, 4) |
| PointMaze Large, step cost | (6e-3, 0.03, 16), (4.5e-3, 0.01, 8), (2e-3, 0.03, 16) | (4.5e-3, 0.01, 8), (6e-3, 0.01, 8) |

Each triple is (learning rate, entropy coefficient, intrinsic weight). 27 units. The fast last layer's best in a
case is the best of 8 or 3 on these seeds, so a difference counts only after a second block.

## Predictions

1. **PointMaze Large, step cost:** the fast last layer is ahead of RND by at least 40 thousand steps with its
   interval below zero, as with reward 1 on arrival, because the reward setting does not matter before the first
   arrival except through the value network's start.
2. **The two PointMaze Medium cases:** it is ahead of RND by a few thousand steps at most; the interval is below
   zero in at least one of the two.
