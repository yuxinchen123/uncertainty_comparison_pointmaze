# Batch 8: the divisor held from the policy's own first arrival, and the error length with a held divisor

Written 2026-11-13 00:27 PT, before any run of the batch. Bonus-only changes (program.md, days 1 to 5).

## What the earlier runs and the literature note showed

1. **The held divisor alone is ahead of CleanRL PPO-RND on the whole-curve mean in all four PointMaze cases**
   (256 validation seeds; `analysis/standing.md`): +0.0109 and +0.0058 on the two Medium cases (second block),
   +0.0095 on both Large cases (first block).
2. **It is close to public practice** (`literature/novelty.md`): the RND module of the RSL-RL library stops
   updating its reward normaliser after a fixed number of samples, and Schwarke et al. (Conference on Robot
   Learning 2023) report that RND's running normalisation keeps the intrinsic reward from falling. No paper
   evaluates the hold.
3. **The moment of the hold is a number to set** (1/16 of the run). The first arrival comes at very different
   times: about 15 thousand steps on PointMaze Medium (1.4% of its run), about 270 thousand on PointMaze Large
   (3.2%), and 0.3 to 2 million on the ant U-maze. A fixed fraction is late on Medium (66 thousand steps) and
   means something else for every budget.
4. **The error length as the bonus** (the square root of the squared error) reached the goal sooner than RND in
   batch 1 (88 against 190 thousand steps on PointMaze Large, 16 development seeds) and was not ahead on the
   whole-curve mean. It was never run with a held divisor.

## Arms

| arm | change against CleanRL PPO-RND | code |
|---|---|---|
| divisor held from the first arrival | each policy's divisor is updated as in RND until that policy first reaches the goal, then held; nothing to set | new option, being added and tested; runs when its tests pass |
| error length with held divisor | the bonus is the error length; divisor held after 1/16 of the run | existing options (commit d7054945); runs now |

Both are judged on the first validation block against RND's best configuration there and against the held
divisor alone (after 1/16), in the two PointMaze cases with reward 1 on arrival first.

A note on scope: the first arm uses one signal from the task, the policy's first arrival, to decide how the bonus
is normalised. I count that as a change of the bonus (its normalisation), not of how the bonus enters PPO.

## Predictions

1. **Divisor held from the first arrival:** on PointMaze Large with reward 1 on arrival it is level with the held
   divisor alone (within 0.005 on the whole-curve mean) and ahead of RND with its interval above zero. On
   PointMaze Medium with reward 1 on arrival it is ahead of the held divisor after 1/16, because its hold comes
   at about 15 thousand steps where the fixed one comes at 66 thousand.
2. **Error length with held divisor:** it reaches the goal sooner than the held divisor alone in both cases
   (first-goal time lower with its interval below zero) and is level with it on the whole-curve mean.
