# Twenty runs at once, four seeds of every arm: what the records say

2026-08-06. Jobs 6534052–6534071, 13,107,200 steps each (800 policy updates, 4 logged rows at the
production 200-update cadence), one GPU each across lotus, cheetah08, cheetah09 and cheetah02.

**All twenty finished with `completed=true` at 100% of target. No failures, no resubmissions.**

## Throughput

| node | GPU | runs on the node | steps per second per run |
|---|---|---|---|
| lotus | Quadro RTX 6000 | 8 | 3,321–3,454 |
| cheetah02 | RTX 4000 Ada | 4 | 3,716–3,769 |
| cheetah08 | RTX A4000 | 4 | 3,048–3,128 |
| cheetah09 | RTX A4000 | 4 | 3,056–3,091 |

Twenty runs at once held about 3,300 steps/s each, against 4,765 for a single run alone on lotus.
The whole wave took 71 minutes, which is why the original one-hour walltime was wrong.

## The arms ran the configuration they were meant to

Every one of the four seeds in every arm agrees, and the invariants hold on every logged row:

| arm | `max_grad_norm` | `joint_grad_clip` | `rnd_max_grad_norm` |
|---|---|---|---|
| arm1_original | 0.5 | True | joint |
| arm2_no_rnd_grad_clip | 0.5 | False | 0.0 |
| arm3_update_proportion_1 | 0.5 | True | joint |
| arm4_shallower_predictor | 0.5 | True | joint |
| arm5_all | 0.5 | False | 0.0 |

Checked on every row of every run: in the joint arms the predictor's clip rate equals the policy's
exactly; in arms 2 and 5 the predictor's clip rate is exactly zero and the scale applied to it is
exactly one. Zero non-finite gradient norms across all twenty runs.

## The finding that matters for the campaign

| arm | joint norm | policy norm | predictor norm | predictor share of squared norm | clip fires | scale applied to the predictor |
|---|---|---|---|---|---|---|
| arm1_original | 1.5441 | 1.5438 | 0.0222 | 0.023% | 0.959 | **0.4914** |
| arm2_no_rnd_grad_clip | 1.2302 | 1.2299 | 0.0230 | 0.030% | 0.919 | **1.0000** |
| arm3_update_proportion_1 | 1.3048 | 1.3047 | 0.0143 | 0.010% | 0.969 | **0.5060** |
| arm4_shallower_predictor | 1.3396 | 1.3395 | 0.0167 | 0.016% | 0.964 | **0.4999** |
| arm5_all | 1.1907 | 1.1906 | 0.0105 | 0.006% | 0.974 | **1.0000** |

Read it in three steps.

1. **The predictor is a rounding error in the joint norm.** It contributes between 0.006% and 0.030%
   of the joint squared norm — its gradient is about one seventieth of the policy's. So the joint
   norm is the policy's norm, to four significant figures, and the clip decision is made entirely by
   the policy.
2. **The clip fires almost every step.** The joint norm sits around 1.2 to 1.5 against a threshold of
   0.5, so 92% to 97% of optimizer steps are clipped.
3. **Therefore the predictor's gradient is halved on almost every step, for reasons that have nothing
   to do with the predictor.** The mean scale applied to it is 0.49 to 0.51 in the three joint arms.

That is the effect arms 2 and 5 remove, and it is a substantial one: the RND predictor's effective
learning rate in those arms is about twice what it is in CleanRL as published. It is also not
something the earlier smoke run showed — at 655,360 steps the mean scale was 0.986, because the
gradient norms had not yet grown. Any conclusion drawn from a run of a few hundred thousand steps
would have been wrong about this.

## Reward, at 0.66% of the campaign's step budget

| arm | episodes | episode length | mean extrinsic (last 200 episodes) | mean intrinsic | seed-to-seed spread of intrinsic |
|---|---|---|---|---|---|
| arm1_original | 25,229 | 561 | 0.00 | 8.680 | 5.42 |
| arm2_no_rnd_grad_clip | 30,908 | 1,285 | 0.50 | 8.789 | 1.94 |
| arm3_update_proportion_1 | 30,230 | 1,136 | 6.25 | 9.262 | 2.41 |
| arm4_shallower_predictor | 27,519 | 349 | 0.00 | 6.953 | 2.40 |
| arm5_all | 33,970 | 453 | 0.00 | 7.302 | 2.32 |

**No conclusion should be drawn from the extrinsic column.** 13.1M steps is 0.66% of the campaign's
2e9, four seeds is not enough to separate anything, and Montezuma's Revenge gives its first reward
only after a long scripted sequence — arm 3's 6.25 comes from one seed finding one key. The column is
here to show the logging works end to end, not to rank the arms.

The intrinsic column is more informative this early: the two arms with the shallower predictor
(4 and 5) produce a visibly lower intrinsic reward than the three with CleanRL's depth, which is what
one would expect from a predictor that can fit its target more easily.

Losses are healthy in every arm — value loss below 0.002, policy loss near zero, approximate KL
around 0.0007, forward loss 0.027 to 0.036. Nothing diverged.

## Record integrity

All twenty records: update rows strictly increasing with no duplicates, train and eval history the
same length, and the episode sidecar's row count equal to `episodes_kept` in every one. The
200-episode trailing window was full (`n_episodes_averaged` = 200) in every final row.

## What was fixed to get here

Two problems, both found by submitting rather than by reading:

1. **The walltime was an hour, and the wave needed 71 minutes.** Seventeen of the twenty would have
   been killed at 75%. Raised to three hours. The lesson generalises: throughput per run at eight
   runs per node is about 70% of a single run's, and the campaign's sizing has to use the packed
   figure.
2. **Slurm's pre-walltime signal would never have reached the trainer.** Slurm signals the batch
   shell, and bash does not forward a signal to a foreground child, so the trainer would have been
   killed outright rather than checkpointing. The job script now runs the trainer in the background
   under a trap that forwards SIGTERM. The 150-run campaign depends on exactly this path, since every
   run will be interrupted several times over 2e9 steps.
