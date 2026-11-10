# Batch 1: diagnostics, even-start parts, decay parts, and the first combinations

Written 2026-11-09 20:55 PT, before any RL unit of the batch had finished. Times are Pacific. Exact settings:
`rl_ideas.json` (RL screen) and `bench_ideas_all.json` (bench). Every idea keeps CleanRL's reward normalization on.

## The control

Online tabular-RND as CleanRL trains it: the predictor shares the policy's Adam (learning rate annealed, joint clip
0.5), the loss is the mean squared feature error, and the bonus is half the summed squared error. It runs at policy
learning rates 1e-3 (as every idea) and 1e-4 (8 extra units, added 20:47 PT, so the control is not weak where the
baseline prefers 1e-4).

## The ideas

Each row is one change against the control. The bench columns come from the first bench (8 field seeds, 4,096
iterations of the RL update shape): the start is $\max_s b_0(s)/\min_s b_0(s)$, and the slopes are the median per-cell
slope of $\log(b_t/b_0)$ on $\log N_t$ under uniform visits, straight / snake grid.

| idea | part of the method | bench: start | bench: slope, straight / snake | prediction for RL |
|---|---|---|---|---|
| control | none | 3.65 / 4.23 | −3.11 / −1.52 | the level to beat |
| separate Adam, constant learning rate 1e-4 | diagnostic: the optimizer sets the decay | 3.65 / 4.23 | −1.21 / −0.71 | better than the control where the control decays fastest |
| length readout ‖g−f‖ | diagnostic: the readout halves the exponent | 1.91 / 2.06 | −1.55 / −0.76 | better than the control |
| initial-ratio readout | even start only | 1.00 | −3.11 / −1.52 | little gain: the decay is unchanged |
| unit start (zero predictor output, unit-length target) | even start only | 1.00 | −5.03 / −5.06 | little gain or a loss: the decay is faster |
| random bias start, standard deviation 0.5 | even start by the random draw | 1.04 | −1.23 / −1.09 | some gain |
| wide one-hot input layer (64) | even start by the random draw | 2.63 / 3.04 | −3.36 / −2.22 | no gain |
| power-law loss q = 4, plain SGD 0.1, length readout | decay only | 1.91 / 2.06 | −0.22 / −0.17 | gain from the slower, per-cell decay |
| shrink-target loss, plain SGD 1, length readout | decay only | 1.91 / 2.06 | −0.30 / −0.22 | gain, as above |
| unit start + AdaGrad 0.1 | even start with a generic optimizer | 1.00 | −0.49 / −0.28 | gain, less than the next rows: its cells do not follow one law (late spread 0.8-1.6) |
| unit start + power-law q = 4, SGD 0.3, length readout | both | 1.00 | −0.46 / −0.42 | the largest gain |
| unit start + shrink target, SGD 2, length readout | both | 1.00 | −0.49 / −0.46 | the largest gain |
| unit start + shrink target, SGD 3, length readout | both | 1.00 | −0.51 / −0.48 | the largest gain |

Ideas tried on the bench and not sent to RL:
- **Power-law q = 6 with the squared readout:** slope −0.15 at learning rate 1e-2, and it diverges at 1e-1.
- **SGD with a 1/t learning rate:** slope about −0.2.
- **AdaGrad without the unit start:** −3 to −5, or −0.2.
- **Only the input rows training** (frozen trunk, or a copy of the target's trunk): slope about −0.1; the rows cannot
  fit the targets through a frozen trunk.
- **Larger learning rates:** the shrink target diverges at 5 and the power-law loss at 0.5 to 1.

## What the batch can decide

1. Whether the even-start parts alone help (rows 4 to 7 against the control).
2. Whether the decay parts alone help (rows 2, 3, 8 and 9).
3. Whether both together help more than either part (the last three rows against the rest). This is the user's
   hypothesis.

The three-law bench adds a third visitation law, in which the second half of the cells is never visited. It measures
how much of its starting bonus a never-visited cell keeps. Its results go into `STATE.md`.
