# Batch 2: where the bonus points at the start, and features that keep the fit local

Written 2026-11-12 14:45 PT, before any run of the batch. Bonus-only changes (program.md, days 1 to 5). Every idea
is tied to something batch 1's records showed.

## What batch 1 showed (PointMaze Large, reward 1 on arrival, 16 development seeds)

Numbers are from `batches/01_.../summary.md` as generated 2026-11-12 14:29 PT (9 of 14 units) and from the control's
maze-grid records.

1. **RND starts with a bonus that rises with the distance from the start.** Before any training the control's bonus
   is 0.6 at the start cell and 56 at the far cells (largest over smallest: 88). The input is whitened with the
   statistics of the random-action rollouts around the start, so a far cell is a large input, and an untrained
   predictor's error grows with the size of its input. Beyond the input clip of ±5 (about 4.5 cells from the start)
   every cell has the same bonus.
2. **Without that slope fewer policies find the goal.** The even start (same bonus everywhere at the start) reaches
   the goal in 9 of 16 policies and has a whole-curve mean of 0.186 against the control's 0.885.
3. **The running divisor is needed.** Freezing it (0.547) or removing it (0.327) is far behind.
4. **A fit that is not local carries no information.** The scheduled last layer on the random body's features has a
   whole-curve mean of 0.000: after the first 32,768 steps its bonus is 0.02 at visited cells and 0.03 at cells never
   visited (start: 1 everywhere), and later the bonus rises to between 0.1 and 80 as the input statistics move.
5. **The predictor's step size makes no difference that 16 seeds can show** (0.887, 0.881 and 0.878 at 0.25, 1 and
   4 times 1e-4, against 0.885), though the slowest has the earliest arrivals at weight 2 (94 thousand steps against
   192 thousand).
6. **Half of the control's missing whole-curve mean (0.058 of 0.115) is in the first 524 thousand steps,** before
   and around the first arrivals; 0.039 is in the rise to 0.94 at 2.1 million steps; 0.018 is after that.
7. **The intrinsic reward the learner sees rises again late in the control** (added 2026-11-12 14:50 PT, from the
   summary's second property table): intrinsic weight times the reward at the probes is 0.0156 after the first
   window, 0.0025 at a quarter of the budget and 0.0045 at the end. The raw bonus stops falling once the learning
   rate has been annealed, while the divisor (a running standard deviation over the whole history) keeps falling.
   Early, the falling divisor is what keeps the reward up until the goal is found: with the divisor frozen from the
   start (batch 1's fixed scale) only 13 of 16 policies find the goal.
8. **A bonus that falls inside the rollout finds the goal much sooner but ends lower** (added 14:50 PT): the fast
   last layer with step 0.1 has its median first arrival at 61 thousand steps at weight 1 and 55 thousand at weight
   2 (control: 353 and 192 thousand), with final levels of 0.96 and 0.75 (control: 0.997 and 0.995); its best
   whole-curve mean is 0.862 ± 0.016 at weight 0.5.

## Screen

As in batch 1: one unit per row, PointMaze Large with reward 1 on arrival at 8,388,608 steps, seeds 0 to 15, learning
rate 2e-3, entropy 0.01, clip 0.1, the 12 intrinsic weights of batch 1, the maze-grid record on in every unit. The
control is trained again in this batch, on this batch's code.

**Validation block (new, for ideas near the control).** Sixteen seeds resolve only differences above about 0.03.
An idea whose interval of the difference in whole-curve mean contains zero or lies above it is trained at its best
weight on 256 further seeds (10000 to 10255) beside the control at its best weight; that resolves about 0.007. The
fresh seeds from 30000 stay unused until a finalist is confirmed.

## The control and the ideas

Define $x$ as the bonus's input after whitening, $\ell$ as a length in whitened units (1 unit is about 0.9 m, a
little under one maze cell, by the statistics of the random-action rollouts), and $D$ as a number of features.
**Local features** are random Fourier features of the input whitened with the statistics of the random-action
rollouts (kept fixed afterwards, no clip): $\phi(x) = \sqrt{2/D}\,\cos(\Omega x + \beta)$ with the rows of $\Omega$
drawn from a normal distribution with standard deviation $1/\ell$ and $\beta$ uniform on $[0, 2\pi)$, drawn once per
seed. Two inputs at distance $d$ then have features with inner product about $\exp(-d^2 / (2\ell^2))$: inputs more
than about $2\ell$ apart share almost nothing.

| name | change against the control | own setting |
|---|---|---|
| control | CleanRL RND as in the baseline | none |
| wider input clip | the whitened input is clipped at ±c instead of ±5, so the starting bonus keeps rising across the whole maze | c = 10, 20, no clip |
| input statistics kept from the random-action rollouts | the bonus's input is whitened with the statistics of the random-action rollouts for the whole run, so a place far from the start stays a large input until the predictor has been trained there | clip: none, ±20 |
| slower predictor | the predictor has its own Adam at a constant rate below batch 1's slowest | 0.0625 and 0.125 times 1e-4 |
| local features as the input of RND | the target and the predictor read $\phi(x)$ ($D$ = 256) instead of $x$; trained and divided as in the control | $\ell$ = 0.5, 1, 2 |
| scheduled last layer on local features | no trained network: the output layer on $\phi(x)$ ($D$ = 512) is fitted in closed form step by step so that each visit moves the visited input's squared error one place along $(1+m)^{-a}$; no divisor | $\ell$ = 0.5, 1, 2 at $a$ = 0.5; $a$ = 1 at $\ell$ = 1 |
| divisor held after a part of the run | the divisor follows the running standard deviation as in the control until a fraction $f$ of the run's planned steps, and keeps the value it has then for the rest of the run, so the reward the learner sees can only fall with the bonus afterwards | $f$ = 1/16, 1/8, 1/4 |

That is 18 units: 1 + 3 + 2 + 2 + 3 + 4 + 3.

## Predictions (written before the runs)

1. **Wider input clip:** the largest over the smallest starting bonus rises above 300 at c = 10 and above 1,000 with
   no clip. The median first arrival at the best weight is at least 15% earlier than the control's at c = 20 or with
   no clip. The whole-curve mean moves by less than 0.02 either way.
2. **Input statistics kept from the random-action rollouts:** the bonus at cells never visited stays more than 10
   times the bonus at visited cells through the first quarter of the budget. Either the whole-curve mean is 0.01 to
   0.03 ahead with earlier arrivals, or it is behind because the predictor's very large errors at far inputs take
   over the shared gradient clip; the records will show which (the divisor after the first window, and the final
   level).
3. **Slower predictor:** arrivals at weight 2 are earlier than the control's; the whole-curve mean is within the
   interval.
4. **Local features as the input of RND** separates the two readings of the even start's failure. Its starting
   bonus is nearly even (largest over smallest below 3). If all 16 policies reach the goal and the whole-curve mean
   is above 0.8, the even start failed because fitting visited inputs lowered the bonus elsewhere, not because the
   slope was missing. If it fails like the even start (12 of 16 or fewer), the slope is what RND needs here.
5. **Scheduled last layer on local features:** at $\ell$ = 0.5 and 1 the bonus at cells never visited stays above
   0.9 until they are visited, and the decay slope is within 0.1 of $-a$. All 16 policies reach the goal at some
   weight. Its whole-curve mean is the highest of the batch if a bonus that only counts is enough; if it is behind
   the control with every policy reaching the goal late, the missing slope costs time at the start.
6. **Divisor held after a part of the run** (written 2026-11-12 14:50 PT): intrinsic weight times the reward at the
   probes at the end of the run is below its value at a quarter of the budget (in the control it is 1.8 times
   higher). The final level at weights 4 and 16 is higher than the control's at the same weights. At the best
   weight the whole-curve mean is 0.005 to 0.015 ahead (a difference the validation block must resolve); $f$ = 1/16
   (524 thousand steps, about when the first arrivals happen) may be too early and lose arrivals.

