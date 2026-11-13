# State of the beat-RND auto research

Times are Pacific. Read `program.md` first. This file is the log: what was done, what was found, what is next.

## Log

- 2026-11-12 00:17 PT: plan approved by the user (2026-11-11 23:5x PT). Rules written and pushed: profiling goes
  before evaluation on serval05 and serval02 card 1; configurations on PointMaze and AntMaze are chosen by the
  whole-curve mean. Speed-study folder created:
  `benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_...`. Monitoring cron every 20
  minutes (session cron `4d4fe516`, expires 2026-11-19; re-create it before then).
- 2026-11-12 00:25 PT: three pieces started in parallel: (1) the trainer changes of plan section 3.1 with their
  tests; (2) the time breakdown of the ant's physics step on an H100 NVL; (3) the copies-per-card skill and its
  first rough table from the existing survey. serval05's card is in use by another account, so profiles use
  serval02 card 1.
- 2026-11-12 00:43 PT: the environment copy `/p/rlprojects/RND/.venvs/platform_jax_warp` (platform_jax plus
  `warp-lang` 1.14.0) is built (job 6873361, 13 minutes) and listed in `.venvs/ENVS.md`. The trainer's test suites
  as they were before any change: 74 tests pass; two fail, both functions named `main` that pytest collects by
  mistake (`tests/agents/test_forward_fixture.py`, `tests/agents/test_selected_settings_blocks.py`). The ant physics
  time breakdown is running on serval02 card 1 since 00:39 PT.
- 2026-11-12 01:00 PT: the skill `/p/rlprojects/.claude/skills/estimate-copies-per-gpu/` is written and pushed
  (29 card models; `capacity.py` with 24 passing tests; the platform's `device/series.py`, not run yet). First
  rough table, from the existing survey alone: 2,048 copies on the H100 NVL and the 80 GB A100, 1,024 on the 40 GB
  A100, 512 or fewer on every other card; about 8,192 environments per process for the ant on an H100 NVL.
- 2026-11-12 01:05 PT: ant physics time breakdown on an H100 NVL (serval02 card 1): the constraint solve is 63% to
  72% of the environment step. Contact points 32 -> 16 take 64% of the time, solver passes 4 -> 2 take 67%,
  line-search passes 8 -> 4 take 79%; frame-loop unroll and the compiler option give nothing. Total speed stops
  growing near 8,192 environments per process (0.30 million steps per second). Table:
  `benchmark_runs/2026-11-12-00-17_*/profiles/antmaze_physics_split/antmaze_physics_split.md`.
- 2026-11-12 01:27 PT: trainer changes committed (`528081b5`): per-copy entropy, clip, value coefficient and reward
  mapping; the two reward settings with goal termination; preset `antmaze_current`; episode log; gather per epoch;
  checkpoint and resume; 131 tests pass, golden parity still bit-identical.
- 2026-11-12 01:29 PT: rollout-shape check started for PointMaze Large (jobs 6873737, 6873738, 6873739 on jaguar03
  RTX A4500s): 4, 16 and 128 environments per policy, intrinsic weights 0.0625, 1, 16, seeds 0 to 15, 50,331,648
  steps per policy. Folder `runs/2026-11-12-01-29_rollout-shape-check_*`. The AntMaze part waits for the ant's
  physics settings, because a kept physics change applies to every later run.
- 2026-11-12 02:05 PT: rollout-shape check on PointMaze Large read (units with 4, 16 and 128 environments per policy
  complete; 2 and 1 running, far enough to compare at equal steps). **PointMaze uses 4 environments per policy.**
  Whole-curve mean at intrinsic weight 1: 0.89 (4), 0.70 (16), 0.24 (128) at 50.3 million steps; 0.54 with 2 against
  0.79 with 4 at 20.7 million; 0.03 with 1 against 0.67 with 4 at 11.8 million. Other things seen: weight 0.0625
  almost never finds the goal (0 to 2 of 16 policies); weight 16 finds it in all 16 but ends low and falls late
  (final level 0.42 at 20.7 million steps, 0.22 at 50.3 million, 4 environments), so a too-large weight shows the
  rise and later fall; weight 1 with 4 environments ends at 0.99 and passes 95% of that near 16 million steps.
- 2026-11-12 02:05 PT: two more ant measurements started on the H100: MuJoCo's Warp implementation, and faster
  settings of the current implementation (fixed-length solver loop, smaller carried state, contact points, line
  search, solver passes) with their physics checks. Sweep machinery, analysis code and PointMaze speed ideas are
  being written and measured in parallel.
- 2026-11-12 02:20 PT: analysis and report code committed (`b805bfe8`, 44 tests); sweep machinery committed
  (`816c626b`, 82 tests). PointMaze idea profiles so far on an RTX A4500 at 4 environments per policy: 2.4 million
  steps per second at 512 and at 960 policies; many cells in one program cost nothing; next values from shifted
  values, gathering per epoch and the packed-key permutation gain 0% to 1% (not kept); target features once per
  iteration gain 3% on the A4500 and 6% on the RTX 5080.
- 2026-11-12 02:31 PT: **RND baseline, PointMaze first pass started** in
  `runs/2026-11-12-02-24_cleanrl-ppo-rnd-baseline_*`: PointMaze Medium and Large, both reward settings in one
  process, 60 configurations x 16 seeds, 4 environments per policy, 50,331,648 steps per policy, 16 units of 256
  policies (8 configurations each) from the frozen code copy of commit `6ad2fa46`. Workers 6874063, 6874076 to
  6874078, 6874082 (yuxinchen) and 6874080, 6874081 (sl5nw) hold 16 cards (RTX 5080, A40, RTX 4000 Ada, RTX A4000);
  persistent monitor job 6874083. All 16 units started within a minute.
- 2026-11-12 02:40 to 10:30 PT: no work by this session (its usage limit was reached at 02:40 PT; four background
  workers stopped with it and were resumed at 10:32 PT: analysis additions, PointMaze speed ideas, the Warp
  measurement, the faster settings of the current ant simulator). The jobs kept running by themselves.
- 2026-11-12 10:30 PT: **PointMaze first pass complete**: 16 of 16 units done, none failed (RTX 5080 units 48 minutes,
  A40 units 1 h 37 min, RTX 4000 Ada and RTX A4000 units 2 h 11 min to 2 h 18 min). Rollout-shape units with 1 and
  2 environments per policy also complete. My quick reading of the records at 50.3 million steps (the analysis
  code's report replaces it):
  - Large, reward 1 on arrival: best learning rate 1e-3, entropy 0.01, intrinsic weight 1: whole-curve mean 0.967,
    final level 0.999, 16 of 16 reach the goal, median first-goal step 421 thousand; the curve is at 0.35 after 1.0
    million steps, 0.75 after 1.6 million, 0.92 after 2.1 million, 0.96 after 4.2 million. With intrinsic weight
    0.00024 no policy reaches the goal, so the bonus is needed here.
  - Large, step cost: best learning rate 1e-3, entropy 0.01, weight 4: whole-curve mean -0.717, final level -0.706
    (about 283 of 400 steps per episode); with weight 0.00024, 10 of 16 reach the goal and the final level is -0.82.
  - Medium, reward 1 on arrival: best learning rate 1e-3, entropy 0.01, weight 1: whole-curve mean 0.998; the curve
    is at 0.63 in the first window (262 thousand steps) and 1.00 by the third; median first-goal step 25 thousand.
    With weight 0.00024, 15 of 16 reach the goal but the final level is 0.69.
  - Medium, step cost: every good configuration ends at -0.30 (about 121 steps per episode) within two windows,
    with intrinsic weights from 0.00024 to 0.25 among the best ten: the bonus makes no difference there.
  - Consequences: (1) the budgets from the rule will be small (about 8 million steps for Large with reward 1 on
    arrival, about 1 to 2 million for Medium), so Medium needs finer logging windows; (2) Medium with the step cost
    cannot separate bonus methods, so reward setting 1-on-arrival is the one where success in all five
    environments is possible; (3) the chosen learning rate and entropy sit at the top of their grids, so the second
    pass extends both.
- 2026-11-12 10:48 PT: PointMaze second pass submitted (16 units; weights at factor 2, learning rate to 3e-3,
  entropy to 0.03, clip 0.1 and 0.2, intrinsic discount 0.99 and 0.999; 256 logging windows; 4.2, 16.8 and 33.6
  million steps). Workers ask for 2 processors and 4 GiB per card now (measured need: one processor, 2.6 GiB). 12 of
  16 units done at 11:44 PT.
- 2026-11-12 11:45 PT: **the ant will use MuJoCo's Warp implementation** (inside MJX 3.11.0, environment
  `platform_jax_warp`). Measured on an H100 NVL: 0.93 million steps per second at 4,096 environments, 2.2 million at
  32,768, 2.4 million at 65,536, against 0.30 million at best for the JAX implementation. Checks: one step against
  C MuJoCo within 1.4e-6 m without contact and 1.3e-2 m in contact (the JAX implementation: 6.8e-5 and 2.75e-2;
  gates 1e-4 and 5e-2); resting height 0.382481 m; reward, termination and reset equal to the JAX environment; no
  non-finite state in 189 million random-torque steps; nothing dropped from its contact pool (at most 8% used).
  Open: its speed on RTX cards, and one case where a NaN written into the torso quaternion is not respawned. Not yet
  wired into the training command.
- 2026-11-12 11:45 PT: three background workers stopped a second time on the usage limit (PointMaze speed ideas,
  faster settings of the JAX ant, the Warp measurement). Their edits are in the working tree, not committed; the
  full test suites are running on it (jobs 6885610 to 6885612) before anything is committed. Speed ideas measured
  so far: 16 for PointMaze or both families and 5 for the ant; only "target features once per iteration" gains
  anything for PointMaze (3% to 6%).
- 2026-11-12 11:50 PT: batch 1 written (`batches/01_start-scale-and-decay-of-the-bonus-on-continuous-inputs/ideas.md`).
- 2026-11-12 12:05 PT: budgets read from the second pass: PointMaze Medium 1,048,576 steps (both reward settings),
  PointMaze Large with reward 1 on arrival 8,388,608. At those budgets the intrinsic discount 0.99 beats 0.999 and
  clip 0.1 beats 0.2, so CleanRL's values stay.
- 2026-11-12 12:46 PT: development at the budget, PointMaze Large with reward 1 on arrival (9 configurations, 16
  seeds, learning rate annealed over the 8,388,608 steps): best learning rate 2e-3, entropy 0.01, intrinsic weight
  2: whole-curve mean 0.888 ± 0.008, final level 0.994, median first arrival at 278 thousand steps. Second: learning
  rate 1e-3, weight 2: 0.881 ± 0.008. **This is RND's configuration for that case.**
- 2026-11-12 12:47 PT: confirmation units built for that case (12 units of 256 policies, seeds 30000 to 31023): the
  chosen configuration, RND at CleanRL's untouched settings, and PPO without a bonus.
- 2026-11-12 12:10 to 12:47 PT: UVA's scheduler is not starting our new workers (3,334 waiting jobs in the card
  partition, 2,901 of one other user; our priority is low). Units run on cards inside job 6884496, which we already
  hold: its own workers and one more worker started on an idle card through `srun --overlap`.
- 2026-11-12 12:56 PT: development at the budget on PointMaze Medium (1,048,576 steps, 16 seeds). Reward 1 on
  arrival: the best learning rate is the largest tried (1e-2, weight 2: whole-curve mean 0.944 ± 0.006), so the
  edge rule applies and learning rates 1e-2, 2e-2 and 3e-2 with weights 2 and 4 were queued. Step cost: the 12
  configurations lie between −0.341 and −0.377, the best four within their intervals; chosen learning rate 6e-3,
  weight 0.0078125 (−0.341 ± 0.005). Its confirmation units (1,024 fresh seeds, with the two reference rows) are
  queued.
- 2026-11-12 13:05 PT: budget for PointMaze Large with the step cost, read from the second pass: the best three
  configurations reach 95% of their final level at 7.7, 7.1 and 8.7 million steps, so the budget is 33,554,432
  steps (twice the largest, rounded up to a power of two).
- 2026-11-12 13:10 to 13:20 PT: UVA workers were not being started because they asked for 4 days. Workers now ask
  for 2 hours, 2 processors and 4 GiB; three started at once. Details in the baseline folder's `infra_history.md`.
- 2026-11-12 13:28 PT: batch 1 code reviewed; suites on the whole working tree pass (jobs 6887438: 80 passed;
  6887440: 54 passed; 6887439: 170 passed, 2 skipped; sweep machinery: 90 passed). Commit `2b84cab1`.
- 2026-11-12 13:35 PT: **batch 1 screen submitted** (14 units of 192 policies: 12 intrinsic weights × 16 seeds, on
  PointMaze Large with reward 1 on arrival at 8,388,608 steps; description
  `batches/01_.../screen_pointmaze-large_reward-arrival-plus-one.json`). Workers 6887738 to 6887755, monitor
  6887756. The first unit started at 13:35 PT on ai02.
- 2026-11-12 13:48 PT: PointMaze Medium with reward 1 on arrival, higher learning rates at the budget (16 seeds):
  learning rate 1e-2 with weight 2 gave 0.917 ± 0.011 in this unit against 0.944 ± 0.006 in the earlier unit (same
  settings and seed numbers, another card model: the two differ by 0.027, about twice the standard error of the
  difference); 2e-2 and 3e-2 are far lower (0.81 and 0.54). **RND's configuration for this case: learning rate
  6e-3, entropy 0.01, weight 2 (0.942 ± 0.007, final level 0.999)**: inside the range of the learning rate and of
  the weight, and above the mean of the two runs at 1e-2 (0.930). Confirmation units built (1,024 fresh seeds, with
  the two reference rows).
  Observation to keep: two runs with the same settings and seed numbers on different cards are separate samples
  (their curves part after the first rounding difference), so "the same seeds" pairs the comparison by seed number
  only.
- 2026-11-12 13:45 PT: Gilbreth (Purdue, standby cards) runs units of the baseline folder: 9 done and verified, 4
  running. Gautschi's L40S queue estimates a start on 2026-11-14, so no unit waits there.
- 2026-11-12 14:10 PT: **first readings of the batch 1 screen** (quick script; the generated summary replaces
  them). Whole-curve mean at each unit's best intrinsic weight, 16 seeds:
  control 0.885 ± 0.010 (weight 2; the baseline's own run gave 0.888 ± 0.008);
  separate predictor step size at 0.25 times 1e-4: 0.887 ± 0.012 (weight 1), at 1 times: 0.881 ± 0.007 (weight 4);
  fixed scale 0.547 ± 0.096 (weight 16, 13 of 16 policies reach the goal);
  even start with no divisor 0.327 ± 0.099 (weight 0.25, 10 of 16);
  even start 0.186 ± 0.085 (weight 0.25, 9 of 16).
  So prediction 1 (even start changes nothing) and prediction 2 (fixed scale and no divisor gain 0.02 to 0.05) did
  not hold: all three are far behind.
- 2026-11-12 14:10 PT: **what the control's bonus looks like on the maze grid** (median over 16 seeds, weight 2).
  Before any training it is 0.6 at the start cell and rises with the distance from it to 56 at the far cells,
  where it is the same number in every cell beyond the input clip of ±5. The inputs are whitened with statistics
  from the random-action rollouts around the start, so a far cell is a large input and an untrained predictor's
  error grows with the size of its input. RND therefore starts with a bonus that points away from the start. After
  32,768 steps the bonus is 0.007 to 0.03 near the start and 0.3 to 0.45 in the far cells; after 262,144 steps it
  is still highest in the corner farthest from the start (0.11) and lower at the goal (0.04). The even start
  removes this slope (0.5 everywhere), and fewer policies then reach the goal at all.
  Two things to check with the full summary: (1) how much of the control's missing whole-curve mean (0.115) is
  before the first arrival (median 192 thousand steps, 2.3% of the budget) and how much is the rise after it;
  (2) whether the bonus left in the far corner keeps policies away from the goal after they have found it.
- 2026-11-12 14:28 PT: PointMaze Large with the step cost, development at the budget (33,554,432 steps, 16
  seeds, learning rates 2e-3 to 6e-3, weights 2 to 8, both intrinsic discounts). Per seed the whole-curve mean
  takes one of two values: about −0.717 (the policy reaches the goal in about 285 of 400 steps, close to the
  shortest time) or about −0.90. A configuration's mean is decided by how many of its 16 seeds end at −0.90: the
  earlier best (intrinsic discount 0.999, learning rate 3e-3, weight 4: −0.718 with 16 of 16 at the higher value)
  gave −0.766 in its repeat (12 of 16). The intrinsic discount 0.999 is therefore not kept.
  **RND's configuration for this case: learning rate 2e-3, entropy 0.01, weight 4, intrinsic discount 0.99
  (−0.7247 ± 0.0009, final level −0.712, 16 of 16).** Confirmation units built (1,024 fresh seeds, with the two
  reference rows); four went to Gilbreth and seven UVA workers were submitted (6889111 to 6889117, 4-hour limit,
  because one unit takes about 2 hours on an RTX 2080 Ti).
  For the comparison this means: with the step cost a method can gain mostly by having fewer seeds at −0.90.
- 2026-11-12 14:25 PT: more first readings of the batch 1 screen. Error length as the bonus: 0.845 ± 0.008 (weight
  0.0625). Separate predictor step size at 4 times: 0.878 ± 0.010 (weight 4). Scheduled last layer with decay 1:
  about 0.0004 at every weight. Its grid records show why: after the first 32,768 steps the bonus is 0.02 at
  visited cells and 0.03 at cells never visited (it started at 1 everywhere), so fitting the visited inputs lowers
  the error everywhere (the fixed random features of different maze positions are nearly the same vector), and
  later the bonus rises again to 0.1 to 1.0 as the input statistics move. Both causes were named as risks in
  `ideas.md`.
- 2026-11-12 14:25 PT: where the control (weight 2) loses its whole-curve mean of 0.115: 0.058 in the first 524
  thousand steps (curve at most 0.32), 0.026 between 524 thousand and 1.05 million, 0.013 between 1.05 and 2.1
  million, 0.018 from there to the end (the curve is 0.944 at 2.1 million and 0.990 at 6.3 million).
- 2026-11-12 14:29 PT: **baseline confirmation results so far** (1,024 fresh seeds; the baseline folder's
  `analysis/analysis.md`): PointMaze Large with reward 1 on arrival: RND 0.881 ± 0.001 (final level 0.993), RND at
  CleanRL's untouched settings 0.453 ± 0.010, PPO without a bonus 0.005. PointMaze Medium with reward 1 on arrival:
  RND 0.932 ± 0.001 (final level 0.994). PointMaze Medium with the step cost: RND −0.351 ± 0.001, PPO without a
  bonus −0.348 ± 0.001, which is ahead of RND by 0.003 [0.001, 0.005]. RND's chosen weight there (0.0078125) was
  the lowest tried, so three lower weights were queued (edge rule); if one is better, RND's configuration and its
  confirmation for that case are replaced.
- 2026-11-12 14:29 PT: batch 1 summary generated with 9 of 14 units
  (`batches/01_.../summary.md`; regenerate with `scripts/maze_campaign/analysis/batch_summary_job.py`). No idea is
  ahead of the control; the three predictor step sizes are within the interval; the rest are behind.
  A fact about the seeds found by the analysis: the random actions of the rollouts before training are drawn per
  place in the process, not per seed number, so the same seed number starts slightly differently at each weight.
  It is left as it is, because the baseline's runs used it.
- 2026-11-12 14:45 PT: batch 2 written (`batches/02_.../ideas.md`, 15 units): wider input clip, input statistics
  kept from the random-action rollouts, slower predictor, local (random Fourier) features as the input of RND,
  scheduled last layer on local features. A validation block of 256 seeds (10000 to 10255) is added for ideas near
  the control. Implementation started.
- 2026-11-12 14:45 PT: PointMaze Medium with the step cost, lower intrinsic weights (16 seeds): weight 0.00195 at
  learning rate 6e-3 gives −0.3405 ± 0.003 (final level −0.304), level with the earlier choice (−0.3408 at weight
  0.0078) and ahead of it on the final level, so by the rule it is **RND's configuration for that case**; its
  confirmation is queued and the earlier confirmation (−0.351 ± 0.001) stays on disk, left out of the report.
- 2026-11-12 14:45 PT: batch 1, fast last layer with step 0.1: best whole-curve mean 0.862 ± 0.016 (weight 0.5);
  median first arrival 61 thousand steps at weight 1 and 55 thousand at weight 2 (control: 353 and 192 thousand)
  but final levels 0.96 and 0.75. Scheduled last layer with decay 0.5: 0.25 at best. Batch 2 got a seventh idea
  from the records: the divisor held after 1/16, 1/8 or 1/4 of the run (18 units now).
- 2026-11-12 14:45 PT: the control was queued on the validation block (seeds 10000 to 10255, weights 1, 2 and 4;
  3 units, handed to Gilbreth), so that every later idea near the control has a reference that resolves about
  0.007.
- 2026-11-12 14:50 PT: **the Warp ant trains from the training command and the sweep machinery** (commit
  2b84cab1). On an H100 NVL, U-maze, the study's learner: 1.68 million steps per second at 16,384 environments
  (1,024 policies × 16, or 128 × 128; 1.25 s per iteration; peak card memory 17.8 GiB), 0.89 million at 4,096
  environments; the rollout is 90% to 94% of the iteration. The cell at 32,768 environments is not measured yet.
- 2026-11-12 14:50 PT: **defect: ants are thrown over the maze walls in both physics implementations.** With
  random torques (16,384 ants, 1,400 steps) the torso is above the 2 m walls in 51,818 steps with MuJoCo Warp
  (largest height 5.5 m; 2 arrivals at the U-maze goal, both over the wall) and in 63,100 steps with MJX's JAX
  implementation (19 m; 15 arrivals). No AntMaze run is started before this is fixed. Fix chosen (routine, applies
  to every method alike, the user can reverse it): a ceiling at the wall-top height that only the torso collides
  with, plus per-window counters of steps near the ceiling and of the largest torso speed, so that a policy that
  gets itself thrown is visible. A worker is implementing and verifying it.
- 2026-11-12 14:55 PT: remote H100 servers, read through the open connections without any new login: MIT (portal11)
  is reachable, but other sessions of this account hold 4 of 4 cards on `mit_preemptable` and have 1 running and 5
  waiting jobs on `mit_normal_gpu`, so new work would wait there; Gautschi's 2 H100 cards are in use by another
  session; ACES (portal05) and DeltaAI (portal11) connections are open. MIT has the earlier study's environment
  `~/orcd/scratch/envs/rnd_platform_jax` (no mujoco, no warp).
- 2026-11-12 15:10 PT: **batch 1 is complete (14 of 14 units, none failed) and closed: no idea is ahead of the
  control** (`batches/01_.../summary.md`). Within the interval: the three predictor step sizes (+0.003, −0.003,
  −0.006) and the fast last layer at steps 0.3 (−0.016 [−0.036, +0.005]) and 0.1 (−0.023 [−0.064, +0.013]).
  Behind: error length (−0.040), fast last layer at step 1 (−0.086), fixed scale (−0.338), even start with no
  divisor (−0.557), the three scheduled last layers (−0.637, −0.779, −0.884), even start (−0.698). No idea goes to
  the development sweep. Of the six predictions, one held (error length), five did not. What batch 1 leaves for
  batch 2 is in `batches/02_.../ideas.md`, "What batch 1 showed".
- 2026-11-12 15:23 PT: RND's confirmation on PointMaze Medium with the step cost at the new weight 0.00195:
  −0.3517 ± 0.0008 (final level −0.3075), the same as at weight 0.0078 (−0.3512 ± 0.0009) and 0.0035 behind PPO
  without a bonus (−0.3481 ± 0.0008). A bonus this small cannot cost that much by its reward, so the cost is
  somewhere else in the RND program; the first suspect is the gradient clip that the policy shares with the
  predictor (the predictor's large early gradient shrinks the policy's step). Test queued on the validation block
  (seeds 10000 to 10255, PointMaze Medium, step cost): RND as it is, the predictor with its own optimizer outside
  the clip at 0.25 and 1 times 1e-4, and PPO without a bonus. The same two predictor settings are queued on
  PointMaze Large with reward 1 on arrival at their two best weights.
- 2026-11-12 15:27 PT: **batch 2 screen submitted** (18 units; frozen code d7054945; description
  `batches/02_.../screen_pointmaze-large_reward-arrival-plus-one.json`). Four units went to Gilbreth at once and 15
  UVA workers were submitted (6890372 to 6890472). The implementation settled two points of the scheduled last
  layer on local features: it has no bias input (a bias would give far inputs a shared part of about 0.5), and
  with 512 features an input far from every fitted input keeps 0.93 to 0.98 of its squared error, not above 0.99,
  because random features of far inputs still overlap by about 0.044.
- 2026-11-12 15:45 PT: **user decision: a secondary metric, the mean capped first-goal time** (the grids'
  metric), to read discovery alone; the primary metric stays the whole-curve mean (discovery and optimization).
  Each method and RND get their own configuration for it; reports carry two tables; a method ahead of RND on it in
  all five environments is noted. Written into the rule file, `program.md` and the memory note. Work it adds:
  (1) RND's own configuration for discovery in every case (a development sweep toward larger intrinsic weights,
  then its confirmation); (2) a second table in the baseline report and in every batch summary; (3) each idea's
  best weight for discovery in the screens.
- 2026-11-12 15:50 PT: first reading of batch 1 on the secondary metric (quick script, 16 development seeds, each
  idea at its own best weight for it; mean capped first-goal time in thousands of steps): fast last layer at step
  0.1: 62 ± 13 (weight 16); error length: 88 ± 13 (64); fast last layer at step 1: 114 ± 22 (16), at step 0.3:
  116 ± 30 (64); separate predictor step size at 1 times: 132 ± 25 (64), at 0.25 times: 137 ± 36 (64), at 4
  times: 177 ± 30 (16); control: 190 ± 51 (64). Several best weights are the largest tried, so weights 256, 1,024
  and 4,096 were queued for the control and these ideas (8 units), and RND's own development sweep for discovery
  was queued in the baseline folder (learning rate × entropy 0.01, 0.03, 0.1 × weight 1 to 1,024; PointMaze Large
  with reward 1 on arrival and PointMaze Medium with both rewards; PointMaze Large with the step cost follows
  around the best cells, because one unit there takes about 50 minutes).
  A worker is adding the second table to the batch summary and to the baseline report.
- 2026-11-12 15:50 PT: **validation block, PointMaze Large with reward 1 on arrival (256 seeds, 10000 to 10255;
  quick script, the generated summary will replace it).** Control: whole-curve mean 0.8759 ± 0.0022 at weight 1,
  0.8781 ± 0.0021 at weight 2, 0.8626 ± 0.0025 at weight 4; mean capped first-goal time 385, 269 and 205 thousand
  steps. **Predictor with its own optimizer at 0.25 times 1e-4, weight 2: 0.8859 ± 0.0024, which is +0.0078 ±
  0.0032 against the control at the same weight (the control's best) on the same seeds, final level 0.9954 against
  0.9920, first-goal time 171 thousand against 269 thousand (−99 ± 16 thousand).** At 1 times: +0.0024 ± 0.0031 at
  weight 2 and 0.8762 at weight 4 (below the control's best). The unit at 0.25 times and weight 1 is still running.
  This is the first idea ahead of the control on the primary metric on seeds it was not chosen on, by a small
  amount; four comparisons were made, so it needs the fresh seeds before it counts. Batch 2 has the slower
  settings 0.0625 and 0.125.
- 2026-11-12 15:40 PT: **the ant ceiling is verified and committed (30b2c609).** With it, 16,384 ants under
  random torques for 1,400 steps had 0 steps above the wall tops and 0 arrivals over a wall, with MuJoCo Warp and
  with MJX's JAX implementation; agreement with C MuJoCo is unchanged; speed is unchanged (1.67 million steps per
  second at 16,384 environments, and 1.93 million at 32,768 environments with 34.5 GiB, on an H100 NVL).
  **The ant leaves the floor by its own leg power in C MuJoCo too** (motors of 150 N m per joint on a body under
  1 kg): of 1,024 ants with random torques for 700 steps, 83 rose above the 2 m wall tops and 4 were over a wall
  cell. So crossing walls is a property of the Gymnasium ant model, not a fault of the fast physics. With the
  ceiling an untrained policy still spends about 0.33% of steps within 0.5 m of it; the per-window counter is read
  against that level.
- 2026-11-12 15:57 PT: **AntMaze baseline folder created and its first unit handed to Gilbreth**:
  `runs/2026-11-12-15-56_cleanrl-ppo-rnd-baseline_antmaze-umaze-medium-large_mujoco-warp-physics_ceiling-on-wall-tops_goal-termination_reward-arrival-plus-1-and-step-cost_environments-per-policy-check-4-16-128_seeds-0-15_confirmation-seeds-from-30000`.
  First phase: the check of 4, 16 and 128 environments per policy on the U-maze with reward 1 on arrival
  (48 configurations: learning rate 1e-4 to 3e-3, entropy 0.001 and 0.01, weight 0.0625 to 64; 16 seeds;
  16,777,216 steps; 8 units). MIT and Gautschi cannot start H100 work now (other sessions hold this account's
  limits), so the units run on Gilbreth's standby A100 and A30 cards.
- 2026-11-12 16:05 PT: **the test on PointMaze Medium with the step cost refutes the gradient-clip suspicion**
  (256 seeds, 10000 to 10255; RND at weight 0.00195): RND −0.3524 ± 0.0018, PPO without a bonus −0.3471 ± 0.0014
  (RND is behind by 0.0053 ± 0.0022 on the same seeds), predictor with its own optimizer outside the clip −0.3548
  ± 0.0016 at 0.25 times and −0.3555 ± 0.0021 at 1 times, which is no better than RND as it is. So what a
  nearly weightless RND costs there is not the predictor's share of the clip. What is left in the RND program and
  absent without a bonus: the intrinsic value head (its loss shares the critic and the clip) and the reward's
  running statistics. That belongs to days 6 to 10 (how the bonus enters PPO, the value heads).
- 2026-11-12 16:05 PT: batch 2, first units (16 development seeds, quick script): **divisor held after 1/16 of the
  run: whole-curve mean 0.895 ± 0.005 at weight 4 and 0.892 ± 0.009 at weight 2** (final levels 0.998 and 0.997;
  batch 1's control: 0.885 at weight 2, and 0.867 with a final level of 0.990 at weight 4); held after 1/4: 0.887 ±
  0.005 at weight 4; held after 1/8: 0.873 ± 0.013 at weight 4. Mean capped first-goal times 162 to 183 thousand
  steps, like the control's 190. Prediction 6 holds so far (higher final level at weight 4; best weight moves up).
- 2026-11-12 16:10 PT: queued (a) on the validation block: divisor held after 1/16 at weights 2 and 4, held after
  1/4 at weight 4, slower predictor with held divisor at weights 2 and 4, slower predictor alone at weight 4;
  (b) **batch 3** (`batches/03_.../ideas.md`, 6 units): combinations of the slower predictor, the held divisor
  and the fast last layer, with predictions.
- 2026-11-12 16:17 PT: the session restarted at about 16:14 PT (a few minutes' gap). The 20-minute monitor was
  created again with the AntMaze folder and the fleet step in its text (0a3e2261; the earlier one deleted), the
  5-minute hand-over loop was started again for the three folders, and the report worker was resumed.
- 2026-11-12 16:17 PT: **first AntMaze unit read by hand and the other 7 released to Gilbreth.** On an NVIDIA A30 one
  iteration at 16,384 environments takes 3.85 s (0.545 million steps per second; an H100 NVL: 1.67 million), so a
  unit with 128 environments per policy takes about 66 minutes there, the unit with 16 about 7 hours and the unit
  with 4 about 11 hours (4-hour jobs, continued from checkpoints). Gilbreth's A100 cards have 1,457 standby jobs
  waiting, so our jobs get A30 cards.
  Two early facts from that unit (128 environments per policy, at 2.4 million steps per policy): some policies
  already reach the U-maze goal; and policies trained with the bonus spend 0.7% to 8.2% of their steps within
  0.5 m of the ceiling (a random policy: about 0.33%), with largest torso speeds of 7.8 to 12.8 m/s. The bonus
  rewards leaving the floor. The Gymnasium AntMaze has no rule against it either.
- 2026-11-12 16:25 PT: **RND's own sweep for discovery, first three cases complete** (72 configurations each, 16
  seeds; mean capped first-goal time in thousands of steps). PointMaze Large with reward 1 on arrival: 111 ± 17 at
  learning rate 6e-3, entropy 0.01, weight 64; the next five cells are 117 to 135, at weights 4 to 1,024. The best
  learning rate is the largest tried, so 1e-2 and 2e-2 were queued. PointMaze Medium: 9.7 ± 1.8 with reward 1 on
  arrival and 8.3 ± 1.5 with the step cost, and about 10 for very different settings: on Medium the goal is found
  within about 20 rollouts whatever the bonus. Extensions queued there too (largest entropy and weight were best),
  and PointMaze Large with the step cost around the best cells.
- 2026-11-12 16:25 PT: **batch 2 on the secondary metric (16 development seeds, each arm at its best weight, all
  at learning rate 2e-3 and entropy 0.01): the scheduled last layer on local features is the fastest so far.**
  Decay 1, length 1: 45 ± 6 thousand steps (weight 4); decay 0.5: 59 ± 11 and 64 ± 11 (two lengths); local
  features as the input of RND, length 2: 84 ± 14; wider input clip: 139 and 145; held divisor: 162 to 183;
  control: 190 ± 51 (RND's own best over 72 configurations: 111 ± 17). All 16 policies reach the goal in each.
  **On the primary metric the same arms are far behind:** scheduled last layer on local features 0.05, local
  features as the input of RND 0.35, against the control's 0.885. They find the goal early and do not settle on
  it: a bonus that only counts stays near 1 at rarely visited states for the whole run. Wider input clip: 0.887
  (clip 10) and 0.883 (clip 20), like the control.
- 2026-11-12 16:35 PT: **the process that ran this session before the 16:14 PT restart was still alive** and
  told this one so (it had fired one more monitoring tick at 16:23 PT and its own copy of the report worker had
  finished there). It made no submission, table edit or commit after the restart and has stood down. The report
  worker therefore ran in both processes on the same files; the combined working tree was checked
  (`tests/maze_campaign_analysis` and `tests/maze_campaign`: 185 passed), committed as 52059227, and the three
  batch summaries and the baseline report were generated again from it.
- 2026-11-12 16:35 PT: **reports now carry both metrics** (baseline `analysis/analysis.md`; `summary.md` of batches
  1, 2 and 3, each with the validation block). RND's configurations for the secondary metric are named there as
  "not confirmed yet". Open point raised by the report: on PointMaze Medium two configurations of the discovery
  sweep have a slightly higher whole-curve mean than RND's chosen one (0.948 ± 0.009 against 0.944 ± 0.006 with
  reward 1 on arrival; −0.339 against −0.341 with the step cost), within their intervals. To do: put RND's best
  few configurations per case on the validation block, as the ideas are, and confirm the winner, so that the
  baseline gets the same selection as a method.
- 2026-11-12 16:45 PT: **ACES is prepared for ant units, not in use yet.** Environment
  `/scratch/group/p.cis261532.000/envs/rnd_platform_jax_mujoco_warp` (the same package list as Gilbreth's, built on
  the login node at no charge); the fleet's entry is in the AntMaze folder
  (`infrastructure/remote_fleet_servers_aces_entry.json`, with a merge script beside the setup files) and is not
  merged into the live table. No job was submitted there. Balance 71,499 SUs (525.7 H100-hours). No H100 node was
  idle at 16:18 PT and 114 jobs waited, so ACES would not start units sooner than Gilbreth today. Whether JAX and
  Warp run on an ACES card is known only after the first real unit.
- 2026-11-12 16:22 to 16:58 PT: **the first AntMaze unit stopped itself at window 64 of 256, and the check was
  started again.** One ant needed 161 constraint rows against a limit of 160, and the trainer's check ends a run
  rather than let MuJoCo Warp drop part of the physics. The limit came from random torques (49 rows); trained
  policies need more. It is now 512 (commit 5c7407bb); all 8 units were rebuilt and handed to Gilbreth again at
  16:57 PT. Nothing of the stopped unit is used. Cost: about 55 minutes on the ant check.
- 2026-11-12 16:36 PT: PointMaze units moved from Gilbreth back to UVA (Gilbreth's standby queue had slowed for
  this account; UVA starts sl5nw workers at once); Gilbreth's standby places go to the ant. `submit.py --accounts`
  limits new workers to chosen accounts.
- 2026-11-12 17:00 PT: **batch 2 with 17 of 18 units (generated summary, 16 development seeds).**
  Whole-curve mean, difference from the control: slower predictor at 0.0625 times 1e-4: +0.012 [−0.010, +0.034]
  (0.897 ± 0.007 at weight 2, median first arrival 67 thousand steps against 192 thousand); divisor held after
  1/16: +0.010 [−0.009, +0.032]; held after 1/4, the three input clips, the slower predictor at 0.125 times: within
  ±0.002; held after 1/8: −0.012; input statistics kept from the random-action rollouts: −0.054 [−0.076, −0.026]
  (behind); local features as the input of RND: −0.53 to −0.70; scheduled last layer on local features: −0.82 to
  −0.87. No clip gives the same numbers as clip 20: the whitened inputs do not reach 20.
  Mean capped first-goal time, difference from the control (190 thousand): scheduled last layer on local features
  −144, −131 and −125 thousand (intervals below zero); input statistics kept: −109 [−221, −29]; local features as
  the input of RND: −106 and −97 (below zero) and −64 (within); the rest within.
  Readings of the predictions: 1 (wider clip) held on the whole-curve mean and is within noise on the arrivals;
  2 (kept statistics) took its second branch, behind on the whole-curve mean, with earlier arrivals; 3 (slower
  predictor) held; 4 (local features as the input): all 16 reach the goal at lengths 1 and 2 but the whole-curve
  mean is 0.20 to 0.35, so the even start's failure to find the goal came from fits that were not local, and what
  is missing here is that the policy settles; 5 (scheduled last layer on local features): all 16 reach the goal,
  the whole-curve mean is the lowest of the batch; 6 (held divisor) held so far.
- 2026-11-12 17:00 PT: **validation block (256 seeds), differences from the control at the same weight.**
  Divisor held after 1/16, weight 2: +0.009 [+0.004, +0.015] (0.888 ± 0.002; final level +0.006). Slower predictor
  (0.25 times) with held divisor: +0.007 [−0.001, +0.014] at weight 2 and 0.888 ± 0.002 at weight 4 (+0.026 against
  the control at weight 4, +0.010 against the control's best, 0.878 at weight 2). Slower predictor alone, weight 2:
  +0.008 [+0.002, +0.014]. The two gains do not add: every arm ends near 0.886 to 0.888. On the first-goal time
  the slower predictor is ahead (171 and 153 thousand steps against 205 to 269) and the held divisor changes
  nothing, as expected of a change that acts after the hold.
- 2026-11-12 17:00 PT: batch 3, first units: all three parts with the divisor held after 1/8, step 0.1: whole-curve
  mean 0.894 ± 0.006 at weight 0.5 (+0.010 [−0.014, +0.037]), final level 0.997; at weight 4 its mean capped
  first-goal time is 63 ± 10 thousand with a whole-curve mean of 0.714 (the fast last layer alone had 0.016 at its
  fastest weight). So the held divisor does remove most of the fast last layer's late cost.
- 2026-11-12 17:03 PT: **RND's configuration for discovery chosen in three cases** (development seeds): PointMaze
  Large with reward 1 on arrival: learning rate 1e-2, entropy 0.01, weight 64 (95 ± 21 thousand steps; 2e-2 is
  worse, so the learning rate is inside its range now); PointMaze Medium with reward 1 on arrival: learning rate
  2e-2, entropy 0.1, weight 4 (9.2 ± 2.2 and 9.9 ± 2.0 in two runs); PointMaze Medium with the step cost:
  learning rate 1e-2, entropy 0.01, weight 4 (8.3 ± 1.5). Their confirmations on the 1,024 fresh seeds are queued
  (phase `confirmation-for-discovery`). PointMaze Large with the step cost waits for its sweep.
  Queued for the ideas that lead on discovery (scheduled last layer on local features with decay 1 and length 1;
  fast last layer at step 0.1; all three parts with the divisor held after 1/8): learning rate 2e-3, 6e-3, 1e-2 ×
  entropy 0.01, 0.03 × weight 4, 16, 64, so that they are tuned as RND was.
- 2026-11-12 17:20 to 17:26 PT: **second restart of the AntMaze check, caused by a fault in the fleet.** After
  the take-back and the second hand-over of the rebuilt units, every fleet step took six units back and handed
  them over again because of a marker the take-back had left on the server; two or three jobs then ran one unit
  at once from 17:01 to 17:20 PT. All of it is discarded (set aside in labelled folders on both sides). The fleet
  is corrected (commit 89cd0c5b, test added, 94 pass) and the 8 units were built fresh and handed over at 17:24 PT;
  a second step at 17:26 PT changed nothing. The PointMaze folders were not affected: their taken-back units were
  never handed to Gilbreth again (their fleet logs show no repeat after 16:36 PT).
  Lesson for the checks: after any hand-over, the next fleet step must report "0 handed over" and the server's
  waiting and running lists must not share a unit.
- 2026-11-12 17:44 PT: all 8 AntMaze units run on Gilbreth, one job each (two had failed at their start on stale
  configuration files left on the server and were built again). Units with 128 environments per policy take about
  1.5 hours on an A10, the unit with 16 about 10 hours, the unit with 4 longer; the three counts can be compared
  on their first 4.2 million steps well before all are done.
- 2026-11-12 17:45 PT: **batch 3 complete (6 units, 16 development seeds).** Whole-curve mean, difference from
  the control: all three parts with the divisor held after 1/16: +0.013 [−0.014, +0.042] (0.898 ± 0.008 at weight
  0.5); held after 1/8: +0.010; fast last layer at step 0.3 with held divisor: +0.004; at step 0.1 with slower
  predictor: −0.002; slower predictor with held divisor: −0.003; fast last layer at step 0.1 with held divisor:
  −0.010. All within their intervals. Mean capped first-goal time: fast last layer at step 0.1 with held divisor
  62 ± 13 thousand (weight 16; whole-curve mean 0.24 there), all three with the divisor held after 1/8: 63 ± 10
  thousand (weight 4; whole-curve mean 0.714 and final level 0.973 there), both ahead of the control's 190.
  Predictions: 1 did not hold (the two gains do not add); 2 held on the final level and the first-goal time, not
  on the whole-curve mean; 3 held (all three: best whole-curve mean of the batch, first-goal time level with the
  best, 1/8 against 1/16 within 0.01).
- 2026-11-12 17:46 PT: **RND confirmed for the discovery table on 1,024 fresh seeds.** PointMaze Large with reward
  1 on arrival (learning rate 1e-2, entropy 0.01, weight 64): mean capped first-goal time 136 ± 3 thousand steps
  (the 16 development seeds had given 95 ± 21). PointMaze Medium with the step cost (learning rate 1e-2, entropy
  0.01, weight 4): 14.1 ± 0.3 thousand (development: 8.3 ± 1.5); PPO without a bonus there: 19.1 ± 0.5. For
  PointMaze Medium with reward 1 on arrival and PointMaze Large with the step cost the report names other
  configurations than the ones run so far (it uses the first finished run of a repeated configuration, and the
  extension units had not been marked as discovery units in its settings); their runs are queued.
- 2026-11-12 17:48 PT: RND's fresh-seed runs for the discovery table queued in the two remaining cases, at the
  configurations the report names (PointMaze Medium with reward 1 on arrival: learning rate 2e-2, entropy 0.01,
  weight 4; PointMaze Large with the step cost: learning rate 4.5e-3, entropy 0.01, weight 8): 8 units in the
  baseline folder, UVA jobs 6894023 to 6894025. The report's settings now also read the extension units as part
  of the discovery sweep.
- 2026-11-12 17:50 PT: **validation block (256 seeds) for the two three-part combinations of batch 3** (fast
  last layer at step 0.1, predictor at 0.25 times 1e-4 with its own optimizer, divisor held after 1/16 or after
  1/8 of the run), intrinsic weights 0.25, 0.5 and 1: 6 units, UVA jobs 6894042 to 6894045, all started within
  a minute. Reason: on 16 seeds both are the best of the batch on the whole-curve mean (+0.013 and +0.010) but
  inside their intervals; the validation block's control is 0.878 at weight 2. The validation-block unit names
  use the short method names `three-parts-held-one-sixteenth` and `three-parts-held-one-eighth` (the unit name
  limit of 200 characters); the batch summary maps them to their full names.
- 2026-11-12 18:02 PT (report regenerated): RND confirmed for the discovery table on PointMaze Medium with reward 1 on arrival (1,024
  fresh seeds; learning rate 2e-2, entropy 0.01, weight 4): mean capped first-goal time 17.4 ± 0.5 thousand
  steps; RND at CleanRL's defaults 210 ± 9, PPO without a bonus 455 ± 15. Three of the four PointMaze cases now
  have RND's row in the discovery table; PointMaze Large with the step cost is running (4 units at 20 to 31 of
  256 windows).
- 2026-11-12, between 18:00 and 18:04 PT: **AntMaze: the physics diverges for trained ants** (largest torso speeds of 1e6 to 1e15 m/s
  in 2 to 9 of 16 policies of most configurations, starting once the policies learn; 7% to 11% of steps with the
  torso above 1.5 m). Details in the AntMaze folder's `infra_history.md`. The rollout-shape check keeps running
  for its records, but nothing is chosen from it. Diagnosis started 18:04 PT (speed study folder,
  `profiles/antmaze_warp_blow_up/`); the AntMaze baseline waits for its result. The reduced solver (4 iterations,
  8 line-search iterations) is the first suspect. This is the third environment problem the gates with random
  torques did not show (walls crossed from above, the constraint-row limit, now this): every later ant gate also
  runs trained policies, not only random torques.
- 2026-11-12 18:25 PT: this folder's monitor job had ended on a refused submission at 18:15 PT (the planner's
  card table is out of date for four nodes); two discovery-sweep units waited about 7 minutes without a worker.
  Planner corrected (commit 28c6681f), workers and a new monitor job submitted; details in `infra_history.md`.
  Open: regenerate the shared resource document for affogato13, lynx01, lynx02, lynx05; freeze this folder's code
  anew at a moment when no unit is part-way through.
- 2026-11-12 18:44 PT: **discovery sweeps of two leading ideas done** (18 configurations each, 16 development
  seeds, PointMaze Large with reward 1 on arrival). Mean capped first-goal time at each idea's best configuration:
  fast last layer at step 0.1: 46 ± 5 thousand steps (learning rate 2e-3, entropy 0.01, weight 16); all three parts
  with the divisor held after 1/8: 58 ± 9 thousand (learning rate 6e-3, entropy 0.03, weight 16). RND's best of 84
  configurations on the same seeds was 95 ± 21 thousand, and 136 ± 3 thousand on 1,024 fresh seeds, so a best
  value from 16 seeds is too low by a wide margin; the ideas' numbers carry the same effect and are not a verdict.
  The sweep of the scheduled last layer on local features is at 112 of 256 windows.
- 2026-11-12 18:45 PT: two studies queued in this folder (19 units, UVA jobs 6894918 to 6894928; 12 cards started
  within a minute, 7 wait):
  1. `discovery-validation`: the three best configurations for the first-goal time of RND, of the fast last layer
     and of all three parts (divisor held after 1/8), each on the 256 seeds of the validation block (9 units). It
     gives the baseline and the ideas the same second selection step before any fresh seed is used.
  2. `development-for-discovery-grid-of-rnd`: the two ideas on the 84 configurations RND's discovery sweeps had
     in this case (72 + 12; 10 units), because the ideas' first sweep had only 18 and each one's best was at an
     end of its grid (lowest learning rate; highest entropy).
  The corrected planner used lynx01, lynx02 and lynx05 (RTX 4000 Ada) for 11 of the cards.
- 2026-11-12 18:41 PT: AntMaze: 5 finished units copied back after two failed fleet steps (missing `data/` folder
  on UVA after my clean-up); they are records for the physics diagnosis only.
- 2026-11-12 19:04 PT: **validation block (256 seeds) for the three-part combinations read.** Each method at its
  best weight against the control at its best weight (weight 2: whole-curve mean 0.8781), paired on the same
  seeds: all three parts with the divisor held after 1/16, weight 0.5: 0.8917, +0.0137 [+0.0072, +0.0203], final
  level +0.0049 [+0.0036, +0.0063], first-goal time −109 thousand [−139, −79]; slower predictor with held divisor,
  weight 4: +0.0102 [+0.0047, +0.0159]; divisor held after 1/16, weight 2: +0.0095; predictor at 0.25 times 1e-4,
  weight 2: +0.0078; all three with the divisor held after 1/8, weight 0.5: +0.0078. The batch summary's
  validation section compares only at equal weights; the comparison at each one's best weight was computed by a
  quick script and is tabled in batch 4's `ideas.md` (to be moved into the summary code).
  **Finalists for the whole-curve mean: all three parts (divisor held after 1/16), and the slower predictor with
  held divisor.** The gain is small (about 0.010 to 0.014 on 0.878) but its interval is above zero on 256 seeds
  and the final level is not behind: the form of the success rule in this one case. Fresh seeds are not used yet:
  they are kept for the method as it stands at the end, and the validation block serves until then.
- 2026-11-12 19:06 PT: **batch 4 queued** (`batches/04_two-finalists_...`): both finalists' development sweeps
  in the other three PointMaze cases (15 configurations each, 16 seeds; 6 units; UVA jobs 6895891 and 6895892 plus
  the workers already running). Predictions are in its `ideas.md`.
- 2026-11-12 19:02 PT: the shared resource file was regenerated from the live scheduler (shared repository commit
  0c179c7): affogato13, lynx01, lynx02 and lynx05 are now listed with RTX 4000 Ada cards; its self-check and the
  contract test pass. The human-readable resource document is not regenerated yet.
- 2026-11-12 19:00 PT: this folder's new monitor job 6894373 is running (on heartpiece) since the memory request
  was lowered.
- 2026-11-12 19:15 PT: **AntMaze physics: cause found and setting decided.** The integrator (`implicitfast` at
  0.01 s) adds rotational energy to fast-turning ants in the air; more solver iterations do not help; RK4 and a
  0.005 s time step remove it. Decision: 0.005 s with 10 physics steps per environment step, plus the guard at
  30 m/s and 100 rad/s; cost 1.97 times per iteration. Details and numbers in the AntMaze folder's
  `infra_history.md`. The old check is stopped; the time-step configuration field, its gates (now including a
  gate with trained policies) and the throughput of the final setting are being done in the background since
  19:17 PT. Then: freeze, rebuild the rollout-shape check under a new phase name, first unit by hand.
- 2026-11-12 19:22 PT (report regenerated): **RND's discovery baseline is confirmed in all four PointMaze cases** (1,024 fresh seeds;
  mean capped first-goal time in thousand steps): Medium with reward 1 on arrival 17.4 ± 0.5; Medium with the
  step cost 14.1 ± 0.3; Large with reward 1 on arrival 136 ± 3; Large with the step cost 131 ± 3 (learning rate
  4.5e-3, weight 8; RND at CleanRL's defaults 2,723 ± 179, PPO without a bonus 10,418 ± 467). The PointMaze
  baseline folder has no unit left (133 done).
- 2026-11-12 19:22 PT (the reading ran in the minute after the script was written): **batch 4, first three units (16 development seeds, against RND's sweeps at the budget on
  the same seeds; quick script `quick_compare_dev.py`, not yet in the summary code).**
  1. PointMaze Medium, reward 1 on arrival, slower predictor with held divisor: best whole-curve mean 0.965
     (learning rate 6e-3, weight 0.5) against RND's best 0.9435 (learning rate 1e-2, weight 2): +0.021
     [+0.008, +0.036]; final level 1.000 against 0.9935.
  2. Same case, all three parts: best 0.946 (learning rate 1e-2, weight 0.25): +0.002 [−0.017, +0.021]; its best
     is at the highest learning rate and the lowest weight of its grid, so the edge rule applies.
  3. PointMaze Medium, step cost, slower predictor with held divisor: best −0.3422 (learning rate 1e-2, weight
     0.00195) against RND's −0.3405: −0.002 [−0.010, +0.007]. Every configuration of the method lies between
     −0.342 and −0.368; no weight is ahead. Prediction 2 of the batch holds so far: a bonus-only change is level
     with RND's nearly bonus-free configuration there, not ahead.
  First reading: the finalist with two parts is at least as good as the one with three outside PointMaze Large.
  On discovery in Medium with reward 1 on arrival both are behind RND's best for that measure (11 to 13 thousand
  steps against 8.7), but these sweeps were not aimed at discovery (weights up to 4 only).
- 2026-11-12 19:21 PT: AntMaze folder: the fleet step had started the two stopped old units again; stopped for
  good (server marked not ready, jobs cancelled by ledger id, units set aside). Details in its `infra_history.md`.
- 2026-11-12 19:44 PT: **discovery check on the validation block read (256 seeds, PointMaze Large, reward 1 on
  arrival; mean capped first-goal time in thousand steps).** The six idea units were at 241 to 249 of 256 windows,
  but every policy had already reached the goal, so their first-goal times are final.
  RND: 119 ± 6 (learning rate 6e-3, entropy 0.01, weight 64); 135 ± 6 (1e-2, 0.01, 64; 136 ± 3 on the fresh
  seeds); 245 ± 46 (1e-3, 0.03, 256). Fast last layer at step 0.1: 56 ± 2 (6e-3, 0.03, 16); 66 ± 3; 78 ± 5.
  All three parts with the divisor held after 1/8: 65 ± 3 (6e-3, 0.03, 16); 73 ± 3; 76 ± 3.
  Each one's best against RND's best, paired: fast last layer −63 [−75, −52]; all three parts −54 [−66, −43].
  So the fast last layer reaches the goal first in under half of RND's steps in this case, with the interval far
  from zero. Each side's best is the best of three on these seeds. At these configurations nobody optimizes
  (whole-curve means of 0.00 to 0.04), which is what the separate discovery table is for.
- 2026-11-12 19:43 PT: queued on the validation block (11 units, UVA jobs 6897191 to 6897213): PointMaze Medium
  with reward 1 on arrival, the three best development configurations of RND and of each finalist; and RND's
  second and fourth best development configurations on PointMaze Large (learning rates 1e-3 and 3e-3 at weight
  2), so the baseline gets the same second selection step as the methods.
- 2026-11-12 19:41 PT: three units of the RND-grid discovery sweep stopped with their state saved when their
  workers neared the 2-hour limit; they continue from their checkpoints with the next workers.
- 2026-11-12 20:01 PT: **AntMaze: stable physics recorded and gated (commit f490e70d); the rollout-shape check
  runs again** under the phase `rollout-shape-check-stable-physics` (18 units; first unit on Gilbreth, job
  12347248, to be read by hand before the other 17 go). Gate with trained policies: 0 fault respawns in 229
  million steps, largest torso speed 14.8 m/s. Throughput on an H100 NVL: 2.72 s per iteration at 16,384
  environments, 0.77 million steps per second. Seen in the stopped check's records (old physics, not a result):
  with 128 environments per policy the best configurations reached the goal in only about 4% of episodes after
  16.8 million steps, while the units with 16 and 4 environments per policy had their first arrivals 4 to 10 times
  earlier in steps; so the count of environments per policy is likely to matter a great deal for the ant.

## Next

1. Batch 1 screen: read each unit as it finishes (whole-curve mean, final level, median first arrival, the two
   property measures); write `batches/01_.../summary.md`; ideas ahead of the control go to the development sweep.
2. RND baseline: confirmation of PointMaze Large with reward 1 on arrival and of Medium with the step cost
   (running); Medium with reward 1 on arrival after its higher learning rates; Large with the step cost at
   33,554,432 steps (development at the budget, then confirmation).
3. AntMaze with the Warp physics: rollout-shape check, first-pass sweep, budgets (H100 work: MIT, then Purdue, then
   the ACCESS servers).
4. Purdue fleet for the PointMaze units (Gautschi L40S, Gilbreth standby cards).
5. The remaining speed ideas and the copies per card for the production workloads.

## Ideas (status)

| batch | idea | status |
|---|---|---|
| 1 | separate predictor step size (0.25, 1, 4 times 1e-4) | within the control's interval on 16 seeds; not pursued alone; slower values are in batch 2 |
| 1 | fast last layer inside the rollout (steps 0.1, 0.3, 1) | earlier arrivals at steps 0.1 and 0.3, lower final level; within or behind; kept in mind for a combination with a held divisor |
| 1 | error length as the bonus | behind (−0.040) |
| 1 | fixed scale; even start; even start with no divisor | far behind; closed |
| 1 | scheduled last layer on the random body's features (decay 0.5, 0.75, 1) | far behind; the fit is not local and the input statistics move; local features are in batch 2 |
| 2 | wider input clip; input statistics kept from the random-action rollouts; slower predictor; local features as the input of RND; scheduled last layer on local features; divisor held after a part of the run | written 2026-11-12; implementation running |

## Decisions I made (routine), with the reason

- PointMaze step limit 400 (the earlier runs' value), AntMaze 700 / 1,000 / 1,000 (Gymnasium's).
- Reward setting B is applied in the trainer as setting A's reward mapped by $(r - 1)/H$ per copy, so both settings
  can share one process on the same seeds.
- The copy count is the smallest power of two at or above 4.5 × A ÷ B (the plan said "largest below", which would
  halve every count and disagrees with the doubling rule's own outcome on the 2026-11-10 series).
- Every proxy factor in the skill's `card_facts.json` for cards outside the UVA survey is a guess until that card's
  first two real units correct it.
- The ant uses MuJoCo's Warp implementation, because it passes the environment's gates with better agreement with
  C MuJoCo than the JAX implementation and is about 8 times faster in total (user, 2026-11-11: another simulator is
  allowed after a time breakdown).
- PointMaze workers ask for 2 processors and 4 GiB per card, from the measured need.
