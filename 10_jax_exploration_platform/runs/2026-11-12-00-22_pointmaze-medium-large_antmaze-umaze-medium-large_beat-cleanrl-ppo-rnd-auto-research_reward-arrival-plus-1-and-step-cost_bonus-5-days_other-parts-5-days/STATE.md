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
- 2026-11-12 20:02 PT: **PointMaze Medium with reward 1 on arrival, validation block (256 seeds): the gain seen on
  the 16 development seeds did not hold.** RND's best of three: whole-curve mean 0.9349 (learning rate 6e-3,
  weight 2, the confirmed configuration; 0.932 on the fresh seeds). Slower predictor with held divisor, best of
  three: 0.9348 (learning rate 3e-3, weight 2), difference −0.0001 [−0.008, +0.008]; its final level is ahead
  (+0.005; 0.9999 against 0.995), and at these configurations it reaches the goal 7 thousand steps sooner (18
  against 25). Its development-best configuration (learning rate 6e-3, weight 0.5; 0.965 on 16 seeds, "+0.021")
  gives 0.928 here. All three parts: 0.906 at its two finished configurations, behind RND by 0.029
  [−0.040, −0.018].
  What this means: (a) a difference of 0.02 on 16 seeds in this case is noise, so every claim goes through the
  validation block first; (b) in this case the method reaches the goal sooner and ends higher but its whole-curve
  mean is only level, so the time between the first arrival and a settled policy is what is left to gain; (c) the
  finalist with two parts is the better of the two outside PointMaze Large.
  Standing of the two-part finalist on the whole-curve mean: PointMaze Large with reward 1 on arrival ahead
  (+0.010 [+0.005, +0.016], validation block); Medium with reward 1 on arrival level; Medium with the step cost
  level (16 seeds); Large with the step cost running.
- 2026-11-12 20:03 PT: the fast last layer on RND's discovery grid is complete (102 configurations with the
  repeats of its first sweep, 16 development seeds): best mean capped first-goal time 43.6 ± 6.9 thousand steps
  (learning rate 2e-3, entropy 0.1, weight 256), then 46.2 (2e-3, 0.01, 16). On the validation seeds its best of
  three was 55.8 ± 2.2 against RND's 119 ± 6, so the wider grid adds little beyond what was already checked.
- 2026-11-12 20:02 PT: **the batch summaries now carry these comparisons** (analysis work in the background,
  commits 82196cbe, 77c651f7, a5fafeba, 73087778; 104 analysis tests pass): each method at its best weight against
  the control at its best on the validation seeds; batch 4 case by case against RND's development sweeps with
  grid-end flags; the discovery checks; the readings of the predictions of batches 2 to 4. From the prediction
  readings: batch 4's prediction 1 did not hold (neither finalist reaches the goal sooner than RND's own best for
  that measure in PointMaze Medium), prediction 2 held (Medium with the step cost is level), prediction 3 waits.
- 2026-11-12 20:28 PT: **AntMaze check with the stable physics: first unit passed its check by hand** (0 fault
  respawns, largest torso speed 13.0 m/s, all values finite, settings as intended), and the other 17 units were
  handed to Gilbreth (17 jobs waiting on priority at 20:28 PT). Speed on an A30: 1.09 s per iteration at 512
  environments; a unit with 4 environments per policy takes about 10 hours. Details in the AntMaze folder's
  `infra_history.md`.
- 2026-11-12 20:21 PT: validation block, PointMaze Large with reward 1 on arrival, with RND's two further
  configurations: learning rate 3e-3 gives 0.8701 and 1e-3 gives 0.8695, both below 0.8781 at 2e-3 (weight 2). So
  RND's best of five on the validation seeds is unchanged and the finalists' differences stand (all three parts
  +0.0137 [+0.007, +0.020]; slower predictor with held divisor +0.0102 [+0.005, +0.016]).
  PointMaze Medium with reward 1 on arrival, all three parts, third configuration: 0.893; its best of three stays
  0.906, behind RND's 0.9349 by 0.029 [−0.040, −0.018].
- 2026-11-12 21:01 PT: **batch 4, PointMaze Large with the step cost, slower predictor with held divisor (16
  development seeds; the unit finished on Gilbreth from its UVA checkpoint):** best whole-curve mean −0.722
  (learning rate 3e-3, weight 8) against RND's −0.725: +0.003 [−0.004, +0.007], level; final level −0.703 against
  −0.712: +0.008 [+0.006, +0.012]. Its best weight is the largest of its grid, so weights 16 and 32 are queued
  (edge rule). RND's same configuration has −0.732 ± 0.001 on the 1,024 confirmation seeds, so its development
  value is 0.007 too high. The unit of all three parts in that case is at 101 of 256 windows and continues on
  Gilbreth.
  Standing of the slower predictor with held divisor on the whole-curve mean: ahead in PointMaze Large with
  reward 1 on arrival (validation block); level in the other three PointMaze cases. Its final level is ahead in
  all four.
- 2026-11-12 21:03 PT: **where it loses on PointMaze Medium with reward 1 on arrival** (mean training return of the
  256 validation policies): ahead of RND until about 50 thousand steps (0.382 against 0.279 at 33 thousand),
  behind from 66 to 200 thousand (0.715 against 0.816 at 98 thousand), level after. In that stretch the intrinsic
  reward the learner sees is 30 to 60 times RND's (0.18 against 0.006 at 98 thousand steps): the slower predictor
  keeps the error high, and the divisor is frozen only at 66 thousand steps, after it has shrunk with the bonus.
  **Batch 5 queued from this** (`batches/05_bonus-that-falls-away-sooner-…`, 4 units on Gilbreth, existing
  options only): held divisor alone (after 1/16 and after 1/64 of the run), slower predictor with the divisor
  held after 1/64, predictor at half speed with held divisor. Predictions in its `ideas.md`.
- 2026-11-12 21:02 PT: **AntMaze check, early look at 1,048,576 steps per policy (not a result; the units are at
  5 to 66 of 256 windows):** policies that had reached the goal by then: 707 of 768 with 16 environments per
  policy; 112 of 128 in the one unit with 4 environments per policy that has got that far; 1 of 768 with 128
  environments per policy. The count of environments per policy decides how soon the ant finds the goal, as the
  stopped check's records suggested.
- 2026-11-12 21:04 PT: all of this folder's units now run on Gilbreth (6: batch 5, the larger weights, and the
  unit of all three parts on PointMaze Large with the step cost, which left UVA when its worker ended). With the
  18 AntMaze units that makes 24 jobs there, the account's limit for idle-card jobs.
- 2026-11-12 21:21 PT: **batch 5, first two arms (16 development seeds, PointMaze Medium with reward 1 on arrival;
  both units ran on Gilbreth in under 15 minutes):** held divisor alone, held after 1/16 of the run: best
  whole-curve mean 0.962 (learning rate 6e-3, weight 2); held after 1/64: 0.964 (same configuration). RND's best on
  these seeds is 0.9435 (0.9421 at that same configuration), so both read +0.02 here. On these seeds the two-part
  finalist read +0.021 and the validation block then gave 0.000, so this is not a result yet: the two best
  configurations of each arm were queued on the validation block (4 units). The arms with the slower predictor
  (divisor held after 1/64) and with the predictor at half speed are running or waiting on Gilbreth.
- 2026-11-12 21:41 PT: batch 5's other two arms on the 16 development seeds (PointMaze Medium, reward 1 on
  arrival): slower predictor with the divisor held after 1/64: best 0.956 (learning rate 1e-2, weight 1);
  predictor at half speed with held divisor: 0.959 (6e-3, weight 1). All four arms read 0.956 to 0.964 here
  against RND's 0.9435, which is inside this case's noise on 16 seeds; the two best configurations of each arm are
  on the validation block (8 units in all, on Gilbreth), and the batch is judged there.
- 2026-11-12 22:01 PT: **batch 5 on the validation block (256 seeds, PointMaze Medium with reward 1 on
  arrival).** Each arm at the better of its two configurations against RND's best of three (0.9349): predictor at
  half speed with held divisor 0.9438, +0.0089 [+0.0012, +0.0166]; held divisor alone after 1/64: +0.0039
  [−0.004, +0.011]; after 1/16: +0.0033 [−0.004, +0.011]; slower predictor with the divisor held after 1/64:
  +0.0013; the two-part finalist: −0.0001. Every arm is ahead on the final level by 0.004 to 0.005. One arm of
  five has its interval above zero with a lower end of +0.001, and it is the best of five on these seeds, so it
  goes to a second validation block (seeds 20000 to 20255) before it counts.
  Readings of batch 5's predictions (quick script; the summary code has no settings for this batch yet):
  1 held (held divisor alone is level with RND within 0.01 and ahead on the final level); 2 did not hold for the
  whole-curve mean (holding the divisor earlier does not put the slower predictor ahead: +0.0013); 3 did not hold
  (the predictor at half speed is not between RND and the finalist: it is above both).
- 2026-11-12 22:01 PT: **batch 4, PointMaze Large with the step cost, 16 development seeds:** all three parts: best
  −0.7187 (learning rate 4.5e-3, weight 0.5), +0.0060 [+0.0044, +0.0075] against RND's −0.7247, final level
  +0.0091 [+0.006, +0.012]; slower predictor with held divisor with the larger weights: best stays −0.7222 at
  weight 8 (16: −0.723; 32: −0.737), so its best weight is now inside its grid; +0.0025 [−0.004, +0.007].
  All of this folder's 121 units were done at 22:00 PT.
- 2026-11-12 22:02 PT: **batch 6 queued** (`batches/06_predictor-at-half-speed-…`, 18 units on Gilbreth): a second
  validation block on PointMaze Medium (RND at three configurations, the predictor at half speed with held
  divisor at two, the held divisor alone at two); the predictor at half speed with held divisor on PointMaze Large
  with reward 1 on arrival (weights 1, 2, 4); and PointMaze Large with the step cost on the first validation block
  (RND, all three parts, the two-part finalist and the predictor at half speed, two configurations each).
  Predictions in its `ideas.md`.
  Open: `summary_settings.json` for batches 5 and 6, so their summaries are generated like batch 4's.
- 2026-11-12 22:18 PT: **placement decided with the user: Gilbreth by default; UVA only for workers that start
  at once (any suitable card type); no worker jobs left waiting in UVA's queue.** Reasons and what was done are
  in `infra_history.md`; the rule is in the session's memory (note `pointmaze-antmaze-placement-gilbreth-first`)
  and in the 20-minute monitor's prompt (cron 9beed1f1 replaced 0a3e2261). At 22:18 PT this folder had 14 units on
  Gilbreth and 2 running on UVA.
- 2026-11-12 22:46 PT: **batch 6, second validation block (seeds 20000 to 20255, used for no choice before),
  PointMaze Medium with reward 1 on arrival.** RND at the best of its three on this block: 0.9308. Arms at
  configurations fixed on the first block: held divisor alone after 1/64 (learning rate 1e-2, weight 2): 0.9417,
  +0.0109 [+0.0039, +0.0182]; predictor at half speed with held divisor (6e-3, weight 2): 0.9416, +0.0108
  [+0.0031, +0.0186], and at weight 1: 0.9410; held divisor alone after 1/16 (6e-3, weight 4): 0.9398, +0.0090
  [+0.0021, +0.0160]. Final level +0.009 for all three (1.000 against 0.991). **All three are ahead of RND with
  intervals above zero on seeds that chose nothing.** My prediction 1 (the gain shrinks and the interval reaches
  zero) did not hold.
- 2026-11-12 22:46 PT: batch 6, PointMaze Large with reward 1 on arrival, predictor at half speed with held
  divisor (first validation block): best at weight 4: 0.8849, +0.0068 [+0.0005, +0.0128] against RND's 0.8781;
  weight 2: 0.8821; weight 1: 0.8597. Prediction 2 held (within 0.005 of the slower predictor with held divisor,
  and ahead of RND).
- **Reading of batches 4 to 6 together: the part that is ahead of RND on the whole-curve mean in both PointMaze
  cases with reward 1 on arrival is the held divisor.** Alone it gives +0.0095 [+0.004, +0.015] on Large (first
  block) and +0.009 to +0.011 on Medium (second block). Slowing the predictor adds nothing consistent to that
  measure (0.25 times: +0.0102 on Large, −0.0001 on Medium; 0.5 times: +0.0068 on Large, +0.0108 on Medium); it
  shortens the first-goal time. For discovery alone the fast last layer leads (56 against RND's 119 thousand
  steps on Large). Why the held divisor helps: RND divides the bonus by the running standard deviation of its
  discounted sum, so as the bonus decays the divisor shrinks with it and the learner keeps seeing an intrinsic
  reward of about the same size; holding the divisor lets the bonus really fall away, and the final level rises
  from about 0.99 to 1.00.
- 2026-11-12 22:48 PT: **batch 7 queued** (`batches/07_held-divisor-alone_in-the-two-step-cost-cases`, 11 units on
  the first validation block): the held divisor alone in PointMaze Medium and Large with the step cost, with RND
  and the half-speed arm beside it on Medium. Predictions in its `ideas.md`. 18 of this folder's units are on
  Gilbreth (8 of batch 6 running there since 22:20 PT, at 25 to 100 of 256 windows at 22:46 PT) and 1 waits for
  the one UVA card that could start a worker (lynx04).
  Open before any finalist: `literature/novelty.md` for the held divisor (closest published work and the exact
  difference), and `summary_settings.json` for batches 5 to 7.
- 2026-11-12 23:06 PT: **batch 7, PointMaze Medium with the step cost (first validation block, 256 seeds).** RND's
  best of three: −0.3505 (learning rate 6e-3, weight 0.00195). Held divisor alone: −0.3499 at weight 0.03125,
  +0.0006 [−0.004, +0.005], level; −0.3518 at 0.00195; −0.3534 at 0.0078. Predictor at half speed with held
  divisor: −0.3539, −0.0034 [−0.008, +0.002]. Prediction 2: level, as predicted; but its second part did not hold:
  the largest weight run (0.03125, sixteen times RND's) is the held divisor's best, not behind. So the held
  divisor makes a real bonus usable where RND needs it nearly off; weights 0.125, 0.5 and 2 are queued (edge
  rule; 5 units).
  PointMaze Large with the step cost: RND's confirmed configuration gives −0.7310 ± 0.0018 on the validation
  block (−0.732 on the fresh seeds); the other 9 units of that case are at 3 to 157 of 256 windows.
- 2026-11-12 23:07 PT: **AntMaze check, the six units with 128 environments per policy are complete (stable
  physics, 16 seeds, 16,777,216 steps per policy).** Best whole-curve mean 0.049 ± 0.001 (learning rate 1e-2,
  entropy 0.001, weight 16), final level 0.065: at the end only about 6% of episodes reach the goal. Every one of
  the eight best configurations has the largest learning rate of the grid. First arrivals at about 1.9 million
  steps. Fault guard: 32 respawns in 12.9 billion environment steps (2.5 per billion); largest torso speed 16.9
  m/s; 7.6% of steps with the torso above 1.5 m. So with 128 environments per policy RND barely learns the U-maze
  in this budget; the units with 16 environments per policy (225 to 228 of 256 windows) and with 4 (67 to 82)
  will say whether more updates per step change that.
- 2026-11-12 23:27 PT: **held divisor alone, PointMaze Medium with the step cost, larger weights (first validation
  block):** weight 2: −0.3463 ± 0.0012 against RND's best −0.3505: +0.0042 [+0.0006, +0.0079], first-goal time
  −6.1 thousand [−8.7, −4.1] (13.9 against 20.0); weight 0.5: −0.3495; 0.125: −0.3518. Its best is again the
  largest weight run, a thousand times RND's, so 4, 8 and 16 are queued, and a second block (seeds 20000 to
  20255) with RND at its three configurations and the held divisor at weights 2, 4 and 8 fixed beforehand,
  because −0.3463 is the best of eight configurations on the first block.
- 2026-11-12 23:27 PT: **PointMaze Large with the step cost, first validation block, slower predictor with held
  divisor:** −0.7230 (learning rate 3e-3, weight 8) against RND's best of two −0.7310: +0.0081 [+0.0033, +0.0130];
  final level +0.0105 [+0.006, +0.015]; first-goal time −64 thousand. The held divisor alone, the half-speed arm
  and all three parts are still running there (73 to 249 of 256 windows at 23:26 PT).
  **Standing on the whole-curve mean, validation seeds:** an arm with the held divisor is ahead of RND in each of
  the four PointMaze cases: Large with reward 1 on arrival (held alone +0.0095), Medium with reward 1 on arrival
  (held alone +0.009 to +0.011, second block), Large with the step cost (slower predictor with held divisor
  +0.0081; held alone pending), Medium with the step cost (held alone at weight 2 +0.0042; second block queued).
- 2026-11-12 23:28 PT: **AntMaze check: 128 and 16 environments per policy are complete; 4 is at 78 to 92 of 256
  windows.** Best whole-curve mean at 16,777,216 steps: 0.0487 ± 0.0014 with 128 (learning rate 1e-2, weight 16)
  and 0.0487 ± 0.0027 with 16 (learning rate 3e-4, weight 0.0625). At the common point of 4,980,736 steps: 0.029
  with 4, 0.029 with 16, 0.033 with 128. First arrivals: about 0.3 million steps with 4 and 16, 1.9 million with
  128. Fault guard: 43 and 32 respawns in 12.9 billion steps each.
  **RND does not learn to reach the ant U-maze goal in this budget with any count:** in the best configurations
  the share of episodes that reach the goal rises slowly (1.4%, 3.8%, 5.1%, 5.9%, 6.3% over the run) and is 3% to
  10% for every seed at the end; arrivals still take about 530 of the 700 steps; the ants spend 12% to 14% of
  their steps above 1.5 m, that is they throw themselves about and sometimes land at the goal. The best
  configuration with 16 environments has a bonus weight of 0.0625, so the bonus hardly matters for this measure.
  A longer run was queued (phase `budget-probe` in the AntMaze folder: 8 times the steps with 128 environments
  per policy and 4 times with 16, learning rates up to 3e-2) to see whether the share keeps rising.
  Open for the user: what the ant comparison should be if RND stays at a few percent (the plan's Part 5, item 2).
- 2026-11-12 23:47 PT: **held divisor alone, PointMaze Medium with the step cost.**
  First block, by weight (learning rate 6e-3): 0.5: −0.3495; 2: −0.3463; 4: −0.3520; 8: −0.3667; 16: −0.3935. Its
  best weight (2) is now inside its grid.
  Second block (seeds 20000 to 20255, held divisor at weights 2, 4, 8 fixed beforehand): weight 2: −0.3477 ±
  0.0012. Against RND at the configuration chosen on the first block (weight 0.00195: −0.3535), which is the rule
  the arm was held to: +0.0058 [+0.0009, +0.0114]. Against RND at its best of three on this second block (weight
  0.0078: −0.3510), which favours RND: +0.0033 [−0.0003, +0.0069], an interval that just reaches zero. First-goal
  time −5.3 thousand steps [−7.2, −3.5] against that same row. Reading: ahead under equal rules, by a small
  margin; not ahead beyond doubt when RND is given the choice on the block.
- 2026-11-12 23:47 PT: **PointMaze Large with the step cost, first block, predictor at half speed with held
  divisor** (learning rate 3e-3, weight 8): −0.7216, +0.0094 [+0.0050, +0.0140] against RND's best of two
  (−0.7310); final level +0.0111; first-goal time −59 thousand. The slower predictor with held divisor has +0.0081
  there. The held divisor alone and all three parts are at 143 to 244 of 256 windows.
- 2026-11-13 00:06 PT: **PointMaze Large with the step cost, first validation block (9 of 11 units complete; RND's
  best of two −0.7310):** predictor at half speed with held divisor (learning rate 3e-3, weight 8): −0.7216,
  +0.0094 [+0.0050, +0.0141]; slower predictor with held divisor (3e-3, weight 8): −0.7230, +0.0081
  [+0.0033, +0.0132]; held divisor alone (2e-3, weight 4): −0.7255, +0.0055 [−0.0003, +0.0111], and (3e-3, weight
  8): −0.7270; all three parts (4.5e-3, weight 0.5): −0.7320, −0.0010 [−0.008, +0.006], so its +0.006 on the 16
  development seeds did not hold. Batch 6's prediction 3 did not hold (all three parts is level, the two-part arm
  is ahead); batch 7's prediction 1 (held divisor alone ahead with its interval above zero) is not met so far:
  the interval's lower end is −0.0003.
  **Standing by arm on the whole-curve mean (validation seeds):**

  | arm | Large, reward 1 on arrival | Medium, reward 1 on arrival | Large, step cost | Medium, step cost |
  |---|---|---|---|---|
  | held divisor alone | ahead, +0.0095 | ahead, +0.009 (second block) | +0.0055, interval reaches zero | ahead under equal rules, +0.0058 (second block); +0.0033 reaching zero when RND chooses on the block |
  | predictor at half speed with held divisor | ahead, +0.0068 | ahead, +0.0108 (second block) | ahead, +0.0094 | only run with RND's tiny weights (level); weights 1, 2, 4 queued |
  | slower predictor with held divisor | ahead, +0.0102 | level, −0.0001 | ahead, +0.0081 | level on 16 seeds with tiny weights |
  | all three parts | ahead, +0.0137 | behind, −0.029 | level, −0.0010 | level on 16 seeds |

  The half-speed arm is the one that could be ahead everywhere; its missing cell is queued (3 units).
- 2026-11-13 00:08 PT (the note itself is dated 00:03 PT): **literature note written (`literature/novelty.md`, background search of about 30 queries
  and 60 pages; not a systematic review). Both ideas are "close" to public work.**
  1. Held divisor: no paper was found that proposes or evaluates stopping the updates of RND's reward divisor
     part-way through training. But (a) Schwarke, Klemm, van der Boon, Bjelonic and Hutter (Conference on Robot
     Learning 2023) report that RND's running reward normalisation keeps the intrinsic reward from falling, and
     run RND without it at a fixed weight; (b) the RND module of the RSL-RL library builds its reward normaliser
     with `until=int(1.0e8)`, that is, it stops updating after 100 million samples (I opened that file and
     checked the line; the paper's title, authors and venue I checked on the proceedings page, its sentence on the
     normalisation I did not check myself). The difference left: the hold after a set fraction of the run inside
     CleanRL's PPO-RND, and an evaluation of it. In batch 1 the arm with no divisor at all was not ahead of RND
     here, so the time of the hold matters in these mazes.
  2. Fast last layer inside the rollout: no work was found where the last layer of an RND predictor takes a
     gradient step at every environment step of an on-policy rollout with the usual epoch updates kept. Updating a
     novelty estimate at every step is published in several forms (elliptical episodic bonuses, neural density
     models; a 2026 arXiv paper trains the whole RND predictor one step per input as a comparison method).
  **Consequence (my reading; the bar itself is the user's):** the held divisor is known practice in public code
  and should not be presented as the study's new method. It can be used and credited, and it raises the question
  whether "beat CleanRL RND" should also be shown against RND with a held divisor. The part of the work that is
  new so far is the fast last layer, which wins on discovery and not on the whole-curve mean. The slowed
  predictor is a setting, not an idea.
- 2026-11-13 00:23 PT: **the summaries of batches 3 to 7 and `analysis/standing.md` are generated** (analysis work
  in the background, commits de35b926, 3d77556a, 4e0f5aa5; 111 analysis tests pass). From the standing page, on the
  validation seeds: **the held divisor alone is ahead of RND on the whole-curve mean in all four PointMaze
  cases** (Medium with reward 1 on arrival +0.0109 and Medium with the step cost +0.0058, both on the second
  block under equal rules; Large with reward 1 on arrival +0.0095 and Large with the step cost +0.0095
  [+0.0057, +0.0137] at weight 8, first block). The half-speed arm is level on Medium with the step cost (+0.0014
  at weights 1, 2, 4), so it is not ahead everywhere. All 167 units of this folder were done at 00:26 PT.
- 2026-11-13 00:28 PT: **batch 8 written** (`batches/08_divisor-held-from-the-first-arrival_…`): (1) the divisor
  held from each policy's own first arrival, a new option being added and tested in the background (no fraction
  to set; not what the public code does); (2) the error length with a held divisor, existing options, 6 units on
  the validation block handed to Gilbreth now. Predictions in its `ideas.md`. This folder's code is frozen anew
  when the new option's tests pass; no unit of it is part-way through then except these 6 short ones.
  Waiting for the user: how strict to be about the held divisor's closeness to public practice, and what the
  AntMaze comparison should be if RND stays at a few percent.
- 2026-11-13 00:47 PT: **batch 8, the error length with a held divisor (first validation block, weights 1, 2, 4).**
  Far behind on the whole-curve mean: PointMaze Large with reward 1 on arrival 0.656 at weight 1 (held divisor
  alone 0.8876; −0.232 [−0.239, −0.225]), 0.352 at weight 2, 0.119 at weight 4; PointMaze Medium 0.877 at weight 1
  (held divisor alone 0.9382; −0.062). First-goal time: 125 thousand steps on Large against 270 for the held
  divisor alone (−145 [−167, −124]); on Medium 23.8 against 21.0 (not sooner). Prediction 2 did not hold: it is
  sooner only on Large, and not level on the whole-curve mean at these weights. Its best weight is the smallest
  one run in both cases, so weights 0.125, 0.25 and 0.5 are queued (6 units, edge rule). Reading: the error
  length decays more slowly than the squared error, so at the same weight the policy keeps exploring; it is a
  discovery arm, like the fast last layer.
- 2026-11-13 01:07 PT: **batch 8, the error length with a held divisor at smaller weights (first validation
  block):** PointMaze Large with reward 1 on arrival: 0.8714 at weight 0.25 and 0.8695 at 0.125 (0.5 still
  running), against RND's 0.8781: −0.0066 [−0.012, −0.002]. PointMaze Medium: 0.9256 at weight 0.5, 0.9177 at
  0.25, 0.8794 at 0.125, against RND's 0.9349: −0.0093 [−0.017, −0.002]. Its best weight is now inside its grid on
  Medium. **Behind RND on the whole-curve mean at every weight; the arm is closed for that measure.**
- 2026-11-13 01:07 PT: **batch 9 queued** (`batches/09_fast-last-layer_discovery-in-the-other-three-pointmaze-cases`,
  27 units, phase `discovery-validation`, first validation block): the fast last layer and RND, each at its own
  configurations for the first-goal time, in PointMaze Medium with both rewards and PointMaze Large with the
  step cost. Predictions in its `ideas.md`.
- 2026-11-13 01:25 PT: **the option `reward_divisor=held_after_first_arrival` is in the platform** (background
  work, commit eab85eba; 5 new tests, the golden, bonus, agent, integration, sweep-machinery and Warp training
  suites pass; existing options give the same results bit for bit). How it works: each policy's divisor is
  updated as in RND through the rollout that contains its first arrival, and is held from the next rollout on; a
  policy that never arrives is exactly RND. The trainer reads the first-arrival step from the episode log it
  already keeps. Card check on serval02 card 1 (00:49 to 00:54 PT, PointMaze Medium, 256 policies): every policy
  arrived (first-arrival step: smallest 1,300, median 16,306, largest 169,568) and held within its arrival's
  rollout (at most 508 steps later); 0.01558 s per iteration against 0.01560 s for the fraction hold, so no
  slowdown.
  **Its units are built only after this folder's jobs of the old code copy have ended on Gilbreth**, because a
  worker does not check which code copy a unit needs: an old-copy job that took a unit with the new option would
  fail it. Batch 9's 27 units run under the old copy now; then: freeze the code anew, build batch 8's first arm
  (the divisor held from the first arrival) on the first validation block in the four PointMaze cases, hand over.
- 2026-11-13 01:27 PT: **batch 9, the two PointMaze Medium cases (discovery check, 256 validation seeds; mean
  capped first-goal time in thousand steps, each side at its best configuration on the block).** Reward 1 on
  arrival: fast last layer 12.5 ± 0.6 (learning rate 1e-2, entropy 0.03, weight 16; best of 8) against RND 17.5 ±
  1.4 (best of 3): −5.0 [−8.3, −2.3]. Step cost: fast last layer 11.4 ± 0.5 against RND 12.4 ± 0.5: −1.0
  [−2.3, +0.3], level. Prediction 2 held (a few thousand steps at most, interval below zero in one of the two).
  The five units on PointMaze Large with the step cost run on Gilbreth (11 to 66 of 256 windows at 01:26 PT).
- 2026-11-13 01:30 PT: **batch 8's main arm is running: the divisor held from the first arrival, 12 units on the
  first validation block in the four PointMaze cases** (code frozen at bed45842). Six UVA workers started at once
  and took the six PointMaze Medium units; the six PointMaze Large units follow on the same workers. Nothing of
  this arm goes to Gilbreth until the five old-copy jobs there have ended.
- 2026-11-13 02:07 PT: **batch 8, the divisor held from the first arrival (first validation block, 256 seeds; 9 of
  12 units complete).**

  | case | its best whole-curve mean | against RND's best | against the hold after 1/16, same configuration |
  |---|---|---|---|
  | PointMaze Large, reward 1 on arrival | 0.8874 (weight 2) | +0.0094 [+0.0033, +0.0155] | −0.0001 [−0.006, +0.006] |
  | PointMaze Medium, reward 1 on arrival | 0.9435 (learning rate 6e-3, weight 4) | +0.0085 [+0.0009, +0.0160] | +0.0053 [−0.001, +0.011] at weight 4; −0.0040 [−0.011, +0.003] at weight 2 |
  | PointMaze Medium, step cost | −0.3476 (weight 1) | +0.0029 [−0.0016, +0.0074] | −0.0020 [−0.006, +0.001] at weight 2; +0.0016 at weight 4 |

  Reading: it behaves like the fixed hold (no difference outside its interval in any case), with nothing to set.
  Ahead of RND in the two cases with reward 1 on arrival; level on Medium with the step cost on this block (the
  fixed hold had +0.0042 there). Prediction 1: its first part held (level with the fixed hold on Large and ahead
  of RND); its second part did not (on Medium it is not ahead of the fixed hold beyond noise). The three units on
  PointMaze Large with the step cost run on UVA (61 to 82 of 256 windows at 02:06 PT); one Large unit at weight 4
  was at 249.
- 2026-11-13 02:07 PT: **batch 9, PointMaze Large with the step cost (discovery check).** RND's best of two: 120.3 ±
  5.6 thousand steps (learning rate 6e-3, weight 8). Fast last layer: 58.9 (6e-3, entropy 0.03, weight 16), 62.1
  and 76.4 thousand; best against RND's best −61 [−73, −50]. Its three units are still running on Gilbreth (62 to
  139 of 256 windows), but every policy has already arrived, so these first-goal times are final. Prediction 1
  held. **Discovery standing on the validation seeds: the fast last layer is ahead of RND in three PointMaze cases
  (Large with reward 1 on arrival −63, Large with the step cost −61, Medium with reward 1 on arrival −5.0) and
  level in the fourth (Medium with the step cost −1.0 [−2.3, +0.3]).**
- 2026-11-13 02:29 PT: first-arrival hold, PointMaze Large with reward 1 on arrival, all three weights complete
  (first block): 0.8874 at weight 2, 0.8852 at weight 4, 0.8680 at weight 1; its best weight is inside its grid.
  **Second block queued for it** (seeds 20000 to 20255, configurations fixed on the first block; 6 units, phase
  `validation-block-two`): PointMaze Medium with reward 1 on arrival (learning rate 6e-3, weight 4), Medium with
  the step cost (weights 1 and 2), and PointMaze Large with reward 1 on arrival together with RND and the fixed
  hold at weight 2, which had no second-block units yet. Three UVA jobs with four cards started at once (an A40
  and three RTX 2080 Ti: 6900604, 6900605, 6900607); two that were still waiting after two minutes were cancelled
  by their ledger ids (6900603, 6900606). PointMaze Large with the step cost follows when its first-block units
  are complete (102 to 141 of 256 windows at 02:26 PT).
- 2026-11-13 02:50 PT: **the divisor held from the first arrival on the second validation block (seeds 20000 to
  20255, configurations fixed on the first block), the two PointMaze Medium cases.**

  | case | first-arrival hold | RND at its first-block best | difference [95%] | against the fixed hold, same seeds |
  |---|---|---|---|---|
  | Medium, reward 1 on arrival | 0.9404 (learning rate 6e-3, weight 4) | 0.9308 | +0.0095 [+0.0015, +0.0178] | +0.0005 [−0.007, +0.008] |
  | Medium, step cost | −0.3470 (learning rate 6e-3, weight 1) | −0.3535 | +0.0065 [+0.0012, +0.0122] | +0.0007 [−0.003, +0.004] |

  Ahead of RND in both under equal rules (each side's configuration fixed on the first block); final level not
  behind in either (+0.0092 [+0.004, +0.018] and +0.0013 [−0.001, +0.004]). On Medium with the step cost, when RND
  is instead given its best configuration on the second block itself, the difference is +0.0040 [−0.0001,
  +0.0080], level; the fixed hold had the same pattern there (+0.0058 and +0.0033). On Medium with reward 1 on
  arrival RND's first-block best is also its best on the second block. Level with the fixed hold in both cases.
  The standing page (`analysis/standing.md`) now has the first-arrival hold as its own family. Still running: its
  second block on PointMaze Large with reward 1 on arrival (162 of 256 windows at 02:46 PT) and its first block on
  PointMaze Large with the step cost (142 to 201).
- 2026-11-13 03:02 PT: **batch 10 is running** (`batches/10_fast-last-layer-with-the-first-arrival-hold_pointmaze_and-ant-umaze-screen-of-the-bonus-arms/ideas.md`,
  seven predictions written before the runs).
  1. PointMaze: the fast last layer together with the first-arrival hold, two configurations in each of the four
     cases on the first validation block; one is paired with the hold alone and one with the fast last layer alone
     on the same seeds. Four units are on Gilbreth and four on UVA.
  2. Ant U-maze, the first screen of bonus arms on the ant: the first-arrival hold, the fast last layer, and both,
     with 128 and 16 environments per policy, 8 configurations and 16 development seeds per unit, against the
     rollout-shape check's RND units at the same configurations and seeds. The six units run in the AntMaze
     baseline folder under the phase `bonus-arms-screen` (routine decision: that folder holds the ant's server
     settings; its code was frozen anew at a2df3c12 for the new option).
  Why now: on the ant U-maze RND finds the goal in every policy (after 0.3 to 2.6 million steps) and does not keep
  it (whole-curve mean 0.04 to 0.05, final level 0.05 to 0.065, the same for intrinsic weights 1 to 64), which is
  where a bonus that falls away after the first arrival should matter most.
- 2026-11-13 03:10 PT: **correction: on the second validation block neither hold is ahead of RND on PointMaze Large
  with reward 1 on arrival.** Seeds 20000 to 20255, learning rate 2e-3, weight 2 for all three, fixed on the first
  block:

  | arm | whole-curve mean | difference from RND (0.8842) [95%] | final level against RND |
  |---|---|---|---|
  | divisor held from the first arrival | 0.8839 | −0.0003 [−0.0067, +0.0059] | +0.0048 [+0.0038, +0.0059] |
  | divisor held after 1/16 of the run | 0.8819 | −0.0023 [−0.0096, +0.0042] | +0.0042 [+0.0028, +0.0056] |

  The first block had +0.0094 and +0.0095 with intervals above zero. Between the blocks RND's mean rose by 0.006
  and the holds' fell by 0.004 to 0.006, each within about two standard errors. By the rule of this study (a
  best-of-several arm counts only after a second block) the lead on this case does not stand: **the holds are
  ahead of RND on the two PointMaze Medium cases (second block) and level on Large with reward 1 on arrival; Large
  with the step cost has the first block only** (fixed hold +0.0095 [+0.0057, +0.0137]; first-arrival hold +0.0074
  [+0.0020, +0.0128] with two of three units). The final level is ahead in every reading. My reading: on Large the
  hold's gain is a higher final level, worth about 0.003 to 0.005 of whole-curve mean, which 256 seeds cannot
  resolve; a larger gain there has to come from finding the goal sooner. Second block on Large with the step cost
  queued at 03:08 PT: RND (2e-3, weight 4), the fixed hold (2e-3, weight 8), the first-arrival hold (3e-3, weight
  8; its third first-block unit, 2e-3 with weight 4, was at 183 of 256 windows and is added if it turns out best).
- 2026-11-13 03:10 PT: **batch 10, first reading: PointMaze Medium with reward 1 on arrival, learning rate 6e-3,
  weight 4, first validation block.** The fast last layer with the first-arrival hold against the hold alone on
  the same seeds: whole-curve mean 0.8424 against 0.9435 (−0.101 [−0.115, −0.089]); final level 0.975 against
  1.000; first-goal time 15.7 against 18.5 thousand steps (−2.8 [−5.4, −0.3]). Against RND's best on the block:
  −0.093 on the whole-curve mean, −9.3 [−12.6, −6.2] thousand steps on the first-goal time. Prediction 3 holds for
  this case: the fast last layer finds the goal sooner and costs return after it. As written in the batch's ideas
  file, the next code change is a fast last layer that stops at each policy's first arrival
  (`fast_last_layer_until_first_arrival`); a background worker is adding and testing it (started 03:09 PT).
- 2026-11-13 03:28 PT: **batch 10, the two PointMaze Medium cases complete (4 units, first validation block, 256
  seeds).** The arm is the fast last layer with the first-arrival hold; differences are arm minus the other unit
  on the same seeds, with 95% intervals.

  | case and configuration | compared with | whole-curve mean | first-goal time, thousand steps |
  |---|---|---|---|
  | reward 1 on arrival, learning rate 1e-2, weight 4 | fast last layer alone | 0.776 against 0.379: +0.397 [+0.378, +0.417] | 14.7 against 13.8: +0.9 [−1.1, +2.9] |
  | reward 1 on arrival, learning rate 1e-2, weight 4 | RND, same configuration | −0.096 [−0.111, −0.081] | −2.9 [−5.4, −0.4] |
  | reward 1 on arrival, learning rate 6e-3, weight 4 | first-arrival hold alone | −0.101 [−0.115, −0.089] | −2.8 [−5.4, −0.3] |
  | step cost, learning rate 1e-2, weight 4 | fast last layer alone | −0.406 against −0.586: +0.180 [+0.170, +0.191] | 15.0 against 11.7: +3.3 [+0.3, +7.5] |
  | step cost, learning rate 1e-2, weight 4 | RND, same configuration | −0.038 [−0.045, −0.031] | +0.5 [−2.5, +4.5] |
  | step cost, learning rate 6e-3, weight 2 | first-arrival hold alone | −0.021 [−0.027, −0.015] | −1.3 [−2.9, +0.2] |

  Predictions: 2 holds (the hold gives back a large part of what the fast last layer costs, and the arm stays
  behind RND); 3 holds on both Medium cases (at the hold's own configurations the fast last layer lowers the
  whole-curve mean by 0.10 and 0.02); 1 holds on Medium with reward 1 on arrival. On Medium with the step cost the
  mean first-goal time is 3.3 thousand steps later with the interval just above zero; the median difference is
  +0.1 and one policy is 401 thousand steps later, so I read it as the tail of the same distribution.
- 2026-11-13 03:28 PT: **a fact about pairing by seed.** Two units of the same program before the first arrival
  (the arm above and the fast last layer alone, both on A30 cards) do not give the same numbers per seed: the
  first-goal step is equal in 44 of 256 seeds, and already in the first 4,096 steps the mean intrinsic reward is
  exactly equal in only 43 of 207 policies that have not arrived. On different card models it is equal in none.
  Rounding differences grow to visible differences within the first updates. So "the same seeds" pairs the start
  of two runs, not their paths, and a paired interval is hardly narrower than an unpaired one.
- 2026-11-13 03:47 PT: **first-arrival hold, first block on PointMaze Large with the step cost, all three units
  complete:** −0.7236 (learning rate 3e-3, weight 8), −0.7268 (2e-3, weight 8), −0.7335 (2e-3, weight 4); best
  against RND's best (−0.7310): +0.0074 [+0.0020, +0.0128]. The unit already running on the second block is the
  best one, so nothing is added there.
- 2026-11-13 03:47 PT: **batch 10, PointMaze Large with reward 1 on arrival, learning rate 2e-3, weight 16** (first
  block): the fast last layer with the hold against the fast last layer alone: whole-curve mean 0.5725 against
  0.036, first-goal time 86.1 against 77.7 thousand steps (+8.4 [−5.7, +22.8], level; prediction 1). Against
  RND's best for the whole-curve mean (2e-3, weight 2): −0.306 on the whole-curve mean, final level 0.85 against
  0.99, first-goal time 86 against 269 thousand steps (−183 [−208, −158]). So at weight 16 the goal is found three
  times sooner and then kept slowly. Whether that slowness comes from the fast last layer after the arrival or
  from the weight is what the option in progress separates.
- 2026-11-13 03:47 PT: **ant U-maze screen, a PARTIAL look** (records copied from Gilbreth at 03:47 PT; 16 development
  seeds; each arm against the check's RND unit over the same first windows; scratchpad script
  `quick_ant_partial.py`). Not a result yet.
  1. First-arrival hold: level with RND at both environment counts (differences in the whole-curve mean between
     −0.006 and +0.008 over the first 3.4 and 5.1 million steps). Prediction 4 is not holding so far.
  2. Fast last layer with 16 environments per policy: ahead of RND in all 8 configurations over the first 3.1
     million steps (+0.002 to +0.009 on a level of 0.015 to 0.025), with earlier first arrivals (0.25 to 0.55
     against 0.32 to 0.83 million steps). The arm with the hold added looks the same.
  3. Fast last layer with 128 environments per policy: behind RND in all 8 configurations (−0.003 to −0.017) and
     later first arrivals (2.2 to 4.1 against 1.9 to 2.75 million steps); the same with the hold added, at 9.8
     million steps.
  4. Fault-guard respawns so far: 0, 0, 0, 0, 1, 1 in the six units.
  If this holds at the end, the choice of environments per policy decides whether the fast last layer helps on
  the ant.
- 2026-11-13 03:55 PT: **the option `fast_last_layer_until_first_arrival` exists (commit 3abfe847) and batch 11 is
  running** (`batches/11_fast-last-layer-until-the-first-arrival_with-the-first-arrival-hold_four-pointmaze-cases/ideas.md`,
  five predictions written before the runs). The option: per policy, the fast last layer scores the bonus until
  the rollout that contains the policy's first arrival; from the next rollout the predictor alone scores it, as in
  RND. The fast last layer is a temporary copy of the predictor's output layer, so nothing is carried over at the
  switch and the saved state is unchanged. Tests: 7 new and 62 existing ones pass, and the golden comparison is
  identical. First seven units (the two Medium cases and Large with reward 1 on arrival, first validation block)
  run on seven UVA cards since 03:54 PT. Four more units (Large with the step cost, and the arm without the hold)
  follow when the four jobs of older code copies on Gilbreth have ended (`infra_history.md`).
- 2026-11-13 04:12 PT: **batch 11, the four PointMaze Medium units (first validation block): the arm with the
  first-arrival hold is far behind; prediction 2 failed.** Fast last layer until the first arrival, then the hold:

  | case and configuration | whole-curve mean | against the hold alone | first-goal time against the hold alone, thousand steps |
  |---|---|---|---|
  | reward 1 on arrival, learning rate 6e-3, weight 4 | 0.7001 | −0.243 [−0.255, −0.232] | −3.9 [−6.5, −1.5] |
  | step cost, learning rate 6e-3, weight 2 | −0.3821 | −0.034 [−0.038, −0.029] | −1.6 [−3.0, −0.2] |
  | step cost, learning rate 6e-3, weight 4 | −0.4307 | −0.080 [−0.086, −0.074] | −0.4 [−2.0, +1.1] |

  At (1e-2, weight 4) with reward 1 on arrival it is 0.5548. It is also behind batch 10's arm, the fast last layer
  throughout with the hold (0.7001 against 0.8424). The goal is still found sooner than by RND's best
  configuration for the whole-curve mean (14.6 against 25.0 thousand steps with reward 1 on arrival; 12.7 against
  20.0 with the step cost). What the records show: at the switch the reward the learner sees doubles, because the
  divisor was built on the fast scores and is then frozen, and afterwards it falls four to eight times more slowly
  than for the hold alone (table in the batch's ideas file, addendum 1). Two readings: the frozen divisor of the
  wrong scale, or a state of the policy left by the fast phase. **Seven units that tell them apart run on UVA
  since 04:10 PT** (existing options): the same arm with RND's running divisor (Medium and Large with reward 1 on
  arrival, 5 configurations) and with the divisor held after 1/16 of the run (2 configurations); predictions 6 to
  8 in the addendum. The batch's planned units on Large with the step cost are not run.
- 2026-11-13 04:30 PT: **second validation block on PointMaze Large with the step cost: both holds are ahead of RND**
  (seeds 20000 to 20255, configurations fixed on the first block; RND −0.7328 at learning rate 2e-3, weight 4).
  First-arrival hold (3e-3, weight 8): −0.7255, +0.0072 [+0.0013, +0.0130]; final level +0.0074 [+0.0011, +0.0135];
  first-goal time 152 against 203 thousand steps (−51 [−77, −24]). Fixed hold (2e-3, weight 8): −0.7232, +0.0095
  [+0.0051, +0.0142]; final level +0.0129. **Standing of the holds on second blocks: ahead of RND on Medium with
  reward 1 on arrival, Medium with the step cost (level there when RND chooses on the block) and Large with the
  step cost; level on Large with reward 1 on arrival.** Both cases with the step cost are ahead.
- 2026-11-13 04:30 PT: **batch 11, the units of addendum 1 on PointMaze Medium with reward 1 on arrival: prediction 6
  failed; the frozen divisor is not the cause.** Whole-curve mean, learning rate 6e-3 (256 validation seeds):

  | arm | weight 4 | weight 2 |
  |---|---|---|
  | RND | 0.9194 | 0.9349 |
  | fast last layer until the first arrival, then RND's running divisor | 0.4487 | 0.6403 |
  | fast last layer until the first arrival, divisor held after 1/16 of the run | 0.6610 | not run |
  | fast last layer until the first arrival, then the first-arrival hold | 0.7001 | not run |
  | fast last layer throughout, with the first-arrival hold (batch 10) | 0.8424 | not run |

  With RND's own divisor after the switch the arm is the worst of all, 0.47 behind RND at the same configuration.
  So the phase with the fast last layer leaves the learner in a state that settles slowly (reading (b) of the
  addendum). What the records show for learning rate 6e-3 and weight 4, by logging window counted from each
  policy's first arrival: 16 windows after it the share of episodes that reach the goal is 0.26 in both arms
  that stop the fast last layer and 0.80 for RND; the maze cells covered are 0.999 against 0.967; the logged
  intrinsic reward is 0.013 against 0.0065. The policies go on to cover the whole maze where RND's policies turn
  to the goal. Prediction 7 held (the divisor held after 1/16 is better than the running divisor, +0.21).
- 2026-11-13 04:30 PT: **batch 11 on PointMaze Large with reward 1 on arrival, the arm with the first-arrival hold**
  (learning rate 2e-3; RND's best 0.8781 with first-goal time 269 thousand steps): weight 2: 0.8496 (−0.028
  [−0.037, −0.021]), first-goal time 106; weight 4: 0.7871, 83; weight 16: 0.3370, 89. The goal is found 2.5 to
  3 times sooner and the return is behind at every weight; prediction 4 failed.
- 2026-11-13 04:30 PT: **ant U-maze screen, first complete unit** (fast last layer with the first-arrival hold, 128
  environments per policy, 16 development seeds, 16,777,216 steps): behind RND at the same configuration in all 8
  configurations (−0.008 to −0.018 on the whole-curve mean; best 0.0385 against RND's best 0.0487), first
  arrivals later (2.1 to 4.1 against 1.9 to 2.75 million steps); 7 fault-guard respawns in the unit.
- 2026-11-13 04:47 PT: **batch 10, PointMaze Large with reward 1 on arrival, learning rate 2e-3, weight 2** (the
  fast last layer throughout, with the first-arrival hold; first validation block): whole-curve mean 0.8707,
  against the hold alone −0.017 [−0.031, −0.005], against RND's best −0.0074 [−0.020, +0.0035] (level); final
  level 0.987 against RND's 0.992 (level); first-goal time 90.7 thousand steps against 269 for RND's best
  configuration for the whole-curve mean (−179 [−201, −156]). It is the first single configuration that is level
  with RND's best on the whole-curve mean and finds the goal about three times sooner. Against RND's best
  configuration for discovery (119.3 thousand steps) its first-goal time is −28.5 [−43.4, −13.5]. First block
  only; one of two configurations of the arm in this case.
- 2026-11-13 04:47 PT: **ant U-maze screen, second complete unit** (fast last layer alone, 128 environments per
  policy): behind RND in all 8 configurations (−0.005 to −0.016 on the whole-curve mean), first arrivals later (2.2
  to 4.1 against 1.9 to 2.75 million steps); 5 fault-guard respawns. With 128 environments per policy the fast
  last layer does not help on the ant, with or without the hold.
- 2026-11-13 04:47 PT: the last job of an older code copy on Gilbreth has ended (the batch 9 unit is complete), and
  the Gilbreth limit of this folder is open again (8 units at once) since 04:46 PT.
- 2026-11-13 05:10 PT: **batch 11, the diagnostic units on PointMaze Large with reward 1 on arrival: prediction 8
  failed.** Learning rate 2e-3, first validation block; RND's best: whole-curve mean 0.8781, final level 0.992,
  first-goal time 269 thousand steps.

  | fast last layer until the first arrival, then | weight | whole-curve mean | final level | first-goal time, thousand steps |
  |---|---|---|---|---|
  | RND's running divisor | 4 | 0.136 | 0.24 | 91 |
  | RND's running divisor | 16 | 0.023 | 0.03 | 83 |
  | divisor held after 1/16 of the run | 4 | 0.572 | 0.88 | 83 |
  | divisor held from the first arrival | 4 | 0.787 | not read | 83 |

  The early first arrival is kept and the goal is then not kept at all with the running divisor. The unit at
  weight 2 with the running divisor was still running. The numbers were passed to the background worker that is
  looking for the mechanism (started 04:31 PT, still running).
- 2026-11-13 05:10 PT: **batch 12 is running** (`batches/12_smaller-steps-of-the-fast-last-layer_with-the-first-arrival-hold_four-pointmaze-cases/ideas.md`,
  three predictions written before the runs): the fast last layer throughout with the first-arrival hold at steps
  0.03 and 0.01 (so far only 0.1), one configuration in each of the four PointMaze cases, first validation block,
  8 units (6 on Gilbreth, 2 on UVA). Reason: at step 0.1 and weight 2 on Large with reward 1 on arrival the arm is
  level with RND on the whole-curve mean and finds the goal three times sooner; a smaller step may keep most of
  the earlier arrival and lose less afterwards.
- 2026-11-13 05:28 PT: **batch 10, PointMaze Large with the step cost, learning rate 2e-3, weight 8: the fast last
  layer (step 0.1) with the first-arrival hold is level with the hold alone on return and finds the goal 2.7
  times sooner** (first validation block, 256 seeds).

  | compared with | whole-curve mean | final level | first-goal time, thousand steps |
  |---|---|---|---|
  | the hold alone, same configuration | −0.7264 against −0.7268: +0.0004 [−0.005, +0.007] | +0.0084 [+0.002, +0.015] | 71.5 against 189: −117 [−139, −97] |
  | RND's best for the whole-curve mean (2e-3, weight 4) | −0.7264 against −0.7310: +0.0046 [−0.0002, +0.0095] | +0.0132 [+0.008, +0.018] | 71.5 against 191: −120 [−142, −98] |

  RND's best configuration for discovery in this case has 120 thousand steps (batch 9), so the arm is ahead on
  discovery as well. On the whole-curve mean it is at the edge of being ahead of RND (first block, one of two
  configurations; the other, learning rate 4.5e-3, is at 234 of 256 windows).
- 2026-11-13 05:28 PT: **batch 12, the two PointMaze Medium cases (steps 0.03 and 0.01 of the fast last layer, with
  the first-arrival hold; first validation block).** Whole-curve mean and first-goal time in thousand steps:

  | case and configuration | step 0.1 | step 0.03 | step 0.01 | hold alone (step 0) | RND's best |
  |---|---|---|---|---|---|
  | reward 1 on arrival (6e-3, weight 4) | 0.8424, 15.7 | 0.9194, 17.3 | 0.9400, 18.5 | 0.9435, 18.5 | 0.9349, 25.0 |
  | step cost (6e-3, weight 2) | −0.3692, 12.9 | −0.3491, 14.0 | −0.3470, 13.8 | −0.3484, 14.2 | −0.3505, 20.0 |

  Predictions 1 and 3 hold: both measures move with the step, toward the hold alone. On Medium the fast last
  layer buys little (the hold alone already arrives within 14 to 19 thousand steps) and step 0.01 costs nothing
  measurable (−0.0035 [−0.011, +0.004] and +0.0013 [−0.002, +0.005] against the hold alone).
- 2026-11-13 05:28 PT: batch 11, the last diagnostic unit (PointMaze Large with reward 1 on arrival, fast until the
  first arrival then RND's running divisor, weight 2): whole-curve mean 0.426, final level 0.63, first-goal time
  96 thousand steps. Same failure as at weights 4 and 16.
- 2026-11-13 05:28 PT: **the picture after batches 10 to 12:** RND with the fast last layer throughout and the divisor
  held from the first arrival is one arm with one setting of its own, the step. The step trades return against
  discovery; on the two Large cases step 0.1 at the hold's weights is level with RND on return (−0.007 and +0.005)
  with first arrivals 2.7 to 3 times sooner; on the two Medium cases a small step gives the hold alone. The Large
  units at steps 0.03 and 0.01 are running.
- 2026-11-13 05:28 PT: **ant U-maze: the rollout-shape check is complete and does not decide between 128 and 16
  environments per policy.** Best whole-curve mean 0.0487 with 128 (final level 0.065, first-goal time 2.1
  million steps), 0.0487 with 16 (0.063, 1.3 million), 0.0341 with 4, which is out. Fault-guard respawns in the
  six units of each count: 32, 43, 16. The longer run will compare 128 and 16 at equal steps; until then ant
  screens run at both counts (the AntMaze baseline folder's `infra_history.md`). **Ant screen, third complete
  unit** (first-arrival hold alone, 128 environments per policy): level with RND in all 8 configurations (−0.003
  to +0.001); 12 fault-guard respawns. Prediction 4 of batch 10 failed at this count.
- 2026-11-13 05:30 PT: **the slow settling after the fast-last-layer phase is explained, and a fix exists**
  (`diagnosis/2026-11-13_slow-settling-after-the-fast-last-layer/notes.md`; option
  `predictor_divisor_after_first_arrival`, commit 9aab504b; a diagnostics field in bb07d8fa).
  1. Mechanism: RND's divisor remembers the predictor's first, large errors (93 at the start of PointMaze Medium,
     still 60 at the first arrival). The fast last layer's scores are about 14 times smaller from the start, so
     its divisor is about 4. When the fast last layer stops, the predictor's own scores are divided by that small
     divisor and the learner sees 10 to 20 times RND's intrinsic reward for the rest of the run.
  2. Ruled out: a larger action spread (it is smaller), a predictor in another state, the switch itself.
  3. Fix: a second divisor follows the predictor's own scores from the first rollout and is used once a policy has
     arrived. On short runs with 64 seeds the failure is gone (Medium, weight 4: return 0.734 against RND's 0.733
     with the first arrival after 16 against 23 thousand steps; Large, weight 2: 0.353 against 0.329 with 101
     against 249 thousand steps). These runs show the mechanism, not a lead.
  4. To know before use: the option adds four arrays per policy to the saved state, so a unit that uses it can
     only be continued by a code copy that has it; weights of fast-last-layer arms are not RND's weights (before
     the arrival the learner sees 2 to 3 times RND's reward at the same weight).
  **Batch 13 is written** (ideas and six predictions,
  `batches/13_fast-last-layer-until-the-first-arrival_then-the-predictors-own-divisor_four-pointmaze-cases/`): 13
  units on the first validation block. It is not built yet: jobs of the code copy 2017e2b8 are still running for
  this folder and would fail units of the new option.
- 2026-11-13 05:46 PT: **batch 13 is running** (13 units, first validation block: the fast last layer until the
  first arrival, then the predictor's own divisor, held or running; 9 on Gilbreth, 4 on UVA). **The folder's code
  is frozen at 42b327e7, copy number 1,** and workers now skip units that need a newer copy than their own
  (commit 42b327e7; `infra_history.md`). The three units of batch 12 that were running on jobs of the older copy
  (Large with reward 1 on arrival at step 0.01, Large with the step cost at steps 0.03 and 0.01) were interrupted
  after 35 minutes and continue from their checkpoints under the new copy.
- 2026-11-13 05:50 PT: **batch 12, PointMaze Large with reward 1 on arrival: the fast last layer at step 0.03 with
  the first-arrival hold is ahead of RND's best by +0.0195 on the first validation block** (learning rate 2e-3,
  weight 2, 256 seeds; the only configuration of this step in this case).

  | compared with | whole-curve mean | final level | first-goal time, thousand steps |
  |---|---|---|---|
  | RND's best (0.8781) | 0.8976: +0.0195 [+0.0136, +0.0254] | +0.0048 [+0.0036, +0.0060] | 128 against 269: −141 [−165, −117] |
  | the hold alone (0.8874) | +0.0101 [+0.0041, +0.0164] | −0.0009 [−0.0017, −0.0002] | 128 against 259: −131 [−155, −108] |

  Step by step in this case (whole-curve mean, first-goal time): step 0.1: 0.8707, 91; step 0.03: 0.8976, 128;
  hold alone: 0.8874, 259; RND: 0.8781, 269. The step 0.01 unit continues on Gilbreth. Prediction 2 of the batch
  holds. This is the largest lead so far in the case where the hold alone was only level on the second block.
  **Second block queued** (seeds 20000 to 20255, same configuration, steps 0.03 and 0.01; RND and the hold alone
  already have their units there: 0.8842 and 0.8839). Rule fixed now: the step with the higher whole-curve mean
  on the first block is the one whose second-block result counts.
- 2026-11-13 05:55 PT: **batch 13, the two PointMaze Medium cases: the failure of batch 11 is gone at full length**
  (the fast last layer until the first arrival, then the predictor's own divisor, held; first validation block,
  256 seeds, learning rate 6e-3). Whole-curve mean, final level, first-goal time in thousand steps:

  | case, weight | the arm | the hold alone, same weight | RND's best |
  |---|---|---|---|
  | reward 1 on arrival, weight 2 | 0.9445, 1.000, 16.0 | 0.9334, 1.000, 29.4 | 0.9349, 0.995, 25.0 |
  | reward 1 on arrival, weight 4 | 0.9416, 1.000, 14.7 | 0.9435, 1.000, 18.5 | (same) |
  | step cost, weight 1 | −0.3489, −0.307, 13.1 | −0.3476, −0.307, 15.4 | −0.3505, −0.308, 20.0 |
  | step cost, weight 2 | −0.3476, −0.307, 12.7 | −0.3484, −0.306, 14.2 | (same) |

  With reward 1 on arrival at weight 2 the arm is ahead of RND's best on the whole-curve mean (+0.0095 [+0.0023,
  +0.0168]) and on the first-goal time (−9.0 [−12.5, −5.7]); batch 11's arm without the second divisor had 0.700
  at weight 4. With the step cost it is level with the hold alone and with RND on the whole-curve mean (+0.0016
  [−0.004, +0.006] against RND) and 7 thousand steps sooner than RND's configuration. Predictions 2 and 5 hold.
  The nine units on the two Large cases are running.
- 2026-11-13 06:06 PT: batch 13, two more units (first validation block). PointMaze Large with reward 1 on
  arrival, the held arm at weight 4: whole-curve mean 0.8806 (against RND's best +0.0025 [−0.004, +0.009], level;
  batch 11's arm without the second divisor had 0.787), final level 0.998 (+0.0059 against RND), first-goal time
  96 against 269 thousand steps (−173 [−197, −149]). Weights 1 and 2 are running. PointMaze Medium with reward 1
  on arrival at weight 2, divisor left running against held: whole-curve mean 0.9385 against 0.9445 (−0.006
  [−0.013, +0.001]), final level 0.996 against 1.000 (−0.004 [−0.005, −0.003]); prediction 6 holds there.
- 2026-11-13 06:30 PT: **PointMaze Large with reward 1 on arrival: the fast last layer at step 0.03 with the
  first-arrival hold is ahead of RND on the second validation block too** (seeds 20000 to 20255, learning rate
  2e-3, weight 2, configuration fixed on the first block; RND 0.8842, hold alone 0.8839 on this block).

  | step | first block against RND's best | second block against RND | second block: final level, first-goal time against RND |
  |---|---|---|---|
  | 0.03 | 0.8976: +0.0195 [+0.0136, +0.0254] | 0.8995: +0.0153 [+0.0096, +0.0208] | +0.0041 [+0.003, +0.005]; 118 against 266 thousand steps (−148 [−172, −125]) |
  | 0.01 | 0.8928: +0.0148 [+0.0082, +0.0212] | 0.8945: +0.0103 [+0.0049, +0.0158] | +0.0047; 180 against 266 (−86 [−113, −61]) |

  Step 0.03 has the higher first-block mean, so its second-block result is the one that counts by the rule fixed
  before the runs; against the hold alone on the second block it is +0.0156 [+0.009, +0.022]. With this the
  whole-curve mean is ahead of RND on a second block in all four PointMaze cases, by one arm with one setting of
  its own: RND with the fast last layer at a step chosen per case (0 on the two Medium cases and, so far, on Large
  with the step cost; 0.03 on Large with reward 1 on arrival) and the divisor held from the first arrival. Fresh
  seeds are still kept for the end.
- 2026-11-13 06:30 PT: batch 13, PointMaze Large with reward 1 on arrival, the held arm at weight 2 (first block):
  whole-curve mean 0.8728 (−0.005 [−0.018, +0.005] against RND's best, level), final level 0.995, first-goal time
  85 against 269 thousand steps (−184 [−208, −161]). With weight 4 (0.8806, 96) this arm is level with RND on
  return and about three times sooner; prediction 3 of batch 13 does not hold so far (weight 1 is running). On
  return the small step throughout (batch 12) is the better arm in this case.
- 2026-11-13 06:30 PT: **batch 14 is running** (`batches/14_smaller-steps-of-the-fast-last-layer-with-the-hold_ant-umaze-screen_and-medium-second-block/ideas.md`):
  the ant U-maze screen of the fast last layer at steps 0.03 and 0.01 with the hold, at 128 and 16 environments
  per policy (4 units in the AntMaze baseline folder), and the second block on the two PointMaze Medium cases for
  step 0.01 with the hold (2 units). The standing page has two new families: the fast last layer throughout with
  the first-arrival hold, and the fast last layer until the first arrival with the predictor's own divisor.
- 2026-11-13 06:49 PT: **ant U-maze screen with 16 environments per policy, complete** (batch 10's three units; 16
  development seeds, 16,777,216 steps, 8 configurations each, against the check's RND unit at the same
  configuration; RND's best over its 48 configurations is 0.0487).

  | arm (step 0.1 where the fast last layer is used) | ahead of RND at the same configuration | differences in the whole-curve mean | arm's best | first-goal time against RND's | fault-guard respawns |
  |---|---|---|---|---|---|
  | fast last layer | 8 of 8 | +0.002 to +0.010 | 0.0516 | sooner in 8 of 8 (0.25 to 1.18 against 0.32 to 1.26 million steps) | 32 |
  | fast last layer with the first-arrival hold | 8 of 8 | +0.002 to +0.008 | 0.0538 | sooner in 8 of 8 | 52 |
  | first-arrival hold alone | 1 of 8 | −0.008 to +0.007 | 0.0558 | the same | 30 |

  The hold alone is ahead only at RND's best configuration (learning rate 3e-4, weight 0.0625), where it has the
  best mean and final level of any ant unit so far (0.0558 and 0.082 against 0.0487 and 0.062). With 128
  environments per policy the same three arms were behind or level. So on the ant the fast last layer helps with
  16 environments per policy and hurts with 128; predictions 5 and 6 of batch 10 hold for the first-goal time,
  and its whole-curve mean is ahead where I predicted it would not be; prediction 4 (the hold) fails. These are
  development seeds and one screen; the choice of the environment count is still open.
- 2026-11-13 06:49 PT: **batch 14, second block on PointMaze Medium, step 0.01 with the hold:** reward 1 on arrival:
  0.9403 against RND's 0.9308, +0.0095 [+0.0022, +0.0170], and equal to the hold alone (0.9404); first-goal time
  17.8 against 26.3 thousand steps. Step cost: −0.3494 against RND's −0.3535, +0.0041 [−0.0017, +0.0105], level
  (the hold alone has −0.3472 there, +0.0065); first-goal time 13.5 against 19.9. Prediction 3 of batch 14 holds.
- 2026-11-13 06:49 PT: **batch 13, three more units (first block).** PointMaze Large with reward 1 on arrival: the
  arm with the predictor's own divisor left running, weight 2: 0.8894, +0.0114 [+0.0054, +0.0173] against RND's
  best, final level level, first-goal time 92 against 269 thousand steps; held, weight 1: 0.8284, behind (−0.050).
  Held at weights 2 and 4 was level (0.8728, 0.8806), so here the running divisor is better than the held one and
  prediction 6 fails in this case. PointMaze Large with the step cost, held at (2e-3, weight 4): −0.7409, behind
  RND's best by 0.010, first-goal time 69 against 191 thousand steps. Three units on Large with the step cost
  are running.
- 2026-11-13 07:08 PT: **batch 12, PointMaze Large with the step cost, step 0.01 with the hold (2e-3, weight 8;
  first block):** −0.7250, against RND's best +0.0060 [+0.0004, +0.0115], final level +0.0069, first-goal time 123
  against 191 thousand steps (−69 [−93, −45]); against the hold alone at the same configuration +0.0018 (level).
  Batch 10's second configuration at step 0.1 (4.5e-3, weight 8) is behind (−0.7638). Second block queued at 07:06
  PT for the three steps 0.1, 0.03 and 0.01 at (2e-3, weight 8); the step with the highest first-block mean
  counts (step 0.03 was at 243 of 256 windows).
- 2026-11-13 07:08 PT: **ant: partial look at the longer RND run, and a concern** (details and numbers in the
  AntMaze baseline folder's `infra_history.md`).
  1. At equal steps (40 million) 16 environments per policy is well ahead of 128: 0.0779 against 0.0503 at each
     count's best configuration. The 16-environment best is at the lowest learning rate (1e-4) and lowest weight
     (0.0625) of its grid.
  2. The fault guard fires about 50 to 200 times more often per step than in the 16.8-million-step check, almost
     all late in the run and in the best configurations (up to one respawn per 0.3 million steps).
  3. Late in these runs 9 to 18% of all steps are above the walls (most in the best configurations) and torso
     speeds reach 14 to 26 m/s. The ants are thrown about; they do not walk.
  A background worker is checking late-training policies against reference MuJoCo (started 07:08 PT), to tell
  whether this motion is a property of the task or something only our physics allows. Until that is known, ant
  results are provisional for RND and for every arm.
- 2026-11-13 07:26 PT: **batch 12, PointMaze Large with the step cost, step 0.03 with the hold (2e-3, weight 8; first
  block):** −0.7223, against RND's best +0.0087 [+0.0039, +0.0136], final level +0.0100 [+0.005, +0.015],
  first-goal time 83 against 191 thousand steps (−108 [−130, −87]). In this case the first-block means by step
  are: 0.03: −0.7223; 0.01: −0.7250; 0.1: −0.7264; hold alone at the same configuration: −0.7268. So step 0.03 is
  the one whose second-block result counts (its unit was at 40 of 256 windows at 07:26 PT). Batch 13's held arm
  at (2e-3, weight 8): −0.7292, level with RND (+0.0018 [−0.005, +0.009]), first-goal time 105.
- 2026-11-13 07:26 PT: **where the arm stands** (RND with the fast last layer at a step chosen per case and the
  divisor held from the first arrival; whole-curve mean against RND, first-goal time in thousand steps):

  | case | step chosen on the first block | second block against RND | first-goal time against RND's |
  |---|---|---|---|
  | Medium, reward 1 on arrival | 0 (hold alone) | +0.0095 [+0.0015, +0.0178] | 20.6 against 26.3 |
  | Medium, step cost | 0 (hold alone) | +0.0065 [+0.0012, +0.0122]; +0.0040 [−0.0001, +0.0080] if RND chooses on the block | 14.7 against 19.9 |
  | Large, reward 1 on arrival | 0.03 | +0.0153 [+0.0096, +0.0208] | 118 against 266 |
  | Large, step cost | 0.03 | running (first block +0.0087; the hold alone has +0.0072 on the second block) | 83 against 191 (first block) |
- 2026-11-13 10:53 PT: **PointMaze Large with the step cost, second validation block: step 0.03 with the hold is
  ahead of RND** (seeds 20000 to 20255, learning rate 2e-3, weight 8, fixed on the first block; RND −0.7328 at its
  first-block best, learning rate 2e-3, weight 4): −0.7216, +0.0112 [+0.0065, +0.0160]; final level +0.0119
  [+0.007, +0.017]; first-goal time 89 against 203 thousand steps (−114 [−138, −91]). Against the hold alone on
  the same block: +0.0039 [−0.0015, +0.0097]; against the fixed hold +0.0017. Steps 0.01 and 0.1 are ahead of RND
  on this block too (+0.0095, +0.0083). **So the arm (RND, divisor held from each policy's first arrival, fast
  last layer at a step chosen per case on the first block) is ahead of RND on a second validation block in all
  four PointMaze cases.** With one step for every case (0.01) it is ahead in three and level on Medium with the
  step cost (+0.0041 [−0.0017, +0.0105]).

  | case | step chosen on the first block | second block against RND | first-goal time, thousand steps, against RND's |
  |---|---|---|---|
  | Medium, reward 1 on arrival | 0 (hold alone) | +0.0095 [+0.0015, +0.0178] | 20.6 against 26.3 |
  | Medium, step cost | 0 (hold alone) | +0.0065 [+0.0012, +0.0122]; level (+0.0040) if RND chooses on the block | 14.7 against 19.9 |
  | Large, reward 1 on arrival | 0.03 | +0.0153 [+0.0096, +0.0208] | 118 against 266 |
  | Large, step cost | 0.03 | +0.0112 [+0.0065, +0.0160] | 89 against 203 |

- 2026-11-13 10:53 PT: batch 13's last two units (PointMaze Large with the step cost, first block, against RND's
  best): the held arm at (3e-3, weight 8) −0.7243, +0.0067 [+0.0008, +0.0124]; the running divisor at (2e-3,
  weight 8) −0.7386, behind (−0.0076). Batch 13's arm is not better than the small step throughout in any case; it
  stays as an explanation of the slow settling, not as a lead.
- 2026-11-13 10:53 PT: **batch 14, ant U-maze screen of the smaller steps (16 development seeds, against RND at the
  same configuration):** with 16 environments per policy step 0.03 is ahead in 8 of 8 configurations (+0.0005 to
  +0.0049; best 0.0530 against RND's best 0.0487) and step 0.01 is level (−0.0026 to +0.0023); with 128
  environments per policy step 0.01 is level (−0.0026 to +0.0008) and step 0.03 behind in 8 of 8 (−0.003 to
  −0.010). Fault-guard respawns 8 to 38 per unit. Predictions 1, 2 and 4 hold.
- 2026-11-13 10:53 PT: **the ant motion check is done; the hopping is a property of the task** (the AntMaze baseline
  folder's `diagnosis/2026-11-13_what-the-trained-ants-do_against-reference-mujoco/notes.md`): the same policies
  in reference MuJoCo hop as much and arrive as often (within −0.017 and +0.021). The 07:08 PT caveat is withdrawn
  for the motion. The guard's runaways come from the implicitfast integrator at 4 solver iterations and are rare.
- 2026-11-13 11:06 PT: **discovery second block (seeds 20000 to 20255), the two PointMaze Medium cases**; the fast
  last layer (step 0.1) and RND each at its first-block best for the first-goal time. Reward 1 on arrival: 13.0
  against 16.8 thousand steps, −3.8 [−7.2, −1.0]; ahead, as on the first block (−5.0). Step cost: 11.4 against
  13.2, −1.8 [−3.1, −0.4]; ahead here although it was level on the first block (−1.0 [−2.3, +0.3]); I count it as
  ahead on one block and level on the other. At these configurations both sides keep the goal badly (whole-curve
  mean 0.12 and 0.008 with reward 1 on arrival), which is why the first-goal time has its own configurations. The
  two Large units of each side run on UVA (12 to 133 of 256 windows at 11:06 PT).
- 2026-11-13 11:31 PT: **AntMaze: Medium and Large are ready** (gates against C MuJoCo pass; configuration files in
  commit e54ab8e8; same cost per unit as the U-maze), and **the plan's budget rule gives B = 134,217,728 steps per
  policy on the U-maze** with 16 environments per policy (details and why this does not yet settle the count of
  environments per policy: the AntMaze baseline folder's `infra_history.md`). Cost, measured: one unit of 128
  policies x 16 environments at that budget is about 21 card-hours on an A100 (0.224 million steps per second; 256
  policies per unit give 0.34 million). A first sweep of 28 RND configurations in each reward setting is about 160
  card-hours, and a validation block of 256 seeds about 40 card-hours per configuration.
- 2026-11-13 11:54 PT: **the user's decisions** (multiple-choice dialog after background with data; plan "Change
  2"):
  1. the divisor frozen at each policy's first arrival counts as part of the method; the baseline stays plain
     RND; its closeness to public practice is disclosed (`literature/novelty.md`);
  2. the AntMaze comparison on Medium and Large is decided after RND probes there;
  3. the ant integrator is measured first (MuJoCo Warp's RK4 and implicit: speed, gates, runaways with trained
     policies; background worker started 11:56 PT); no ant sweep starts before the user has seen it;
  4. ant work: H100/H200 servers allowed in the order MIT, Purdue, ACCESS (a background worker prepares MIT and
     Gautschi `ai` for ant units, started 11:56 PT, servers left not ready); full protocol on the U-maze first,
     Medium and Large only the RND probe until it shows RND reaches the goal; 128 seeds per validation block on
     the ant, 256 fresh seeds; the budget stays at the rule's value (134,217,728 steps per policy on the U-maze).
- 2026-11-13 11:56 PT: **batch summaries 8 to 14 are written** (each batch folder's `summary.md`, generated; generator
  change 032f6313, summaries a642c1fe; 118 analysis tests pass). Corrections to entries above, from the summaries:
  1. batch 12, prediction 1 did not hold (on Medium with the step cost the first-goal time does not rise from
     step 0.03 to 0.01, 14.0 then 13.8 thousand steps; on both Large cases the whole-curve mean is highest at
     step 0.03, not at the smallest step);
  2. batch 13, prediction 6 holds on Large with reward 1 on arrival when read as written, on the final level
     (0.9945 held against 0.9931 running, within noise); the entry of 06:49 judged it on the whole-curve mean;
  3. batch 14, prediction 4 did not hold if "a few per unit" means at most 10 (13, 23, 8 and 38 respawns);
  4. batch 8, prediction 2, PointMaze Medium: with each side at its best first-goal time the error length reaches
     the goal sooner (18.2 against 21.0 thousand steps, −2.8 [−5.6, −0.1]); the prediction as a whole still did
     not hold (the whole-curve mean is behind);
  5. interval ends can differ in the third figure from the quick readings above (other resampling seeds); the
     means agree.
- 2026-11-13 11:56 PT: the longer ant run (`budget-probe`): the 16-environment unit is complete (35 ant units done);
  the 128-environment unit continues on Gilbreth. **One reading for question 3:** in the 128-environment run the
  guard respawns rose from 4,992 at 83 million steps to 177,192 at 122 million (78 per 10,000 episodes, 212 in the
  configuration with learning rate 1e-2 and weight 64, which is 2% of its episodes).
- 2026-11-13 11:57 PT: **discovery second block, PointMaze Large with reward 1 on arrival:** fast last layer (6e-3,
  entropy 0.03, weight 16) 57.3 against RND (6e-3, 0.01, weight 64) 129.8 thousand steps, −72.5 [−84.3, −61.1]
  (first block −63). Ahead on both blocks. Large with the step cost: RND's unit is complete, the fast last layer's
  at 152 of 256 windows (UVA, job 6908671).
- 2026-11-13 12:06 PT: **discovery second block, PointMaze Large with the step cost:** fast last layer 58.2 against
  RND 120.9 thousand steps, −62.7 [−74.5, −51.2] (final: all 256 policies have arrived; the unit's remaining
  windows cannot change it; first block −61). **On discovery the fast last layer is ahead of RND on both blocks in
  three PointMaze cases and on the second block in the fourth:**

  | case | first block, thousand steps [95%] | second block, thousand steps [95%] |
  |---|---|---|
  | Medium, reward 1 on arrival | −5.0 [−8.3, −2.4] | −3.8 [−7.2, −1.0] |
  | Medium, step cost | −1.0 [−2.3, +0.3] (level) | −1.8 [−3.1, −0.4] |
  | Large, reward 1 on arrival | −63 [−75, −52] | −72.5 [−84.3, −61.1] |
  | Large, step cost | −61 [−73, −50] | −62.7 [−74.5, −51.2] |

  Each side at its own best configuration for the first-goal time. Prediction of batch 9's addendum: the three
  cases held; Medium with the step cost was predicted level and is ahead on this block.
- 2026-11-13 12:59 PT: **the user's decisions after the integrator measurement** (plan "Change 3"): the ant runs
  MuJoCo Warp with RK4 at 0.01 s x 5 from now on (no runaways, every check passes, 1.85 times the time; units of
  the old physics are not compared with new ones); 16 environments per policy, settled against one
  128-environment RND unit at the same budget by the equal-steps rule. Integrator notes: the AntMaze folder's
  `diagnosis/2026-11-13_ant-integrators-in-mujoco-warp/notes.md`.
- 2026-11-13 13:13 PT: **the RK4 ant work is running**: RND sweep on the U-maze in both reward settings, the
  128-environment unit, RND probes on Medium and Large, and batch 15 (the method on the U-maze; ideas and six
  predictions written before the runs), 17 units at 134,217,728 steps per policy; 13 on UVA, 4 on Gilbreth (the
  AntMaze folder's `infra_history.md`). Each unit needs about 1.5 to 2 days of card time.
- 2026-11-13 13:18 PT: **correction: on the second block the method is ahead of RND in three PointMaze cases, not
  four.** On Medium with the step cost the rule fixed before the second blocks were read (the step with the highest
  first-block whole-curve mean) chooses step 0.01 (−0.3470 against −0.3476 for step 0), and step 0.01 is level on
  the second block: +0.0041 [−0.0017, +0.0103]. The table of 10:53 PT listed step 0 there, chosen after both
  second-block readings were known; that is a choice on the test seeds and does not count. In reward setting B the
  PointMaze part of the success rule is therefore not met yet. Found by the worker that wrote the development
  document's section, which now follows the rule (commit 6f7aa0af).
- 2026-11-13 13:18 PT: **the development document has the study's section** (`development_document/beat_rnd_pointmaze_antmaze.tex`,
  section 6, PDF pages 108 to 122; every number generated from the records by `development_document/code/beat_rnd_tables.py`;
  compiles with no overfull box and no undefined reference).
- 2026-11-13 13:30 PT: all 17 RK4 ant units are running (13 on UVA at 1 to 2 of 256 windows after 18 minutes, about
  10 minutes per window, so about 40 hours per unit; 4 on Gilbreth, just started). Because the wall time of an ant
  unit now sets the schedule, a background worker measures where the time of one RK4 ant iteration goes and the
  speed ideas that keep results identical (plan Part 3; serval05 profile units, started 13:30 PT).
- 2026-11-13 14:25 PT: **RK4 ant speed:** the iteration is 98% physics and three quarters fixed cost (kernel launches),
  so a unit at 134 million steps per policy needs at least about 26 hours on an H100 whatever its number of
  policies; no identical-result idea reaches 5% (best +4.3%). Two start-up options added (off by default; about 1%
  of a 2-hour UVA job). Details in the AntMaze folder's `infra_history.md`. For the schedule: the RTX A4500 units
  (about 45 hours) are the slowest; near the end of the sweep they move to faster free cards
  ([[move-slow-card-units-to-fast-cards-near-the-end]] in memory).
- 2026-11-13 15:50 PT: **RND probes on AntMaze Medium and Large, an early reading** (RK4 physics, 16 environments per
  policy, 15 of 256 windows = 7.9 of 134 million steps per policy, 8 configurations x 16 seeds): on Medium all 128
  policies have reached the goal at least once, and 0.5 to 0.9% of the latest episodes arrive; on Large 1 of 128
  policies has reached the goal (learning rate 1e-2, weight 1). Not a result yet; the question for the user (what
  the comparison is where RND rarely reaches the goal) waits for the full probes.
- 2026-11-13 20:10 PT: **mid-run look at the RK4 ant sweep, U-maze, reward 1 on arrival** (75 of 256 windows, about
  39 million steps per policy; a check, not a result): every policy of the two most advanced RND units has reached
  the goal (one of 16 at the smallest weight not yet); the mean return so far is highest at small intrinsic
  weights: 0.0896 at learning rate 1e-4 and weight 1/16, 0.0824 at 1/64, against 0.045 to 0.047 at weights 1 to
  16; about 10% of the latest episodes arrive at the best configurations. The method at step 0.1 (56 windows) has
  0.0800 at (1e-4, 1/16). No fault-guard respawn in any of the 17 units.

## Next

Rewritten 2026-11-13 22:31 PT.

1. **RK4 ant sweep and batch 15** (U-maze, 17 units at 134,217,728 steps per policy, 16 environments per policy;
   the slowest units end on 15 November at the rates of 13 November evening): each tick restarts the UVA units
   after their 2-hour jobs (the session's scratchpad script `ant_tick.sh`, free cards only, waiting jobs cancelled
   by ledger id and their units handed to Gilbreth) and reads every unit's fault-guard count. Near the end, units
   on the slowest cards move to faster free cards ([[move-slow-card-units-to-fast-cards-near-the-end]]).
2. **When the sweep is complete:**
   1. RND's configurations in both reward settings: the best for the whole-curve mean and the best for the mean
      capped first-goal time, on the 16 development seeds;
   2. the equal-steps rule on the pair of environment counts (16 against the 128-environment unit at the same
      budget); the count changes only if 128 is ahead;
   3. batch 15 against RND at the same configurations and seeds (its six predictions), and its summary.
3. **First ant validation block** (U-maze, seeds 10000 to 10127): each side's best three configurations for the
   whole-curve mean and its best for the first-goal time, in both reward settings; placement Gilbreth, free UVA
   cards, then MIT, Gautschi `ai`, ACCESS. About 26 to 40 hours per unit.
4. **Ant fresh seeds** (256 from 30000) for RND and the method as it stands at the end. My default, told to the
   user on 2026-11-13: on the ant the 256 fresh seeds also serve as the check on unused seeds; no separate second
   block unless the user asks for one.
5. **RND probes on AntMaze Medium and Large:** when complete, their curves go to the user, who decides the
   comparison there.
6. **PointMaze:** on the second block the method is ahead of RND in three of the four cases; Medium with the step
   cost is level (+0.0041 [−0.0017, +0.0103]). Fresh seeds (1,024) for the method as it stands at the end.
7. **Days 6 to 10** (changes outside the bonus, from about 2026-11-16 22:40 PT), each tied to an observation of
   days 1 to 5. Schedule fact: an ant unit takes at least about 26 hours, so a change found then reaches the ant
   only through one sweep, one validation block and the fresh seeds back to back (about 4.5 days), which ends
   close to 2026-11-21 22:40 PT with no time to spare.
8. Re-create the monitoring cron before 2026-11-20 13:00 PT (created 2026-11-13 about 13:00 PT; it expires after
   7 days).

## Ideas (status)

| batch | idea | status |
|---|---|---|
| 1 | separate predictor step size (0.25, 1, 4 times 1e-4) | within the control's interval on 16 seeds; not pursued alone; slower values are in batch 2 |
| 1 | fast last layer inside the rollout (steps 0.1, 0.3, 1) | earlier arrivals at steps 0.1 and 0.3, lower final level; within or behind; kept in mind for a combination with a held divisor |
| 1 | error length as the bonus | behind (−0.040) |
| 1 | fixed scale; even start; even start with no divisor | far behind; closed |
| 1 | scheduled last layer on the random body's features (decay 0.5, 0.75, 1) | far behind; the fit is not local and the input statistics move; local features are in batch 2 |
| 2 | wider input clip; input statistics kept from the random-action rollouts; local features as the input of RND; scheduled last layer on local features | see `batches/02_.../summary.md`; none went further |
| 2, 5, 7 | divisor held after a part of the run (1/16) | ahead of RND on the whole-curve mean on the two Medium cases (second block: +0.011, +0.006); level on Large with reward 1 on arrival on the second block (−0.002; +0.0095 on the first); Large with the step cost +0.0095 on the first block only; final level ahead everywhere; close to public practice (`literature/novelty.md`) |
| 2, 5, 6 | slower predictor, or predictor at half speed, with the held divisor | ahead in some cases, level in others; not better than the held divisor alone |
| 3 | all three parts (fast last layer, slower predictor, held divisor) | ahead on Large with reward 1 on arrival (+0.0137), behind on Medium with reward 1 on arrival (−0.029), level on Large with the step cost |
| 8 | error length with the held divisor | behind RND at every weight; closed for the whole-curve mean |
| 8 | divisor held from each policy's first arrival | level with the fixed hold, nothing to set; ahead of RND on both Medium cases (second block: +0.0095, +0.0065); level on Large with reward 1 on arrival on the second block (−0.0003; +0.0094 on the first); Large with the step cost +0.0074 on the first block, second block running |
| 3, 9 | fast last layer, for discovery | first-goal time ahead of RND in three PointMaze cases (−63, −61, −5.0 thousand steps), level on Medium with the step cost; its whole-curve mean is far behind at those configurations |
| 10 | fast last layer with the first-arrival hold (PointMaze); first-arrival hold, fast last layer, and both on the ant U-maze | running since 2026-11-13 02:58 PT |

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
