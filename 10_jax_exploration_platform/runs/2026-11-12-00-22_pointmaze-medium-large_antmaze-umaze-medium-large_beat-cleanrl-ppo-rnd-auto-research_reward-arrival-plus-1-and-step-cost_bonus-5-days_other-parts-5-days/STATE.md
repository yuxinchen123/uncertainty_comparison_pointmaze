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

## Next

1. Review and commit the trainer changes; run the whole-iteration time breakdown for both families.
2. Order the 30 speed ideas from the breakdown; profile them (15 minutes each, H100 NVL first).
3. Rollout-shape check (4, 16, 128 environments per policy); RND first-pass sweep; budgets.

## Ideas (status)

None yet. The first diagnostic (before batch 1): how much one predictor update lowers the error at states not yet
visited, per maze cell, for RND.

## Decisions I made (routine), with the reason

- PointMaze step limit 400 (the earlier runs' value), AntMaze 700 / 1,000 / 1,000 (Gymnasium's).
- Reward setting B is applied in the trainer as setting A's reward mapped by $(r - 1)/H$ per copy, so both settings
  can share one process on the same seeds.
- The copy count is the smallest power of two at or above 4.5 × A ÷ B (the plan said "largest below", which would
  halve every count and disagrees with the doubling rule's own outcome on the 2026-11-10 series).
- Every proxy factor in the skill's `card_facts.json` for cards outside the UVA survey is a guess until that card's
  first two real units correct it.
