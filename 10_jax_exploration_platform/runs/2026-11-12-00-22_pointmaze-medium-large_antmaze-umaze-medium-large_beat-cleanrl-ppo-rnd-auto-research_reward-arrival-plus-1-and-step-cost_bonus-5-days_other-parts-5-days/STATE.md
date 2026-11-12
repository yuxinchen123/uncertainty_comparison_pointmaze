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
