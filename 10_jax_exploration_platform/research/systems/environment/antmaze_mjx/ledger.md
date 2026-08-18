# antmaze_mjx — round ledger

The AntMaze environment family: the Gymnasium Ant in the umaze / medium / large mazes, fused
into the platform's single compiled iteration through MuJoCo MJX. Specification and deviations:
`src/exploration_platform/envs/antmaze/spec.md` (`antmaze_<map>_cont<cap>_nonoise@1`).

Times are Pacific with a `PT` marker; the machines run on Eastern and the times were converted
for display. Result files are not committed (the repository ignores `*.json`), so each round
keeps its own result JSON under `benchmarks/*/results/` on disk and the tables below carry what
it said.

## The research question this family answers

Can the platform train PPO with an exploration bonus end to end — environment, agent and bonus
in one compiled program — on an environment whose physics is real articulated rigid-body
dynamics rather than a hand-derived ball, and what does one such environment step cost next to
PointMaze's?

## The gate this family had to pass

1. **Correctness before speed.** The full test suite: the PointMaze golden-parity gate still
   bit-identical after the trainer was made environment-generic, and the AntMaze gates of
   `tests/envs/` (stepper semantics, C-MuJoCo parity, end-to-end composition).
2. **End to end.** `Runner(PPOConfig(...), bonus="rnd_next_state", env_cfg=preset(map))`
   composes, compiles and trains, with coverage tracked, for `rnd_next_state` and `none`; the
   PointMaze-only visit-count family is refused at composition time.
3. **Measured throughput** on the H100 NVL of serval05, env step alone and whole training
   iteration, at several copy counts, recorded below.

## Rounds

| round | when | what was tried | outcome |
|---|---|---|---|
| 1 — [MJX fusion](rounds/2026-08-18-r01-mjx-fusion/notes.md) | 2026-08-18, from 00:20 PT | load the reference model through MJX as shipped, then find settings that are both fast and faithful | as shipped (RK4, full solver, one box per wall cell) an env step costs 2,134 ms at 1,024 copies — unusable. Accepted: `implicitfast` with Newton iterations 4 / line-search 8 (settles at the reference height where iterations 1, the MJX example-model setting, is unstable), wall runs merged into single boxes, float32 under a trace-time x64-off boundary. The MJX contact-cap numerics (`max_geom_pairs` / `max_contact_points`) were found broken in MJX 3.11's JAX backend — any setting silently zeroes every contact — and are not used |

## Where this family stands — environment step alone (H100 NVL, serval05)

(to be filled by `benchmarks/env/bench_antmaze_env.py` in round 1; one row per (map, copies))

## Where this family stands — whole training iteration (H100 NVL, serval05)

(to be filled by `benchmarks/end_to_end/bench_antmaze_train.py` in round 1)

## Correctness, after round 1

(to be recorded when the full suite has run on the refactored trainer)
