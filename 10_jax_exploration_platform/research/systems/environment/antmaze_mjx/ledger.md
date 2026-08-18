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
| 1 — [MJX fusion](rounds/2026-08-18-r01-mjx-fusion/notes.md) | 2026-08-18, from 00:20 PT | load the reference model through MJX as shipped, then find settings that are both fast and faithful | as shipped (RK4, full solver, one box per wall cell) an env step costs 2,134 ms at 1,024 copies — unusable. Accepted: `implicitfast` with Newton iterations 4 / line-search 8 (settles at the reference height where iterations 1, the MJX example-model setting, is unstable), wall runs merged into single boxes, float32 under a trace-time x64-off boundary. The MJX contact-cap numerics (`max_geom_pairs` / `max_contact_points`) appeared broken — any setting silently zeroed every contact — and were shelved until round 2 explained them |
| 2 — the contact budget, and honest timing | 2026-08-18 05:30–06:55 PT | diagnose why the first benchmark grid read a flat ~250 ms per step at every copy count | two separate causes found. (a) The 250 ms was an artifact: XLA compiles a SECOND executable on the first call whose input buffers are donated, and the harness's single warm-up call left that compile inside the timed rounds; steady-state was 3 ms all along. Benchmarks now warm up twice. (b) The cap numerics fail because MJX 3.11 reads them as `numeric_data[id]` instead of `numeric_data[adr[id]]`; ant.xml's 15-element `init_qpos` numeric sits first, so any cap read from inside it as ~0 and deleted every contact. Dropping the unused `init_qpos` numeric and registering the two size-1 caps first aligns id with address, and the caps work: ncon padding falls from 225 to 32 on every map, the per-step cost becomes map-independent, and the whole antmaze gate set re-runs green with the capped model |

## Where this family stands — environment step alone (H100 NVL, serval05)

Measured 2026-08-18 ~06:50 PT by `benchmarks/env/bench_antmaze_env.py` (medians over 11 rounds
of 10 env steps, random torques, the state donated; result JSON
`benchmarks/env/results/2026-08-18_antmaze_env_h100nvl_serval05.json`). One env step is 5
physics steps. n_envs = 1, so copies = environments. The per-step cost is nearly independent of
the map — the contact budget (32 slots) equalizes the three — so the copy count is the only
knob that matters:

| map | copies | seconds per env step | total env steps per second | env steps per second per copy | hours per million steps per copy | peak device memory |
|---|---|---|---|---|---|---|
| umaze | 512 | 0.00626 | 0.082 M | 159.7 | 1.74 | 0.11 GiB |
| umaze | 1,024 | 0.00807 | 0.127 M | 124.0 | 2.24 | 0.12 GiB |
| umaze | 2,048 | 0.01140 | 0.180 M | 87.7 | 3.17 | 0.16 GiB |
| umaze | 4,096 | 0.01765 | 0.232 M | 56.7 | 4.90 | 0.24 GiB |
| medium | 512 | 0.00633 | 0.081 M | 157.9 | 1.76 | 0.24 GiB |
| medium | 1,024 | 0.00815 | 0.126 M | 122.7 | 2.26 | 0.24 GiB |
| medium | 2,048 | 0.01157 | 0.177 M | 86.4 | 3.21 | 0.24 GiB |
| medium | 4,096 | 0.01788 | 0.229 M | 55.9 | 4.97 | 0.24 GiB |
| large | 512 | 0.00645 | 0.079 M | 155.0 | 1.79 | 0.24 GiB |
| large | 1,024 | 0.00822 | 0.125 M | 121.6 | 2.28 | 0.24 GiB |
| large | 2,048 | 0.01167 | 0.175 M | 85.7 | 3.24 | 0.24 GiB |
| large | 4,096 | 0.01805 | 0.227 M | 55.4 | 5.01 | 0.24 GiB |

Next to PointMaze this is what real articulated physics costs: the fused PointMaze reaches
tens of millions of env steps per second where the ant reaches a quarter of one million — a
factor of roughly 200 — and the aggregate is still climbing at 4,096 copies while each copy
slows, so a run picks its copy count by which side of that trade it needs. Compile time is 8 s
per (map, copies) shape.

## Where this family stands — whole training iteration (H100 NVL, serval05)

(to be filled by `benchmarks/end_to_end/bench_antmaze_train.py` in round 1)

## Correctness, after round 2

The full CPU suite (16 tests) runs green: the PointMaze golden-parity gate is still
bit-identical after the trainer was made environment-generic; the AntMaze stepper gates
(shapes, determinism, copy isolation, goal semantics, truncation and auto-reset, the x64
boundary, cell indexing, preset connectivity) pass; the three C-MuJoCo parity gates pass with
the capped model (contact-free one-step worst 1.76e-6, in-contact worst 2.75e-2 within the
documented bound, equilibrium shared at z = 0.38248 to 3.7e-11 both ways); and the end-to-end
composition trains for `rnd_next_state` and `none` with the visit-count family refused.
