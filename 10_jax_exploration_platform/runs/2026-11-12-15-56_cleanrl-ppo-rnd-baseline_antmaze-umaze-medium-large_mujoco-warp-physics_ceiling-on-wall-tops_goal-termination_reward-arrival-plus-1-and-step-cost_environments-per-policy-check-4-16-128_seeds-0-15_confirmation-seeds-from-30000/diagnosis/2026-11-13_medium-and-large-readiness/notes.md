# AntMaze Medium and Large: ready for training units

Work of 2026-11-13, 11:02 to 11:27 PT, by a helper of the main session. Times are Pacific, converted from
machines running Eastern. Every Slurm job id is in `slurm/submitted_jobids_medium-large-readiness_sl5nw.txt` of this
run folder; the test jobs are also in the speed study's own test ledger, because its test runner writes there. All
jobs have ended.

## Result

1. **Both maps pass every gate** (table "The gates" below), with the physics the AntMaze baseline uses: MuJoCo
   Warp, implicitfast, 4 solver and 8 line-search iterations, 10 physics steps of 0.005 s per environment step, the
   fault guard at 30 m/s and 100 rad/s, the ceiling on the wall tops, the episode ending at the goal.
2. **Nothing map-specific in the environment code is wrong**, so no environment code changed. One docstring count
   in `am_common.py` was wrong ("45 -> 20 on the large map"; the boxes are 62 -> 31) and was corrected.
3. **One unit costs what it costs on the U-maze**: at 16 environments per policy the Medium and Large maps take
   0.5% and 1.1% longer per iteration than the U-maze on the same card (A40), at 128 environments per policy up to
   4% (A100 40 GB against the U-maze's real units on A100 80 GB). Card memory is the same on all three maps.
4. **Two configuration files** (the U-maze stable-physics file on each map; only the map, its start and goal
   cells, its step limit and the runtime selector differ):
   - `configs/experiments/antmaze_medium_warp-physics_time-step-0.005_frame-skip-10_fault-guard-30-metres-and-100-radians-per-second_cleanrl-ppo-rnd_goal-termination_reward-arrival-plus-one_example_steps-204800.toml`
   - `configs/experiments/antmaze_large_warp-physics_time-step-0.005_frame-skip-10_fault-guard-30-metres-and-100-radians-per-second_cleanrl-ppo-rnd_goal-termination_reward-arrival-plus-one_example_steps-204800.toml`

## What the code does on the larger maps

Read in `am_common.py`, `warp_antmaze.py`, `faults.py`, `scripts/run_training.py` and
`scripts/maze_campaign/build_units.py`, then checked by the gates.

| item | U-maze | Medium | Large |
|---|---|---|---|
| cells (rows x columns), 4 m each | 5 x 5 | 8 x 8 | 9 x 12 |
| start cell, its centre (m) | (3, 1), (-4, -4) | (6, 1), (-10, -10) | (7, 1), (-18, -12) |
| goal cell, its centre (m) | (1, 1), (-4, 4) | (1, 6), (10, 10) | (1, 10), (18, 12) |
| step limit | 700 | 1,000 | 1,000 |
| open cells (the coverage units) | 7 | 26 | 46 |
| wall cells, merged into boxes | 18 into 8 | 38 into 20 | 62 into 31 |
| outer wall faces, the guard's volume (m) | x and y within 10 | x and y within 16 | x within 24, y within 18 |
| geoms in the model | 23 | 35 | 46 |
| geom pairs MuJoCo Warp tests per environment | 118 | 274 | 417 |

- **Same on every map**: the goal radius (0.45 m), the reset (start cell centre, 0.75 m high, at rest, no noise),
  the ceiling (one plane at 2 m; a plane collides without bound, its size of 40 m only sets how it is drawn), the
  contact pool (48 slots per environment, shared) and the limit of 512 constraint rows per environment.
- **The contact pool does not fill faster on larger maps.** MuJoCo Warp tests every allowed geom pair (13 ant geoms
  against the floor and every wall box, and the torso against the ceiling), but keeps a pair only when the
  geoms' oriented bounding boxes overlap (its default filter is plane, sphere and oriented box: `io.py` line 420
  of the installed MuJoCo Warp). A long merged wall therefore adds no candidate pairs unless the ant is at it.
  Measured: at most 4.0 candidate pairs and 1.6 contacts per environment on all three maps (pool: 48).
- **The MJX contact caps** (`max_geom_pairs`, `max_contact_points`) are read by MJX's JAX implementation only.
- **The observation** holds the torso's x and y in metres, so its range grows from 10 m (U-maze) to 16 m (Medium)
  and 24 m (Large). The policy and value networks get it raw (`normalize_agent_obs = false`, as in the U-maze
  file); the random network distillation input is normalised. This is a learner setting and was not changed.
- **The unit builder** already lists both environments with the right cells and step limits, and replaces the map
  values of whatever base file a description names. Two things a description must give for these maps:
  1. the workload explicitly, because the default name has no time step in it, for example
     `antmaze-medium_environments-per-policy-16_update-epoch-minibatch_bonus-rnd-next-state_physics-warp_time-step-0.005_frame-skip-10`;
  2. `policies_per_unit` as a number and `minimum_card_memory_mb`, because `device/estimates/` has no AntMaze
     estimate file (the U-maze descriptions do the same).

## The gates

Script `scripts/map_gates.py`, on puma01's processors (MuJoCo Warp on the processor device, C MuJoCo in double
precision), jobs 6908709 to 6908738. Records: `records/gates_<map>.json` (file, reset, goal),
`records/gates_<map>_physics.json` (step limit, guard, walls, ceiling with the guard on, agreement) and
`records/gates_<map>_ceiling-with-the-guard-off.json`. The agreement bounds are those of `spec.md` ("Correctness
gates"); the U-maze's numbers at this time step are in `spec.md`, deviation 10.

| gate | what is measured | bound | Medium | Large |
|---|---|---|---|---|
| file | the file's map values against the preset; 49 points in every cell inside a wall box exactly when the cell is a wall | equal; 0 cells wrong | passed; 0 | passed; 0 |
| reset | 6 environments: torso xy minus the start cell's centre; height; speed | 0; 0.75 m; 0 | 0; 0.75 m; 0 | 0; 0.75 m; 0 |
| goal | 28 torsos placed 0 to 1 m from the goal in 4 directions, one step | ends exactly the 16 within 0.45 m; reward 1 exactly there | 16 ended, all within 0.45 m, reward 1; restart at the spawn | the same |
| step limit | 2 ants (no torque, random torques), 1,000 steps from the spawn | one truncation each, at step 1,000 | at step 1,000 only; no termination, no respawn | the same |
| walls | an ant thrown at 8 m/s toward every wall face next to an open cell, 1.5 s, Warp and C | every ant touches a wall; no body origin or geom centre inside a wall | 48 of 48 touch; deepest body origin 0.094 m outside a wall | 88 of 88 touch; 0.094 m outside |
| walls, agreement | one physics step from C states around each ant's first wall contact | 5e-2 m | 192 states, at most 4.9e-7 m | 352 states, at most 1.5e-6 m |
| ceiling | ants thrown upward at 10, 20, 30 m/s at the spawn, the goal and the open cells nearest the corners, 2 s, Warp and C | touch the ceiling; stay below 2 m | 1.796, 1.843, 1.917 m at every cell, Warp and C alike | the same |
| agreement, one step | one environment step from 50 states of a random-torque trajectory: contact-free; in contact | 1e-4 m; 5e-2 m | 1.7e-6 m (26 states); 8.4e-3 m (24) | 3.6e-6 m (26); 8.4e-3 m (24) |
| agreement, 20 steps | shared random torques from the spawn: before the first contact; over all 20 steps | 1e-4 m; 4 m | 1.5e-6 m; 0.16 m | 3.3e-6 m; 0.16 m |
| resting height | 1,200 passive steps; a C-settled state kept by Warp | 0.38248 m within 1e-4; moved under 1e-3 m | 0.38248 m in both; moved 1.6e-8 m | the same |
| guard | the six forced ants of `tests/envs/test_antmaze_fault_guard.py` (one 2 m outside this map's outer wall); the volume bound 1 cm on each side of the outer faces | exactly the four broken ants respawned; inside kept, outside respawned | respawned: the ant at 60 m/s, the ant moving up at 40 m/s at the goal, the ant outside the wall, the ant turning at 150 rad/s; the ant at rest at the goal ended there with reward 1; both faces right | the same |

- **Wall details.** In C MuJoCo the throws touched every wall box (20 of 20, 31 of 31). The torso centre never came
  closer than 0.379 m to a wall face and never entered a wall cell; Warp and C agree on these numbers to 1e-5 m.
- **Ceiling with the guard on.** The first run kept the guard on, and the four throws at 30 m/s (one per cell; 30 m/s
  is the guard's speed bound) were respawned by the guard at their first environment step in Warp, while C MuJoCo
  has no guard; which of the guard's tests fired was not recorded. Respawning an ant at its bound is what the guard
  is for, so the ceiling gate runs with the guard off, as the U-maze's ceiling test does. The heights then equal the U-maze's at this time step (1.796, 1.843 and 1.917 m) at every cell tested.
- **Why the contact-free difference is 2 to 4 times the U-maze's** (9.3e-7 m there): the torso sits at |x| = 10 m
  (Medium) and 18 m (Large) instead of 4 m, and single-precision numbers are spaced 9.5e-7 m apart between 8 and
  16 m and 1.9e-6 m between 16 and 32 m.

## Speed and memory per card

Script `scripts/profile_unit_shapes.py`: one shape per process, built exactly as `scripts/run_training.py` builds a
unit from the map's configuration file (through `device/series.py`), untrained policies, 2 priming iterations,
two compile calls, then the median of at least 5 timed iterations. Card memory is the largest nvidia-smi reading
while the process ran, with JAX's preallocation off. Jobs 6908692 to 6908694 and 6908741 (one card each, at most 15
minutes per job).

Define $r$ as the environment steps per second of one copy (one policy with its environments). The last throughput
column is $\dfrac{10^{6}}{3600 \times r}$, the hours one copy needs for a million steps.

| card | map | environments per policy | copies | seconds per iteration | total steps per second (millions) | steps per second per copy (thousands) | hours per million steps per copy | card memory (MiB) | candidate pairs per environment, most | constraint rows of one environment, most |
|---|---|---|---|---|---|---|---|---|---|---|
| A100 40 GB | Medium | 16 | 64 | 0.966 | 0.136 | 2.12 | 0.131 | 1,719 | 4.0 | 31 |
| A100 40 GB | Medium | 16 | 128 | 1.171 | 0.224 | 1.75 | 0.159 | 2,779 | 4.0 | 38 |
| A100 40 GB | Medium | 16 | 256 | 1.540 | 0.340 | 1.33 | 0.209 | 4,941 | 3.9 | 37 |
| A100 40 GB | Medium | 16 | 512 | 2.352 | 0.446 | 0.87 | 0.319 | 9,263 | 3.9 | 42 |
| A100 40 GB | Large | 16 | 64 | 0.964 | 0.136 | 2.12 | 0.131 | 1,717 | 4.0 | 31 |
| A100 40 GB | Large | 16 | 128 | 1.177 | 0.223 | 1.74 | 0.160 | 2,783 | 3.9 | 36 |
| A100 40 GB | Large | 16 | 256 | 1.553 | 0.338 | 1.32 | 0.211 | 4,941 | 3.9 | 36 |
| A100 40 GB | Large | 16 | 512 | 2.387 | 0.439 | 0.86 | 0.324 | 9,265 | 3.9 | 39 |
| A100 40 GB | Medium | 128 | 64 | 2.338 | 0.448 | 7.01 | 0.040 | 8,989 | 3.9 | 37 |
| A100 40 GB | Medium | 128 | 128 | 3.996 | 0.525 | 4.10 | 0.068 | 17,311 | 3.9 | 38 |
| A100 40 GB | Large | 128 | 64 | 2.370 | 0.442 | 6.91 | 0.040 | 8,983 | 3.9 | 37 |
| A100 40 GB | Large | 128 | 128 | 4.055 | 0.517 | 4.04 | 0.069 | 17,311 | 3.9 | 38 |
| A40 | U-maze | 16 | 128 | 1.098 | 0.239 | 1.87 | 0.149 | 2,609 | 3.9 | 37 |
| A40 | Medium | 16 | 128 | 1.103 | 0.238 | 1.86 | 0.150 | 2,607 | 4.0 | 34 |
| A40 | Large | 16 | 128 | 1.110 | 0.236 | 1.84 | 0.151 | 2,607 | 4.0 | 35 |
| A40 | Large | 128 | 128 | 5.217 | 0.402 | 3.14 | 0.088 | 17,137 | 3.9 | 39 |

- **The map adds little.** At 16 environments per policy and 128 policies on the A40: U-maze 1.098 s, Medium
  1.103 s, Large 1.110 s. On the A100 40 GB, Medium and Large take 1.171 and 1.177 s; the U-maze's real units of
  this shape took 1.161 to 1.177 s on A100 80 GB cards at Gilbreth (`device/observed/`). At 128 environments per
  policy and 128 policies: 4.00 and 4.06 s, against 3.89 to 4.03 s for the U-maze's real units.
- **Nothing was dropped** in any shape: at most 4.0 candidate pairs per environment (48 slots) and 42 constraint
  rows in one environment (limit 512). These are untrained policies. Trained U-maze ants need more: the first
  U-maze training unit stopped at 161 rows under the earlier limit of 160, and trained policies reached 179 rows in
  the gate of `spec.md`, deviation 10; hence the limit of 512. The walls are the same flat faces on every map, so
  trained ants should not need more rows on Medium or Large, but the first units' `constraint_rows_largest` in
  their window records is the check.
- **Total steps per second keeps rising with the policies at 16 environments per policy**: on the A100 40 GB
  +65% from 64 to 128 policies, +52% from 128 to 256, +31% from 256 to 512 (Medium; +30% on Large; 9,265 MiB
  of card memory). Each doubling still gains more than 10%, so by the copies-per-card rule 512 is not yet where
  the gain stops; 1,024 was not measured.

## Sizing

### Steps per policy in one 4-hour Gilbreth job, 16 environments per policy

Define $s$ as the seconds per training iteration, $E = 16$ the environments per policy and $T = 128$ the rollout
steps, so one iteration adds $E T = 2{,}048$ environment steps to every policy. Define $U$ as the seconds of a
4-hour job left after start-up; start-up measured here is 2 to 3 minutes (building, priming, two compile calls),
and a unit primes 50 iterations instead of 2, so $U = 14{,}400 - 300 = 14{,}100$ s is assumed. The steps per policy
one job reaches are $S = \dfrac{U E T}{s}$.

| policies per unit | card | seconds per iteration | steps per policy in one 4-hour job (millions) | hours for a unit of 16.8 million steps per policy, start-up included |
|---|---|---|---|---|
| 64 | A100 (40 GB measured) | 0.965 | 29.9 | 2.3 |
| 128 | A100 (40 GB measured) | 1.174 | 24.6 | 2.8 |
| 256 | A100 (40 GB measured) | 1.547 | 18.7 | 3.6 |
| 512 | A100 (40 GB measured) | 2.370 | 12.2 | 5.5 (two jobs) |
| 128 | A30 (from the U-maze's A30 units) | about 1.53 | about 18.9 | about 3.6 |
| 128 | A10 (not measured) | 1.1 to 1.5 expected | 19 to 26 | 2.6 to 3.5 |
| 128 | A40 (UVA, measured) | 1.106 | 26.1 | 2.6 |

The seconds per iteration are the mean of Medium and Large where both were measured.

- **A100 80 GB**: the A100 40 GB's times are used. The U-maze's real units of 128 policies took the same time on
  both (1.163 s median on the 80 GB card against 1.171 to 1.177 s here for Medium and Large).
- **A30**: not measured on these maps. The U-maze's real units of 128 policies took 1.31 times as long on an A30
  as on an A100 80 GB (1.50 to 1.53 s against 1.16 s), and the map adds about 1%, so 1.53 s is used for 128
  policies. That ratio grows with the work per iteration (1.06 at 4 environments per policy, 1.80 at 128), so for
  256 or more policies an A30 is slower than the factor says; no number is given.
- **A10**: not measured on any map, at UVA or at Gilbreth. The card facts take it as the RTX A4500 / A40 class;
  the A40 took 1.10 to 1.11 s for 128 policies, so between 1.1 and 1.5 s is expected. The first unit on an A10 is
  the measurement.
- **A whole unit of 16.8 million steps per policy** (8,192 iterations at 16 environments) takes, with 128
  policies: 2.8 hours on an A100 (80 or 40 GB) and about 3.6 hours on an A30, start-up included, so it fits one
  4-hour job on both; with 256 policies, 3.6 hours on an A100, and on an A30 it needs a second job.

### Policies per unit, and `minimum_card_memory_mb`

Card memory is the same on the three maps (A40, 128 policies x 16 environments: 2,609, 2,607 and 2,607 MiB), so
the U-maze's real units give the memory a unit really needs. Their largest readings: 4,903 MiB for 128 policies x
16 environments (most units read about 2,860; four read 4,690 to 4,903), and 17,385 MiB for 128 x 128 on A30 and
A100 cards (four units on A100 80 GB cards read 33,765 to 33,769 MiB; the A30 units of that shape ran at 17,169 on
a 24 GB card, so the larger readings are not what the unit needs). `build_units.py` puts a margin of 1.25 on the
largest peak.

Define $P$ as the policies in a unit at 16 environments per policy. On the A100 40 GB the card memory of the
profile grows by 16.9 MiB per policy (4,941 MiB at $P = 256$, 9,263 MiB at $P = 512$), and the U-maze's largest
real-unit reading at $P = 128$ is 1.76 times this profile's reading of the same shape (4,903 against 2,779 MiB). The
column `minimum_card_memory_mb` below is 1.25 times the profile's reading times 1.76, rounded up to a thousand (for
$P = 128$, 1.25 times the real units' 4,903 MiB). It sets which cards may take a unit; the largest $P$ per card is
the largest power of two whose value fits the card. (The readings are in MiB and the field in MB, 4.9% apart; the
margin covers it.)

| policies per unit, 16 environments each | profile card memory (MiB) | `minimum_card_memory_mb` | card classes it fits |
|---|---|---|---|
| 64 | 1,719 | 4,000 | every card of 8 GB or more |
| 128 | 2,779 | 7,000 | every card of 8 GB or more |
| 256 | 4,941 | 11,000 | 11 GB cards and up |
| 512 | 9,263 | 21,000 | A30 and A10 (24 GB), A40, A100 40 and 80 GB; not 20 GB cards |
| 128, with 128 environments each | 17,311 | 22,000 (1.25 x 17,385, the U-maze's real units) | A30 and A10 (24 GB), A40, A100 |

Largest unit by memory alone: 512 policies on A30 and A10 (24 GB), 1,024 on A100 40 GB and A40, 2,048 on A100
80 GB (not measured above 512).

- **128 policies per unit** stays the choice when a 16.8-million-step unit must finish inside one 4-hour job on
  every Gilbreth card class.
- **256 policies per unit** gives 52% more total steps per second on an A100, and a 16.8-million-step unit still
  fits one job there (3.6 hours), but not on an A30.
- `minimum_card_memory_mb = 20000`, which the U-maze descriptions use for both shapes, also works for Medium and
  Large: it is above the largest real reading of both (4,903 and 17,385 MiB), with less than the 1.25 margin for
  128 environments per policy. Every Gilbreth card has at least 24 GB, so on Gilbreth the value changes nothing;
  a smaller value only lets UVA cards of 8 to 16 GB take the 16-environment units.

## Configuration files and tests

- The two files above were written from the U-maze file by replacing seven lines and the header; every trainer
  field, every physics field and the runtime other than the selector are the U-maze's, byte for byte.
- `tests/integration/test_experiment_configuration_files.py`: the two files join the study examples; the count of
  AntMaze files goes from 4 to 6; a new test checks that each file is the U-maze file with the map values of
  `am_common.preset` (read from the source, not imported) and the runtime selector, and that a file whose map and
  selector disagree is refused.
- Test jobs on puma01 (through the speed study's `infrastructure/submit_tests.sh`, interpreter platform_jax):
  1. job 6908716: `tests/integration/test_experiment_configuration_files.py`,
     `tests/integration/test_training_configuration_file_cli.py` (rebuilds every recorded file's trainer and
     environment through the training command, the two new files included) and `tests/maze_campaign/`:
     126 passed;
  2. job 6908740, after the commit: the two integration files, `tests/envs/test_antmaze_mjx.py` and
     `tests/envs/test_antmaze_fault_guard.py`: 30 passed, 6 skipped (the Warp cases of the guard test, which
     need the platform_jax_warp interpreter; the Warp physics of these maps is covered by the gates above).
- READMEs: `configs/experiments/README.md` (the three per-map stable-physics files) and
  `src/exploration_platform/envs/antmaze/README.md` (the files and these notes).
- Commits: `e54ab8e8` (the two configuration files, the test, the two READMEs, the docstring count); these
  notes and the scripts are in the commit after it.

## Scripts and records

| file | what it is |
|---|---|
| `scripts/map_gates.py` | the gates of one map, against C MuJoCo, on processors |
| `scripts/processor_job.sh` | runs a script of this folder with the platform_jax_warp interpreter on processors |
| `scripts/profile_unit_shapes.py` | one unit shape per process on a card: time, card memory, Warp pool use |
| `scripts/card_job.sh` | runs groups of shapes on one card within a budget |
| `scripts/summarize.py` | prints the throughput table and the gate numbers from `records/` |
| `records/` | the gate records (JSON) and the profile records (one JSON line per shape) |
| `logs/` | the job logs |
