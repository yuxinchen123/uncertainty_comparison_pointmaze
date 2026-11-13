# antmaze

The Gymnasium Ant in the three maze maps (U-maze, medium, large), specified in `spec.md`.

| file | what it holds |
|---|---|
| `am_common.py` | the configuration (`AntMazeConfig`), the three presets, the model's XML with the walls and the ceiling |
| `mjx_antmaze.py` | `JaxAntMaze`: the physics stepped by MJX's JAX implementation |
| `warp_antmaze.py` | `WarpAntMaze`: the same model stepped by MuJoCo Warp, called from JAX |
| `faults.py` | the fault record both implementations keep: steps in the ceiling zone and the largest torso speed, per policy |

`AntMazeConfig.physics` chooses between the two: `"jax"` (the default, and the stepper of every
run before 2026-11-12) or `"warp"`. The composer (`training/compose.py`) reads it; the training
command sets it with `--physics` or from a configuration file's `[environment]`.

## The model

`am_common.build_antmaze_xml` builds one model per map from the vendored `assets/ant.xml`:

1. **The ant** of Gymnasium, unchanged: a torso sphere of radius 0.25 m and four legs of three
   capsules each, 8 motors of 150 N m at full control.
2. **The floor**, the plane of `ant.xml`.
3. **The walls**: boxes 2 m high, one per horizontal run of wall cells (a cell is 4 m x 4 m).
4. **The ceiling** (since 2026-11-12): one plane on the wall tops, at 2 m, whose normal points
   down. It has the floor's contact parameters and is not drawn.
5. The solver settings of `spec.md` (implicitfast, 4 solver iterations, 8 line-search iterations).

Which geoms collide is decided by two bit masks per geom (a pair collides when the `contype` of
one shares a bit with the `conaffinity` of the other):

| geom | `contype` | `conaffinity` | collides with |
|---|---:|---:|---|
| floor, wall boxes | 1 | 1 | every geom of the ant |
| torso sphere | 3 | 0 | floor, walls, ceiling |
| leg capsules (12) | 1 | 0 | floor, walls |
| ceiling | 0 | 2 | the torso sphere only |

The ceiling adds exactly one geom pair, torso against ceiling; the floor and wall pairs are those
of the model before it (`tests/envs/test_antmaze_ceiling.py` compares the pair sets of all three
maps).

## The ceiling

The ant's motors are strong for its weight (150 N m per joint on a body of 0.91 kg), so torques
can lift it off the floor. Before the ceiling, random torques sent some ants above the 2 m walls
and across them (`spec.md`, deviation 8). This is a property of the model and not of one
implementation: in C MuJoCo itself, on the model without the ceiling, 83 of 1,024 ants with
random torques had the torso above the wall tops within 700 steps and 4 of them were over a wall
cell (largest height 4.73 m). With the ceiling the torso cannot rise above the wall tops, so it
cannot cross a wall: the same torques in C MuJoCo then give no step above the wall tops and none
over a wall cell.

- The torso sphere comes within the contact margin (0.01 m) of the ceiling when the torso centre
  is at **1.74 m** (`CEILING_CONTACT_HEIGHT`).
- An ant thrown upward from the spawn stops there. Largest height of the torso centre, in C
  MuJoCo, MJX JAX and MuJoCo Warp alike: 1.775 m when thrown at 10 m/s and 1.785 m at 20 m/s; in
  C MuJoCo at most 1.981 m for throws of up to 40 m/s. One physics step from states in ceiling
  contact agrees with C MuJoCo to 4.9e-7 m (JAX) and 5.1e-8 m (Warp).
- Ants that stood, walked, fell over or lay on their back did not touch it. Largest height of the
  torso centre over 10 s from the spawn (0.75 m), on processors, in C MuJoCo, JAX and Warp:

  | ant | largest torso height (m) |
  |---|---:|
  | stands with no torque | 0.79 |
  | stands with all four ankles held at an end of their range by full torque | 0.77 to 0.80 |
  | walks, one leg after the other (1 Hz at a quarter of the full torque; 0.5 Hz at full torque) | 0.79 to 0.83 |
  | falls over, tumbling at 8 rad/s with no torque | 1.23 to 1.26 |
  | lies on its back, then moves its legs with the walking torques | 0.74 |

- An ant that pushes hard with several legs at once does leave the floor and can reach the
  ceiling by its own leg power: of 36 periodic torque patterns tried in C MuJoCo, 9 brought the
  torso to the ceiling (7 of the 12 with full torque, and 2 of the 4 with half torque at 2 Hz;
  none of the 12 with a quarter of the full torque, whose largest height was 1.44 m). The
  Gymnasium Ant environment calls a torso height from 0.2 m to 1.0 m healthy; AntMaze does not
  end the episode outside it, and neither does this environment.

Measurements: section "After the ceiling" of
`benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_*/profiles/antmaze_warp_training/antmaze_warp_training.md`.

## The physics fault guard

Two fields of the configuration, both 0 by default (guard off): `fault_torso_speed` in metres per
second and `fault_torso_angular_speed` in radians per second. With either above zero, an
environment is respawned exactly like one whose state is not finite (no reward, no arrival, a
truncation whose final observation is the spawn, one more in `nan_count` and in the window's
`physics_fault_respawns`) when, after any physics step of the environment step,

1. the torso centre moved faster than `fault_torso_speed` (when above zero), or
2. the torso turned faster than `fault_torso_angular_speed` (when above zero), or
3. the torso centre was outside the maze's volume: beyond the outer faces of the outer walls,
   below the floor or above the ceiling plane.

Why, the measurements and the values the study uses (30 m/s and 100 rad/s): spec.md, deviation 9.
Tests: `tests/envs/test_antmaze_fault_guard.py`.

## The fault record

A policy that learns to get itself thrown upward would still reach the ceiling, and an ant that
turns fast in the air can be sped up by the integrator itself (spec.md, deviation 9). Both
implementations therefore keep four numbers per policy in the environment state, summed inside
the compiled rollout (`faults.py`):

| field of the state | window record of the training command | what it holds |
|---|---|---|
| `ceiling_zone_steps` | `ceiling_zone_steps_per_copy` | environment steps in which the torso centre was above 1.5 m (`CEILING_ZONE_HEIGHT`: 0.5 m under the ceiling, 0.24 m under first contact) after any of the step's physics steps |
| `largest_torso_speed` | `largest_torso_speed_per_copy` | the largest speed of the torso centre after any physics step, in metres per second |
| `physics_fault_respawns` | `physics_fault_respawns_per_copy` | environment steps that ended in a respawn because the physics state was not finite or, with the guard on, beyond the guard's bounds |
| `largest_torso_angular_speed` | `largest_torso_angular_speed_per_copy` | the largest angular speed of the torso after any physics step, in radians per second |

- The training command reads all four at the end of each logging window and sets them back to
  zero (`fault_window`); they are part of the state, so a unit that is stopped and continued
  keeps the window's partial numbers.
- `respawn` sets them to zero, so the first window holds no priming step.
- The two largest speeds include the physics steps of an environment step that ended in a
  respawn, so a window shows what the guard stopped; the policy never sees those values.
- A height or speed that is not a finite number counts as outside the zone and as speed 0; that
  environment is put back at the start and counted in `nan_count`, as before.
- PointMaze has no such record and its window records are unchanged.

What an untrained policy gives, for comparison (random torques, 16,384 ants for 1,400 steps on
the U-maze, H100 NVL, 2026-11-12, with the ceiling in the model):

| physics | environment steps in the ceiling zone | episodes of 700 steps with such a step | largest torso speed (m/s) | steps with the torso above the wall tops | episodes that ended at the goal |
|---|---:|---:|---:|---:|---:|
| MuJoCo Warp | 75,724 of 22,937,600 (0.33%) | 9,886 of 32,768 | 9.75 | 0 (51,818 before the ceiling) | 0 (2 before) |
| MJX JAX | 82,377 of 22,937,600 (0.36%) | 10,569 of 32,768 | 19.64 | 0 (63,100 before the ceiling) | 0 (15 before) |

A policy whose windows show a much larger share of steps in the ceiling zone than 0.33% has
learned to leave the floor.

## What differs in the Warp stepper

It needs an interpreter with warp-lang (`/p/rlprojects/RND/.venvs/platform_jax_warp/bin/python`).

1. **One contact pool for all environments**, of `warp_contact_slots_per_environment` (48) slots
   per environment, and at most `warp_constraint_rows_per_environment` (512) constraint rows per
   environment. MuJoCo Warp drops what does not fit; `raise_on_overflow` then raises, and the
   training command calls it after priming, at every window end, before every checkpoint, and
   between window ends about once a minute. The torso-ceiling pair enters the pool only while the
   torso is within 0.27 m of the ceiling.
2. **A kernel folder** on node-local storage (`prepare_kernel_folder`), created if missing.
3. **Non-finite states**: an environment whose positions or velocities are non-finite when the
   step starts, or after any of its physics steps, is respawned and counted. The test at the
   start is needed because MuJoCo Warp turns a torso quaternion that holds a NaN into a valid
   one.
4. **Repeatability**: bit for bit on processors. On a card the order of the contacts in the pool
   changes from run to run, so two runs of the same program differ.

Measurements and checks:
`benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_*/profiles/antmaze_warp/antmaze_warp.md`
(the environment) and `.../profiles/antmaze_warp_training/antmaze_warp_training.md` (the whole
training iteration).

The two counting states of the plan, `gt_position_maze_cell` (the maze cell the ant is in) and
`gt_position_1m` (position rounded to a 1 metre grid), are not implemented.
