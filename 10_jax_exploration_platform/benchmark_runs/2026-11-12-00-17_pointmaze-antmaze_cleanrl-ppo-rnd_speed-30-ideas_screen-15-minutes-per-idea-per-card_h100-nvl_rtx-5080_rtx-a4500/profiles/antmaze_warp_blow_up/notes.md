# AntMaze with MuJoCo Warp: ants whose torso speed runs away

Times are Pacific (PT), read from the clock on 2026-11-12; the machines run on Eastern time.

## What was seen in training

The partial records of the rollout-shape check on the U-maze (read 18:05 PT) showed, in most
configurations, 2 to 9 of 16 policies with a largest torso speed above 10,000 m/s at some logging
window (up to 2.5e15 m/s). Untrained ants stay near 9 to 10 m/s.

## Finding

1. **The cause is the integrator, not the solver.** `implicitfast` at a time step of 0.01 s (our
   deviation from the reference model, which uses RK4) is not stable for an ant that turns fast
   while it is off the floor. It takes the forces that depend on the turning rate from the start of
   the physics step, which adds rotational energy at every step; in the air nothing takes it away.
2. **More solver or line-search iterations change nothing** (the first suspect, refuted).
3. **RK4, or `implicitfast` at 0.005 s, do not run away** with the same policies; they cost 3.59
   and 1.97 times the time per training iteration.
4. **A guard that respawns an environment whose torso passes 30 m/s or 100 rad/s** ends every
   runaway before a large speed exists, at no measurable cost. It does not correct the inflated
   turning below its bounds.

## How it was reproduced

One checkpoint of the running check was copied from Gilbreth (read only): the unit with 128
environments per policy and cells 33 to 40 of 48 (learning rates 1e-3 and 3e-3), written at
iteration 336 of 1,024. `units/roll_out_checkpoint_policies.py` rebuilds the
unit's trainer, restores the checkpoint and steps the 16,384 environments with the 128 saved
policies, with fresh action noise and no learning. An environment step whose torso speed passes
100 m/s, while the step before it in the same episode had not, is an onset; the script keeps the
state before each of the 24 environment steps that lead up to it, and the actions.

With the physics of training, 300 s of rollout on serval05's H100 NVL (unit 1, 18:09 to 18:17 PT)
gave 33 onsets in 362 million environment steps (0.09 per million), 38 episodes that reached 30
to 100 m/s, and 26 respawns for a non-finite state.

## The mechanism, from the 33 recorded onsets

- **Where the ant is.** Off the floor: the median torso height over the 24 steps before an onset
  is 1.1 to 1.3 m (an ant at rest has its torso at 0.38 m), mostly with no contact at all.
- **What grows first.** The torso's angular speed. Its median over the 33 onsets, before each of
  the last 25 environment steps (rad/s): 32, 36, 36, 37, 38, 39, 41, 41, 44, 48, 49, 53, 56, 55,
  62, 67, 73, 78, 86, 89, 94, 109, 138, 188, 326. The torso speed stays near 3 to 5 m/s until the
  last few steps.
- **Then** joints pass their ranges by whole radians, the ant is driven into the ceiling, a wall
  or the floor at hundreds of metres per second, and the values grow without bound.
- **Replayed in C MuJoCo** (`units/replay_events.py`, puma01) from the state 2, 4, 8 or 24
  environment steps before the onset, with the recorded actions. Number of the 33 onsets that
  reach 100 m/s again:

  | integrator and settings in C MuJoCo | from 2 steps before | from 4 | from 8 | from 24 |
  |---|---|---|---|---|
  | `implicitfast`, 4 solver and 8 line-search iterations (training) | 32 | 32 | 31 | 20 |
  | `implicitfast`, 8 and 16 | 32 | 32 | 31 | 20 |
  | `implicitfast`, 16 and 16 | 32 | 32 | 31 | 20 |
  | `implicitfast`, 100 and 50 (MuJoCo's defaults) | 32 | 32 | 31 | 20 |
  | `Euler`, 4 and 8 | 32 | 32 | 31 | 20 |
  | `RK4`, 100 and 50 (the reference model) | 0 | 0 | 0 | 0 |
  | `RK4`, 4 and 8 | 0 | 0 | 0 | 0 |
  | `implicit` (with the turning-rate derivatives), 4 and 8 | 0 | 0 | 0 | 0 |
  | `implicitfast`, 4 and 8, time step 0.005 s | 27 | 11 | 1 | 2 |
  | `implicitfast`, 4 and 8, time step 0.0025 s | 7 | 0 | 0 | 0 |

  From 24 steps before, RK4 keeps the largest torso speed at 5.4 m/s and the largest angular speed
  at 44 rad/s (medians); `implicitfast` reaches 205 m/s and 816 rad/s.
- **The 38 episodes that reached 30 to 100 m/s are the same thing:** 27 to 28 of them reach
  100 m/s in C MuJoCo with `implicitfast` from 2 to 8 steps before, 0 or 1 with RK4.
- **MuJoCo Warp on processors** repeats the onsets with every iteration count: 32 or 33 of the 33
  at 4/8, 8/16, 16/16 and 50/50, from every start.
- C MuJoCo prints "Nan, Inf or huge value in QACC ... The simulation is unstable" in these replays
  and puts the simulation back to the model's default state. Our environment has no such reset
  for finite values.

## Corrections tried on the card

The same checkpoint and policies, 16,384 environments. Rollouts of 300 s on serval05 (units 1, 5,
7, 8, 12) and of 45 s on serval02 card 1 (units 9, 10, 11, 13).

| physics | machine | environment steps (million) | onsets above 100 m/s | respawns (non-finite or guard) | largest torso speed after any physics step, m/s | environment steps ending above 60 rad/s |
|---|---|---|---|---|---|---|
| `implicitfast`, 0.01 s (training) | serval05 | 362 | 33 | 26 | 5.7e13 | not counted |
| `implicitfast`, 0.01 s (training) | serval02 card 1 | 57 | 5 | 4 | 1.2e11 | 208 |
| `implicitfast`, 0.01 s, 16 solver and 16 line-search iterations | serval02 card 1 | 46 | 3 | 3 | 2.1e10 | 113 |
| `RK4`, 0.01 s | serval05 | 104 | 0 | 0 | 14.6 | 0 |
| `RK4`, 0.01 s | serval02 card 1 | 18 | 0 | 0 | 13.5 | 0 |
| `implicitfast`, 0.005 s, 10 physics steps | serval05 | 127 | 0 | 0 | 14.3 | 1 |
| `implicitfast`, 0.005 s, 10 physics steps | serval02 card 1 | 31 | 0 | 0 | 13.9 | 0 |
| training physics, guard at 30 m/s | serval05 | 109 | hidden by the guard | 7 | 1.1e3 | 298 |
| training physics, guard at 30 m/s | serval02 card 1 | 62 | hidden by the guard | 12 | 2.4e7 | 269 |
| training physics, guard at 30 m/s and 100 rad/s | serval05 | 92 | hidden by the guard | 22 | 17.9 | 235, none above 100 |
| training physics, guard at 30 m/s and 100 rad/s | serval02 card 1 | 62 | hidden by the guard | 12 | 15.4 | 137, none above 100 |

- With only the speed bound, a runaway is stopped one environment step later than with the angular
  bound: inside that last step the speed still reaches 1e3 to 1e7 m/s (the policy does not see it,
  the final observation of a respawned environment is the spawn), and some steps before it end
  with 100 to 1,000 rad/s in the observation. With both bounds the largest speed after any physics
  step is 15 to 18 m/s and the largest angular speed 127 to 135 rad/s.
- Share of environment steps ending with the torso turning at 20 to 40 rad/s: 0.53% with the
  training physics, 0.28% with RK4, 0.29% at 0.005 s. `implicitfast` at 0.01 s inflates the
  turning of airborne ants also where no episode runs away.
- The full `implicit` integrator is refused by MJX 3.11.0 (`NotImplementedError` in
  `_put_option`, unit 6). Forced past the check (`units/full_implicit_in_warp.py`), MuJoCo Warp's
  step differs from C MuJoCo's by up to 6.9e-3 m in one step at contact-free states (the gate is
  1e-4) and 30 to 33 of the 33 onsets run away again. Not usable.

## Cost

Whole training iterations (128-step rollout and update) of the unit, 128 policies with 128
environments each, on serval02 card 1 (H100 NVL), 6 iterations timed after 2 warm-up calls. The
hours per million steps per copy are $\dfrac{10^{6}}{3600 \times r}$ with $r$ the steps per second
of one copy.

| physics | copies | seconds per iteration | total steps per second (millions) | steps per second per copy (thousands) | hours per million steps per copy |
|---|---|---|---|---|---|
| `implicitfast`, 0.01 s (training), guard off | 128 | 1.374 | 1.526 | 11.92 | 0.0233 |
| same, guard at 30 m/s | 128 | 1.375 | 1.525 | 11.92 | 0.0233 |
| same, guard at 30 m/s and 100 rad/s | 128 | 1.380 | 1.520 | 11.88 | 0.0234 |
| `implicitfast`, 0.01 s, 16 solver and 16 line-search iterations, guard off | 128 | 1.690 | 1.241 | 9.70 | 0.0286 |
| `implicitfast`, 0.005 s, 10 physics steps | 128 | 2.707 | 0.775 | 6.05 | 0.0459 |
| `RK4`, 0.01 s | 128 | 4.927 | 0.426 | 3.33 | 0.0835 |

serval05 gave 1.388 s for the training physics in unit 1. From 18:23 PT processes of another
account (pyt9dp) were on serval05's card, so the timings of units 5, 7, 8 and 12 there are
discarded (the units end with exit code 90); their counts of onsets and speeds do not depend on
the timing and are used.

## The guard

Code: `src/exploration_platform/envs/antmaze/faults.py` (`beyond_bounds`), both implementations,
configuration fields `fault_torso_speed` and `fault_torso_angular_speed`, both 0 (off) by default.
Description and the reason for the bounds: `spec.md` of the environment, deviation 9. Tests:
`tests/envs/test_antmaze_fault_guard.py`.

Bounds for the study: **30 m/s and 100 rad/s**.

- Under RK4 and under `implicitfast` at 0.005 s (280 million environment steps in all) the largest
  torso speed after any physics step was 14.6 m/s, no environment step ended above 80 rad/s and
  one ended above 60.
- 30 m/s is twice that speed. 100 rad/s is above every turning rate the two stable settings
  showed and below the 138 rad/s (median) of two steps before an onset.
- With both bounds: about 0.2 respawns per million environment steps for these policies (one
  episode in about 6,000).

## Units and records

- `units/roll_out_checkpoint_policies.py`: the rollout, the onsets, the timing.
- `units/replay_events.py`: replay of the onsets in C MuJoCo and in MuJoCo Warp on processors.
- `units/full_implicit_in_warp.py`: the forced full implicit integrator and two gates with it.
- `units/read_record.py`: prints the main numbers of rollout records.
- `units/h100-nvl_unit-*.sh` (serval05 queue, kind profile), `units/serval02-card-1_unit-*.sh`
  (serval02 card 1 under its lock), `units/replay_job.sh` (puma01).
- `records/*.json` and `records/*_events.npz` (not in git), `logs/`.
- Units 2, 3 and 4 (8/16, 16/16, 50/50 on the card) stopped after seconds at 18:17 PT because the
  environment's new configuration field was not yet in the copied configuration file. The replays
  on processors then showed that the iteration counts do not matter; 16/16 was run on the card
  as unit 13 (3 onsets in 46 million steps, 1.23 times the time per training iteration).
