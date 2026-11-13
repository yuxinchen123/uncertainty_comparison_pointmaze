# Ant integrators in MuJoCo Warp: speed, gates and runaways

Measurements of 2026-11-13, 11:57 to 12:45 PT, by a helper of the main session; measurement only. No default,
configuration file under `configs/`, queue, fleet table, `STATE.md` or running job was changed. Every Slurm job id
is in `slurm/submitted_jobids_integrator-check_sl5nw.txt` of this run folder. Times are Pacific, converted from
machines running Eastern.

## Result

1. **RK4 at 0.01 s x 5 (the reference ant model's integrator and time step) removes the runaways at the smallest
   cost among the settings that keep the ant's motion.**
   - It passes every gate.
   - With the trained policies it had 0 respawns in about 65,600 episodes on each 128-environment checkpoint, where
     the current physics had 33.6 per 10,000 episodes at 83 million steps and 486.9 per 10,000 at 125 million.
   - Its arrival rates are those of RK4 at 0.005 s (within 0.004 in total).
   - It costs 1.86 times the current time per training iteration at 128 policies x 16 environments and 1.83 times
     at 512 x 16 (A100 40 GB); 2.01 times on the H100 NVL at 128 x 16 (another account's process was on that card).
   - It needs no code change: the configuration's three values (`integrator = "RK4"`, `physics_time_step = 0.01`,
     `frame_skip = 5`) reach the Warp model.
2. **RK4 at 0.005 s x 10** also has no respawns, at 3.66 and 3.56 times the current time (about 4.0 on the H100).
3. **The `implicit` integrator as MJX 3.11.0 ships it in MuJoCo Warp is wrong.** Its turning-rate term has the
   opposite sign to C MuJoCo's. It fails the agreement gate by a factor of 50 to 170 and runs away more often than
   the current physics: 468 to 553 respawns per 10,000 episodes at 0.005 s, and about half of all episodes at
   0.01 s. MJX also refuses it without a patch.
4. **`implicit` with the sign corrected** (two patches inside this folder's scripts) passes every gate and had no
   respawns. It costs 1.55 to 1.71 times the current time at 0.005 s x 10, and 0.79 to 0.80 times at 0.01 s x 5:
   at 0.01 s it is cheaper than the current physics. But the same policies arrive two thirds as often as under RK4
   at 0.005 s and half as often at 0.01 s, so it damps the ant's motion and changes the task.
5. **The late gains of the runaway configurations depend on the current physics.** At 125 million steps the four
   configurations with learning rate 1e-2 arrive in 0.090 to 0.125 of episodes in the current physics against
   0.076 to 0.083 under RK4, and in fewer steps (454 to 493 against 514 to 528). The configurations with learning
   rate 3e-2, which never run away, arrive equally often in both. Those policies use part of what the
   implicitfast error allows.

My reading: if the physics is changed, RK4 at 0.01 s x 5 is the candidate. It is the reference model's setting,
its arrival rates are those of RK4 at half the time step, and it costs about 1.85 times the current time per
iteration. Corrected `implicit` is cheaper but needs a patched library and gives a more damped ant. Any change
makes every unit trained so far a different environment. The decision is the user's.

## The settings

All with 4 solver and 8 line-search iterations, the fault guard at 30 m/s and 100 rad/s, the ceiling on the wall
tops, the episode ending at the goal, and one environment step of 0.05 s. No candidate needed more iterations to
pass.

| setting | integrator | physics time step | physics steps per environment step | Warp model integrator number |
|---|---|---|---|---|
| implicitfast, 0.005 s x 10 (current) | implicitfast | 0.005 s | 10 | 3 |
| implicit as shipped, 0.005 s x 10 | implicit | 0.005 s | 10 | 2 |
| RK4, 0.005 s x 10 | RK4 | 0.005 s | 10 | 1 |
| RK4, 0.01 s x 5 | RK4 | 0.01 s | 5 | 1 |
| implicit as shipped, 0.01 s x 5 | implicit | 0.01 s | 5 | 2 |
| implicit with the sign corrected, 0.005 s x 10 | implicit, sign patch | 0.005 s | 10 | 2 |
| implicit with the sign corrected, 0.01 s x 5 | implicit, sign patch | 0.01 s | 5 | 2 |

The configuration files (`configurations/`, written by `scripts/write_configurations.py`) are the U-maze
stable-physics example file with only `integrator`, `physics_time_step` and `frame_skip` changed. The last column
is `int(env.model.opt.integrator)` read in every gate and rollout record (MuJoCo's numbering: 1 RK4, 2 implicit,
3 implicitfast).

## Speed

Define $s$ as the median seconds of one whole training iteration (128-step rollout of every environment and the
update), $C$ as the number of policies and $e = 16$ as the environments per policy. Then:

- total steps per second: $\dfrac{128\thinspace e\thinspace C}{s}$;
- steps per second per policy: $r = \dfrac{128\thinspace e}{s}$;
- hours per million steps per policy: $\dfrac{10^{6}}{3600\thinspace r}$;
- the last column is $\dfrac{s}{s_{\text{current}}}$ at the same $C$ on the same card.

The trainer is the real one, built from the configuration file as `scripts/run_training.py` builds it, with
untrained policies, two warm-up calls before timing (compile time apart), and rounds of about 2 s
(`scripts/profile_integrators.py`).

**A100 40 GB, cheetah01** (job 6908901, 12:05 to 12:31 PT; no other process on the card at the end of the job):

| setting | policies | seconds per<br>iteration | total steps per<br>second (millions) | steps per second<br>per policy (thousands) | hours per million<br>steps per policy | time against<br>current |
|---|---:|---:|---:|---:|---:|---:|
| implicitfast, 0.005 s x 10 (current) | 128 | 1.162 | 0.226 | 1.762 | 0.158 | 1.00 |
| implicit as shipped, 0.005 s x 10 | 128 | 1.823 | 0.144 | 1.123 | 0.247 | 1.57 |
| RK4, 0.005 s x 10 | 128 | 4.260 | 0.062 | 0.481 | 0.578 | 3.66 |
| RK4, 0.01 s x 5 | 128 | 2.168 | 0.121 | 0.945 | 0.294 | 1.86 |
| implicit as shipped, 0.01 s x 5 | 128 | 0.930 | 0.282 | 2.203 | 0.126 | 0.80 |
| implicitfast, 0.005 s x 10 (current) | 512 | 2.327 | 0.451 | 0.880 | 0.316 | 1.00 |
| implicit as shipped, 0.005 s x 10 | 512 | 3.610 | 0.290 | 0.567 | 0.490 | 1.55 |
| RK4, 0.005 s x 10 | 512 | 8.283 | 0.127 | 0.247 | 1.123 | 3.56 |
| RK4, 0.01 s x 5 | 512 | 4.264 | 0.246 | 0.480 | 0.578 | 1.83 |
| implicit as shipped, 0.01 s x 5 | 512 | 1.841 | 0.570 | 1.112 | 0.250 | 0.79 |

The sign correction changes one `atomic_sub` into an `atomic_add` in one kernel, so the corrected `implicit` costs
what the shipped one costs. A second A100 job for the corrected rows (6909096) did not start within two minutes
(another user took the card) and was cancelled; the H100 rows below measure the corrected setting directly.

**H100 NVL, serval05** (serval05 queue, profile unit `2026-11-13-12-10_profile-antmaze-integrator-check-speed-h100`,
from 12:25 PT). A process of another account (pyt9dp, 7,558 MiB) was on the card for the whole unit, so these
timings are not clean (the unit ends with exit code 90); they are shown for the ratios, which agree with the A100's:

| setting | policies | seconds per<br>iteration | total steps per<br>second (millions) | steps per second<br>per policy (thousands) | hours per million<br>steps per policy | time against<br>current |
|---|---:|---:|---:|---:|---:|---:|
| implicitfast, 0.005 s x 10 (current) | 128 | 1.006 | 0.260 | 2.035 | 0.137 | 1.00 |
| RK4, 0.005 s x 10 | 128 | 4.020 | 0.065 | 0.509 | 0.545 | 3.99 |
| RK4, 0.01 s x 5 | 128 | 2.026 | 0.129 | 1.011 | 0.275 | 2.01 |
| implicit with the sign corrected, 0.005 s x 10 | 128 | 1.716 | 0.153 | 1.193 | 0.233 | 1.71 |
| implicit with the sign corrected, 0.01 s x 5 | 128 | 0.806 | 0.325 | 2.542 | 0.109 | 0.80 |

At 512 x 16 the same unit gave 1.751 s (current), 12.536 s (RK4 at 0.005 s, 7.16 times) and 5.207 s (corrected
`implicit` at 0.005 s, 2.97 times); RK4 at 0.01 s was not in the unit at this shape. Those two ratios
are twice the A100's at the same shape (3.56 and 1.55), so I read them as the other process taking the card during
those minutes and do not use them. The unit ran from 12:25 to 12:40 PT.

With the trained policies the rollout alone gives the same order (seconds per 70 steps of 16,384 environments on
an RTX 5080, from the rollouts below): current 1.23, RK4 at 0.01 s 2.33 (1.9 times), RK4 at 0.005 s 4.58 (3.7
times). Card memory hardly depends on the setting (2.8 to 2.9 GB at 128 x 16 and 9.3 to 9.4 GB at 512 x 16 on the
A100). No contact or constraint row was dropped in any measurement.

## Gates

`scripts/integrator_gates.py` on puma01's processors (MuJoCo Warp on the processor device, C MuJoCo in double
precision), jobs 6908916 to 6908920 and 6909052, 6909053. The reset, goal, step-limit, guard, wall and ceiling gates
are those of `diagnosis/2026-11-13_medium-and-large-readiness/scripts/map_gates.py` on the U-maze. The agreement
gates compare Warp with C MuJoCo **running the same integrator, time step and iterations**. Bounds: one environment
step from 50 random-torque states, 1e-4 m contact-free and 5e-2 m in contact; 20 shared random-torque steps from
the spawn, 1e-4 m before the first contact and under one 4 m cell over all; resting height 0.38248 m within
1e-4 m. Records: `records/gates_<setting>.json`.

| setting | reset, goal, step limit,<br>guard, walls, ceiling | one step,<br>contact-free (m) | one step,<br>in contact (m) | 20 steps, before<br>first contact (m) | 20 steps,<br>all (m) | resting height,<br>Warp and C (m) | all |
|---|---|---:|---:|---:|---:|---|---|
| implicitfast, 0.005 s x 10 (current) | all passed | 9.3e-7 | 8.4e-3 | 7.5e-7 | 0.16 | 0.38248, 0.38248 | passed |
| RK4, 0.005 s x 10 | all passed | 6.8e-7 | 2.3e-2 | 3.3e-7 | 0.34 | 0.38248, 0.38248 | passed |
| RK4, 0.01 s x 5 | all passed | 5.1e-5 | 1.1e-2 | 3.3e-7 | 0.38 | 0.38248, 0.38248 | passed |
| implicit as shipped, 0.005 s x 10 | all passed | 5.0e-3 | 6.6e-3 | 6.6e-3 | 0.19 | 0.38248, 0.38248 | **failed** |
| implicit as shipped, 0.01 s x 5 | all passed | 1.7e-2 | 1.9e-2 | 4.1e-2 | 0.58 | 0.38248, 0.38248 | **failed** |
| implicit with the sign corrected, 0.005 s x 10 | all passed | 1.2e-6 | 5.9e-3 | 4.2e-7 | 0.15 | 0.38248, 0.38248 | passed |
| implicit with the sign corrected, 0.01 s x 5 | all passed | 1.5e-6 | 1.0e-2 | 4.6e-7 | 0.11 | 0.38248, 0.38248 | passed |

RK4 at 0.01 s is 75 times further from C at contact-free states than at 0.005 s (5.1e-5 m against 6.8e-7 m), still
under the bound. Both sides run the same method at the same step, so I read it as single-precision rounding over the
longer step.

### Why the shipped `implicit` fails

`scripts/implicit_four_way.py` (jobs 6908939, 6909039) steps the 50 states of the one-step gate with four steppers:
C implicit, C implicitfast, Warp implicit as shipped, Warp implicitfast. Define the implicit correction of a
simulator as its implicit step minus its implicitfast step from the same state (the vector of position
differences): $w$ for Warp and $c$ for C. The length ratio is the length of $w$ divided by the length of $c$. At the
contact-free states, largest position differences in metres:

| time step | C implicit against<br>C implicitfast | Warp implicitfast against<br>C implicitfast | Warp implicit against<br>C implicit | median cosine<br>of $w$ and $c$ | median<br>length ratio |
|---|---:|---:|---:|---:|---:|
| 0.005 s x 10 | 2.5e-3 | 1.2e-6 | 5.0e-3 | -0.9999 | 0.997 |
| 0.01 s x 5 | 8.3e-3 | 1.3e-6 | 1.7e-2 | -0.9995 | 0.991 |

Warp's correction has C's length and the opposite direction. Define $M$ as the mass matrix, $\Delta t$ as the
physics time step, $D_{\text{smooth}}$ as the velocity derivative of the actuator and passive forces, and
$D_{\text{rne}}$ as the velocity derivative of the turning-rate (Coriolis and centrifugal) forces. C MuJoCo solves
with $M - \Delta t\thinspace D_{\text{smooth}} + \Delta t\thinspace D_{\text{rne}}$ (its `mjd_rne_vel` subtracts
the turning-rate term from the velocity derivative). MuJoCo Warp's `implicit` in
`mujoco/mjx/third_party/mujoco_warp/_src/forward.py` calls `derivative.deriv_rne_vel(m, d, d.qLU,
flg_subtract=True)` and so solves with $M - \Delta t\thinspace D_{\text{smooth}} - \Delta t\thinspace D_{\text{rne}}$.
Flipping that flag (`integrator_options.correct_full_implicit_sign`) brings Warp to 1.2e-6 and 1.5e-6 m from C.
This also explains the earlier finding of the speed study (`profiles/antmaze_warp_blow_up/notes.md`) that the
forced full implicit ran away again in Warp although it does not in C; its parity number there (6.9e-3 m)
compared Warp implicit with C implicitfast.

## Runaways with the trained policies

`scripts/roll_out_with_setting.py`: a checkpoint's trainer is rebuilt from the unit's configuration file with the
setting's three values written in, the checkpoint is restored, every environment is put at the spawn, and the saved
policies act with fresh action noise (the same noise for every setting) and no learning. Define an episode as one
that ended during the rollout. An episode that ends by truncation before the 700-step limit was ended by a respawn
for a broken physics state (beyond the guard's bounds, or not finite). Then:

- respawns per 10,000 episodes: $10^{4}\thinspace \dfrac{n_{\text{respawned}}}{n_{\text{episodes}}}$;
- arrival rate: $\dfrac{n_{\text{arrived}}}{n_{\text{episodes}}}$; a respawned episode counts as not arrived.

The respawn counts equal the environment's own counter, except for the shipped `implicit` (up to 40 fewer of
42,319): a respawn at the 700th step is counted here as reaching the limit. When no respawn is seen in $N$
episodes, the 95% upper bound on the rate is about $3/N$: 0.46 per 10,000 for $N = 65{,}560$.

Checkpoints:

1. **128 environments per policy, 83.4 million steps per policy** (iteration 5,088; the motion check's copy). 128
   policies x 128 environments, 2,800 steps per environment.
2. **128 environments per policy, 125.3 million steps per policy** (iteration 7,648). Copied from Gilbreth at 12:01
   PT (written there at 11:51 PT; its size, time and inode were the same before and after the copy). It loaded and
   restored. The unit is still training there; this is not its final checkpoint.
3. **16 environments per policy, 39.6 million steps per policy** (iteration 19,328; the motion check's copy). 128
   policies x 16 environments, 5,600 steps per environment.

Cards: the current, shipped-implicit and RK4 rows of the 128-environment checkpoints ran on RTX 5080 (nekomata01),
the corrected-implicit rows on RTX 2080 Ti (ai02, ai03), and the 16-environment checkpoint on RTX 2080 Ti (ai01,
ai04). The counts do not depend on the card's speed.

**125.3 million steps, 128 environments per policy:**

| setting | episodes | respawned | per 10,000 | arrival rate | largest torso speed after<br>any physics step (m/s) |
|---|---:|---:|---:|---:|---:|
| implicitfast, 0.005 s x 10 (current) | 68,191 | 3,320 | 486.9 | 0.083 | 21.8 |
| RK4, 0.005 s x 10 | 65,570 | 0 | 0 | 0.069 | 15.0 |
| RK4, 0.01 s x 5 | 65,569 | 0 | 0 | 0.065 | 15.9 |
| implicit as shipped, 0.005 s x 10 | 66,389 | 3,310 | 498.6 | 0.074 | 41.1 |
| implicit as shipped, 0.01 s x 5 | 84,352 | 44,203 | 5,240.3 | 0.052 | 8,264 |
| implicit with the sign corrected, 0.005 s x 10 | 65,546 | 0 | 0 | 0.046 | 14.8 |
| implicit with the sign corrected, 0.01 s x 5 | 65,540 | 0 | 0 | 0.033 | 16.4 |

**83.4 million steps, 128 environments per policy:**

| setting | episodes | respawned | per 10,000 | arrival rate | largest torso speed after<br>any physics step (m/s) |
|---|---:|---:|---:|---:|---:|
| implicitfast, 0.005 s x 10 (current) | 65,693 | 221 | 33.6 | 0.064 | 18.0 |
| RK4, 0.005 s x 10 | 65,561 | 0 | 0 | 0.064 | 15.1 |
| RK4, 0.01 s x 5 | 65,559 | 0 | 0 | 0.063 | 15.6 |
| implicit as shipped, 0.005 s x 10 | 66,336 | 3,102 | 467.6 | 0.069 | 28.9 |
| implicit as shipped, 0.01 s x 5 | 83,188 | 42,279 | 5,082.3 | 0.051 | 2,074 |
| implicit with the sign corrected, 0.005 s x 10 | 65,546 | 0 | 0 | 0.044 | 15.5 |
| implicit with the sign corrected, 0.01 s x 5 | 65,540 | 0 | 0 | 0.031 | 16.6 |

**39.6 million steps, 16 environments per policy:**

| setting | episodes | respawned | per 10,000 | arrival rate | largest torso speed after<br>any physics step (m/s) |
|---|---:|---:|---:|---:|---:|
| implicitfast, 0.005 s x 10 (current) | 16,396 | 2 | 1.2 | 0.050 | 15.2 |
| RK4, 0.005 s x 10 | 16,395 | 0 | 0 | 0.058 | 14.7 |
| RK4, 0.01 s x 5 | 16,401 | 0 | 0 | 0.061 | 15.6 |
| implicit as shipped, 0.005 s x 10 | 16,722 | 924 | 552.6 | 0.064 | 41.5 |
| implicit as shipped, 0.01 s x 5 | 22,103 | 10,970 | 4,963.1 | 0.044 | 1,081 |
| implicit with the sign corrected, 0.005 s x 10 | 16,388 | 0 | 0 | 0.040 | 14.7 |
| implicit with the sign corrected, 0.01 s x 5 | 16,385 | 0 | 0 | 0.032 | 15.7 |

**Per configuration at 125.3 million steps** (16 policies each; respawns, arrival rate). Learning rate 1e-2,
intrinsic weights 1, 4, 16, 64:

| setting | weight 1 | weight 4 | weight 16 | weight 64 | learning rate 3e-2,<br>arrival rate, 4 weights |
|---|---|---|---|---|---|
| implicitfast, 0.005 s x 10 (current) | 609, 0.125 | 500, 0.090 | 1,317, 0.109 | 894, 0.107 | 0.054 to 0.061, no respawn |
| RK4, 0.005 s x 10 | 0, 0.083 | 0, 0.080 | 0, 0.078 | 0, 0.080 | 0.056 to 0.059 |
| RK4, 0.01 s x 5 | 0, 0.079 | 0, 0.077 | 0, 0.077 | 0, 0.076 | 0.048 to 0.056 |
| implicit with the sign corrected, 0.005 s x 10 | 0, 0.061 | 0, 0.055 | 0, 0.060 | 0, 0.053 | 0.029 to 0.039 |
| implicit with the sign corrected, 0.01 s x 5 | 0, 0.041 | 0, 0.038 | 0, 0.042 | 0, 0.044 | 0.024 to 0.028 |

Mean steps of the arriving episodes in the four learning-rate-1e-2 configurations at 125.3 million steps: 454 to
493 in the current physics, 514 to 525 under RK4 at 0.005 s, 516 to 528 under RK4 at 0.01 s.

What the tables show:

1. The current physics' runaways grow with training: 1.2 per 10,000 episodes at 40 million steps (16 environments),
   33.6 at 83 million and 486.9 at 125 million (128 environments), almost all in the configurations with learning
   rate 1e-2. Both RK4 settings and corrected `implicit` had none in any checkpoint. The largest torso speed stayed
   at 14.7 to 16.6 m/s, the speed of the ants that do not run away.
2. Under RK4 the 125-million-step policies arrive about as often as under the current physics at 83 million steps.
   In the current physics they arrive more often (0.083 against 0.065 to 0.069), entirely in the configurations
   that run away.
3. Corrected `implicit` lowers the arrival rate of every configuration, including those that never run away, to
   about half of RK4's at 0.01 s and two thirds at 0.005 s. C MuJoCo's own `implicit` integrator did the same in the
   motion check (`diagnosis/2026-11-13_what-the-trained-ants-do_against-reference-mujoco/notes.md`, Result 1: 0.054
   against 0.069 for RK4 at the same step). It is a property of the integrator, which takes energy out of a fast
   hopping ant, and the reference ant model (RK4) does not have it.

## What a change of the physics would need

1. **RK4 at 0.01 s x 5:** only the three values in the configuration file. `build_antmaze_xml` writes the
   integrator into the model's option line and MuJoCo Warp reads it (integrator number 1 in every record). The
   guard can stay as a second line of defence: it never fired.
2. **RK4 at 0.005 s x 10:** the same, with `physics_time_step = 0.005` and `frame_skip = 10`.
3. **`implicit`:** two patches of the installed library, in `warp_antmaze.py` behind `integrator == "implicit"`,
   with tests:
   - MJX's `_put_option` must accept it (`integrator_options.allow_full_implicit`);
   - MuJoCo Warp's sign must be corrected (`integrator_options.correct_full_implicit_sign`).

   I did not make this change. The task was measurement only, and the change would alter a vendored library's
   behaviour, which is the user's decision. Newer MuJoCo Warp releases may have fixed the sign; I did not check.

For every option, units trained before the change ran in a different environment, so they are not comparable with
units trained after it.

## Not measured

1. AntMaze Medium and Large were not gated with RK4; the U-maze gates and the code path are the same for all three
   maps.
2. The speeds are for untrained policies. The trained policies' rollouts give the same ratios (section "Speed").
3. Training under RK4 was not run: how fast a policy learns, and whether it still learns to hop, is not known.
4. The H100 timings had another account's process on the card. A clean H100 row would need serval05 free or
   serval02 card 1.
5. The 125.3-million-step checkpoint is the latest one on Gilbreth at 11:51 PT, not the unit's last.

## Files

- `scripts/integrator_options.py`: the seven settings, the two patches for `implicit`, the configuration rewrite.
- `scripts/write_configurations.py`, `configurations/`: one configuration file per setting.
- `scripts/profile_integrators.py`, `scripts/speed_job.sh`, `scripts/h100_speed_unit.sh`: training-iteration
  timing (A100 job 6908901; serval05 profile unit).
- `scripts/integrator_gates.py`, `scripts/gates_job.sh`: the gates.
- `scripts/implicit_four_way.py`, `scripts/processor_job.sh`: the four steppers and the sign of the correction.
- `scripts/roll_out_with_setting.py`, `scripts/rollout_job.sh`: the rollouts with the trained policies (jobs
  6908909 to 6908911, 6909049 to 6909051).
- `scripts/summarize.py`: the tables from `records/`.
- `records/`: gate, speed and rollout summaries (`*_episodes.npz`, one row per episode, not in git); `logs/` (not in
  git); `checkpoint_copy/`: the 125.3-million-step checkpoint (1.5 GB, not in git).
