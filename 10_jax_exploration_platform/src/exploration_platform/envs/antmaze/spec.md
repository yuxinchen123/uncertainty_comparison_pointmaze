# AntMaze on the platform — reference semantics and deviations

Specification name: `antmaze_<map>_cont<cap>_nonoise@1` (maps `umaze` / `medium` / `large`).

The reference is Gymnasium-Robotics 1.3.1 `AntMaze_UMaze-v5` / `AntMaze_Medium-v5` /
`AntMaze_Large-v5` (`gymnasium_robotics/envs/maze/ant_maze_v5.py` over `maze_v4.py`, read
2026-08-18 from the `exploration` env's installed copy), with the Gymnasium Ant of `ant_v5.py`
and its `assets/ant.xml`. The `ant.xml` vendored in `assets/` here is a byte copy of that file
(Gymnasium 1.2.3, MIT license).

## What the reference fixes, and this implementation keeps

| item | value | reference source |
|---|---|---|
| robot | Gymnasium Ant: 8 torque actuators in [-1, 1], free-floating torso, 15 qpos / 14 qvel | `ant_maze_v5.py` builds `AntEnv` |
| control cadence | one env step = 5 physics steps of 0.01 s (20 Hz) | `AntEnv` `frame_skip=5` |
| maze maps | the same `umaze` / `medium` / `large` grids the PointMaze family uses | `maps.U_MAZE / MEDIUM_MAZE / LARGE_MAZE` |
| maze scale | `maze_size_scaling=4`: a cell is 4 m x 4 m; walls are boxes 2 m tall (`maze_height=0.5`) centred at z = 1 m | `ant_maze_v5.py` `super().__init__` |
| reward | sparse: 1 when the torso xy is within 0.45 m of the goal xy, else 0 | `maze_v4.compute_reward` |
| task type | continuing: reaching the goal never terminates the episode | `continuing_task=True` default |
| episode caps | 700 (umaze), 1000 (medium, large) env steps, truncation only | the v5 registrations |
| joint reset noise | zero — v5 constructs `AntEnv(reset_noise_scale=0.0)` | `ant_maze_v5.py` |
| unhealthy termination | none: AntMaze discards the Ant's own `terminated` (a flipped ant keeps running until the cap) | `ant_maze_v5.step` ignores it |

## Deviations, and why

1. **Fixed start and goal cells, zero cell noise.** The reference's letterless maps make every
   open cell a legal random start/goal and add uniform(-0.25, 0.25) m of position noise; this
   platform pins one start and one goal per map (the PointMaze convention since 2026-08-16:
   bottom-left start, far-end goal) and sets the noise to zero, so two copies differing only in
   policy sampling see the same task. Consequence: with the ant's joint noise already zero,
   every reset is fully deterministic and the environment holds no RNG at all.
   Presets: umaze start (3,1) goal (1,1); medium start (6,1) goal (1,6); large start (7,1)
   goal (1,10) — (row, col), row 0 the top row.
2. **The goal never moves.** The reference's continuing task re-samples a goal cell when the
   current one is reached; with one fixed goal cell there is nothing to re-sample, so the ant
   keeps collecting reward while it stays inside the goal radius (exactly the platform's
   PointMaze semantics).
3. **Integrator and solver budget.** The reference ant.xml integrates with RK4 at MuJoCo's
   default Newton settings. Under MJX that compiles to a program about 400x slower (probe 1,
   2026-08-18: 2,134 ms per env step at 1,024 copies on an H100). The fused model runs
   `implicitfast` with `iterations=4, ls_iterations=8`, chosen on this evidence: the MJX
   example-model setting `iterations=1` is unstable for this ant (flies to z = 19 m on CPU,
   NaNs in float32), while at `iterations=4` the tuned model shares the reference's resting
   equilibrium exactly — settled 1,200 env steps, both models rest at z = 0.38248 m and a
   state settled under either stays settled under the other to 3.7e-11 (gate 3 below).
   That evidence is about an ant at rest. For an ant that turns fast in the air `implicitfast`
   at 0.01 s is not stable, which the reference's RK4 is (measured 2026-11-12; deviation 9).
4. **Wall boxes merged.** The reference emits one box geom per wall cell; here horizontal runs
   of wall cells are merged into single boxes (62 wall cells -> 31 boxes on the large map,
   38 -> 20 on the medium, 18 -> 8 on the umaze). The union of
   boxes — hence the collision geometry — is identical; only the number of geom pairs MJX
   enumerates shrinks.
5. **Observation layout.** The reference returns a dict: 27-d `observation` (qpos without the
   torso xy, then qvel), 2-d `achieved_goal` (the torso xy), 2-d `desired_goal`. The platform
   returns one flat 29-d vector — the full qpos (torso xy included, indices 0–1) then qvel —
   because the exploration bonus scores the next state and the torso position is the part that
   says where the ant is. The goal is not observed, as in the platform's PointMaze. Same
   numbers, different packaging; nothing is added or dropped except the (constant) desired
   goal.
6. **Float32 physics.** The platform runs `jax_enable_x64=True` globally for its running
   statistics; every MJX call is wrapped in `jax.enable_x64(False)` at trace time so the
   compiled physics is float32 (the reference's C MuJoCo is float64). The parity gate bounds
   the resulting drift.

7. **Non-finite states respawn.** A float32 contact solve can fail on a knife-edge state and
   leave NaNs — measured at order one event per hundred thousand copy-iterations under a
   training policy, and not reproducible (the card's non-deterministic reductions decide
   whether the edge tips). (The measurement of 2026-11-12 under deviation 9 found another source
   of non-finite states that IS reproducible: an ant turning fast in the air.) Such an environment is treated as an infrastructure truncation:
   respawned at the spawn, reward forced to 0 for that step, its observation sanitised so the
   networks never see a NaN, and counted in the state's `nan_count` so a run can report how
   often it happened. The C-MuJoCo reference in float64 does not need this.

8. **A ceiling on the wall tops (decided in the session of 2026-11-12 and reported to the user, who may reverse it).** The reference has none. Its ant
   is strong enough to leave the floor (motors of 150 N m per joint on a body of under 1 kg), and
   under random torques some ants rose above the 2 m walls and crossed them: measured on an H100
   NVL with 16,384 ants for 1,400 steps, 63,100 steps with the torso above the wall tops and 15
   arrivals at the U-maze goal over the wall with MJX's JAX implementation, 51,818 steps and 2
   arrivals with MuJoCo Warp. C MuJoCo does the same on the model without a ceiling: of 1,024
   ants with random torques for 700 steps, 83 had the torso above the wall tops and 4 were over
   a wall cell. A maze whose walls can be crossed is not the task. The model of all
   three maps therefore holds one plane at z = 2 m whose normal points down, with the floor's
   contact parameters, that collides with the torso sphere only: the plane has `contype=0
   conaffinity=2`, the torso sphere `contype=3` and every other geom keeps `contype=1`, so the
   model gains exactly one geom pair and the floor and wall pairs are those of before. The torso
   sphere (radius 0.25 m) meets the plane when the torso centre is at 1.74 m. The environment
   state carries a fault record per policy (`faults.py`): the environment steps with the torso
   centre above 1.5 m after any physics step, and the largest torso speed; the training command
   writes both into every window record (`ceiling_zone_steps_per_copy`,
   `largest_torso_speed_per_copy`). Checks: `tests/envs/test_antmaze_ceiling.py`.
9. **A physics fault guard, off by default (added 2026-11-12 in the session and reported to the
   user, who may reverse it; `fault_torso_speed`, `fault_torso_angular_speed`).** The reference
   has none in its own code; C MuJoCo itself puts a simulation back to the model's default state
   when a position, velocity or acceleration is not finite or above 1e10.
   - **What was seen.** In training with RND on the U-maze (MuJoCo Warp, `implicitfast`, time
     step 0.01 s) most configurations had, at some logging window, a policy whose largest torso
     speed was above 10,000 m/s (up to 2.5e15 m/s). The policies of one checkpoint (iteration 336
     of 1,024; 128 policies x 128 environments) rolled out without learning on an H100 NVL gave
     33 such episodes in 362 million environment steps (0.09 per million) and 26 respawns for a
     non-finite state.
   - **The mechanism.** An ant that is off the floor (median torso height 1.1 to 1.3 m) and
     turning. `implicitfast`, like Euler, takes the forces that depend on the turning rate
     (Coriolis and centrifugal) from the start of the physics step. For a body that turns fast
     this adds rotational energy at every step, and in the air nothing takes it away: over the 24
     environment steps before an episode's torso speed first passed 100 m/s, the median angular
     speed of the torso rose from 32 to 55 (12 steps before), 94 (4 before), 138, 188 and
     326 rad/s. Contacts, joint limits and the solver are not the cause: the growth happens with
     no contact, and from the recorded states C MuJoCo does the same with `implicitfast` at 4/8,
     8/16, 16/16 and 100/50 solver and line-search iterations and with `Euler` (32 of 33 episodes
     from 2 or 4 environment steps before, 20 of 33 from 24 steps before). With the reference's
     `RK4`, or with the integrator `implicit` (which includes the turning-rate derivatives of
     those forces), C MuJoCo reaches 100 m/s in none of the 33 from any of these starts; from 24
     steps before, the largest torso speed stays near 5 m/s and the largest angular speed near
     45 rad/s (medians).
   - **Other physics settings in MuJoCo Warp, same policies** (H100 NVL, 16,384 environments;
     seconds per training iteration measured on serval02 card 1):

     | physics | environment steps rolled out | episodes above 100 m/s | largest torso speed after any physics step | environment steps ending above 60 rad/s | seconds per training iteration |
     |---|---|---|---|---|---|
     | `implicitfast`, 0.01 s (training) | 362 million | 33 | 5.7e13 m/s | not counted in that run | 1.374 |
     | `RK4`, 0.01 s | 104 million | 0 | 14.6 m/s | 0 | 4.927 |
     | `implicitfast`, 0.005 s, 10 physics steps per environment step | 127 million | 0 | 14.3 m/s | 1 | 2.707 |

     With `implicitfast` at 0.01 s and only the speed bound on, 298 of 109 million environment
     steps ended above 60 rad/s, and 0.53% between 20 and 40 rad/s against 0.28% with `RK4` and
     0.29% at 0.005 s: the integrator inflates the turning of airborne ants also below the level
     at which an episode runs away. MJX 3.11.0 refuses the integrator `implicit` for both of its
     implementations; forced past that check, MuJoCo Warp's step did not agree with C MuJoCo
     (one step at contact-free states differed by up to 6.9e-3 m against the gate's 1e-4) and
     still ran away in 30 to 33 of the 33 episodes, so it is not used.
   - **The guard.** With `fault_torso_speed` or `fault_torso_angular_speed` above zero, an
     environment whose torso, after any physics step of an environment step, moved faster than
     the first, turned faster than the second, or had its centre outside the maze's volume
     (outer faces of the outer walls, floor, ceiling plane) is treated exactly like a non-finite
     one (deviation 7): respawned, reward 0, no arrival, a truncation whose final observation is
     the spawn, counted in `nan_count` and in the window record's
     `physics_fault_respawns_per_copy`. The window record also holds
     `largest_torso_angular_speed_per_copy`.
   - **The bounds the study uses: 30 m/s and 100 rad/s.** With `RK4` and with `implicitfast` at
     0.005 s the same policies never passed 14.6 m/s after any physics step and never ended an
     environment step above 80 rad/s (one step in 231 million above 60). 30 m/s is twice that
     speed; 100 rad/s is above every turning rate the two stable settings showed and below the
     138 rad/s (median) of two steps before a runaway. With both bounds on and the training
     physics, the same policies gave 22 respawns in 92 million environment steps (0.24 per
     million, about one episode in 6,000), no environment step ending above 100 rad/s, and a
     largest torso speed after any physics step of 17.9 m/s. The guard costs nothing measurable
     (1.380 s per training iteration against 1.374 s).
   - **What the guard does not do.** It ends a runaway; it does not correct the inflated turning
     below its bounds. Only a stable integration does (the 0.005 s time step at 1.97 times the
     time, or `RK4` at 3.59 times).
   Measurements, scripts and records:
   `benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_*/profiles/antmaze_warp_blow_up/notes.md`.
   Checks: `tests/envs/test_antmaze_fault_guard.py`.

## Correctness gates

Run by `tests/envs/test_antmaze_mjx.py` (fast, CPU) and
`tests/envs/test_antmaze_parity_cpu_mujoco.py` (slower):

1. MJX against C MuJoCo of the SAME model (same XML, same options), one env step from 50
   states along a random-torque trajectory, split by contact. Measured 2026-08-18: at
   contact-free states the fused float32 stepper agrees with C float64 to 3.9e-7 m (median),
   and MJX float64 agrees with C float64 to 4.4e-16 — the same algorithm to machine epsilon;
   float32 against float64 MJX differs by at most 1.8e-6, so the float32 choice costs nothing.
   At states whose 5-substep window contains a contact event the one-step difference reaches
   the centimetre level (worst 2.75e-2 m), because MJX's collision functions differ from C's
   by documented design (different contact-point sets for the same geom pair). The gate
   bounds the two regimes separately (1e-4 contact-free — measured worst 1.76e-6 — and 5e-2
   in contact).
2. Trajectory divergence structure: 20 shared-torque env steps from the spawn agree to
   roundoff before the first contact (measured 1.67e-6), then diverge exponentially — two
   samples of the same chaotic dynamics — bounded by one maze cell over the horizon.
   Equilibrium preservation: settled 1,200 env steps, both integrators rest at exactly
   z = 0.38248 m, and each keeps the other's settled state to ~3.5e-11.
3. Semantics: reward is 1 exactly within 0.45 m of the goal; truncation at the cap and
   auto-reset restore the spawn exactly; the continuing task never terminates; copies are
   bit-isolated (stepping copy k never changes copy j).
4. The x64 boundary: composing the env under the platform's global x64 produces float32
   physics state and float32 observations.
