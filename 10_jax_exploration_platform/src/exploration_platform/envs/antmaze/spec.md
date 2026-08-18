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
   `implicitfast` with `iterations=4, ls_iterations=8`, chosen on this evidence (probe 3):
   the C-MuJoCo passive settling height at these settings equals the RK4 full-solver height
   (0.0427 m, identical to 4 decimals), while the MJX example-model setting `iterations=1`
   is unstable for this ant (flies to z = 19 m on CPU, NaNs in float32).
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

## Correctness gates

Run by `tests/envs/test_antmaze_mjx.py` (fast, CPU) and
`tests/envs/test_antmaze_parity_cpu_mujoco.py` (slower):

1. MJX against C MuJoCo of the SAME model (same XML, same options), one env step from 50
   states along a random-torque trajectory, split by contact. Measured 2026-08-18: at
   contact-free states the fused float32 stepper agrees with C float64 to 3.9e-7 m (median),
   and MJX float64 agrees with C float64 to 4.4e-16 — the same algorithm to machine epsilon;
   float32 against float64 MJX differs by at most 1.8e-6, so the float32 choice costs nothing.
   At states inside a contact event the one-step difference reaches the centimetre level
   (worst 2.75e-2 m), because MJX's collision functions differ from C's by documented design
   (different contact-point sets for the same geom pair). The gate bounds the two regimes
   separately (1e-4 contact-free, 5e-2 in contact), plus a 20-env-step shared-torque
   trajectory bound.
2. Settling equivalence: the tuned model's passive settling height equals the reference RK4
   full-solver model's on C MuJoCo (the integrator deviation does not change where the ant
   comes to rest).
3. Semantics: reward is 1 exactly within 0.45 m of the goal; truncation at the cap and
   auto-reset restore the spawn exactly; the continuing task never terminates; copies are
   bit-isolated (stepping copy k never changes copy j).
4. The x64 boundary: composing the env under the platform's global x64 produces float32
   physics state and float32 observations.
