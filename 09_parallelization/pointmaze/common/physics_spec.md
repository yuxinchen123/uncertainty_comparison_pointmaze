# PointMaze physics spec — ground truth for the GPU reimplementations

Source: Gymnasium-Robotics 1.3.1 `PointMaze_*-v3` (MuJoCo). Every constant and rule below was
read from the installed package and verified empirically by
`code/probe_mujoco_dynamics.py` (run 2026-08-15 with the `exploration` env). The reference
fixtures in `fixtures/` are produced by `code/make_fixtures.py`.

## Constants (read from the MuJoCo model)

| constant | value | source |
|---|---|---|
| timestep $h$ | 0.01 | point.xml `option timestep` |
| frame skip | 1 (one env step = one integrator step) | PointEnv `frame_skip=1` |
| ball mass $m$ | 4.1887902047863905 | sphere r=0.1, density 1000: $m = \frac{4}{3}\pi r^3 \cdot 1000$ |
| joint damping $d$ | 1.0 | point.xml `joint damping` |
| actuator gear $g$ | 100.0 | point.xml `motor gear` |
| ball radius $r$ | 0.1 | point.xml `geom size` |
| gravity | 0 | point.xml |
| velocity clip | ±5.0 per axis | PointEnv `_clip_velocity` |
| action clip | ±1.0 per axis | PointEnv `step` |
| cell size | 1.0 m (PointMaze `maze_size_scaling=1`) | point_maze.py |
| contact | condim=1 (frictionless), margin 0.002 | point.xml defaults |

## Per-step update rule (verified EXACT in free space, error 0 / 2e-16)

Define $q, v \in \mathbb{R}^2$ as position and velocity, $a$ as the action. One env step:

1. $a_c = \mathrm{clip}(a, -1, 1)$
2. $v_c = \mathrm{clip}(v, -5, 5)$  (per axis, BEFORE the integrator step)
3. $v' = (m v_c + h g a_c) / (m + h d)$  (MuJoCo Euler integrates joint damping implicitly)
4. $q' = q + h v'$
5. wall contact resolution (below) adjusts $q', v'$
6. observation = $(q'_x, q'_y, v'_x, v'_y)$, float64 in the reference env

Verified: 200 random free-space steps match to 0 (position) / 2.2e-16 (velocity); with the
clip engaged the terminal speed is 5.226256 = $(5m + hg)/(m + hd)$, matched to 6 decimals.

## Walls — the EXACT contact model (probe-verified to 8e-16; see `code/fit_contact_model.py`)

The maze map is a grid of 1 m cells (map arrays in `gymnasium_robotics.envs.maze.maps`;
Large = 9 rows x 12 cols). World frame: cell $(i, j)$ has center
$x = (j + 0.5) - W/2$, $y = H/2 - (i + 0.5)$ with $W$=cols, $H$=rows (row 0 at top, y up).
Every `1` cell is a static box spanning the full cell (half-size 0.5, full height).

MuJoCo's contact pipeline for this system was extracted exactly (solver internals `efc_*`
probed over a 405-point grid, and a 2-contact solve verified against a live dump):

1. Contacts are detected at the CURRENT (pre-step) state, one candidate per ball-box geom
   pair. Only the 8 neighbor cells of the ball's cell can ever be in range. For each wall
   box: nearest point on the rectangle to the center gives the surface distance
   $\mathrm{dist} = \lVert p - p_{\mathrm{near}} \rVert - r$ and the separation direction
   $s$ (nearest point toward center; a flat face when the center is over the box, an oblique
   corner normal otherwise — including INTERIOR corners of coplanar box runs).
2. A contact is active when $\mathrm{dist} \le \mathrm{margin}$ (0.002; boundary inclusive).
3. Force law per contact (solref [0.02, 1], solimp [0.9, 0.95, 0.001, 0.5, 2]): define
   $\rho = \mathrm{dist} - \mathrm{margin}$; impedance $d$ = solimp sigmoid of
   $\lvert \rho \rvert / 0.001$ (0.9 to 0.95, power 2, midpoint 0.5); $b = 2/(0.95 \cdot 0.02)$;
   $k = d/(0.95^2 \cdot 0.02^2)$; reference acceleration
   $a_{\mathrm{ref}} = -b \cdot v_s - k \cdot \rho$ with $v_s$ = separation velocity
   (computed from the pre-clipped-to-±5 velocity); regularization $R = (1-d)/(d \cdot m)$.
4. Joint solve over active contacts (measured: never more than 2 here): the strictly convex
   QP $\min_{f \ge 0} \frac{1}{2} f^\top (A+R) f - f^\top (a_{\mathrm{ref}} - a_{\mathrm{unc}})$
   with $A_{ij} = (s_i \cdot s_j)/m$ and $a_{\mathrm{unc}} = (G a_c - D v_c)/m$ projected on
   each $s_i$. Solved exactly by 4-case enumeration for the top-2 candidates by distance.
5. Total force $F = G a_c + \sum_i f_i s_i$, then the implicit-damping Euler step of the
   update rule above. There is NO position projection: transient penetration (up to ~8 mm on
   a full-speed impact) is real MuJoCo behavior and is reproduced exactly.

Verified results (fixture checker): one-step teacher-forced error <= 4.4e-16 (float64) on
every fixture case including slides across coplanar box boundaries and corner wraps;
closed-loop 400-step multi-contact rollouts match to 1e-13. One fixture (`corner_graze`)
passes through a knife-edge equilibrium (ball balanced on a corner against a diagonal push)
where closed-loop paths separate on rounding noise; its one-step comparison stays exact.
float32 mode: one-step <= 5e-7, episode rollouts within ~1.5 mm.

## Episode logic (default config = the run-6 setup `initial_single_large_pointmaze_max_400`)

- Map `PointMaze_Large-v3`, fixed start cell (7, 1), fixed goal cell (1, 10).
- Reset: $q$ = start-cell center + per-axis uniform(−0.25, 0.25) noise; $v = 0$. The goal
  position likewise gets fresh uniform(−0.25, 0.25) noise at every reset.
- Reward (sparse): $1.0$ if $\lVert q' - \mathrm{goal} \rVert \le 0.45$ else $0.0$, computed on
  the post-step position. Optional constant `reward_shift` added every step (0 by default;
  −1 gives the ExPLORe convention).
- `continuing_task=True`: never terminated; truncated at 400 steps (auto-reset in the batched
  envs, with the final observation exposed for correct PPO bootstrapping).
- Alternative config knobs (must stay supported, they are the section-8 setups): any of the 4
  maps, any start/goal cells, `continuing_task=False` (terminate at goal distance ≤ 0.45),
  noise 0, `reward_shift=-1`, different step cap. Registered default caps per map (used when
  a section-8 setup passes `max_episode_steps=None`): UMaze 300, Open 300, Medium 600,
  Large 800.
- `reset_target` semantics (reference `update_goal`): goal resampling on reach applies only
  when `continuing_task=True` AND `reset_target=True`. Both project setups use
  `reset_target=False`, so the goal never moves within an episode; the GPU envs implement
  only this case.
- The goal CELL is fixed but the goal POSITION is per-env state: redrawn with fresh noise at
  every reset, and not observable. Observation after the project's wrapper stack
  (RemoveGoal): the flat 4-d state $(x, y, v_x, v_y)$; reference dtype float64 (GPU envs
  default float32).
- Action space: Box(−1, 1, (2,), float32). The env clips internally; the PPO policy must
  CLIP (not tanh-squash) to match the reference training setup.

## Seeding

Reset noise must follow the project's keyed-RNG rule (`~/.claude/rules/rng-seeding.md`): each
draw is a function of (base_seed, copy_index, env_index, reset_counter, axis), via a
counter-based generator (hash / philox / JAX fold_in) — never a shared sequential stream. The
reference MuJoCo env's noise stream is NOT reproduced draw-for-draw; validation of reset noise
is distributional (uniform bounds, per-axis independence), not trajectory-exact.

## Validation contract (every implementation must pass, `code/check_against_fixtures.py`)

1. One-step teacher-forced error on every fixture case ≤ 1e-9 (float64) / 1e-3 (float32).
2. Closed-loop rollout error ≤ 1e-6 (float64) / 5e-2 (float32) on every case except the
   knife-edge `corner_graze` (reported only); penetration never exceeds 0.06 m (the
   one-max-speed-step physical bound).
3. Reward / termination / truncation / auto-reset match the rules above bit-exactly given the
   same positions (unit tests per implementation).
4. Random-policy behavior check: 256 episodes, distribution of episode return and of visited
   cells statistically indistinguishable from the reference env (same episode cap).
