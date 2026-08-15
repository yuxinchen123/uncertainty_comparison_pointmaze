# Adversarial review of `pointmaze/common/physics_spec.md`

22 findings against the installed Gymnasium-Robotics 1.3.1 source, the 07_reconstruction wrapper
stack, and three new MuJoCo probes run 2026-08-15 with
`/p/rlprojects/RND/.venvs/exploration/bin/python` (probe scripts kept in the session scratchpad;
every number below is reproducible from the commands quoted in the evidence lines).

Versions in use: gymnasium 1.0.0, gymnasium-robotics 1.3.1, mujoco 3.1.6.

Severity meaning:

- **blocker** — an implementation built exactly to the current text will be wrong in a way that
  changes trajectories, acceptance results, or a hard-coded constant.
- **should-fix** — the text is silent on something each of the three implementations must decide,
  so the three will decide it differently.
- **note** — correct but imprecise, or a fact worth recording so nobody "corrects" it later.

---

## 1. Wall impacts bounce. The "no bounce" claim is wrong — blocker

**Spec text.** "Contact model in MuJoCo: soft, frictionless (condim=1), margin 0.002, default
solref (critically damped — no bounce)." And the approximation list calls the only impact effect
"(b) impact transients".

**Evidence.** Ball launched at the velocity clip toward the wall face at `x = 5.0`, zero action
(probe 1):

```
t= 7 x=+4.895737 vx=+4.905525 ncon=1
t= 8 x=+4.944675 vx=+4.893842 ncon=1
t= 9 x=+4.933145 vx=-1.153028 ncon=2
t=14 x=+4.875904 vx=-1.139363 ncon=1     <- still leaving the wall at -1.14 m/s
```

The outgoing velocity is not a transient: the ball leaves the wall and keeps travelling at
about -1.14 m/s. Measured over a range of impact speeds (probe 2, free flight, zero action):

| impact speed | deepest surface overlap | outgoing speed | ratio |
|---|---|---|---|
| 0.25 | 0.000000 | 0.0246 | 0.099 |
| 0.50 | 0.000707 | 0.0664 | 0.133 |
| 1.00 | 0.000000 | 0.0156 | 0.016 |
| 2.00 | 0.013598 | 0.3819 | 0.191 |
| 3.00 | 0.025322 | 0.6700 | 0.223 |
| 4.00 | 0.016589 | 0.4515 | 0.113 |
| 5.2263 | 0.046682 | 1.1944 | 0.229 |

The ratio is not a material constant — it depends on how far into the wall the last free step
landed, which depends on the phase of the approach. `solref = [0.02, 1.0]` (dampratio 1) makes the
contact critically damped only in the linear regime; one step at the velocity clip moves the ball
0.0523 m, far past the 0.002 m margin, so the response is never in that regime for a fast impact.

**Correction.**

- Delete "critically damped — no bounce".
- Replace it with: the contact is a soft spring-damper; a step that lands the ball inside the wall
  produces an outgoing velocity of up to about 1.2 m/s at the velocity clip, with the ratio of
  outgoing to incoming speed between 0.01 and 0.23 depending on where in the step the overlap
  starts.
- Move the bounce from "approximation (b) impact transients" to a named, first-order divergence:
  the hard-clamp model returns exactly 0 normal velocity where MuJoCo returns up to -1.2 m/s.
- State the design decision explicitly (hard clamp, no restitution) and record the measured gap in
  each implementation's `progress_and_changes.md`, instead of implying the two agree.

---

## 2. Contact force starts at center distance 0.102, not 0.1 — blocker

**Spec text.** "the ball center settles 0.101372 from the face (hard-contact value 0.1) —
soft-contact compression ~1.4 mm" and "clamp the normal position to face ± r".

**Evidence.** `point.xml` line 6 sets `margin="0.002"`; the compiled model has
`geom_margin` uniformly 0.002 and `geom_gap` uniformly 0. With gap 0 every detected contact is
also an active constraint, so force begins at surface separation 0.002 — center distance
`r + margin = 0.102`. Probe 1 sweeps the separation and reads `ncon`:

```
surface sep 0.0021  x=4.89790  ncon=1 [ground only]
surface sep 0.0020  x=4.89800  ncon=2 [ground, block_1_11 dist=0.002]
surface sep 0.0015  x=4.89850  ncon=2 [ground, block_1_11 dist=0.0015]
```

Sustained full force into the face settles at `x = 4.898628`, i.e. 0.101372 from the face, with
`qfrc_constraint = [-100, 0]`. Releasing the force pushes the ball back out past 0.102 and it keeps
drifting outward. So 0.102 is the contact-free boundary and 0.101372 is where a 100 N push
compresses it to — 0.1 is not a value the reference ever targets.

**Correction.**

- Replace "hard-contact value 0.1" with: MuJoCo's contact-free boundary is `r + margin = 0.102`;
  under the maximum actuator force the ball is compressed 0.628 mm past it, to 0.101372.
- Change the hard-wall rule to clamp at `r_eff = r + margin = 0.102`, not `r = 0.1`. This halves
  the settle error (0.63 mm instead of 1.37 mm) and reproduces the point where deceleration
  begins. If 0.1 is kept instead, say so and say that every corridor is then 4 mm wider than the
  reference.
- Add `margin = 0.002`, `gap = 0`, `solref = [0.02, 1.0]`, `solimp = [0.9, 0.95, 0.001, 0.5, 2]`,
  `condim = 1`, `impratio = 1`, solver Newton, 20 iterations, tolerance 1e-8 to the constants
  table — they are currently summarised as "default solref" with no values.

---

## 3. Validation rule "no wall penetration ever" contradicts the reference — blocker

**Spec text.** Validation contract item 2: "Wall fixtures (head-on, slide, corner, high-speed
impact): no wall penetration ever (center-to-face distance >= r - 2 mm)".

**Evidence.** The reference itself violates that bound by about 45 mm. Probe 1, velocity-clip
impact: the ball center reaches `x = 4.944675` with the face at `x = 5.0`, so center-to-face is
0.055 while `r - 2 mm` is 0.098. Probe 2 measures a deepest surface overlap of 0.046682 at the
velocity clip. The `impact_fast` fixture in `make_fixtures.py` (line 65) is exactly this case, so
the fixture checker as specified would fail on the recorded reference data.

The ball center still never enters a wall cell: the largest single-step travel is
`0.01 * 5.226256 = 0.0523 m`, and contact starts at 0.102 from the face, so the worst center
position is about 0.05 m short of the face. Tunnelling through a 1 m wall is impossible.

**Correction.** Split item 2 into two independent clauses:

1. A property of the GPU env alone: the ball center never comes closer than `r_eff` to any wall
   face and never enters a wall cell. This is checked on the GPU env's own rollouts, not against
   MuJoCo.
2. A measured divergence against the fixtures: report max position and velocity difference per
   wall fixture, with a stated pass threshold per case. The `impact_fast` case cannot pass a tight
   threshold — give it its own recorded expected divergence (about 0.05 m position, about 1.2 m/s
   velocity) rather than a bound it will fail.

---

## 4. The per-axis clamp mis-fires at convex corners — blocker

**Spec text.** "for each wall face the ball would overlap, clamp the normal position to face ± r
and zero the inward normal velocity component. Per-axis decomposition is exact for face contacts
because contacts are frictionless".

**Evidence.** Probe 4 aims the ball at the convex corner `(-2.0, 2.5)` of `block_2_3`
(spans x in [-3,-2], y in [1.5,2.5]):

```
t=15 q=(-1.935199,+2.564801) v=(-1.443850,-1.443850) []
t=16 q=(-1.933405,+2.566595) v=(+0.179390,+0.179390) [block_2_3 dist=-0.008357 normal=(-0.7071,-0.7071,0)]
```

MuJoCo uses the oblique edge normal and reflects both velocity components together. The block's
right face is at `x = -2` and its top face at `y = 2.5`. A per-axis test at
`q = (-1.935, 2.565)` finds an overlap on **both** axes (`x < -2 + 0.102 = -1.898` and
`y < 2.5 + 0.102 = 2.602`), so a naive per-axis clamp would move the ball to `x = -1.898` and
`y = 2.602` at once — 0.037 m in each axis — and zero both velocity components. That is a larger
position error than the bounce it is trying to avoid, and it triggers in a region MuJoCo treats as
a single oblique contact.

The face case itself is fine: probe 2 shows the tangential component untouched by a face contact
(`vx` continues to decay only by damping while `vy` reverses), consistent with `condim = 1`.

**Correction.** Replace the per-face rule with the closest-point circle-versus-box test, stated
explicitly:

```
for each wall cell with center (cx, cy), half extent 0.5:
    px = clamp(qx, cx-0.5, cx+0.5);  py = clamp(qy, cy-0.5, cy+0.5)
    ex = qx - px;  ey = qy - py;  dist = sqrt(ex*ex + ey*ey)
    if dist < r_eff:
        n = (ex, ey) / dist                      # face contact gives an axis-aligned n
        q += (r_eff - dist) * n
        vn = v . n;  if vn < 0: v -= vn * n      # zero only the inward normal component
```

Also specify: what happens when `dist == 0` (center inside a box — assert it cannot happen), and
the resolution order when two wall cells overlap at once (the interior-corner case settles
per-axis in MuJoCo: probe 3 measured `qpos = (-4.898628, 3.398628)`, exactly 0.101372 from each of
the two faces, `qfrc_constraint = [100, -100]`).

---

## 5. Registered episode caps per map are missing — should-fix

**Spec text.** Alternative knobs mention "different step cap"; nothing states the defaults.

**Evidence.** `gymnasium_robotics/__init__.py`: UMaze 300 (line 970), Open 300 (line 982),
Medium 600 (line 1018), Large 800 (line 1054). The section-8 setups in `env_setups.py` line 48 use
`max_episode_steps=None`, which `train.py` line 389 turns into "do not pass the kwarg", i.e. the
registered default. Only `initial_single_large_pointmaze_max_400` overrides it, to 400.

**Correction.** Add a per-map row: UMaze 300, Open 300, Medium 600, Large 800, and say that the
section-8 configs use those defaults while the run-6 config uses 400 on Large. Also record the
TimeLimit convention: the episode carries exactly `max_episode_steps` env steps, truncation is
raised on the last one, `compute_truncated` in `maze_v4.py` line 414 always returns False so
truncation comes only from the external wrapper, and the reference env does not auto-reset (the
auto-reset in the training runs comes from Stable-Baselines3's `DummyVecEnv`).

---

## 6. `reset_target` is never mentioned, and the "single goal location" premise is false — should-fix

**Spec text.** "`continuing_task=True`: never terminated". Nothing about `reset_target` or
`update_goal`.

**Evidence.** `point_maze.py` line 322 defaults `reset_target=False`; `maze_v4.py` line 394
`update_goal` fires only when `continuing_task and reset_target and dist <= 0.45 and
len(unique_goal_locations) > 1`. The last condition is **true for all four maps**: the map arrays
contain no `r`/`g`/`c` cells, so `make_maze` (lines 218-236) assigns every open cell to both lists.
Probe 1 counted 46 goal locations on Large (7 UMaze, 15 Open, 26 Medium). So `reset_target=False`
is the only thing that disables goal resampling.

With `reset_target=True` (probe 1):

```
step at goal: reward 1.0 terminated False truncated False info {'success': True} new goal [0.5 2.]
```

The goal jumped from `(4.5, 3.0)` to a different open cell on the same step, after the reward was
computed.

**Correction.** Add a line: `reset_target=False` in both the run-6 setup (`env_setups.py` line 111)
and every section-8 setup (line 46), and it is a supported knob. State its semantics if enabled:
after the step's reward and termination are computed, if the post-step position is within 0.45 of
the goal, resample the goal cell uniformly from all open cells and re-apply the position noise,
repeating until the new goal is more than 0.45 away. Delete any implication that the map has a
single goal location.

---

## 7. Order of operations inside one env step is not stated — should-fix

**Evidence.** `point_maze.py` lines 392-406: integrate (`point_env.step`), build the observation,
`compute_reward` on the post-step position against the **current** goal, `compute_terminated`,
`compute_truncated` (always False), set `info["success"]`, then `update_goal` which may move the
goal. So a success step is rewarded against the old goal and the next step is scored against the
new one.

**Correction.** Add the ordered list to the episode-logic section. It is the only place that fixes
whether the goal move is visible to the current step's reward.

---

## 8. "Goal is fixed" is wrong — only the goal cell is fixed — should-fix

**Spec text.** "Observation after the project's wrapper stack (RemoveGoal): the flat 4-d state
(x, y, vx, vy). Goal is fixed, so no goal keys are exposed."

**Evidence.** `FixedGoalWrapper` (`point_maze_wrappers.py` line 22) pins the goal **cell**, and
`MazeEnv.reset` line 328 then calls `add_xy_position_noise` on it every reset. Probe 1, four resets
with the same `goal_cell`:

```
[[4.65663512 3.20637779]
 [4.5218125  3.21753621]
 [4.67870214 2.76679279]
 [4.68158946 3.02073061]]
```

2000 resets give per-axis noise in [-0.2487, +0.2500], mean about 0 — uniform(-0.25, 0.25) on both
axes, as documented. The goal position is not in the observation.

**Correction.** Rewrite as: the goal cell is fixed; the goal position is redrawn every reset within
a 0.5 m by 0.5 m square and is not observable, so the task is partially observed and the goal must
be per-environment state in the batched env, not a compile-time constant. The reward test uses the
current episode's goal. Under `position_noise_range = 0.0` (the section-8 setups) the goal is a
true constant.

---

## 9. Action-space dtype and the double clip are missing — should-fix

**Evidence.** Probe 1: `action_space = Box(-1.0, 1.0, (2,), float32)`, observation dict entries all
float64. Two independent clips apply: `np.clip(action, -1.0, 1.0)` in `point.py` line 56, and
MuJoCo's own `ctrllimited="true" ctrlrange="-1.0 1.0"` on both motors (`point.xml` lines 30-31).
Probe 3 confirms the second one: `do_simulation` with `ctrl = 5.0` gives exactly the same
`qvel = 0.23816384` as `ctrl = 1.0`, even though `d.ctrl` still reads 5.0.

Action precision: replaying the same action as float64 versus float32-rounded changes one step's
position by 2.8e-11 and velocity by 2.8e-9 (probe 2). Over a 400-step episode that accumulates to
roughly 1e-8 in position — above the spec's 1e-9 float64 tolerance. The fixtures store float64
actions (`make_fixtures.py` line 78) while the training loop feeds float32 actions from the policy.

**Correction.**

- Add the action space line: `Box(-1, 1, (2,), float32)`, clipped twice.
- State that a Gaussian PPO policy must reproduce this by **clipping** the sampled action, not by
  a tanh squash, and that the log-probability is taken on the unclipped sample.
- State that the 1e-9 float64 tolerance applies only when the implementation is fed the fixtures'
  float64 actions; a float32 action path needs about 1e-8 over 400 steps.

---

## 10. Visit-count semantics are absent although the validation contract scores them — should-fix

**Spec text.** Validation item 4 scores "distribution of visited cells"; nothing in the spec
defines the mapping. `STRUCTURE.md` line 63 lists "optional per-cell visit counts" as reproduced
wrapper semantics.

**Evidence.** `point_maze_utils.py` `observation_to_grid` (lines 19-61) and `velocity_to_grid`
(lines 64-80), used by `PositionVisitCountWrapper` and `PositionVelocityVisitCountWrapper`
(`point_maze_wrappers.py` lines 112-118, 156-163).

Rules that must be copied exactly:

- Position to cell: `col = int((x + W/2) / cell_size)`, `row = int((H/2 - y) / cell_size)`, with the
  edge cases `x < -W/2` to col 0, `x >= W/2` to the last column, `y <= -H/2` to the last row,
  `y > H/2` to row 0. The x and y edge tests are not symmetric — copy them as written.
- Velocity binning: clamp each component to [-5, 5], then 10 uniform bins over that range. The
  observed velocity reaches 5.226256, so the top bin absorbs the overshoot.
- Counts increment on `step` only, never on `reset`, and only when the cell is open. A position
  over a wall cell or out of bounds yields count 0 and no increment.

**Correction.** Add a short "Visit-count surfaces" section with those three rules, or state
explicitly that visit counts are out of scope for the GPU env and remove them from validation
item 4.

---

## 11. Velocity clip placement and what does *not* clip — note

The spec's step order is right; the surrounding facts are worth recording because they are the
easy thing to get backwards.

**Evidence.** `point.py` lines 55-58: the clip runs at the **start** of `step` and writes the
clipped velocity back through `set_state`, before `do_simulation`. Neither `set_state` nor `reset`
clips. Probe 1:

```
set_state with qvel 50: stored [ 50. -50.]
one step after: obs [ 0.04988092 -0.04988092  4.98809181 -4.98809181]
```

so the out-of-range velocity survives in the state until the next step clips it. Reset zeroes
velocity through `init_qvel`, which `PointMazeEnv.reset` never touches (only `init_qpos[:2]` is
overwritten, `point_maze.py` line 382) — probe 3 confirms a mid-flight reset returns
`(x, y, 0, 0)`.

**Correction.** Add one sentence: the stored velocity between steps may exceed the clip (up to
5.226256), the observation reports that unclipped value, and the clip is applied to the state at
the start of the next step. A batched env that clips at the end of the step produces different
observations.

---

## 12. Ball mass constant is off by one unit in the last place — note

**Evidence.** The spec writes `4.1887902047863905`; the model and `fixtures/constants.json` both
carry `4.188790204786391`. They are different doubles (bit patterns `...7365` and `...7366`),
differing by 8.88e-16, a relative 2.1e-16.

**Correction.** Use `4.188790204786391`, or better, state that implementations should read the
value from `fixtures/constants.json` rather than retyping it. The error is far below the 1e-9
tolerance but a constants table that does not match the model it claims to quote invites a
copy-paste of the wrong value into three implementations.

---

## 13. 0.45 versus 0.5 — 0.45 is the goal threshold everywhere in code — note

**Evidence.** 0.45 appears in all four goal tests: `compute_reward` (`maze_v4.py` line 382),
`compute_terminated` (line 389), `update_goal` (lines 400 and 405), and `info["success"]`
(`point_maze.py` lines 387 and 400). 0.5 appears in two unrelated places: the class docstring
(`point_maze.py` lines 251 and 282), which is simply wrong, and `generate_reset_pos`
(`maze_v4.py` line 284), where `0.5 * maze_size_scaling` is the minimum start-to-goal separation
used only when the start cell is auto-generated — never in this project, because
`FixedStartWrapper` always supplies `reset_cell`.

**Correction.** Add this as an explicit note so the threshold is not "corrected" to 0.5 later.

---

## 14. Wall box geometry and the permanent ground contact — note

**Spec text.** "Every `1` cell is a static box spanning the full cell (half-size 0.5, full
height)."

**Evidence.** Probe 1: `block_0_0 pos = [-5.5, 4.0, 0.2]`, `size = [0.5, 0.5, 0.2]`, so the box
spans z in [0, 0.4] and the ball center at z = 0 lies exactly in the plane of the box's bottom
face. That is why the sphere-box contact reduces to the 2-d circle-box problem with no z
component. Separately, the ball is permanently in contact with the ground plane (distance exactly
0, always in the contact list), but it contributes nothing: probe 3 in free space shows
`ncon = 1`, `qfrc_constraint = [0, 0]`.

**Correction.** Replace "full height" with the actual half extents `(0.5, 0.5, 0.2)` at
z-center 0.2, and add one line saying the ball-ground contact is always present and always
generates zero generalised force, so it can be ignored.

---

## 15. Control rate is 100 Hz — note

`timestep 0.01` with `frame_skip = 1` gives 100 control steps per simulated second. The
gymnasium-robotics class docstring (`point_maze.py` line 39) says "The control frequency of the
ball is of f = 10 Hz", and the two metadata blocks disagree with each other (`render_fps` 100 in
`point.py` line 30, 50 in `point_maze.py` line 313). The spec's constants are right; add a line
saying the docstring is wrong so nobody reconciles against it.

---

## 16. Reset without options randomises both the goal cell and the start cell — note

**Evidence.** `MazeEnv.reset` lines 307-311 pick a random goal cell and a random start cell when
`options` is None. Probe 1, four bare `reset()` calls, gives four different goal and start
positions across the map. The project's wrappers always pass both cells, so this never fires in
training — but any comparison harness or fixture generator that calls `reset()` without options
silently gets a different task. Worth one sentence, since the GPU env's reset must be compared
against the option-supplying path only.

---

## 17. `TerminateOnTimeLimitWrapper` exists and is off — note

**Evidence.** `point_maze_wrappers.py` lines 209-218 converts truncation into termination;
`train.py` line 401 applies it only when `apply_termination_wrapper` is set, and the default is
False (`train.py` line 127). No run-6 launcher sets it.

**Correction.** State that all recorded runs left it off, so the batched env must expose the
time-limit end as a **truncation** and the PPO value target must bootstrap through it. If the
wrapper is ever enabled, the same step becomes a true terminal and the bootstrap must be dropped —
a one-line switch worth naming in the spec so it is not hard-coded.

---

## 18. Reward repeats every step inside the goal radius — note

With `continuing_task=True` the episode does not end at the goal, so the sparse reward is +1 on
**every** step whose post-step position is within 0.45 m. A 400-step episode can return up to 400.
The spec says "never terminated" but does not say the reward repeats; an implementation that gives
the reward once per episode would look plausible and be wrong.

---

## 19. Validation item 4's criterion is not achievable as written — note

"distribution of episode return and of visited cells statistically indistinguishable from the
reference env" cannot be met: findings 1, 2 and 4 are systematic differences in wall behaviour, and
a two-sample test on 256 episodes will eventually detect them. Restate as a bounded-difference
criterion with numbers, for example a stated maximum relative difference in mean episode return and
in per-cell visit share, plus the effect size reported rather than a pass/fail on a hypothesis
test.

---

## 20. Per-map data beyond Large is missing — should-fix

The spec must support all four maps but records only the Large geometry, and the fixtures cover
only Large. Add:

| map | shape | open cells | step cap |
|---|---|---|---|
| UMaze | 5 x 5 | 7 | 300 |
| Open | 5 x 7 | 15 | 300 |
| Medium | 8 x 8 | 26 | 600 |
| Large | 9 x 12 | 46 | 800 |

Also add the section-8 start and goal cells from `env_setups.py` lines 62-82: UMaze pairs
`(3,1)`/`(1,1)`; Open `(3,1)`/`(1,5)` and `(1,1)`/`(3,5)`; Medium `(6,1)`/`(1,6)` and
`(1,1)`/`(6,6)`; Large `(7,1)`/`(1,10)` and `(1,1)`/`(7,10)`. Fixtures for at least one non-Large
map would catch a hard-coded 9 x 12 in any of the three implementations.

---

## 21. Observation dtype through the training stack — note

The reference observation is float64 (`point.py` line 71, spaces declared float64 in
`point_maze.py` lines 355-360), `FlattenObservation` keeps float64, and Stable-Baselines3 casts to
float32 at the network boundary. So the recorded runs simulate in float64 and learn in float32.
The spec should say which precision each GPU implementation simulates in and that the network input
is float32 either way, so the float32-mode tolerance in validation item 1 is understood as a
simulation-precision choice, not a change to the training pipeline.

---

## 22. Dense reward is `exp(-distance)`, not the negative distance — note

`maze_v4.py` line 380 returns `np.exp(-distance)` for `reward_type="dense"`, while the class
docstring (`point_maze.py` line 252) says "the negative Euclidean distance". Only sparse is used
here, but if a dense variant is ever added to the GPU envs, the code is the authority.
