"""The AntMaze ceiling and the fault record: the model, C MuJoCo, and both physics implementations.

The model has one plane on the wall tops (2 m) whose normal points down and which only the torso
sphere meets, so an ant thrown upward cannot rise above the walls and cross them
(`am_common.build_antmaze_xml`). Checked here:

1. the model of all three maps: one ceiling plane, and exactly one geom pair more than the model
   without it (torso against ceiling);
2. the fault record's arithmetic on a hand-made case (`faults.add_step`);
3. nine ants stepped one physics step at a time (`CASES`): two thrown upward at 10 and 20 m/s,
   which must touch the ceiling and stay below the wall tops, and seven that stand (with no
   torque, and with all four ankles pushed to either end of their range by full torque), walk,
   fall over or lie on their back, which must never touch the ceiling. In C MuJoCo of the same
   model, in the JAX implementation and, where the interpreter has warp-lang, in the Warp
   implementation;
4. the fault record the environment keeps for those nine ants against the heights and speeds the
   test recorded itself;
5. one physics step from C-MuJoCo states in contact with the ceiling, against C MuJoCo;
6. the training command: AntMaze window records hold the two fault fields, PointMaze records do
   not.

Run on a compute node, never on a login node:
  PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu <platform_jax or platform_jax_warp python> -m pytest -q -s \
      tests/envs/test_antmaze_ceiling.py
"""
import functools
import importlib.util
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for folder in ("src", "scripts"):
    sys.path.insert(0, str(ROOT / folder))
import exploration_platform  # noqa: E402,F401  (64-bit statistics on, as in training)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from exploration_platform.envs.antmaze import faults  # noqa: E402
from exploration_platform.envs.antmaze.am_common import (CEILING_CONTACT_HEIGHT, CEILING_HEIGHT,  # noqa: E402
                                                         CEILING_ZONE_HEIGHT, TORSO_GEOM,
                                                         build_antmaze_xml, preset)

HAS_WARP = importlib.util.find_spec("warp") is not None
IMPLEMENTATIONS = ["jax", pytest.param("warp", marks=pytest.mark.skipif(
    not HAS_WARP, reason="this interpreter has no warp-lang (platform_jax_warp holds it)"))]
EXAMPLES = ROOT / "configs/experiments"
PHYSICS_STEPS = 1000     # 10 seconds of 0.01 s physics steps
# one environment step = one physics step, so every physics step is seen; no step limit, no goal end
CFG = replace(preset("umaze"), frame_skip=1, max_episode_steps=10 ** 6)
CASES = ("thrown upward at 10 m/s", "thrown upward at 20 m/s",
         "pushes all four ankles toward the bent end of their range with full torque",
         "pushes all four ankles toward the straight end of their range with full torque",
         "stands, no torque",
         "walks: one leg after the other at 1 Hz with a quarter of the full torque",
         "walks: one leg after the other at 0.5 Hz with full torque",
         "falls over: tumbling at 8 rad/s, no torque",
         "on its back: no torque for 2 s, then the torques of the first walk")
THROWN, ORDINARY, WALKING = (0, 1), (2, 3, 4, 5, 6, 7, 8), (5, 6)


def case_inputs(init_qpos):
    """The nine ants' start states and torques: (qpos [9, 15], qvel [9, 14], torques [steps, 9, 8]).

    The actuators of ant.xml in order: hip 4, ankle 4, hip 1, ankle 1, hip 2, ankle 2, hip 3,
    ankle 3. Ankles 1 and 4 turn between 30 and 70 degrees, ankles 2 and 3 between -70 and -30.
    """
    n = len(CASES)
    qpos = np.tile(np.asarray(init_qpos, np.float64), (n, 1))
    qvel = np.zeros((n, 14))
    torques = np.zeros((PHYSICS_STEPS, n, 8))
    # GOAL: the two thrown ants leave the spawn upward; the tumbling ant turns about its x axis;
    # the ant on its back starts turned over (a half turn about x: quaternion 0 1 0 0).
    qvel[0, 2], qvel[1, 2], qvel[7, 3] = 10.0, 20.0, 8.0
    qpos[8, 3:7] = (0.0, 1.0, 0.0, 0.0)
    # GOAL: constant full ankle torques toward each end of the ankles' ranges, all four at once.
    ankle_sign = np.array([0, 1, 0, 1, 0, -1, 0, -1], np.float64)
    torques[:, 2], torques[:, 3] = ankle_sign, -ankle_sign
    # GOAL: the two walks (`walking_torques`), chosen in C MuJoCo among 36 periodic patterns
    # (units/diagnose_ordinary_and_thrown_ants.py of the speed study): both moved the ant more
    # than 1 m in 10 s and left it upright.
    slow_walk = walking_torques(amplitude=0.25, hertz=1.0)
    torques[:, 5], torques[:, 6] = slow_walk, walking_torques(amplitude=1.0, hertz=0.5)
    torques[200:, 8] = slow_walk[200:]
    return qpos, qvel, torques


def walking_torques(amplitude, hertz, phase=(1.5 * np.pi, 0.0, 0.5 * np.pi, np.pi), ankle_lag=0.5 * np.pi):
    """Periodic torques [steps, 8]: each hip a sine, its ankle the same sine `ankle_lag` ahead.

    phase: the hips' phases in actuator order (legs 4, 1, 2, 3); the default drives one leg after
    the other, a quarter period apart.
    before: time t in seconds
    after:  hip torque amplitude x sin(2 pi hertz t + phase), ankle torque amplitude x
            sin(2 pi hertz t + phase + ankle_lag) with the sign of the ankle's range
    """
    t = np.arange(PHYSICS_STEPS)[:, None] * 0.01
    angle = 2 * np.pi * hertz * t + np.asarray(phase)[None]
    torques = np.zeros((PHYSICS_STEPS, 8))
    torques[:, 0::2] = amplitude * np.sin(angle)
    torques[:, 1::2] = amplitude * np.sin(angle + ankle_lag) * np.array([1.0, 1.0, -1.0, -1.0])
    return torques


def spawn_qpos(cfg):
    """The spawn of a configuration: the model's own start pose with the torso over the start cell."""
    from exploration_platform.envs.antmaze.am_common import cell_center
    from exploration_platform.envs.pointmaze.pm_common import MAPS
    model = mujoco.MjModel.from_xml_string(build_antmaze_xml(cfg))
    rows, cols = len(MAPS[cfg.map_name]), len(MAPS[cfg.map_name][0])
    qpos = np.asarray(model.qpos0, np.float64).copy()
    qpos[0:2] = cell_center(cfg.start_cell, rows, cols)
    return qpos


def c_rollout(qpos0, qvel0, torques, cfg=CFG):
    """Ants in C MuJoCo: per physics step the torso position, its velocity, the whole
    state, and whether the step's collision pass held a contact with the ceiling.

    Returns {"position" [steps, n, 3], "velocity" [steps, n, 3], "ceiling_contact" [steps, n] bool,
    "qpos" [steps + 1, n, 15], "qvel" [steps + 1, n, 14]} (the two last include the start state).
    """
    model = mujoco.MjModel.from_xml_string(build_antmaze_xml(cfg))
    ceiling = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ceiling")
    steps, n = torques.shape[:2]
    out = {"position": np.zeros((steps, n, 3)), "velocity": np.zeros((steps, n, 3)),
           "ceiling_contact": np.zeros((steps, n), bool),
           "qpos": np.zeros((steps + 1, n, model.nq)), "qvel": np.zeros((steps + 1, n, model.nv))}
    for b in range(n):
        data = mujoco.MjData(model)
        data.qpos[:], data.qvel[:] = qpos0[b], qvel0[b]
        mujoco.mj_forward(model, data)
        out["qpos"][0, b], out["qvel"][0, b] = data.qpos, data.qvel
        for k in range(steps):
            data.ctrl[:] = torques[k, b]
            mujoco.mj_step(model, data)
            out["position"][k, b], out["velocity"][k, b] = data.qpos[0:3], data.qvel[0:3]
            out["ceiling_contact"][k, b] = bool((data.contact.geom[:data.ncon] == ceiling).any())
            out["qpos"][k + 1, b], out["qvel"][k + 1, b] = data.qpos, data.qvel
    return out


def make_environment(implementation, cfg, n):
    """One environment per ant (n policies x 1 environment) with the named physics."""
    if implementation == "warp":
        from exploration_platform.envs.antmaze.warp_antmaze import WarpAntMaze
        return WarpAntMaze(replace(cfg, physics="warp"), n, 1)
    from exploration_platform.envs.antmaze.mjx_antmaze import JaxAntMaze
    return JaxAntMaze(cfg, n, 1)


def ceiling_contacts(implementation, env, data, ceiling):
    """[B] int32: each environment's contacts with the ceiling in the physics data of its last step.

    JAX: every environment has its own contact arrays, [B, slots]; a slot is a contact when its
    distance is under its margin. Warp: all environments share one pool; the first `nacon` slots
    are contacts, and each names its environment.
    """
    if implementation == "jax":
        contact = data._impl.contact
        live = (contact.geom == ceiling).any(-1) & (contact.dist < contact.includemargin)
        return live.sum(-1).astype(jnp.int32)
    impl = data._impl
    slot = jnp.arange(impl.contact__dist.shape[0])
    live = ((slot < impl.nacon.max()) & (impl.contact__geom == ceiling).any(-1)
            & (impl.contact__dist < impl.contact__includemargin))
    return jnp.zeros((env.B,), jnp.int32).at[impl.contact__worldid].add(live.astype(jnp.int32))


@functools.lru_cache(maxsize=None)
def reference():
    """The nine ants in C MuJoCo, and the inputs they were given."""
    qpos0, qvel0, torques = case_inputs(spawn_qpos(CFG))
    return {"inputs": (qpos0, qvel0, torques), **c_rollout(qpos0, qvel0, torques)}


@functools.lru_cache(maxsize=None)
def stepped(implementation):
    """The nine ants in one implementation, one physics step per environment step.

    Returns the environment, its compiled single step, the state after the rollout, and per physics
    step the torso position [steps, n, 3], its velocity [steps, n, 3] and the ceiling contacts
    [steps, n].
    """
    qpos0, qvel0, torques = reference()["inputs"]
    n = len(CASES)
    env = make_environment(implementation, CFG, n)
    ceiling = mujoco.mj_name2id(mujoco.MjModel.from_xml_string(build_antmaze_xml(CFG)),
                                mujoco.mjtObj.mjOBJ_GEOM, "ceiling")

    def rollout(state, torque_steps):
        """Scan the environment's own step; per step the torso before any reset and the contacts."""
        def body(st, torque):
            """One environment step of all nine ants."""
            st, _, _, _, _, final_obs = env.step(st, torque.reshape(n, 1, 8))
            return st, (final_obs[:, 0, 0:3], final_obs[:, 0, 15:18],
                        ceiling_contacts(implementation, env, st.data, ceiling))
        return jax.lax.scan(body, state, torque_steps)

    start = env.reset()
    start = start._replace(data=start.data.replace(qpos=jnp.asarray(qpos0, jnp.float32),
                                                   qvel=jnp.asarray(qvel0, jnp.float32)))
    state, (position, velocity, contacts) = jax.jit(rollout)(start, jnp.asarray(torques, jnp.float32))
    return {"env": env, "step": jax.jit(env.step), "state": state,
            "position": np.asarray(position, np.float64), "velocity": np.asarray(velocity, np.float64),
            "ceiling_contacts": np.asarray(contacts)}


def one_step_in_ceiling_contact(implementation, thrown_ant=0):
    """One physics step of an implementation from consecutive C-MuJoCo states of a thrown ant, the
    first of them one step before its first ceiling contact, against C MuJoCo's next states.

    Returns the C states used (their physics-step numbers), which of them had a ceiling contact in
    C, and per state the largest absolute position difference after the step, in metres.
    """
    ref, run = reference(), stepped(implementation)
    first = int(np.argmax(ref["ceiling_contact"][:, thrown_ant]))
    # as many states as the environment has ants, so the compiled step of `stepped` is used again
    states = np.arange(first - 1, first - 1 + len(CASES))
    env = run["env"]
    start = env.reset()
    start = start._replace(data=start.data.replace(
        qpos=jnp.asarray(ref["qpos"][states, thrown_ant], jnp.float32),
        qvel=jnp.asarray(ref["qvel"][states, thrown_ant], jnp.float32)))
    after, *_ = run["step"](start, jnp.zeros((len(states), 1, 8), jnp.float32))
    difference = np.abs(np.asarray(after.data.qpos, np.float64) - ref["qpos"][states + 1, thrown_ant]).max(-1)
    return {"c_states": states.tolist(),
            "states_in_ceiling_contact": [int(k) for k in states if ref["ceiling_contact"][k, thrown_ant]],
            "largest_position_difference_per_state": [float(x) for x in difference]}


def case_numbers(position, contacts):
    """Per ant: its largest torso height, its physics steps with a ceiling contact, and how far
    it moved on the floor plane; a list of dictionaries in the order of `CASES`."""
    return [{"case": case, "largest_torso_height": float(position[:, b, 2].max()),
             "physics_steps_with_a_ceiling_contact": int((contacts[:, b] > 0).sum()),
             "physics_steps": int(position.shape[0]),
             "moved_on_the_floor_plane": float(np.linalg.norm(position[-1, b, 0:2] - position[0, b, 0:2]))}
            for b, case in enumerate(CASES)]


def height_table(name, position, contacts):
    """Print one line per ant: the largest torso height and the steps with a ceiling contact."""
    for row in case_numbers(position, contacts):
        print(f"{name}: {row['case']}: largest torso height {row['largest_torso_height']:.3f} m, "
              f"{row['physics_steps_with_a_ceiling_contact']} of {row['physics_steps']} physics steps "
              f"with a ceiling contact, moved {row['moved_on_the_floor_plane']:.2f} m on the floor plane")


@pytest.mark.parametrize("map_name", ["umaze", "medium", "large"])
def test_the_model_has_one_ceiling_and_one_pair_more(map_name):
    """Every map's model holds one plane on the wall tops facing down, and the pairs that the
    collision bits allow are those of the model without it plus torso against ceiling."""
    xml = build_antmaze_xml(preset(map_name))
    model = mujoco.MjModel.from_xml_string(xml)
    ceiling = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ceiling")
    assert ceiling >= 0 and model.geom_type[ceiling] == mujoco.mjtGeom.mjGEOM_PLANE
    np.testing.assert_allclose(model.geom_pos[ceiling], [0.0, 0.0, CEILING_HEIGHT])
    assert CEILING_HEIGHT == 2.0 and abs(CEILING_CONTACT_HEIGHT - 1.74) < 1e-12
    # the plane's normal is its frame's z axis: (0, 0, -1)
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, model.geom_quat[ceiling])
    np.testing.assert_allclose(rotation.reshape(3, 3)[:, 2], [0.0, 0.0, -1.0], atol=1e-12)
    # the ceiling has the floor's contact parameters
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    for field in ("geom_margin", "geom_gap", "geom_condim", "geom_friction", "geom_solref", "geom_solimp"):
        np.testing.assert_array_equal(getattr(model, field)[ceiling], getattr(model, field)[floor])

    def pairs(m):
        """Names of the geom pairs on different bodies whose collision bits match."""
        name = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
        return {frozenset((name(a), name(b))) for a in range(m.ngeom) for b in range(a + 1, m.ngeom)
                if m.geom_bodyid[a] != m.geom_bodyid[b]
                and ((m.geom_contype[a] & m.geom_conaffinity[b]) or (m.geom_contype[b] & m.geom_conaffinity[a]))}

    # GOAL: the same model as it was before the ceiling: the plane removed, the torso's bit back.
    root = ET.fromstring(xml)
    worldbody = root.find("worldbody")
    worldbody.remove(worldbody.find("geom[@name='ceiling']"))
    del root.find(f".//geom[@name='{TORSO_GEOM}']").attrib["contype"]
    before = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    assert before.ngeom == model.ngeom - 1
    assert pairs(model) - pairs(before) == {frozenset(("ceiling", TORSO_GEOM))}
    assert pairs(before) - pairs(model) == set()
    # nothing else of the model changed: the same time step, solver and bodies
    assert (before.opt.timestep, before.opt.iterations, before.opt.ls_iterations, before.opt.integrator) == (
        model.opt.timestep, model.opt.iterations, model.opt.ls_iterations, model.opt.integrator)
    np.testing.assert_array_equal(before.body_mass, model.body_mass)


def test_fault_record_on_a_hand_made_case():
    """Two policies with two environments each, two physics steps: the zone count adds one per
    environment that was above 1.5 m after any physics step, the speed is the largest finite one."""
    assert CEILING_ZONE_HEIGHT == 1.5
    # environments 0, 1 belong to policy 0 and 2, 3 to policy 1
    heights = jnp.asarray([[0.6, 1.4, 1.51, 1.49], [0.7, 1.6, 1.9, 1.5]], jnp.float32)
    speeds = jnp.asarray([[1.0, 3.0, 0.5, 7.0], [2.0, jnp.nan, jnp.inf, 6.0]], jnp.float32)
    zone, speed = faults.add_step(jnp.asarray([5, 0], jnp.int32), jnp.asarray([2.5, 9.0], jnp.float32),
                                  heights, speeds, 2, 2)
    # policy 0: environment 1 was above 1.5 m once -> 5 + 1; policy 1: environment 2 in both
    # physics steps counts once, environment 3 at exactly 1.5 m does not count -> 0 + 1
    np.testing.assert_array_equal(np.asarray(zone), [6, 1])
    # policy 0: the largest finite speed 3.0 replaces 2.5; policy 1: 7.0 does not replace 9.0
    np.testing.assert_array_equal(np.asarray(speed), [3.0, 9.0])
    assert zone.dtype == jnp.int32 and speed.dtype == jnp.float32
    # a second step adds to the first; a height that is not a number counts as outside the zone
    zone, speed = faults.add_step(zone, speed, jnp.asarray([[jnp.nan, 1.7, 0.0, 0.0]], jnp.float32),
                                  jnp.asarray([[4.0, 0.0, 0.0, 12.5]], jnp.float32), 2, 2)
    np.testing.assert_array_equal(np.asarray(zone), [7, 1])
    np.testing.assert_array_equal(np.asarray(speed), [4.0, 12.5])
    # torso height and speed from a state: z is qpos[2], the speed the length of qvel[0:3]
    qpos = jnp.zeros((1, 15), jnp.float32).at[0, 2].set(1.25)
    qvel = jnp.zeros((1, 14), jnp.float32).at[0, 0:3].set(jnp.asarray([3.0, 4.0, 12.0], jnp.float32))
    height, speed_now = faults.torso_height_and_speed(qpos, qvel)
    assert float(height[0]) == 1.25 and float(speed_now[0]) == 13.0


def test_c_mujoco_thrown_ants_stop_at_the_ceiling_and_the_others_never_touch_it():
    """In C MuJoCo of the same model: the thrown ants touch the ceiling and stay below the wall
    tops, where without a ceiling they would rise to 5.8 m and 21 m; the seven others never
    touch the ceiling and stay under the ceiling zone."""
    ref = reference()
    height_table("C MuJoCo", ref["position"], ref["ceiling_contact"])
    for b in THROWN:
        assert ref["ceiling_contact"][:, b].any(), CASES[b]
        assert CEILING_CONTACT_HEIGHT < ref["position"][:, b, 2].max() < CEILING_HEIGHT, CASES[b]
    for b in ORDINARY:
        assert not ref["ceiling_contact"][:, b].any(), CASES[b]
        assert ref["position"][:, b, 2].max() < CEILING_ZONE_HEIGHT, CASES[b]
    for b in WALKING:
        assert np.linalg.norm(ref["position"][-1, b, 0:2] - ref["position"][0, b, 0:2]) > 1.0, CASES[b]
    # GOAL: an ant thrown upward at up to 40 m/s keeps its torso centre below the wall tops.
    qpos0, qvel0, torques = ref["inputs"]
    for speed in (5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0):
        launch = qvel0[0:1].copy()
        launch[0, 2] = speed
        thrown = c_rollout(qpos0[0:1], launch, torques[:200, 0:1])
        print(f"C MuJoCo: thrown upward at {speed:.0f} m/s: largest torso height "
              f"{thrown['position'][:, 0, 2].max():.3f} m")
        assert thrown["position"][:, 0, 2].max() < CEILING_HEIGHT, speed


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_thrown_ants_stay_below_the_wall_tops(implementation):
    """An ant thrown upward at 10 or 20 m/s touches the ceiling and its torso centre stays below
    the wall tops (2 m) at every physics step."""
    run = stepped(implementation)
    height_table(implementation, run["position"], run["ceiling_contacts"])
    for b in THROWN:
        assert run["ceiling_contacts"][:, b].sum() > 0, CASES[b]
        assert CEILING_CONTACT_HEIGHT < run["position"][:, b, 2].max() < CEILING_HEIGHT, CASES[b]
    assert int(np.asarray(run["state"].nan_count).sum()) == 0
    if implementation == "warp":
        assert not run["env"].overflow_report(run["state"])["dropped"]


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_ants_that_stand_walk_fall_over_or_lie_on_their_back_never_touch_the_ceiling(implementation):
    """Zero contacts with the ceiling in 1,000 physics steps for each of the seven ordinary ants,
    and their torso centres stay under the ceiling zone (1.5 m)."""
    run = stepped(implementation)
    for b in ORDINARY:
        assert run["ceiling_contacts"][:, b].sum() == 0, CASES[b]
        assert run["position"][:, b, 2].max() < CEILING_ZONE_HEIGHT, CASES[b]
    # the walking torques move the ant along the floor
    for b in WALKING:
        assert np.linalg.norm(run["position"][-1, b, 0:2] - run["position"][0, b, 0:2]) > 0.3, CASES[b]


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_the_environment_fault_record_equals_the_recorded_heights_and_speeds(implementation):
    """The environment's own two numbers per policy after the rollout: the steps above 1.5 m and
    the largest torso speed, as the test's record of every physics step gives them."""
    run = stepped(implementation)
    record, cleared = run["env"].fault_window(run["state"])
    expected_steps = (run["position"][:, :, 2] > CEILING_ZONE_HEIGHT).sum(0)
    expected_speed = np.linalg.norm(run["velocity"], axis=-1).max(0)
    print(f"{implementation}: fault record {record}")
    assert record["ceiling_zone_steps_per_copy"] == expected_steps.tolist()
    np.testing.assert_allclose(record["largest_torso_speed_per_copy"], expected_speed, rtol=1e-5)
    # the thrown ants were in the zone and fast; the ordinary ants never were in the zone
    assert all(record["ceiling_zone_steps_per_copy"][b] > 0 for b in THROWN)
    assert all(record["ceiling_zone_steps_per_copy"][b] == 0 for b in ORDINARY)
    assert record["largest_torso_speed_per_copy"][1] > 15.0
    # reading sets the two numbers back to zero and leaves everything else as it was
    assert not np.asarray(cleared.ceiling_zone_steps).any()
    assert not np.asarray(cleared.largest_torso_speed).any()
    np.testing.assert_array_equal(np.asarray(cleared.step_count), np.asarray(run["state"].step_count))
    # a respawn also sets them to zero
    respawned, _ = run["env"].respawn(run["state"])
    assert not np.asarray(respawned.ceiling_zone_steps).any()
    assert not np.asarray(respawned.largest_torso_speed).any()


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_one_step_from_states_in_ceiling_contact_agrees_with_c_mujoco(implementation):
    """One physics step from nine consecutive C-MuJoCo states of the ant thrown at 10 m/s, the
    first of them one step before its first ceiling contact: within the in-contact gate of spec.md
    (5e-2 m) of C MuJoCo's next state."""
    result = one_step_in_ceiling_contact(implementation)
    print(f"{implementation}: one physics step from C states of the ant thrown at 10 m/s: {result}")
    assert len(result["states_in_ceiling_contact"]) > 0
    assert max(result["largest_position_difference_per_state"]) < 5e-2


def run_command(*options) -> None:
    """Call the training command in this process with the given command line."""
    import run_training
    command_line = sys.argv
    sys.argv = ["run_training.py", *options]
    try:
        run_training.main()
    finally:
        sys.argv = command_line
        run_training.STOP["requested"] = False


def test_window_records_hold_the_fault_fields_for_antmaze_only():
    """The AntMaze example's window records hold one count and one speed per policy; the PointMaze
    example's records hold neither field, and the PointMaze environment has no fault record."""
    from exploration_platform.envs.pointmaze.jax_pointmaze import JaxPointMaze
    assert not hasattr(JaxPointMaze, "fault_window")
    short = ["--copies", "2", "--iterations", "2", "--window-iterations", "1",
             "--prime-iterations", "1", "--rollout-steps", "16"]
    names = set(faults.RECORD_NAMES.values())
    assert names == {"ceiling_zone_steps_per_copy", "largest_torso_speed_per_copy"}
    cases = (("pointmaze_large_cleanrl-ppo-rnd_goal-termination_reward-arrival-plus-one_example"
              "_steps-1024000.toml", False),
             ("antmaze_umaze_cleanrl-ppo-rnd_goal-termination_reward-arrival-plus-one_example"
              "_steps-204800.toml", True))
    for file_name, has_fields in cases:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_command("--run-dir", str(run_dir), "--unit-id", "unit",
                        "--configuration-file", str(EXAMPLES / file_name), *short)
            records = [json.loads(line) for line in (run_dir / "data" / "unit.jsonl").read_text()
                       .splitlines() if line.strip()]
            windows = [r for r in records if r["record"] == "episode_window"]
            assert len(windows) == 2
            for record in records:
                assert bool(names & set(record)) == (has_fields and record["record"] == "episode_window")
            if has_fields:
                for window in windows:
                    steps, speed = (window["ceiling_zone_steps_per_copy"],
                                    window["largest_torso_speed_per_copy"])
                    assert len(steps) == 2 and all(isinstance(x, int) and x >= 0 for x in steps)
                    # 16 steps x 4 environments per policy in a window: the count cannot exceed it
                    assert all(x <= 16 * 4 for x in steps)
                    assert len(speed) == 2 and all(0.0 < x < 100.0 for x in speed)
