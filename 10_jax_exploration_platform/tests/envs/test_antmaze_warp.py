"""The Warp AntMaze stepper against the gates of the JAX one, with MuJoCo Warp on processors.

Needs an interpreter with warp-lang (`/p/rlprojects/RND/.venvs/platform_jax_warp`); the
canonical `platform_jax` has none, so this file is skipped there. The measurements live in
`benchmarks/env/antmaze_warp_checks.py`; this file only asserts on them, with the bounds of
`spec.md` ("Correctness gates") where a gate exists.

Run on a compute node, never on a login node:
  WARP_CACHE_PATH=<node-local folder> PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu \
      <platform_jax_warp python> -m pytest -q tests/envs/test_antmaze_warp.py
"""
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("warp")
BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(BASE / "benchmarks" / "env"))
import antmaze_warp_checks as checks  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from exploration_platform.envs.antmaze.am_common import cell_center, preset  # noqa: E402
from exploration_platform.envs.antmaze.warp_antmaze import WarpAntMaze, prepare_kernel_folder  # noqa: E402


def test_reset_shapes_and_spawn():
    """Reset puts every env at the start cell's centre in float32; the contact pool is shared."""
    env = WarpAntMaze(preset("umaze"), 2, 3)
    s = env.reset()
    assert s.data.qpos.shape == (6, env.nq) and s.data.qpos.dtype == jnp.float32
    start = cell_center((3, 1), env.rows, env.cols)
    np.testing.assert_array_equal(np.asarray(s.data.qpos[:, 0:2]), np.tile(np.float32(start), (6, 1)))
    assert float(s.data.qpos[:, 2].min()) == 0.75 and float(jnp.abs(s.data.qvel).max()) == 0.0
    # one pool of 6 x 48 contact slots for all six environments, and 160 rows for each of them
    assert s.data._impl.contact__dist.shape == (6 * 48,)
    assert s.data._impl.efc__force.shape == (6, 160)
    _, obs = env.respawn(s)
    assert obs.shape == (2, 3, 29) and obs.dtype == jnp.float32
    # the environment's own record of its configuration names the physics that runs
    assert env.cfg.physics == "warp" and (env.naconmax, env.njmax) == (6 * 48, 160)


def test_pool_sizes_come_from_the_configuration():
    """The shared pool is slots per environment x environments; both sizes are configuration fields."""
    cfg = replace(preset("umaze"), warp_contact_slots_per_environment=8,
                  warp_constraint_rows_per_environment=64)
    env = WarpAntMaze(cfg, 2, 3)
    s = env.reset()
    assert (env.naconmax, env.njmax) == (6 * 8, 64)
    assert s.data._impl.contact__dist.shape == (48,) and s.data._impl.efc__force.shape == (6, 64)
    # a measurement may still pass its own sizes, which replace the configuration's
    assert WarpAntMaze(cfg, 2, 3, naconmax=100, njmax=32).naconmax == 100


def test_kernel_folder_is_node_local_and_created():
    """The kernel folder is under WARP_CACHE_PATH when the job sets it; without it, under the
    node's /localtmp or temporary folder, in the user's own folder, and never in the home folder."""
    env = WarpAntMaze(preset("umaze"), 1, 1)
    assert Path(env.kernel_folder).is_dir() and env.kernel_folder == prepare_kernel_folder()
    if "WARP_CACHE_PATH" in os.environ:
        assert Path(os.path.realpath(os.environ["WARP_CACHE_PATH"])) in Path(env.kernel_folder).parents
    # a new process with the variable unset chooses the folder itself and creates it
    script = ("import sys; sys.path.insert(0, sys.argv[1]);"
              "from exploration_platform.envs.antmaze.warp_antmaze import prepare_kernel_folder;"
              "print('KERNEL_FOLDER', prepare_kernel_folder())")
    environment = {name: value for name, value in os.environ.items() if name != "WARP_CACHE_PATH"}
    output = subprocess.run([sys.executable, "-c", script, str(BASE / "src")], env=environment,
                            capture_output=True, text=True, check=True).stdout
    folder = Path(next(line.split(" ", 1)[1] for line in output.splitlines()
                       if line.startswith("KERNEL_FOLDER ")))
    user = os.environ.get("USER") or os.environ["LOGNAME"]
    assert folder.is_dir() and folder.parent.name == "warp_kernel_cache"
    assert folder.parent.parent.name == user and Path.home() not in folder.parents


def test_step_contract_determinism_and_x64_boundary():
    """Six outputs with the platform shapes, float32 under global x64, and repeatable numbers."""
    assert jax.config.jax_enable_x64
    env = WarpAntMaze(preset("umaze"), 2, 2)
    call = checks.rollout_call(env)
    act = checks.torques("contract torques", 12, 4)
    s1, (obs1, rew1, term1, trunc1, final1) = call(env.reset(), act)
    s2, (obs2, *_) = call(env.reset(), act)
    assert obs1.shape == (12, 2, 2, 29) and rew1.shape == (12, 2, 2) and final1.shape == obs1.shape
    assert obs1.dtype == jnp.float32 and rew1.dtype == jnp.float32 and s1.data.qpos.dtype == jnp.float32
    assert term1.dtype == jnp.bool_ and trunc1.dtype == jnp.bool_
    np.testing.assert_array_equal(np.asarray(obs1), np.asarray(obs2))
    assert bool(jnp.isfinite(obs1).all())
    # the ants fell onto the floor during the 12 steps, so the pool was used and nothing was dropped
    report = env.raise_on_overflow(s1)
    assert report["nacon_peak"] > 0 and report["nefc_peak"] > 0


def test_one_step_parity_against_c_mujoco_and_jax():
    """Gate 1 of spec.md for the Warp stepper: 1e-4 at contact-free states, 5e-2 in contact."""
    result = checks.one_step_parity()
    print("one-step parity:", result)
    free, contact = result["contact_free"], result["in_contact"]
    assert free["warp_against_c"]["count"] > 0 and contact["warp_against_c"]["count"] > 0
    assert free["warp_against_c"]["largest"] < 1e-4
    assert contact["warp_against_c"]["largest"] < 5e-2
    assert free["warp_against_jax"]["largest"] < 1e-4
    assert not result["pool"]["dropped"]


def test_trajectory_divergence_structure():
    """Gate 2 of spec.md: roundoff before the first contact, under one maze cell over 20 steps."""
    result = checks.trajectory_divergence()
    print("trajectory divergence:", result)
    assert result["warp"]["all_finite"]
    assert result["warp"]["largest_before_the_first_contact"] < 1e-4
    assert result["warp"]["largest_over_all_steps"] < 4.0


def test_resting_height():
    """After 1,200 passive steps the Warp ant rests at the reference height, and keeps a C-settled state."""
    result = checks.resting_height()
    print("resting height:", result)
    assert abs(result["warp"]["height"] - checks.REFERENCE_RESTING_HEIGHT) < 1e-4
    assert abs(result["c_mujoco"]["height"] - checks.REFERENCE_RESTING_HEIGHT) < 1e-4
    assert result["warp"]["c_settled_state_moved_by"] < 1e-3


def test_copy_isolation():
    """Changing the other environments' torques never changes a watched environment; a nearly full pool neither."""
    for margin in (None, 1.03):
        result = checks.copy_isolation(16, pool_margin=margin)
        print("copy isolation:", result)
        assert result["all_finite"] and result["unwatched_environments_that_differ"] > 0
        assert not result["pool_run_a"]["dropped"] and not result["pool_run_b"]["dropped"]
        # on processors MuJoCo Warp is deterministic, so both comparisons are exact
        assert result["others_changed"]["bitwise_equal"] == result["watched"]
        assert result["same_run_repeated"]["bitwise_equal"] == result["watched"]
        assert result["others_changed"]["first_step_with_a_difference"] is None
    assert result["pool_fill"] > 0.9


def test_small_pool_is_reported():
    """A contact pool below the need is reported as dropped contacts, and the guard raises."""
    report = checks.small_pool_is_reported()
    print("small pool:", report)
    assert report["dropped"] and max(report["nacon_peak"], report["ncollision_peak"]) > report["naconmax"]
    env = WarpAntMaze(preset("umaze"), 1, 1)
    state = env.reset()._replace(nacon_peak=jnp.asarray(env.naconmax + 1, jnp.int32))
    with pytest.raises(RuntimeError, match="dropped part of the physics") as stop:
        env.raise_on_overflow(state)
    assert "warp_contact_slots_per_environment" in str(stop.value)


@pytest.mark.parametrize("continuing_task", [True, False])
def test_semantics_match_the_jax_environment(continuing_task):
    """Reward, termination, truncation, the non-finite respawn and the reset equal the JAX environment's."""
    result = checks.semantics_against_jax(continuing_task)
    print("semantics:", result)
    assert result["mismatches"] == []
    assert result["largest_observation_difference"] < 1e-3


def test_trainer_like_rollout_with_resets_and_donated_state():
    """A donated state through lax.scan with resets inside: exact reset counts, float32 physics."""
    result = checks.trainer_like_rollout(2, 3, rollout_steps=16, calls=4, episode_steps=10)
    print("trainer-like rollout:", result)
    assert result["non_finite_resets"] == 0
    assert result["resets_counted"] == result["resets_expected_from_the_step_limit"] == result["dones_in_outputs"]
    assert result["step_count_values"] == [result["step_count_expected"]]
    assert result["dtypes"] == {"qpos": "float32", "observation": "float32", "statistic": "float64",
                                "x64_enabled": True}
    # two, as for the JAX environment: the state reset() returns and the state a compiled call returns
    # compile separately although every leaf has the same shape and type (measured 2026-11-12)
    assert result["compiled_programs"] <= 2
    assert not result["pool"]["dropped"]


def test_non_finite_inputs_are_respawned():
    """Every poisoned coordinate is respawned and counted, as in the JAX environment.

    MuJoCo Warp normalises the torso quaternion at the start of a physics step and turns a quaternion
    with a NaN into a valid one, so the state after the step is finite. The environment therefore
    also tests the state it is handed (and the state after each physics step), and that case is
    respawned like the others.
    """
    result = checks.non_finite_inputs()
    print("non-finite inputs:", result)
    assert len(result["cases"]) == 12
    for case in result["cases"]:
        assert case["jax_respawned"] and case["warp_respawned"], case
    assert result["healthy_environment_respawned"] == {"warp": False, "jax": False}
    assert result["healthy_environment_unchanged"] == {"warp": True, "jax": True}
