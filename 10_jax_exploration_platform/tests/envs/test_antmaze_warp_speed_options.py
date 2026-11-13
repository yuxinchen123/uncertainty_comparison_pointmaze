"""Speed options of the Warp AntMaze trainer that change no number, checked on processors.

1. `frame_unroll` of `WarpAntMaze` (set only by the measurement scripts of the RK4 iteration study): the
   loop over the frame_skip physics steps of one environment step, unrolled. The environment's state,
   observations, rewards and episode ends are bit-identical with it on and off, with the RK4 integrator
   at 0.01 s x 5 that the AntMaze units run.
2. The state put on the card before calls (the training command's `--put-state-on-card`): the
   iteration is compiled once, through its second call and a window end, and the state is bit-identical
   to the state of the same iterations without it.
3. Through the training command: a unit stopped after its first iteration without any option, then
   continued in a new process with `--put-state-on-card` and `--compilation-cache-dir`, writes the same
   window records, number for number, as the unit run in one go without them; the cache folder holds
   the compiled programs. So a unit checkpointed today can continue with both options on: neither is in
   the checkpoint or its identity.

On processors MuJoCo Warp is deterministic except the 14 island arrays a physics step returns
unwritten (`warp_antmaze.ISLAND_ARRAYS`), which are left out. Needs the platform_jax_warp interpreter.

Run on a compute node, never on a login node:
  PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu <platform_jax_warp python> -m pytest -q \
      tests/envs/test_antmaze_warp_speed_options.py
"""
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("warp")
ROOT = Path(__file__).resolve().parents[2]
for folder in ("src", "scripts", "tests/envs"):
    sys.path.insert(0, str(ROOT / folder))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from exploration_platform.agents.ppo.config import PPOConfig  # noqa: E402
from exploration_platform.envs.antmaze.am_common import preset  # noqa: E402
from exploration_platform.envs.antmaze.warp_antmaze import ISLAND_ARRAYS, WarpAntMaze  # noqa: E402
from exploration_platform.training import episode_log  # noqa: E402
from exploration_platform.training.runner import Runner  # noqa: E402
from test_antmaze_warp_training import records_of, run_command  # noqa: E402

RK4 = replace(preset("umaze"), physics="warp", continuing_task=False, max_episode_steps=6,
              integrator="RK4", physics_time_step=0.01, frame_skip=5)


def arrays(tree) -> dict:
    """Every array of a tree by its path, the island arrays of the Warp data left out."""
    flat, _ = jax.tree_util.tree_flatten_with_path(tree)
    named = {jax.tree_util.keystr(path): np.asarray(leaf) for path, leaf in flat}
    return {name: value for name, value in named.items() if not any(part in name for part in ISLAND_ARRAYS)}


def differing(one: dict, other: dict) -> list:
    """The paths of the arrays whose bytes differ."""
    assert one.keys() == other.keys()
    return [name for name in one if one[name].tobytes() != other[name].tobytes()]


def stepped(frame_unroll: int, steps: int = 8) -> dict:
    """The arrays of every environment step's outputs, from the spawn with fixed random torques."""
    env = WarpAntMaze(RK4, 2, 2, frame_unroll=frame_unroll)
    step = jax.jit(env.step)
    state = env.reset()
    torques = jax.random.uniform(jax.random.PRNGKey(4), (steps, 2, 2, 8), jnp.float32, -1.0, 1.0)
    outputs = []
    for k in range(steps):
        state, *rest = step(state, torques[k])
        outputs.append((state, *rest))
    return arrays(outputs)


def test_frame_unroll_changes_no_number():
    """Unrolled 5 (no loop left) and 2 (bodies of 2 and a remainder): bit-identical to the loop."""
    reference = stepped(1)
    # the 8 steps of 6-step episodes hold a step-limit reset (the truncation flags, output 4), so the
    # reset path is compared too
    assert any(value.any() for name, value in reference.items() if name.endswith("][4]"))
    for unroll in (5, 2):
        assert differing(reference, stepped(unroll)) == [], unroll


def test_frame_unroll_outside_one_to_frame_skip_is_refused():
    """0 and frame_skip + 1 are not a layout of the frame loop."""
    for unroll in (0, 6):
        with pytest.raises(ValueError, match="frame_unroll must lie in 1..frame_skip"):
            WarpAntMaze(RK4, 1, 1, frame_unroll=unroll)


def iterated(put_on_card: bool):
    """Two iterations, a window end as the training command makes it, one more iteration; returns the
    state's arrays and the number of programs the iteration compiled."""
    cfg = PPOConfig(configuration_preset="antmaze_current", n_copies=2, n_envs=2, num_steps=8,
                    prime_iterations=1, update_style="epoch_minibatch", episode_log=True)
    runner = Runner(cfg, bonus="rnd_next_state", env_cfg=RK4)
    on_card = (lambda tree: jax.device_put(tree, jax.devices()[0])) if put_on_card else (lambda tree: tree)
    state = on_card(runner.prime(runner.init_state(run_seed=3)))
    for k in (1, 2):
        state, _ = runner.iterate(state, runner.lr_argument(k, 3))
    # GOAL: the window end: the episode log's sums and the fault record set back to zero outside the program
    state = state._replace(agent_state={**state.agent_state,
                                        "episode_log": episode_log.clear_window(state.agent_state["episode_log"])})
    state = on_card(state._replace(env_state=runner.env.fault_window(state.env_state)[1]))
    state, _ = runner.iterate(state, runner.lr_argument(3, 3))
    return arrays(state), runner.iterate._cache_size()


def test_state_put_on_card_compiles_the_iteration_once_and_changes_no_number():
    """With the state put on the card, one program through the second call and a window end; the
    final state equals the state without it bit for bit."""
    placed, programs = iterated(True)
    as_built, _ = iterated(False)
    assert programs == 1
    assert differing(as_built, placed) == []


def test_a_checkpointed_unit_continues_with_both_options_and_equals_the_whole_unit():
    """Stopped after its first iteration without options, continued in a new process with the state put
    on the card and the compilation cache on: every window record equals the unit run in one go."""
    options = ["--unit-id", "unit", "--env", "antmaze-umaze", "--physics", "warp",
               "--configuration-preset", "antmaze_current", "--copies-per-cell", "1",
               "--envs-per-copy", "2", "--rollout-steps", "8", "--iterations", "3",
               "--window-iterations", "2", "--episode-steps", "7", "--update-style",
               "epoch_minibatch", "--clip-coef", "0.1", "--prime-iterations", "1",
               "--episode-log", "--reward-setting", "arrival_plus_one_and_step_cost"]
    with tempfile.TemporaryDirectory() as directory:
        whole, parts, cache = (Path(directory) / name for name in ("whole", "parts", "cache"))
        run_command("--run-dir", str(whole), *options)
        with pytest.raises(SystemExit, match="run the same command again"):
            run_command("--run-dir", str(parts), *options, "--time-limit-seconds", "1e-9")
        # the continuation is a new process, as a restarted unit is: JAX decides at a process's
        # first compilation whether it uses the cache, and this test process has compiled already
        subprocess.run([sys.executable, str(ROOT / "scripts/run_training.py"), "--run-dir", str(parts),
                        *options, "--checkpoint-seconds", "1e-9", "--put-state-on-card",
                        "--compilation-cache-dir", str(cache)], check=True, cwd=ROOT)
        # GOAL: the cache was written, and the continued unit says what it ran with
        assert any(path.is_file() for path in cache.rglob("*")), "the compilation cache stayed empty"
        log = (parts / "logs" / "unit.log").read_text()
        assert f"JAX compilation cache: {cache}; state put on the card before calls: yes" in log

        # GOAL: the same window records, number for number, apart from times
        windows = lambda records: [r for r in records if r["record"] == "episode_window"]
        expected, actual = windows(records_of(whole, "unit")), windows(records_of(parts, "unit"))
        assert len(expected) == len(actual) == 2
        timing = {"written_at", "seconds_since_first_iteration"}
        for one, other in zip(expected, actual):
            assert {k: v for k, v in one.items() if k not in timing} == {
                k: v for k, v in other.items() if k not in timing}, one["window_index"]
