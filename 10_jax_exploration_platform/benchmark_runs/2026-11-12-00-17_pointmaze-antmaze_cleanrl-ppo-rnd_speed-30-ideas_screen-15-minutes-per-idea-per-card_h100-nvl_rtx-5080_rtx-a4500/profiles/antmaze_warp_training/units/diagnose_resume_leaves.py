"""Which arrays of the training state differ between a unit run in one go and the same unit saved and restored?

Run on a compute node (processors):  <platform_jax_warp python> diagnose_resume_leaves.py
Trains a small Warp AntMaze unit three ways: in one go (A), in one go a second time (B), and with a save to a
checkpoint file and a restore into a newly built trainer after the first iteration (C). Prints the arrays in
which A and C differ and those in which A and B differ. An array that differs between two runs in one go is
memory MuJoCo Warp never writes with this model (the step hands back a buffer it did not fill).
"""
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

BASE = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(BASE / "src"))
import jax  # noqa: E402
import numpy as np  # noqa: E402
from exploration_platform.agents.ppo.config import PPOConfig  # noqa: E402
from exploration_platform.envs.antmaze.am_common import preset  # noqa: E402
from exploration_platform.training.checkpoint import read_checkpoint, restore_state, save_checkpoint  # noqa: E402
from exploration_platform.training.runner import Runner  # noqa: E402

TOTAL, STOP_AFTER = 3, 1
ENV = replace(preset("umaze"), physics="warp", continuing_task=False, max_episode_steps=6)
CFG = PPOConfig(configuration_preset="antmaze_current", n_copies=2, n_envs=1, num_steps=4, prime_iterations=1,
                update_style="epoch_minibatch", num_minibatches=2, episode_log=True)


def arrays(state) -> dict:
    """Every array of a training state, by its path in the tree."""
    flat, _ = jax.tree_util.tree_flatten_with_path(state)
    return {jax.tree_util.keystr(path): np.asarray(leaf) for path, leaf in flat}


def trained(stop_after=None) -> dict:
    """The state's arrays after TOTAL iterations; with stop_after, saved and restored after that many."""
    runner = Runner(CFG, bonus="rnd_next_state", env_cfg=ENV)
    state = runner.prime(runner.init_state(run_seed=17))
    for k in range(1, (stop_after or TOTAL) + 1):
        state, _ = runner.iterate(state, runner.lr_argument(k, TOTAL))
    if stop_after is not None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "unit.npz"
            save_checkpoint(path, state, {}, runner.bonus.constants)
            saved = read_checkpoint(path)
        runner = Runner(CFG, bonus="rnd_next_state", env_cfg=ENV, bonus_constants=saved["bonus_constants"])
        state = restore_state(runner.init_state(run_seed=17), saved["leaves"])
        for k in range(stop_after + 1, TOTAL + 1):
            state, _ = runner.iterate(state, runner.lr_argument(k, TOTAL))
    return arrays(state)


def differing(one: dict, other: dict) -> list:
    """The paths of the arrays whose bytes differ, with each array's shape and type."""
    return [f"{name} {one[name].shape} {one[name].dtype}" for name in one
            if one[name].tobytes() != other[name].tobytes()]


first, second, resumed = trained(), trained(), trained(STOP_AFTER)
print(f"arrays in the state: {len(first)}; inside the Warp data: {sum('.env_state.data' in name for name in first)}")
for label, other in (("run in one go against saved and restored", resumed), ("run in one go against run in one go again", second)):
    names = differing(first, other)
    print(f"{label}: {len(names)} arrays differ")
    for name in names:
        print("   ", name)
