"""Tests for `copy_seed_offset`, the knob that lets one logical run be cut into chunks.

A run of C copies can be split across cards by giving each chunk a slice of the copy indices. The
seed index of a copy keys both its initial weights and the environment draws it meets, so the
properties that make a split trustworthy are:

  1. offset 0 is exactly what the platform did before the knob existed — no existing run moves,
  2. two chunks of a run cover disjoint seed indices, and together cover every index once,
  3. chunk j's copy i has BITWISE the weights that copy (j x per_chunk + i) of the whole run has,
     so a chunk is a slice of the same run rather than a different run of the same size,
  4. the offset survives the non-sweep path as well as the sweep path.

Run with the platform's canonical JAX environment registered in
/p/rlprojects/RND/.venvs/ENVS.md (currently /p/rlprojects/RND/.venvs/platform_jax):
PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python test_copy_seed_offset.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from exploration_platform.agents.ppo.config import PPOConfig  # noqa: E402
from exploration_platform.agents.ppo.networks import init_agent_params  # noqa: E402
from exploration_platform.training.sweep import build_sweep, sweep_config  # noqa: E402


def test_offset_zero_is_the_old_behaviour():
    """A configuration with no offset gives exactly the seed indices it always gave."""
    cfg = sweep_config(learning_rates=(1e-3, 1e-4), betas=(), copies_per_group=3)
    assert list(build_sweep(cfg).copy_seed_index) == [0, 1, 2, 0, 1, 2]
    # and the non-sweep path, which has one seed per copy
    assert list(build_sweep(PPOConfig(n_copies=4)).copy_seed_index) == [0, 1, 2, 3]


def test_two_chunks_partition_the_copies():
    """Two chunks of a 8-copy run cover 0..7 exactly once between them, and share nothing."""
    # before: one run of 8 copies has seed indices [0..7]
    # after:  chunk 0 of 4 copies has [0,1,2,3] and chunk 1 has [4,5,6,7]
    first = build_sweep(sweep_config(learning_rates=(1e-3,), copies_per_group=4,
                                     copy_seed_offset=0))
    second = build_sweep(sweep_config(learning_rates=(1e-3,), copies_per_group=4,
                                      copy_seed_offset=4))
    a, b = list(first.copy_seed_index), list(second.copy_seed_index)
    assert a == [0, 1, 2, 3] and b == [4, 5, 6, 7]
    assert set(a).isdisjoint(b), "chunks must share no seed index"
    assert sorted(a + b) == list(range(8)), "the chunks together must cover the run exactly once"


def test_a_chunk_holds_the_whole_run_s_own_copies_bitwise():
    """Chunk 1's weights are bit for bit the second half of the whole run's weights."""
    whole = init_agent_params(8, 0, list(range(8)))
    chunk = init_agent_params(4, 0, list(range(4, 8)))
    for key in ("actor", "critic"):
        for name, value in chunk[key].items():
            expected = np.asarray(whole[key][name])[4:]
            assert np.array_equal(np.asarray(value), expected), (
                f"{key}/{name} of the chunk is not the whole run's copies 4..7")


def test_the_offset_reaches_the_non_sweep_path():
    """A plain configuration with an offset also shifts, so an unswept chunk works too."""
    cfg = PPOConfig(n_copies=3, copy_seed_offset=6)
    assert list(build_sweep(cfg).copy_seed_index) == [6, 7, 8]


def main() -> None:
    """Run every test in this file and report."""
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"ok  {name}")
    print("all copy-seed-offset tests passed")


if __name__ == "__main__":
    main()
