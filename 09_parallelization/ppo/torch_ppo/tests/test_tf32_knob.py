"""The matrix-precision knob takes in both directions, whatever the process was left in.

The knob used to be applied only when it was ON, so a trainer asking for exact single precision
inherited the reduced-precision setting from any trainer built before it in the same process.
This test builds the two orders that expose that.

Run: PYTHONNOUSERSITE=1 <python> test_tf32_knob.py   (CPU)
"""
import sys
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from torch_ppo_rnd import PPOConfig, PPORND  # noqa: E402

SMALL = dict(n_copies=2, n_envs=2, num_steps=4, obs_norm_init_iters=1)


def test_exact_after_reduced_in_one_process():
    """A tf32=False trainer built after a tf32=True one still runs at exact single precision."""
    PPORND(PPOConfig(tf32=True, **SMALL), device="cpu")
    assert torch.get_float32_matmul_precision() == "high"
    PPORND(PPOConfig(tf32=False, **SMALL), device="cpu")
    assert torch.get_float32_matmul_precision() == "highest", \
        "a tf32=False trainer inherited the reduced-precision setting"
    print("ok test_exact_after_reduced_in_one_process")


def test_reduced_after_exact_in_one_process():
    """The other order: a tf32=True trainer built after a tf32=False one asks for reduced."""
    PPORND(PPOConfig(tf32=False, **SMALL), device="cpu")
    assert torch.get_float32_matmul_precision() == "highest"
    PPORND(PPOConfig(tf32=True, **SMALL), device="cpu")
    assert torch.get_float32_matmul_precision() == "high", \
        "a tf32=True trainer did not turn the reduced-precision matrix units back on"
    print("ok test_reduced_after_exact_in_one_process")


def test_setting_survives_a_hostile_starting_state():
    """The knob is set from the config, not from whatever the process was left in."""
    for start, want_tf32, expect in [("highest", True, "high"), ("high", False, "highest"),
                                     ("medium", False, "highest"), ("medium", True, "high")]:
        torch.set_float32_matmul_precision(start)
        PPORND(PPOConfig(tf32=want_tf32, **SMALL), device="cpu")
        assert torch.get_float32_matmul_precision() == expect, (start, want_tf32)
    print("ok test_setting_survives_a_hostile_starting_state")


if __name__ == "__main__":
    test_exact_after_reduced_in_one_process()
    test_reduced_after_exact_in_one_process()
    test_setting_survives_a_hostile_starting_state()
    print("ALL TF32-KNOB TESTS PASSED")
