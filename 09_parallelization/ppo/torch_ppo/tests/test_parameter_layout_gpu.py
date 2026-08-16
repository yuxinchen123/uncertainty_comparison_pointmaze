"""GPU-only: the two buffer layouts hold the same numbers, and what that is worth on the card.

`parameter_layout="copy_major"` gives the buffer one row per copy, so a parameter's window is
strided across copies. `parameter_layout="parameter_major"` gives each parameter one contiguous
block, so every window a multiplication or the optimizer touches is contiguous. Nothing about the
arithmetic is written differently: the same expressions run over the same numbers at different
addresses, and on the processor the two train to BITWISE equal parameters
(tests/test_torch_ppo.py::test_the_two_buffer_layouts_train_identically).

On the graphics card they are not bitwise equal, and the reason is worth stating rather than
hiding behind a tolerance: the multiplication library chooses its kernel partly from the operand's
layout, so a contiguous weight and a strided one can be multiplied by different kernels, which sum
the same products in a different order. This test measures how far apart that puts one training
iteration, and requires it to stay inside float32 rounding.

It also checks that each build really used the layout it was asked for — the contiguous form's
weight windows must be contiguous and the strided form's must not — because a test that compared
two identical builds would pass while measuring nothing.

Run on serval05: PYTHONNOUSERSITE=1 <torch python> test_parameter_layout_gpu.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from torch_ppo_rnd import PPOConfig, PPORND  # noqa: E402

CFG = dict(n_copies=8, n_envs=4, num_steps=32, obs_norm_init_iters=1,
           rollout_mode="compile-step", tf32=False, compile_opt=True,
           gradient_buffer=False)


def one_iteration(layout):
    """One training iteration in one layout, from the same seed, with the pieces to compare."""
    torch.manual_seed(5)
    torch.cuda.manual_seed(5)
    t = PPORND(PPOConfig(parameter_layout=layout, **CFG), device="cuda")
    t.prime_obs_rms()
    batch = t.rollout()
    loss = t._loss_fn(batch, style_a=False)
    grads = [g.detach().clone() for g in t._backward(loss)]
    before = [w.detach().clone() for w in t.param_windows]
    t._clip_per_copy_and_step([g.clone() for g in grads])
    return {"trainer": t, "grads": grads, "before": before,
            "after": [w.detach().clone() for w in t.param_windows]}


def main():
    """Run one iteration in each layout and measure how far apart the card puts them."""
    got = {layout: one_iteration(layout) for layout in ("copy_major", "parameter_major")}
    cm, pm = got["copy_major"], got["parameter_major"]

    # each build used the layout it was asked for: the weights of the contiguous form are
    # contiguous and those of the strided form are not. Without this the comparison is vacuous.
    weights = [i for i, w in enumerate(pm["trainer"].param_windows) if w.numel() // w.shape[0] > 4]
    assert weights, "no parameter big enough to be a multiplication operand was found"
    assert all(pm["trainer"].param_windows[i].is_contiguous() for i in weights), \
        "the parameter-major build has non-contiguous weights"
    assert not any(cm["trainer"].param_windows[i].is_contiguous() for i in weights), \
        "the copy-major build has contiguous weights, so the two builds are the same"

    # the two started from the same weights, or nothing below means anything
    for a, b in zip(cm["before"], pm["before"]):
        assert torch.equal(a, b), "the two layouts started from different weights"
    biggest = max(g.abs().max().item() for g in cm["grads"])
    worst_grad = max((a - b).abs().max().item() for a, b in zip(cm["grads"], pm["grads"]))
    moved = max((a - b).abs().max().item() for a, b in zip(cm["after"], cm["before"]))
    worst_param = max((a - b).abs().max().item() for a, b in zip(cm["after"], pm["after"]))
    scale = max(max(w.abs().max().item() for w in cm["after"]), 1e-12)

    print(f"parameters moved by {moved:.3e} (a zero here would make the test vacuous)")
    print(f"gradients differ by {worst_grad:.3e} absolute against a largest of {biggest:.3e}")
    print(f"parameters after one step differ by {worst_param:.3e} absolute, "
          f"{worst_param / scale:.3e} relative")
    assert moved > 0, "the optimizer did not move the parameters"
    assert worst_grad / biggest <= 1e-5, "the two layouts compute different gradients"
    assert worst_param / scale <= 1e-5, "the two layouts compute different parameters"
    print("ok test_the_two_layouts_agree_to_float32_rounding")


if __name__ == "__main__":
    main()
