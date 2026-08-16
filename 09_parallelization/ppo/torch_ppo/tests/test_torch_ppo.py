"""Torch PPO+RND tests: determinism, copy isolation, both update styles (spec section 17).

Run: PYTHONNOUSERSITE=1 <python> test_torch_ppo.py   (CPU)
"""
import sys
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from torch_ppo_rnd import PPOConfig, PPORND  # noqa: E402

SMALL = dict(n_copies=4, n_envs=2, num_steps=8, obs_norm_init_iters=1)


def run_iters(trainer, iters, seed=123):
    """Deterministic short run: fix the global RNG so both trainers draw identical noise."""
    torch.manual_seed(seed)
    trainer.prime_obs_rms()
    for it in range(iters):
        batch = trainer.rollout()
        if trainer.cfg.update_style == "full_batch":
            trainer.update_full_batch(batch)
        else:
            trainer.update_epoch_minibatch(batch)


def test_same_seed_bit_identical():
    """Two same-seed trainers end bit-identical after 2 iterations (both styles)."""
    for style in ["full_batch", "epoch_minibatch"]:
        a = PPORND(PPOConfig(update_style=style, **SMALL), device="cpu")
        b = PPORND(PPOConfig(update_style=style, **SMALL), device="cpu")
        run_iters(a, 2)
        run_iters(b, 2)
        for pa, pb in zip(a.trainable, b.trainable):
            assert torch.equal(pa, pb), style
    print("ok test_same_seed_bit_identical")


def test_copy_isolation():
    """Perturbing copy 2's weights must leave copies 0, 1, 3 bit-identical after training.

    Catches any cross-copy coupling: shared statistics, global grad norm, mean over copies.
    """
    for style in ["full_batch", "epoch_minibatch"]:
        a = PPORND(PPOConfig(update_style=style, **SMALL), device="cpu")
        b = PPORND(PPOConfig(update_style=style, **SMALL), device="cpu")
        with torch.no_grad():
            b.actor["W0"][2] += 0.05         # perturb only copy 2
        run_iters(a, 2)
        run_iters(b, 2)
        keep = [0, 1, 3]
        for pa, pb in zip(a.trainable, b.trainable):
            assert torch.equal(pa[keep], pb[keep]), f"{style}: copies 0/1/3 diverged"
            # copy 2 must actually have moved differently (the perturbation did something)
        assert not torch.equal(a.actor["W0"][2], b.actor["W0"][2])
    print("ok test_copy_isolation")


def test_finite_and_learns_predictor():
    """Losses stay finite and the RND predictor error falls over a few iterations."""
    t = PPORND(PPOConfig(**SMALL), device="cpu")
    torch.manual_seed(0)
    t.prime_obs_rms()
    first = last = None
    for it in range(4):
        batch = t.rollout()
        loss = t.update_epoch_minibatch(batch)
        assert loss == loss, "loss is NaN"
        m = float(batch["rint_mean"].mean())
        first = m if first is None else first
        last = m
    assert last < first, f"intrinsic reward did not fall: {first} -> {last}"
    print("ok test_finite_and_learns_predictor")


def test_gradients_land_in_the_flat_buffer_without_accumulating():
    """The gradient path writes the flat buffer, never accumulates, and never touches padding.

    Three things must hold, and each one would hide a real defect if it did not:
    no parameter carries an attached gradient (an attached one would make autograd ADD, which
    is the cost this path exists to avoid); the flat buffer holds exactly what autograd
    produced; and a second backward pass from the same inputs OVERWRITES rather than doubles,
    which is what makes zeroing unnecessary. The alignment padding must also stay zero, since
    the optimiser reads the whole buffer including it.
    """
    t = PPORND(PPOConfig(**SMALL), device="cpu")
    torch.manual_seed(0)
    t.prime_obs_rms()
    batch = t.rollout()

    loss = t._losses(batch, style_a=False)
    reference = [g.detach().clone() for g in torch.autograd.grad(loss, t.trainable)]
    t._backward(t._losses(batch, style_a=False))
    assert all(p.grad is None for p in t.trainable), "a parameter still carries a gradient"
    for window, ref in zip(t.grad_windows, reference):
        assert torch.equal(window, ref), "the flat buffer does not hold what autograd produced"
    assert max(float(r.abs().max()) for r in reference) > 0, "no gradient, nothing was compared"

    # the padding: everything in the buffer that no window covers must still be zero
    covered = torch.zeros_like(t._flat_grad, dtype=torch.bool)
    flat_base = t._flat_grad.reshape(-1)
    for window in t.grad_windows:
        start = (window.data_ptr() - t._flat_grad.data_ptr()) // t._flat_grad.element_size()
        rows, per_row = window.shape[0], window.numel() // window.shape[0]
        for c in range(rows):
            covered.reshape(-1)[start + c * t._flat.shape[1]:
                                start + c * t._flat.shape[1] + per_row] = True
    assert float(flat_base[~covered.reshape(-1)].abs().max()) == 0.0, "padding was written"

    # a second pass from identical inputs must overwrite, not double
    t._backward(t._losses(batch, style_a=False))
    for window, ref in zip(t.grad_windows, reference):
        assert torch.equal(window, ref), "the second backward pass accumulated instead of writing"
    print("ok test_gradients_land_in_the_flat_buffer_without_accumulating")


def test_every_parameter_window_is_aligned():
    """Every parameter window starts on a sixteen-byte boundary, for every copy, in both layouts.

    This is what lets the matrix-multiply library use its four-numbers-at-a-time kernels. In the
    copy-major layout it holds only if BOTH each window's offset and the per-copy row length are
    multiples of four numbers; in the parameter-major layout only if each block's per-copy length
    is. Either way the second copy's addresses are the ones that catch a wrong row length, so the
    test checks them as well as the first copy's.
    """
    for layout, buffer in (("copy_major", True), ("copy_major", False),
                           ("parameter_major", False)):
        t = PPORND(PPOConfig(parameter_layout=layout, gradient_buffer=buffer, **SMALL),
                   device="cpu")
        # counted rather than zipped: a zip against an empty list checks nothing and still
        # passes, and the gradient buffer is optional, so the count is what makes this
        # non-vacuous
        assert len(t.trainable) == 21, f"{len(t.trainable)} parameters, not twenty-one"
        groups = [t.trainable, t.param_windows, t.m_windows, t.v_windows]
        if buffer:
            assert len(t.grad_windows) == 21, "the buffer form built no gradient windows"
            groups.append(t.grad_windows)
        checked = 0
        for group in groups:
            for tensor in group:
                for copy in range(min(2, tensor.shape[0])):
                    assert tensor[copy].data_ptr() % 16 == 0, \
                        (f"{layout}: window {tuple(tensor.shape)} copy {copy} is not on a "
                         f"sixteen-byte boundary")
                    checked += 1
        assert checked == len(groups) * 21 * 2, f"only {checked} addresses were checked"
    print("ok test_every_parameter_window_is_aligned")


def test_the_two_buffer_layouts_train_identically():
    """The parameter-major layout must be BITWISE equal to the copy-major one, no-buffer both.

    The two hold the same numbers in a different order and run the same twenty-one programs over
    them, so nothing about the arithmetic changes and there is no tolerance to argue about. A
    difference here would mean a window points at the wrong numbers.
    """
    out = {}
    for layout in ("copy_major", "parameter_major"):
        torch.manual_seed(5)
        t = PPORND(PPOConfig(parameter_layout=layout, gradient_buffer=False, **SMALL),
                   device="cpu")
        t.prime_obs_rms()
        for _ in range(2):
            t.update_epoch_minibatch(t.rollout())
        out[layout] = [p.detach().clone() for p in t.trainable]
    moved = max(float(p.abs().max()) for p in out["copy_major"])
    assert moved > 0, "the parameters are all zero, so nothing was compared"
    for a, b in zip(out["copy_major"], out["parameter_major"]):
        assert torch.equal(a, b), "the two layouts trained to different parameters"
    print("ok test_the_two_buffer_layouts_train_identically")


if __name__ == "__main__":
    test_same_seed_bit_identical()
    test_copy_isolation()
    test_finite_and_learns_predictor()
    test_gradients_land_in_the_flat_buffer_without_accumulating()
    test_every_parameter_window_is_aligned()
    test_the_two_buffer_layouts_train_identically()
