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
    t._backward_into_flat(t._losses(batch, style_a=False))
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
    t._backward_into_flat(t._losses(batch, style_a=False))
    for window, ref in zip(t.grad_windows, reference):
        assert torch.equal(window, ref), "the second backward pass accumulated instead of writing"
    print("ok test_gradients_land_in_the_flat_buffer_without_accumulating")


if __name__ == "__main__":
    test_same_seed_bit_identical()
    test_copy_isolation()
    test_finite_and_learns_predictor()
    test_gradients_land_in_the_flat_buffer_without_accumulating()
