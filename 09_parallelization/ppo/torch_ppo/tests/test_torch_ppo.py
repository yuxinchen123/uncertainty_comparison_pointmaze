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


if __name__ == "__main__":
    test_same_seed_bit_identical()
    test_copy_isolation()
    test_finite_and_learns_predictor()
