"""GPU-only equivalence tests for the CUDA-graph fast paths.

1. Rollout: graph replay must equal the same compiled step run uncaptured, bitwise.
2. Update: the captured update (capturable Adam, static buffers) must track the uncaptured
   update to float32 noise (capturable Adam computes bias correction on-GPU, so bitwise
   equality is not guaranteed; the gate is max parameter difference <= 1e-5 after 3 iters).

Run on serval05: PYTHONNOUSERSITE=1 <torch python> test_capture_gpu.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from torch_ppo_rnd import PPOConfig, PPORND  # noqa: E402

CFG = dict(n_copies=4, n_envs=4, num_steps=32, obs_norm_init_iters=1)


def test_rollout_capture_bitwise():
    """compile-step (uncaptured) vs capture: identical kernels, so bitwise-equal batches."""
    a = PPORND(PPOConfig(rollout_mode="compile-step", **CFG), device="cuda")
    b = PPORND(PPOConfig(rollout_mode="capture", **CFG), device="cuda")
    torch.manual_seed(7); a.prime_obs_rms()
    torch.manual_seed(7); b.prime_obs_rms()
    b._build_rollout_graph()
    for it in range(3):
        torch.manual_seed(100 + it)
        ba = a.rollout()
        torch.manual_seed(100 + it)
        bb = b.rollout()
        for k in ["obs", "actions", "old_logprob", "adv", "ret_ext", "ret_int", "rnd_input"]:
            assert torch.equal(ba[k], bb[k]), \
                f"iter {it} {k} max|d|={(ba[k]-bb[k]).abs().max().item():.3e}"
        torch.manual_seed(200 + it)
        a.update_epoch_minibatch(ba)
        torch.manual_seed(200 + it)
        b.update_epoch_minibatch(bb)
    print("ok test_rollout_capture_bitwise")


def test_update_capture_tracks_eager():
    """Captured update vs uncaptured update: same rollouts, parameters within 1e-5."""
    for style in ["full_batch", "epoch_minibatch"]:
        a = PPORND(PPOConfig(rollout_mode="compile-step", update_style=style,
                             fused_adam=True, **CFG), device="cuda")
        b = PPORND(PPOConfig(rollout_mode="compile-step", update_style=style,
                             fused_adam=True, capture_update=True, **CFG), device="cuda")
        torch.manual_seed(7); a.prime_obs_rms()
        torch.manual_seed(7); b.prime_obs_rms()
        upd_a = a.update_full_batch if style == "full_batch" else a.update_epoch_minibatch
        for it in range(3):
            torch.manual_seed(100 + it)
            ba = a.rollout()
            torch.manual_seed(100 + it)
            bb = b.rollout()
            torch.manual_seed(200 + it)   # style B permutation draw parity
            upd_a(ba)
            torch.manual_seed(200 + it)
            b.update_captured(bb)
            worst = max((pa - pb).abs().max().item()
                        for pa, pb in zip(a.trainable, b.trainable))
            assert worst <= 1e-5, f"{style} iter {it}: worst param diff {worst:.3e}"
        print(f"ok test_update_capture_tracks_eager[{style}] (worst {worst:.3e})")


def test_learning_rate_reaches_the_graph():
    """A zero learning rate must freeze the parameters when the update is captured.

    The annealed learning rate is written into a device tensor that the captured Adam reads.
    If capture had instead frozen a python float (the classic capture defect), the replay
    would keep stepping at the old rate and this test would fail — nothing else in the suite
    exercises it, because the benchmarks all run at a constant rate.
    """
    t = PPORND(PPOConfig(rollout_mode="capture", capture_update=True, fused_adam=True,
                         one_graph=True, **CFG), device="cuda")
    torch.manual_seed(13)
    t.prime_obs_rms()
    t._build_iteration_graph()
    t.iteration_captured()                      # one normal iteration at the configured rate
    before = [p.detach().clone() for p in t.trainable]
    moved = False
    t._lr_t.fill_(0.0)                          # anneal all the way down
    for _ in range(3):
        t.iteration_captured()
    after = [p.detach().clone() for p in t.trainable]
    worst = max((a - b).abs().max().item() for a, b in zip(before, after))
    print(f"parameter movement over 3 iterations at learning rate 0: {worst:.3e}")
    assert worst == 0.0, "the captured update ignored the annealed learning rate"
    # and a non-zero rate must move them again, so the test cannot pass by never updating
    t._lr_t.fill_(3e-4)
    t.iteration_captured()
    moved = max((a - p.detach()).abs().max().item() for a, p in zip(after, t.trainable)) > 0
    assert moved, "parameters never move, so the zero-rate check proves nothing"
    print("ok test_learning_rate_reaches_the_graph")


if __name__ == "__main__":
    test_rollout_capture_bitwise()
    test_update_capture_tracks_eager()
    test_learning_rate_reaches_the_graph()
