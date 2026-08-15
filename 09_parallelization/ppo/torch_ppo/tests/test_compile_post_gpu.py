"""GPU-only: the compiled post-rollout body must compute what the eager one computes.

Two checks, because the naive "run both trainers for a few iterations and compare" mixes the
question with accumulation: the intrinsic filter and the running statistics are carried across
iterations, so a last-bit difference in iteration 1 is amplified by iteration 3 and the absolute
deviation says nothing about whether the two bodies compute the same function.

  1. Isolation: one trainer, one rollout, then run the eager body and the compiled body from
     BYTE-IDENTICAL inputs (buffers and statistics snapshotted and restored between the two
     calls) and compare every output relative to its own magnitude. Compiling fuses reductions
     and may reorder floating-point accumulation, so the gate is 1e-5 RELATIVE — far tighter
     than any real formula or ordering mistake could hide under.
  2. Drift: three full iterations of two independently built trainers, reported (not gated) so
     the accumulated difference is on the record.

Run on serval05: PYTHONNOUSERSITE=1 <torch python> test_compile_post_gpu.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from torch_ppo_rnd import PPOConfig, PPORND  # noqa: E402

CFG = dict(n_copies=4, n_envs=4, num_steps=32, obs_norm_init_iters=1,
           rollout_mode="compile-step", fused_adam=True)
FIELDS = ["obs", "actions", "old_logprob", "adv", "ret_ext", "ret_int", "vext_old",
          "rnd_input", "rnd_tf"]


def rel(a, b):
    """Largest deviation relative to the magnitude of the quantity itself."""
    scale = max(a.abs().max().item(), b.abs().max().item(), 1e-12)
    return (a - b).abs().max().item() / scale


def test_isolation():
    """Eager body and compiled body, run from identical inputs, must agree relatively."""
    t = PPORND(PPOConfig(compile_post=True, **CFG), device="cuda")
    torch.manual_seed(3)
    t.prime_obs_rms()
    # fill the rollout buffers WITHOUT running the post body, so both bodies below start from
    # the same statistics (the post body updates them, so a rollout() call would advance them)
    torch.manual_seed(21)
    t._Z.normal_()
    t._rollout_body()

    stats = [t.int_filter, t.obs_rms.mean, t.obs_rms.var, t.obs_rms.count,
             t.int_rms.mean, t.int_rms.var, t.int_rms.count]
    before = [x.clone() for x in stats]

    t._post_fn()                                   # the compiled body
    compiled_out = {k: t._U[k].clone() for k in FIELDS}
    compiled_stats = [x.clone() for x in stats]

    for dst, src in zip(stats, before):            # restore the inputs exactly
        dst.copy_(src)
    t._post_body_impl()                            # the eager body, byte-identical inputs
    eager_out = {k: t._U[k].clone() for k in FIELDS}
    eager_stats = [x.clone() for x in stats]

    worst_field = max((rel(eager_out[k], compiled_out[k]), k) for k in FIELDS)
    worst_stat = max(rel(a, b) for a, b in zip(eager_stats, compiled_stats))
    print(f"isolation: worst relative field deviation {worst_field[0]:.3e} ({worst_field[1]})")
    print(f"isolation: worst relative statistic deviation {worst_stat:.3e}")
    assert worst_field[0] <= 1e-5 and worst_stat <= 1e-5, "compiled post body computes something else"
    print("ok test_isolation")


def test_drift_report():
    """Three full iterations of both builds; reported so the accumulation is on the record."""
    a = PPORND(PPOConfig(compile_post=False, **CFG), device="cuda")
    b = PPORND(PPOConfig(compile_post=True, **CFG), device="cuda")
    torch.manual_seed(3); a.prime_obs_rms()
    torch.manual_seed(3); b.prime_obs_rms()
    for it in range(3):
        torch.manual_seed(700 + it)
        ba = a.rollout()
        torch.manual_seed(700 + it)
        bb = b.rollout()
        torch.manual_seed(800 + it)
        a.update_epoch_minibatch(ba)
        torch.manual_seed(800 + it)
        b.update_epoch_minibatch(bb)
    worst_param = max((pa - pb).abs().max().item() for pa, pb in zip(a.trainable, b.trainable))
    worst_param_rel = max(rel(pa, pb) for pa, pb in zip(a.trainable, b.trainable))
    print(f"drift after 3 iterations: worst parameter deviation {worst_param:.3e} "
          f"({worst_param_rel:.3e} relative)")
    assert worst_param_rel <= 1e-4, "parameters drifted more than accumulation explains"
    print("ok test_drift_report")


if __name__ == "__main__":
    test_isolation()
    test_drift_report()
