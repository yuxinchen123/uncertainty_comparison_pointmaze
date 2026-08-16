"""GPU-only: the flat-buffer clip and Adam must compute what the previous per-tensor form did.

The previous form walked twenty-one parameter tensors — summing squared gradients per tensor for
the per-copy norm, rescaling each one, then handing them to torch.optim.Adam. The new form does
the same arithmetic over one [C, P] buffer. Rewriting a reduction changes the order in which
floating-point numbers are added, so the two cannot be bitwise equal; what must hold is that a
SINGLE step from identical inputs agrees to float32 rounding, which is what this checks. A
whole-run comparison cannot answer the question, because one differing bit in the first step
changes the actions taken in the second.

Run on serval05: PYTHONNOUSERSITE=1 <torch python> test_flat_optimizer_gpu.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from torch_ppo_rnd import PPOConfig, PPORND  # noqa: E402

CFG = dict(n_copies=4, n_envs=4, num_steps=32, obs_norm_init_iters=1,
           rollout_mode="compile-step", compile_opt=False)


def reference_step(params, grads, m, v, t, lr, eps, max_norm):
    """The previous form: per-tensor norm accumulation, per-tensor rescale, then Adam."""
    C = params[0].shape[0]
    g2 = torch.zeros(C, device=params[0].device, dtype=torch.float32)
    for g in grads:
        g2 = g2 + g.reshape(C, -1).square().sum(1)
    scale = (max_norm / (g2.sqrt() + 1e-6)).clamp(max=1.0)
    b1, b2 = 0.9, 0.999
    bc1 = 1.0 - b1 ** t
    bc2 = 1.0 - b2 ** t
    out = []
    for p, g, mi, vi in zip(params, grads, m, v):
        gs = g * scale.view(C, *([1] * (g.dim() - 1)))
        mi = mi * b1 + gs * (1.0 - b1)
        vi = vi * b2 + gs * gs * (1.0 - b2)
        denom = (vi / bc2).sqrt() + eps
        out.append(p - lr * (mi / bc1) / denom)
    return out


def main():
    """One optimizer step, computed both ways from byte-identical inputs."""
    t = PPORND(PPOConfig(**CFG), device="cuda")
    torch.manual_seed(5)
    t.prime_obs_rms()
    batch = t.rollout()

    # produce a real set of gradients, through the trainer's own gradient path
    loss = t._loss_fn(batch, style_a=False)
    grad_source = t._backward(loss)

    # snapshot the inputs the two forms will share
    params_before = [p.detach().clone() for p in t.trainable]
    grads = [g.detach().clone() for g in t.grad_windows]
    flat_before = t._flat.detach().clone()

    ref = reference_step(params_before, grads,
                         [torch.zeros_like(p) for p in params_before],
                         [torch.zeros_like(p) for p in params_before],
                         t=1, lr=t.cfg.learning_rate, eps=t.cfg.adam_eps,
                         max_norm=t.cfg.max_grad_norm)

    t._clip_per_copy_and_step(grad_source)      # the flat form, same inputs, moments start at 0

    worst_abs = worst_rel = 0.0
    for p, r in zip(t.trainable, ref):
        d = (p.detach() - r).abs().max().item()
        scale = max(r.abs().max().item(), 1e-12)
        worst_abs = max(worst_abs, d)
        worst_rel = max(worst_rel, d / scale)
    moved = (t._flat.detach() - flat_before).abs().max().item()
    print(f"parameters actually moved by {moved:.3e} (a zero here would make the test vacuous)")
    print(f"flat form against the per-tensor form: worst absolute {worst_abs:.3e}, "
          f"worst relative {worst_rel:.3e}")
    assert moved > 0, "the optimizer did not move the parameters, so nothing was compared"
    assert worst_rel <= 1e-5, "the flat optimizer computes something else"
    print("ok test_flat_optimizer_matches_per_tensor_form")


if __name__ == "__main__":
    main()
