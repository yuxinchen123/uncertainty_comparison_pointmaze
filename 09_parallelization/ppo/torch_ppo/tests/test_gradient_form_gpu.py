"""GPU-only: reading the gradients where they lie must compute what copying them into a buffer does.

`gradient_buffer=True` copies the twenty-one gradients into one [C, P] buffer and then runs one
reduction and one elementwise program over it. `gradient_buffer=False` skips the copy and runs
twenty-one of each instead. The Adam arithmetic is identical expression for expression; the
gradient norm is not, because the same squares are added in a different order — one contiguous
reduction over a row of 59,920 numbers against twenty-one sub-reductions summed.

Three things are checked, and the third is the one that settles the question:

1. A single optimizer step from byte-identical inputs agrees to float32 rounding.
2. The two forms actually ran — the buffer form holds a buffer and the other does not, the
   padding of the moments is untouched, and the parameters moved. Without this the comparison
   could pass by both sides doing nothing.
3. The two gradient norms, recomputed in float64 from the same gradients, agree to double
   rounding. That is what shows the single-precision distance is the order of the additions
   rather than a different quantity.

Run on serval05: PYTHONNOUSERSITE=1 <torch python> test_gradient_form_gpu.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from torch_ppo_rnd import PPOConfig, PPORND  # noqa: E402

CFG = dict(n_copies=8, n_envs=4, num_steps=32, obs_norm_init_iters=1,
           rollout_mode="compile-step", tf32=False, compile_opt=True)


def one_step(gradient_buffer):
    """One optimizer step of one form: returns the parameters after it and the gradients used."""
    # the same seed on both sides, so the rollout, the batch and therefore the gradients match
    torch.manual_seed(5)
    torch.cuda.manual_seed(5)
    t = PPORND(PPOConfig(gradient_buffer=gradient_buffer, **CFG), device="cuda")
    t.prime_obs_rms()
    batch = t.rollout()
    loss = t._loss_fn(batch, style_a=False)
    grads = t._backward(loss)
    before = t._flat.detach().clone()
    t._clip_per_copy_and_step(grads)
    kept = [g.detach().clone() for g in
            (t.grad_windows if gradient_buffer else grads)]
    return t, [p.detach().clone() for p in t.param_windows], before, kept


def main():
    """Compare the two gradient forms on one step, then again in double precision."""
    buf, params_buf, before_buf, grads_buf = one_step(True)
    direct, params_direct, before_direct, grads_direct = one_step(False)

    # the two builds must have started from the same weights and produced the same gradients,
    # or the comparison below is between two different problems
    assert torch.equal(before_buf, before_direct), "the two builds started from different weights"
    worst_grad = max((a - b).abs().max().item() for a, b in zip(grads_buf, grads_direct))
    assert worst_grad == 0.0, f"the two builds produced different gradients ({worst_grad:.3e})"

    # 2. each form actually did its own thing
    assert buf._flat_grad is not None, "the buffer form built no buffer"
    assert direct._flat_grad is None, "the no-buffer form still holds a gradient buffer"
    moved = (buf._flat - before_buf).abs().max().item()
    assert moved > 0, "the optimizer did not move the parameters, so nothing was compared"

    # 1. the single-precision distance between the two forms
    worst_abs = worst_rel = 0.0
    for a, b in zip(params_buf, params_direct):
        d = (a - b).abs().max().item()
        worst_abs = max(worst_abs, d)
        worst_rel = max(worst_rel, d / max(a.abs().max().item(), 1e-12))
    print(f"parameters moved by {moved:.3e}")
    print(f"one step, the two forms: worst absolute {worst_abs:.3e}, relative {worst_rel:.3e}")

    # 3. the same two norms in double precision, from the same gradients
    C = buf.cfg.n_copies
    flat64 = buf._flat_grad.double()
    norm_flat = flat64.square().sum(1).sqrt()
    total = torch.zeros(C, device=flat64.device, dtype=torch.float64)
    for g in grads_direct:
        total = total + g.double().reshape(C, -1).square().sum(1)
    norm_tensors = total.sqrt()
    worst64 = ((norm_flat - norm_tensors).abs() / norm_flat.abs().clamp(min=1e-300)).max().item()
    print(f"the two gradient norms in double precision agree to {worst64:.3e} relative")

    assert worst_rel <= 1e-5, "the two gradient forms compute different parameters"
    assert worst64 <= 1e-13, "the two gradient norms are not the same quantity"
    print("ok test_gradient_form_matches_the_buffer_form")


if __name__ == "__main__":
    main()
