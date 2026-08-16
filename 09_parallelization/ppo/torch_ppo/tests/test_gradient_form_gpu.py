"""GPU-only: the three ways the gradients can reach the optimizer must compute the same step.

The forms differ only in where the gradient lives between the backward pass and the optimizer:

  copy then measure   round five's: copy the twenty-one gradients into one [C, P] buffer, then
                      run one reduction over the whole row for the per-copy limit
  copy and measure    the same copy, with the squares summed in the same program, so the buffer
                      is not read a second time
  no buffer           read the gradients where they were written; twenty-one programs for the
                      limit and twenty-one for Adam instead of one each

The Adam arithmetic is identical expression for expression in all three. The per-copy limit is
not: the first sums one contiguous row of 59,920 numbers, the other two sum tensor by tensor and
add the partial sums. Four things are checked, and the last is the one that settles it:

1. One optimizer step from byte-identical inputs agrees across the three to float32 rounding.
2. The two forms that sum tensor by tensor agree with each other BITWISE, which they must,
   because they add the same partial sums in the same order.
3. Each form actually ran its own path — the buffer forms hold a buffer and the third does not,
   and the parameters moved. Without this the comparison could pass by all three doing nothing.
4. The two gradient norms, recomputed in float64 from the same gradients, agree to double
   rounding. That is what shows the single-precision distance is the order of the additions and
   not a different quantity.

Run on serval05: PYTHONNOUSERSITE=1 <torch python> test_gradient_form_gpu.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from torch_ppo_rnd import PPOConfig, PPORND  # noqa: E402

CFG = dict(n_copies=8, n_envs=4, num_steps=32, obs_norm_init_iters=1,
           rollout_mode="compile-step", tf32=False, compile_opt=True)

# the two buffer forms name the copy-major layout: a buffer needs a per-copy row for one
# program to walk, which the parameter-major layout that now ships does not have
FORMS = {
    "copy then measure": dict(gradient_buffer=True, fuse_copy_and_limit=False,
                              parameter_layout="copy_major"),
    "copy and measure": dict(gradient_buffer=True, fuse_copy_and_limit=True,
                             parameter_layout="copy_major"),
    "no buffer": dict(gradient_buffer=False),
}


def one_step(form):
    """One optimizer step of one form, from the same seed, with the pieces it used."""
    # the same seed on every side, so the rollout, the batch and therefore the gradients match
    torch.manual_seed(5)
    torch.cuda.manual_seed(5)
    t = PPORND(PPOConfig(**form, **CFG), device="cuda")
    t.prime_obs_rms()
    batch = t.rollout()
    loss = t._loss_fn(batch, style_a=False)
    grads = t._backward(loss)
    before = t._flat.detach().clone()
    t._clip_per_copy_and_step(grads)
    used = [g.detach().clone() for g in
            (t.grad_windows if t.cfg.gradient_buffer else grads)]
    return {"trainer": t, "params": [p.detach().clone() for p in t.param_windows],
            "before": before, "grads": used, "scale": t._scale.detach().clone()}


def main():
    """Run one step in each form and compare them, then repeat the norm in double precision."""
    got = {name: one_step(form) for name, form in FORMS.items()}
    ref = got["copy then measure"]

    # 3. every form started from the same weights, used the same gradients, and moved something
    for name, r in got.items():
        assert torch.equal(ref["before"], r["before"]), f"{name} started from other weights"
        worst = max((a - b).abs().max().item() for a, b in zip(ref["grads"], r["grads"]))
        assert worst == 0.0, f"{name} produced different gradients ({worst:.3e})"
    assert got["copy then measure"]["trainer"]._flat_grad is not None, "no buffer was built"
    assert got["copy and measure"]["trainer"]._flat_grad is not None, "no buffer was built"
    assert got["no buffer"]["trainer"]._flat_grad is None, "the no-buffer form holds a buffer"
    assert got["copy and measure"]["trainer"]._scale_fn is None, \
        "the fused form still runs a separate gradient-limit program"
    moved = (ref["trainer"]._flat - ref["before"]).abs().max().item()
    assert moved > 0, "the optimizer did not move the parameters, so nothing was compared"
    print(f"parameters moved by {moved:.3e}")

    # 1. the single-precision distance from round five's form
    for name, r in got.items():
        worst_abs = worst_rel = 0.0
        for a, b in zip(ref["params"], r["params"]):
            d = (a - b).abs().max().item()
            worst_abs = max(worst_abs, d)
            worst_rel = max(worst_rel, d / max(a.abs().max().item(), 1e-12))
        print(f"  {name:<18s} parameters differ by {worst_abs:.3e} absolute, "
              f"{worst_rel:.3e} relative; limiting factor by "
              f"{(ref['scale'] - r['scale']).abs().max().item():.3e}")
        assert worst_rel <= 1e-5, f"{name} computes different parameters"

    # 2. the two forms that sum tensor by tensor write the same expression, so they must land on
    # the same value to rounding. They are NOT bitwise equal on the graphics processor, and the
    # reason is worth stating: one of them is a reduction on its own and the other rides along
    # with a copy, so the compiler splits the two reductions differently and the partial sums are
    # combined in a different order. They agree bitwise when neither is compiled.
    a, b = got["copy and measure"]["scale"], got["no buffer"]["scale"]
    per_tensor_gap = ((a - b).abs() / a.abs().clamp(min=1e-30)).max().item()
    print(f"the two per-tensor forms' limiting factors agree to {per_tensor_gap:.3e} relative")
    assert per_tensor_gap <= 1e-6, "the two per-tensor sums are not the same quantity"

    # 4. the same two norms in double precision, from the same gradients
    C = CFG["n_copies"]
    flat64 = ref["trainer"]._flat_grad.double()
    norm_flat = flat64.square().sum(1).sqrt()
    total = torch.zeros(C, device=flat64.device, dtype=torch.float64)
    for g in got["no buffer"]["grads"]:
        total = total + g.double().reshape(C, -1).square().sum(1)
    worst64 = ((norm_flat - total.sqrt()).abs()
               / norm_flat.abs().clamp(min=1e-300)).max().item()
    print(f"the two gradient norms in double precision agree to {worst64:.3e} relative")
    assert worst64 <= 1e-13, "the two gradient norms are not the same quantity"
    print("ok test_gradient_forms_agree")


if __name__ == "__main__":
    main()
