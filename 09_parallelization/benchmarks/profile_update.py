"""Break the update stage into its parts, to see what the sixteen minibatch steps actually pay.

The update stage is the largest phase of an iteration. It runs sixteen times: gather the
minibatch out of the batch buffers, forward, backward, clip the per-copy gradient norm, take an
Adam step, zero the gradients. This times each part on the real shapes so the optimisation
targets are chosen from measurement rather than from reading the code.

Usage (through the H100 lock wrapper): python profile_update.py --n-copies 128
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "ppo" / "torch_ppo"))
RESULTS = Path(__file__).resolve().parent / "results"


def cuda_time(fn, reps=30, warmup=8):
    """Median CUDA time of fn() in microseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000.0)
    return sorted(ts)[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-copies", type=int, default=128)
    args = ap.parse_args()
    from torch_ppo_rnd import PPORND, production_config

    C = args.n_copies
    t = PPORND(production_config(C), device="cuda")
    t.prime_obs_rms()
    t._build_iteration_graph()
    t.iteration_captured()

    cfg = t.cfg
    Brows = cfg.num_steps * cfg.n_envs
    mb = Brows // cfg.num_minibatches
    steps = cfg.update_epochs * cfg.num_minibatches

    # one minibatch, gathered the way the update body does it
    idx = t._perm[0][:, :mb]
    def gather_one():
        out = {}
        for key in t._U_KEYS:
            u = t._U[key]
            ix = idx.unsqueeze(-1).expand(C, mb, u.shape[-1]) if u.dim() == 3 else idx
            out[key] = u.gather(1, ix)
        return out
    batch = gather_one()

    def fwd():
        return t._loss_fn(batch, style_a=False)

    def fwd_bwd():
        t._loss_fn(batch, style_a=False).backward()

    # gradients must exist before the clip can be timed on its own
    fwd_bwd()

    if hasattr(t, "_opt_fn"):
        # the flat-buffer build: clip, Adam and zeroing are one chain over one buffer
        parts = {
            "gather the minibatch (9 tensors)": cuda_time(gather_one),
            "forward": cuda_time(fwd),
            "forward and backward": cuda_time(fwd_bwd),
            "clip, Adam and zero over the flat buffer": cuda_time(lambda: t._opt_fn()),
            "Adam step": 0.0,
            "zero the gradients (19 tensors)": 0.0,
        }
    else:
        def clip_only():
            g2 = torch.zeros(C, device=t.device)
            for p in t.trainable:
                g2 = g2 + p.grad.reshape(C, -1).square().sum(1)
            scale = (cfg.max_grad_norm / (g2.sqrt() + 1e-6)).clamp(max=1.0)
            for p in t.trainable:
                p.grad.mul_(scale.view(C, *([1] * (p.dim() - 1))))

        def adam_only():
            t.opt.step()

        def zero_only():
            for p in t.trainable:
                p.grad.zero_()

        parts = {
            "gather the minibatch (9 tensors)": cuda_time(gather_one),
            "forward": cuda_time(fwd),
            "forward and backward": cuda_time(fwd_bwd),
            "clip the per-copy gradient norm (walks 19 tensors twice)": cuda_time(clip_only),
            "Adam step": cuda_time(adam_only),
            "zero the gradients (19 tensors)": cuda_time(zero_only),
        }
    parts["backward alone (difference)"] = parts["forward and backward"] - parts["forward"]

    clip_key = ("clip, Adam and zero over the flat buffer"
                if "clip, Adam and zero over the flat buffer" in parts
                else "clip the per-copy gradient norm (walks 19 tensors twice)")
    per_step = (parts["gather the minibatch (9 tensors)"] + parts["forward and backward"]
                + parts[clip_key] + parts["Adam step"] + parts["zero the gradients (19 tensors)"])
    print(f"\n== one minibatch step at {C} copies, {mb} rows per copy ==")
    for k, v in sorted(parts.items(), key=lambda kv: -kv[1]):
        print(f"  {v:9.1f} us  {k}")
    print(f"  {per_step:9.1f} us  one step, summed")
    print(f"  {per_step*steps/1000:9.2f} ms  x {steps} steps = the update stage, uncaptured")
    print(f"\nShare of one step: "
          f"gather {parts['gather the minibatch (9 tensors)']/per_step*100:.0f}%, "
          f"forward+backward {parts['forward and backward']/per_step*100:.0f}%, "
          f"clip+Adam+zero {parts[clip_key]/per_step*100:.0f}%, "
          f"Adam {parts['Adam step']/per_step*100:.0f}%, "
          f"zero {parts['zero the gradients (19 tensors)']/per_step*100:.0f}%")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{time.strftime('%Y-%m-%d-%H-%M-%S')}_profile_update_C{C}.json"
    out.write_text(json.dumps({"n_copies": C, "rows_per_minibatch": mb, "steps": steps,
                               "parts_us": parts, "per_step_us": per_step}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
