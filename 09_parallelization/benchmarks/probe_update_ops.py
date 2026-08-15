"""What the update stage's individual operations cost against what their bytes should cost.

At 1,024 copies and above the trainer's tensors are far larger than any cache, so every
operation in the update stage should be limited by memory bandwidth, and the useful question
is what fraction of the card's bandwidth each one reaches. This probe measures, at one copy
count:

  1. a plain large copy, which is the bandwidth this card actually delivers to a streaming
     kernel and therefore the reference every other line is judged against;
  2. the batched matrix multiplication of every layer of the update's forward pass, in the
     two forms the trainer could write it — `baddbmm` (bias folded into the multiplication)
     and `bmm` followed by a separate bias-and-activation pass;
  3. the optimiser chain over the flat parameter buffer, against its own byte count;
  4. the gradient accumulation the backward pass performs into the flat gradient buffer;
  5. the permutation gather of one epoch's batch.

Usage (through the H100 lock wrapper): python probe_update_ops.py --n-copies 1024
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).resolve().parent / "results"
F32 = 4

# update-stage layers as (name, rows, in, out); the first entry is the packed actor+critic
# first layer, which reads 4 inputs and writes 128 outputs
LAYERS = [("actor+critic layer 1 (packed)", 128, 4, 128),
          ("actor layer 2", 128, 64, 64),
          ("critic layer 2", 128, 64, 64),
          ("predictor layer 1", 128, 4, 256),
          ("predictor layer 2", 128, 256, 128),
          ("predictor layer 3", 128, 128, 128)]


def timed(fn, reps=30, warmup=8):
    """Median seconds of fn() measured with cuda events."""
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
        ts.append(a.elapsed_time(b) / 1e3)
    return sorted(ts)[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-copies", type=int, default=1024)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    torch.set_float32_matmul_precision("high")
    C = args.n_copies
    dev = torch.device("cuda")
    rows = []

    def record(name, seconds, byte_count, flops=0.0):
        """Store one measurement together with the bandwidth and arithmetic rate it reached."""
        rows.append({"name": name, "seconds": seconds, "bytes": byte_count, "flops": flops,
                     "gb_per_s": byte_count / seconds / 1e9,
                     "tflop_per_s": flops / seconds / 1e12 if flops else 0.0})
        print(f"  {seconds*1e6:9.1f} us  {byte_count/1e9:7.3f} GB  "
              f"{byte_count/seconds/1e9:7.0f} GB/s  "
              f"{flops/seconds/1e12 if flops else 0:6.1f} TFLOP/s  {name}")

    print(f"\n== reference: what a streaming kernel gets at {C} copies ==")
    n = 256 * 1024 * 1024                      # 1 GiB of single-precision numbers
    x = torch.randn(n, device=dev)
    y = torch.empty_like(x)
    record("copy one gibibyte (read 1, write 1)", timed(lambda: y.copy_(x)), 2 * n * F32)
    z = torch.randn(n, device=dev)
    record("y = x * 2 + z (read 2, write 1)", timed(lambda: torch.add(x * 2, z, out=y)),
           3 * n * F32)
    del x, y, z
    torch.cuda.empty_cache()

    print(f"\n== one forward layer of the update, both ways, {C} copies ==")
    for name, m, k, nout in LAYERS:
        a = torch.randn(C, m, k, device=dev)
        w = torch.randn(C, k, nout, device=dev)
        b = torch.randn(C, nout, device=dev)
        out = torch.empty(C, m, nout, device=dev)
        flops = 2.0 * C * m * k * nout
        # baddbmm: torch expands the bias to the output shape, writes it into the result and
        # asks the multiply to accumulate on top of it, so the output tensor is touched twice
        # before the activation reads it again
        record(f"baddbmm, {name}", timed(lambda: torch.baddbmm(b.unsqueeze(1), a, w, out=out)),
               F32 * C * (m * k + k * nout + 3 * m * nout), flops)
        record(f"bmm alone, {name}", timed(lambda: torch.bmm(a, w, out=out)),
               F32 * C * (m * k + k * nout + m * nout), flops)
        h = torch.empty_like(out)
        record(f"bias and tanh pass, {name}",
               timed(lambda: torch.tanh(out + b.unsqueeze(1), out=h)),
               F32 * C * 2 * m * nout)
        del a, w, b, out, h
        torch.cuda.empty_cache()

    print(f"\n== the optimiser over the flat parameter buffer, {C} copies ==")
    P = 59910                                   # trainable parameters per copy
    flat = torch.randn(C, P, device=dev)
    grad = torch.randn(C, P, device=dev)
    mm = torch.zeros(C, P, device=dev)
    vv = torch.zeros(C, P, device=dev)
    lr_col = torch.full((C, 1), 3e-4, device=dev)

    def norm_only():
        """The per-copy gradient norm: one reduction over the whole gradient buffer."""
        return (0.5 / (grad.square().sum(1, keepdim=True).sqrt() + 1e-6)).clamp(max=1.0)

    scale = norm_only()

    def adam_chain():
        """Clip and one Adam step, written the way the trainer writes it."""
        gs = grad * scale
        m_new = mm * 0.9 + gs * 0.1
        v_new = vv * 0.999 + gs * gs * 0.001
        step = m_new * lr_col / (v_new.sqrt() + 1e-5)
        mm.copy_(m_new)
        vv.copy_(v_new)
        flat.sub_(step)
        grad.zero_()

    compiled_chain = torch.compile(adam_chain, dynamic=False)
    record("gradient-norm reduction (read 1)", timed(norm_only), F32 * C * P)
    record("clip and Adam, eager", timed(adam_chain), F32 * C * P * 8)
    record("clip and Adam, compiled", timed(compiled_chain), F32 * C * P * 8)
    fresh = torch.randn(C, P, device=dev)
    record("gradient accumulation (read 2, write 1)",
           timed(lambda: grad.add_(fresh)), F32 * C * P * 3)
    del flat, grad, mm, vv, fresh
    torch.cuda.empty_cache()

    print(f"\n== the style-B epoch gather, {C} copies ==")
    B = 512
    tf_field = torch.randn(C, B, 128, device=dev)
    dst = torch.empty_like(tf_field)
    idx = torch.rand(C, B, device=dev).argsort(dim=-1)
    ix = idx.unsqueeze(-1).expand(C, B, 128)
    record("gather the cached target features (read 1, write 1)",
           timed(lambda: torch.gather(tf_field, 1, ix, out=dst)), F32 * C * B * 128 * 2)
    record("draw one epoch's permutation (argsort of 512 keys per copy)",
           timed(lambda: torch.rand(C, B, device=dev).argsort(dim=-1)), F32 * C * B * 4)

    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / f"{time.strftime('%Y-%m-%d-%H-%M-%S')}_probe_update_ops_C{C}{args.tag}.json"
    p.write_text(json.dumps({
        "n_copies": C, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "git": subprocess.run(["git", "-C", str(BASE), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip(),
        "rows": rows}, indent=1))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
