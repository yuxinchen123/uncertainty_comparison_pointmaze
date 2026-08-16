"""Where the two gradient forms differ, program by program, on the trainer's real shapes.

The whole-iteration comparison says which form is faster; this says why. It builds one trainer of
each form at the same copy count, produces a real set of gradients through each one's own path,
and times the three programs that differ: the copy into the flat buffer (which only the buffer
form runs), the per-copy gradient limit, and the Adam step.

The reason a decomposition is needed rather than a byte count: the operation probe at 4,096
copies measures a write into a STRIDED window of the flat buffer at 2,022 gigabytes per second
against 3,588 for the same write into a contiguous tensor. The buffer form pays that rate for its
copy; the other form pays it for every parameter, moment and gradient the optimizer touches, if
the compiler's generated kernels lose the same thing eager kernels do. Which of the two costs
more is a measurement, not an argument.

Usage (through the H100 lock wrapper):
  python probe_gradient_form.py --n-copies 4096
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "ppo" / "torch_ppo"))
RESULTS = Path(__file__).resolve().parent / "results"


def cuda_time(fn, reps=30, warmup=8):
    """Median device time of fn() in microseconds."""
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


def build(n_copies, gradient_buffer):
    """A trainer of one gradient form, with one real minibatch and one real set of gradients."""
    from torch_ppo_rnd import PPORND, production_config
    # graph capture off so the three programs can be called one at a time and timed
    t = PPORND(production_config(n_copies, gradient_buffer=gradient_buffer,
                                 one_graph=False, capture_update=False), device="cuda")
    t.prime_obs_rms()
    Brows = t.cfg.num_steps * t.cfg.n_envs
    t._rollout_body()
    t._post_body()
    mb = {k: t._U[k][:, :Brows // t.cfg.num_minibatches] for k in t._U_KEYS}
    raw = list(torch.autograd.grad(t._loss_fn(mb, style_a=False), t.trainable))
    grads = t._backward(t._loss_fn(mb, style_a=False))
    return t, raw, grads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-copies", type=int, default=4096)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    C = args.n_copies

    rows = []

    def record(name, seconds_us, nbytes):
        """One timed program with the bytes it must move and the rate that implies."""
        rows.append({"name": name, "microseconds": seconds_us, "bytes": nbytes,
                     "gb_per_s": nbytes / (seconds_us * 1e-6) / 1e9})
        print(f"  {seconds_us:9.1f} us  {nbytes/1e9:6.3f} GB  {rows[-1]['gb_per_s']:7.0f} GB/s  "
              f"{name}")

    print(f"\n== the programs the two gradient forms differ in, {C} copies ==")
    buf, raw_buf, flat_grad = build(C, True)
    P = buf._flat.shape[1]
    F32 = 4
    one_buffer = F32 * C * P
    record("copy the gradients into the flat buffer (read 1, write 1), buffer form",
           cuda_time(lambda: torch._foreach_copy_(buf.grad_windows, raw_buf)), 2 * one_buffer)
    record("gradient limit over the flat buffer (read 1), buffer form",
           cuda_time(lambda: buf._scale_fn(flat_grad)), one_buffer)
    record("Adam over the flat buffer (read 4, write 3), buffer form",
           cuda_time(lambda: buf._adam_fn(flat_grad)), 7 * one_buffer)
    del buf, raw_buf, flat_grad
    torch.cuda.empty_cache()

    direct, raw_direct, grads = build(C, False)
    # the twenty-one gradients are contiguous tensors, so their bytes are the unpadded count
    grad_bytes = sum(g.numel() * F32 for g in grads)
    record("gradient limit over the twenty-one gradients (read 1), no-buffer form",
           cuda_time(lambda: direct._scale_fn(grads)), grad_bytes)
    record("Adam over the twenty-one windows (read 4, write 3), no-buffer form",
           cuda_time(lambda: direct._adam_fn(grads)), 7 * grad_bytes)

    per_step = {
        "buffer form": rows[0]["microseconds"] + rows[1]["microseconds"] + rows[2]["microseconds"],
        "no-buffer form": rows[3]["microseconds"] + rows[4]["microseconds"],
    }
    print(f"\n  one minibatch step's optimizer work:")
    for k, v in per_step.items():
        print(f"    {v:9.1f} us  {k}")
    print(f"  difference over the sixteen steps of an iteration: "
          f"{(per_step['buffer form'] - per_step['no-buffer form']) * 16 / 1000:+.2f} ms")

    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / f"{time.strftime('%Y-%m-%d-%H-%M-%S')}_probe_gradient_form_C{C}{args.tag}.json"
    p.write_text(json.dumps({
        "n_copies": C, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "git": subprocess.run(["git", "-C", str(BASE), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip(),
        "per_step_us": per_step, "rows": rows}, indent=1))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
