"""Microbenchmark 3: how step time grows with the number of independent copies.

Two shapes of work, both with weights stored [C, in, out] and torch.baddbmm per layer,
compiled with mode="reduce-overhead" (CUDA graphs):

  rollout - forward only, no grad, per-copy batch = n_envs (the acting step)
  update  - forward + backward + Adam, per-copy batch = minibatch size

Run under the H100 lock.
"""

import argparse
import json
import time

import torch

OBS_DIM = 4
OUT_DIM = 5


def build(copies, hidden, device, dtype, grad):
    dims = [(OBS_DIM, hidden), (hidden, hidden), (hidden, OUT_DIM)]
    ps = []
    for i, o in dims:
        w = (torch.randn(copies, i, o, device=device, dtype=dtype) * (i**-0.5)).requires_grad_(grad)
        b = torch.zeros(copies, 1, o, device=device, dtype=dtype, requires_grad=grad)
        ps += [w, b]
    return ps


def fwd(x, ps):
    h = torch.baddbmm(ps[1], x, ps[0]).tanh()
    h = torch.baddbmm(ps[3], h, ps[2]).tanh()
    return torch.baddbmm(ps[5], h, ps[4])


def timeit(fn, iters=50, warmup=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", default="/p/rlprojects/RND/09_parallelization/benchmarks/results/copy_scaling.json")
    a = ap.parse_args()
    device, dtype = "cuda", torch.float32
    torch.backends.cuda.matmul.allow_tf32 = True
    prop = torch.cuda.get_device_properties(0)
    out = {
        "device": prop.name,
        "sms": prop.multi_processor_count,
        "torch": torch.__version__,
        "hidden": a.hidden,
        "per_copy_batch": a.batch,
        "rows": [],
    }

    for copies in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
        row = {"copies": copies, "total_rows": copies * a.batch}
        # rollout: forward only
        ps = build(copies, a.hidden, device, dtype, grad=False)
        x = torch.randn(copies, a.batch, OBS_DIM, device=device, dtype=dtype)

        @torch.compile(mode="reduce-overhead", dynamic=False)
        def roll(x):
            with torch.no_grad():
                return fwd(x, ps)

        try:
            t = timeit(lambda: roll(x))
            row["rollout_us"] = round(t * 1e6, 1)
            row["rollout_rows_per_s_M"] = round(copies * a.batch / t / 1e6, 1)
        except Exception as e:
            row["rollout_us"] = f"err:{type(e).__name__}"
        torch._dynamo.reset()
        torch.cuda.empty_cache()

        # update: fwd + bwd + adam
        ps2 = build(copies, a.hidden, device, dtype, grad=True)
        opt = torch.optim.Adam(ps2, lr=3e-4, foreach=True, capturable=True)

        @torch.compile(mode="reduce-overhead", dynamic=False)
        def upd(x):
            y = fwd(x, ps2)
            loss = (y**2).mean(dim=(1, 2)).sum()
            gs = torch.autograd.grad(loss, ps2)
            for p, g in zip(ps2, gs):
                p.grad = g
            opt.step()
            opt.zero_grad(set_to_none=False)
            return loss

        try:
            t = timeit(lambda: upd(x), iters=30, warmup=10)
            row["update_us"] = round(t * 1e6, 1)
            row["update_rows_per_s_M"] = round(copies * a.batch / t / 1e6, 1)
        except Exception as e:
            row["update_us"] = f"err:{type(e).__name__}:{str(e)[:80]}"
        torch._dynamo.reset()
        torch.cuda.empty_cache()

        out["rows"].append(row)
        print(json.dumps(row), flush=True)

    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
