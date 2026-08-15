"""Microbenchmark: how to hold n_copies independent small MLPs on one H100.

Compares four ways of evaluating C independent MLPs, each on its own batch of M rows:
  loop      - a python for-loop over copies, one nn.Linear stack per copy
  bmm       - weights stored [C, in, out], torch.baddbmm per layer
  vmap      - torch.func.stack_module_state + functional_call + vmap
  blockdiag - one dense [C*in, C*out] block-diagonal weight, single mm

Reports median wall time per call, achieved useful TFLOP/s (block-diagonal counted on the
USEFUL flops only, so its waste shows up as low throughput), and peak memory.

Run under the H100 lock:  bash locks/gpu_run.sh "<venv>/bin/python <this file>"
"""

import argparse
import json
import time

import torch
import torch.nn as nn
from torch.func import functional_call, stack_module_state, vmap

OBS_DIM = 4
OUT_DIM = 5  # 2 action means + 2 value heads + 1 rnd scalar stand-in


def make_mlp(hidden: int, device, dtype):
    return nn.Sequential(
        nn.Linear(OBS_DIM, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
        nn.Linear(hidden, OUT_DIM),
    ).to(device=device, dtype=dtype)


def useful_flops(copies, batch, hidden, backward):
    # 2*M*K*N per layer, x2 for the multiply-add already counted
    f = 2 * batch * (OBS_DIM * hidden + hidden * hidden + hidden * OUT_DIM)
    f *= copies
    return f * 3 if backward else f


def timeit(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


def bench_loop(copies, batch, hidden, device, dtype, backward):
    mlps = [make_mlp(hidden, device, dtype) for _ in range(copies)]
    x = torch.randn(copies, batch, OBS_DIM, device=device, dtype=dtype)

    def run():
        out = 0.0
        for c in range(copies):
            y = mlps[c](x[c])
            if backward:
                y.sum().backward()
            else:
                out = out + y.sum()
        return out

    return timeit(run, iters=10 if copies >= 64 else 30)


def bench_bmm(copies, batch, hidden, device, dtype, backward, compiled=False):
    dims = [(OBS_DIM, hidden), (hidden, hidden), (hidden, OUT_DIM)]
    # scale FIRST, then mark as a leaf: `randn(requires_grad=True) * c` is a non-leaf whose
    # graph is retained, and the second backward then fails with "backward a second time".
    ws = [
        (torch.randn(copies, i, o, device=device, dtype=dtype) * (i**-0.5)).requires_grad_(backward)
        for i, o in dims
    ]
    bs = [torch.zeros(copies, 1, o, device=device, dtype=dtype, requires_grad=backward) for _, o in dims]
    x = torch.randn(copies, batch, OBS_DIM, device=device, dtype=dtype)

    def fwd(x):
        h = torch.baddbmm(bs[0], x, ws[0]).tanh()
        h = torch.baddbmm(bs[1], h, ws[1]).tanh()
        return torch.baddbmm(bs[2], h, ws[2])

    f = torch.compile(fwd, mode="max-autotune-no-cudagraphs") if compiled else fwd

    def run():
        y = f(x)
        if backward:
            y.sum().backward()
        return y

    return timeit(run)


def bench_vmap(copies, batch, hidden, device, dtype, backward):
    mlps = [make_mlp(hidden, device, dtype) for _ in range(copies)]
    params, buffers = stack_module_state(mlps)
    base = make_mlp(hidden, "meta", dtype)
    x = torch.randn(copies, batch, OBS_DIM, device=device, dtype=dtype)
    if backward:
        for p in params.values():
            p.requires_grad_(True)

    def fmodel(p, b, xi):
        return functional_call(base, (p, b), (xi,))

    vf = vmap(fmodel)

    def run():
        y = vf(params, buffers, x)
        if backward:
            y.sum().backward()
        return y

    return timeit(run)


def bench_blockdiag(copies, batch, hidden, device, dtype, backward):
    dims = [(OBS_DIM, hidden), (hidden, hidden), (hidden, OUT_DIM)]
    bytes_needed = sum((copies * i) * (copies * o) for i, o in dims) * (2 if dtype == torch.bfloat16 else 4)
    if bytes_needed > 20e9:
        return None, bytes_needed
    ws = []
    for i, o in dims:
        w = torch.zeros(copies * i, copies * o, device=device, dtype=dtype, requires_grad=backward)
        with torch.no_grad():
            for c in range(copies):
                w[c * i : (c + 1) * i, c * o : (c + 1) * o] = torch.randn(i, o, device=device, dtype=dtype) * (i**-0.5)
        ws.append(w)
    x = torch.randn(batch, copies * OBS_DIM, device=device, dtype=dtype)

    def run():
        h = (x @ ws[0]).tanh()
        h = (h @ ws[1]).tanh()
        y = h @ ws[2]
        if backward:
            y.sum().backward()
        return y

    return timeit(run, iters=20), bytes_needed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/p/rlprojects/RND/09_parallelization/benchmarks/results/batched_mlp_layout.json")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "tf32", "bf16"])
    args = ap.parse_args()

    device = "cuda"
    if args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
        torch.backends.cuda.matmul.allow_tf32 = args.dtype == "tf32"
        torch.backends.cudnn.allow_tf32 = args.dtype == "tf32"

    prop = torch.cuda.get_device_properties(0)
    results = {
        "device": prop.name,
        "sms": prop.multi_processor_count,
        "torch": torch.__version__,
        "dtype": args.dtype,
        "rows": [],
    }

    grid = []
    for copies in (8, 32, 128):
        for hidden in (64, 256):
            for batch in (64, 256, 2048):
                grid.append((copies, hidden, batch))

    for copies, hidden, batch in grid:
        row = {"copies": copies, "hidden": hidden, "batch": batch}
        for backward in (False, True):
            tag = "bwd" if backward else "fwd"
            flops = useful_flops(copies, batch, hidden, backward)
            try:
                t = bench_bmm(copies, batch, hidden, device, dtype, backward)
                row[f"bmm_{tag}_us"] = round(t * 1e6, 1)
                row[f"bmm_{tag}_tflops"] = round(flops / t / 1e12, 1)
            except RuntimeError as e:
                row[f"bmm_{tag}_us"] = f"err:{e}"
            torch.cuda.empty_cache()
            try:
                t = bench_vmap(copies, batch, hidden, device, dtype, backward)
                row[f"vmap_{tag}_us"] = round(t * 1e6, 1)
                row[f"vmap_{tag}_tflops"] = round(flops / t / 1e12, 1)
            except Exception as e:
                row[f"vmap_{tag}_us"] = f"err:{type(e).__name__}"
            torch.cuda.empty_cache()
            try:
                res, nbytes = bench_blockdiag(copies, batch, hidden, device, dtype, backward)
                row[f"blockdiag_{tag}_weight_gb"] = round(nbytes / 1e9, 2)
                if res is None:
                    row[f"blockdiag_{tag}_us"] = "skipped_oom"
                else:
                    row[f"blockdiag_{tag}_us"] = round(res * 1e6, 1)
                    row[f"blockdiag_{tag}_tflops"] = round(flops / res / 1e12, 1)
            except RuntimeError as e:
                row[f"blockdiag_{tag}_us"] = f"err:{type(e).__name__}"
            torch.cuda.empty_cache()
        # loop baseline: forward only, and only where it will not take forever
        if batch == 256:
            try:
                t = bench_loop(copies, batch, hidden, device, dtype, False)
                row["loop_fwd_us"] = round(t * 1e6, 1)
            except RuntimeError as e:
                row["loop_fwd_us"] = f"err:{type(e).__name__}"
            torch.cuda.empty_cache()
        results["rows"].append(row)
        print(json.dumps(row), flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
