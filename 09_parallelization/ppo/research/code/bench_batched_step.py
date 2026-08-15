"""Microbenchmark 2: a full batched train step (forward + backward + Adam) for C copies.

Compares:
  bmm   - weights [C, in, out], torch.baddbmm per layer, Adam over the stacked tensors
  vmap  - torch.func.stack_module_state + functional_call + vmap, same Adam

each in three execution modes: eager, torch.compile(default), torch.compile(reduce-overhead)
(the last one wires in CUDA graphs).

Run under the H100 lock.
"""

import argparse
import json
import time

import torch
import torch.nn as nn
from torch.func import functional_call, stack_module_state, vmap

OBS_DIM = 4
OUT_DIM = 5


def make_mlp(hidden, device, dtype):
    return nn.Sequential(
        nn.Linear(OBS_DIM, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
        nn.Linear(hidden, OUT_DIM),
    ).to(device=device, dtype=dtype)


def useful_flops(copies, batch, hidden):
    f = 2 * batch * (OBS_DIM * hidden + hidden * hidden + hidden * OUT_DIM)
    return 3 * f * copies  # fwd + bwd


def timeit(fn, iters=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def build_bmm(copies, hidden, device, dtype):
    dims = [(OBS_DIM, hidden), (hidden, hidden), (hidden, OUT_DIM)]
    params = []
    for i, o in dims:
        w = (torch.randn(copies, i, o, device=device, dtype=dtype) * (i**-0.5)).requires_grad_(True)
        b = torch.zeros(copies, 1, o, device=device, dtype=dtype, requires_grad=True)
        params += [w, b]

    def fwd(x, ps):
        h = torch.baddbmm(ps[1], x, ps[0]).tanh()
        h = torch.baddbmm(ps[3], h, ps[2]).tanh()
        return torch.baddbmm(ps[5], h, ps[4])

    return fwd, params


def build_vmap(copies, hidden, device, dtype):
    mlps = [make_mlp(hidden, device, dtype) for _ in range(copies)]
    stacked, buffers = stack_module_state(mlps)
    base = make_mlp(hidden, "meta", dtype)
    plist = []
    for v in stacked.values():
        v.requires_grad_(True)
        plist.append(v)

    def fmodel(p, b, xi):
        return functional_call(base, (p, b), (xi,))

    vf = vmap(fmodel)

    def fwd(x, ps):
        pd = dict(zip(stacked.keys(), ps))
        return vf(pd, buffers, x)

    return fwd, plist


def run_case(kind, mode, copies, hidden, batch, device, dtype):
    if kind == "bmm":
        fwd, params = build_bmm(copies, hidden, device, dtype)
    else:
        fwd, params = build_vmap(copies, hidden, device, dtype)
    opt = torch.optim.Adam(params, lr=3e-4, foreach=True, capturable=(mode == "reduce-overhead"))
    x = torch.randn(copies, batch, OBS_DIM, device=device, dtype=dtype)

    def step(x):
        y = fwd(x, params)
        loss = (y**2).mean(dim=(1, 2)).sum()  # per-copy loss, summed -> independent grads
        grads = torch.autograd.grad(loss, params)
        for p, g in zip(params, grads):
            p.grad = g
        opt.step()
        opt.zero_grad(set_to_none=False)
        return loss

    if mode == "eager":
        f = step
    elif mode == "compile":
        f = torch.compile(step, dynamic=False)
    else:
        f = torch.compile(step, mode="reduce-overhead", dynamic=False)

    t = timeit(lambda: f(x))
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/p/rlprojects/RND/09_parallelization/benchmarks/results/batched_step_modes.json")
    args = ap.parse_args()
    device, dtype = "cuda", torch.float32
    torch.backends.cuda.matmul.allow_tf32 = True

    prop = torch.cuda.get_device_properties(0)
    out = {"device": prop.name, "sms": prop.multi_processor_count, "torch": torch.__version__, "rows": []}

    configs = [(128, 256, 256), (128, 256, 2048), (128, 64, 2048), (32, 256, 2048), (8, 256, 256)]
    for copies, hidden, batch in configs:
        row = {"copies": copies, "hidden": hidden, "batch": batch}
        for kind in ("bmm", "vmap"):
            for mode in ("eager", "compile", "reduce-overhead"):
                key = f"{kind}_{mode}"
                try:
                    t = run_case(kind, mode, copies, hidden, batch, device, dtype)
                    row[key + "_us"] = round(t * 1e6, 1)
                    row[key + "_tflops"] = round(useful_flops(copies, batch, hidden) / t / 1e12, 1)
                except Exception as e:
                    row[key + "_us"] = f"err:{type(e).__name__}:{str(e)[:120]}"
                torch.cuda.empty_cache()
                torch._dynamo.reset()
        out["rows"].append(row)
        print(json.dumps(row), flush=True)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
