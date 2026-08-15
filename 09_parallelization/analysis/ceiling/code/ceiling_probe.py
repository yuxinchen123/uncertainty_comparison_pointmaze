"""Measure the achievable ceilings of this H100 NVL: matmul throughput and memory bandwidth.

Reports, for each measurement, the raw time and the derived rate, so the arithmetic is checkable.
"""

import json
import time

import torch

dev = torch.device("cuda")
props = torch.cuda.get_device_properties(0)
out = {
    "gpu": props.name,
    "sm_count": props.multi_processor_count,
    "l2_bytes": props.L2_cache_size,
    "clock_khz": props.clock_rate,
    "total_memory_bytes": props.total_memory,
    "torch": torch.__version__,
}


def timeit(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


# ---------------------------------------------------------------- big square matmul
def square_gemm(n, tf32):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    a = torch.randn(n, n, device=dev, dtype=torch.float32)
    b = torch.randn(n, n, device=dev, dtype=torch.float32)
    t = timeit(lambda: torch.mm(a, b))
    flops = 2.0 * n * n * n
    return {"n": n, "tf32": tf32, "seconds": t, "flops": flops, "flops_per_s": flops / t}


out["square_gemm"] = []
for n in (4096, 8192, 16384):
    for tf32 in (True, False):
        out["square_gemm"].append(square_gemm(n, tf32))
        torch.cuda.empty_cache()

# ------------------------------------------------- batched GEMM at the trainer's real shapes
# One copy's update-phase layer-2 matmul: (C, M, 64) x (C, 64, 64)
torch.backends.cuda.matmul.allow_tf32 = True


def batched_gemm(c, m, k, n):
    a = torch.randn(c, m, k, device=dev, dtype=torch.float32)
    b = torch.randn(c, k, n, device=dev, dtype=torch.float32)
    t = timeit(lambda: torch.bmm(a, b))
    flops = 2.0 * c * m * k * n
    return {"C": c, "M": m, "K": k, "N": n, "seconds": t, "flops": flops, "flops_per_s": flops / t}


out["batched_gemm"] = [
    batched_gemm(128, 4, 4, 64),  # rollout step: actor layer 1, N=4 rows per copy
    batched_gemm(128, 4, 64, 64),  # rollout step: actor layer 2
    batched_gemm(128, 128, 64, 64),  # update minibatch: trunk layer 2
    batched_gemm(128, 512, 64, 64),  # full-batch pass: trunk layer 2
    batched_gemm(128, 512, 256, 128),  # RND target/predictor layer 2, whole rollout
    batched_gemm(128, 2048, 256, 128),  # the same widened 4x
]

# ---------------------------------------------------------------- memory bandwidth
def copy_bandwidth(n_bytes):
    n = n_bytes // 4
    a = torch.empty(n, device=dev, dtype=torch.float32).normal_()
    b = torch.empty_like(a)
    t = timeit(lambda: b.copy_(a))
    moved = 2.0 * n_bytes  # one read + one write
    return {"bytes_each": n_bytes, "seconds": t, "bytes_moved": moved, "bytes_per_s": moved / t}


out["copy_bandwidth"] = [copy_bandwidth(b) for b in (1 << 28, 1 << 30, 4 << 30)]


def add_bandwidth(n_bytes):
    n = n_bytes // 4
    a = torch.empty(n, device=dev, dtype=torch.float32).normal_()
    b = torch.empty(n, device=dev, dtype=torch.float32).normal_()
    c = torch.empty_like(a)
    t = timeit(lambda: torch.add(a, b, out=c))
    moved = 3.0 * n_bytes  # two reads + one write
    return {"bytes_each": n_bytes, "seconds": t, "bytes_moved": moved, "bytes_per_s": moved / t}


out["add_bandwidth"] = [add_bandwidth(b) for b in (1 << 28, 1 << 30)]

# ---------------------------------------------------------------- empty-kernel launch cost
small = torch.zeros(1, device=dev)


def launch_cost(k):
    def body():
        for _ in range(k):
            small.add_(1.0)

    t = timeit(body, warmup=3, iters=10)
    return {"launches": k, "seconds": t, "seconds_per_launch": t / k}


out["launch_cost"] = [launch_cost(k) for k in (100, 1000, 10000)]

print(json.dumps(out, indent=1))
