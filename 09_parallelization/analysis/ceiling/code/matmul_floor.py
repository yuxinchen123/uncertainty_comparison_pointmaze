"""Sum the isolated best-case time of every matrix multiply in one training iteration.

For each layer of each network, the forward pass is one batched matrix multiply and the
backward pass is two (one for the gradient with respect to the input, one for the gradient
with respect to the weights). Each is timed on its own, many repeats inside one timed region
so that synchronisation cost is not counted. The sum is a lower bound on the iteration time
for this algorithm as written: it is what the iteration would cost if the matrix multiplies
were the only work and nothing else on the device took any time at all.
"""

import json
import time

import torch

torch.backends.cuda.matmul.allow_tf32 = True
dev = torch.device("cuda")
C = 128

cache = {}


def bmm_time(c, m, k, n, repeats=200):
    """Median seconds for one bmm of shape (c, m, k) x (c, k, n)."""
    key = (c, m, k, n)
    if key in cache:
        return cache[key]
    a = torch.randn(c, m, k, device=dev, dtype=torch.float32)
    b = torch.randn(c, k, n, device=dev, dtype=torch.float32)
    out = torch.empty(c, m, n, device=dev, dtype=torch.float32)
    for _ in range(10):
        torch.bmm(a, b, out=out)
    torch.cuda.synchronize()
    samples = []
    for _ in range(7):
        t0 = time.perf_counter()
        for _ in range(repeats):
            torch.bmm(a, b, out=out)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) / repeats)
    samples.sort()
    cache[key] = samples[len(samples) // 2]
    del a, b, out
    return cache[key]


def layer_cost(m, k, n, backward):
    """Forward = (m,k)x(k,n). Backward adds dgrad (m,n)x(n,k) and wgrad (k,m)x(m,n)."""
    fwd = bmm_time(C, m, k, n)
    if not backward:
        return fwd, 2.0 * C * m * k * n
    dgrad = bmm_time(C, m, n, k)
    wgrad = bmm_time(C, k, m, n)
    return fwd + dgrad + wgrad, 6.0 * C * m * k * n


# layer widths, from ppo/research/ppo_rnd_algorithm_spec.md section 3
ACTOR = [(4, 64), (64, 64), (64, 2)]
CRITIC = [(4, 64), (64, 64), (64, 2)]  # two 1-wide heads, packed into one width-2 matmul
TARGET = [(4, 256), (256, 128)]
PREDICTOR = [(4, 256), (256, 128), (128, 128)]

result = {"C": C, "note": "seconds, TF32 on, H100 NVL"}

# ---- rollout: T = 128 sequential steps, N = 4 rows per copy per step; actor only
T, N = 128, 4
roll_t = roll_f = 0.0
for (k, n) in ACTOR:
    t, f = layer_cost(N, k, n, backward=False)
    roll_t += T * t
    roll_f += T * f
result["rollout_actor"] = {"seconds": roll_t, "flops": roll_f}

# ---- post-rollout wide passes, read off _post_body_impl in ppo/torch_ppo/torch_ppo_rnd.py:
#   critic once on the concatenation of observations and next observations, 2B rows
#   RND target and predictor on B rows, for the curiosity bonus, under the old statistics
#   RND target a second time on B rows, under the new statistics, cached for the update
# The log-probability needs no matrix multiply: it is a function of the stored action noise.
B = 512
post_t = post_f = 0.0
for rows, group in ((2 * B, CRITIC), (B, TARGET), (B, PREDICTOR), (B, TARGET)):
    for (k, n) in group:
        t, f = layer_cost(rows, k, n, backward=False)
        post_t += t
        post_f += f
result["post_wide_passes"] = {"seconds": post_t, "flops": post_f}

# ---- update, style B: 16 steps of 128 rows; actor, critic, predictor forward and backward
upd_t = upd_f = 0.0
for group in (ACTOR, CRITIC, PREDICTOR):
    for (k, n) in group:
        t, f = layer_cost(128, k, n, backward=True)
        upd_t += 16 * t
        upd_f += 16 * f
result["update_styleB"] = {"seconds": upd_t, "flops": upd_f}

# ---- update, style A: 1 step of 512 rows
updA_t = updA_f = 0.0
for group in (ACTOR, CRITIC, PREDICTOR):
    for (k, n) in group:
        t, f = layer_cost(512, k, n, backward=True)
        updA_t += t
        updA_f += f
result["update_styleA"] = {"seconds": updA_t, "flops": updA_f}

result["iteration_styleB"] = {
    "seconds": roll_t + post_t + upd_t,
    "flops": roll_f + post_f + upd_f,
}
result["iteration_styleA"] = {
    "seconds": roll_t + post_t + updA_t,
    "flops": roll_f + post_f + updA_f,
}

# ---- per-kernel floor inside a replayed graph
def graph_kernel_floor(n_kernels):
    x = torch.zeros(1024, device=dev)
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            for _ in range(n_kernels):
                x.add_(1.0)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        for _ in range(n_kernels):
            x.add_(1.0)
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(9):
        t0 = time.perf_counter()
        for _ in range(20):
            g.replay()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) / 20)
    samples.sort()
    t = samples[len(samples) // 2]
    return {"kernels": n_kernels, "seconds": t, "seconds_per_kernel": t / n_kernels}


result["graph_kernel_floor"] = [graph_kernel_floor(k) for k in (64, 512, 2048)]

print(json.dumps(result, indent=1))
