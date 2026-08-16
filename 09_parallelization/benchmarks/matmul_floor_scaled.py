"""The isolated matrix-multiply floor of one training iteration, at any copy count.

This is the same instrument as `analysis/ceiling/code/matmul_floor.py`, which was written for
128 copies only, generalised to the copy counts the trainer is actually used at. For every
layer of every network it times the batched matrix multiplication on its own — forward, and
for the trained layers the two backward multiplications as well — and sums them. The sum is a
lower bound on the iteration time for this algorithm as written: what the iteration would cost
if the matrix multiplications were the only work on the device.

Usage (through the H100 lock wrapper):
  python matmul_floor_scaled.py --n-copies 1024 4096
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).resolve().parent / "results"

# layer widths (input, output) from ppo/research/ppo_rnd_algorithm_spec.md section 3
ACTOR = [(4, 64), (64, 64), (64, 2)]
CRITIC = [(4, 64), (64, 64), (64, 2)]   # two 1-wide heads, packed into one width-2 multiply
TARGET = [(4, 256), (256, 128)]
PREDICTOR = [(4, 256), (256, 128), (128, 128)]


def make_timer(dev):
    """Return a memoised timer for one batched matrix multiplication shape."""
    cache = {}

    def bmm_time(c, m, k, n, repeats=50):
        """Median seconds for one bmm of shape (c, m, k) x (c, k, n)."""
        key = (c, m, k, n)
        if key in cache:
            return cache[key]
        a = torch.randn(c, m, k, device=dev)
        b = torch.randn(c, k, n, device=dev)
        out = torch.empty(c, m, n, device=dev)
        for _ in range(5):
            torch.bmm(a, b, out=out)
        torch.cuda.synchronize()
        samples = []
        for _ in range(5):
            t0 = time.perf_counter()
            for _ in range(repeats):
                torch.bmm(a, b, out=out)
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) / repeats)
        samples.sort()
        cache[key] = samples[len(samples) // 2]
        del a, b, out
        torch.cuda.empty_cache()
        return cache[key]

    return bmm_time


def layer_cost(bmm_time, C, m, k, n, backward):
    """(seconds, flops) of one layer: forward, plus input- and weight-gradient if trained."""
    fwd = bmm_time(C, m, k, n)
    if not backward:
        return fwd, 2.0 * C * m * k * n
    # backward is two more multiplications: (m,n)x(n,k) for the input gradient and
    # (k,m)x(m,n) for the weight gradient, so three multiplications and 6*m*k*n flops
    return fwd + bmm_time(C, m, n, k) + bmm_time(C, k, m, n), 6.0 * C * m * k * n


def floor_for(C, dev):
    """Isolated matrix-multiply floor of one iteration at C copies, both update styles."""
    bmm_time = make_timer(dev)
    T, N, B, MB = 128, 4, 512, 128
    res = {"n_copies": C}

    # rollout: T sequential steps, N rows per copy per step, actor only (the critic, the
    # log-probability and the RND bonus were hoisted out of the loop in round two)
    roll_t = roll_f = 0.0
    for (k, n) in ACTOR:
        t, f = layer_cost(bmm_time, C, N, k, n, backward=False)
        roll_t += T * t
        roll_f += T * f
    res["rollout"] = {"seconds": roll_t, "flops": roll_f}

    # post-rollout wide passes: critic on 2B rows (values and bootstrap values in one call),
    # RND target and predictor on B rows for the bonus, and the target once more on B rows
    # under the updated statistics, cached for the update stage
    post_t = post_f = 0.0
    for rows, group in ((2 * B, CRITIC), (B, TARGET), (B, PREDICTOR), (B, TARGET)):
        for (k, n) in group:
            t, f = layer_cost(bmm_time, C, rows, k, n, backward=False)
            post_t += t
            post_f += f
    res["post"] = {"seconds": post_t, "flops": post_f}

    # update: style B is 16 steps of MB rows, style A one step of B rows; the frozen target is
    # not recomputed here because its features were cached in the post-rollout stage
    for name, rows, steps in (("update_styleB", MB, 16), ("update_styleA", B, 1)):
        ut = uf = 0.0
        for group in (ACTOR, CRITIC, PREDICTOR):
            for (k, n) in group:
                t, f = layer_cost(bmm_time, C, rows, k, n, backward=True)
                ut += steps * t
                uf += steps * f
        res[name] = {"seconds": ut, "flops": uf}

    for style, key in (("styleA", "update_styleA"), ("styleB", "update_styleB")):
        res[f"iteration_{style}"] = {
            "seconds": roll_t + post_t + res[key]["seconds"],
            "flops": roll_f + post_f + res[key]["flops"]}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-copies", type=int, nargs="+", default=[1024, 2048, 4096])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    torch.set_float32_matmul_precision("high")     # TF32, as the trainer runs
    dev = torch.device("cuda")

    rows = []
    for C in args.n_copies:
        r = floor_for(C, dev)
        rows.append(r)
        for style in ("styleA", "styleB"):
            it = r[f"iteration_{style}"]
            print(f"C={C:>5d} {style}: matrix-multiply floor {it['seconds']*1e3:8.2f} ms  "
                  f"({it['flops']/1e9:8.1f} GFLOP, "
                  f"{it['flops']/it['seconds']/1e12:6.2f} TFLOP/s while multiplying)")
        print(f"        rollout {r['rollout']['seconds']*1e3:7.2f} ms  "
              f"post {r['post']['seconds']*1e3:7.2f} ms  "
              f"updateA {r['update_styleA']['seconds']*1e3:7.2f} ms  "
              f"updateB {r['update_styleB']['seconds']*1e3:7.2f} ms")

    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / f"{time.strftime('%Y-%m-%d-%H-%M-%S')}_matmul_floor_scaled{args.tag}.json"
    p.write_text(json.dumps({
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "git": subprocess.run(["git", "-C", str(BASE), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip(),
        "rows": rows}, indent=1))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
