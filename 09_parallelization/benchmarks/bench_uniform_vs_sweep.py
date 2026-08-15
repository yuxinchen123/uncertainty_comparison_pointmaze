"""What does training at DIFFERENT learning rates cost, against one rate for everything?

The obvious comparison — a uniform run against a swept run — mixes two separate changes:

  1. the learning rate becomes a per-copy VECTOR, which means a different optimizer: torch's
     fused Adam takes one scalar rate per parameter group, so a sweep uses a hand-written
     batched Adam instead;
  2. the values in that vector actually DIFFER, which changes no code at all — only the
     numbers the same kernels read.

So this measures three arms at each copy count, matched in total copies:

  uniform     one rate for every copy, torch's fused Adam (the default path)
  same-rates  the sweep path, but every group given the SAME rate
  diff-rates  the sweep path with genuinely different rates

uniform against same-rates prices the optimizer change. same-rates against diff-rates prices
the rates differing, and should be zero: identical kernels, identical shapes, different
constants. Each pair is measured in ABBA order in separate processes, so the within-arm spread
gives a noise floor and a difference below it is reported as no effect.

Usage (through the H100 lock wrapper):
  python bench_uniform_vs_sweep.py --copies 128 256 512 1024 2048 4096 --rates 16 --iters 20
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "ppo" / "torch_ppo"))
RESULTS = Path(__file__).resolve().parent / "results"
ARMS = ("uniform", "same-rates", "diff-rates")


def rates_for(n, same):
    """The rate list for one arm: all equal, or spread log-uniformly over a realistic range."""
    if same:
        return [3e-4] * n
    lo, hi = 1e-5, 1e-2
    return [lo * (hi / lo) ** (i / (n - 1)) for i in range(n)] if n > 1 else [3e-4]


def measure(arm, total_copies, n_rates, iters, warmup, style):
    """Time one arm at one copy count; prints a RESULT line."""
    import torch
    from torch_ppo_rnd import PPORND, production_config, sweep_config
    if arm == "uniform":
        cfg = production_config(total_copies, style=style)
    else:
        cfg = sweep_config(rates_for(n_rates, arm == "same-rates"),
                           total_copies // n_rates, style=style)
    t = PPORND(cfg, device="cuda")
    t.prime_obs_rms()
    t._build_iteration_graph()
    for _ in range(warmup):
        t.iteration_captured()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        t.iteration_captured()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    print("RESULT " + json.dumps({
        "arm": arm, "total_copies": cfg.n_copies, "n_rates": n_rates,
        "sec_per_iteration": times[len(times) // 2],
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 2 ** 20}))


def run_child(arm, c, n_rates, iters, warmup, style):
    """One arm in its own process; returns its parsed result or None."""
    cmd = [sys.executable, __file__, "--child", arm, "--copies", str(c), "--rates",
           str(n_rates), "--iters", str(iters), "--warmup", str(warmup), "--style", style]
    out = subprocess.run(cmd, capture_output=True, text=True)
    line = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
    if not line:
        print(f"  {arm} at {c} copies FAILED: {(out.stdout + out.stderr)[-300:]}")
        return None
    return json.loads(line[0][len("RESULT "):])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copies", type=int, nargs="+", default=[128, 256, 512, 1024, 2048, 4096])
    ap.add_argument("--rates", type=int, default=16)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--style", default="epoch_minibatch")
    ap.add_argument("--child")
    args = ap.parse_args()

    if args.child:
        measure(args.child, args.copies[0], args.rates, args.iters, args.warmup, args.style)
        return

    rows = []
    print(f"{'copies':>7} {'uniform ms':>11} {'same-rates':>11} {'diff-rates':>11} "
          f"{'noise':>7} {'vector cost':>12} {'differing cost':>15}")
    for c in args.copies:
        # ABBA over the three arms so drift cannot favour one of them
        order = list(ARMS) + list(reversed(ARMS))
        got = {a: [] for a in ARMS}
        for arm in order:
            r = run_child(arm, c, args.rates, args.iters, args.warmup, args.style)
            if r:
                got[arm].append(r)
        if not all(got[a] for a in ARMS):
            continue
        med = {a: sorted(x["sec_per_iteration"] for x in got[a])[len(got[a]) // 2] for a in ARMS}
        noise = max(max(x["sec_per_iteration"] for x in got[a])
                    - min(x["sec_per_iteration"] for x in got[a]) for a in ARMS)
        row = {"total_copies": c, "n_rates": args.rates,
               "uniform_sec": med["uniform"], "same_rates_sec": med["same-rates"],
               "diff_rates_sec": med["diff-rates"], "noise_floor_sec": noise,
               "vector_cost_percent": (med["same-rates"] / med["uniform"] - 1) * 100,
               "differing_cost_percent": (med["diff-rates"] / med["same-rates"] - 1) * 100,
               "total_cost_percent": (med["diff-rates"] / med["uniform"] - 1) * 100,
               "peak_vram_mb": {a: got[a][0]["peak_vram_mb"] for a in ARMS}}
        rows.append(row)
        print(f"{c:>7} {med['uniform']*1e3:>11.2f} {med['same-rates']*1e3:>11.2f} "
              f"{med['diff-rates']*1e3:>11.2f} {noise*1e3:>7.2f} "
              f"{row['vector_cost_percent']:>11.1f}% {row['differing_cost_percent']:>14.1f}%")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{time.strftime('%Y-%m-%d-%H-%M-%S')}_uniform_vs_sweep.json"
    out.write_text(json.dumps({"style": args.style, "n_rates": args.rates, "rows": rows},
                              indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
