"""How a learning-rate sweep scales: in the number of rates, and in the copies per rate.

Three studies, because "a sweep costs X" depends on which knob is being turned:

  rates      copies per rate fixed, number of rates grows (so the total copy count grows with
             it). This is the shape you see when you decide to try more learning rates.
  copies     number of rates fixed, copies per rate grows (total grows again). This is the
             shape you see when you want more seeds behind each rate.
  groups     TOTAL copies fixed, split into more and more groups. Nothing about the work
             changes — only how many distinct learning rates are in play — so this isolates
             whether having more groups costs anything by itself.

Reported for every point: milliseconds per iteration, total environment steps per second,
per-copy environment steps per second (one copy is one independent training run), and per-env
environment steps per second (one copy holds n_envs environments).

Run it through the project's H100 lock wrapper (locks/gpu_run.sh); the script itself does
not take the lock.

Usage:
  python bench_sweep_scaling.py --study rates  --copies-per-rate 128 --rate-counts 1 2 4 8 16 32
  python bench_sweep_scaling.py --study copies --rate-count 16 --copies-list 8 16 32 64 128 256
  python bench_sweep_scaling.py --study groups --total-copies 2048 --rate-counts 1 2 4 8 16 32 64
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


def rates_for(n):
    """n learning rates spread log-uniformly over the range a sweep would realistically try."""
    lo, hi = 1e-5, 1e-2
    if n == 1:
        return [3e-4]
    return [lo * (hi / lo) ** (i / (n - 1)) for i in range(n)]


def measure(n_rates, copies_per_rate, iters, warmup, style):
    """One configuration, in its own process; prints a RESULT line."""
    import torch
    from torch_ppo_rnd import PPORND, sweep_config
    cfg = sweep_config(rates_for(n_rates), copies_per_rate, style=style)
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
    sec = times[len(times) // 2]
    C, N, T = cfg.n_copies, cfg.n_envs, cfg.num_steps
    print("RESULT " + json.dumps({
        "n_rates": n_rates, "copies_per_rate": copies_per_rate, "total_copies": C,
        "n_envs_per_copy": N, "total_envs": C * N, "style": style,
        "sec_per_iteration": sec,
        "env_steps_per_sec": T * C * N / sec,
        "env_steps_per_sec_per_copy": T * N / sec,
        "env_steps_per_sec_per_env": T / sec,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 2 ** 20}))


def child_run(n_rates, copies_per_rate, iters, warmup, style):
    """Run one configuration in a subprocess and return its parsed result (None if it failed)."""
    cmd = [sys.executable, __file__, "--child", "--rate-count", str(n_rates),
           "--copies-per-rate", str(copies_per_rate), "--iters", str(iters),
           "--warmup", str(warmup), "--style", style]
    out = subprocess.run(cmd, capture_output=True, text=True)
    line = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
    if not line:
        tail = (out.stdout + out.stderr)[-400:].replace("\n", " ")
        print(f"  {n_rates} rates x {copies_per_rate} copies: FAILED — {tail}")
        return None
    return json.loads(line[0][len("RESULT "):])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", choices=["rates", "copies", "groups"], default="rates")
    ap.add_argument("--rate-counts", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--rate-count", type=int, default=16)
    ap.add_argument("--copies-list", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
    ap.add_argument("--copies-per-rate", type=int, default=128)
    ap.add_argument("--total-copies", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--style", default="epoch_minibatch")
    ap.add_argument("--child", action="store_true")
    args = ap.parse_args()

    if args.child:
        measure(args.rate_count, args.copies_per_rate, args.iters, args.warmup, args.style)
        return

    # each study is a list of (number of rates, copies per rate) pairs
    if args.study == "rates":
        points = [(g, args.copies_per_rate) for g in args.rate_counts]
    elif args.study == "copies":
        points = [(args.rate_count, k) for k in args.copies_list]
    else:
        points = [(g, args.total_copies // g) for g in args.rate_counts
                  if args.total_copies % g == 0]

    rows, failures = [], []
    print(f"{'rates':>6} {'per rate':>9} {'copies':>7} {'ms/iter':>9} "
          f"{'total env-steps/s':>18} {'per copy':>10} {'per env':>9} {'VRAM MB':>8}")
    for g, k in points:
        r = child_run(g, k, args.iters, args.warmup, args.style)
        if r is None:
            # a configuration that did not run is recorded, not silently dropped
            failures.append({"n_rates": g, "copies_per_rate": k, "total_copies": g * k})
            continue
        rows.append(r)
        print(f"{g:>6} {k:>9} {r['total_copies']:>7} {r['sec_per_iteration']*1e3:>9.1f} "
              f"{r['env_steps_per_sec']:>18.3e} {r['env_steps_per_sec_per_copy']:>10.0f} "
              f"{r['env_steps_per_sec_per_env']:>9.0f} {r['peak_vram_mb']:>8.0f}")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{time.strftime('%Y-%m-%d-%H-%M-%S')}_sweep_scaling_{args.study}.json"
    out.write_text(json.dumps({"study": args.study, "style": args.style, "rows": rows,
                               "failures": failures,
                               "order": [[g, k] for g, k in points]}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
