"""How should a learning-rate sweep be laid out on one GPU? Three strategies, measured.

A sweep trains G groups of copies, each group at its own learning rate. There is more than one
way to arrange that:

  fused      one trainer holding all G*K copies, with a per-copy learning-rate vector, in one
             captured graph. Every group advances in the same batched kernels.
  separate   G trainers of K copies each, run one after another — what you would write without
             a per-copy rate. Each has its own graph and its own scalar rate.
  uniform    one trainer of G*K copies at a SINGLE rate, using the stock fused Adam. Not a
             sweep; it is the reference that prices the per-copy Adam against torch's.

Reported per strategy: seconds per iteration for the WHOLE sweep (for "separate", the sum over
its G trainers, since they share one GPU and run in turn), and peak memory.

Usage (serval05, under the H100 lock):
  python bench_sweep.py --rates 1e-4 3e-4 1e-3 3e-3 --copies-per-rate 128 --iters 30
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


def timed(trainer, iters, warmup):
    """Median seconds per captured iteration."""
    import torch
    trainer.prime_obs_rms()
    trainer._build_iteration_graph()
    for _ in range(warmup):
        trainer.iteration_captured()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        trainer.iteration_captured()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


def run_strategy(strategy, rates, per_rate, iters, warmup, style):
    """One strategy, in its own process; prints a RESULT line."""
    import torch
    from torch_ppo_rnd import PPORND, production_config, sweep_config
    total = len(rates) * per_rate

    if strategy == "fused":
        t = PPORND(sweep_config(rates, per_rate, style=style), device="cuda")
        sec = timed(t, iters, warmup)
    elif strategy == "uniform":
        t = PPORND(production_config(total, style=style), device="cuda")
        sec = timed(t, iters, warmup)
    else:                                   # separate: one trainer per rate, run in turn
        sec = 0.0
        for r in rates:
            t = PPORND(production_config(per_rate, style=style, learning_rate=r), device="cuda")
            sec += timed(t, iters, warmup)
            del t
            torch.cuda.empty_cache()
    print("RESULT " + json.dumps({
        "strategy": strategy, "total_copies": total, "groups": len(rates),
        "copies_per_rate": per_rate, "sec_per_iteration": sec,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 2 ** 20}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3, 3e-3])
    ap.add_argument("--copies-per-rate", type=int, default=128)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--style", default="epoch_minibatch")
    ap.add_argument("--child")
    args = ap.parse_args()

    if args.child:
        run_strategy(args.child, args.rates, args.copies_per_rate, args.iters, args.warmup,
                     args.style)
        return

    rows = []
    for strategy in ("uniform", "fused", "separate"):
        cmd = [sys.executable, __file__, "--child", strategy, "--rates",
               *[str(r) for r in args.rates], "--copies-per-rate", str(args.copies_per_rate),
               "--iters", str(args.iters), "--warmup", str(args.warmup), "--style", args.style]
        out = subprocess.run(cmd, capture_output=True, text=True).stdout
        line = [l for l in out.splitlines() if l.startswith("RESULT ")]
        assert line, f"{strategy} produced no result:\n{out[-3000:]}"
        r = json.loads(line[0][len("RESULT "):])
        rows.append(r)
        print(f"{strategy:>9}: {r['sec_per_iteration']*1e3:8.2f} ms per sweep iteration "
              f"({r['total_copies']} copies, {r['groups']} rates), "
              f"peak {r['peak_vram_mb']:.0f} MB")

    fused = next(r for r in rows if r["strategy"] == "fused")["sec_per_iteration"]
    sep = next(r for r in rows if r["strategy"] == "separate")["sec_per_iteration"]
    uni = next(r for r in rows if r["strategy"] == "uniform")["sec_per_iteration"]
    print(f"fused sweep is {sep/fused:.2f}x faster than running the groups separately, "
          f"and costs {(fused/uni - 1)*100:+.1f}% against a uniform-rate run of the same size")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{time.strftime('%Y-%m-%d-%H-%M-%S')}_sweep_strategies.json"
    out.write_text(json.dumps({"rates": args.rates, "copies_per_rate": args.copies_per_rate,
                               "style": args.style, "rows": rows,
                               "fused_speedup_vs_separate": sep / fused,
                               "fused_overhead_vs_uniform": fused / uni - 1}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
