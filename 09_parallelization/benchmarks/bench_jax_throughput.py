"""Throughput of the JAX trainer as it stands, per copy and in total.

One row per copy count, reporting both rates: what all copies together produce, and what a
single copy gets. The per-copy rate is restated as time, normalised to one million environment
steps per copy whatever a real run's step budget would be.

Each copy collects `num_steps` x `n_envs` environment steps per iteration, so the total rate is
that number times the copy count divided by the time an iteration takes.

Usage (through the lock wrapper):
  python bench_jax_throughput.py --copies 8 32 128 --timing sync
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "ppo" / "jax_ppo"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
RESULTS = Path(__file__).resolve().parent / "results"


def measure(total_copies, iters, warmup, style, timing, rounds):
    """Median seconds per iteration for one copy count; prints one RESULT line."""
    import jax
    from jax_ppo_rnd import PPOConfig, JaxPPORND
    from bench_jax_parity import time_round

    trainer = JaxPPORND(PPOConfig(n_copies=total_copies, update_style=style))
    state = trainer.init_state()
    state = trainer.prime_obs_rms(state, jax.random.PRNGKey(0))
    for i in range(warmup):                        # covers compilation
        state, m = trainer._iterate(state, jax.random.PRNGKey(i),
                                    trainer.lr_argument(1, 10 ** 6))
    jax.block_until_ready(state)

    per_round = []
    for rnd in range(rounds):
        sec, state = time_round(trainer, state, iters, timing, 1000 * (rnd + 1))
        per_round.append(sec)
    print("RESULT " + json.dumps({
        "total_copies": total_copies, "timing": timing, "style": style,
        "steps_per_iteration_per_copy": trainer.cfg.num_steps * trainer.cfg.n_envs,
        "scan_unroll": trainer.scan_unroll, "per_round_sec": per_round}))


def run_child(c, iters, warmup, style, timing, rounds):
    """One copy count in its own process, so a failure at one size does not lose the rest."""
    cmd = [sys.executable, __file__, "--child", "1", "--copies", str(c), "--iters", str(iters),
           "--warmup", str(warmup), "--style", style, "--timing", timing,
           "--rounds", str(rounds)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    line = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
    if not line:
        print(f"  {c} copies ({timing}) FAILED: {(out.stdout + out.stderr)[-600:]}")
        return None
    return json.loads(line[0][len("RESULT "):])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copies", type=int, nargs="+", default=[8, 32, 128])
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--style", default="epoch_minibatch")
    ap.add_argument("--timing", default="sync", choices=["sync", "pipelined"])
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--child")
    args = ap.parse_args()

    if args.child:
        measure(args.copies[0], args.iters, args.warmup, args.style, args.timing, args.rounds)
        return

    rows = []
    print(f"JAX trainer throughput, style {args.style}, timing {args.timing}, "
          f"{args.rounds} rounds of {args.iters} iterations")
    print(f"{'copies':>7} {'seconds per':>13} {'total steps per':>17} "
          f"{'steps per second':>18} {'hours per million':>19}")
    print(f"{'':>7} {'iteration':>13} {'second (millions)':>17} "
          f"{'per copy (thousands)':>18} {'steps per copy':>19}")
    for c in args.copies:
        r = run_child(c, args.iters, args.warmup, args.style, args.timing, args.rounds)
        if r is None:
            continue
        sec = sorted(r["per_round_sec"])[len(r["per_round_sec"]) // 2]
        per_copy_steps = r["steps_per_iteration_per_copy"]
        per_copy_rate = per_copy_steps / sec                       # steps per second, one copy
        total_rate = per_copy_rate * c                             # steps per second, all copies
        hours = 1e6 / (3600 * per_copy_rate)          # hours one copy needs for a million steps
        rows.append({**r, "median_sec": sec, "total_steps_per_second": total_rate,
                     "steps_per_second_per_copy": per_copy_rate,
                     "hours_per_million_steps_per_copy": hours})
        print(f"{c:>7} {sec:>13.5f} {total_rate/1e6:>17.3f} "
              f"{per_copy_rate/1e3:>18.2f} {hours:>19.4f}")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / (f"{time.strftime('%Y-%m-%d-%H-%M-%S')}_jax_throughput_"
                     f"{args.style}_{args.timing}.json")
    out.write_text(json.dumps({"rows": rows}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
