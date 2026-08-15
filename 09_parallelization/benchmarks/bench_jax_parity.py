"""What do the two parity features cost the JAX trainer?

Mirrors the torch-side bench_uniform_vs_sweep.py, and adds the coverage feature. Comparing a
swept run against a uniform one mixes two separate changes, so the arms separate them:

  uniform      one rate for every copy; the optimizer multiplies by a scalar
  same-rates   the sweep path, every group given the SAME rate — this is the cost of the
               mechanism (a per-copy rate vector broadcast along the copy axis)
  diff-rates   the sweep path with genuinely different rates — changes no code at all, only
               the constants the same kernels read, so it should cost nothing
  coverage     uniform, plus the per-copy visited-cell map maintained on the device
  both         the full parity configuration: differing rates and coverage together

All arms are built in ONE process and then timed round-robin for many rounds, so every arm
meets the same clocks, the same memory state and the same drift. An earlier version of this
benchmark ran each arm in its own process: process-to-process variation alone was 0.7 ms on a
13 ms iteration, which is larger than any of the effects being measured, and raising the
iteration count did not reduce it. The noise floor reported here is the spread of the same arm
across rounds, which is the right yardstick for a difference of a few percent.

Both timing modes are measured because they answer different questions: "sync" waits for each
iteration, "pipelined" lets the host run ahead and waits once per block.

Usage (through the H100 lock wrapper):
  python bench_jax_parity.py --copies 8 128 --rates 16 --iters 40
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "ppo" / "jax_ppo"))
RESULTS = Path(__file__).resolve().parent / "results"
ARMS = ("uniform", "same-rates", "diff-rates", "coverage", "both")


def rates_for(n, same):
    """The rate list for one arm: all equal, or spread log-uniformly over a realistic range."""
    if same:
        return [3e-4] * n
    lo, hi = 1e-5, 1e-2
    return [lo * (hi / lo) ** (i / (n - 1)) for i in range(n)] if n > 1 else [3e-4]


def build(arm, total_copies, n_rates, style):
    """A trainer and a primed state for one arm."""
    import jax
    from jax_ppo_rnd import PPOConfig, JaxPPORND, sweep_config
    cover = arm in ("coverage", "both")
    if arm in ("uniform", "coverage"):
        cfg = PPOConfig(n_copies=total_copies, update_style=style, track_coverage=cover)
    else:
        # a sweep cannot have more groups than copies; at small copy counts the group count
        # falls back to the largest divisor of the copy count that is at most n_rates
        g = min(n_rates, total_copies)
        while total_copies % g:
            g -= 1
        cfg = sweep_config(rates_for(g, arm == "same-rates"), total_copies // g,
                           style=style, track_coverage=cover)
    trainer = JaxPPORND(cfg)
    state = trainer.init_state()
    state = trainer.prime_obs_rms(state, jax.random.PRNGKey(0))
    return trainer, state


def time_round(trainer, state, iters, timing, salt):
    """One timing round for one arm; returns (seconds per iteration, new state)."""
    import jax
    lr = trainer.lr_argument(1, 10 ** 6)
    if timing == "sync":
        times = []
        for i in range(iters):
            t0 = time.perf_counter()
            state, m = trainer._iterate(state, jax.random.PRNGKey(salt + i), lr)
            jax.block_until_ready(m["loss"])
            times.append(time.perf_counter() - t0)
        times.sort()
        return times[len(times) // 2], state
    t0 = time.perf_counter()                       # pipelined: wait once for the whole block
    for i in range(iters):
        state, m = trainer._iterate(state, jax.random.PRNGKey(salt + i), lr)
    jax.block_until_ready(m["loss"])
    return (time.perf_counter() - t0) / iters, state


def measure_all(total_copies, n_rates, iters, warmup, style, timing, rounds):
    """Build every arm, then time them round-robin; prints a RESULT line with all arms."""
    import jax
    built = {}
    for arm in ARMS:
        trainer, state = build(arm, total_copies, n_rates, style)
        for i in range(warmup):                    # covers compilation
            state, m = trainer._iterate(state, jax.random.PRNGKey(i),
                                        trainer.lr_argument(1, 10 ** 6))
        jax.block_until_ready(state)
        built[arm] = (trainer, state)
    per_round = {a: [] for a in ARMS}
    for rnd in range(rounds):
        # forward then reversed arm order within each pair of rounds, so a monotone drift
        # cannot systematically favour whichever arm is measured first
        order = ARMS if rnd % 2 == 0 else tuple(reversed(ARMS))
        for arm in order:
            trainer, state = built[arm]
            sec, state = time_round(trainer, state, iters, timing, 1000 * (rnd + 1))
            built[arm] = (trainer, state)
            per_round[arm].append(sec)
    print("RESULT " + json.dumps({
        "total_copies": total_copies, "n_rates": n_rates, "timing": timing, "style": style,
        "per_round_sec": per_round}))


def run_child(c, n_rates, iters, warmup, style, timing, rounds):
    """All arms for one copy count, in one process; returns the parsed result or None."""
    cmd = [sys.executable, __file__, "--child", "all", "--copies", str(c), "--rates",
           str(n_rates), "--iters", str(iters), "--warmup", str(warmup), "--style", style,
           "--timing", timing, "--rounds", str(rounds)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    line = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
    if not line:
        print(f"  {c} copies ({timing}) FAILED: {(out.stdout + out.stderr)[-500:]}")
        return None
    return json.loads(line[0][len("RESULT "):])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copies", type=int, nargs="+", default=[8, 128])
    ap.add_argument("--rates", type=int, default=16)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--style", default="epoch_minibatch")
    ap.add_argument("--timing", default="sync", choices=["sync", "pipelined"])
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--child")
    args = ap.parse_args()

    if args.child:
        measure_all(args.copies[0], args.rates, args.iters, args.warmup,
                    args.style, args.timing, args.rounds)
        return

    rows = []
    print(f"style {args.style}, {args.rates} rates, timing {args.timing}, "
          f"{args.rounds} interleaved rounds of {args.iters} iterations")
    print(f"{'copies':>7} " + " ".join(f"{a:>11}" for a in ARMS) +
          f" {'noise':>7} {'mechanism':>10} {'differing':>10} {'coverage':>9} {'both':>7}")
    for c in args.copies:
        r = run_child(c, args.rates, args.iters, args.warmup, args.style, args.timing,
                      args.rounds)
        if r is None:
            continue
        got = r["per_round_sec"]
        med = {a: sorted(got[a])[len(got[a]) // 2] for a in ARMS}
        # the noise floor is how much the SAME arm moves between rounds, worst over arms
        noise = max(max(got[a]) - min(got[a]) for a in ARMS)
        pct = lambda x, y: (med[x] / med[y] - 1) * 100
        row = {"total_copies": c, "n_rates": args.rates, "timing": args.timing,
               "style": args.style, "median_sec": med, "noise_floor_sec": noise,
               "mechanism_cost_percent": pct("same-rates", "uniform"),
               "differing_cost_percent": pct("diff-rates", "same-rates"),
               "coverage_cost_percent": pct("coverage", "uniform"),
               "both_cost_percent": pct("both", "uniform"),
               "per_round_sec": got}
        rows.append(row)
        print(f"{c:>7} " + " ".join(f"{med[a]*1e3:>11.2f}" for a in ARMS) +
              f" {noise*1e3:>7.2f} {row['mechanism_cost_percent']:>9.1f}% "
              f"{row['differing_cost_percent']:>9.1f}% {row['coverage_cost_percent']:>8.1f}% "
              f"{row['both_cost_percent']:>6.1f}%")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / (f"{time.strftime('%Y-%m-%d-%H-%M-%S')}_jax_parity_"
                     f"{args.style}_{args.timing}.json")
    out.write_text(json.dumps({"style": args.style, "timing": args.timing,
                               "n_rates": args.rates, "rows": rows}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
