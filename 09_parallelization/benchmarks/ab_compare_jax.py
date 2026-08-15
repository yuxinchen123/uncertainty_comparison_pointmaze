"""Paired A/B comparison of two JAX trainer configurations, with a noise floor.

Same protocol as the torch-side ab_compare.py: the two configurations run in ABBA order, each
in its own process, and the verdict states plainly whether the difference clears the spread of
the same configuration measured twice. Comparing against a number measured hours earlier is not
a comparison; this is.

Usage (serval05, under the H100 lock):
  python ab_compare_jax.py --name hoist-C128 \\
      --a '{"n_copies":128,"hoist_rollout":false}' --b '{"n_copies":128}' --iters 60
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

DEFAULTS = dict(update_style="epoch_minibatch")


def run_one(overrides, iters, warmup):
    """Median seconds per iteration for one configuration (own process)."""
    import jax
    from jax_ppo_rnd import PPOConfig, JaxPPORND
    cfg = PPOConfig(**{**DEFAULTS, **overrides})
    trainer = JaxPPORND(cfg)
    state = trainer.init_state()
    state = trainer.prime_obs_rms(state, jax.random.PRNGKey(0))
    # warmup covers compilation; the timed loop then measures steady-state replay
    for i in range(warmup):
        state, _ = trainer._iterate(state, jax.random.PRNGKey(i), 3e-4)
    jax.block_until_ready(state)
    times = []
    for i in range(iters):
        t0 = time.perf_counter()
        state, _ = trainer._iterate(state, jax.random.PRNGKey(1000 + i), 3e-4)
        jax.block_until_ready(state)
        times.append(time.perf_counter() - t0)
    times.sort()
    return {"median": times[len(times) // 2], "p10": times[len(times) // 10],
            "p90": times[int(len(times) * 0.9)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--a", default="{}")
    ap.add_argument("--b", required=False)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--child")
    args = ap.parse_args()

    if args.child:
        print("RESULT " + json.dumps(run_one(json.loads(args.child), args.iters, args.warmup)))
        return

    got = {"a": [], "b": []}
    for side in ("a", "b", "b", "a"):
        payload = args.a if side == "a" else args.b
        out = subprocess.run([sys.executable, __file__, "--name", args.name, "--child", payload,
                              "--iters", str(args.iters), "--warmup", str(args.warmup)],
                             capture_output=True, text=True).stdout
        line = [l for l in out.splitlines() if l.startswith("RESULT ")]
        assert line, f"side {side} produced no result:\n{out[-3000:]}"
        got[side].append(json.loads(line[0][len("RESULT "):]))

    med = {k: sorted(x["median"] for x in v)[len(v) // 2] for k, v in got.items()}
    spread = max(max(x["median"] for x in v) - min(x["median"] for x in v) for v in got.values())
    diff = med["a"] - med["b"]
    verdict = ("B is faster" if diff > spread else "A is faster" if -diff > spread else
               "no effect above the noise floor")
    row = {"name": args.name, "a_overrides": json.loads(args.a), "b_overrides": json.loads(args.b),
           "a_ms": med["a"] * 1e3, "b_ms": med["b"] * 1e3, "difference_ms": diff * 1e3,
           "noise_floor_ms": spread * 1e3, "relative_change_percent": diff / med["a"] * 100,
           "a_iters_per_sec": 1 / med["a"], "b_iters_per_sec": 1 / med["b"],
           "verdict": verdict, "iters": args.iters,
           "a_runs_ms": [x["median"] * 1e3 for x in got["a"]],
           "b_runs_ms": [x["median"] * 1e3 for x in got["b"]]}
    print(f"{args.name}: A {row['a_ms']:.2f} ms | B {row['b_ms']:.2f} ms | "
          f"difference {row['difference_ms']:+.2f} ms ({row['relative_change_percent']:+.1f}%) | "
          f"noise floor {row['noise_floor_ms']:.2f} ms -> {verdict}")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{time.strftime('%Y-%m-%d-%H-%M-%S')}_abjax_{args.name}.json"
    out.write_text(json.dumps(row, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
