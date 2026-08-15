"""Paired A/B comparison of two trainer configurations, with a noise floor and a decision rule.

Round one compared configurations by running each once and reading the numbers. That cannot tell a
2% change from 2% drift. This harness runs the two configurations in ABBA order, each in its own
process (so one configuration's memory pool or compile cache cannot affect the other), and reports:
  - the median per-iteration time of each configuration,
  - the paired difference,
  - the within-configuration spread (the same configuration measured twice, in different slots),
  - a verdict: a difference smaller than the spread is NOT evidence of an effect.

Usage (serval05, under the H100 lock):
  python ab_compare.py --name post-compile \\
      --a '{"n_copies":128}' --b '{"n_copies":128,"compile_post":true}' --iters 60
Every key of the JSON objects is a PPOConfig field; both sides start from the production defaults
below and apply their own overrides.
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

PRODUCTION = dict(update_style="epoch_minibatch", rollout_mode="capture", capture_update=True,
                  fused_adam=True, one_graph=True, tf32=True, compile_post=True,
                  compile_opt=True)


def load_module(rev):
    """The trainer module: the working tree, or a git revision (for before/after pairing)."""
    if not rev:
        import torch_ppo_rnd
        return torch_ppo_rnd
    import importlib.util
    repo = BASE.parent
    rel = "09_parallelization/ppo/torch_ppo/torch_ppo_rnd.py"
    src = subprocess.run(["git", "-C", str(repo), "show", f"{rev}:{rel}"],
                         capture_output=True, text=True, check=True).stdout
    src = src.replace("BASE = Path(__file__).resolve().parent.parent.parent",
                      f'BASE = Path("{BASE}")')
    path = Path(f"/localtmp/sl5nw/torch_ppo_rnd_rev_{rev}.py")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(src)
    spec = importlib.util.spec_from_file_location(f"trainer_rev_{rev}", path)
    mod = importlib.util.module_from_spec(spec)
    # register before executing: the compiler re-imports the defining module by name when it
    # traces a function, and a module loaded only from a file path is not importable that way
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def run_one(overrides, iters, warmup):
    """Median seconds per iteration for one configuration (called in its own process).

    An override key "__rev__" selects a git revision of the trainer instead of the working
    tree, so a change can be timed against its own predecessor; keys the older revision does
    not define are dropped, with a note, so an added knob does not make the pairing impossible.
    """
    import torch
    overrides = dict(overrides)
    rev = overrides.pop("__rev__", None)
    mod = load_module(rev)
    PPOConfig, PPORND = mod.PPOConfig, mod.PPORND
    fields = set(PPOConfig.__dataclass_fields__)
    merged = {**PRODUCTION, **overrides}
    dropped = sorted(k for k in merged if k not in fields)
    for k in dropped:
        merged.pop(k)
    if dropped:
        print(f"NOTE revision {rev} has no field(s) {dropped}; using its defaults",
              file=sys.stderr)
    cfg = PPOConfig(**merged)
    trainer = PPORND(cfg, device="cuda")
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
    return {"median": times[len(times) // 2], "p10": times[len(times) // 10],
            "p90": times[int(len(times) * 0.9)], "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20}


def child(payload, iters, warmup):
    """Entry point for the per-configuration subprocess."""
    r = run_one(json.loads(payload), iters, warmup)
    print("RESULT " + json.dumps(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="short name of the experiment for the record")
    ap.add_argument("--a", default="{}", help="JSON overrides for side A (the baseline)")
    ap.add_argument("--b", required=False, help="JSON overrides for side B (the change)")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--child", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child:
        child(args.child, args.iters, args.warmup)
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
    verdict = ("B is faster" if diff > spread else
               "A is faster" if -diff > spread else
               "no effect above the noise floor")
    row = {"name": args.name, "a_overrides": json.loads(args.a), "b_overrides": json.loads(args.b),
           "a_ms": med["a"] * 1e3, "b_ms": med["b"] * 1e3, "difference_ms": diff * 1e3,
           "noise_floor_ms": spread * 1e3, "relative_change_percent": diff / med["a"] * 100,
           "verdict": verdict, "iters": args.iters,
           "a_runs_ms": [x["median"] * 1e3 for x in got["a"]],
           "b_runs_ms": [x["median"] * 1e3 for x in got["b"]],
           "peak_vram_mb": {"a": got["a"][0]["peak_vram_mb"], "b": got["b"][0]["peak_vram_mb"]}}
    print(f"{args.name}: A {row['a_ms']:.2f} ms | B {row['b_ms']:.2f} ms | "
          f"difference {row['difference_ms']:+.2f} ms ({row['relative_change_percent']:+.1f}%) | "
          f"noise floor {row['noise_floor_ms']:.2f} ms -> {verdict}")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{time.strftime('%Y-%m-%d-%H-%M-%S')}_ab_{args.name}.json"
    out.write_text(json.dumps(row, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
