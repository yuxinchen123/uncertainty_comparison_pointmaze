"""Paired, in-process, round-by-round comparison of two torch trainer configurations.

This is the torch twin of `bench_jax_change.py`, and it exists for the reason that script
records: comparing two configurations in separate processes has a noise floor of 0.7 ms on an
8-13 ms iteration, which does not shrink with more timed iterations because the variation is
between processes rather than within them. Both configurations are therefore built in ONE
process and timed round-robin, with the order reversed on alternate rounds so that a drift in
the machine cannot systematically favour whichever is measured first.

The verdict statistic is the one the jax round five adopted after finding that the older rule
(two medians against the largest spread a single version shows across rounds) called a real 3%
effect "noise": the two arms are timed in the SAME round on the SAME machine, so whatever drift
moves one moves the other, and the difference is taken round by round. A change that wins every
round is real however large the between-round spread is; the sign test over R rounds has
p = 2^-R under the null.

Usage (through the H100 lock wrapper):
  python bench_torch_change.py --knob fused_gradient_norm --off False --on True \
      --copies 1024 4096 --rounds 11
  python bench_torch_change.py --knob set --off "a=1,b=2" --on "a=3,b=4" --copies 4096
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


def as_value(text):
    """Coerce one command-line knob value to the type PPOConfig expects."""
    # booleans first, because int("True") raises and bool("False") is True
    if text in ("True", "False"):
        return text == "True"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def overrides(knob, text):
    """The config overrides one arm applies, as a dict.

    before: knob="fused_gradient_norm", text="True"   ; after: {"fused_gradient_norm": True}
    before: knob="set", text="a=1,b=False"            ; after: {"a": 1, "b": False}
    """
    if knob != "set":
        return {knob: as_value(text)}
    return {k: as_value(v) for k, v in (p.split("=", 1) for p in text.split(",") if p)}


def build(cfg_overrides, n_copies, style, device="cuda"):
    """Build one warmed trainer with its iteration graph captured, ready to be timed."""
    import torch
    from torch_ppo_rnd import PPORND, production_config
    cfg = production_config(n_copies, style=style, **cfg_overrides)
    # the assertion that the arm under test is actually the arm asked for: a knob silently
    # dropped by a dataclass default would make the two arms identical and the comparison
    # meaningless, which is the failure this project has already hit twice
    for k, v in cfg_overrides.items():
        assert getattr(cfg, k) == v, f"the config did not take {k}={v}"
    trainer = PPORND(cfg, device=device)
    trainer.prime_obs_rms()
    trainer._build_iteration_graph()
    torch.cuda.synchronize()
    return trainer


def time_round(trainer, iters, timing, salt):
    """Seconds per iteration for one arm in one round: median (sync) or mean (pipelined)."""
    import torch
    # the same salt in both arms of a round, so the two see the same action noise and the same
    # permutations; the timing does not depend on the values, but a difference in them would
    # make a numerical comparison of the two arms impossible afterwards
    torch.cuda.manual_seed(salt)
    torch.manual_seed(salt)
    if timing == "pipelined":
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            trainer.iteration_captured()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        trainer.iteration_captured()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2]


def run_child(args):
    """Measure one copy count: build both arms, warm them, time them round by round."""
    import torch
    off_over = overrides(args.knob, args.off)
    on_over = overrides(args.knob, args.on)
    assert off_over != on_over, "the two arms are the same configuration"
    arms = {"off": build(off_over, args.copies[0], args.style),
            "on": build(on_over, args.copies[0], args.style)}
    for t in arms.values():
        for _ in range(args.warmup):
            t.iteration_captured()
    torch.cuda.synchronize()

    # round-robin with the order flipped on odd rounds, so a machine drift cannot favour
    # whichever arm is always measured first
    got = {"off": [], "on": []}
    for rnd in range(args.rounds):
        order = ["off", "on"] if rnd % 2 == 0 else ["on", "off"]
        for name in order:
            got[name].append(time_round(arms[name], args.iters, args.timing, 1000 * (rnd + 1)))

    # paired difference per round; positive means the "on" arm was faster in that round
    paired = [o - n for o, n in zip(got["off"], got["on"])]
    wins = sum(1 for d in paired if d > 0)
    med = {k: sorted(v)[len(v) // 2] for k, v in got.items()}
    floor = max(max(v) - min(v) for v in got.values())
    verdict = "faster" if wins == len(paired) else "slower" if wins == 0 else "mixed"
    row = {
        "n_copies": args.copies[0], "style": args.style, "timing": args.timing,
        "off": off_over, "on": on_over,
        "median_sec": med, "noise_floor_sec": floor,
        "change_percent": (med["on"] / med["off"] - 1.0) * 100.0,
        "median_paired_diff_sec": sorted(paired)[len(paired) // 2],
        "paired_diff_sec": paired, "per_round_sec": got,
        "rounds_favouring_on": wins, "rounds": len(paired), "verdict": verdict,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
    }
    print("RESULT " + json.dumps(row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knob", required=True)
    ap.add_argument("--off", default="False")
    ap.add_argument("--on", default="True")
    ap.add_argument("--copies", type=int, nargs="+", default=[1024, 4096])
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--style", default="epoch_minibatch")
    ap.add_argument("--timing", default="sync", choices=["sync", "pipelined"])
    ap.add_argument("--rounds", type=int, default=11)
    ap.add_argument("--tag", default="")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child:
        run_child(args)
        return

    # one child process per copy count, so an out-of-memory failure at one size does not lose
    # the sizes already measured
    import torch  # noqa: F401  (imported only for the version recorded below)
    rows = []
    for c in args.copies:
        cmd = [sys.executable, __file__, "--child", "--knob", args.knob, "--off", args.off,
               "--on", args.on, "--copies", str(c), "--iters", str(args.iters),
               "--warmup", str(args.warmup), "--style", args.style,
               "--timing", args.timing, "--rounds", str(args.rounds)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT ")), None)
        if line is None:
            print(proc.stdout[-3000:])
            print(proc.stderr[-4000:], file=sys.stderr)
            raise RuntimeError(f"the child measuring {c} copies produced no result")
        row = json.loads(line[len("RESULT "):])
        rows.append(row)
        print(f"C={c:>5d} {args.style} {args.timing}: off {row['median_sec']['off']*1e3:8.2f} ms  "
              f"on {row['median_sec']['on']*1e3:8.2f} ms  "
              f"{row['change_percent']:+6.2f}%  floor {row['noise_floor_sec']*1e3:.2f} ms  "
              f"{row['rounds_favouring_on']}/{row['rounds']} rounds favour on -> {row['verdict']}",
              flush=True)

    RESULTS.mkdir(exist_ok=True)
    stamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    out = RESULTS / f"{stamp}_torch_change_{args.knob}_{args.style}_{args.timing}{args.tag}.json"
    out.write_text(json.dumps({
        "knob": args.knob, "off": args.off, "on": args.on, "style": args.style,
        "timing": args.timing, "rounds": args.rounds, "iters": args.iters,
        "torch": __import__("torch").__version__,
        "git": subprocess.run(["git", "-C", str(BASE), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip(),
        "rows": rows}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
