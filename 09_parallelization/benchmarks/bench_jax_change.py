"""Paired timing of one trainer change against the unchanged trainer.

The autoresearch loop needs to answer one question per change: is the iteration faster, by more
than the measurement moves on its own? Comparing two processes cannot answer it — process to
process variation is around 0.7 ms on an iteration of 8 to 13 ms, larger than most single
changes. So both versions are built in ONE process and timed round-robin, with the order
reversed on alternate rounds, exactly as the parity harness does; the same-version spread
between rounds is reported as the noise floor and any difference smaller than it means nothing.

The change under test is named by a config field and the two values to compare, so the same
harness serves every round-five change:

  python bench_jax_change.py --knob flat_params --off False --on True --copies 8 32 128

Several knobs at once, for comparing whole configurations, by naming the knob "set":

  python bench_jax_change.py --knob set --off scan_unroll=4,update_unroll=1 \
                                        --on  scan_unroll=0,update_unroll=2

Both timing modes are reported by running it twice (--timing sync, --timing pipelined); they
measure different things and are never compared against each other.
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


def as_value(text):
    """A command-line knob value as the python value it names (strings pass through)."""
    if text in ("True", "False"):
        return text == "True"
    try:
        return int(text)
    except ValueError:
        return text


def as_settings(knob, value):
    """One arm's config fields, from either the single-knob or the several-knob form.

    before: knob "scan_unroll", value 4            -> {"scan_unroll": 4}
    before: knob "set", value "scan_unroll=4,update_unroll=1"
    after:  {"scan_unroll": 4, "update_unroll": 1}
    """
    if knob != "set":
        return {knob: value}
    return {part.split("=")[0]: as_value(part.split("=")[1])
            for part in str(value).split(",")}


def build(knob, value, total_copies, style):
    """A trainer and its primed state, with one or several config fields set."""
    import jax
    from jax_ppo_rnd import PPOConfig, JaxPPORND
    cfg = PPOConfig(n_copies=total_copies, update_style=style, **as_settings(knob, value))
    trainer = JaxPPORND(cfg)
    state = trainer.init_state()
    state = trainer.prime_obs_rms(state, jax.random.PRNGKey(0))
    return trainer, state


def measure(knob, off, on, total_copies, iters, warmup, style, timing, rounds):
    """Build both versions, time them round-robin, print one RESULT line."""
    import jax
    from bench_jax_parity import time_round

    arms = {"off": off, "on": on}
    built = {}
    for name, value in arms.items():
        trainer, state = build(knob, value, total_copies, style)
        for i in range(warmup):                    # covers compilation
            state, m = trainer._iterate(state, jax.random.PRNGKey(i),
                                        trainer.lr_argument(1, 10 ** 6))
        jax.block_until_ready(state)
        built[name] = (trainer, state)

    per_round = {name: [] for name in arms}
    for rnd in range(rounds):
        # forward then reversed order on alternate rounds, so a drift in the machine cannot
        # systematically favour whichever version is measured first
        order = ["off", "on"] if rnd % 2 == 0 else ["on", "off"]
        for name in order:
            trainer, state = built[name]
            sec, state = time_round(trainer, state, iters, timing, 1000 * (rnd + 1))
            built[name] = (trainer, state)
            per_round[name].append(sec)

    print("RESULT " + json.dumps({
        "knob": knob, "off": str(off), "on": str(on), "total_copies": total_copies,
        "timing": timing, "style": style, "per_round_sec": per_round}))


def run_child(knob, off, on, c, iters, warmup, style, timing, rounds):
    """One copy count in its own process, so a failure in one size does not lose the rest."""
    cmd = [sys.executable, __file__, "--child", "1", "--knob", knob, "--off", str(off),
           "--on", str(on), "--copies", str(c), "--iters", str(iters), "--warmup", str(warmup),
           "--style", style, "--timing", timing, "--rounds", str(rounds)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    line = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
    if not line:
        print(f"  {c} copies ({timing}) FAILED: {(out.stdout + out.stderr)[-600:]}")
        return None
    return json.loads(line[0][len("RESULT "):])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knob", required=True)
    ap.add_argument("--off", default="False")
    ap.add_argument("--on", default="True")
    ap.add_argument("--copies", type=int, nargs="+", default=[8, 32, 128])
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--style", default="epoch_minibatch")
    ap.add_argument("--timing", default="sync", choices=["sync", "pipelined"])
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--child")
    args = ap.parse_args()

    off, on = as_value(args.off), as_value(args.on)
    if args.child:
        measure(args.knob, off, on, args.copies[0], args.iters, args.warmup,
                args.style, args.timing, args.rounds)
        return

    rows = []
    print(f"{args.knob}: {off} against {on}, style {args.style}, timing {args.timing}, "
          f"{args.rounds} interleaved rounds of {args.iters} iterations")
    print(f"{'copies':>7} {'off ms':>9} {'on ms':>9} {'change':>9} {'noise ms':>9} "
          f"{'paired ms':>9} {'won':>7} {'verdict':>8}")
    for c in args.copies:
        r = run_child(args.knob, off, on, c, args.iters, args.warmup, args.style,
                      args.timing, args.rounds)
        if r is None:
            continue
        got = r["per_round_sec"]
        med = {k: sorted(v)[len(v) // 2] for k, v in got.items()}
        # the noise floor is how much the SAME version moves between rounds, worst of the two
        noise = max(max(v) - min(v) for v in got.values())
        change = (med["on"] / med["off"] - 1) * 100
        # the two versions are measured in the same round on the same machine, so the honest
        # comparison is round by round: the drift that moves one moves the other with it.
        # before: off = [8.1, 8.4, 8.0], on = [7.9, 8.2, 7.8] (spread 0.4, differences 0.2)
        # after:  won 3 of 3 rounds, median difference 0.2 ms — visible although the spread is
        #         twice as large as the effect
        paired = [o - n for o, n in zip(got["off"], got["on"])]
        wins = sum(1 for d in paired if d > 0)
        med_paired = sorted(paired)[len(paired) // 2]
        verdict = ("faster" if wins == len(paired) else
                   "slower" if wins == 0 else "mixed")
        rows.append({**r, "median_sec": med, "noise_floor_sec": noise,
                     "change_percent": change, "paired_diff_sec": paired,
                     "rounds_favouring_on": wins, "median_paired_diff_sec": med_paired,
                     "verdict": verdict})
        print(f"{c:>7} {med['off']*1e3:>9.2f} {med['on']*1e3:>9.2f} {change:>8.1f}% "
              f"{noise*1e3:>9.2f} {med_paired*1e3:>+9.2f} {wins:>3}/{len(paired):<3} "
              f"{verdict:>8}")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / (f"{time.strftime('%Y-%m-%d-%H-%M-%S')}_jax_change_{args.knob}_"
                     f"{args.style}_{args.timing}.json")
    out.write_text(json.dumps({"rows": rows}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
