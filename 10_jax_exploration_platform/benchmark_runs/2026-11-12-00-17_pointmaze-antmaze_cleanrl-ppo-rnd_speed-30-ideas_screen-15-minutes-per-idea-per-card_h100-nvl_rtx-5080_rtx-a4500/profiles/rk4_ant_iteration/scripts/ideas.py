"""Speed ideas for the RK4 AntMaze iteration that change the program's layout and no number.

The unit's trainer is built once (common.build_trainer); every variant is the same iteration
function built again with one thing changed, compiled in this process, and all variants are timed
in alternating rounds of chained calls (the order reversed every second round), each on its own
copy of the same state. Recorded per variant: the seconds of its first call (tracing, compiling,
Warp's graph captures and one iteration), the rounds, the graph captures during the timed rounds,
and the throughput. The gain of a variant is the median over rounds of
(the unit's seconds per iteration) / (the variant's seconds per iteration), paired by round.

Variants (names given with --variants, comma-separated):
  rollout-unroll-<n>          the rollout scan holds n steps per body (the unit: 16)
  frame-unroll-<n>            the environment's loop over its 5 physics steps holds n per body (the unit: 1)
  rollout-unroll-<n>_frame-unroll-<k>   both
  update-unroll-<n>           the update's scan over its 16 gradient steps holds n per body (the unit: 2)
  solver-plain-loop           MuJoCo Warp's solver iterations as a plain loop of 4, not a conditional graph loop

  python ideas.py --copies 128 --variants rollout-unroll-1,frame-unroll-5 --out records/x.json
"""
import argparse
import copy
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CaptureCounter, build_trainer, chained_seconds, lr_of, machine, median,  # noqa: E402
                    say, throughput, write_record)


def variant_program(runner, name: str):
    """The compiled iteration of one variant (donating its state, as the unit's program does)."""
    import jax
    from exploration_platform.training.train_step import build_iteration

    cfg, env = runner.cfg, runner.env
    # GOAL: read the changes the name asks for
    # before: "rollout-unroll-4_frame-unroll-5" -> {"rollout-unroll": 4, "frame-unroll": 5}
    changes = {}
    for part in name.split("_"):
        if part == "solver-plain-loop":
            changes["solver-plain-loop"] = True
        else:
            key, value = part.rsplit("-", 1)
            changes[key] = int(value)
    if "rollout-unroll" in changes:
        cfg = replace(cfg, scan_unroll=changes.pop("rollout-unroll"))
    if "update-unroll" in changes:
        cfg = replace(cfg, update_unroll=changes.pop("update-unroll"))
    if "frame-unroll" in changes or "solver-plain-loop" in changes:
        env = copy.copy(env)
        env.frame_unroll = changes.pop("frame-unroll", env.frame_unroll)
        if changes.pop("solver-plain-loop", False):
            # MuJoCo Warp's per-world done flags make every world stop at the same iteration in both loops
            option = env.model.opt
            env.model = env.model.replace(opt=option.replace(_impl=option._impl.replace(graph_conditional=False)))
    if changes:
        raise ValueError(f"unknown parts in the variant name {name!r}: {sorted(changes)}")
    return jax.jit(build_iteration(cfg, env, runner.bonus, runner.sweep, runner.layout), donate_argnums=(0,))


def main():
    """Build the unit's trainer, compile every variant, time them in alternating rounds, write the record."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--copies", type=int, required=True)
    parser.add_argument("--variants", required=True)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=float, default=840.0,
                        help="no new round starts after this many seconds of wall time")
    arguments = parser.parse_args()
    started = time.time()
    import jax

    counter = CaptureCounter()
    runner, state, _ = build_trainer(arguments.copies)
    jax.block_until_ready(state)
    record = {"record": "RK4 AntMaze speed ideas", "copies": arguments.copies, "environments_per_policy": 16,
              "machine": machine(), "build_and_prime_seconds": time.time() - started, "variants": {}}
    say(f"built and primed {arguments.copies} policies in {record['build_and_prime_seconds']:.1f} s")

    # GOAL: every program with its own copy of the state, compiled and warmed by two calls
    lr = lr_of(runner, 1)
    names = ["unit's program"] + arguments.variants.split(",")
    programs, states = {}, {}
    for name in names:
        program = runner.iterate if name == "unit's program" else variant_program(runner, name)
        own = jax.tree.map(jax.numpy.copy, state)
        calls = []
        for _ in range(2):
            start = time.perf_counter()
            own = program(own, lr)[0]
            jax.block_until_ready(own)
            calls.append(time.perf_counter() - start)
        programs[name], states[name] = program, own
        record["variants"][name] = {"first_call_seconds": calls[0], "second_call_seconds": calls[1],
                                    "rounds_seconds": [], "graph_captures_during_timed_rounds": 0}
        say(f"compiled {name}: first call {calls[0]:.1f} s, second {calls[1]:.2f} s")
        write_record(arguments.out, record)

    # GOAL: alternating rounds of chained calls, order reversed every second round
    per_round = max(2, round(3.0 / record["variants"]["unit's program"]["second_call_seconds"]))
    record["chained_calls_per_round"] = per_round
    for r in range(arguments.rounds):
        if time.time() - started > arguments.budget_seconds:
            say(f"budget reached after {r} rounds")
            break
        for name in (names if r % 2 == 0 else names[::-1]):
            before = counter.count
            step = lambda s, program=programs[name]: program(s, lr)[0]
            seconds, states[name] = chained_seconds(step, states[name], per_round)
            record["variants"][name]["rounds_seconds"].append(seconds)
            record["variants"][name]["graph_captures_during_timed_rounds"] += counter.count - before
        write_record(arguments.out, record)

    # GOAL: the throughput and the paired gain of every variant
    reference = record["variants"]["unit's program"]["rounds_seconds"]
    for name in names:
        entry = record["variants"][name]
        entry["throughput"] = throughput(arguments.copies, 16, runner.cfg.num_steps, median(entry["rounds_seconds"]))
        entry["gain"] = median([ref / own for ref, own in zip(reference, entry["rounds_seconds"])])
        say(f"{name}: {median(entry['rounds_seconds']):.4f} s per iteration, gain {entry['gain']:.3f}, "
            f"{entry['graph_captures_during_timed_rounds']} captures while timed")
    record["wall_seconds"] = time.time() - started
    write_record(arguments.out, record)


if __name__ == "__main__":
    main()
