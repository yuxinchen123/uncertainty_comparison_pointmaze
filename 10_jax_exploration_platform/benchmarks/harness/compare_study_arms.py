"""Time a reference program and one or more changed programs of the PointMaze study's learner.

Every arm is the study's learner (see `study_learner.py`) with some trainer keys replaced. For
each rollout shape, all arms are built in this one process, each is warmed up with two iterations
(the second call, on donated buffers, compiles once more), and then they are timed in alternating
rounds: a round times one block of chained iterations per arm, with one wait for the card at the
end of the block, as a training loop runs; the order of the arms is reversed on every second
round. The first arm is the reference. Each round gives one paired ratio per arm,

    ratio = (reference seconds per iteration) / (arm seconds per iteration),

which is also the arm's total steps per second relative to the reference, since both hold the same
number of policies and environments. A drift of the machine moves both arms of a round alike, so
the paired ratios are steadier than the two medians.

The whole unit stays inside its time budget: the shapes share it equally, an arm that cannot be
built in its shape's share is recorded as not measured, and fewer than five rounds is recorded as
inconclusive. The record is rewritten after every round.

An arm is written `name` or `name:key=value,key=value`. The key `sweep_cells=<n>` (1, 8 or 60)
splits the policies into that many cells with different learning rate, entropy coefficient and
intrinsic weight. The key `iterations_per_host_call=<k>` runs k iterations inside one compiled
loop per call (`training/blocks.py`); its time is divided by k. Every other key is a trainer
field. `--start-rounds-at <epoch seconds>` makes the rounds of the first shape begin at a given
time, so that two processes sharing a card time their rounds together.

Usage (inside a Slurm job on one card):
  python compare_study_arms.py --idea 18_gather-minibatches-per-epoch --shapes 16:256,128:64 \
      --arm "gathered before the first step:gather_minibatches_per_epoch=False" \
      --arm "gathered one epoch at a time:gather_minibatches_per_epoch=True" --output <file.json>
"""
import argparse
import ast
import gc
import json
import sys
import time
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from study_learner import (block_seconds, machine_record, median, study_configuration,  # noqa: E402
                           throughput_columns)

# the cells of the study's first sweep pass: 3 learning rates x 2 entropy coefficients x 10
# intrinsic weights (factor 4 from 4^-6 to 4^3); the 8-cell program takes 2 x 2 x 2 of them
SWEEP_LISTS = {
    8: ((1e-4, 3e-4), (0.001, 0.01), (0.25, 1.0)),
    60: ((1e-4, 3e-4, 1e-3), (0.001, 0.01), tuple(4.0 ** k for k in range(-6, 4))),
}


def parse_arm(text: str):
    """One arm as (name, {trainer key: value}).

    before: "32-bit batch statistics:batch_stats_f32=True,scan_unroll=8"
    after:  ("32-bit batch statistics", {"batch_stats_f32": True, "scan_unroll": 8})
    """
    name, _, settings = text.partition(":")
    overrides = {}
    for item in filter(None, settings.split(",")):
        key, _, value = item.partition("=")
        try:
            overrides[key.strip()] = ast.literal_eval(value.strip())
        except (ValueError, SyntaxError):
            overrides[key.strip()] = value.strip()
    return name.strip(), overrides


def arm_configuration(copies: int, envs_per_copy: int, overrides: dict):
    """The study's learner with one arm's keys, as (PPOConfig, EnvConfig).

    before: copies 240, {"sweep_cells": 8}
    after:  8 cells of 30 policies, each cell its own (learning rate, entropy, intrinsic weight)
    """
    overrides = dict(overrides)
    overrides.pop("iterations_per_host_call", None)
    cells = overrides.pop("sweep_cells", 1)
    if cells != 1:
        if cells not in SWEEP_LISTS or copies % cells:
            raise ValueError(f"sweep_cells must be 1, 8 or 60 and divide the {copies} policies; "
                             f"got {cells}")
        overrides.update(setting_names=("learning_rate", "ent_coef", "int_coef"),
                         parameter_settings=tuple(product(*SWEEP_LISTS[cells])),
                         copies_per_group=copies // cells)
    return study_configuration(copies, envs_per_copy, **overrides)


def measure_shape(copies: int, envs_per_copy: int, arms: list, deadline: float, min_rounds: int,
                  max_rounds: int, on_round, start_rounds_at: float = 0.0) -> dict:
    """Build every arm at one shape, then time them in alternating rounds until the deadline.

    on_round(result) is called after the build and after every round with the shape's result so
    far, so the caller can write it out. The rounds begin no earlier than `start_rounds_at`
    (seconds since the epoch).
    """
    import jax
    import jax.numpy as jnp
    from exploration_platform.training.blocks import iteration_block
    from exploration_platform.training.runner import Runner

    result = {"copies": copies, "envs_per_copy": envs_per_copy, "arms": {}, "not_measured": [],
              "rounds": 0}
    built = {}
    shape_started = time.time()
    for index, (name, overrides) in enumerate(arms):
        # a third arm or later is built only while a third of the shape's time is still left
        t0 = time.time()
        if index >= 2 and t0 > shape_started + (deadline - shape_started) * 2 / 3:
            result["not_measured"].append(name)
            continue
        cfg, env_cfg = arm_configuration(copies, envs_per_copy, overrides)
        runner = Runner(cfg, env_cfg=env_cfg)
        lr = runner.lr_argument(1, 1)
        state = runner.prime(runner.init_state(run_seed=0))
        # one call runs one iteration, or k iterations inside one compiled loop
        per_call = int(overrides.get("iterations_per_host_call", 1))
        if per_call == 1:
            step = lambda s, runner=runner, lr=lr: runner.iterate(s, lr)[0]
        else:
            block, rates = iteration_block(runner, per_call), jnp.full((per_call,), lr)
            step = lambda s, block=block, rates=rates: block(s, rates)[0]
        for _ in range(2):
            state = step(state)
            jax.block_until_ready(state)
        one_call, state = block_seconds(step, state, 3)
        one_call /= per_call
        built[name] = [step, state, per_call]
        result["arms"][name] = {"overrides": {k: repr(v) for k, v in overrides.items()},
                                "seconds_to_build_and_warm_up": time.time() - t0,
                                "seconds_per_iteration_rounds": []}
        print(f"  built {name!r} in {time.time() - t0:.1f} s; first timed iterations "
              f"{one_call * 1e3:.2f} ms", flush=True)
    on_round(result)

    # alternating rounds: each block lasts about one second, at least three iterations
    names = list(built)
    reference = names[0]
    one_call, built[reference][1] = block_seconds(built[reference][0], built[reference][1], 3)
    block = max(3, int(1.0 / (one_call / built[reference][2])))
    result["iterations_per_block"] = block
    time.sleep(max(0.0, start_rounds_at - time.time()))
    result["rounds_started_epoch"] = time.time()
    while result["rounds"] < max_rounds and (
            time.time() < deadline or result["rounds"] < min(min_rounds, 3)):
        for name in (names if result["rounds"] % 2 == 0 else names[::-1]):
            step, state, per_call = built[name]
            calls = max(1, round(block / per_call))
            seconds, built[name][1] = block_seconds(step, state, calls)
            result["arms"][name]["seconds_per_iteration_rounds"].append(seconds / per_call)
        result["rounds"] += 1
        on_round(result)

    # medians, the five throughput columns, and the paired ratios against the reference
    T = study_configuration(copies, envs_per_copy)[0].num_steps
    reference_rounds = result["arms"][reference]["seconds_per_iteration_rounds"]
    for name in names:
        arm = result["arms"][name]
        rounds = arm["seconds_per_iteration_rounds"]
        arm["throughput"] = throughput_columns(copies, envs_per_copy, T, median(rounds))
        ratios = [r / a for r, a in zip(reference_rounds, rounds)]
        arm["paired_ratio_median"] = median(ratios)
        arm["paired_ratio_least"], arm["paired_ratio_most"] = min(ratios), max(ratios)
        arm["rounds_faster_than_reference"] = sum(ratio > 1.0 for ratio in ratios)
    result["conclusive"] = result["rounds"] >= min_rounds
    result["peak_card_bytes_all_arms"] = (jax.devices()[0].memory_stats() or {}).get(
        "peak_bytes_in_use")
    on_round(result)
    built.clear()
    gc.collect()
    return result


def main():
    """Measure every shape inside one time budget and write the record."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--idea", required=True, help="the idea's number and name, for the record")
    ap.add_argument("--shapes", default="16:256,128:64",
                    help="comma-separated <environments per policy>:<policies>")
    ap.add_argument("--arm", action="append", required=True,
                    help="name[:key=value,...]; the first arm is the reference")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--budget-seconds", type=float, default=810.0)
    ap.add_argument("--min-rounds", type=int, default=5)
    ap.add_argument("--max-rounds", type=int, default=25)
    ap.add_argument("--start-rounds-at", type=float, default=0.0,
                    help="seconds since the epoch before which the first shape's rounds wait")
    args = ap.parse_args()
    started = time.time()

    arms = [parse_arm(text) for text in args.arm]
    shapes = [tuple(int(part) for part in shape.split(":")) for shape in args.shapes.split(",")]
    record = {"record": "paired timing of a reference and changed programs", "idea": args.idea,
              "environment": "pointmaze-large", "arms": [name for name, _ in arms],
              "reference": arms[0][0], "machine": machine_record(), "shapes": []}

    def write(partial=None):
        """Write the record with the finished shapes and the one in progress."""
        shown = dict(record, shapes=record["shapes"] + ([partial] if partial else []),
                     wall_seconds=time.time() - started)
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(shown, indent=1))
        temporary.replace(args.output)

    for index, (envs_per_copy, copies) in enumerate(shapes):
        # the shapes still to run share what is left of the budget equally
        left = started + args.budget_seconds - time.time()
        deadline = time.time() + left / (len(shapes) - index)
        print(f"[{time.time() - started:6.1f} s] {copies} policies x {envs_per_copy} "
              "environments", flush=True)
        result = measure_shape(copies, envs_per_copy, arms, deadline, args.min_rounds,
                               args.max_rounds, write,
                               args.start_rounds_at if index == 0 else 0.0)
        record["shapes"].append(result)
        for name, arm in result["arms"].items():
            print(f"  {arm['throughput']['seconds_per_iteration'] * 1e3:9.3f} ms  "
                  f"{arm['throughput']['total_steps_per_second_millions']:7.3f} M steps/s  "
                  f"ratio {arm['paired_ratio_median']:.3f} "
                  f"({arm['paired_ratio_least']:.3f} to {arm['paired_ratio_most']:.3f}, faster "
                  f"in {arm['rounds_faster_than_reference']} of {result['rounds']} rounds)  "
                  f"{name}", flush=True)
    write()
    print(f"[{time.time() - started:6.1f} s] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
