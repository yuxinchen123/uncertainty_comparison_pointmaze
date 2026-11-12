"""On one card, compare the training state a changed program reaches with the reference's.

For a key whose results may differ from the reference only by rounding (risk class "dist"), this
is the check on the card itself, beside the leaf-by-leaf check on the processor: every arm is the
PointMaze study's learner (see `study_learner.py`) with some trainer keys replaced, all arms start
from the same seed and run the same number of iterations, and each arm's final state is compared
with the first arm's, array by array.

The first arm is the reference. An arm named "reference built again" with no keys shows what two
builds of the same program differ by on this card, which is the size a rounding difference has
here. Reported per arm: how many state arrays are bit-identical to the reference's, and the
largest difference among the others, |a - b| / (1 + |a|).

Usage (inside a Slurm job on one card):
  python compare_study_states_on_card.py --copies 64 --envs-per-copy 4 --iterations 10 \
      --arm "reference" --arm "reference built again" \
      --arm "target features once per iteration:target_features_once_per_iteration=True" \
      --output <file.json>
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_study_arms import arm_configuration, parse_arm  # noqa: E402
from study_learner import machine_record  # noqa: E402


def final_state(copies: int, envs_per_copy: int, overrides: dict, iterations: int) -> dict:
    """Train one arm from seed 0 for `iterations` iterations; every state array by its path."""
    import jax
    import numpy as np
    from exploration_platform.training.runner import Runner

    cfg, env_cfg = arm_configuration(copies, envs_per_copy, overrides)
    runner = Runner(cfg, env_cfg=env_cfg)
    state = runner.prime(runner.init_state(run_seed=0))
    for k in range(1, iterations + 1):
        state, _ = runner.iterate(state, runner.lr_argument(k, iterations))
    flat, _ = jax.tree_util.tree_flatten_with_path(state)
    return {jax.tree_util.keystr(path): np.asarray(leaf) for path, leaf in flat}


def compare(reference: dict, other: dict) -> dict:
    """How two final states differ: identical arrays, and the largest difference of the others.

    before: two dictionaries of arrays with the same names
    after:  {"arrays": 87, "bit_identical": 20, "largest_difference": 3.1e-06,
             "largest_difference_in": ".opt.m[...]", "non_float_arrays_differing": []}
    """
    import numpy as np
    if reference.keys() != other.keys():
        raise ValueError("the two programs do not hold the same state arrays")
    identical, worst, worst_name, non_float = 0, 0.0, "", []
    for name, a in reference.items():
        b = other[name]
        if a.tobytes() == b.tobytes():
            identical += 1
        elif a.dtype.kind != "f":
            non_float.append(name)
        else:
            difference = float(np.max(np.abs(a - b) / (1.0 + np.abs(a))))
            if difference > worst:
                worst, worst_name = difference, name
    return {"arrays": len(reference), "bit_identical": identical, "largest_difference": worst,
            "largest_difference_in": worst_name, "non_float_arrays_differing": non_float}


def main():
    """Run every arm, compare each with the first, write the record."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--copies", type=int, default=64)
    ap.add_argument("--envs-per-copy", type=int, default=4)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--arm", action="append", required=True,
                    help="name[:key=value,...]; the first arm is the reference")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    started = time.time()

    arms = [parse_arm(text) for text in args.arm]
    record = {"record": "final states of changed programs against the reference, on one card",
              "environment": "pointmaze-large", "copies": args.copies,
              "envs_per_copy": args.envs_per_copy, "iterations": args.iterations,
              "reference": arms[0][0], "machine": machine_record(), "arms": {}}
    reference = final_state(args.copies, args.envs_per_copy, arms[0][1], args.iterations)
    for name, overrides in arms[1:]:
        result = compare(reference, final_state(args.copies, args.envs_per_copy, overrides,
                                                args.iterations))
        record["arms"][name] = {"overrides": {k: repr(v) for k, v in overrides.items()}, **result}
        print(f"[{time.time() - started:6.1f} s] {name}: {result['bit_identical']} of "
              f"{result['arrays']} arrays bit-identical; largest difference of the others "
              f"{result['largest_difference']:.3e} in {result['largest_difference_in'] or '-'}; "
              f"non-float arrays differing: {result['non_float_arrays_differing'] or 'none'}",
              flush=True)
        args.output.write_text(json.dumps(record, indent=1))
    record["wall_seconds"] = time.time() - started
    args.output.write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
