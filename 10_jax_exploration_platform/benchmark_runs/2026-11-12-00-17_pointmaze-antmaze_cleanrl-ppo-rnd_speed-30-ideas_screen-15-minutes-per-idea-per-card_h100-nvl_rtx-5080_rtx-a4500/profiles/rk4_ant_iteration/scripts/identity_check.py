"""Is the training state after a few iterations identical, array by array, with each speed idea on and off?

Runs on a compute node's processors (MuJoCo Warp on its processor device, JAX on the processor backend),
where two runs of the same program are equal bit for bit except the 14 island arrays of the Warp data that
a physics step returns unwritten (`warp_antmaze.ISLAND_ARRAYS`, left out here). The trainer is the RK4 unit's
(common.unit_configuration) at a small shape: 2 policies x 2 environments, 32-step rollouts with the
rollout unroll given explicitly as 16 (the unit's value), one priming rollout. Every variant of
`ideas.py` starts from the same primed state and runs the same iterations; the reference program is built
twice, so the check also shows that the processor run repeats itself.

  python identity_check.py --variants frame-unroll-5,rollout-unroll-1 --iterations 3 --out records/x.json
"""
import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import lr_of, machine, say, unit_configuration, write_record  # noqa: E402
from ideas import variant_program  # noqa: E402


def arrays(tree) -> dict:
    """Every array of a tree by its path, the island arrays of the Warp data left out."""
    import jax
    import numpy as np
    from exploration_platform.envs.antmaze.warp_antmaze import ISLAND_ARRAYS
    flat, _ = jax.tree_util.tree_flatten_with_path(tree)
    named = {jax.tree_util.keystr(path): np.asarray(leaf) for path, leaf in flat}
    return {name: value for name, value in named.items() if not any(part in name for part in ISLAND_ARRAYS)}


def compare(reference: dict, other: dict) -> dict:
    """How many arrays are equal bit for bit, and the largest difference of the others."""
    import numpy as np
    differing = {name: float(np.max(np.abs(reference[name].astype(np.float64) - other[name].astype(np.float64))))
                 for name in reference if reference[name].tobytes() != other[name].tobytes()}
    return {"arrays": len(reference), "bit_identical": len(reference) - len(differing),
            "differing": dict(sorted(differing.items(), key=lambda item: -item[1])[:20])}


def main():
    """Run the reference twice and every variant from the same primed state; write the comparison."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", required=True)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    started = time.time()
    import jax
    from exploration_platform.training.runner import Runner
    from exploration_platform.training.train_step import build_iteration

    # GOAL: the unit's trainer at a small shape, primed once
    trainer_arguments, config, environment = unit_configuration(2, 2, prime_iterations=1)
    config = replace(config, num_steps=32, scan_unroll=16)
    runner = Runner(config, bonus=trainer_arguments.bonus, env_cfg=environment)
    primed = runner.prime(runner.init_state(run_seed=5))
    jax.block_until_ready(primed)
    say(f"built and primed on {jax.devices()[0]} in {time.time() - started:.0f} s")

    # GOAL: every program from its own copy of the primed state, the same iterations
    def run(program):
        """The state's arrays and the last iteration's metrics after the iterations."""
        state = jax.tree.map(jax.numpy.copy, primed)
        for k in range(1, arguments.iterations + 1):
            state, metrics = program(state, lr_of(runner, k))
        return arrays(state), arrays(metrics)

    record = {"record": "RK4 AntMaze identity of speed ideas on processors", "iterations": arguments.iterations,
              "shape": "2 policies x 2 environments, 32-step rollouts, reference rollout unroll 16",
              "machine": machine(), "results": {}}
    reference_state, reference_metrics = run(runner.iterate)
    names = ["unit's program, built again"] + arguments.variants.split(",")
    for name in names:
        start = time.time()
        program = (jax.jit(build_iteration(runner.cfg, runner.env, runner.bonus, runner.sweep, runner.layout),
                           donate_argnums=(0,))
                   if name == "unit's program, built again" else variant_program(runner, name))
        state, metrics = run(program)
        result = {"state": compare(reference_state, state), "metrics": compare(reference_metrics, metrics),
                  "seconds": time.time() - start}
        result["identical"] = (result["state"]["bit_identical"] == result["state"]["arrays"]
                               and result["metrics"]["bit_identical"] == result["metrics"]["arrays"])
        record["results"][name] = result
        say(f"{name}: state {result['state']['bit_identical']} of {result['state']['arrays']} arrays bit-identical, "
            f"metrics {result['metrics']['bit_identical']} of {result['metrics']['arrays']} "
            f"({result['seconds']:.0f} s)")
        if not result["identical"]:
            say("  differing: " + json.dumps(result["state"]["differing"])[:600])
        write_record(arguments.out, record)
    record["wall_seconds"] = time.time() - started
    write_record(arguments.out, record)


if __name__ == "__main__":
    main()
