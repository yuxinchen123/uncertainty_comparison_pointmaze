"""Start-up seconds of one RK4 AntMaze process, with or without JAX's persistent compilation cache.

One process does what a unit does before its first window: imports, builds the trainer, primes it
(a fresh unit) or not (a resumed unit), and runs its first three iterations. Recorded: the seconds of
each part, the graphs Warp captured in each call, and the files in the cache folder before and after.
The unit runs this script several times in a row on one card: without the cache, without it but with
the state committed to the card before the first call (--commit-state: does the second call then still
compile?), then twice with a new cache folder (the first fills it, the second reads it), so the
difference of the first iterations between the runs is what each change saves on every restart.

  python startup.py --copies 128 --cache-dir /localtmp/<user>/jax_cache_test --prime --out records/x.jsonl
"""
import argparse
import json
import time

STARTED = time.time()

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))


def files_in(folder):
    """The number of files under a folder (0 when it does not exist)."""
    return sum(1 for path in Path(folder).rglob("*") if path.is_file()) if folder and Path(folder).exists() else 0


def main():
    """Measure one start-up and append its record."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--copies", type=int, required=True)
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--prime", action="store_true", help="prime as a fresh unit does (a resumed unit does not)")
    parser.add_argument("--commit-state", action="store_true",
                        help="put the state on the card (committed arrays) before the first iteration, so the "
                             "first call's arguments match the arrays every later call gets")
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    # GOAL: the cache is switched on before JAX compiles anything, as the training option does
    import jax
    if arguments.cache_dir:
        jax.config.update("jax_compilation_cache_dir", arguments.cache_dir)
    from common import CaptureCounter, lr_of, machine, unit_configuration
    from exploration_platform.training.runner import Runner
    counter = CaptureCounter()
    record = {"label": arguments.label, "copies": arguments.copies, "cache_dir": arguments.cache_dir,
              "cache_files_before": files_in(arguments.cache_dir), "primed": arguments.prime,
              "import_seconds": time.time() - STARTED}

    # GOAL: build (and prime) as the training command does
    start = time.time()
    trainer_arguments, config, environment = unit_configuration(arguments.copies, prime_iterations=2)
    runner = Runner(config, bonus=trainer_arguments.bonus, env_cfg=environment)
    state = runner.init_state(run_seed=0)
    jax.block_until_ready(state)
    record["build_seconds"] = time.time() - start
    if arguments.prime:
        start, before = time.time(), counter.count
        state = runner.prime(state)
        jax.block_until_ready(state)
        record["prime_seconds"], record["prime_graph_captures"] = time.time() - start, counter.count - before

    # GOAL: with --commit-state, the state as committed arrays on the card, as an iteration returns them
    record["state_committed_before_first_iteration"] = arguments.commit_state
    if arguments.commit_state:
        state = jax.device_put(state, jax.devices()[0])

    # GOAL: the first three iterations, each waited for
    record["iteration_call_seconds"], record["iteration_graph_captures"] = [], []
    for k in (1, 2, 3):
        start, before = time.time(), counter.count
        state, metrics = runner.iterate(state, lr_of(runner, k))
        jax.block_until_ready((state, metrics))
        record["iteration_call_seconds"].append(time.time() - start)
        record["iteration_graph_captures"].append(counter.count - before)
        # the number of compiled programs of the iteration so far (2 after the second call: it compiled again)
        record.setdefault("iteration_programs_compiled", []).append(runner.iterate._cache_size())
    record["total_seconds_to_third_iteration"] = time.time() - STARTED

    # GOAL: a window end as the training command makes it (the episode log's sums and the fault
    # record set back to zero outside the compiled program), then a fourth iteration: does the
    # changed state compile the program once more?
    from exploration_platform.training import episode_log
    state = state._replace(agent_state={**state.agent_state,
                                        "episode_log": episode_log.clear_window(state.agent_state["episode_log"])})
    state = state._replace(env_state=runner.env.fault_window(state.env_state)[1])
    if arguments.commit_state:
        state = jax.device_put(state, jax.devices()[0])
    start, before = time.time(), counter.count
    state, metrics = runner.iterate(state, lr_of(runner, 4))
    jax.block_until_ready((state, metrics))
    record["iteration_after_a_window_end_seconds"] = time.time() - start
    record["iteration_after_a_window_end_graph_captures"] = counter.count - before
    record["iteration_programs_compiled"].append(runner.iterate._cache_size())
    record["cache_files_after"] = files_in(arguments.cache_dir)
    record["loss_after_four_iterations"] = float(metrics["loss"])
    record["machine"] = machine()
    with arguments.out.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "machine"}), flush=True)


if __name__ == "__main__":
    main()
