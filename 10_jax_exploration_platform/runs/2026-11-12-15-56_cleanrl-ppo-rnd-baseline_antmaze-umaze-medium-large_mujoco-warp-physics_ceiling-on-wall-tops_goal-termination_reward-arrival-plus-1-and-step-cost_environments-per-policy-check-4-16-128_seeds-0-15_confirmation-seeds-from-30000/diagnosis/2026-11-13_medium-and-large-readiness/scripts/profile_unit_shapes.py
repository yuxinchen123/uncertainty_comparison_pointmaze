"""Seconds per training iteration, card memory and Warp pool use of AntMaze units, one shape per process.

A shape is (configuration file, environments per policy, policies). Each shape runs in its own process
(this script with --only-copies), so the card's memory reading belongs to that shape alone. A shape
builds the trainer exactly as scripts/run_training.py does (through device/series.py), primes it,
calls the compiled iteration twice to compile it, times rounds of about two seconds, and then reads:

  1. the median seconds per iteration of the rounds;
  2. JAX's peak bytes in use, and the largest card memory nvidia-smi read while the process ran
     (sampled every second; the job runs with XLA_PYTHON_CLIENT_PREALLOCATE=false, so the reading is
     what the process holds, MuJoCo Warp's own buffers included);
  3. MuJoCo Warp's pool report (`WarpAntMaze.overflow_report`): the most contacts and candidate pairs in
     the shared pool, the most constraint rows of one environment, and whether anything was dropped.

The policies are untrained (random-like torques). Run on a compute node with one card:
  python profile_unit_shapes.py --configuration-file F --map medium --environments-per-policy 16 \
      --copies 64,128 --out records/<name>.jsonl
"""
import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

PLATFORM = Path("/p/rlprojects/RND/10_jax_exploration_platform")
sys.path.insert(0, str(PLATFORM / "device"))
sys.path.insert(0, str(PLATFORM / "scripts"))
sys.path.insert(0, str(PLATFORM / "src"))
import series  # noqa: E402  (the doubling-series runner: its trainer command line and spread)

TARGET_ROUND_SECONDS = 2.0
MIN_ROUNDS, MAX_ROUNDS = 5, 12


def parse_arguments(argv=None):
    """The shapes to measure and where to write them."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configuration-file", type=Path, required=True)
    parser.add_argument("--map", required=True)
    parser.add_argument("--environments-per-policy", type=int, required=True)
    parser.add_argument("--copies", required=True, help="comma-separated policy counts, e.g. 64,128")
    parser.add_argument("--out", type=Path, required=True, help="a .jsonl file; one line per shape is appended")
    parser.add_argument("--timing-seconds", type=float, default=30.0)
    parser.add_argument("--prime-iterations", type=int, default=2,
                        help="priming sets only the observation statistics; 2 saves time and does not change the rate")
    parser.add_argument("--budget-minutes", type=float, default=15.0,
                        help="no new shape starts after this much wall time (the 15-minute rule per idea per card)")
    parser.add_argument("--only-copies", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--point-out", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def series_arguments(arguments):
    """The argument object device/series.py's trainer_command_line reads."""
    return argparse.Namespace(configuration_file=arguments.configuration_file, environment_family="antmaze",
                              out_dir=arguments.out.parent, environments_per_policy=arguments.environments_per_policy,
                              update_style="epoch_minibatch", bonus="rnd_next_state",
                              trainer_argument=[f"--prime-iterations={arguments.prime_iterations}"])


def card_memory_sampler(stop, readings):
    """Append nvidia-smi's memory-used reading of every visible card, in MiB, once a second until stopped."""
    while not stop.is_set():
        output = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, check=True).stdout
        readings.append(max(int(line) for line in output.split()))
        stop.wait(1.0)


def measure(arguments):
    """One shape in this process: build, prime, compile, time; returns its record."""
    import jax
    import run_training
    from exploration_platform.training.runner import Runner

    # GOAL: read the card's memory for the whole life of the shape, construction included.
    stop, readings = threading.Event(), []
    sampler = threading.Thread(target=card_memory_sampler, args=(stop, readings), daemon=True)
    sampler.start()

    # GOAL: the trainer and environment exactly as the training command builds them from the file.
    options = series.trainer_command_line(series_arguments(arguments), arguments.only_copies)
    trainer_arguments = run_training.parse_args(options)
    config = run_training.build_config(trainer_arguments)
    environment = run_training.build_env_config(trainer_arguments)
    if environment.map_name != arguments.map or environment.physics != "warp":
        raise ValueError(f"the file builds {environment.map_name!r} with {environment.physics!r} physics")
    device = jax.local_devices()[0]
    if device.platform != "gpu":
        raise RuntimeError(f"the shapes must run on a card, got {device}")
    start = time.perf_counter()
    runner = Runner(config, bonus=trainer_arguments.bonus, env_cfg=environment)
    config = runner.cfg
    state = runner.prime(runner.init_state(run_seed=0))
    jax.block_until_ready(state)
    build_seconds = time.perf_counter() - start

    def one_iteration(state, iteration):
        """One compiled training iteration at a learning rate that does not move."""
        return runner.iterate(state, runner.lr_argument(iteration, series.ANNEAL_ITERATIONS))

    # GOAL: two warm-up calls (compile, then compile again for the donated state), then timed rounds.
    warmup = []
    for iteration in (1, 2):
        start = time.perf_counter()
        state, metrics = one_iteration(state, iteration)
        jax.block_until_ready((state, metrics))
        warmup.append(time.perf_counter() - start)
    per_round = max(1, round(TARGET_ROUND_SECONDS / warmup[1]))
    deadline, rounds, iteration = time.perf_counter() + arguments.timing_seconds, [], 3
    while len(rounds) < MAX_ROUNDS:
        start = time.perf_counter()
        for _ in range(per_round):
            state, metrics = one_iteration(state, iteration)
            iteration += 1
        jax.block_until_ready((state, metrics))
        rounds.append((time.perf_counter() - start) / per_round)
        if len(rounds) >= MIN_ROUNDS and (series.spread_of(rounds) <= series.STABLE_SPREAD
                                          or time.perf_counter() > deadline):
            break

    # GOAL: the record of the shape: the five throughput quantities, memory, and the pool report.
    stop.set()
    sampler.join()
    seconds = statistics.median(rounds)
    steps_per_copy = config.num_steps * config.n_envs
    report = runner.env.overflow_report(state.env_state)
    memory = device.memory_stats() or {}
    return {
        "map": environment.map_name, "configuration_file": str(arguments.configuration_file),
        "copies": config.n_copies, "environments_per_copy": config.n_envs, "rollout_steps": config.num_steps,
        "environments": config.n_copies * config.n_envs,
        "seconds_per_iteration": seconds,
        "total_steps_per_second_millions": config.n_copies * steps_per_copy / seconds / 1e6,
        "steps_per_second_per_copy_thousands": steps_per_copy / seconds / 1e3,
        "hours_per_million_steps_per_copy": 1e6 / (3600 * steps_per_copy / seconds),
        "card_memory_used_mb_largest_reading": max(readings) if readings else None,
        "jax_peak_bytes_in_use_mb": memory.get("peak_bytes_in_use", 0) / 2 ** 20,
        "pool": report,
        "contacts_per_environment_largest": report["nacon_peak"] / (config.n_copies * config.n_envs),
        "candidate_pairs_per_environment_largest": report["ncollision_peak"] / (config.n_copies * config.n_envs),
        "iterations_timed": len(rounds) * per_round, "rounds": rounds,
        "relative_spread_middle_half": series.spread_of(rounds) if len(rounds) >= 4 else None,
        "build_and_prime_seconds": build_seconds, "warmup_call_seconds": warmup,
        "prime_iterations": config.prime_iterations, "physics_time_step": environment.physics_time_step,
        "frame_skip": environment.frame_skip, "max_episode_steps": environment.max_episode_steps,
        "device_name": device.device_kind, "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "written_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def main():
    """Every requested shape in its own process, one record line each; or one shape when started by itself."""
    arguments = parse_arguments()
    if arguments.only_copies is not None:
        arguments.point_out.write_text(json.dumps(measure(arguments)))
        return
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.perf_counter() + 60 * arguments.budget_minutes
    for copies in (int(text) for text in arguments.copies.split(",")):
        if time.perf_counter() > deadline:
            print(f"budget of {arguments.budget_minutes} minutes used: {copies} copies not started", flush=True)
            break
        # GOAL: one shape per process; a failed shape is recorded with the end of its error output.
        point = arguments.out.with_name(f"{arguments.out.stem}_copies-{copies}.json")
        command = [sys.executable, "-u", __file__, *sys.argv[1:], "--only-copies", str(copies), "--point-out", str(point)]
        started = time.perf_counter()
        process = subprocess.run(command, capture_output=True, text=True)
        if process.returncode == 0:
            record = json.loads(point.read_text())
            point.unlink()
        else:
            record = {"map": arguments.map, "copies": copies, "environments_per_copy": arguments.environments_per_policy,
                      "failed": True, "returncode": process.returncode, "stderr_tail": process.stderr[-3000:]}
        record["wall_seconds"] = time.perf_counter() - started
        with arguments.out.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps({key: record.get(key) for key in (
            "map", "copies", "environments_per_copy", "seconds_per_iteration", "card_memory_used_mb_largest_reading",
            "contacts_per_environment_largest", "candidate_pairs_per_environment_largest", "failed", "wall_seconds")}),
            flush=True)
        if process.returncode != 0:
            print(process.stderr[-3000:], flush=True)


if __name__ == "__main__":
    main()
