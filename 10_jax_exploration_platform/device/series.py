"""Doubling series: seconds per iteration and peak card memory of one workload at 64, 128, 256, ... copies.

Run it on a compute node (or serval05 / serval02 card 1) that holds ONE card, never on a login
node. It follows the shared skill `/p/rlprojects/.claude/skills/estimate-copies-per-gpu/` (step 2)
and writes one record per series into `device/probes/`, in the format that skill's `capacity.py`
reads.

The series starts at 64 copies and doubles until one of these stops it:

  1. a doubling raised total steps per second by under 10%;
  2. the card ran out of memory;
  3. the time budget (15 minutes, start-up and compilation included) is used;
  4. the largest allowed copy count is reached, or a size failed for another reason.

Each size runs in its own process (this same script with `--only-copies`), so a size that runs out
of memory loses only itself. A size is timed after two warm-up calls of the compiled iteration
(the first compiles the program, the second compiles it again for donated buffers), in rounds of
about two seconds; the reported time is the median round.

The trainer configuration comes from one recorded file under `configs/experiments/`; the copies,
the environments per policy, the update style and the bonus come from this script's arguments.
Every other trainer option goes through `--trainer-argument`, unchanged, to `scripts/run_training.py`.

Example (one line):
  PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python device/series.py
    --configuration-file configs/experiments/pointmaze_large_current_returns_random_network_distillation_steps-100-million.toml
    --environment-family pointmaze --map large --environments-per-policy 4
    --update-style full_batch --bonus rnd_next_state --card "H100 NVL" --out-dir device/probes
"""
import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PLATFORM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLATFORM / "scripts"))

PACIFIC = ZoneInfo("America/Los_Angeles")
GAIN = 0.10                   # a doubling must raise total steps per second by at least this share
TARGET_ROUND_SECONDS = 2.0    # iterations per round are chosen to fill about this much time
MIN_ROUNDS = 6                # rounds before a size may be called settled
MIN_ROUNDS_WHEN_LATE = 3      # rounds a size still takes once its own timing budget is used
MAX_ROUNDS = 24
STABLE_SPREAD = 0.02          # settled: the middle half of the rounds lies within 2% of the median
ANNEAL_ITERATIONS = 10 ** 6   # a learning-rate horizon so long that the rate does not move in a series
ENVIRONMENT_FAMILIES = {"EnvConfig": "pointmaze", "AntMazeConfig": "antmaze"}
# the phrases the runtime prints when the card's memory is the cause (the survey's list)
OUT_OF_MEMORY_MARKERS = ("RESOURCE_EXHAUSTED", "Out of memory", "out of memory",
                         "OUT_OF_MEMORY", "CUDA_ERROR_OUT_OF_MEMORY")


def parse_arguments(argv=None):
    """The workload, the card's name, and the limits of the series."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configuration-file", type=Path, required=True,
                        help="a recorded configuration under configs/experiments/")
    parser.add_argument("--environment-family", required=True, choices=sorted(ENVIRONMENT_FAMILIES.values()))
    parser.add_argument("--map", required=True, help="the maze map, as the environment configuration names it")
    parser.add_argument("--environments-per-policy", type=int, required=True)
    parser.add_argument("--update-style", required=True, choices=["full_batch", "epoch_minibatch"])
    parser.add_argument("--bonus", required=True, help="the bonus family, as bonuses/registry.py names it")
    parser.add_argument("--trainer-argument", action="append", default=[],
                        help="one more scripts/run_training.py option, e.g. --trainer-argument=--map=medium; "
                             "repeatable; needs --workload, because it changes the workload")
    parser.add_argument("--workload", help="the workload's name in words; default: built from the four arguments above")
    parser.add_argument("--card", required=True, help="the card's name as card_facts.json of the skill has it")
    parser.add_argument("--out-dir", type=Path, required=True, help="device/probes of this platform")
    parser.add_argument("--first-copies", type=int, default=64)
    parser.add_argument("--largest-copies", type=int, default=65536)
    parser.add_argument("--budget-minutes", type=float, default=15.0, help="wall time of the whole series")
    parser.add_argument("--timing-seconds", type=float, default=60.0, help="timing budget of one size, after its warm-up")
    # set by the series itself when it starts one size as its own process
    parser.add_argument("--only-copies", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--point-out", type=Path, help=argparse.SUPPRESS)
    arguments = parser.parse_args(argv)
    if arguments.trainer_argument and not arguments.workload:
        parser.error("--trainer-argument changes the workload: name it with --workload")
    return arguments


def workload_name(arguments):
    """The workload in words. after: pointmaze-large_environments-per-policy-4_update-full-batch_bonus-rnd-next-state"""
    if arguments.workload:
        return arguments.workload
    return (f"{arguments.environment_family}-{arguments.map}"
            f"_environments-per-policy-{arguments.environments_per_policy}"
            f"_update-{arguments.update_style.replace('_', '-')}"
            f"_bonus-{arguments.bonus.replace('_', '-')}")


def trainer_command_line(arguments, copies):
    """The scripts/run_training.py options of one size.

    before: 256 copies, a configuration file that lists 1 explicit (rate, weight) setting
    after:  [..., "--copies", "256", "--copies-per-cell", "256", "--envs-per-copy", "4", ...]
    """
    from experiment_configuration import read_experiment_configuration
    document = read_experiment_configuration(arguments.configuration_file)
    if document["metadata"]["environment_family"] != arguments.environment_family:
        raise ValueError(f"{arguments.configuration_file} records the family "
                         f"{document['metadata']['environment_family']!r}, not {arguments.environment_family!r}")
    # CHECK WHEN THE TRAINER CHANGES OF PLAN SECTION 3.1 ARE IN: every option name in this list is
    # run_training.py's own (--copies, --copies-per-cell, --envs-per-copy, --update-style, --bonus).
    # The run folder and unit name are required by its parser and are not used: nothing is written there.
    options = ["--run-dir", str(arguments.out_dir), "--unit-id", "doubling-series",
               "--configuration-file", str(arguments.configuration_file),
               "--copies", str(copies),
               "--envs-per-copy", str(arguments.environments_per_policy),
               "--update-style", arguments.update_style,
               "--bonus", arguments.bonus]
    # CHECK: a file with explicit (rate, weight) settings sizes the process as settings x copies per
    # cell, so the series divides its copies over the file's settings; the per-copy settings of plan
    # section 3.1 (setting_names) may change how a file lists its cells.
    cells = len(document["trainer"]["parameter_settings"])
    if cells:
        if copies % cells:
            raise ValueError(f"{copies} copies cannot be divided over the file's {cells} settings")
        options += ["--copies-per-cell", str(copies // cells)]
    return options + list(arguments.trainer_argument)


def spread_of(round_seconds):
    """Relative width of the middle half of the per-iteration times of the rounds.

    before: [0.402, 0.399, 0.405, 0.401, 0.400, 0.398] ; after: about 0.01 (the middle half lies
            within 1% of the median, so the size is called settled)
    """
    first_quarter, _, third_quarter = statistics.quantiles(round_seconds, n=4)
    return (third_quarter - first_quarter) / statistics.median(round_seconds)


def measure(arguments):
    """One size, in its own process: build the trainer, warm up twice, time rounds; returns the point."""
    import jax
    import run_training
    from exploration_platform.training.runner import Runner

    # CHECK WHEN THE TRAINER CHANGES ARE IN: parse_args, build_config and build_env_config are
    # run_training.py's own construction of the configuration, so new options and presets are
    # followed without a change here, as long as these three names and their arguments stay.
    trainer_arguments = run_training.parse_args(trainer_command_line(arguments, arguments.only_copies))
    config = run_training.build_config(trainer_arguments)
    environment = run_training.build_env_config(trainer_arguments)
    family = ENVIRONMENT_FAMILIES.get(type(environment).__name__)
    if (family, environment.map_name) != (arguments.environment_family, arguments.map):
        raise ValueError(f"the configuration builds {family} {environment.map_name!r}, "
                         f"not {arguments.environment_family} {arguments.map!r}")
    device = jax.local_devices()[0]
    if device.platform != "gpu":
        raise RuntimeError(f"a series must run on a graphics card, got {device}")

    # the trainer and its first state, as scripts/run_training.py builds them
    # CHECK: Runner(config, bonus=..., env_cfg=...), init_state, prime, iterate and lr_argument
    build_start = time.perf_counter()
    runner = Runner(config, bonus=trainer_arguments.bonus, env_cfg=environment)
    config = runner.cfg
    state = runner.prime(runner.init_state(run_seed=0))
    jax.block_until_ready(state)
    build_seconds = time.perf_counter() - build_start

    def one_iteration(state, iteration):
        """One compiled training iteration at a learning rate that does not move."""
        return runner.iterate(state, runner.lr_argument(iteration, ANNEAL_ITERATIONS))

    # two warm-up calls, timed apart: the first compiles the program, the second compiles it again
    # for the donated state, and neither belongs to the rate
    warmup_seconds = []
    for iteration in (1, 2):
        start = time.perf_counter()
        state, metrics = one_iteration(state, iteration)
        jax.block_until_ready((state, metrics))
        warmup_seconds.append(time.perf_counter() - start)

    # rounds of about two seconds, until the middle half agrees within 2% or the size's budget is used
    # before: second warm-up call 0.04 s ; after: 50 iterations per round
    iterations_per_round = max(1, min(200, round(TARGET_ROUND_SECONDS / warmup_seconds[1])))
    deadline = time.perf_counter() + arguments.timing_seconds
    round_seconds, iteration = [], 3
    while len(round_seconds) < MAX_ROUNDS:
        start = time.perf_counter()
        for _ in range(iterations_per_round):
            state, metrics = one_iteration(state, iteration)
            iteration += 1
        jax.block_until_ready((state, metrics))
        round_seconds.append((time.perf_counter() - start) / iterations_per_round)
        if len(round_seconds) >= MIN_ROUNDS and spread_of(round_seconds) <= STABLE_SPREAD:
            break
        if time.perf_counter() > deadline and len(round_seconds) >= MIN_ROUNDS_WHEN_LATE:
            break

    # the point: the median round, the five throughput quantities, and the card's peak memory
    seconds = statistics.median(round_seconds)
    steps_per_copy = config.num_steps * config.n_envs
    memory = device.memory_stats()
    spread = spread_of(round_seconds)
    return {
        "copies": config.n_copies, "status": "measured",
        "seconds_per_iteration": seconds,
        "total_steps_per_second": config.n_copies * steps_per_copy / seconds,
        "steps_per_second_per_copy": steps_per_copy / seconds,
        "hours_per_million_steps_per_copy": 1e6 / (3600 * steps_per_copy / seconds),
        "peak_memory_mb": memory["peak_bytes_in_use"] / 1e6 if memory else None,
        "seconds_per_iteration_rounds": round_seconds, "iterations_per_round": iterations_per_round,
        "relative_spread_middle_half": spread, "settled": spread <= STABLE_SPREAD,
        "build_and_prime_seconds": build_seconds, "warmup_call_seconds": warmup_seconds,
        "environments_per_copy": config.n_envs, "rollout_steps": config.num_steps,
        "steps_per_iteration_per_copy": steps_per_copy,
        "device_name": device.device_kind, "jax_version": jax.__version__,
    }


def run_size(arguments, copies, seconds_left, log_dir):
    """Start one size as its own process and return its point, or the reason it has none."""
    point_path = log_dir / f"copies-{copies}.json"
    log_path = log_dir / f"copies-{copies}.log"
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:],
               "--only-copies", str(copies), "--point-out", str(point_path)]
    start = time.perf_counter()
    # a size that would run past the series' budget is stopped and recorded as out of time
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=seconds_left)
    except subprocess.TimeoutExpired as stopped:
        # what the stopped process had printed arrives as bytes or as text, depending on the Python version
        printed = stopped.stderr or ""
        printed = printed.decode(errors="replace") if isinstance(printed, bytes) else printed
        log_path.write_text(f"$ {' '.join(command)}\n\nstopped after {seconds_left:.0f} s\n"
                            f"--- stderr ---\n{printed}\n")
        return {"copies": copies, "status": "out_of_time", "wall_seconds": time.perf_counter() - start,
                "log": str(log_path)}
    log_path.write_text(f"$ {' '.join(command)}\n\n--- stdout ---\n{process.stdout}\n"
                        f"--- stderr ---\n{process.stderr}\n")
    wall_seconds = time.perf_counter() - start
    if process.returncode == 0 and point_path.exists():
        return {**json.loads(point_path.read_text()), "wall_seconds": wall_seconds}
    status = ("out_of_memory" if any(marker in process.stderr for marker in OUT_OF_MEMORY_MARKERS)
              else "failed")
    return {"copies": copies, "status": status, "returncode": process.returncode,
            "wall_seconds": wall_seconds, "stderr_tail": process.stderr[-2000:], "log": str(log_path)}


def write_record(path, record):
    """Write the series record whole, so a reader never sees half of it."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=1) + "\n")
    os.replace(temporary, path)


def run_series(arguments):
    """The whole series: one size after another, the record rewritten after every size."""
    # the settings every job of this platform needs, and the card taken whole for the measurement
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise SystemExit("set PYTHONNOUSERSITE=1: the user-site packages hide the environment's own jax")
    host = socket.gethostname()
    if host.startswith("portal") or "login" in host:
        raise SystemExit(f"{host} is a login node: run the series on a compute node")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")
    trainer_command_line(arguments, arguments.first_copies)   # refuses a wrong file before any process starts

    # the record's name: workload, card and start time (Pacific)
    workload = workload_name(arguments)
    started = datetime.now().astimezone()
    stem = (f"{workload}__{arguments.card.lower().replace(' ', '-')}"
            f"__{started.astimezone(PACIFIC).strftime('%Y-%m-%d-%H-%M')}")
    arguments.out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = arguments.out_dir / "logs" / stem
    log_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "record": "doubling_series", "workload": workload, "card": arguments.card,
        "host": host, "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "configuration_file": str(arguments.configuration_file),
        "environment_family": arguments.environment_family, "map": arguments.map,
        "update_style": arguments.update_style, "bonus": arguments.bonus,
        "trainer_arguments": arguments.trainer_argument,
        "memory_fraction": os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"],
        "first_copies": arguments.first_copies, "budget_minutes": arguments.budget_minutes,
        "gain_threshold": GAIN, "started_at": started.isoformat(timespec="seconds"),
        "device_name": None, "environments_per_copy": None, "rollout_steps": None,
        "steps_per_iteration_per_copy": None, "stop_reason": "running", "points": [],
    }
    path = arguments.out_dir / f"{stem}.json"

    deadline = time.perf_counter() + arguments.budget_minutes * 60
    copies, previous = arguments.first_copies, None
    while True:
        # a size starts only when the time left is at least twice what the size before it took,
        # because building and compiling grow with the copies
        seconds_left = deadline - time.perf_counter()
        if seconds_left <= (2 * previous["wall_seconds"] if previous else 0):
            record["stop_reason"] = "out_of_time"
            break
        point = run_size(arguments, copies, seconds_left, log_dir)
        record["points"].append(point)
        if point["status"] != "measured":
            record["stop_reason"] = point["status"]
            break
        for field in ("device_name", "environments_per_copy", "rollout_steps", "steps_per_iteration_per_copy"):
            record[field] = point[field]
        gain = point["total_steps_per_second"] / previous["total_steps_per_second"] - 1 if previous else None
        point["gain_from_doubling"] = gain
        print(f"{copies:>6} copies: {point['seconds_per_iteration']:.4f} s per iteration, "
              f"{point['total_steps_per_second'] / 1e6:.3f} million steps per second"
              + (f", {gain:+.1%} from the doubling" if previous else ""), flush=True)
        write_record(path, record)
        if previous and gain < GAIN:
            record["stop_reason"] = "doubling_gained_under_10_percent"
            break
        if 2 * copies > arguments.largest_copies:
            record["stop_reason"] = "largest_copies_reached"
            break
        copies, previous = 2 * copies, point

    record["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    write_record(path, record)
    print(f"stopped: {record['stop_reason']}; wrote {path}", flush=True)


def main():
    """The series, or one size of it when the series started this process."""
    arguments = parse_arguments()
    if arguments.only_copies is None:
        run_series(arguments)
    else:
        arguments.point_out.write_text(json.dumps(measure(arguments), indent=1))


if __name__ == "__main__":
    main()
