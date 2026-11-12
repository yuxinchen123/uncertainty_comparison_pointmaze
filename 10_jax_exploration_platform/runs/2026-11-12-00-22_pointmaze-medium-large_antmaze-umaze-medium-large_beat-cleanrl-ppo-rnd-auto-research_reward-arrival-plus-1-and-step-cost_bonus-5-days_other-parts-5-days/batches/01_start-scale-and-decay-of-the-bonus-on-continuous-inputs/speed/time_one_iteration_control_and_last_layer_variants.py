"""Time one training iteration of PointMaze Large for the control and the two last-layer variants.

The learner is the PointMaze example configuration file (CleanRL PPO with random network
distillation, 4 environments per policy, 128-step rollouts, 4 epochs x 4 minibatches) with the
number of policies given on the command line. Each variant is built, primed, run for three
iterations that are not timed (the first compiles the program, and a program whose input is
donated compiles once more on its first donated call), and then timed over a fixed number of
iterations with one wait for the card at the end.

One record per variant is appended to the output file as soon as it is measured.

Run on a card, through the profile job of the speed study:
  submit_profile.sh rtx-5080-nekomata02 <name> <this script> --policies 160 --out <file>.jsonl
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PLATFORM = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PLATFORM / "src"))
sys.path.insert(0, str(PLATFORM / "scripts"))
EXAMPLE = PLATFORM / "configs/experiments" / (
    "pointmaze_large_cleanrl-ppo-rnd_goal-termination_reward-arrival-plus-one_example_steps-1024000.toml")

# each variant in words, and the bonus options that select it
VARIANTS = (
    ("control: CleanRL random network distillation as in the baseline", {}),
    ("fast last layer inside the rollout, step 0.3 of the exact fit",
     {"fast_last_layer_step": 0.3}),
    ("scheduled last layer on fixed features, decay 0.75",
     {"scheduled_last_layer_decay": 0.75}),
)


def card_memory_used_mb() -> int:
    """The card's memory in use, in MiB, as nvidia-smi reports it for the first card."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return int(out.splitlines()[0])


def measure(name: str, options: dict, config, env_cfg, timed_iterations: int) -> dict:
    """Build, prime, warm up and time one variant; returns its record."""
    import jax
    from exploration_platform.bonuses.registry import make_bonus
    from exploration_platform.training.runner import Runner

    # build and prime
    start = time.time()
    runner = Runner(config, bonus=make_bonus("rnd_next_state", options), env_cfg=env_cfg)
    state = runner.prime(runner.init_state(0))
    jax.block_until_ready(state.obs)
    seconds_build_and_prime = time.time() - start

    # three iterations that are not timed
    horizon = 2000
    start = time.time()
    for k in (1, 2, 3):
        state, metrics = runner.iterate(state, runner.lr_argument(k, horizon))
        jax.block_until_ready(metrics["loss"])
    seconds_warm_up = time.time() - start

    # the timed iterations: dispatched without waiting, one wait at the end
    start = time.time()
    for k in range(4, 4 + timed_iterations):
        state, metrics = runner.iterate(state, runner.lr_argument(k, horizon))
    jax.block_until_ready(metrics["loss"])
    seconds_per_iteration = (time.time() - start) / timed_iterations

    # the throughput columns of every table: total and per copy
    copies = config.n_copies
    steps_per_iteration = copies * config.n_envs * config.num_steps
    total_per_second = steps_per_iteration / seconds_per_iteration
    per_copy_per_second = total_per_second / copies
    return {"variant": name, "bonus_options": options, "copies": copies,
            "environments_per_copy": config.n_envs, "rollout_steps": config.num_steps,
            "timed_iterations": timed_iterations,
            "seconds_per_iteration": seconds_per_iteration,
            "total_steps_per_second": total_per_second,
            "steps_per_second_per_copy": per_copy_per_second,
            "hours_per_million_steps_per_copy": 1e6 / (3600 * per_copy_per_second),
            "seconds_build_and_prime": seconds_build_and_prime,
            "seconds_three_untimed_iterations": seconds_warm_up,
            "card_memory_used_mb_after": card_memory_used_mb(),
            "rint_mean_last_iteration": float(metrics["rint_mean"].mean()),
            "measured_at": datetime.now().astimezone().isoformat()}


def main() -> None:
    """Measure every variant in turn, writing each record as it is ready."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies", type=int, default=160)
    parser.add_argument("--timed-iterations", type=int, default=60)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    from run_training import build_config, build_env_config, hardware_record, parse_args
    unit = parse_args(["--run-dir", "not-written", "--unit-id", "not-written",
                       "--configuration-file", str(EXAMPLE), "--copies", str(arguments.policies)])
    config, env_cfg = build_config(unit), build_env_config(unit)

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    with open(arguments.out, "a", buffering=1) as out:
        out.write(json.dumps(hardware_record()) + "\n")
        for name, options in VARIANTS:
            record = measure(name, options, config, env_cfg, arguments.timed_iterations)
            out.write(json.dumps(record) + "\n")
            out.flush()
            print(f"{name}: {record['seconds_per_iteration'] * 1e3:.1f} ms per iteration, "
                  f"{record['total_steps_per_second'] / 1e6:.3f} million steps per second in "
                  f"total, {record['steps_per_second_per_copy'] / 1e3:.2f} thousand per copy, "
                  f"{record['hours_per_million_steps_per_copy']:.3f} hours per million steps "
                  f"per copy, card memory {record['card_memory_used_mb_after']} MiB", flush=True)


if __name__ == "__main__":
    main()
