"""Card check of `reward_divisor=held_after_first_arrival` on the study's own PointMaze Medium unit.

Part 1, the hold against the first arrival: the unit's trainer (built exactly as scripts/run_training.py
builds it) is trained for its whole budget; after every iteration the divisor's variance and the
first-arrival step of every policy are read. The hold step of a policy is the end of the last
rollout that moved its divisor; it must lie at or after the first-arrival step and less than one
rollout after it. A policy with no arrival must still be moved by the last rollout.

Part 2, the speed against `held_after_fraction`: two fresh trainers, one per option, trained in
alternating blocks of iterations; seconds per iteration per block, the first two iterations of
each trainer (compilation) left out.

usage: check_hold_and_speed.py --configuration-file <unit .toml> --out <record .json>
                               [--iterations N] [--block-iterations 256] [--wall-seconds 780]
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PLATFORM / "src"))
sys.path.insert(0, str(PLATFORM / "scripts"))
START = time.time()
OPTIONS = {"held after the first arrival": "reward_divisor=held_after_first_arrival",
           "held after 1/16 of the run": "reward_divisor=held_after_fraction,divisor_hold_fraction=0.0625"}


def build_trainer(configuration_file: Path, bonus_options: str, scratch: Path):
    """The trainer of the unit with the given bonus options, built as the training command builds
    it; returns (trainer, the command's arguments)."""
    import run_training as command
    from exploration_platform.bonuses.registry import make_bonus
    from exploration_platform.training.runner import Runner
    args = command.parse_args(["--run-dir", str(scratch), "--unit-id", "card-check",
                               "--configuration-file", str(configuration_file),
                               "--reward-setting", "arrival_plus_one",
                               "--bonus-options", bonus_options])
    if args.episode_steps == 0:
        args.episode_steps = command.ENVIRONMENTS[args.env]["episode_steps"]
    config, env_cfg = command.build_config(args), command.build_env_config(args)
    trainer = Runner(config, bonus=make_bonus(args.bonus, command.bonus_options_of(args)),
                     env_cfg=env_cfg)
    return trainer, args


def primed_state(trainer, args):
    """A fresh training state after the random-action rollouts, ready on the card."""
    import jax
    state = trainer.prime(trainer.init_state(run_seed=args.run_seed))
    jax.block_until_ready(state.obs)
    return state


def spread(values) -> dict:
    """The smallest, median and largest of a set of step counts; empty when no policy is in it."""
    if len(values) == 0:
        return {}
    return {"smallest": int(values.min()), "median": float(np.median(values)),
            "largest": int(values.max())}


def hold_against_first_arrival(trainer, args, iterations: int) -> dict:
    """Part 1: train the whole unit and compare each policy's hold step with its first arrival."""
    cfg = trainer.cfg
    steps_per_iteration = cfg.n_envs * cfg.num_steps
    state = primed_state(trainer, args)
    variance = np.asarray(state.agent_state["int_rms"].var)[:, 0].copy()
    last_moved = np.zeros(cfg.n_copies, np.int64)
    began = time.time()
    for iteration in range(1, iterations + 1):
        state, _ = trainer.iterate(state, trainer.lr_argument(
            iteration, args.lr_anneal_iterations or args.iterations))
        now = np.asarray(state.agent_state["int_rms"].var)[:, 0]
        # before: last_moved [3, 7], this iteration 8 changed policy 0's variance only
        # after:  last_moved [8, 7]
        last_moved = np.where(now != variance, iteration, last_moved)
        variance = now.copy()
    first_goal = np.asarray(state.agent_state["episode_log"]["first_goal_step"])
    arrived = first_goal >= 0
    hold_step = last_moved * steps_per_iteration
    after = (hold_step - first_goal)[arrived]
    return {
        "iterations": iterations, "environment_steps_per_policy_and_iteration": steps_per_iteration,
        "policies": int(cfg.n_copies), "policies_with_an_arrival": int(arrived.sum()),
        "first_arrival_step": spread(first_goal[arrived]),
        "hold_step_minus_first_arrival_step": spread(after),
        "policies_whose_hold_is_within_one_rollout_after_the_arrival":
            int(((after >= 0) & (after < steps_per_iteration)).sum()),
        "policies_without_an_arrival_still_moved_by_the_last_rollout":
            int((last_moved[~arrived] == iterations).sum()),
        "policies_without_an_arrival": int((~arrived).sum()),
        "seconds": time.time() - began,
        "per_policy": {"first_arrival_step": first_goal.tolist(), "hold_step": hold_step.tolist()},
    }


def speed_in_alternating_blocks(trainers: dict, iterations: int, block: int, deadline: float) -> dict:
    """Part 2: seconds per iteration of each option, measured block by block in alternation."""
    import jax
    states, done, seconds = {}, {}, {name: [] for name in trainers}
    # two iterations of each trainer first: the first compiles, the second may compile again for
    # donated buffers; neither is timed
    for name, (trainer, args) in trainers.items():
        state = primed_state(trainer, args)
        for iteration in (1, 2):
            state, _ = trainer.iterate(state, trainer.lr_argument(
                iteration, args.lr_anneal_iterations or args.iterations))
        jax.block_until_ready(state.obs)
        states[name], done[name] = state, 2
    while all(count + block <= iterations for count in done.values()) and time.time() < deadline:
        for name, (trainer, args) in trainers.items():
            state, began = states[name], time.time()
            for iteration in range(done[name] + 1, done[name] + block + 1):
                state, _ = trainer.iterate(state, trainer.lr_argument(
                    iteration, args.lr_anneal_iterations or args.iterations))
            jax.block_until_ready(state.obs)
            seconds[name].append((time.time() - began) / block)
            states[name], done[name] = state, done[name] + block
    cfg = next(iter(trainers.values()))[0].cfg
    steps = cfg.n_copies * cfg.n_envs * cfg.num_steps
    table = {}
    for name, blocks in seconds.items():
        mean = statistics.fmean(blocks)
        per_copy = steps / mean / cfg.n_copies
        table[name] = {"blocks": len(blocks), "iterations_per_block": block,
                       "seconds_per_iteration_by_block": blocks,
                       "copies": int(cfg.n_copies), "seconds_per_iteration": mean,
                       "total_steps_per_second_millions": steps / mean / 1e6,
                       "steps_per_second_per_copy_thousands": per_copy / 1e3,
                       "hours_per_million_steps_per_copy": 1e6 / (3600 * per_copy)}
    return table


def main() -> None:
    """Run both parts and write one record."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=0, help="0: the unit's own number")
    parser.add_argument("--block-iterations", type=int, default=256)
    parser.add_argument("--wall-seconds", type=float, default=780.0)
    own = parser.parse_args()
    scratch = own.out.parent / "scratch_run_folder_of_the_card_check"
    import jax
    record = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(START)),
              "devices": [str(device) for device in jax.devices()],
              "configuration_file": str(own.configuration_file), "options": OPTIONS}

    # part 1: the hold of every policy against its first arrival, over the unit's whole budget
    first_name = "held after the first arrival"
    trainer, args = build_trainer(own.configuration_file, OPTIONS[first_name], scratch)
    iterations = own.iterations or args.iterations
    record["hold_against_first_arrival"] = hold_against_first_arrival(trainer, args, iterations)
    print(json.dumps({key: value for key, value in record["hold_against_first_arrival"].items()
                      if key != "per_policy"}), flush=True)

    # part 2: both options from fresh states, in alternating blocks, until the budget or the time ends
    trainers = {first_name: (trainer, args)}
    for name in OPTIONS:
        if name != first_name:
            trainers[name] = build_trainer(own.configuration_file, OPTIONS[name], scratch)
    record["speed"] = speed_in_alternating_blocks(trainers, iterations, own.block_iterations,
                                                  START + own.wall_seconds)
    record["seconds_in_all"] = time.time() - START
    own.out.parent.mkdir(parents=True, exist_ok=True)
    own.out.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({name: {key: value for key, value in row.items()
                             if key != "seconds_per_iteration_by_block"}
                      for name, row in record["speed"].items()}), flush=True)
    print(f"wrote {own.out} after {record['seconds_in_all']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
