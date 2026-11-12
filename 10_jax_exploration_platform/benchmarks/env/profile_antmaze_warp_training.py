"""The whole training iteration of the study's learner with the Warp AntMaze physics, one cell per process.

A cell is a number of policies and a number of environments per policy on AntMaze U-maze. The
learner is the study's, read from the Warp example configuration file: clipped PPO, 4 epochs x 4
minibatches, 128-step rollouts, random network distillation on the next state, the episode ending
at the goal, the episode log and the coverage map on. Only the two numbers of the cell, the
per-epoch minibatch gather (on from 128 environments per policy, as the study runs that shape) and
the number of priming rollouts (1 instead of 50; priming is not part of an iteration) are changed.

What one process measures:

1. The whole iteration (`Runner.iterate`), as the training loop calls it: rounds of chained calls
   with one wait for the card at the end of each round. The first two calls are timed apart,
   because they include compilation.
2. What the process holds on the card after those rounds (nvidia-smi's figure for this process;
   JAX does not hand memory back, so this is the largest amount the iterations needed), JAX's own
   peak, and Warp's.
3. The use of MuJoCo Warp's shared contact pool, and whether anything was dropped (a cell with a
   drop fails).
4. Two pieces alone, to say where the iteration's time goes: the rollout scan (policy forward and
   environment, 128 steps, the state donated and carried from call to call as in the iteration)
   and the update (shuffle, gather, sixteen gradients, clip, Adam, on a batch of random numbers of
   the right shapes). A piece timed alone waits for the card after every call.
5. The processes of anyone else on the same card, read every 15 seconds while the cell runs: a
   cell that saw one is written with the word "discarded" and is not shown in the table.

The record of a cell is one JSON file; `--render` writes the Markdown table from all records in a
folder. Never run on a login node. `--processor-dry-run` runs a cell on a compute node's
processors to check the script itself (`test_profile_antmaze_warp_training.py`); its record says
so and holds no card facts.

  <platform_jax_warp python> profile_antmaze_warp_training.py --policies 1024 --environments-per-policy 16 \
      --machine "serval02 card 1" --unit "<unit name>" --records <folder>
  <any python 3.11+> profile_antmaze_warp_training.py --render --records <folder> --markdown <file.md>
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(BASE / "scripts"))
sys.path.insert(0, str(BASE / "benchmarks" / "harness"))

STUDY_CONFIGURATION = BASE / "configs/experiments" / (
    "antmaze_umaze_warp-physics_cleanrl-ppo-rnd_goal-termination_reward-arrival-plus-one"
    "_example_steps-204800.toml")
PACIFIC = ZoneInfo("America/Los_Angeles")
OUT_OF_MEMORY_EXIT_CODE = 77
NO_CARD_EXIT_CODE = 76


def pacific(iso: str) -> str:
    """A stored timestamp for the reader: Pacific, with the zone named."""
    return datetime.fromisoformat(iso).astimezone(PACIFIC).strftime("%Y-%m-%d %H:%M PT")


def median(values) -> float:
    """The middle value of a list (the upper of the two middle values for an even count)."""
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def study_configuration(policies: int, environments_per_policy: int):
    """The study's learner and the Warp U-maze at one cell, as (PPOConfig, AntMazeConfig).

    before: the example file's trainer (16 policies x 4 environments, 50 priming rollouts)
    after:  the same trainer with n_copies and n_envs of the cell, 1 priming rollout, and
            minibatches gathered one epoch at a time from 128 environments per policy
    """
    from experiment_configuration import read_experiment_configuration
    from exploration_platform.agents.ppo.config import PPOConfig
    from exploration_platform.envs.antmaze.am_common import AntMazeConfig
    document = read_experiment_configuration(STUDY_CONFIGURATION)
    trainer = {**document["trainer"], "n_copies": policies, "n_envs": environments_per_policy,
               "gather_minibatches_per_epoch": environments_per_policy >= 128,
               "prime_iterations": 1}
    return PPOConfig(**trainer), AntMazeConfig(**document["environment"])


def card_rows() -> list:
    """Every process on every card of this machine, as (card uuid, process id, MiB it holds)."""
    # before: lines "GPU-fed6..., 12345, 2311"; after: [("GPU-fed6...", 12345, 2311)]
    rows = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                           "--format=csv,noheader,nounits"], capture_output=True, text=True,
                          check=True).stdout.strip().splitlines()
    parsed = []
    for row in rows:
        uuid, pid, mib = (part.strip() for part in row.split(","))
        parsed.append((uuid, int(pid), int(mib) if mib.isdecimal() else None))
    return parsed


def card_record() -> dict:
    """The card this process runs on: its uuid, name, total memory and driver.

    The card is the one nvidia-smi lists this process on (JAX must have touched the card
    already); when the process is not listed, the first card CUDA_VISIBLE_DEVICES names.
    """
    own = [uuid for uuid, pid, _ in card_rows() if pid == os.getpid()]
    target = own[0] if own else os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    row = subprocess.run(["nvidia-smi", "-i", target, "--query-gpu=uuid,name,memory.total,driver_version",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True,
                         check=True).stdout.strip().splitlines()[0]
    uuid, name, total, driver = (part.strip() for part in row.split(","))
    return {"uuid": uuid, "name": name, "memory_total_mib": int(total), "driver": driver}


def watch_card(card_uuid: str, seen: list, stop: threading.Event, seconds: float = 15.0) -> None:
    """Until told to stop, add to `seen` every process of anyone else found on the card."""
    while not stop.wait(seconds):
        seen.extend(other_processes(card_uuid))


def other_processes(card_uuid: str) -> list:
    """Processes on this card that are not this process, as "pid <n> user <name> <MiB> MiB"."""
    others = []
    for uuid, pid, mib in card_rows():
        if uuid == card_uuid and pid != os.getpid():
            user = subprocess.run(["ps", "-o", "user=", "-p", str(pid)], capture_output=True,
                                  text=True).stdout.strip() or "unknown"
            others.append(f"pid {pid} user {user} {mib} MiB")
    return others


def memory_record(card_uuid: str) -> dict:
    """What this process holds on the card: nvidia-smi's figure, JAX's peak and Warp's peak."""
    import jax
    import warp as wp
    own = [mib for uuid, pid, mib in card_rows() if uuid == card_uuid and pid == os.getpid()]
    used = subprocess.run(["nvidia-smi", "-i", card_uuid, "--query-gpu=memory.used",
                           "--format=csv,noheader,nounits"], capture_output=True, text=True,
                          check=True).stdout.split()[0]
    return {"process_mib_on_card": own[0] if own else None, "card_mib_used": int(used),
            "jax_peak_bytes": jax.local_devices()[0].memory_stats()["peak_bytes_in_use"],
            "warp_pool_peak_bytes": wp.get_mempool_used_mem_high("cuda:0")}


def is_out_of_memory(error) -> bool:
    """Whether an exception says the card's memory ran out."""
    text = str(error).lower()
    return "resource_exhausted" in text or "out of memory" in text or "failed to allocate" in text


def write_record(path: Path, record: dict) -> None:
    """Write the record through a temporary file, so a reader never sees half of it."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(record, indent=1))
    temporary.replace(path)


def rollout_piece(runner):
    """The rollout scan as the iteration runs it, compiled alone with its state donated.

    Returns a function (environment state, observation) -> (environment state, observation):
    128 steps of policy forward and environment step, storing what the rest of the iteration
    reads, with fresh action noise of fixed values (timing depends on shapes, not on values).
    """
    import jax
    from exploration_platform import F32
    from exploration_platform.agents.ppo.networks import actor_mean
    from exploration_platform.training.state import named_params, stored_params
    from exploration_platform.training.train_step import rollout_unroll
    cfg, env = runner.cfg, runner.env

    def rollout(env_state, obs, actor, noise):
        """One rollout: the carried state after 128 steps, and one number per stored buffer."""
        def body(carry, z):
            """One rollout step: the action from the policy's mean and noise, then the environment."""
            env_state, obs = carry
            action = actor_mean(actor, obs) + jax.numpy.exp(actor["logstd"])[:, None, :] * z
            stepped, next_obs, reward, terminated, truncated, final_obs = env.step(env_state, action)
            return (stepped, next_obs), (obs, action, final_obs, reward, terminated.astype(F32),
                                         (terminated | truncated).astype(F32))
        carry, buffers = jax.lax.scan(body, (env_state, obs), noise, unroll=rollout_unroll(cfg))
        # every buffer is read, so none is removed from the program as unused
        return carry, sum(buffer.sum() for buffer in buffers)

    compiled = jax.jit(rollout, donate_argnums=(0, 1))

    def bind(state):
        """The piece bound to one training state's policy; returns the function that is timed."""
        actor = named_params(stored_params(state, runner.layout), runner.layout)["agent"]["actor"]
        noise = jax.random.normal(jax.random.PRNGKey(11), (cfg.num_steps, cfg.n_copies, cfg.n_envs,
                                                           env.act_dim), F32)
        return lambda env_state, obs: compiled(env_state, obs, actor, noise)[0]
    return bind


def update_piece(runner, state):
    """The update alone on a batch of random numbers of the iteration's shapes; a function of no arguments."""
    import jax
    from exploration_platform.agents.ppo.update import build_update
    from exploration_platform.training.state import stored_params
    from profile_jax_phases import make_batch
    cfg = runner.cfg
    batch = make_batch(cfg, jax.random.PRNGKey(7), runner.env.obs_dim, runner.env.act_dim)
    update = build_update(cfg, runner.total_loss, runner.sweep.lr_per_copy)
    learning_rate, key = runner.lr_argument(1, 1), jax.random.PRNGKey(8)
    compiled = jax.jit(lambda params, opt, rows: update(params, opt, rows, learning_rate, key))
    stored = stored_params(state, runner.layout)
    return lambda: compiled(stored, state.opt, batch)


def measure(args) -> int:
    """Measure one cell and write its record; returns the process's exit code."""
    started = time.time()
    unit_start = args.unit_start_epoch or started
    deadline = unit_start + args.unit_seconds
    say = lambda text: print(f"[{time.time() - started:6.1f} s] {text}", flush=True)

    import jax
    import exploration_platform  # noqa: F401  (64-bit statistics on before any array exists, as in training)
    on_card = jax.devices()[0].platform == "gpu"
    if not on_card and not args.processor_dry_run:
        print(f"JAX did not load the graphics card (device {jax.devices()[0]}); nothing measured")
        return NO_CARD_EXIT_CODE
    import importlib.metadata
    import numpy as np
    from exploration_platform.training.runner import Runner

    policies, per_policy = args.policies, args.environments_per_policy
    # GOAL: name the card (JAX touches it first, so nvidia-smi lists this process on it) and
    # start reading who else is on it; a dry run on processors has no card.
    jax.block_until_ready(jax.numpy.zeros(1))
    card = card_record() if on_card else {"uuid": "", "name": "processors (dry run)",
                                          "memory_total_mib": 0, "driver": ""}
    others, stop_watching = [], threading.Event()
    if on_card:
        others.extend(other_processes(card["uuid"]))
        threading.Thread(target=watch_card, args=(card["uuid"], others, stop_watching),
                         daemon=True).start()
    name = f"{args.card_slug}_policies-{policies}_environments-per-policy-{per_policy}"
    path = args.records / f"{name}.json"
    record = {
        "record": "training iteration with the Warp AntMaze physics", "environment": "antmaze-umaze",
        "policies": policies, "environments_per_policy": per_policy,
        "environments": policies * per_policy, "machine": args.machine, "unit": args.unit,
        "host": socket.gethostname(), "card": card, "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "started_iso": datetime.now().astimezone().isoformat(),
        "versions": {package: importlib.metadata.version(package)
                     for package in ("jax", "jaxlib", "mujoco", "mujoco-mjx", "warp-lang")},
        "git_commit": subprocess.run(["git", "-C", str(BASE), "rev-parse", "--short", "HEAD"],
                                     capture_output=True, text=True).stdout.strip(),
        "configuration_file": STUDY_CONFIGURATION.name, "status": "running",
        "processor_dry_run": not on_card,
    }
    write_record(path, record)
    say(f"cell {policies} policies x {per_policy} environments on {args.machine} ({card['name']}); "
        f"{deadline - time.time():.0f} s of the unit are left")

    try:
        # GOAL: the study's program and a primed state.
        cfg, env_cfg = study_configuration(policies, per_policy)
        runner = Runner(cfg, bonus="rnd_next_state", env_cfg=env_cfg)
        record.update(rollout_steps=cfg.num_steps, kernel_folder=runner.env.kernel_folder,
                      kernel_folders_at_start=sum(1 for entry in os.scandir(runner.env.kernel_folder)
                                                  if entry.is_dir()),
                      gather_minibatches_per_epoch=cfg.gather_minibatches_per_epoch)
        state = runner.prime(runner.init_state(run_seed=0))
        jax.block_until_ready(state.obs)
        runner.check_environment(state)
        record["seconds_to_build_and_prime"] = time.time() - started
        say(f"built and primed; kernel folder {runner.env.kernel_folder}")
        step = lambda s: runner.iterate(s, runner.lr_argument(1, 1))

        # GOAL: three calls with a wait each: the first two compile, the third sets the block size.
        single, losses = [], []
        for _ in range(3):
            t0 = time.perf_counter()
            state, metrics = step(state)
            jax.block_until_ready(metrics["loss"])
            single.append(time.perf_counter() - t0)
            losses.append(float(np.asarray(metrics["loss"])))
        record.update(first_two_iterations_seconds=single[:2], one_iteration_with_wait_seconds=single[2],
                      seconds_to_first_timed_call=time.time() - started)
        say(f"first two iterations {single[0]:.1f} s and {single[1]:.1f} s; the third {single[2]:.3f} s")

        # GOAL: rounds of chained calls with one wait at the end, as the training loop runs.
        block = max(2, min(20, int(args.round_seconds / single[2])))
        rounds = []
        for _ in range(args.rounds):
            t0 = time.perf_counter()
            for _ in range(block):
                state, metrics = step(state)
            jax.block_until_ready(metrics["loss"])
            rounds.append((time.perf_counter() - t0) / block)
            losses.append(float(np.asarray(metrics["loss"])))
        seconds = median(rounds)
        per_policy_rate = per_policy * cfg.num_steps / seconds
        log = {key: np.asarray(value) for key, value in state.agent_state["episode_log"].items()}
        record.update(
            chained_calls_per_round=block, seconds_per_iteration_rounds=rounds,
            seconds_per_iteration=seconds,
            total_steps_per_second_millions=policies * per_policy_rate / 1e6,
            steps_per_second_per_policy=per_policy_rate,
            hours_per_million_steps_per_policy=1e6 / (3600.0 * per_policy_rate),
            loss_finite=bool(np.isfinite(losses).all()),
            episodes_ended=int(log["episodes"].sum()), arrivals=int(log["arrivals"].sum()),
            non_finite_resets=int(np.asarray(state.env_state.nan_count).sum()),
            memory=memory_record(card["uuid"]) if on_card else None,
            pool=runner.check_environment(state))
        write_record(path, record)
        say(f"whole iteration {seconds:.4f} s chained ({block} calls a round, {args.rounds} rounds): "
            f"{record['total_steps_per_second_millions']:.3f} million steps per second in total, "
            f"{per_policy_rate:.0f} per policy; memory {record['memory']}; pool {record['pool']}")

        # GOAL: the update alone, then the rollout alone (which consumes the state), if time is left.
        pieces = {}
        if deadline - time.time() > 3 * single[1] + 20:
            piece = update_piece(runner, state)
            for _ in range(2):
                jax.block_until_ready(piece())
            times = []
            for _ in range(args.piece_calls):
                t0 = time.perf_counter()
                jax.block_until_ready(piece())
                times.append(time.perf_counter() - t0)
            pieces["update"] = {"median_seconds": median(times), "calls": len(times)}
            say(f"update alone {median(times):.4f} s")
        if deadline - time.time() > 3 * single[1] + 20:
            piece = rollout_piece(runner)(state)
            env_state, obs = state.env_state, state.obs
            for _ in range(2):
                env_state, obs = piece(env_state, obs)
                jax.block_until_ready(obs)
            times = []
            for _ in range(args.piece_calls):
                t0 = time.perf_counter()
                env_state, obs = piece(env_state, obs)
                jax.block_until_ready(obs)
                times.append(time.perf_counter() - t0)
            pieces["rollout"] = {"median_seconds": median(times), "calls": len(times)}
            runner.env.raise_on_overflow(env_state)
            say(f"rollout alone {median(times):.4f} s")
        record["pieces"] = pieces
    except Exception as error:
        # the card's memory running out is a result of the cell; anything else is a failure
        out_of_memory = is_out_of_memory(error)
        stop_watching.set()
        record.update(status="not measured" if out_of_memory else "failed",
                      reason=("the card's memory ran out: " if out_of_memory else
                              f"{type(error).__name__}: ") + str(error)[:400],
                      other_processes_on_card=sorted(set(others)),
                      wall_seconds=time.time() - started)
        write_record(path, record)
        if out_of_memory:
            say(f"not measured: {record['reason']}")
            return OUT_OF_MEMORY_EXIT_CODE
        raise

    # GOAL: a cell that shared the card with anyone else is kept, but marked as not usable.
    stop_watching.set()
    if on_card:
        others.extend(other_processes(card["uuid"]))
    record["other_processes_on_card"] = sorted(set(others))
    record["status"] = "discarded" if others else "measured"
    record["wall_seconds"] = time.time() - started
    record["kernel_folders_at_end"] = sum(1 for entry in os.scandir(runner.env.kernel_folder)
                                          if entry.is_dir())
    write_record(path, record)
    say(f"cell {record['status']} in {record['wall_seconds']:.0f} s"
        + (f": another process was on the card: {record['other_processes_on_card']}"
           if record["other_processes_on_card"] else ""))
    return 0


# ----------------------------------------------------------------------------------------------
# the Markdown table
# ----------------------------------------------------------------------------------------------

def share(record: dict, piece: str) -> str:
    """A piece's time as a share of the chained iteration, in percent; "not measured" without it."""
    if piece not in record.get("pieces", {}):
        return "not measured"
    return f"{100 * record['pieces'][piece]['median_seconds'] / record['seconds_per_iteration']:.0f}%"


def table_rows(records: list) -> list:
    """One table row per measured cell, ordered by environments per policy and then by policies."""
    rows = []
    for r in sorted(records, key=lambda r: (r["environments_per_policy"], r["policies"])):
        # a processor dry run has no card memory; its row shows 0
        memory, pool = r["memory"] or {"process_mib_on_card": 0, "card_mib_used": 0}, r["pool"]
        held = memory["process_mib_on_card"] or memory["card_mib_used"]
        fullest = max(pool["nacon_peak"], pool["ncollision_peak"])
        rows.append(
            f"| {r['policies']:,} | {r['environments_per_policy']} | {r['environments']:,} | "
            f"{r['seconds_per_iteration']:.4f} | {r['total_steps_per_second_millions']:.3f} | "
            f"{r['steps_per_second_per_policy']:,.0f} | {r['hours_per_million_steps_per_policy']:.4f} | "
            f"{share(r, 'rollout')} | {share(r, 'update')} | {held / 1024:.2f} | "
            f"{100 * fullest / pool['naconmax']:.1f}% | {pacific(r['started_iso'])} |")
    return rows


def render(records_folder: Path) -> str:
    """The Markdown document of every record in a folder: one table per card, and what was not measured."""
    records = [json.loads(path.read_text()) for path in sorted(records_folder.glob("*.json"))]
    lines = [
        "# Training iteration with the Warp AntMaze physics (AntMaze U-maze)",
        "",
        "Generated by `benchmarks/env/profile_antmaze_warp_training.py --render` from the records in "
        f"`{records_folder.name}/`; change the script, not this file. Times are Pacific, converted from "
        "machines running on Eastern.",
        "",
        "## What was measured",
        "",
        "`Runner.iterate` of the platform with the study's learner, read from "
        f"`configs/experiments/{STUDY_CONFIGURATION.name}`: clipped PPO in 4 epochs of 4 minibatches, "
        "128-step rollouts, random network distillation on the next state, the episode ending at the "
        "goal, the episode log and the coverage map on, MuJoCo Warp physics with a shared pool of 48 "
        "contact slots per environment. One process per cell. Define $c$ as the number of policies, "
        "$e$ as the environments per policy and $s$ as the median seconds per iteration over rounds "
        "of chained calls with one wait for the card per round (the first three calls, which include "
        "compilation, are left out).",
        "",
        "- Total steps per second: $\\dfrac{128 \\times c \\times e}{s}$, shown in millions.",
        "- Steps per second per policy: $r = \\dfrac{128 \\times e}{s}$.",
        "- Hours per million steps per policy: $\\dfrac{10^{6}}{3600 \\times r}$.",
        "- Rollout share and update share: the median seconds of that piece compiled and timed alone "
        "(the rollout scan of 128 steps of policy forward and environment; the update with its "
        "shuffle, gather, sixteen gradients, clip and Adam), divided by $s$. A piece alone waits for "
        "the card after every call, and the rest of the iteration (value pass, distillation forward, "
        "statistics, advantages, episode log, coverage map) is not timed alone, so the two shares do "
        "not add up to 100%.",
        "- Peak card memory: what the process held on the card after the timed rounds, as nvidia-smi "
        "reports it for the process. JAX does not hand memory back, so this is the most the "
        "iterations needed, with the memory JAX took but no longer uses included.",
        "- Largest pool use: the larger of the contacts and the candidate pairs in the shared pool at "
        "any physics step, divided by the pool's slots. Nothing was dropped in any row shown.",
        "",
    ]
    cards = {}
    for record in records:
        cards.setdefault((record["card"]["name"], record["machine"]), []).append(record)
    for (card, machine), group in sorted(cards.items()):
        measured = [r for r in group if r["status"] == "measured"]
        lines += [f"## {card} ({machine})", ""]
        if measured:
            sample = measured[0]
            lines += [
                f"JAX {sample['versions']['jax']}, MuJoCo MJX {sample['versions']['mujoco-mjx']}, "
                f"warp-lang {sample['versions']['warp-lang']}, driver {sample['card']['driver']}, card "
                f"memory {sample['card']['memory_total_mib']:,} MiB; kernel folder "
                f"`{sample['kernel_folder']}`.",
                "",
                "| policies | environments<br>per policy | environments<br>in total | seconds per<br>iteration | "
                "total steps per<br>second (millions) | steps per second<br>per policy | "
                "hours per million<br>steps per policy | rollout<br>share | update<br>share | "
                "peak card<br>memory (GiB) | largest<br>pool use | measured |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
                *table_rows(measured), ""]
            lines += ["First two iterations (with compilation), seconds: " + "; ".join(
                f"{r['policies']:,} x {r['environments_per_policy']}: "
                f"{r['first_two_iterations_seconds'][0]:.0f} and {r['first_two_iterations_seconds'][1]:.0f}"
                for r in sorted(measured, key=lambda r: (r["environments_per_policy"], r["policies"]))) + ".",
                ""]
            in_use = [f"{r['policies']:,} x {r['environments_per_policy']}: "
                      f"{r['memory']['jax_peak_bytes'] / 2 ** 30:.2f} and "
                      f"{r['memory']['warp_pool_peak_bytes'] / 2 ** 30:.2f}"
                      for r in sorted(measured, key=lambda r: (r["environments_per_policy"], r["policies"]))
                      if r["memory"]]
            if in_use:
                lines += ["Largest memory in use as the two libraries count it themselves, JAX's and then "
                          "Warp's own pool, GiB (the column above also holds what JAX took and no longer "
                          "uses, and the card's context): " + "; ".join(in_use) + ".", ""]
            checks = [f"{r['policies']:,} x {r['environments_per_policy']}: loss finite "
                      f"{'yes' if r['loss_finite'] else 'NO'}, {r['episodes_ended']:,} episodes ended, "
                      f"{r['arrivals']:,} at the goal, {r['non_finite_resets']} non-finite resets"
                      for r in sorted(measured, key=lambda r: (r["environments_per_policy"], r["policies"]))]
            lines += ["Checks in the same processes: " + "; ".join(checks) + ".", ""]
        unused = [r for r in group if r["status"] != "measured"]
        if unused:
            lines += ["Not measured or not used:", ""]
            for r in unused:
                reason = (f"another process was on the card: {r['other_processes_on_card']}"
                          if r["status"] == "discarded" else
                          r.get("reason", "the process ended before the cell was measured"))
                lines.append(f"- {r['policies']:,} policies x {r['environments_per_policy']} environments, "
                             f"{pacific(r['started_iso'])}: {reason}.")
            lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    """Measure one cell, or write the Markdown table from a folder of records."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", type=Path, required=True, help="the folder of the cells' records")
    parser.add_argument("--render", action="store_true", help="write the Markdown table and measure nothing")
    parser.add_argument("--markdown", type=Path, help="with --render: the file to write")
    parser.add_argument("--policies", type=int)
    parser.add_argument("--environments-per-policy", type=int)
    parser.add_argument("--machine", help='the machine and card in words, e.g. "serval02 card 1"')
    parser.add_argument("--card-slug", help="the card's name in a record's file name, e.g. h100-nvl")
    parser.add_argument("--card-name", help="with --not-measured-reason: the card's name as nvidia-smi "
                                            'prints it, e.g. "NVIDIA H100 NVL"')
    parser.add_argument("--unit", default="", help="the name of the unit this cell is part of")
    parser.add_argument("--unit-start-epoch", type=float, default=0.0,
                        help="when the unit started, in seconds since 1970; default: now")
    parser.add_argument("--unit-seconds", type=float, default=840.0,
                        help="no new piece starts when too little of this is left")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--round-seconds", type=float, default=6.0,
                        help="about how long one round of chained calls lasts")
    parser.add_argument("--piece-calls", type=int, default=5)
    parser.add_argument("--processor-dry-run", action="store_true",
                        help="run the cell on processors to check this script; nothing about a card is recorded")
    parser.add_argument("--not-measured-reason",
                        help="measure nothing: write a record that says why this cell was not measured")
    args = parser.parse_args(argv)
    if args.render:
        args.markdown.write_text(render(args.records))
        print(f"wrote {args.markdown}")
        return 0
    for required in ("policies", "environments_per_policy", "machine", "card_slug"):
        if getattr(args, required) is None:
            parser.error(f"--{required.replace('_', '-')} is required for a measurement")
    args.records.mkdir(parents=True, exist_ok=True)
    if args.not_measured_reason:
        # a cell its unit could not start still gets a record, so the table's notes name it
        name = f"{args.card_slug}_policies-{args.policies}_environments-per-policy-{args.environments_per_policy}"
        write_record(args.records / f"{name}.json", {
            "policies": args.policies, "environments_per_policy": args.environments_per_policy,
            "environments": args.policies * args.environments_per_policy, "machine": args.machine,
            "unit": args.unit, "card": {"name": args.card_name or args.card_slug},
            "started_iso": datetime.now().astimezone().isoformat(), "status": "not measured",
            "reason": args.not_measured_reason, "other_processes_on_card": []})
        return 0
    return measure(args)


if __name__ == "__main__":
    raise SystemExit(main())
