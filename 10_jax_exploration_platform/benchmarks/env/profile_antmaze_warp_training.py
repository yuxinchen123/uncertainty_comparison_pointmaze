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
# the order of the cells in the document: by the environments in total, then by those per policy
CELL_ORDER = lambda record: (record["environments"], record["environments_per_policy"])
OUT_OF_MEMORY_EXIT_CODE = 77
NO_CARD_EXIT_CODE = 76


def pacific(iso: str) -> str:
    """A stored timestamp for the reader: Pacific, with the zone named."""
    return datetime.fromisoformat(iso).astimezone(PACIFIC).strftime("%Y-%m-%d %H:%M PT")


def median(values) -> float:
    """The middle value of a list (the upper of the two middle values for an even count)."""
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def study_configuration(policies: int, environments_per_policy: int, configuration_file: Path = None):
    """The study's learner and the Warp U-maze at one cell, as (PPOConfig, AntMazeConfig).

    configuration_file: the recorded file read (default `STUDY_CONFIGURATION`); another file of the
    same study measures another physics configuration, for example the stable time step.

    before: the example file's trainer (16 policies x 4 environments, 50 priming rollouts)
    after:  the same trainer with n_copies and n_envs of the cell, 1 priming rollout, and
            minibatches gathered one epoch at a time from 128 environments per policy
    """
    from experiment_configuration import read_experiment_configuration
    from exploration_platform.agents.ppo.config import PPOConfig
    from exploration_platform.envs.antmaze.am_common import AntMazeConfig
    document = read_experiment_configuration(configuration_file or STUDY_CONFIGURATION)
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
        "configuration_file": (args.configuration_file or STUDY_CONFIGURATION).name, "status": "running",
        "processor_dry_run": not on_card,
        # the model has had its ceiling since 2026-11-12; a record without this field is older
        "model_has_ceiling": True,
    }
    write_record(path, record)
    say(f"cell {policies} policies x {per_policy} environments on {args.machine} ({card['name']}); "
        f"{deadline - time.time():.0f} s of the unit are left")

    try:
        # GOAL: the study's program and a primed state.
        cfg, env_cfg = study_configuration(policies, per_policy, args.configuration_file)
        runner = Runner(cfg, bonus="rnd_next_state", env_cfg=env_cfg)
        record.update(rollout_steps=cfg.num_steps, kernel_folder=runner.env.kernel_folder,
                      # the physics steps and the fault guard of this cell's environment
                      physics_time_step=env_cfg.physics_time_step, frame_skip=env_cfg.frame_skip,
                      fault_torso_speed=env_cfg.fault_torso_speed,
                      fault_torso_angular_speed=env_cfg.fault_torso_angular_speed,
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
            # the lengths of the episodes that ended at the goal, summed, and the policies' first
            # training steps with an arrival (-1: none): an arrival this early is not a walk
            arrival_steps_sum=int(log["steps_to_goal_sum"].sum()),
            first_goal_steps=sorted(int(step) for step in log["first_goal_step"] if step >= 0),
            iterations_run=3 + args.rounds * block,
            non_finite_resets=int(np.asarray(state.env_state.nan_count).sum()),
            # the environment's fault record over all iterations of this process, summed over policies
            ceiling_zone_steps=int(np.asarray(state.env_state.ceiling_zone_steps, np.int64).sum()),
            largest_torso_speed=float(np.asarray(state.env_state.largest_torso_speed).max()),
            physics_fault_respawns=int(np.asarray(state.env_state.physics_fault_respawns, np.int64).sum()),
            largest_torso_angular_speed=float(np.asarray(state.env_state.largest_torso_angular_speed).max()),
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
    """One table row per measured cell, ordered by the environments in total and then by those per policy."""
    rows = []
    for r in sorted(records, key=CELL_ORDER):
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
            with_ceiling = sum(1 for r in measured if r.get("model_has_ceiling"))
            lines += [
                f"JAX {sample['versions']['jax']}, MuJoCo MJX {sample['versions']['mujoco-mjx']}, "
                f"warp-lang {sample['versions']['warp-lang']}, driver {sample['card']['driver']}, card "
                f"memory {sample['card']['memory_total_mib']:,} MiB; kernel folder "
                f"`{sample['kernel_folder']}`. "
                + ("Every row was measured with the ceiling in the model." if with_ceiling == len(measured)
                   else "Every row was measured before the model had its ceiling; the cells measured "
                        "again with it are in the section \"After the ceiling\"." if with_ceiling == 0
                   else f"{with_ceiling} of the {len(measured)} rows were measured with the ceiling in "
                        "the model: " + ", ".join(f"{r['policies']:,} x {r['environments_per_policy']}"
                                                  for r in sorted(measured, key=CELL_ORDER)
                                                  if r.get("model_has_ceiling")) + "."),
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
                for r in sorted(measured, key=CELL_ORDER)) + ".",
                ""]
            in_use = [f"{r['policies']:,} x {r['environments_per_policy']}: "
                      f"{r['memory']['jax_peak_bytes'] / 2 ** 30:.2f} and "
                      f"{r['memory']['warp_pool_peak_bytes'] / 2 ** 30:.2f}"
                      for r in sorted(measured, key=CELL_ORDER)
                      if r["memory"]]
            if in_use:
                lines += ["Largest memory in use as the two libraries count it themselves, JAX's and then "
                          "Warp's own pool, GiB (the column above also holds what JAX took and no longer "
                          "uses, and the card's context): " + "; ".join(in_use) + ".", ""]
            checks = [f"{r['policies']:,} x {r['environments_per_policy']}: loss finite "
                      f"{'yes' if r['loss_finite'] else 'NO'}, {r['episodes_ended']:,} episodes ended, "
                      f"{r['arrivals']:,} at the goal"
                      + (f" (after {r['arrival_steps_sum'] / r['arrivals']:.0f} steps on average)"
                         if r["arrivals"] and "arrival_steps_sum" in r else "")
                      + f", {r['non_finite_resets']} non-finite resets"
                      for r in sorted(measured, key=CELL_ORDER)]
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
    return "\n".join(lines + render_arrivals(records_folder.parent / "records_arrivals")
                     + render_after_the_ceiling(records_folder.parent))


def render_arrivals(folder: Path) -> list:
    """The section on episodes that end at the goal under random torques, from the records of
    `units/diagnose_arrivals_with_random_torques.py`; no lines when there is no such record."""
    records = [json.loads(path.read_text()) for path in sorted(folder.glob("*.json"))]
    if not records:
        return []
    names = {"jax": "MJX JAX", "warp": "MuJoCo Warp"}
    lines = [
        "## Episodes that end at the goal without a walk (H100 NVL, serval02 card 1)",
        "",
        "The measured cells above counted episodes that ended at the goal after 15 iterations of an "
        "untrained policy. The start cell and the goal cell of the U-maze are 8 m apart with a wall "
        "cell between them (the way around is about 16 m), so these are not walks. "
        "`units/diagnose_arrivals_with_random_torques.py` stepped ants with random torques (standard "
        "normal, clipped to [-1, 1], fresh at every step), the episode ending at the goal or after 700 "
        "steps, with each implementation of the physics, and kept the torso's position and velocity "
        "of every step. The walls are 2 m high. This measurement was made before the model had its "
        "ceiling.",
        "",
        *arrival_rows(records),
    ]
    for r in records:
        arrivals = r["arrivals"]
        if not arrivals:
            continue
        # a wall cell of the U-maze between the start row and the goal row: (2, 1) or (2, 2)
        over = sum(1 for a in arrivals if [2, 1] in a["cells_in_order"] or [2, 2] in a["cells_in_order"])
        steps = sorted(a["step_of_the_episode"] for a in arrivals)
        heights = sorted(a["largest_height_of_the_episode"] for a in arrivals)
        lines.append(
            f"- {names[r['physics']]}: {over} of the {len(arrivals)} episodes that ended at the goal went "
            f"through the wall cells between the start row and the goal row, above the wall: the largest "
            f"torso height of these episodes was between {heights[0]:.2f} m and {heights[-1]:.2f} m, and "
            f"they ended at the goal after {steps[0]} to {steps[-1]} steps.")
    lines += ["", "In both implementations some ants rose above the walls and crossed them; the records hold "
              "every such episode that ended at the goal (`records_arrivals/`). Nothing in the environment "
              "of that time ended or discarded an episode in which the torso was above the walls.", ""]
    return lines


def arrival_rows(records: list) -> list:
    """The table of the random-torque measurement: one row per physics implementation."""
    names = {"jax": "MJX JAX", "warp": "MuJoCo Warp"}
    lines = [
        "| physics | environments | environment<br>steps | episodes<br>ended | ended at<br>the goal | "
        "steps with the torso<br>above the walls' top | steps faster<br>than 10 m/s | "
        "steps with the torso<br>inside a wall cell | steps with the torso<br>outside the maze | "
        "largest torso<br>height (m) | largest torso<br>speed (m/s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in records:
        lines.append(
            f"| {names[r['physics']]} | {r['environments']:,} | {r['environment_steps']:,} | "
            f"{r['episodes_ended']:,} | {r['episodes_ended_at_the_goal']} | "
            f"{r['steps_with_the_torso_above_the_walls_top']:,} | "
            f"{r['steps_faster_than_10_metres_per_second']:,} | "
            f"{r['steps_with_the_torso_inside_a_wall_cell']:,} | "
            f"{r['steps_with_the_torso_outside_the_maze']:,} | {r['largest_torso_height']:.2f} | "
            f"{r['largest_torso_speed']:.2f} |")
    return lines + [""]


def render_after_the_ceiling(profile_folder: Path) -> list:
    """The section "After the ceiling" from the records made with the ceiling in the model: the
    checks against C MuJoCo before and after, the nine ants on processors, the random-torque
    measurement again, and the speed cells again. A part whose record does not exist is left out;
    no lines when none exists."""
    read = lambda path: json.loads(path.read_text()) if path.exists() else None
    checks_folder = profile_folder.parent / "antmaze_warp"
    before = read(checks_folder / "antmaze_warp_checks_processors_before-the-ceiling.json")
    after = read(checks_folder / "antmaze_warp_checks_processors.json")
    ants = read(profile_folder / "records_ceiling" / "ordinary-and-thrown-ants_processors.json")
    torque_folder = profile_folder / "records_arrivals_after_the_ceiling"
    torques = [json.loads(path.read_text()) for path in sorted(torque_folder.glob("*.json"))]
    earlier_torques = {r["physics"]: r for r in (json.loads(path.read_text()) for path in sorted(
        (profile_folder / "records_arrivals").glob("*.json")))}
    cells = [json.loads(path.read_text()) for path in
             sorted((profile_folder / "records_after_the_ceiling").glob("*.json"))]
    earlier_cells = {(r["policies"], r["environments_per_policy"]): r for r in (
        json.loads(path.read_text()) for path in sorted((profile_folder / "records").glob("h100-nvl_*.json")))
        if r.get("status") == "measured"}
    if not (before and after) and not ants and not torques and not cells:
        return []
    names = {"jax": "MJX JAX", "warp": "MuJoCo Warp", "c_mujoco": "C MuJoCo"}
    lines = [
        "## After the ceiling",
        "",
        "Since 2026-11-12 the model of all three maps holds one plane on the wall tops (2 m) whose "
        "normal points down and which collides with the torso sphere only (`envs/antmaze/am_common.py`). "
        "The torso sphere (radius 0.25 m) comes within the contact margin of it when the torso centre "
        "is at 1.74 m. The environment also keeps a fault record per policy "
        "(`envs/antmaze/faults.py`): the environment steps with the torso centre above 1.5 m (the "
        "ceiling zone) after any physics step, and the largest torso speed; the training command "
        "writes both into every window record of an AntMaze unit.",
        "",
    ]

    # GOAL: the environment's checks against C MuJoCo of the same model, before and after.
    if before and after:
        check = "check a: one step against C MuJoCo and against JAX, 50 states"
        lines += [
            "### Checks against C MuJoCo of the same model, before and after (processors)",
            "",
            f"`benchmarks/env/antmaze_warp_checks.py` on {after['host']}, MuJoCo Warp on the processor device: "
            f"before the ceiling {before['measured_pacific']} (Slurm job {before['slurm_job']}), with it "
            f"{after['measured_pacific']} (Slurm job {after['slurm_job']}). The C MuJoCo reference loads the "
            "same XML, so it has the ceiling too. One environment step from 50 states along a C-MuJoCo "
            "random-torque trajectory; the largest absolute position difference, in metres:",
            "",
            "| kind of state | pair | states | largest difference<br>before (m) | largest difference<br>with the ceiling (m) | "
            "median<br>before (m) | median with<br>the ceiling (m) | gate of<br>`spec.md` |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
        gates = {("contact_free", "warp_against_c"): "below 1e-4", ("contact_free", "jax_against_c"): "below 1e-4",
                 ("in_contact", "warp_against_c"): "below 5e-2", ("in_contact", "jax_against_c"): "below 5e-2"}
        for kind in ("contact_free", "in_contact"):
            for pair in ("warp_against_c", "jax_against_c", "warp_against_jax"):
                b, a = before["checks"][check][kind][pair], after["checks"][check][kind][pair]
                lines.append(f"| {kind.replace('_', ' ')} | {pair.replace('_', ' ')} | {a['count']} | "
                             f"{b['largest']:.2e} | {a['largest']:.2e} | {b['median']:.2e} | {a['median']:.2e} | "
                             f"{gates.get((kind, pair), '')} |")
        divergence = "check a: 20 shared-torque steps from the spawn against C MuJoCo"
        resting = "check b: resting height after 1,200 passive steps"
        part = lambda record, name, implementation, field: record["checks"][name][implementation][field]
        lines += [
            "",
            "- 20 shared-torque steps from the spawn, largest position difference from C MuJoCo before the "
            "first contact: Warp " + f"{part(before, divergence, 'warp', 'largest_before_the_first_contact'):.2e} m "
            f"before and {part(after, divergence, 'warp', 'largest_before_the_first_contact'):.2e} m with the "
            f"ceiling; JAX {part(before, divergence, 'jax', 'largest_before_the_first_contact'):.2e} m and "
            f"{part(after, divergence, 'jax', 'largest_before_the_first_contact'):.2e} m. Over all 20 steps: Warp "
            f"{part(before, divergence, 'warp', 'largest_over_all_steps'):.3f} m and "
            f"{part(after, divergence, 'warp', 'largest_over_all_steps'):.3f} m; JAX "
            f"{part(before, divergence, 'jax', 'largest_over_all_steps'):.3f} m and "
            f"{part(after, divergence, 'jax', 'largest_over_all_steps'):.3f} m (gate: below 4 m).",
            "- Resting height after 1,200 passive steps (reference 0.38248 m): C MuJoCo "
            f"{part(before, resting, 'c_mujoco', 'height'):.5f} m before and "
            f"{part(after, resting, 'c_mujoco', 'height'):.5f} m with the ceiling; Warp "
            f"{part(before, resting, 'warp', 'height'):.5f} m and {part(after, resting, 'warp', 'height'):.5f} m; "
            f"JAX {part(before, resting, 'jax', 'height'):.5f} m and {part(after, resting, 'jax', 'height'):.5f} m.",
            "",
        ]

    # GOAL: the nine ants of the ceiling test, stepped one physics step at a time on processors.
    if ants:
        lines += [
            "### Which ants touch the ceiling (processors, one physics step at a time)",
            "",
            f"`units/diagnose_ordinary_and_thrown_ants.py` on {ants['host']}, {pacific(ants['measured_iso'])} "
            f"(Slurm job {ants['slurm_job']}): nine ants from the spawn, {ants['physics_steps']:,} physics steps "
            "of 0.01 s each, in C MuJoCo of the same model and in both implementations (float32, MuJoCo "
            "Warp on the processor device). The spawn height of the torso centre is 0.75 m; the ceiling is "
            "touched from 1.74 m; the wall tops are at 2 m.",
            "",
            "| ant | largest torso height (m)<br>C MuJoCo | largest torso height (m)<br>MJX JAX | "
            "largest torso height (m)<br>MuJoCo Warp | physics steps with a<br>ceiling contact<br>"
            "C MuJoCo, JAX, Warp | moved on the floor plane (m)<br>C MuJoCo, JAX, Warp |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        rows = ants["nine_ants"]
        for c, j, w in zip(rows["c_mujoco"], rows["jax"], rows["warp"]):
            lines.append(
                f"| {c['case']} | {c['largest_torso_height']:.3f} | {j['largest_torso_height']:.3f} | "
                f"{w['largest_torso_height']:.3f} | {c['physics_steps_with_a_ceiling_contact']}, "
                f"{j['physics_steps_with_a_ceiling_contact']}, {w['physics_steps_with_a_ceiling_contact']} | "
                f"{c['moved_on_the_floor_plane']:.2f}, {j['moved_on_the_floor_plane']:.2f}, "
                f"{w['moved_on_the_floor_plane']:.2f} |")
        thrown = ants["thrown_upward_c_mujoco"]
        lines += [
            "",
            "- An ant thrown upward from the spawn in C MuJoCo, largest height of the torso centre: "
            + "; ".join(f"{row['speed']:.0f} m/s: {row['largest_torso_height']:.3f} m" for row in thrown) + ".",
        ]
        for implementation, result in ants["one_step_in_ceiling_contact"].items():
            contact = set(result["states_in_ceiling_contact"])
            differences = [d for k, d in zip(result["c_states"], result["largest_position_difference_per_state"])
                           if k in contact]
            lines.append(
                f"- One physics step of {names[implementation]} from {len(differences)} C-MuJoCo states of the "
                "ant thrown at 10 m/s that are in contact with the ceiling, against C MuJoCo's next state: "
                f"largest position difference {max(differences):.2e} m (the in-contact gate of `spec.md` is 5e-2 m).")
        # GOAL: the 36 periodic patterns, one row per torque size and frequency (four patterns each).
        patterns = ants["periodic_patterns_c_mujoco"]
        touching = [p for p in patterns if p["physics_steps_with_a_ceiling_contact"]]
        lines += [
            "",
            "Periodic leg torques in C MuJoCo, 10 s each from the spawn: each hip gets a sine and its ankle "
            "the same sine a quarter period ahead or behind, with diagonal leg pairs in turn or one leg after "
            f"the other. {len(touching)} of the {len(patterns)} patterns brought the torso to the ceiling: the "
            f"ant's own legs lift it off the floor (the model's motors give {ants['motor_torque_at_full_control']:g} "
            f"N m per joint at full control to a body of {ants['ant_mass']:.2f} kg). The Gymnasium Ant "
            "environment itself calls a torso height from 0.2 m to 1.0 m healthy.",
            "",
            "| share of the<br>full torque | frequency (Hz) | largest torso height<br>of the four patterns (m) | "
            "patterns that touched<br>the ceiling | most physics steps<br>with a ceiling contact | "
            "moved on the floor<br>plane (m) |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for torque in sorted({p["torque"] for p in patterns}):
            for hertz in sorted({p["hertz"] for p in patterns}):
                group = [p for p in patterns if p["torque"] == torque and p["hertz"] == hertz]
                heights = [p["largest_torso_height"] for p in group]
                moved = [p["moved_on_the_floor_plane"] for p in group]
                steps = [p["physics_steps_with_a_ceiling_contact"] for p in group]
                lines.append(f"| {torque:g} | {hertz:g} | {min(heights):.2f} to {max(heights):.2f} | "
                             f"{sum(1 for x in steps if x)} of {len(group)} | {max(steps)} | "
                             f"{min(moved):.2f} to {max(moved):.2f} |")
        lines.append("")
        # GOAL: random torques in C MuJoCo itself, with the ceiling and without it.
        if "random_torques_c_mujoco" in ants:
            both = ants["random_torques_c_mujoco"]
            sample = both["with_the_ceiling"]
            lines += [
                f"Random torques in C MuJoCo itself (float64), {sample['ants']} ants from the spawn for "
                f"{sample['environment_steps'] // sample['ants']} environment steps of 5 physics steps each, "
                "no reset and no goal, the same torques on the model with the ceiling and on the same model "
                "without it (the torso is looked at after the last physics step of each environment step):",
                "",
                "| model in C MuJoCo | environment<br>steps | steps with the torso<br>above the walls' top | "
                "ants that were<br>above the walls' top | steps with the torso<br>inside a wall cell | "
                "steps with the torso<br>outside the maze | steps in the<br>ceiling zone | "
                "ants that were in<br>the ceiling zone | steps faster<br>than 5 m/s | steps faster<br>than 10 m/s | "
                "largest torso<br>height (m) | largest torso<br>speed (m/s) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
            for name, key in (("without the ceiling", "without_the_ceiling"), ("with the ceiling", "with_the_ceiling")):
                r = both[key]
                lines.append(
                    f"| {name} | {r['environment_steps']:,} | {r['steps_with_the_torso_above_the_walls_top']:,} | "
                    f"{r['ants_with_the_torso_above_the_walls_top']:,} of {r['ants']:,} | "
                    f"{r['steps_with_the_torso_inside_a_wall_cell']:,} | "
                    f"{r['steps_with_the_torso_outside_the_maze']:,} | "
                    f"{r['steps_with_the_torso_in_the_ceiling_zone']:,} | "
                    f"{r['ants_with_the_torso_in_the_ceiling_zone']:,} of {r['ants']:,} | "
                    f"{r['steps_faster_than_5_metres_per_second']:,} | "
                    f"{r['steps_faster_than_10_metres_per_second']:,} | {r['largest_torso_height']:.2f} | "
                    f"{r['largest_torso_speed']:.2f} |")
            without = both["without_the_ceiling"]
            lines += [
                "",
                f"Without the ceiling, {without['ants_with_the_torso_above_the_walls_top']:,} of the "
                f"{without['ants']:,} ants rose above the wall tops in C MuJoCo and "
                f"{without['ants_with_the_torso_inside_a_wall_cell']} of them were over a wall cell: the ant "
                "model itself leaves the floor under random torques, in the reference's own stepper as in "
                "the two implementations.",
                ""]

    # GOAL: the random-torque measurement again, beside the earlier one.
    if torques:
        lines += [
            "### Random torques again (H100 NVL, serval02 card 1)",
            "",
            "The measurement of the section above with the ceiling in the model: 16,384 ants, 1,400 "
            "environment steps, random torques, the episode ending at the goal or after 700 steps.",
            "",
            *arrival_rows(torques),
            "| physics | steps with the torso in<br>the ceiling zone<br>(above 1.5 m) | steps with the torso at<br>"
            "the ceiling (above 1.74 m) | episodes with a step<br>in the ceiling zone | the environment's own<br>"
            "count of ceiling-zone<br>steps | the environment's own<br>largest torso<br>speed (m/s) | "
            "steps faster<br>than 5 m/s | steps faster<br>than 2 m/s |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in torques:
            own = r["environment_fault_record"]
            lines.append(
                f"| {names[r['physics']]} | {r['steps_with_the_torso_in_the_ceiling_zone']:,} | "
                f"{r['steps_with_the_torso_at_the_ceiling']:,} | "
                f"{r['episodes_with_the_torso_in_the_ceiling_zone']:,} of {r['episodes_ended']:,} | "
                f"{own['ceiling_zone_steps']:,} | {own['largest_torso_speed']:.2f} | "
                f"{r['steps_faster_than_5_metres_per_second']:,} | {r['steps_faster_than_2_metres_per_second']:,} |")
        lines += [
            "",
            "The script looks at the torso after the last physics step of each environment step; the "
            "environment's own count looks after each of the 5 physics steps, so it is the larger one.",
            "",
        ]
        for r in torques:
            share = lambda count: f"{100 * count / r['environment_steps']:.4f}%"
            height, speed = r["torso_height_percentiles"], r["torso_speed_percentiles"]
            earlier = earlier_torques.get(r["physics"])
            arrivals = r["arrivals"]
            around = [a for a in arrivals if a["went_around_the_wall"]]
            text = (
                f"- {names[r['physics']]}: {r['steps_with_the_torso_above_the_walls_top']:,} steps with the torso "
                f"above the wall tops"
                + (f" ({earlier['steps_with_the_torso_above_the_walls_top']:,} before the ceiling)" if earlier else "")
                + f", {r['steps_with_the_torso_inside_a_wall_cell']:,} inside a wall cell"
                + (f" ({earlier['steps_with_the_torso_inside_a_wall_cell']:,} before)" if earlier else "")
                + f", {r['steps_with_the_torso_outside_the_maze']:,} outside the maze"
                + (f" ({earlier['steps_with_the_torso_outside_the_maze']:,} before)" if earlier else "")
                + f". The torso was in the ceiling zone in {share(r['steps_with_the_torso_in_the_ceiling_zone'])} "
                f"of the steps and at the ceiling in {share(r['steps_with_the_torso_at_the_ceiling'])}. Torso "
                f"height: median {height['50']:.2f} m, 99% of the steps below {height['99']:.2f} m, 99.99% below "
                f"{height['99.99']:.2f} m. Torso speed: median {speed['50']:.2f} m/s, 99% of the steps below "
                f"{speed['99']:.2f} m/s, 99.99% below {speed['99.99']:.2f} m/s. {r['non_finite_resets']} "
                f"non-finite resets. Episodes that ended at the goal: {len(arrivals)}"
                + (f" ({earlier['episodes_ended_at_the_goal']} before)" if earlier else ""))
            if arrivals:
                heights = sorted(a["largest_height_of_the_episode"] for a in arrivals)
                steps = sorted(a["step_of_the_episode"] for a in arrivals)
                text += (f"; {len(around)} of them went around the wall through the open cells, "
                         f"{len(arrivals) - len(around)} did not; their largest torso height was between "
                         f"{heights[0]:.2f} m and {heights[-1]:.2f} m, and they ended after {steps[0]} to "
                         f"{steps[-1]} steps")
            lines.append(text + ".")
        lines.append("")
        # GOAL: the two implementations on the card beside C MuJoCo on processors, as shares of the steps.
        if ants and "random_torques_c_mujoco" in ants:
            c = ants["random_torques_c_mujoco"]["with_the_ceiling"]
            percent = lambda count, steps: f"{100 * count / steps:.4f}%"
            lines += [
                "The same counts as shares of the environment steps, beside C MuJoCo with the same kind of "
                f"torques ({c['environment_steps']:,} steps on processors, section above; C MuJoCo ran "
                "without resets and without the goal):",
                "",
                "| physics | steps in the<br>ceiling zone | steps faster<br>than 5 m/s | steps faster<br>than 10 m/s | "
                "largest torso<br>speed (m/s) |",
                "|---|---:|---:|---:|---:|",
                f"| C MuJoCo | {percent(c['steps_with_the_torso_in_the_ceiling_zone'], c['environment_steps'])} | "
                f"{percent(c['steps_faster_than_5_metres_per_second'], c['environment_steps'])} | "
                f"{percent(c['steps_faster_than_10_metres_per_second'], c['environment_steps'])} "
                f"({c['steps_faster_than_10_metres_per_second']} steps) | {c['largest_torso_speed']:.2f} |",
            ]
            for r in sorted(torques, key=lambda record: record["physics"], reverse=True):
                lines.append(
                    f"| {names[r['physics']]} | "
                    f"{percent(r['steps_with_the_torso_in_the_ceiling_zone'], r['environment_steps'])} | "
                    f"{percent(r['steps_faster_than_5_metres_per_second'], r['environment_steps'])} | "
                    f"{percent(r['steps_faster_than_10_metres_per_second'], r['environment_steps'])} "
                    f"({r['steps_faster_than_10_metres_per_second']:,} steps) | {r['largest_torso_speed']:.2f} |")
            lines.append("")
            for r in torques:
                fast = r["steps_faster_than_10_metres_per_second"]
                if fast:
                    expected = fast * c["environment_steps"] / r["environment_steps"]
                    lines += [
                        f"At the rate of {names[r['physics']]} ({fast:,} steps faster than 10 m/s in "
                        f"{r['environment_steps']:,}), C MuJoCo's {c['environment_steps']:,} steps would hold "
                        f"about {expected:.0f} such steps; they hold {c['steps_faster_than_10_metres_per_second']}.",
                        ""]

    # GOAL: the speed cells measured again, beside the earlier values of the same cells.
    measured = [r for r in cells if r.get("status") == "measured"]
    if cells:
        lines += ["### Speed with the ceiling in the model (H100 NVL, serval02 card 1)", ""]
    if measured:
        lines += [
            "| policies | environments<br>per policy | environments<br>in total | seconds per<br>iteration | "
            "total steps per<br>second (millions) | steps per second<br>per policy | "
            "hours per million<br>steps per policy | rollout<br>share | update<br>share | "
            "peak card<br>memory (GiB) | largest<br>pool use | measured |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            *table_rows(measured), ""]
        for r in sorted(measured, key=CELL_ORDER):
            earlier = earlier_cells.get((r["policies"], r["environments_per_policy"]))
            held = lambda record: (record["memory"]["process_mib_on_card"] or record["memory"]["card_mib_used"]) / 1024
            text = (f"- {r['policies']:,} x {r['environments_per_policy']}: "
                    + (f"before the ceiling {earlier['seconds_per_iteration']:.4f} s per iteration, "
                       f"{earlier['total_steps_per_second_millions']:.3f} million steps per second and "
                       f"{held(earlier):.2f} GiB; with it {r['seconds_per_iteration']:.4f} s "
                       f"({100 * (r['seconds_per_iteration'] / earlier['seconds_per_iteration'] - 1):+.1f}%), "
                       f"{r['total_steps_per_second_millions']:.3f} million and {held(r):.2f} GiB. "
                       if earlier else "not measured before the ceiling. ")
                    + f"First two iterations (with compilation) {r['first_two_iterations_seconds'][0]:.0f} s and "
                    f"{r['first_two_iterations_seconds'][1]:.0f} s; loss finite "
                    f"{'yes' if r['loss_finite'] else 'NO'}; {r['episodes_ended']:,} episodes ended, "
                    f"{r['arrivals']:,} at the goal; {r['non_finite_resets']} non-finite resets; "
                    f"{r['ceiling_zone_steps']:,} environment steps in the ceiling zone of "
                    f"{r['iterations_run'] * r['rollout_steps'] * r['environments']:,}; largest torso speed "
                    f"{r['largest_torso_speed']:.2f} m/s.")
            lines.append(text)
        lines.append("")
    unused = [r for r in cells if r.get("status") != "measured"]
    if unused:
        lines += ["Not measured or not used:", ""]
        for r in unused:
            reason = (f"another process was on the card: {r['other_processes_on_card']}"
                      if r["status"] == "discarded" else
                      r.get("reason", "the process ended before the cell was measured"))
            lines.append(f"- {r['policies']:,} policies x {r['environments_per_policy']} environments, "
                         f"{pacific(r['started_iso'])}: {reason}.")
        lines.append("")
    return lines


def main(argv=None) -> int:
    """Measure one cell, or write the Markdown table from a folder of records."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", type=Path, required=True, help="the folder of the cells' records")
    parser.add_argument("--render", action="store_true", help="write the Markdown table and measure nothing")
    parser.add_argument("--markdown", type=Path, help="with --render: the file to write")
    parser.add_argument("--configuration-file", type=Path,
                        help="the recorded configuration file of the cell's learner and environment "
                             f"(default configs/experiments/{STUDY_CONFIGURATION.name})")
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
