"""Write the queue this run's chosen submission plan asks for, and the run's launch-time files.

One run is one sweep. This run is an EXTENSION of train run 1.1
(`runs/2026-08-16-20-39_..._first-batch`): it takes the single best configuration each of that
run's four algorithm arms found and re-runs it at 1,024 copies for about 10^9 environment steps
per copy — four times the copies of one parent cell and a hundred times its step budget.

A unit here is one algorithm arm at one configuration, 1,024 copies. A unit may be CUT INTO CHUNKS
by copy index, because the copies are independent seeds and a chunk of them runs at a better
per-copy rate on its own card; `code/plan_submission.py` decides how many chunks each unit gets and
which card each chunk goes to, and this module writes exactly the queue that plan describes. Each
chunk is one queue entry, one job, one shard.

**How a chunk stays a slice of the same run rather than a different run.** The seed index of a copy
keys both its initial weights and the environment draws it meets, so chunk j of k passes
`--copy-seed-offset j x (1024 / k)` and holds copy indices `j x (1024/k)` onward: the k chunks
partition 0..1023, no index appears twice, and copy i of the union has the weights and the
environments copy i of a single 1,024-copy run would have had. The policy's own sampling stream is
keyed by `--run-seed`, which is the chunk index, so no copy shares its action noise with a copy of
another chunk either.

Run after the plan, before any science submission:
  PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python code/plan_submission.py
  PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python code/build_queue.py
"""
import json
import subprocess
import sys
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent.parent
PLATFORM_ROOT = RUN_DIR.parent.parent
REPO_ROOT = PLATFORM_ROOT.parent
sys.path.insert(0, str(PLATFORM_ROOT / "src"))
sys.path.insert(0, str(PLATFORM_ROOT / "scripts"))
sys.path.insert(0, str(RUN_DIR / "code"))

import plan_submission as ps  # noqa: E402  (the run's shape and the chosen plan live there)

PARENT_RUN = ("2026-08-16-20-39_jax-ppo_pointmaze-large-nonoise_full-batch_rollout-128"
              "_envs-per-copy-4_steps-per-copy-10M"
              "_bonus-rnd-visit-count-sqrt-visit-count-linear-none"
              "_learning-rate-1e-3-1e-4-1e-5_intrinsic-weight-1e-5-to-1e5_copies-per-cell-256"
              "_first-batch")
PARENT_COPIES_PER_CELL = 256
PARENT_STEPS_PER_COPY = 10035200
LEARNING_RATE = "1e-3"


def chunk_id(chunk: dict) -> str:
    """The chunk's name, with every load-bearing knob spelled into it.

    before: {"order": 1, "arm": "rnd_next_state", "intrinsic_weight": "10", "chunk": 1,
             "chunks": 2, "copies": 512, "copy_index_first": 512, "copy_index_last": 1023}
    after:  "unit-1_rnd-next-state_learning-rate-1e-3_intrinsic-weight-10_chunk-2-of-2_
             copies-512_copy-index-512-1023"
    A chunk of one is still named as such, so the shard name says outright whether a unit was cut.
    """
    weight_part = ("no-intrinsic-weight" if chunk["intrinsic_weight"] is None
                   else f"intrinsic-weight-{chunk['intrinsic_weight']}")
    return (f"unit-{chunk['order']}_{chunk['arm'].replace('_', '-')}"
            f"_learning-rate-{LEARNING_RATE}_{weight_part}"
            f"_chunk-{chunk['chunk'] + 1}-of-{chunk['chunks']}_copies-{chunk['copies']}"
            f"_copy-index-{chunk['copy_index_first']}-{chunk['copy_index_last']}")


def chunk_record(chunk: dict) -> dict:
    """One queue entry: the identity of the chunk and the exact arguments that run it."""
    arguments = ["--unit-id", chunk_id(chunk), "--bonus", chunk["arm"],
                 "--learning-rates", LEARNING_RATE]
    # the arm with no bonus has no weight, so its single cell is the learning rate alone
    if chunk["intrinsic_weight"] is not None:
        arguments += ["--intrinsic-weights", chunk["intrinsic_weight"]]
    arguments += ["--copies-per-cell", str(chunk["copies"]),
                  "--rollout-steps", str(ps.ROLLOUT_STEPS),
                  "--envs-per-copy", str(ps.ENVS_PER_COPY),
                  "--update-style", "full_batch",
                  "--iterations", str(ps.ITERATIONS),
                  "--window-iterations", str(ps.WINDOW_ITERATIONS),
                  "--episode-steps", str(ps.EPISODE_STEPS),
                  "--base-seed", "0",
                  # the chunk index keys the policy's sampling stream, so two chunks of one unit
                  # never draw the same action noise
                  "--run-seed", str(chunk["chunk"]),
                  # and the copy-index offset keys the weights and the environments, so the chunk
                  # holds exactly its slice of the whole run's copies
                  "--copy-seed-offset", str(chunk["copy_index_first"]),
                  "--track-coverage"]
    return {
        "unit_id": chunk_id(chunk),
        "order": chunk["order"],
        "bonus": chunk["arm"],
        "extends_parent_run": PARENT_RUN,
        "chunk": chunk["chunk"],
        "chunks": chunk["chunks"],
        "copies": chunk["copies"],
        "copies_per_cell": chunk["copies"],
        "copies_in_the_whole_unit": ps.COPIES_PER_UNIT,
        "copy_index_first": chunk["copy_index_first"],
        "copy_index_last": chunk["copy_index_last"],
        "cells": 1,
        "learning_rate": float(LEARNING_RATE),
        "intrinsic_weight": (None if chunk["intrinsic_weight"] is None
                             else float(chunk["intrinsic_weight"])),
        "iterations": ps.ITERATIONS,
        "window_iterations": ps.WINDOW_ITERATIONS,
        "windows": ps.ITERATIONS // ps.WINDOW_ITERATIONS,
        "env_steps_per_copy": ps.STEPS_PER_COPY,
        "planned_node": chunk["node"],
        "planned_node_class": chunk["node_class"],
        "planned_seconds": chunk["seconds"],
        "planned_rate_source": chunk["rate_source"],
        "arguments": arguments,
    }


def check_the_chunks_partition_every_unit(records: list) -> None:
    """Fail loudly unless each unit's chunks cover 0..1023 exactly once between them.

    This is the invariant that makes the union of the shards the 1,024-copy run: if two chunks
    overlapped, some copies would be counted twice and others never run, and the aggregate would be
    quietly wrong rather than obviously broken.
    """
    by_unit = {}
    for record in records:
        by_unit.setdefault(record["order"], []).append(record)
    for order, group in sorted(by_unit.items()):
        covered = []
        for record in group:
            covered += list(range(record["copy_index_first"], record["copy_index_last"] + 1))
        if sorted(covered) != list(range(ps.COPIES_PER_UNIT)):
            raise SystemExit(
                f"unit {order}'s {len(group)} chunks do not partition its "
                f"{ps.COPIES_PER_UNIT} copies: {len(covered)} indices covered, "
                f"{len(set(covered))} of them distinct")


def as_yaml_text(value, indent: int = 0) -> str:
    """Write nested dictionaries, lists and scalars as YAML text; the platform's own writer."""
    from run_training import as_yaml
    return as_yaml(value, indent)


def resolved_trainer_configs(records: list) -> dict:
    """Every chunk's fully resolved trainer configuration, keyed by chunk id.

    Resolved through the same `sweep_config` the runner calls, so the file records what will
    actually run rather than what was typed.
    """
    from exploration_platform.training.sweep import sweep_config
    resolved = {}
    for record in records:
        weights = () if record["intrinsic_weight"] is None else (record["intrinsic_weight"],)
        config = sweep_config(learning_rates=(record["learning_rate"],), betas=weights,
                              copies_per_group=record["copies"], style="full_batch",
                              n_envs=ps.ENVS_PER_COPY, num_steps=ps.ROLLOUT_STEPS, base_seed=0,
                              track_coverage=True,
                              copy_seed_offset=record["copy_index_first"])
        resolved[record["unit_id"]] = {field: getattr(config, field)
                                       for field in sorted(config.__dataclass_fields__)}
    return resolved


def write_launch_files(records: list, plan: dict, commit: str) -> None:
    """Write command.txt, manifest.yaml and config_resolved.yaml before the first submission."""
    (RUN_DIR / "command.txt").write_text(
        "# the sweep as a whole: measure the rates, plan the split, build the queue, submit\n"
        f"RUN_DIR={RUN_DIR} bash {RUN_DIR}/code/submit_probe.sh <node> 00:40:00 "
        "<copy counts> <arms>\n"
        "PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python "
        f"{RUN_DIR}/code/measured_cells.py\n"
        "PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python "
        f"{RUN_DIR}/code/plan_submission.py\n"
        "PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python "
        f"{RUN_DIR}/code/build_queue.py\n"
        f"# then, per chunk, one independent job on the node the plan named:\n"
        f"#   RUN_DIR={RUN_DIR} bash code/submit_one.sh real <chunk_id> <node> 4-00:00:00\n"
        f"# code commit at launch: {commit}\n"
        "# one chunk runs as:\n"
        + "".join(
            "PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python "
            f"{PLATFORM_ROOT}/scripts/run_training.py --run-dir {RUN_DIR} "
            + " ".join(record["arguments"]) + "\n" for record in records))

    by_arm = {}
    for record in records:
        by_arm.setdefault(record["bonus"], []).append(record)
    manifest = {
        "schema_version": 1,
        "run_id": RUN_DIR.name,
        "kind": "extension",
        "extends": {
            "parent_run": PARENT_RUN,
            "parent_run_path": f"runs/{PARENT_RUN}",
            "what_changed": (f"each arm's best configuration of the parent run, re-run at "
                             f"{ps.COPIES_PER_UNIT} copies for {ps.STEPS_PER_COPY} environment "
                             f"steps per copy: 4 times the copies of one parent cell and about 100 "
                             f"times its step budget"),
            "configurations_extended": {
                arm: {"learning_rate": group[0]["learning_rate"],
                      "intrinsic_weight": ("none — this arm has no bonus"
                                           if group[0]["intrinsic_weight"] is None
                                           else group[0]["intrinsic_weight"]),
                      "parent_copies": PARENT_COPIES_PER_CELL,
                      "extension_copies": ps.COPIES_PER_UNIT,
                      "extension_chunks": len(group),
                      "parent_env_steps_per_copy": PARENT_STEPS_PER_COPY,
                      "extension_env_steps_per_copy": ps.STEPS_PER_COPY}
                for arm, group in by_arm.items()},
        },
        "description": ("The extension of train run 1.1 at a thousand million environment steps "
                        "per copy. Four algorithm arms, each at the single configuration it won "
                        "with in the parent run: random network distillation at intrinsic weight "
                        "10, the two oracle visit-count bonuses at intrinsic weight 1, and no "
                        "bonus at all, every one at learning rate 1e-3. 1,024 copies per arm, "
                        "seeded 0 to 1,023, 1,000,038,400 environment steps per copy. A unit is "
                        "cut into chunks by copy index where the submission plan's arithmetic says "
                        "that finishes the whole set sooner."),
        "git": {"commit": commit},
        "specs": {
            "env": "pointmaze_large_cont400_nonoise@1",
            "agent": "ppo_full_batch@1",
            "bonus": ("rnd_next_state@1 + gt_position_velocity_sqrt@1 + "
                      "gt_position_velocity_linear@1 + none@1"),
            "update_schedule": ("post_rollout_update@1 for the visit-count arms, per_rollout@1 for "
                                "random network distillation, none@1 for the arm with no bonus"),
        },
        "compile_signature": {
            "copies_per_unit": ps.COPIES_PER_UNIT,
            "copies_per_chunk": sorted({record["copies"] for record in records}),
            "envs_per_copy": ps.ENVS_PER_COPY, "rollout_steps": ps.ROLLOUT_STEPS,
            "update_style": "full_batch",
            "epochs": "not applicable — full_batch takes one gradient step on the whole rollout",
            "minibatches": "not applicable — full_batch takes one gradient step on the whole "
                           "rollout",
            "dtype": "float32", "stats_dtype": "float64",
            "iterations": ps.ITERATIONS, "window_iterations": ps.WINDOW_ITERATIONS,
            "windows": ps.ITERATIONS // ps.WINDOW_ITERATIONS,
            "env_steps_per_copy": ps.STEPS_PER_COPY,
        },
        "seeding": {
            "base_seed": 0, "mode": "paired",
            "run_seed": "the chunk index, so two chunks of one unit never draw the same action "
                        "noise",
            "copy_seed_offset": "the chunk's first copy index, so the chunks of a unit partition "
                                "copies 0..1023 and copy i of the union has the weights and the "
                                "environment draws copy i of a single 1,024-copy run would have",
        },
        "submission_plan": {
            "generated_at_pacific": plan["generated_at_pacific"],
            "makespan_hours": round(plan["chosen"]["makespan_seconds"] / 3600, 3),
            "chunk_count": plan["chosen"]["chunk_count"],
            "splits_per_unit": plan["chosen"]["splits"],
            "written_by": "code/plan_submission.py, recorded in code/submission_plan.json",
        },
        "units": [record["unit_id"] for record in records],
        "artifacts": {"metrics": "metrics.jsonl", "summary": "summary.json",
                      "shards": "data/*.jsonl", "aggregator": "code/aggregate.py",
                      "plan": "code/submission_plan.json",
                      "own_rate_measurements": "code/measured_cells.json"},
        "checkpoints": "none — this platform saves no model state, so a chunk resumes by being "
                       "re-run whole",
    }
    (RUN_DIR / "manifest.yaml").write_text(as_yaml_text(manifest) + "\n")

    resolved = {"chunks": resolved_trainer_configs(records),
                "environment": {"specification": "pointmaze_large_cont400_nonoise@1",
                                "map_name": "large", "start_cell": "(7, 1)", "goal_cell": "(1, 10)",
                                "position_noise": 0.0,
                                "max_episode_steps": ps.EPISODE_STEPS,
                                "continuing_task": True, "goal_radius": 0.45, "reward_shift": 0.0},
                "runtime": {"interpreter": "/p/rlprojects/RND/.venvs/platform_jax/bin/python",
                            "iterations": ps.ITERATIONS,
                            "window_iterations": ps.WINDOW_ITERATIONS}}
    (RUN_DIR / "config_resolved.yaml").write_text(as_yaml_text(resolved) + "\n")


def main() -> None:
    """Write one queue entry per planned chunk, plus the run's launch-time files."""
    plan_path = RUN_DIR / "code" / "submission_plan.json"
    if not plan_path.exists():
        raise SystemExit(f"no plan at {plan_path}; run code/plan_submission.py first")
    plan = json.loads(plan_path.read_text())

    pending = RUN_DIR / "queue" / "pending"
    pending.mkdir(parents=True, exist_ok=True)
    records = [chunk_record(chunk) for chunk in
               sorted(plan["chosen"]["assignment"], key=lambda c: (c["order"], c["chunk"]))]
    check_the_chunks_partition_every_unit(records)
    for record in records:
        (pending / f"{record['unit_id']}.json").write_text(json.dumps(record, indent=2) + "\n")

    commit = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    write_launch_files(records, plan, commit)

    print(f"{len(records)} chunks written to {pending}")
    for record in records:
        print(f"  {record['unit_id']}")
        print(f"      {record['copies']} copies (indices {record['copy_index_first']}-"
              f"{record['copy_index_last']}) on {record['planned_node']} "
              f"({record['planned_node_class']}), estimated "
              f"{record['planned_seconds'] / 3600:.2f} h")
    print(f"makespan of the plan: {plan['chosen']['makespan_seconds'] / 3600:.2f} h")


if __name__ == "__main__":
    main()
