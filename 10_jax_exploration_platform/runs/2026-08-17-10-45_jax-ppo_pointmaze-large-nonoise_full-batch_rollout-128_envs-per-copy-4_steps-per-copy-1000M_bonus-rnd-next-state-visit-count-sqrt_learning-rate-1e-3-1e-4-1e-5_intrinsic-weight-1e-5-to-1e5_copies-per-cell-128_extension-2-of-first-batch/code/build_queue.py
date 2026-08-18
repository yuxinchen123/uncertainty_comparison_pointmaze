"""Write the queue this run's chosen submission plan asks for, and the run's launch-time files.

One run is one sweep. This run is the SECOND extension of train run 1.1
(`runs/2026-08-16-20-39_..._first-batch`): it takes the two algorithm arms that were still
improving at the parent's budget and re-runs their WHOLE learning-rate by intrinsic-weight grid at
about 1,000M environment steps per copy, 128 copies per configuration.

A unit here is one arm's whole grid fused into one compiled program: 3 learning rates x 11
intrinsic weights = 33 cells x 128 copies = 4,224 copies. A unit may be CUT INTO CHUNKS by copy
index, because the copies are independent seeds and a chunk of them runs at a better per-copy rate
on its own card; `code/plan_submission.py` decides how many chunks each unit gets and which card
each chunk goes to, and this module writes exactly the queue that plan describes. Each chunk is one
queue entry, one job, one shard, and every chunk carries all 33 configurations.

**How a chunk stays a slice of the same run rather than a different run.** The seed index of a copy
keys both its initial weights and the environment draws it meets, so chunk j of k passes
`--copy-seed-offset j x (128 / k)` and holds copy indices `j x (128/k)` onward IN EVERY CELL: the k
chunks partition each cell's 0..127, no index appears twice, and copy i of a cell in the union has
the weights and the environments copy i of a single 128-copy cell would have. The policy's own
sampling stream is keyed by `--run-seed`, which is the chunk index, so no copy shares its action
noise with a copy of another chunk either.

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
FIRST_EXTENSION_RUN = ("2026-08-16-22-47_jax-ppo_pointmaze-large-nonoise_full-batch_rollout-128"
                       "_envs-per-copy-4_steps-per-copy-1000M"
                       "_bonus-rnd-visit-count-sqrt-visit-count-linear-none_learning-rate-1e-3"
                       "_intrinsic-weight-rnd-10-visit-count-1_copies-1024_extension-of-first-batch")


def chunk_id(chunk: dict) -> str:
    """The chunk's name, with every load-bearing knob spelled into it.

    before: {"order": 1, "arm": "rnd_next_state", "chunk": 1, "chunks": 8, "copies_per_cell": 16,
             "copies": 528, "copy_index_first": 16, "copy_index_last": 31}
    after:  "unit-1_rnd-next-state_grid-learning-rate-1e-3-1e-4-1e-5_intrinsic-weight-1e-5-to-1e5_
             chunk-2-of-8_copies-per-cell-16_copies-528_copy-index-16-31"
    A chunk of one is still named as such, so the shard name says outright whether a unit was cut.
    """
    return (f"unit-{chunk['order']}_{chunk['arm'].replace('_', '-')}"
            f"_grid-learning-rate-1e-3-1e-4-1e-5_intrinsic-weight-1e-5-to-1e5"
            f"_chunk-{chunk['chunk'] + 1}-of-{chunk['chunks']}"
            f"_copies-per-cell-{chunk['copies_per_cell']}_copies-{chunk['copies']}"
            f"_copy-index-{chunk['copy_index_first']}-{chunk['copy_index_last']}")


def chunk_record(chunk: dict) -> dict:
    """One queue entry: the identity of the chunk and the exact arguments that run it."""
    arguments = ["--unit-id", chunk_id(chunk), "--bonus", chunk["arm"],
                 "--learning-rates", ",".join(ps.LEARNING_RATES),
                 "--intrinsic-weights", ",".join(ps.INTRINSIC_WEIGHTS),
                 "--copies-per-cell", str(chunk["copies_per_cell"]),
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
                 # holds exactly its slice of every cell's copies
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
        "copies_per_cell": chunk["copies_per_cell"],
        "copies_per_cell_in_the_whole_unit": ps.COPIES_PER_CELL,
        "copies_in_the_whole_unit": ps.COPIES_PER_UNIT,
        "copy_index_first": chunk["copy_index_first"],
        "copy_index_last": chunk["copy_index_last"],
        "cells": ps.CELLS,
        "learning_rates": list(ps.LEARNING_RATES),
        "intrinsic_weights": list(ps.INTRINSIC_WEIGHTS),
        "iterations": ps.ITERATIONS,
        "window_iterations": ps.WINDOW_ITERATIONS,
        "windows": ps.ITERATIONS // ps.WINDOW_ITERATIONS,
        "env_steps_per_copy": ps.STEPS_PER_COPY,
        "planned_node": chunk["node"],
        "planned_node_class": chunk["node_class"],
        "planned_reservation": chunk.get("reservation", ""),
        "planned_seconds": chunk["seconds"],
        "planned_rate_source": chunk["rate_source"],
        "arguments": arguments,
    }


def check_the_chunks_partition_every_unit(records: list) -> None:
    """Fail loudly unless each unit's chunks cover copy indices 0..127 exactly once between them.

    This is the invariant that makes the union of the shards the 128-copy-per-configuration run: if
    two chunks overlapped, some copies would be counted twice and others never run, and the
    aggregate would be quietly wrong rather than obviously broken.
    """
    by_unit = {}
    for record in records:
        by_unit.setdefault(record["order"], []).append(record)
    for order, group in sorted(by_unit.items()):
        covered = []
        for record in group:
            covered += list(range(record["copy_index_first"], record["copy_index_last"] + 1))
        if sorted(covered) != list(range(ps.COPIES_PER_CELL)):
            raise SystemExit(
                f"unit {order}'s {len(group)} chunks do not partition its "
                f"{ps.COPIES_PER_CELL} copies per cell: {len(covered)} indices covered, "
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
        config = sweep_config(learning_rates=[float(x) for x in record["learning_rates"]],
                              betas=[float(x) for x in record["intrinsic_weights"]],
                              copies_per_group=record["copies_per_cell"], style="full_batch",
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
        f"RUN_DIR={RUN_DIR} bash {RUN_DIR}/code/submit_probe.sh <label> <node> 00:40:00 "
        "<copies per cell> <arms>\n"
        "PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python "
        f"{RUN_DIR}/code/measured_cells.py\n"
        "PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python "
        f"{RUN_DIR}/code/plan_submission.py\n"
        "PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python "
        f"{RUN_DIR}/code/build_queue.py\n"
        "# then, per chunk, one independent job on the node the plan named. The job runs its own\n"
        "# canary and its own resume check on that card before the science run, so the card is\n"
        "# never released between the two:\n"
        f"#   RUN_DIR={RUN_DIR} bash code/submit_one.sh <chunk_id> <node> 4-00:00:00\n"
        f"# code commit at launch: {commit}\n"
        "# one chunk's science run is:\n"
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
            "first_extension_of_the_same_parent": FIRST_EXTENSION_RUN,
            "what_changed": (f"the WHOLE learning-rate by intrinsic-weight grid of the two arms "
                             f"that were still improving at the parent's budget, re-run at "
                             f"{ps.COPIES_PER_CELL} copies per configuration for "
                             f"{ps.STEPS_PER_COPY} environment steps per copy: half the copies of "
                             f"one parent cell and about 100 times its step budget"),
            "arms_extended": {
                arm: {"learning_rates": group[0]["learning_rates"],
                      "intrinsic_weights": group[0]["intrinsic_weights"],
                      "configurations": ps.CELLS,
                      "parent_copies_per_cell": PARENT_COPIES_PER_CELL,
                      "extension_copies_per_cell": ps.COPIES_PER_CELL,
                      "extension_chunks": len(group),
                      "parent_env_steps_per_copy": PARENT_STEPS_PER_COPY,
                      "extension_env_steps_per_copy": ps.STEPS_PER_COPY}
                for arm, group in by_arm.items()},
        },
        "description": ("The second extension of train run 1.1, at a thousand million environment "
                        "steps per copy. Two algorithm arms — random network distillation on the "
                        "next state, and the square root of the oracle position-and-velocity "
                        "visit count — each swept over its whole grid of 3 learning rates x 11 "
                        "intrinsic weights, 128 copies per configuration, 1,000,038,400 "
                        "environment steps per copy. A unit is one arm's 33 configurations fused "
                        "into one compiled program, cut into chunks by copy index where the "
                        "submission plan's arithmetic says that finishes the whole set sooner."),
        "git": {"commit": commit},
        "specs": {
            "env": "pointmaze_large_cont400_nonoise@1",
            "agent": "ppo_full_batch@1",
            "bonus": "rnd_next_state@1 + gt_position_velocity_sqrt@1",
            "update_schedule": ("per_rollout@1 for random network distillation, "
                                "post_rollout_update@1 for the oracle visit-count arm"),
        },
        "compile_signature": {
            "cells": ps.CELLS,
            "copies_per_cell_in_the_whole_unit": ps.COPIES_PER_CELL,
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
                                "copies 0..127 of EVERY cell and copy i of a cell in the union has "
                                "the weights and the environment draws copy i of a single "
                                "128-copy cell would have",
        },
        "submission_plan": {
            "generated_at_pacific": plan["generated_at_pacific"],
            "makespan_hours": round(plan["chosen"]["makespan_seconds"] / 3600, 3),
            "chunk_count": plan["chosen"]["chunk_count"],
            "cards_used": plan["chosen"]["cards_used"],
            "splits_per_unit": plan["chosen"]["splits"],
            "written_by": "code/plan_submission.py, recorded in code/submission_plan.json",
        },
        "units": [record["unit_id"] for record in records],
        "artifacts": {"metrics": "metrics.jsonl", "summary": "summary.json",
                      "shards": "data/*.jsonl", "aggregator": "code/aggregate.py",
                      "plan": "code/submission_plan.json",
                      "own_rate_measurements": "code/measured_cells.json",
                      "canary_shards": "canary/data/*.jsonl"},
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
        print(f"      {record['copies']} copies ({record['copies_per_cell']} per cell, indices "
              f"{record['copy_index_first']}-{record['copy_index_last']}) on "
              f"{record['planned_node']} ({record['planned_node_class']}), estimated "
              f"{record['planned_seconds'] / 3600:.2f} h")
    print(f"makespan of the plan: {plan['chosen']['makespan_seconds'] / 3600:.2f} h")


if __name__ == "__main__":
    main()
