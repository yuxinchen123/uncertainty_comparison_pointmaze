"""Cut the chunks stranded by jaguar03's failure into smaller ones the free cards can carry.

jaguar03 went down at 2026-08-17 23:03 PT with eight of this run's chunks on it. Four of them —
the oracle arm's chunks 3 to 6, holding copy indices 32 to 95 at 16 copies per configuration —
could not be re-placed as they were: a 528-copy chunk of that arm needs an H100 or an A100 to
finish in the time the rest of the run has left, and every such card on the cluster is either
allocated or projected free more than a day out. On the fastest card actually free, an RTX 2080 Ti,
one of those chunks would take 18 hours.

Cutting the same copies finer fixes that, because the per-copy rate rises as the chunk shrinks:
the same 64 copies per configuration as EIGHT chunks of 8 copies each take about 7.4 hours on an
RTX 2080 Ti, running side by side instead of one after another.

What this writes, and what it moves:

1. Eight new queue entries covering copy indices 32..95, eight copies per configuration each. Their
   `--copy-seed-offset` is their first copy index, so every copy keeps the seed it would have had
   in a single 128-copy run, and their `--run-seed` is 8 to 15, outside the 0 to 7 the original
   eight-way cutting used, so no chunk shares another's action noise.
2. The four superseded shards move from `data/` to `data_superseded/`, and their queue entries from
   `queue/running/` to `queue/superseded/`. Nothing is edited or deleted: the shards are the record
   of what jaguar03 managed before it failed. They have to leave `data/` because the aggregator
   checks that a configuration's chunks partition its copies exactly once, and indices 32..95 would
   otherwise be covered twice — by the superseded chunks and by the new ones.

Run once, after the stranded jobs have been cancelled:
  PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python code/recut_stranded_chunks.py
"""
import json
import shutil
import sys
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RUN_DIR / "code"))

import plan_submission as ps  # noqa: E402

ARM = "gt_position_velocity_sqrt"
SUPERSEDED_CHUNKS = ("chunk-3-of-8", "chunk-4-of-8", "chunk-5-of-8", "chunk-6-of-8")
FIRST_COPY_INDEX = 32          # the first copy index the superseded chunks covered
LAST_COPY_INDEX = 95           # ... and the last
COPIES_PER_CELL = 8            # the new cutting: eight copies of every configuration per chunk
RUN_SEED_BASE = 8              # outside the 0..7 the original eight-way cutting used


def new_chunk_id(first: int, last: int) -> str:
    """The new chunk's name, saying outright that it came from the re-cut and what it holds."""
    return (f"unit-2_{ARM.replace('_', '-')}"
            f"_grid-learning-rate-1e-3-1e-4-1e-5_intrinsic-weight-1e-5-to-1e5"
            f"_recut-chunk-{(first - FIRST_COPY_INDEX) // COPIES_PER_CELL + 1}-of-8"
            f"_copies-per-cell-{COPIES_PER_CELL}_copies-{COPIES_PER_CELL * ps.CELLS}"
            f"_copy-index-{first}-{last}")


def new_record(first: int, index: int) -> dict:
    """One queue entry for the re-cut, in the same shape `build_queue` writes."""
    last = first + COPIES_PER_CELL - 1
    unit_id = new_chunk_id(first, last)
    arguments = ["--unit-id", unit_id, "--bonus", ARM,
                 "--learning-rates", ",".join(ps.LEARNING_RATES),
                 "--intrinsic-weights", ",".join(ps.INTRINSIC_WEIGHTS),
                 "--copies-per-cell", str(COPIES_PER_CELL),
                 "--rollout-steps", str(ps.ROLLOUT_STEPS),
                 "--envs-per-copy", str(ps.ENVS_PER_COPY),
                 "--update-style", "full_batch",
                 "--iterations", str(ps.ITERATIONS),
                 "--window-iterations", str(ps.WINDOW_ITERATIONS),
                 "--episode-steps", str(ps.EPISODE_STEPS),
                 "--base-seed", "0",
                 "--run-seed", str(RUN_SEED_BASE + index),
                 "--copy-seed-offset", str(first),
                 "--track-coverage"]
    return {"unit_id": unit_id, "order": 2, "bonus": ARM,
            "chunk": RUN_SEED_BASE + index, "chunks": 12,
            "copies": COPIES_PER_CELL * ps.CELLS, "copies_per_cell": COPIES_PER_CELL,
            "copies_per_cell_in_the_whole_unit": ps.COPIES_PER_CELL,
            "copies_in_the_whole_unit": ps.COPIES_PER_UNIT,
            "copy_index_first": first, "copy_index_last": last, "cells": ps.CELLS,
            "learning_rates": list(ps.LEARNING_RATES),
            "intrinsic_weights": list(ps.INTRINSIC_WEIGHTS),
            "iterations": ps.ITERATIONS, "window_iterations": ps.WINDOW_ITERATIONS,
            "windows": ps.ITERATIONS // ps.WINDOW_ITERATIONS,
            "env_steps_per_copy": ps.STEPS_PER_COPY,
            "planned_node": "", "planned_node_class": "ai01-04_lynx10",
            "planned_reservation": "",
            # 55,427 environment steps a second per copy measured on an RTX A4500 at THIS copy
            # count -- 264 copies, not the 132-copy row -- carried onto the RTX 2080 Ti by the
            # survey's ratio between the two classes
            "planned_seconds": ps.STEPS_PER_COPY / (55427.0 * 6.76 / 9.96),
            "planned_rate_source": ("probed on jaguar03 at this copy count, carried by the "
                                    "survey's card ratio at 512 copies"),
            "replaces": [c for c in SUPERSEDED_CHUNKS],
            "arguments": arguments}


def supersede(run_dir: Path) -> list:
    """Move the four superseded shards and queue entries out of the aggregator's path."""
    moved = []
    (run_dir / "data_superseded").mkdir(exist_ok=True)
    (run_dir / "queue" / "superseded").mkdir(exist_ok=True)
    for token in SUPERSEDED_CHUNKS:
        for shard in (run_dir / "data").glob(f"*{token}*.jsonl"):
            shutil.move(str(shard), run_dir / "data_superseded" / shard.name)
            moved.append(f"data/{shard.name}")
        for state in ("running", "pending"):
            for entry in (run_dir / "queue" / state).glob(f"*{token}*.json"):
                shutil.move(str(entry), run_dir / "queue" / "superseded" / entry.name)
                moved.append(f"queue/{state}/{entry.name}")
    return moved


def main() -> None:
    """Supersede the four stranded chunks and write the eight that replace them."""
    moved = supersede(RUN_DIR)
    print(f"moved {len(moved)} superseded files out of the aggregator's path:")
    for name in moved:
        print(f"  {name}")

    pending = RUN_DIR / "queue" / "pending"
    pending.mkdir(parents=True, exist_ok=True)
    records = [new_record(first, index) for index, first in
               enumerate(range(FIRST_COPY_INDEX, LAST_COPY_INDEX + 1, COPIES_PER_CELL))]
    covered = [i for r in records for i in range(r["copy_index_first"], r["copy_index_last"] + 1)]
    if sorted(covered) != list(range(FIRST_COPY_INDEX, LAST_COPY_INDEX + 1)):
        raise SystemExit(f"the re-cut does not cover {FIRST_COPY_INDEX}..{LAST_COPY_INDEX} exactly "
                         f"once: {len(covered)} indices, {len(set(covered))} distinct")
    for record in records:
        (pending / f"{record['unit_id']}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nwrote {len(records)} re-cut queue entries, each "
          f"{records[0]['copies']} copies, estimated "
          f"{records[0]['planned_seconds'] / 3600:.2f} h on an RTX 2080 Ti:")
    for record in records:
        print(f"  {record['unit_id']}")


if __name__ == "__main__":
    main()
