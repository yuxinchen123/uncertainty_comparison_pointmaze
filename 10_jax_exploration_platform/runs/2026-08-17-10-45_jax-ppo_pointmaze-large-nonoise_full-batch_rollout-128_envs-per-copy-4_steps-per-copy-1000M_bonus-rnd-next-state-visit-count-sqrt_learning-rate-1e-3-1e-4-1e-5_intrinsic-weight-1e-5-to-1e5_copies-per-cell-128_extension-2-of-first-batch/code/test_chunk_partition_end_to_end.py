"""End-to-end test that two chunks of a MULTI-CELL unit really are a partition of it.

`tests/agents/test_copy_seed_offset.py` checks the seed arithmetic and
`code/test_plan_submission.py` checks the planner's index ranges, but neither runs the trainer, and
neither covers the thing that is new in this run: a chunk carries EVERY configuration of its arm at
a slice of their copies, so the aggregator has to split each shard by cell and then pool across
chunks. This test trains two chunks of a toy multi-cell unit through the same
`scripts/run_training.py` the science jobs use, on the processor, and asks the run's own aggregator
what it made of them. The properties it pins are the ones a wrong answer would corrupt silently:

  1. each chunk's shard records the slice of copy indices it was given, and holds every cell,
  2. the two slices are disjoint and together cover the toy unit's copies exactly once,
  3. the aggregator's partition check accepts them and produces one row per CELL, each pooling the
     cell's copies from both chunks — N is the sum of the chunks' per-cell copies, not one of them
     and not twice it,
  4. a cell's pooled curve has one point per window, and the cells of one shard select disjoint
     copies that together cover it.

Run (a couple of minutes on the processor):
  PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu \\
    /p/rlprojects/RND/.venvs/platform_jax/bin/python code/test_chunk_partition_end_to_end.py
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent.parent
PLATFORM_ROOT = RUN_DIR.parent.parent
sys.path.insert(0, str(RUN_DIR / "code"))

import aggregate  # noqa: E402

# a unit small enough to train on the processor in seconds: 2 rates x 2 weights = 4 cells of 8
# copies each, cut into two chunks of 4 copies per cell
TOY_RATES = "1e-3,1e-4"
TOY_WEIGHTS = "1,10"
TOY_CELLS = 4
TOY_COPIES_PER_CELL = 8
TOY_CHUNKS = 2
TOY_ITERATIONS = 100
# the episode clock turns every 400 / 8 = 50 iterations for an 8-step rollout, so a 50-iteration
# window is one whole turn of it and its records are phase-blocked and therefore scored
TOY_WINDOW = 50
TOY_ROLLOUT = 8
TOY_ENVS = 2


def train_chunk(run_dir: Path, chunk: int) -> None:
    """Run one chunk of the toy multi-cell unit through the platform's own runner."""
    per_cell = TOY_COPIES_PER_CELL // TOY_CHUNKS
    first = chunk * per_cell
    unit_id = (f"toy_chunk-{chunk + 1}-of-{TOY_CHUNKS}_copies-per-cell-{per_cell}"
               f"_copy-index-{first}-{first + per_cell - 1}")
    command = [str(Path(sys.executable)), str(PLATFORM_ROOT / "scripts" / "run_training.py"),
               "--run-dir", str(run_dir), "--unit-id", unit_id,
               "--bonus", "gt_position_velocity_sqrt",
               "--learning-rates", TOY_RATES, "--intrinsic-weights", TOY_WEIGHTS,
               "--copies-per-cell", str(per_cell), "--rollout-steps", str(TOY_ROLLOUT),
               "--envs-per-copy", str(TOY_ENVS), "--update-style", "full_batch",
               "--iterations", str(TOY_ITERATIONS), "--window-iterations", str(TOY_WINDOW),
               "--episode-steps", "400", "--base-seed", "0",
               "--run-seed", str(chunk), "--copy-seed-offset", str(first), "--track-coverage"]
    environment = dict(os.environ, PYTHONNOUSERSITE="1", JAX_PLATFORMS="cpu")
    result = subprocess.run(command, capture_output=True, text=True, env=environment)
    if result.returncode != 0:
        raise SystemExit(f"toy chunk {chunk} failed:\n{result.stdout[-3000:]}\n"
                         f"{result.stderr[-3000:]}")


def cell_rows(run_dir: Path) -> list:
    """The aggregator's table for a toy run, with the toy unit's copy count as the denominator."""
    saved = aggregate.COPIES_PER_CONFIGURATION
    aggregate.COPIES_PER_CONFIGURATION = TOY_COPIES_PER_CELL
    try:
        return aggregate.cell_table(run_dir)
    finally:
        aggregate.COPIES_PER_CONFIGURATION = saved


def test_two_chunks_partition_a_multi_cell_unit_end_to_end() -> None:
    """Train both chunks, then check the shards and what the aggregator makes of them."""
    scratch = Path(tempfile.mkdtemp(prefix="chunk_partition_"))
    try:
        for chunk in range(TOY_CHUNKS):
            train_chunk(scratch, chunk)

        # 1. each shard records its own slice and holds every cell
        chunks = aggregate.read_chunks(scratch)
        assert len(chunks) == TOY_CHUNKS, f"expected {TOY_CHUNKS} shards, found {len(chunks)}"
        slices = []
        for chunk_id, chunk in sorted(chunks.items()):
            start = chunk["start"]
            assert chunk["complete"] is not None, f"{chunk_id} has no completion record"
            assert len(start["cell_settings"]) == TOY_CELLS, f"{chunk_id} lost a cell"
            first, last = start["copy_seed_index_first"], start["copy_seed_index_last"]
            assert last - first + 1 == TOY_COPIES_PER_CELL // TOY_CHUNKS
            slices.append((first, last))
        # 2. disjoint, and together the whole unit's per-cell copies exactly once
        covered = [index for first, last in slices for index in range(first, last + 1)]
        assert sorted(covered) == list(range(TOY_COPIES_PER_CELL)), f"{slices} is not a partition"

        # 3. one row per cell, each pooling both chunks' copies of that cell
        group = aggregate.group_by_arm(chunks)["gt_position_velocity_sqrt"]
        aggregate.check_partition("gt_position_velocity_sqrt", group,
                                  copies=TOY_COPIES_PER_CELL)
        rows = cell_rows(scratch)
        assert len(rows) == TOY_CELLS, f"expected {TOY_CELLS} scored cells, found {len(rows)}"
        for row in rows:
            assert row["copies"] == TOY_COPIES_PER_CELL, (
                f"pooled N is {row['copies']}, not the {TOY_COPIES_PER_CELL} copies of the cell")
            assert row["chunks"] == TOY_CHUNKS
        settings = {(row["learning_rate"], row["intrinsic_weight"]) for row in rows}
        assert len(settings) == TOY_CELLS, f"cells collapsed into {len(settings)} settings"

        # 4. each cell's pooled curve has one point per window, and the cells select DISJOINT
        # copies of each shard. The curves themselves are not compared: a 100-iteration toy run
        # never reaches the goal, so every cell's return is zero and equal values would prove
        # nothing either way.
        curve = aggregate.configuration_curve(group, sorted(settings)[0])
        assert len(curve["mean_episode_return"]) == TOY_ITERATIONS // TOY_WINDOW
        assert all(value is not None for value in curve["mean_episode_return"])
        positions = aggregate.cell_copies(group[0])
        every = [index for members in positions.values() for index in members]
        assert sorted(every) == list(range(TOY_CELLS * TOY_COPIES_PER_CELL // TOY_CHUNKS)), (
            "the cells of one shard do not partition its copies")
        assert all(len(members) == TOY_COPIES_PER_CELL // TOY_CHUNKS
                   for members in positions.values())
        print(f"ok  two chunks of {TOY_CELLS} cells pooled into {len(rows)} rows of "
              f"N={rows[0]['copies']} over {len(curve['mean_episode_return'])} windows")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def main() -> None:
    """Run the end-to-end test and report."""
    test_two_chunks_partition_a_multi_cell_unit_end_to_end()
    print("chunk-partition end-to-end test passed")


if __name__ == "__main__":
    main()
