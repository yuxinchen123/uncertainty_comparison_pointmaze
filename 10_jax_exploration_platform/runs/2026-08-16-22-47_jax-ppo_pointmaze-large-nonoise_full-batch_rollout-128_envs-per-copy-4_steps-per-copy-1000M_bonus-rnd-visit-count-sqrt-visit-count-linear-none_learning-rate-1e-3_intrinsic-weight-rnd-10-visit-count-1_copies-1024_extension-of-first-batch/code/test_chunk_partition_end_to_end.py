"""End-to-end test that two chunks of a unit really are a partition of it, on real shards.

`tests/agents/test_copy_seed_offset.py` checks the seed arithmetic and
`code/test_plan_submission.py` checks the planner's index ranges, but neither runs the trainer. This
one does: it trains two chunks of a toy unit through the same `scripts/run_training.py` the science
jobs use, on the processor, and then asks the run's own aggregator what it made of them. The
properties it pins are the ones a wrong answer would corrupt silently:

  1. each chunk's shard records the slice of copy indices it was given,
  2. the two slices are disjoint and together cover the toy unit exactly once,
  3. the aggregator's partition check accepts them, and pools them into ONE configuration whose N
     is the sum of the chunks' copies, not one of them and not twice it,
  4. the pooled curve has one point per window with every copy of both chunks in it.

Run (a couple of minutes on the processor):
  PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu \\
    /p/rlprojects/RND/.venvs/platform_jax/bin/python code/test_chunk_partition_end_to_end.py
"""
import json
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

# a unit small enough to train on the processor in seconds: 16 copies cut into two chunks of 8
TOY_COPIES = 16
TOY_CHUNKS = 2
TOY_ITERATIONS = 100
# the episode clock is 400 / gcd(rollout, 400) iterations, which is 50 for an 8-step rollout, so a
# 50-iteration window is one whole turn of it and its records are phase-blocked and therefore scored
TOY_WINDOW = 50
TOY_ROLLOUT = 8
TOY_ENVS = 2


def train_chunk(run_dir: Path, chunk: int) -> None:
    """Run one chunk of the toy unit through the platform's own runner."""
    size = TOY_COPIES // TOY_CHUNKS
    first = chunk * size
    unit_id = (f"toy_chunk-{chunk + 1}-of-{TOY_CHUNKS}_copies-{size}"
               f"_copy-index-{first}-{first + size - 1}")
    command = [str(Path(sys.executable)), str(PLATFORM_ROOT / "scripts" / "run_training.py"),
               "--run-dir", str(run_dir), "--unit-id", unit_id,
               "--bonus", "gt_position_velocity_sqrt",
               "--learning-rates", "1e-3", "--intrinsic-weights", "1",
               "--copies-per-cell", str(size), "--rollout-steps", str(TOY_ROLLOUT),
               "--envs-per-copy", str(TOY_ENVS), "--update-style", "full_batch",
               "--iterations", str(TOY_ITERATIONS), "--window-iterations", str(TOY_WINDOW),
               "--episode-steps", "400", "--base-seed", "0",
               "--run-seed", str(chunk), "--copy-seed-offset", str(first), "--track-coverage"]
    environment = dict(os.environ, PYTHONNOUSERSITE="1", JAX_PLATFORMS="cpu")
    result = subprocess.run(command, capture_output=True, text=True, env=environment)
    if result.returncode != 0:
        raise SystemExit(f"toy chunk {chunk} failed:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}")


def test_two_chunks_partition_the_unit_end_to_end() -> None:
    """Train both chunks, then check the shards and what the aggregator makes of them."""
    scratch = Path(tempfile.mkdtemp(prefix="chunk_partition_"))
    try:
        for chunk in range(TOY_CHUNKS):
            train_chunk(scratch, chunk)

        # 1. each shard records its own slice
        chunks = aggregate.read_chunks(scratch)
        assert len(chunks) == TOY_CHUNKS, f"expected {TOY_CHUNKS} shards, found {len(chunks)}"
        slices = []
        for chunk_id, chunk in sorted(chunks.items()):
            start = chunk["start"]
            assert chunk["complete"] is not None, f"{chunk_id} has no completion record"
            first, last = start["copy_seed_index_first"], start["copy_seed_index_last"]
            assert last - first + 1 == start["copies"], f"{chunk_id} slice does not match its copies"
            slices.append((first, last))
        # 2. disjoint, and together the whole unit exactly once
        covered = [index for first, last in slices for index in range(first, last + 1)]
        assert sorted(covered) == list(range(TOY_COPIES)), f"slices {slices} are not a partition"
        assert len(set(covered)) == len(covered), f"slices {slices} overlap"

        # 3. the aggregator accepts the partition and pools the chunks into ONE configuration
        groups = aggregate.group_by_configuration(chunks)
        assert len(groups) == 1, f"expected one configuration, found {len(groups)}"
        key, group = next(iter(groups.items()))
        aggregate.check_partition(key, group, copies=TOY_COPIES)
        rows = [row for row in configuration_rows(scratch)]
        assert len(rows) == 1, f"expected one scored configuration, found {len(rows)}"
        assert rows[0]["copies"] == TOY_COPIES, (
            f"pooled N is {rows[0]['copies']}, not the {TOY_COPIES} copies of the whole unit")
        assert rows[0]["chunks"] == TOY_CHUNKS

        # 4. the pooled curve has one point per window, over every copy of both chunks
        curve = aggregate.configuration_curve(group)
        assert len(curve["mean_episode_return"]) == TOY_ITERATIONS // TOY_WINDOW, (
            f"{len(curve['mean_episode_return'])} points for "
            f"{TOY_ITERATIONS // TOY_WINDOW} windows")
        assert all(value is not None for value in curve["mean_episode_return"])
        print(f"ok  two chunks of {TOY_COPIES} copies pooled into one configuration of "
              f"N={rows[0]['copies']} over {len(curve['mean_episode_return'])} windows")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def configuration_rows(run_dir: Path) -> list:
    """The aggregator's table for a toy run, with the toy unit's copy count as the denominator."""
    saved = aggregate.COPIES_PER_CONFIGURATION
    aggregate.COPIES_PER_CONFIGURATION = TOY_COPIES
    try:
        return aggregate.cell_table(run_dir)
    finally:
        aggregate.COPIES_PER_CONFIGURATION = saved


def main() -> None:
    """Run the end-to-end test and report."""
    test_two_chunks_partition_the_unit_end_to_end()
    print("chunk-partition end-to-end test passed")


if __name__ == "__main__":
    main()
