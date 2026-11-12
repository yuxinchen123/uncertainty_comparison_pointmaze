"""The training-iteration measurement script, run on a compute node's processors at a tiny cell.

Checks that one cell goes through the whole script (build, prime, the timed rounds, the two pieces,
the record) and that the Markdown table is written from the records. The numbers themselves mean
nothing on processors. Needs warp-lang (the platform_jax_warp interpreter); skipped without it.

Run on a compute node, never on a login node:
  PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu <platform_jax_warp python> -m pytest -q \
      benchmarks/env/test_profile_antmaze_warp_training.py
"""
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("warp")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import profile_antmaze_warp_training as profile  # noqa: E402


def test_median_and_pacific_time():
    """The middle value (the upper one of an even count), and a stored time shown in Pacific."""
    assert profile.median([3.0, 1.0, 2.0]) == 2.0 and profile.median([4.0, 1.0]) == 4.0
    assert profile.pacific("2026-11-12T15:30:00-05:00") == "2026-11-12 12:30 PT"


def test_a_cell_is_measured_recorded_and_rendered(tmp_path):
    """A 2 x 2 cell on processors fills its record; an unstarted cell is named in the notes."""
    records = tmp_path / "records"
    common = ["--records", str(records), "--machine", "a compute node's processors",
              "--card-slug", "processors"]
    assert profile.main([*common, "--policies", "2", "--environments-per-policy", "2",
                         "--processor-dry-run", "--rounds", "2", "--round-seconds", "0.5",
                         "--piece-calls", "2"]) == 0
    record = json.loads((records / "processors_policies-2_environments-per-policy-2.json").read_text())
    assert record["status"] == "measured" and record["processor_dry_run"]
    assert (record["environments"], record["rollout_steps"]) == (4, 128)
    assert record["loss_finite"] and record["non_finite_resets"] == 0
    assert set(record["pieces"]) == {"update", "rollout"}
    assert not record["pool"]["dropped"] and record["pool"]["naconmax"] == 4 * 48
    assert Path(record["kernel_folder"]).is_dir()
    # the five throughput columns agree with each other: 128 steps x 2 environments per policy
    seconds, rate = record["seconds_per_iteration"], record["steps_per_second_per_policy"]
    assert abs(rate - 128 * 2 / seconds) < 1e-6 * rate
    assert abs(record["total_steps_per_second_millions"] - 2 * rate / 1e6) < 1e-12
    assert abs(record["hours_per_million_steps_per_policy"] - 1e6 / (3600 * rate)) < 1e-9

    # a cell its unit could not start, then the table of both
    assert profile.main([*common, "--policies", "4", "--environments-per-policy", "2",
                         "--card-name", "processors (dry run)",
                         "--not-measured-reason", "the unit's time was used"]) == 0
    markdown = tmp_path / "table.md"
    assert profile.main(["--render", "--records", str(records), "--markdown", str(markdown)]) == 0
    text = markdown.read_text()
    assert "## processors (dry run) (a compute node's processors)" in text
    assert "| 2 | 2 | 4 |" in text and text.count("\n| ") == 2
    assert "- 4 policies x 2 environments" in text and "the unit's time was used" in text
