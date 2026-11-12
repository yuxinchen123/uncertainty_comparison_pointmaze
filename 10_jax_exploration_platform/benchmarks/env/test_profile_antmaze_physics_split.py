"""Tests of the table arithmetic and the Markdown rendering of profile_antmaze_physics_split.py.

Standard library only: no JAX is imported, so these run anywhere, under pytest or directly
(python test_profile_antmaze_physics_split.py).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import profile_antmaze_physics_split as split  # noqa: E402


def timed(ms_per_env_step, first_call_seconds=8.0, rounds=11, **extra):
    """A timed-function record whose every round took the given milliseconds per environment step."""
    seconds = ms_per_env_step / 1000.0 * split.ENV_STEPS_PER_CALL
    return dict(first_call_seconds=first_call_seconds, call_seconds=[seconds] * rounds, **extra)


def start_state():
    """A start-state record of a healthy cell."""
    return {"contact_slots_per_environment": 32, "constraint_rows_per_environment": 136,
            "mean_contacts_within_margin_per_environment": 3.5, "share_of_environments_with_a_contact": 0.97,
            "environments_reset_for_a_non_finite_state": 0, "state_is_finite": True}


def split_cell(n_envs, prefix_ms, physics_ms, env_ms, rounds=11, others=()):
    """A finished split cell at one batch size."""
    functions = {f"stages through: {piece}": timed(ms, rounds=rounds)
                 for piece, ms in zip(split.PIECES[:6], prefix_ms)}
    functions[split.PHYSICS_OWN_STATE] = timed(physics_ms, rounds=rounds)
    functions[split.ENVIRONMENT_STEP] = timed(env_ms, rounds=rounds)
    return {"kind": "split", "name": split.split_name("umaze", n_envs), "status": "measured",
            "measured_pacific": "2026-11-12 01:00 PT", "env_steps_per_call": split.ENV_STEPS_PER_CALL,
            "rounds_done": rounds, "functions": functions, "start_state": start_state(),
            "composition_check": {"largest_relative_difference": 0.0, "passed": True},
            "other_processes_on_card": list(others), "wall_seconds": 100.0}


def run_record(cells, unit="unit 1", discarded=None):
    """A finished run holding the given cells."""
    run = {"unit": unit, "machine": "serval02 card 1", "host": "serval02",
           "card": {"name": "NVIDIA H100 NVL", "driver": "590.48.01"},
           "started_pacific": "2026-11-12 01:00 PT",
           "versions": {"jax": "0.11.0", "jaxlib": "0.11.0", "mujoco": "3.11.0", "mujoco-mjx": "3.11.0"},
           "python": "3.12.13", "cells": cells, "finished": True, "wall_seconds": 600.0}
    if discarded:
        run["discarded"] = discarded
    return run


def test_split_rows_sum_to_the_environment_step():
    """The seven pieces add up to the environment step, with the feeding cost taken off piece 1."""
    rows, feed_ms = split.split_rows([2.0, 5.0, 6.0, 8.0, 15.0, 16.0], physics_ms=15.5, env_ms=16.0)
    assert feed_ms == 0.5
    assert [r["added_ms"] for r in rows] == [1.5, 3.0, 1.0, 2.0, 7.0, 1.0, 0.5]
    assert abs(sum(r["added_ms"] for r in rows) - 16.0) < 1e-12
    assert abs(sum(r["share"] for r in rows) - 1.0) < 1e-12
    assert [r["prefix_ms"] for r in rows] == [2.0, 5.0, 6.0, 8.0, 15.0, 16.0, 16.0]


def test_split_rows_with_no_feeding_cost():
    """Edge case: when prefix 6 equals the physics step, piece 1 is the whole of prefix 1."""
    rows, feed_ms = split.split_rows([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], physics_ms=6.0, env_ms=6.0)
    assert feed_ms == 0.0 and rows[0]["added_ms"] == 1.0 and rows[6]["added_ms"] == 0.0


def test_two_point_line():
    """A time of 4.8 ms + 3.125 ms per 1,000 environments is recovered from its two points."""
    fixed, per_thousand = split.two_point_line(8.0, 17.6, 1024, 4096)
    assert abs(fixed - 4.8) < 1e-9 and abs(per_thousand - 3.125) < 1e-9
    # edge case: a time that does not depend on the batch is all fixed part
    assert split.two_point_line(2.0, 2.0, 1024, 4096) == (2.0, 0.0)


def test_throughput_row():
    """4,096 environments at 16 ms per environment step: 0.256 million steps per second in total."""
    row = split.throughput_row(4096, 0.016)
    assert abs(row["total_steps_per_second_millions"] - 0.256) < 1e-12
    assert abs(row["steps_per_second_per_environment"] - 62.5) < 1e-9
    assert abs(row["hours_per_million_steps_per_environment"] - 1e6 / 3600 / 62.5) < 1e-9


def test_solver_arithmetic():
    """The five solver estimates from the four settings rows, on round numbers."""
    got = split.solver_arithmetic(default_ms=18.0, one_iteration_ms=8.0, two_iterations_ms=12.0,
                                  four_line_search_ms=14.0)
    assert got == {"one solver iteration inside the loop": 3.0,
                   "one solver pass without the loop": 2.0,
                   "added per iteration by the loop form": 1.0,
                   "the 8 line-search iterations of all 4 solver iterations": 8.0,
                   "everything outside the solver iterations": 6.0}
    # edge case: when the loop form adds nothing, the pass without the loop equals the pass inside it
    same = split.solver_arithmetic(18.0, 9.0, 12.0, 18.0)
    assert same["added per iteration by the loop form"] == 0.0
    assert same["the 8 line-search iterations of all 4 solver iterations"] == 0.0


def test_usable_cells_skips_shared_card_few_rounds_and_discarded_runs():
    """Only a finished cell with enough rounds, an unshared card and a kept run reaches the tables."""
    good = split_cell(4096, [5, 8, 10, 12, 17, 18], 17.8, 18.0)
    shared = split_cell(1024, [2, 3, 4, 5, 8, 8.2], 8.1, 8.3, others=["pid 7 user other 500 MiB"])
    short = dict(split_cell(8192, [1] * 6, 1, 1, rounds=5), name="short")
    stopped = dict(split_cell(8192, [1] * 6, 1, 1), name="stopped", status="running")
    record = {"runs": [run_record([good, shared, short, stopped]),
                       run_record([dict(good, name="in a discarded run")], "unit 2", "card shared")]}
    assert list(split.usable_cells(record)) == [split.split_name("umaze", 4096)]
    reasons = {name: reason for name, _, reason in split.unusable_cells(record)}
    assert "another process" in reasons[split.split_name("umaze", 1024)]
    assert reasons["short"] == "only 5 rounds"
    assert "unit ended" in reasons["stopped"]
    assert reasons["in a discarded run"] == "run discarded: card shared"


def test_a_later_measurement_replaces_an_earlier_one():
    """The same cell measured in two units: the tables use the later one."""
    first = split_cell(4096, [5, 8, 10, 12, 17, 18], 17.8, 18.0)
    second = split_cell(4096, [5, 8, 10, 12, 17, 18], 17.8, 19.0)
    cells = split.usable_cells({"runs": [run_record([first]), run_record([second], "unit 2")]})
    assert split.median_ms(cells[split.split_name("umaze", 4096)], split.ENVIRONMENT_STEP) == 19.0


def test_render_full_record_and_missing_cells():
    """The report renders with every table filled, and says 'Not measured' where a cell is missing."""
    variations = {"kind": "variations", "name": split.variations_name(4096), "status": "measured",
                  "measured_pacific": "2026-11-12 01:05 PT", "env_steps_per_call": split.ENV_STEPS_PER_CALL,
                  "rounds_done": 11, "other_processes_on_card": [], "functions": {
                      name: timed(18.0 - i, environments=4096, start_state=start_state())
                      for i, (name, _) in enumerate(split.VARIATIONS)}}
    large = {"kind": "large map", "name": split.LARGE_MAP_NAME, "status": "measured",
             "measured_pacific": "2026-11-12 01:04 PT", "env_steps_per_call": split.ENV_STEPS_PER_CALL,
             "rounds_done": 11, "other_processes_on_card": [], "functions": {
                 "large, 1,024 environments": timed(8.5, environments=1024, start_state=start_state()),
                 "large, 4,096 environments": timed(18.5, environments=4096, start_state=start_state())}}
    wide = {"kind": "wide batch", "name": split.wide_name(8192), "status": "measured",
            "measured_pacific": "2026-11-12 01:06 PT", "env_steps_per_call": split.ENV_STEPS_PER_CALL,
            "rounds_done": 11, "other_processes_on_card": [], "functions": {
                split.ENVIRONMENT_STEP: timed(31.0, environments=8192, start_state=start_state())}}
    skipped = {"kind": "wide batch", "name": split.wide_name(16384), "status": "not measured",
               "reason": "20 s were left of the 15-minute unit and the cell was expected to take 100 s"}
    record = {"runs": [run_record([split_cell(4096, [5, 8, 10, 12, 17, 18], 17.8, 18.0),
                                   split_cell(1024, [2, 3, 4, 5, 8, 8.2], 8.1, 8.3),
                                   large, variations, wide, skipped])]}
    text = split.render(record)
    assert "| 7. reward, reset and observation | 8.300 | 0.200 | 2.4% | 18.000 | 0.200 | 1.1% |" in text
    assert "| whole environment step |" in text
    assert "| 4,096 | 18.500 | 18.000 | +2.8% | 8.0 |" in text
    assert "| solver iterations 1 | 4,096 | 17.000 | 94.4% |" in text
    assert "| 16,384 | not measured |" in text
    assert "20 s were left of the 15-minute unit" in text
    assert "## Frame loop unrolled, and one compiler option (U-maze, 4,096 environments)\n\nNot measured." in text
    assert "| 4,096 | 18.000 | split of the physics step, umaze, 4,096 environments, timed in turn |" in text
    assert "| 4,096 | 18.000 | variations of the environment step, umaze, 4,096 environments |" in text
    # solver estimates from the settings rows 18, 17, 16, 15 ms: (18 - 16) / 2 = 1 ms per iteration
    assert "| one solver iteration inside the loop | 1.000 | 5.6% |" in text
    assert "| the 8 line-search iterations of all 4 solver iterations | 6.000 | 33.3% |" in text
    assert "- 4,096 environments: the 2 cells" not in text   # one whole-step cell only: no spread line
    # edge case: an empty record renders, with every table marked as not measured
    assert split.render({"runs": []}).count("Not measured.") == 4


def test_split_uses_back_to_back_times_when_every_function_has_them():
    """A split cell with both timing orders is read back to back, and the two orders are compared."""
    cell = split_cell(4096, [5, 8, 10, 12, 17, 18], 17.8, 18.0)
    assert split.split_order(cell) == "in turn"
    for function in cell["functions"].values():
        function["call_seconds_back_to_back"] = [0.9 * seconds for seconds in function["call_seconds"]]
    assert split.split_order(cell) == "back to back"
    assert abs(split.median_ms(cell, split.ENVIRONMENT_STEP, "back to back") - 16.2) < 1e-9
    text = split.render({"runs": [run_record([cell])]})
    assert "| 7. reward, reset and observation | 16.200 | 0.180 | 1.1% |" in text
    assert "| environment step | 18.000 | 16.200 | +11.1% |" in text
    assert "split of the physics step, umaze, 4,096 environments, timed back to back |" in text
    # edge case: one function with too few back-to-back rounds sends the whole cell back to the in-turn times
    cell["functions"][split.ENVIRONMENT_STEP]["call_seconds_back_to_back"] = [0.1] * 3
    assert split.split_order(cell) == "in turn"


def test_a_replaced_measurement_is_listed():
    """A cell measured in two units: the earlier one is listed as replaced, with the later unit named."""
    first = split_cell(4096, [5, 8, 10, 12, 17, 18], 17.8, 18.0)
    second = split_cell(4096, [5, 8, 10, 12, 17, 18], 17.8, 19.0)
    record = {"runs": [run_record([first]), run_record([second], "unit 2")]}
    reason = "replaced by the later measurement of this cell in unit `unit 2`"
    assert split.unusable_cells(record) == [(split.split_name("umaze", 4096), "unit 1", reason)]


def test_render_uses_the_cell_that_timed_all_batch_sizes_in_turn():
    """When one cell timed every batch size, the throughput table reads that cell, not the separate ones."""
    together = {"kind": "batch sizes", "name": split.BATCH_SIZES_NAME, "status": "measured",
                "measured_pacific": "2026-11-12 01:20 PT", "env_steps_per_call": split.ENV_STEPS_PER_CALL,
                "rounds_done": 11, "other_processes_on_card": [], "functions": {
                    f"{n:,} environments": timed(ms, environments=n, start_state=start_state())
                    for n, ms in ((1024, 9.0), (2048, 12.0), (4096, 17.5), (8192, 28.0), (16384, 52.0))}}
    record = {"runs": [run_record([split_cell(4096, [5, 8, 10, 12, 17, 18], 17.8, 18.0), together])]}
    text = split.render(record)
    assert "All batch sizes were timed in turn in one cell" in text
    assert "| 4,096 | 0.01750 | 0.234 (+37% total) | 57.1 | 4.86 |" in text
    assert "| 2,048 | 0.01200 | 0.171 (+50% total) |" in text
    # both readings of the default step at 4,096 environments are listed
    assert "| 4,096 | 18.000 | split of the physics step, umaze, 4,096 environments, timed in turn |" in text
    assert "| 4,096 | 17.500 | batch sizes, umaze, whole environment step |" in text
    # edge case: with no settings cell the solver estimates are left out
    assert "What the settings rows say about the solver" not in text


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"passed: {name}")
