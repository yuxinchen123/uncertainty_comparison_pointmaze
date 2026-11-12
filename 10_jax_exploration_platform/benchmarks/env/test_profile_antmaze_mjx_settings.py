"""Tests of profile_antmaze_mjx_settings.py: the table arithmetic, and a trial of every measuring part.

The arithmetic tests use the standard library only. The trial steps a few environments through
every measuring part on processors and renders the result; it initializes JAX, so it runs only
inside a Slurm job (never on a login node) and skips itself anywhere else.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import profile_antmaze_mjx_settings as settings  # noqa: E402

BOTH = {"solver_loop": "fixed_count", "carried_state": "next_step_inputs"}


def test_models_are_parsed_and_named_in_words():
    """A command-line model becomes its changed fields, and those become a name in words."""
    assert settings.parse_changes("default") == {}
    changes = settings.parse_changes("solver_loop=fixed_count,ls_iterations=4,max_contact_points=16")
    assert changes == {"solver_loop": "fixed_count", "ls_iterations": 4, "max_contact_points": 16}
    assert settings.model_name({}) == "default model"
    assert settings.model_name(changes) == ("contact points 16, line-search iterations 4,"
                                            " fixed-count solver loop")
    assert settings.model_name(BOTH) == "fixed-count solver loop, only the next step's inputs carried"
    try:
        settings.parse_changes("frame_skip=4")
    except ValueError:
        return
    raise AssertionError("an unknown field was not refused")


def test_rates_and_emphasis():
    """Counts become rates per million steps; the best of a column is bold, the second underlined."""
    assert settings.per_million(3, 2_000_000) == 1.5
    assert settings.emphasize(["3.0", "1.0", "2.0"], [3.0, 1.0, 2.0], "smallest") == ["3.0", "**1.0**", "<u>2.0</u>"]
    assert settings.emphasize(["3", "1", "3"], [3, 1, 3], "largest") == ["**3**", "<u>1</u>", "**3**"]
    # a value that was not measured takes no mark and no rank
    assert settings.emphasize(["not measured", "0.0"], [None, 0.0], "smallest") == ["not measured", "**0.0**"]
    mean, spread = settings.mean_and_spread([1.0, 2.0, 3.0])
    assert (mean, spread) == (2.0, 1.0)


def test_gate_numbers_are_read_from_what_the_gates_print():
    """The numbers of the three gates are taken from their printed lines, also when a gate stops early."""
    one_step = ("one-step parity: 30 contact-free states, worst |dqpos| 1.76e-06; 20 states in contact,"
                " worst 2.75e-02\nok test_one_step_parity\n")
    assert settings.gate_numbers("test_one_step_parity", one_step) == {
        "one step, no contact (m)": 1.76e-06, "one step, in contact (m)": 2.75e-02}
    all_free = "one-step parity: all 50 states contact-free, worst 3.90e-07\n"
    assert settings.gate_numbers("test_one_step_parity", all_free) == {"one step, no contact (m)": 3.9e-07}
    divergence = "20-step trajectory: first contact at state 5, pre-contact worst 1.67e-06, final divergence 3.10e-01\n"
    assert settings.gate_numbers("test_trajectory_divergence_structure", divergence) == {
        "before the first contact (m)": 1.67e-06, "20 steps (m)": 0.31}
    rest = ("reference -> tuned: settled z 0.38248, after handover z 0.38248, max |dqpos| 3.70e-11\n"
            "tuned -> reference: settled z 0.38251, after handover z 0.38251, max |dqpos| 3.50e-11\n")
    numbers = settings.gate_numbers("test_equilibrium_preservation", rest)
    assert numbers["handover drift (m)"] == 3.7e-11 and abs(numbers["resting heights apart (m)"] - 3e-5) < 1e-12
    # the gate stopped at its first assertion: one line printed, no gap between two heights
    assert settings.gate_numbers("test_equilibrium_preservation", rest.splitlines()[0]) == {
        "handover drift (m)": 3.7e-11}


def run_record(cells):
    """A finished run on a card holding the given cells."""
    return {"unit": "unit 1", "machine": "serval02 card 1", "host": "serval02",
            "card": {"name": "NVIDIA H100 NVL", "driver": "590.48.01", "uuid": "GPU-x"},
            "started_pacific": "2026-11-12 03:00 PT", "unit_started_pacific": "2026-11-12 03:00 PT",
            "unit_start_epoch": 1000.0, "end_epoch": 1600.0,
            "versions": {"jax": "0.11.0", "jaxlib": "0.11.0", "mujoco": "3.11.0", "mujoco-mjx": "3.11.0"},
            "python": "3.12.13", "cells": cells, "finished": True}


def measured(kind, name, **content):
    """A finished cell with no other process on the card."""
    return dict(kind=kind, name=name, status="measured", measured_pacific="2026-11-12 03:01 PT",
                env_steps_per_call=10, rounds_done=11, other_processes_on_card=[], **content)


def test_speed_table_sorts_by_time_and_compares_with_the_default():
    """The speed table lists the fastest model first, marks the best and gives the time against the default."""
    timed = lambda ms: {"environments": 4096, "first_call_seconds": 14.0, "call_seconds": [ms / 100.0] * 11,
                        "start_state": {"environments_reset_for_a_non_finite_state": 0}}
    cell = measured("speed", settings.speed_name("the two speed options", 4096),
                    functions={"default model": timed(20.0), "fixed-count solver loop": timed(15.0)})
    report = settings.render({"runs": [run_record([cell])]}, None)
    rows = [line for line in report.splitlines() if line.startswith("| ") and "4,096" in line]
    assert rows[0].startswith("| fixed-count solver loop | 4,096 | **0.01500** | **0.273** |"), rows[0]
    assert "-25.0%" in rows[0] and "+0.0%" in rows[1] and rows[1].startswith("| default model |")
    assert "10.0 minutes from the unit's start" in report
    # a cell measured while another process was on the card is listed as not used, and a processor trial is ignored
    shared = dict(cell, other_processes_on_card=["pid 7 user other 100 MiB"])
    trial = run_record([cell])
    trial["card"] = {"name": "processors", "driver": "", "uuid": ""}
    report = settings.render({"runs": [run_record([shared]), trial]}, None)
    assert "another process was on the card" in report and "| default model |" not in report


def test_processor_trial_of_every_measuring_part():
    """Every measuring part runs on a few environments on processors and its record renders."""
    if "SLURM_JOB_ID" not in os.environ:
        print("skipped test_processor_trial_of_every_measuring_part: it initializes JAX, so it runs only in a Slurm job")
        return
    import exploration_platform  # noqa: F401  (64-bit statistics on, as in training)
    save = lambda: None
    # GOAL: speed, two models at 4 environments.
    speed = measured("speed", settings.speed_name("trial", 4), functions={})
    settings.measure_speed(speed, [{}, BOTH], 4, 2, save)
    assert set(speed["functions"]) == {"default model", settings.model_name(BOTH)}
    assert all(len(f["call_seconds"]) == 2 for f in speed["functions"].values())
    # GOAL: equality; on processors the same program gives the same bits twice, and carrying less changes nothing.
    equality = measured("equality", settings.EQUALITY_NAME)
    settings.measure_equality(equality, 4, save)
    rows = equality["comparisons"]
    assert len(rows) == 8 and all(row["twice_environments_not_equal"] == 0 for row in rows)
    assert all(0.0 <= row["share_over"] <= 1.0 for row in rows)
    # the scan form (rows 2, 3 and 5) and the smaller carried state (rows 6 and 7) change no bit on processors
    assert all(rows[k]["environments_not_equal"] == 0 for k in (2, 3, 5, 6, 7)), rows
    # GOAL: the physics checks of one model with both options and tight contact limits, short rollouts.
    model = {"max_contact_points": 8, "max_geom_pairs": 4, **BOTH}
    checks = measured("checks", settings.checks_name(settings.model_name(model)))
    settings.measure_checks(checks, model, 4, save, travel_steps=50, rest_steps=50)
    assert checks["contact_slots"] == 8 and set(checks["rollouts"]) == {
        settings.RANDOM_ROLLOUT, settings.WALL_ROLLOUT, settings.REST_ROLLOUT}
    assert checks["solver_sample"] == 4
    for facts in checks["rollouts"].values():
        counts = facts["counts"]
        assert facts["environment_steps"] == 200 and sum(counts["passes_histogram"]) == 200, counts
        assert (counts["sampled_steps_error_over_10_percent"] <= counts["sampled_steps_error_over_1_percent"]
                <= counts["sampled_steps_error_over_0.1_percent"] <= facts["sampled_steps"] == 200)
        assert counts["steps_missed"] >= counts["steps_over_slots"] >= 0
        assert counts["steps_missed_with_free_slots"] == counts["steps_missed"] - counts["steps_over_slots"]
    assert len(checks["rollouts"][settings.RANDOM_ROLLOUT]["distance_first_sequences"]) == 4
    # after 50 steps without torques the ant has landed: between the spawn height and the floor
    assert 0.2 < checks["rollouts"][settings.REST_ROLLOUT]["resting_height_mean"] < 0.75
    report = settings.render({"runs": [run_record([speed, equality, checks])]}, None)
    assert "### Contact limits" in report and "### Solver accuracy" in report
    assert "contact points 8, geometry pairs 4" in report
    print(report)


if __name__ == "__main__":
    test_models_are_parsed_and_named_in_words()
    test_rates_and_emphasis()
    test_gate_numbers_are_read_from_what_the_gates_print()
    test_speed_table_sorts_by_time_and_compares_with_the_default()
    test_processor_trial_of_every_measuring_part()
    print("ok")
