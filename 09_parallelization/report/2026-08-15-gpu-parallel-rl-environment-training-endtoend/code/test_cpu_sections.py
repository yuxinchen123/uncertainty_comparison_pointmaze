"""Tests for the table-building helpers in cpu_sections.

Run: <python with matplotlib> -m pytest test_cpu_sections.py
They read the committed measurement files under benchmarks/results/, so they also catch a
results file being moved or renamed out from under the report.
"""
import cpu_sections as c


def test_hours_per_million():
    """A per-copy rate becomes the hours one copy needs for a million steps."""
    # 1574.34 steps per second: 1e6 / (3600 x 1574.34) = 0.1764 hours
    assert round(c.hours_per_million(1574.3418670692142), 4) == 0.1764
    # a rate nearly four times higher takes nearly four times less time
    assert round(c.hours_per_million(5895.025), 4) == 0.0471


def test_mark_best():
    """The best cell of a column is bold, the second best underlined, ties included."""
    # higher is better: 24.1 wins, 7.57 is second
    assert c.mark_best(["24.1", "7.57", "3.80"], [24.1, 7.57, 3.80], True) == [
        "**24.1**", "<u>7.57</u>", "3.80"]
    # tie for best: both are bold, and the next distinct value takes the underline
    assert c.mark_best(["1.0", "1.0", "2.0"], [1.0, 1.0, 2.0], False) == [
        "**1.0**", "**1.0**", "<u>2.0</u>"]


def test_best_under_pins_one_copy_per_worker():
    """The pinned configuration is the 224-worker one-copy-each setting, not its maximum total."""
    rows = c.best_under()
    by_name = {r["name"]: r for r in rows}
    solo = by_name["processor, independent processes, one update, one copy per worker"]
    assert solo["pinned"] and solo["copies"] == 224
    # the numbers themselves are no longer pinned here: the row is whichever measurement of that
    # setting has the best methodology, and it moved once when the five-iteration reading was
    # replaced by a ninety-second one. What is pinned is that the row is a real, sustained
    # measurement of that setting and that its rates agree with each other.
    assert solo["hours_1M"] > 0
    assert abs(solo["per_copy"] * solo["copies"] - solo["total"]) < 1e-6
    # selected by maximum total instead of pinned: a setting with many copies per worker
    assert by_name["processor, independent processes, one update"]["copies"] >= 1792
    # the table is sorted by total throughput, whatever the selection picked for each row
    assert [r["total"] for r in rows] == sorted((r["total"] for r in rows), reverse=True)


def test_best_under_small_limit_keeps_the_pinned_row():
    """A limit below the large settings drops them and leaves the pinned 224-copy row standing."""
    rows = c.best_under(limit=256)
    assert max(r["copies"] for r in rows) <= 256
    assert any(r["pinned"] and r["copies"] == 224 for r in rows)


def test_gain_is_the_step_up_from_the_rung_below():
    """Each rung's total against the one below it; the first rung has nothing to gain over."""
    rows = [{"env_steps_per_sec": 2e6}, {"env_steps_per_sec": 3e6}, {"env_steps_per_sec": 3.03e6}]
    g = c.gain(rows)
    assert g[0] is None
    assert round(g[1], 3) == 0.5
    assert round(g[2], 3) == 0.01


def test_plateau_rung_finds_the_first_rung_that_bought_little():
    """The curve has flattened at the first rung whose gain falls under the threshold."""
    rows = [{"env_steps_per_sec": 2e6, "n_copies": 16},
            {"env_steps_per_sec": 3e6, "n_copies": 32},
            {"env_steps_per_sec": 3.03e6, "n_copies": 64}]
    assert c.plateau_rung(rows, 0.02)["n_copies"] == 64


def test_plateau_rung_is_none_while_the_curve_is_still_climbing():
    """A sweep whose every rung gained more than the threshold has not reached a plateau."""
    rows = [{"env_steps_per_sec": 2e6, "n_copies": 16},
            {"env_steps_per_sec": 3e6, "n_copies": 32},
            {"env_steps_per_sec": 4e6, "n_copies": 64}]
    assert c.plateau_rung(rows, 0.02) is None


def test_gigabytes_cell():
    """Megabytes become a gigabyte figure with two decimals."""
    assert c.GB(2048.0) == "2.00"
    assert c.GB(620.5) == "0.61"


def test_timed_seconds_reads_a_window_an_iteration_count_or_the_old_default():
    """Three kinds of file, one rule for how long each of them actually timed."""
    assert c.timed_seconds({"window_seconds": 92.4, "sec_per_iteration": 7.0}) == 92.4
    assert round(c.timed_seconds({"_iters": 150, "sec_per_iteration": 0.443}), 1) == 66.5
    # no iteration count recorded: the benchmark's default of five, which is what wrote it
    assert round(c.timed_seconds({"sec_per_iteration": 0.481}), 3) == 2.405


def test_declared_iters_reads_the_long_runs_from_their_file_names():
    """The 150-iteration files predate the field, and their names are the only record of it."""
    assert c.declared_iters({"_file": "2026-08-15_trainbench_cpu_processes_sustained_j3_p112.json",
                             "sec_per_iteration": 0.44}) == 150
    assert c.declared_iters({"_file": "2026-08-15_trainbench_cpu_processes_j3_p16.json",
                             "sec_per_iteration": 0.44}) == 5


def test_is_sustained_separates_a_long_window_from_an_opening_burst():
    """A minute of timed work is a settled rate; two seconds is the opening of the load."""
    assert c.is_sustained({"window_seconds": 92.0, "sec_per_iteration": 7.0})
    assert not c.is_sustained({"_file": "burst.json", "sec_per_iteration": 0.48})


def test_method_rank_puts_a_common_window_above_a_sum_of_rates_above_a_burst():
    """Sustained beats burst, and within sustained a shared window beats a sum of worker rates."""
    window = {"window_seconds": 92.0, "sec_per_iteration": 1.0}
    long_sum = {"_iters": 150, "sec_per_iteration": 0.44}
    burst = {"_file": "x.json", "sec_per_iteration": 0.48}
    assert c.method_rank(window) > c.method_rank(long_sum) > c.method_rank(burst)


def test_median_row_takes_the_middle_repeat_and_records_the_spread():
    """Three repeats give the middle one, not the best one, and the spread between the extremes."""
    rows = [{"env_steps_per_sec": 2.10e6, "window_seconds": 90.0, "sec_per_iteration": 1.0},
            {"env_steps_per_sec": 2.12e6, "window_seconds": 90.0, "sec_per_iteration": 1.0},
            {"env_steps_per_sec": 2.11e6, "window_seconds": 90.0, "sec_per_iteration": 1.0}]
    chosen = c.median_row(rows)
    assert chosen["env_steps_per_sec"] == 2.11e6
    assert chosen["_repeats"] == 3
    assert round(chosen["_spread"], 4) == round(2.12 / 2.10 - 1, 4)
    assert chosen["_sustained"]


def test_choose_per_setting_prefers_the_sustained_measurement():
    """Where both kinds exist for a setting, the burst one is dropped rather than being faster."""
    rows = [{"workers": 224, "n_copies": 16, "env_steps_per_sec": 3.8e6, "_file": "burst.json",
             "sec_per_iteration": 0.48},
            {"workers": 224, "n_copies": 16, "env_steps_per_sec": 1.2e6, "window_seconds": 92.0,
             "sec_per_iteration": 1.6}]
    chosen, = c.choose_per_setting(rows, lambda r: (r["workers"], r["n_copies"]))
    assert chosen["env_steps_per_sec"] == 1.2e6 and chosen["_sustained"]


def test_choose_per_setting_falls_back_to_a_burst_row_when_nothing_else_exists():
    """A setting measured only the old way still appears, marked as what it is."""
    rows = [{"workers": 8, "n_copies": 1, "env_steps_per_sec": 0.0117e6, "_file": "burst.json",
             "sec_per_iteration": 0.349}]
    chosen, = c.choose_per_setting(rows, lambda r: (r["workers"], r["n_copies"]))
    assert not chosen["_sustained"]
