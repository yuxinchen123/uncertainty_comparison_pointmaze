"""Unit tests for the arithmetic of the copies-per-worker sweep."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_copies_per_worker import (aggregate, burst_and_settled, iters_for,  # noqa: E402
                                     per_worker_memory_model, predicted_node_gb,
                                     predicted_point_gb)


def test_iters_for_hits_the_upper_bound_on_a_fast_point():
    """A fast iteration would need more than the bound to fill a minute, so the bound applies."""
    assert iters_for(0.4, target=60.0, cap=300.0, lo=3, hi=150) == 150


def test_iters_for_times_about_a_minute_on_a_slow_point():
    """Seven seconds an iteration gives nine iterations, about a minute of work."""
    assert iters_for(7.0, target=60.0, cap=300.0, lo=3, hi=150) == 9


def test_iters_for_respects_the_ceiling_on_a_very_slow_point():
    """A two-hundred-second iteration cannot be run three times inside a five-minute ceiling."""
    assert iters_for(200.0, target=60.0, cap=300.0, lo=1, hi=150) == 1


def test_iters_for_never_goes_below_the_floor():
    """The floor wins over the ceiling, so a point is always measured at least that many times."""
    assert iters_for(200.0, target=60.0, cap=300.0, lo=3, hi=150) == 3


def test_memory_model_fits_a_line_through_the_end_points():
    """Base and slope come from the smallest and largest probe points."""
    base, slope = per_worker_memory_model([{"copies": 1, "peak_rss_mb": 420.0},
                                           {"copies": 101, "peak_rss_mb": 1420.0}])
    assert abs(slope - 10.0) < 1e-9
    assert abs(base - 410.0) < 1e-9


def test_memory_model_survives_a_single_repeated_copy_count():
    """Two probes at the same copy count give a flat line rather than a division by zero."""
    base, slope = per_worker_memory_model([{"copies": 8, "peak_rss_mb": 500.0},
                                           {"copies": 8, "peak_rss_mb": 505.0}])
    assert slope == 0.0
    assert base == 500.0


def test_predicted_point_gb_multiplies_by_the_worker_count():
    """Two hundred workers of one gigabyte each is two hundred gigabytes."""
    assert abs(predicted_point_gb((1024.0, 0.0), 16, 200) - 200.0) < 1e-9


def test_predicted_node_gb_extrapolates_from_the_two_largest_points():
    """Doubling the copies again adds the same memory the last doubling added."""
    rows = [{"copies": 16, "node_peak_used_gb": 100.0},
            {"copies": 64, "node_peak_used_gb": 190.0},
            {"copies": 128, "node_peak_used_gb": 300.0}]
    assert abs(predicted_node_gb(rows, 256) - 520.0) < 1e-9


def test_predicted_node_gb_with_one_measured_copy_count_returns_that_reading():
    """Two readings at the same copy count give no slope, so the reading itself is the answer."""
    rows = [{"copies": 32, "node_peak_used_gb": 140.0},
            {"copies": 32, "node_peak_used_gb": 142.0}]
    assert predicted_node_gb(rows, 256) == 142.0


def test_aggregate_sums_the_worker_rates():
    """Workers run at the same time, so the machine's rate is the sum of theirs."""
    rows = [{"sec_per_iteration": 0.5, "sec_per_iteration_opening_three": 0.4},
            {"sec_per_iteration": 0.6, "sec_per_iteration_opening_three": 0.4}]
    point = aggregate(rows, copies=16, procs=2)
    assert point["total_copies"] == 32
    assert abs(point["env_steps_per_sec"] - (512 * 16 / 0.5 + 512 * 16 / 0.6)) < 1e-6
    assert abs(point["env_steps_per_sec_per_copy"]
               - point["env_steps_per_sec"] / 32) < 1e-9


def test_aggregate_reports_the_opening_burst_separately():
    """The opening-three reading is faster here, and is carried as its own set of fields."""
    rows = [{"sec_per_iteration": 0.5, "sec_per_iteration_opening_three": 0.4}]
    point = aggregate(rows, copies=1, procs=1)
    assert point["env_steps_per_sec_opening_three"] > point["env_steps_per_sec"]
    assert abs(point["sec_per_iteration_opening_three"] - 0.4) < 1e-9


def test_burst_and_settled_separates_the_opening_from_the_closing_iterations():
    """The first three iterations and the last third are reported apart from the whole run."""
    r = burst_and_settled([0.40, 0.41, 0.42, 0.50, 0.51, 0.52])
    assert abs(r["sec_per_iteration_opening_three"] - 0.41) < 1e-9
    assert abs(r["sec_per_iteration_last_third"] - 0.52) < 1e-9
    assert abs(r["sec_per_iteration"] - 0.50) < 1e-9
    assert r["sec_fastest"] == 0.40 and r["sec_slowest"] == 0.52


def test_burst_and_settled_handles_a_three_iteration_run():
    """With three iterations the opening reading and the whole-run reading coincide."""
    r = burst_and_settled([1.0, 2.0, 3.0])
    assert r["sec_per_iteration"] == r["sec_per_iteration_opening_three"] == 2.0
