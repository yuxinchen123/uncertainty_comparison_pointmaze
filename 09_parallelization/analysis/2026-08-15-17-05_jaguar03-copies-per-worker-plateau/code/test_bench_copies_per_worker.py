"""Unit tests for the arithmetic of the copies-per-worker sweep."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_copies_per_worker import (aggregate, common_window, iteration_times,  # noqa: E402
                                     iters_for, per_worker_memory_model, predicted_node_gb,
                                     predicted_point_gb, warmup_for)


def test_iters_for_hits_the_upper_bound_on_a_fast_point():
    """A fast iteration would need more than the bound to fill the target, so the bound applies."""
    assert iters_for(0.4, target=90.0, cap=420.0, lo=5, hi=150) == 150


def test_iters_for_times_about_the_target_on_a_slow_point():
    """Seven seconds an iteration gives thirteen iterations, about ninety seconds of work."""
    assert iters_for(7.0, target=90.0, cap=420.0, lo=5, hi=150) == 13


def test_iters_for_respects_the_ceiling_on_a_very_slow_point():
    """A two-hundred-second iteration already overshoots the target, so it is run once."""
    assert iters_for(200.0, target=90.0, cap=420.0, lo=1, hi=150) == 1


def test_iters_for_never_goes_below_the_floor():
    """The floor wins over the ceiling, so a point is always measured at least that many times."""
    assert iters_for(600.0, target=90.0, cap=420.0, lo=5, hi=150) == 5


def test_warmup_runs_about_twenty_seconds():
    """A slow iteration needs few warm-up passes, a fast one needs many, both bounded."""
    assert warmup_for(7.0, target=20.0, lo=1, hi=30) == 3
    assert warmup_for(0.4, target=20.0, lo=1, hi=30) == 30
    assert warmup_for(100.0, target=20.0, lo=1, hi=30) == 1


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


def test_iteration_times_are_the_gaps_between_the_stamps():
    """A worker's iteration durations are the differences of its start and its end stamps."""
    assert iteration_times({"start": 10.0, "ends": [11.0, 13.0, 16.0]}) == [1.0, 2.0, 3.0]


def test_common_window_counts_only_iterations_inside_the_shared_window():
    """The window opens when the last worker starts and closes when the first one finishes.

    Worker one runs iterations of one second from t=0, worker two from t=0.2. The window is
    [0.2, 2.0]: worker one's second iteration and worker two's first lie inside it, so two
    iterations of work were done in 1.8 seconds.
    """
    rows = [{"start": 0.0, "ends": [1.0, 2.0]}, {"start": 0.2, "ends": [1.2, 2.2]}]
    w = common_window(rows, copies=1)
    assert abs(w["window_seconds"] - 1.8) < 1e-9
    assert w["iterations_in_window"] == 2
    assert abs(w["env_steps_per_sec"] - 2 * 512 / 1.8) < 1e-6


def test_common_window_is_lower_than_the_sum_of_worker_rates_when_workers_are_staggered():
    """The old aggregate counts every worker at full rate however little they overlapped."""
    rows = [{"start": 0.0, "ends": [1.0, 2.0]}, {"start": 0.9, "ends": [1.9, 2.9]}]
    point = aggregate(rows, copies=1, procs=2)
    assert point["env_steps_per_sec"] < point["env_steps_per_sec_sum_of_worker_rates"]


def test_aggregate_reports_the_opening_three_iterations_separately():
    """The opening reading is the old aggregate over the first three iterations only."""
    rows = [{"start": 0.0, "ends": [0.4, 0.8, 1.2, 2.2, 3.2, 4.2, 5.2]},
            {"start": 0.0, "ends": [0.4, 0.8, 1.2, 2.2, 3.2, 4.2, 5.2]}]
    point = aggregate(rows, copies=1, procs=2)
    # the first three iterations took 0.4s each, the whole run has a median of 1.0s, so the
    # opening reading is the faster one
    assert (point["env_steps_per_sec_sum_of_worker_rates_opening_three"]
            > point["env_steps_per_sec_sum_of_worker_rates"])
    assert abs(point["sec_per_iteration"] - 1.0) < 1e-9
