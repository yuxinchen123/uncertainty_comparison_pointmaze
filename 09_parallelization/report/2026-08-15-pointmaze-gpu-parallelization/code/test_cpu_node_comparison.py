"""Tests for the helper functions of the processor-node comparison section."""
import matplotlib
matplotlib.use("Agg")

import pytest

from cpu_node_comparison import (at_workers, best_config, burst_change, hardware, item,
                                 largest_burst_change, median_per_worker_count, para,
                                 per_core_l3_mib, size_to_mib, throughput_table, verdict_sentence)


def row(workers, total):
    """A results row with only the fields these helpers read."""
    return {"workers": workers, "total_copies": workers, "env_steps_per_sec": total,
            "env_steps_per_sec_per_copy": total / workers, "sec_per_iteration": 0.4}


def test_size_to_mib_handles_each_unit():
    assert size_to_mib("18M") == 18.0
    assert size_to_mib("512K") == pytest.approx(0.5)
    assert size_to_mib("1.3M") == pytest.approx(1.3)


def test_size_to_mib_refuses_an_unknown_unit():
    with pytest.raises(KeyError):
        size_to_mib("18T")


def test_per_core_l3_divides_a_block_by_the_cores_sharing_it():
    # 18 MiB shared by 8 cores is 2.25 MiB each; 32 MiB shared by 7 is about 4.57
    assert per_core_l3_mib({"l3_per_instance": "18M", "cores_sharing_one_l3": 8}) == 2.25
    assert per_core_l3_mib({"l3_per_instance": "32M",
                            "cores_sharing_one_l3": 7}) == pytest.approx(4.571, abs=1e-3)


def test_median_per_worker_count_takes_the_middle_of_three_repeats():
    rows = [row(112, 129_574), row(112, 127_600), row(112, 131_000)]
    out = median_per_worker_count(rows)
    assert len(out) == 1
    assert out[0]["env_steps_per_sec"] == 129_574
    assert out[0]["_repeats"] == 3


def test_median_per_worker_count_sorts_and_keeps_every_worker_count():
    rows = [row(32, 23_000), row(1, 1_372), row(16, 20_475)]
    assert [r["workers"] for r in median_per_worker_count(rows)] == [1, 16, 32]


def test_median_per_worker_count_on_an_empty_list():
    assert median_per_worker_count([]) == []


def test_best_config_picks_the_highest_total_not_the_most_workers():
    # the point of the function: on jaguar03 the 224-worker setting is WORSE than 112, so the
    # largest worker count must not be mistaken for the best setting
    rows = [row(112, 129_574), row(224, 113_822), row(32, 47_200)]
    assert best_config(rows)["workers"] == 112


def test_at_workers_finds_a_row_and_returns_none_when_absent():
    rows = [row(16, 20_475), row(32, 23_000)]
    assert at_workers(rows, 32)["env_steps_per_sec"] == 23_000
    assert at_workers(rows, 64) is None


def test_burst_change_measures_the_fall_from_short_to_sustained():
    # 224 workers: 1,566 steps/s per copy in five iterations, 507 in 150, a fall of about 68%
    short, long = [row(224, 350_784)], [row(224, 113_568)]
    out = burst_change(short, long, 224)
    assert out["burst"] == pytest.approx(1_566, abs=1)
    assert out["sustained"] == pytest.approx(507, abs=1)
    assert out["change"] == pytest.approx(-0.676, abs=1e-3)


def test_burst_change_returns_none_for_a_worker_count_measured_only_one_way():
    assert burst_change([row(224, 350_784)], [row(112, 129_574)], 224) is None


def test_largest_burst_change_takes_the_biggest_move_either_way():
    # 1 worker rises 4%, 16 workers fall 9%, and the 32-worker count measured only one way is
    # ignored, so the biggest move is the fall
    short = [row(1, 1_320), row(16, 21_120), row(32, 22_000)]
    long = [row(1, 1_372), row(16, 19_200)]
    assert largest_burst_change(short, long) == pytest.approx(-0.0909, abs=1e-3)


def test_largest_burst_change_with_no_shared_worker_count():
    assert largest_burst_change([row(1, 1_320)], [row(16, 20_320)]) is None


def base_extrapolation(share_threads, share_best):
    """An extrapolation dictionary with only the fields the verdict sentence reads."""
    return {"share_of_big_at_threads": share_threads, "share_of_big_best": share_best,
            "big_cores": 112, "big_threads": 224, "projected_total": 158_900,
            "big_total_at_threads": 113_822, "big_best_total": 127_600, "big_best_workers": 112}


def test_verdict_says_it_beats_the_big_node_when_the_projection_wins():
    text = verdict_sentence(base_extrapolation(1.40, 1.25))
    assert "beats" in text
    assert "140%" in text and "125%" in text


def test_verdict_says_it_falls_short_when_the_projection_loses():
    # the sentence must follow the numbers rather than a hand-written conclusion, because the
    # sustained measurements reversed what the short ones said
    text = verdict_sentence(base_extrapolation(0.45, 0.40))
    assert "falls short of" in text
    assert "45%" in text


def test_para_never_breaks_a_hyphenated_word_across_two_lines():
    # markdown rejoins wrapped lines with a space, so a break at the hyphen would render as
    # "last- level cache"
    text = para("filler " * 20 + "last-level cache " + "filler " * 20)
    assert "last-\n" not in text
    assert max(len(line) for line in text.splitlines()) <= 95


def test_item_hangs_its_continuation_lines_under_the_number():
    lines = item(2, "word " * 60).splitlines()
    assert lines[0].startswith("2. ")
    assert all(line.startswith("   ") for line in lines[1:])


def test_throughput_table_keeps_the_columns_the_convention_requires():
    if hardware() is None:
        pytest.skip("the nodes' hardware record is not on this machine")
    header = throughput_table().splitlines()[0]
    for column in ("worker processes", "copies per worker", "copies in total",
                   "seconds per iteration", "million steps per second",
                   "thousand steps per second per copy", "hours per million steps per copy"):
        assert column in header
