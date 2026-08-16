"""Tests for the helper functions of the processor-node comparison section."""
import matplotlib
matplotlib.use("Agg")

import pytest

from cpu_node_comparison import (at_workers, best_config, median_per_worker_count,
                                 per_core_l3_mib, size_to_mib, verdict_sentence)


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


def base_extrapolation(share_threads, share_best):
    """An extrapolation dictionary with only the fields the verdict sentence reads."""
    return {"share_of_big_at_threads": share_threads, "share_of_big_best": share_best,
            "big_cores": 112, "big_threads": 224, "projected_total": 160_988,
            "big_total_at_threads": 113_822, "big_best_total": 129_574, "big_best_workers": 112}


def test_verdict_says_yes_when_the_projection_wins():
    text = verdict_sentence(base_extrapolation(1.41, 1.24))
    assert "answer is yes" in text
    assert "141%" in text and "124%" in text


def test_verdict_says_no_when_the_projection_loses():
    # the sentence must follow the numbers rather than a hand-written conclusion, because the
    # sustained measurements reversed what the short ones said
    text = verdict_sentence(base_extrapolation(0.45, 0.40))
    assert "answer is no" in text
    assert "45%" in text
