"""Tests for the submission-strategy program's arithmetic.

The plan decides how long this run takes, so its arithmetic is tested on a WORKED EXAMPLE with
fixed inputs — a hand-made survey and a hand-made card list — rather than on live data, which
changes between runs and cannot anchor an assertion. Two properties are checked separately:

  1. the per-chunk time, the schedule and the makespan come out of the arithmetic the docstring
     describes, on inputs whose answer can be worked out by hand;
  2. the chunks of a unit partition its copy indices — disjoint, and together exactly the run —
     which is what makes the union of the shards the 1,024-copy run rather than an overlap.

Run:
  PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python code/test_plan_submission.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plan_submission as ps  # noqa: E402


def toy_survey() -> dict:
    """A two-class survey with round numbers, so every expected time is exact.

    fast: 100 million steps per second at 1,024 copies, 200 million at 512 — so the per-copy rate
    doubles when a unit is halved, which makes the split's benefit a factor of exactly two.
    slow: a tenth of that. Build and compile are zero so setup is only the two constants.
    """
    cell = lambda total: {"total_steps_per_second": total, "build_seconds": 0.0,
                          "compile_seconds": 0.0, "status": "measured"}
    return {("fast", 1024): cell(100e6), ("fast", 512): cell(200e6),
            ("slow", 1024): cell(10e6), ("slow", 512): cell(20e6),
            ("fast", 4096): cell(100e6), ("slow", 4096): cell(10e6)}


def toy_catalog() -> dict:
    """Two classes with cards big enough for anything this test asks of them."""
    return {"fast": {"gpu_mem_mb": 95830}, "slow": {"gpu_mem_mb": 95830}}


def test_arm_factor_is_the_measured_rate_over_the_survey_s():
    """The factor is this platform's measured total rate divided by the survey's on the same cell."""
    survey = toy_survey()
    anchor = ps.ARM_ANCHORS["rnd_next_state"]
    expected = ((anchor["copies"] * ps.ROLLOUT_STEPS * ps.ENVS_PER_COPY
                 / anchor["seconds_per_iteration"]) / 100e6)
    # the anchor's class is renamed to the toy one for the check
    ps.ARM_ANCHORS["rnd_next_state"]["node_class"] = "fast"
    try:
        assert abs(ps.arm_factor("rnd_next_state", survey) - expected) < 1e-9
    finally:
        ps.ARM_ANCHORS["rnd_next_state"]["node_class"] = "serval06-09"


def test_chunk_time_is_steps_over_rate_plus_setup():
    """One chunk's seconds are the steps it must advance over its rate, plus its setup."""
    survey, factors = toy_survey(), {"rnd_next_state": 1.0}
    setup = ps.WARM_CACHE_FRACTION * ps.PROCESS_START_SECONDS + ps.EXPECTED_QUEUE_SECONDS
    # 1,024 copies on `fast`: per-copy rate 100e6 / 1024, so the time is steps x 1024 / 100e6
    whole = ps.chunk_seconds("rnd_next_state", "fast", 1024, survey, factors)
    assert abs(whole - (ps.STEPS_PER_COPY * 1024 / 100e6 + setup)) < 1e-6
    # 512 copies: per-copy rate 200e6 / 512, which is four times better, so the chunk is a quarter
    half = ps.chunk_seconds("rnd_next_state", "fast", 512, survey, factors)
    assert abs(half - (ps.STEPS_PER_COPY * 512 / 200e6 + setup)) < 1e-6
    assert abs((whole - setup) / (half - setup) - 4.0) < 1e-9


def test_an_unmeasured_copy_count_is_not_priced():
    """A cell the survey never measured returns no time, so no plan can be built on it."""
    assert ps.chunk_seconds("rnd_next_state", "fast", 256, toy_survey(),
                            {"rnd_next_state": 1.0}) is None
    assert ps.per_copy_rate("rnd_next_state", "fast", 256, toy_survey(),
                            {"rnd_next_state": 1.0}) is None


def two_whole_chunks() -> list:
    """Two whole units as single chunks, for the scheduling tests."""
    return [{"order": 1, "arm": "rnd_next_state", "intrinsic_weight": "10", "chunk": 0,
             "chunks": 1, "copies": 1024, "copy_index_first": 0, "copy_index_last": 1023},
            {"order": 4, "arm": "none", "intrinsic_weight": None, "chunk": 0, "chunks": 1,
             "copies": 1024, "copy_index_first": 0, "copy_index_last": 1023}]


def test_a_free_slow_card_is_refused_when_stacking_the_fast_one_finishes_sooner():
    """`slow` is ten times slower here, so both chunks stack on the fast card rather than split.

    The fast chunk is 10,240 s and the slow one 102,400 s, so the second chunk lands at 20,480 s on
    the fast card against 102,400 s on the free slow card. Using the free card would be five times
    worse — the arithmetic, not a preference for keeping cards busy, decides.
    """
    survey, factors = toy_survey(), {"rnd_next_state": 1.0, "none": 1.0}
    cards = [{"node": "f1", "node_class": "fast"}, {"node": "s1", "node_class": "slow"}]
    assignment, makespan = ps.schedule(two_whole_chunks(), cards, survey, factors, toy_catalog())
    assert [row["node"] for row in assignment] == ["f1", "f1"]
    fast = ps.chunk_seconds("none", "fast", 1024, survey, factors)
    assert abs(makespan - 2 * fast) < 1e-6


def test_a_second_card_of_the_same_speed_is_used_and_halves_the_makespan():
    """Two equal cards take one chunk each, so the makespan is one chunk rather than two."""
    survey, factors = toy_survey(), {"rnd_next_state": 1.0, "none": 1.0}
    cards = [{"node": "f1", "node_class": "fast"}, {"node": "f2", "node_class": "fast"}]
    assignment, makespan = ps.schedule(two_whole_chunks(), cards, survey, factors, toy_catalog())
    assert sorted(row["node"] for row in assignment) == ["f1", "f2"]
    fast = ps.chunk_seconds("none", "fast", 1024, survey, factors)
    assert abs(makespan - fast) < 1e-6


def test_a_card_that_cannot_hold_the_chunk_is_skipped():
    """A card too small for the arm at that copy count never receives it."""
    catalog = {"fast": {"gpu_mem_mb": 1024}, "slow": {"gpu_mem_mb": 95830}}
    assert not ps.fits("rnd_next_state", "fast", 1024, catalog)
    assert ps.fits("rnd_next_state", "slow", 1024, catalog)


def test_chunks_partition_the_unit_s_copy_indices():
    """Every split covers 0..1023 exactly once, with no index in two chunks."""
    for k in (1, 2, 4, 8):
        chunks = ps.chunks_of(ps.UNITS[0], k)
        assert len(chunks) == k
        covered = []
        for chunk in chunks:
            covered += list(range(chunk["copy_index_first"], chunk["copy_index_last"] + 1))
        assert sorted(covered) == list(range(ps.COPIES_PER_UNIT)), f"k={k} does not cover the run"
        assert len(set(covered)) == len(covered), f"k={k} repeats a copy index"


def test_the_iteration_count_is_whole_windows_and_close_to_the_asked_budget():
    """1,953,200 iterations is 9,766 whole 200-iteration windows and 10^9 steps to 0.005 per cent."""
    assert ps.ITERATIONS % ps.WINDOW_ITERATIONS == 0
    # a window must be a whole number of turns of the episode clock, 400 / gcd(128, 400) = 25
    assert ps.WINDOW_ITERATIONS % 25 == 0
    assert abs(ps.STEPS_PER_COPY - 1e9) / 1e9 < 5e-5


def test_the_proportionality_clause_prefers_fewer_chunks_at_equal_finish():
    """Given two cards of one speed, splitting a unit that gains nothing is not chosen."""
    # `fast` at 512 is only marginally better here, so a split saves less than its setup
    cell = lambda total: {"total_steps_per_second": total, "build_seconds": 0.0,
                          "compile_seconds": 0.0, "status": "measured"}
    survey = {("fast", 1024): cell(100e6), ("fast", 512): cell(50.0e6), ("fast", 4096): cell(100e6)}
    factors = {unit["arm"]: 1.0 for unit in ps.UNITS}
    cards = [{"node": f"f{i}", "node_class": "fast"} for i in range(8)]
    result = ps.plan(cards, survey, factors, {"fast": {"gpu_mem_mb": 95830}})
    # halving the copies here leaves the per-copy rate unchanged (50e6/512 = 100e6/1024), so a
    # split costs one more setup and gains nothing: every unit must stay whole
    assert result["chosen"]["splits"] == [1, 1, 1, 1], result["chosen"]["splits"]


def main() -> None:
    """Run every test in this file and report."""
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"ok  {name}")
    print("all submission-plan tests passed")


if __name__ == "__main__":
    main()
