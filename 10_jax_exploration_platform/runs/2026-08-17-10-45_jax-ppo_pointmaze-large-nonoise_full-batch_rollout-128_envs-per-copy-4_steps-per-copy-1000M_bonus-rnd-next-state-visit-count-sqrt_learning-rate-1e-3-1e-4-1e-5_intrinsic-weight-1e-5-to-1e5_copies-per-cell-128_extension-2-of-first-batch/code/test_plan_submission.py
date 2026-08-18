"""Tests for this run's submission planner: the arithmetic it decides the split by.

The planner spends card-hours, so every claim it makes is checked here against a hand-computable
case rather than against its own output. Run:
  PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python code/test_plan_submission.py
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import plan_submission as ps


def card(node: str, node_class: str, partition: str = "gpu", reservation: str = "",
         free_in_seconds: float = 0.0) -> dict:
    """One free card, in the shape `live_cards` produces."""
    return {"node": node, "node_class": node_class, "partition": partition,
            "reservation": reservation, "free_in_seconds": free_in_seconds}


def test_chunks_partition_every_cell():
    """Each unit's chunks cover copy indices 0..127 of every cell exactly once between them."""
    for k in ps.SPLITS:
        chunks = ps.chunks_of(ps.UNITS[0], k)
        assert len(chunks) == k
        covered = []
        for chunk in chunks:
            assert chunk["copies"] == chunk["copies_per_cell"] * ps.CELLS
            covered += list(range(chunk["copy_index_first"], chunk["copy_index_last"] + 1))
        assert sorted(covered) == list(range(ps.COPIES_PER_CELL)), k
    print("ok  chunks partition every cell's 128 copies, at every split")


def test_the_run_is_a_thousand_million_steps_per_copy():
    """The iteration count is a whole number of windows and lands within 0.01% of 10^9 steps."""
    # the episode clock turns once every lcm(400, 128) / 128 = 25 iterations, and a window must
    # span a whole number of turns or its reward sum carries the episode phase
    clock = math.lcm(ps.EPISODE_STEPS, ps.ROLLOUT_STEPS) // ps.ROLLOUT_STEPS
    assert clock == 25
    assert ps.ITERATIONS % ps.WINDOW_ITERATIONS == 0
    assert ps.WINDOW_ITERATIONS % clock == 0
    assert abs(ps.STEPS_PER_COPY - 1e9) / 1e9 < 1e-4
    assert ps.COPIES_PER_UNIT == 4224 and ps.CELLS == 33
    print("ok  1,953,200 iterations is 9,766 whole windows and 1,000,038,400 steps per copy")


def test_a_probed_cell_is_used_as_measured():
    """A rate this run probed on this card at this copy count is returned unchanged."""
    measured = {("rnd_next_state", "serval06-09", 528):
                {"steps_per_second_per_copy": 55022.0, "build_and_prime_seconds": 21.5,
                 "first_iteration_with_compile_seconds": 27.9}}
    rate = ps.per_copy_rate("rnd_next_state", "serval06-09", 528, ps.load_survey(), measured)
    assert rate == 55022.0
    assert "probed on this card" in ps.rate_source("rnd_next_state", "serval06-09", 528,
                                                   ps.load_survey(), measured)
    print("ok  a probed cell is used as measured, and says so")


def test_an_unprobed_class_is_carried_by_the_card_ratio_only():
    """A class nothing probed gets the probed rate times the survey's ratio at the same count.

    before: distillation probed at 55,022 steps/s/copy on an H100 NVL at 528 copies; the survey
            measures 36.21 M steps/s on that class and 9.97 M on the A40 class, both at 512;
    after:  55,022 x 9.97 / 36.21 = 15,149 steps/s/copy on the A40 — the arm and the copy count
            stay measured, only the card ratio is transferred.
    """
    survey = ps.load_survey()
    measured = {("rnd_next_state", "serval06-09", 528):
                {"steps_per_second_per_copy": 55022.0, "build_and_prime_seconds": 21.5,
                 "first_iteration_with_compile_seconds": 27.9}}
    expected = (55022.0 * survey[("jaguar01", 512)]["total_steps_per_second"]
                / survey[("serval06-09", 512)]["total_steps_per_second"])
    rate = ps.per_copy_rate("rnd_next_state", "jaguar01", 528, survey, measured)
    assert abs(rate - expected) < 1e-6, (rate, expected)
    assert "carried by the survey's card ratio at 512 copies" in ps.rate_source(
        "rnd_next_state", "jaguar01", 528, survey, measured)
    print("ok  an unprobed class is carried by the card ratio alone, at the same copy count")


def test_no_arm_factor_crosses_copy_counts():
    """A chunk size no probe covers has no price at all, rather than an extrapolated one."""
    survey = ps.load_survey()
    measured = {("rnd_next_state", "serval06-09", 528): {"steps_per_second_per_copy": 55022.0}}
    assert ps.per_copy_rate("rnd_next_state", "jaguar01", 1056, survey, measured) is None
    assert ps.per_copy_rate("rnd_next_state", "jaguar01", 264, survey, measured) is None
    print("ok  a copy count no probe covers is unpriced, not extrapolated")


def test_the_scheduler_prefers_a_free_slow_card_only_when_it_finishes_sooner():
    """Chunks over one fast card and one slow card land wherever they finish soonest.

    before: three distillation chunks, an H100 NVL card and an A40 card, the A40 about 3.6 times
            slower for the same chunk;
    after:  all three stay on the H100 because three of them still end before one A40 chunk would;
            with eight, the A40 does finish some of them sooner and gets them.
    """
    survey = ps.load_survey()
    catalog = ps.load_class_catalog()
    measured = {("rnd_next_state", "serval06-09", 528):
                {"steps_per_second_per_copy": 55022.0, "build_and_prime_seconds": 0.0,
                 "first_iteration_with_compile_seconds": 0.0}}
    cards = [card("serval06", "serval06-09"), card("jaguar01", "jaguar01")]
    fast = ps.chunk_seconds("rnd_next_state", "serval06-09", 528, survey, measured)
    slow = ps.chunk_seconds("rnd_next_state", "jaguar01", 528, survey, measured)
    assert slow > 3 * fast > 0, (fast, slow)

    assignment, makespan = ps.schedule(ps.chunks_of(ps.UNITS[0], 8)[:3], cards, survey, catalog,
                                       measured)
    assert [row["node"] for row in assignment] == ["serval06"] * 3
    assert abs(makespan - 3 * fast) < 1e-6

    assignment, makespan = ps.schedule(ps.chunks_of(ps.UNITS[0], 8), cards, survey, catalog,
                                       measured)
    on_slow = sum(1 for row in assignment if row["node"] == "jaguar01")
    assert on_slow == 1, on_slow
    assert abs(makespan - max(7 * fast, slow)) < 1e-6
    print("ok  the scheduler places by earliest finish, not by card speed")


def test_a_card_that_is_not_free_yet_carries_its_wait():
    """A card with a nonzero free_in_seconds finishes that much later than a free one."""
    survey = ps.load_survey()
    measured = {("rnd_next_state", "serval06-09", 528):
                {"steps_per_second_per_copy": 55022.0, "build_and_prime_seconds": 0.0,
                 "first_iteration_with_compile_seconds": 0.0}}
    chunks = ps.chunks_of(ps.UNITS[0], 8)[:1]
    catalog = ps.load_class_catalog()
    _, now = ps.schedule(chunks, [card("serval06", "serval06-09")], survey, catalog, measured)
    _, later = ps.schedule(chunks, [card("serval06", "serval06-09", free_in_seconds=3600.0)],
                           survey, catalog, measured)
    assert abs((later - now) - 3600.0) < 1e-6
    print("ok  a card that is not free yet carries its wait into the finish time")


def test_memory_excludes_a_card_too_small_for_an_uncut_unit():
    """The whole distillation unit needs 18 GiB, so a 16 GB card may hold an eighth but not it."""
    catalog = ps.load_class_catalog()
    assert not ps.fits("rnd_next_state", "nekomata01", 4224, catalog)
    assert ps.fits("rnd_next_state", "nekomata01", 528, catalog)
    assert ps.fits("gt_position_velocity_sqrt", "nekomata01", 4224, catalog)
    print("ok  card memory excludes an uncut distillation unit from a 16 GB card")


def test_the_reservation_name_reaches_the_plan():
    """A node under our own reservation carries its name, because a job must pass it."""
    assert ps.reservation_of("reserved sl5nw_156 (ours)") == "sl5nw_156"
    assert ps.reservation_of("open") == ""
    assert ps.reservation_of("reserved cs_admin_maint") == ""
    print("ok  our reservation's name is parsed out of the availability note")


def main() -> None:
    """Run every test in this file."""
    test_chunks_partition_every_cell()
    test_the_run_is_a_thousand_million_steps_per_copy()
    test_a_probed_cell_is_used_as_measured()
    test_an_unprobed_class_is_carried_by_the_card_ratio_only()
    test_no_arm_factor_crosses_copy_counts()
    test_the_scheduler_prefers_a_free_slow_card_only_when_it_finishes_sooner()
    test_a_card_that_is_not_free_yet_carries_its_wait()
    test_memory_excludes_a_card_too_small_for_an_uncut_unit()
    test_the_reservation_name_reaches_the_plan()
    print("all planner tests passed")


if __name__ == "__main__":
    main()
