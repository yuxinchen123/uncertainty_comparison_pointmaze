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
    assert round(solo["sec_per_iteration"], 3) == 0.326
    assert round(solo["total"] / 1e6, 4) == 0.3527
    assert round(solo["per_copy"] / 1e3, 2) == 1.57
    # same measurement file, selected by maximum total instead: the 3,584-copy setting
    assert by_name["processor, independent processes, one update"]["copies"] == 3584
    # the pinned row is the fastest processor row per copy, and the table is sorted by total
    cpu_rows = [r for r in rows if r["kind"] == "cpu"]
    assert max(cpu_rows, key=lambda r: r["per_copy"])["copies"] == 224
    assert [r["total"] for r in rows] == sorted((r["total"] for r in rows), reverse=True)


def test_best_under_small_limit_keeps_the_pinned_row():
    """A limit below the large settings drops them and leaves the pinned 224-copy row standing."""
    rows = c.best_under(limit=256)
    assert max(r["copies"] for r in rows) <= 256
    assert any(r["pinned"] and r["copies"] == 224 for r in rows)
