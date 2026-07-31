#!/usr/bin/env python
"""Unit tests for the novel logic in monitoring_report.py: worker-log parsing and per-metric marking.
Run:  <exploration python> -m pytest 20_mins_monitoring/test_monitoring_report.py -q
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import monitoring_report as mr  # noqa: E402


def test_parse_worker_log(tmp_path):
    """parse_worker_log counts disjoint claim / done / failed lines and captures the done seconds."""
    # a log with 3 claims, 2 rc=0 finishes (45273s, 46247s), 1 rc=1 finish -> 2 done, 1 failed
    log = tmp_path / "pmam-w30-1_9999999.log"
    log.write_text(
        "[2026-07-23T03:08:56 worker 9.1.2 sweep=s] claimed 00031_of_92400_x_seed0.json :: env algo\n"
        "[2026-07-23T03:08:57 worker 9.2.3 sweep=s] claimed 00032_of_92400_y_seed0.json :: env algo\n"
        "[2026-07-23T03:08:58 worker 9.3.4 sweep=s] claimed 00033_of_92400_z_seed0.json :: env algo\n"
        "[2026-07-23T15:43:35 worker 9.1.2 sweep=s] 00031_of_92400_x_seed0.json rc=0 in 45273s -> done "
        "(running totals: 1 done, 0 failed)\n"
        "[2026-07-23T15:59:43 worker 9.2.3 sweep=s] 00032_of_92400_y_seed0.json rc=0 in 46247s -> done "
        "(running totals: 2 done, 0 failed)\n"
        "[2026-07-23T06:36:50 worker 9.3.4 sweep=s] 00033_of_92400_z_seed0.json rc=1 in 12457s -> "
        "failed (running totals: 2 done, 1 failed)\n")
    claims, done, failed, fin = mr.parse_worker_log(str(log))
    assert (claims, done, failed) == (3, 2, 1)
    assert fin == [45273, 46247]
    # running_now = claims - done - failed = 3 - 2 - 1 = 0 (floor already satisfied)
    assert max(0, claims - done - failed) == 0


def test_mark_rows():
    """mark_rows bolds the best and underlines the second best per metric column, respecting direction
    (steps lower-is-better), skips None cells, and bolds all tied-best without an underline."""
    rows = [
        {"reward": "-79.99", "steps": "80.3", "success": "1.000", "reward100": "-78.9",
         "mcov": "100.00", "cov1m": "100.00",
         "_raw": {"reward": -79.99, "steps": 80.3, "success": 1.0, "reward100": -78.9,
                  "mcov": 100.0, "cov1m": 100.0}},
        {"reward": "-80.37", "steps": "80.7", "success": "1.000", "reward100": "-78.6",
         "mcov": "100.00", "cov1m": "100.00",
         "_raw": {"reward": -80.37, "steps": 80.7, "success": 1.0, "reward100": -78.6,
                  "mcov": 100.0, "cov1m": 100.0}},
        {"reward": "-81.01", "steps": "—", "success": "1.000", "reward100": "-79.0",
         "mcov": "93.85", "cov1m": "93.85",
         "_raw": {"reward": -81.01, "steps": None, "success": 1.0, "reward100": -79.0,
                  "mcov": 93.85, "cov1m": 93.85}},
    ]
    mr.mark_rows(rows)
    # reward higher-is-better: -79.99 best (bold), -80.37 second (underline), -81.01 unmarked
    assert rows[0]["reward"] == "**-79.99**"
    assert rows[1]["reward"] == "<u>-80.37</u>"
    assert rows[2]["reward"] == "-81.01"
    # steps lower-is-better: 80.3 best (bold), 80.7 second (underline); the None ("—") row is skipped
    assert rows[0]["steps"] == "**80.3**"
    assert rows[1]["steps"] == "<u>80.7</u>"
    assert rows[2]["steps"] == "—"
    # success all tied at 1.000 -> all bold, none underlined
    assert all(r["success"] == "**1.000**" for r in rows)
    # mcov: two tied best at 100.00 (both bold), 93.85 is the second distinct value (underline)
    assert rows[0]["mcov"] == "**100.00**"
    assert rows[1]["mcov"] == "**100.00**"
    assert rows[2]["mcov"] == "<u>93.85</u>"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
