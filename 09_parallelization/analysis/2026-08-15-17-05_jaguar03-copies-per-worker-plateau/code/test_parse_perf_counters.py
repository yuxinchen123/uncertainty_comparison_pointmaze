"""Unit tests for reading perf's printed output back into numbers."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from parse_perf_counters import derived, parse  # noqa: E402

SAMPLE = """host jaguar03
===== procs=112 copies=64 style=full_batch est=0.6 2026-08-16T02:52:35-04:00 =====
--- group 1: cache and instruction counts ---
    43,132,083,238      cache-misses              #   14.662 % of all cache refs
 4,144,289,282,931      instructions              #    1.34  insn per cycle
 3,085,871,485,176      cycles
      20.207548762 seconds time elapsed
--- group 2: address translation ---
     1,404,002,243      dTLB-load-misses          #    3.64% of all dTLB cache accesses
    38,593,966,944      dTLB-loads
===== procs=1 copies=64 style=full_batch est=0.6 2026-08-16T02:46:38-04:00 =====
   147,005,475,006      instructions              #    1.92  insn per cycle
    76,506,552,955      cycles
"""


def test_parse_makes_one_record_per_window():
    """Each header starts a new record and the counts under it belong to that record."""
    records = parse(SAMPLE)
    assert [r["workers"] for r in records] == [112, 1]
    assert records[0]["n_copies"] == 64 and records[0]["style"] == "full_batch"
    assert records[0]["cache-misses"] == 43132083238.0
    assert records[0]["dTLB-loads"] == 38593966944.0


def test_parse_ignores_lines_before_the_first_window():
    """Text printed before any header belongs to no record and is dropped."""
    assert parse("host jaguar03\n   1,234      cache-misses\n") == []


def test_derived_ratios():
    """Instructions per cycle and the translation miss share come from the counts."""
    r = derived(parse(SAMPLE)[0])
    assert abs(r["instructions_per_cycle"] - 4144289282931 / 3085871485176) < 1e-12
    assert abs(r["dtlb_load_miss_percent"] - 100 * 1404002243 / 38593966944) < 1e-9


def test_derived_leaves_out_a_ratio_whose_counts_are_missing():
    """A window that did not count the translation events carries no translation ratio."""
    r = derived(parse(SAMPLE)[1])
    assert "dtlb_load_miss_percent" not in r
    assert abs(r["instructions_per_cycle"] - 147005475006 / 76506552955) < 1e-12
