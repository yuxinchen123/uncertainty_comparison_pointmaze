"""Turn the counter job's text output into one record per measured setting.

`perf stat` prints for people, not for programs, so this reads its output back into numbers: one
record per counting window, carrying the setting it was taken under and every event that window
counted. The report quotes ratios from these — instructions per cycle, the share of address
translations that missed — rather than converting counts into bytes, because the generic
`cache-misses` event does not have one fixed meaning on this processor and a byte figure derived
from it would be a guess wearing a unit.

Usage:
  python parse_perf_counters.py
"""
import json
import re
from pathlib import Path

RUN = Path(__file__).resolve().parent.parent
# both counter jobs, the later-settled one last so its records win where they overlap
SOURCES = [RUN / "logs" / "perf_counters.txt",
           RUN / "logs" / "perf_counters_settled.txt"]
OUT = RUN / "data" / "perf_counters.json"

HEADER = re.compile(r"=====\s+procs=(\d+)\s+copies=(\d+)\s+style=(\S+)")
COUNT = re.compile(r"^\s*([\d,]+)\s+([A-Za-z0-9_.-]+)\b")


def parse(text):
    """One record per counting window: the setting, and every event count inside it.

    before: a header line "===== procs=112 copies=64 style=full_batch est=0.6 ... =====" followed
            by lines like "    43,132,083,238      cache-misses              # 14.662 % ..."
    after:  {"workers": 112, "n_copies": 64, "style": "full_batch",
             "cache-misses": 43132083238.0, ...}
    """
    # a window counts its events in three passes, so an event can appear twice; the LAST
    # reading wins, because the later passes count fewer events at once and perf shares the
    # hardware counters between events by turns when it is asked for too many
    records, current = [], None
    for line in text.splitlines():
        head = HEADER.search(line)
        if head:
            current = {"workers": int(head.group(1)), "n_copies": int(head.group(2)),
                       "style": head.group(3)}
            records.append(current)
            continue
        if current is None:
            continue
        hit = COUNT.match(line)
        if hit:
            current[hit.group(2)] = float(hit.group(1).replace(",", ""))
    return records


def derived(record):
    """The ratios the report quotes, added to a record that has the counts they come from."""
    out = dict(record)
    if record.get("cycles"):
        out["instructions_per_cycle"] = record.get("instructions", 0) / record["cycles"]
    if record.get("dTLB-loads"):
        out["dtlb_load_miss_percent"] = (100 * record.get("dTLB-load-misses", 0)
                                         / record["dTLB-loads"])
    if record.get("L1-dcache-loads"):
        out["l1_load_miss_percent"] = (100 * record.get("L1-dcache-load-misses", 0)
                                       / record["L1-dcache-loads"])
    if record.get("cycles") and record.get("stalled-cycles-backend"):
        out["backend_stall_percent"] = 100 * record["stalled-cycles-backend"] / record["cycles"]
    return out


def main():
    """Parse the counter output and write one record per setting."""
    # one record per setting, the later-settled job's reading replacing the earlier one, since
    # its counting window is certainly inside the load's timed region
    # before: four records from the first job and three from the second, overlapping at two
    #         settings
    # after:  five records, the overlapping two taken from the second job
    by_setting = {}
    for source in SOURCES:
        if not source.exists():
            continue
        for r in parse(source.read_text()):
            row = derived(r)
            row["settled"] = source.name.endswith("settled.txt")
            by_setting[(row["workers"], row["n_copies"], row["style"])] = row
    records = sorted(by_setting.values(), key=lambda r: (r["style"], r["workers"], r["n_copies"]))
    OUT.write_text(json.dumps(records, indent=1))
    print(f"wrote {OUT} with {len(records)} records")


if __name__ == "__main__":
    main()
