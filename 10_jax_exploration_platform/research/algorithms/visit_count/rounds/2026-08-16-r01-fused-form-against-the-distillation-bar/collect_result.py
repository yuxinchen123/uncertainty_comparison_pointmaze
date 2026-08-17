"""Assemble this round's `result.json` from the benchmark output it rests on.

The raw output of every measurement lands under `benchmark_runs/` and is not committed, so the
round keeps its own small copy of the numbers its verdict rests on. Re-running this script against
the same benchmark folder reproduces the file exactly.

  /p/rlprojects/RND/.venvs/platform_jax/bin/python collect_result.py \
    --benchmark-run <platform>/benchmark_runs/2026-08-16_visit-count-gate
"""
import argparse
import json
from pathlib import Path


def load(folder: Path, pattern: str) -> dict:
    """The one benchmark file in `folder` matching `pattern`, refusing an ambiguous match."""
    hits = sorted(folder.glob(pattern))
    if len(hits) != 1:
        raise FileNotFoundError(f"{len(hits)} files match {pattern} in {folder}; the round's "
                                f"record must name exactly one measurement per row")
    return json.loads(hits[0].read_text())


def gate_rows(paired: dict, solo_peak: dict) -> list:
    """One row per arm of a paired measurement: its throughput, its memory, its verdict.

    before: the paired file's `throughput` list and `gate` dict, and a {(bonus, copies): peak MiB}
            map from the one-arm processes. The key carries the copy count as well as the name,
            because the same bonus was measured alone at both 8,448 and 768 copies and a
            name-only key would let the smaller run's memory overwrite the larger one's.
    after:  rows carrying copies, seconds per iteration, both rates, the memory, and — for the
            candidate arms — the ratio against the bar and how many rounds it won
    """
    bar = paired["arms"][0]
    rows = []
    for row in paired["throughput"]:
        name = row["bonus"]
        verdict = paired["gate"].get(name)
        rows.append({**row,
                     "peak_device_mib_measured_alone": solo_peak[(name, row["copies"])],
                     "is_the_bar": name == bar,
                     "throughput_ratio_against_bar": verdict["throughput_ratio_against_bar"]
                     if verdict else 1.0,
                     "paired_rounds_faster_than_bar": verdict["paired_rounds_faster_than_bar"]
                     if verdict else None,
                     "rounds": paired["rounds"],
                     "passes_gate": verdict["passes_gate"] if verdict else None})
    return rows


def main():
    """Read every measurement of the round and write the round's own record beside this script."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark-run", required=True)
    args = ap.parse_args()
    run = Path(args.benchmark_run)

    solo = [json.loads(p.read_text()) for p in sorted((run / "solo").glob("*.json"))]
    peaks = {(f["arms"][0], f["copies"]): f["peak_device_mib"] for f in solo}
    sqrt = load(run / "paired", "*copies-8448*sqrt.json")
    linear = load(run / "paired", "*copies-8448*linear.json")
    none = load(run / "paired", "*copies-768*none.json")
    # every profile run is kept, not just the last: the bonus's share is a difference of two
    # ~33.8 ms numbers, so its noise is only visible across repeats of the same measurement
    profiles = [json.loads(p.read_text())
                for p in sorted((run / "profile").glob("*visit_count_phases*.json"))]
    if not profiles:
        raise FileNotFoundError(f"no phase profile under {run / 'profile'}")
    profile = profiles[-1]

    out = Path(__file__).resolve().parent / "result.json"
    out.write_text(json.dumps({
        "round": "2026-08-16-r01-fused-form-against-the-distillation-bar",
        "family": "visit_count",
        "device": sqrt["devices"], "jax": sqrt["jax"], "git": sqrt["git"],
        "measured_at": [f["measured_at"] for f in (sqrt, linear, none, *profiles)],
        "protocol": {
            "paired": "the arms alternate round by round in one process on one card, the order "
                      "flips every round, every iteration is waited for",
            "rounds": sqrt["rounds"], "iterations_per_round": sqrt["iterations_per_round"],
            "peak_memory": "measured in separate one-arm processes, because the allocator's "
                           "high-water mark belongs to the process",
            "shape": "the shipped sweep: learning rates x intrinsic weights x 256 copies per cell,"
                     " one update per batch, 128 rollout steps of 4 environments"},
        "gate_at_8448_copies_visit_count_sqrt": gate_rows(sqrt, peaks),
        "gate_at_8448_copies_visit_count_linear": gate_rows(linear, peaks),
        "gate_at_768_copies_no_bonus": gate_rows(none, peaks),
        "phase_profile_seconds_at_8448_copies": profile["seconds"],
        "phase_profile_seconds_every_run": [p["seconds"] for p in profiles],
        "phase_profile_context": {k: profile[k] for k in
                                  ("bonus", "copies", "update_style", "table_entries_per_copy",
                                   "rows_scattered_per_copy", "distinct_indices_per_copy_mean")},
        "ran_check": {"visit_count_sqrt": sqrt["ran_check"], "no_bonus": none["ran_check"]},
    }, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
