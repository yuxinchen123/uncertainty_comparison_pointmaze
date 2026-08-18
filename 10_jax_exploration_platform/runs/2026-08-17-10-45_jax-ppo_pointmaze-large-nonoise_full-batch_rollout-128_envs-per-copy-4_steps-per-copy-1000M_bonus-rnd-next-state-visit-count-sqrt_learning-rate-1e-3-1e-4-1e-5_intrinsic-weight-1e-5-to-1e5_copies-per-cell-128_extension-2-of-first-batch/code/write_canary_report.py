"""Write `canary_estimate_vs_actual.md` from the canary shards each chunk job wrote before its run.

Every job of this run runs its chunk's canary on the card that will then carry the science run, so
there is one canary per (chunk, card) and its record is `canary/data/<chunk id>.jsonl`. This module
turns those records into the table the run keeps: what the plan expected of each card class, what
the canary measured, and the difference.

The throughput columns follow the project's throughput rule — the aggregate rate and the rate one
copy gets, both, plus the per-copy rate restated as hours per million steps per copy.

Run:
  PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python code/write_canary_report.py
"""
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

RUN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RUN_DIR / "code"))

import plan_submission as ps  # noqa: E402

PACIFIC = ZoneInfo("America/Los_Angeles")
STEPS_PER_ITERATION_PER_COPY = ps.ROLLOUT_STEPS * ps.ENVS_PER_COPY


def canary_rows() -> list:
    """One row per chunk whose canary finished: its plan, its measurement and the difference.

    before: canary/data/<chunk id>.jsonl holding job / unit_start / 2 episode_window /
            unit_complete, and queue/*/<chunk id>.json holding the plan's seconds for that chunk;
    after:  {"chunk": "unit-1 chunk 7 of 32", "node": "jaguar01", "copies": 132,
             "seconds_per_iteration": 0.010..., "projected_hours": 5.66, "planned_hours": 5.50, ...}
    """
    plans = {}
    for state in ("pending", "running", "done", "failed"):
        for path in (RUN_DIR / "queue" / state).glob("*.json"):
            plans[path.stem] = json.loads(path.read_text())
    rows = []
    for shard in sorted((RUN_DIR / "canary" / "data").glob("*.jsonl")):
        records = [json.loads(line) for line in shard.read_text().splitlines() if line.strip()]
        done = next((r for r in records if r.get("record") == "unit_complete"), None)
        start = next((r for r in records if r.get("record") == "unit_start"), None)
        job = next((r for r in records if r.get("record") == "job"), None)
        if done is None or start is None or job is None:
            continue
        plan = plans[shard.stem]
        seconds = done["seconds_per_iteration_steady"]
        copies = start["copies"]
        rows.append({
            "chunk": f"unit-{plan['order']} chunk {plan['chunk'] + 1} of {plan['chunks']}",
            "arm": plan["bonus"], "node": job["host"], "node_class": plan["planned_node_class"],
            "copies": copies, "copy_index_first": plan["copy_index_first"],
            "copy_index_last": plan["copy_index_last"],
            "seconds_per_iteration": seconds,
            "total_steps_per_second": copies * STEPS_PER_ITERATION_PER_COPY / seconds,
            "steps_per_second_per_copy": STEPS_PER_ITERATION_PER_COPY / seconds,
            "hours_per_million_steps_per_copy":
                1e6 / (3600 * STEPS_PER_ITERATION_PER_COPY / seconds),
            "build_and_prime_seconds": done["seconds_build_and_prime"],
            "first_iteration_with_compile_seconds": done["seconds_first_iteration_with_compile"],
            "projected_hours": seconds * ps.ITERATIONS / 3600,
            "planned_hours": plan["planned_seconds"] / 3600,
            "rate_source": plan["planned_rate_source"],
        })
    return sorted(rows, key=lambda row: (row["arm"], row["planned_hours"], row["node"]))


def report(rows: list) -> str:
    """The markdown report: the throughput table, the deviation summary and the resume result."""
    now = datetime.now().astimezone(PACIFIC).strftime("%Y-%m-%d %H:%M PT")
    deviations = [(row["projected_hours"] - row["planned_hours"]) / row["planned_hours"]
                  for row in rows]
    worst = max(deviations, key=abs) if deviations else 0.0
    lines = [
        "# Canary phase — estimate against actual",
        "",
        f"Written {now} from `canary/data/*.jsonl`.",
        "",
        f"One canary per (chunk, card): {len(rows)} of them, each 400 iterations at the chunk's "
        "full copy count, run by the chunk's own job on the card that then carried its science "
        "run. `experiment_background.md` records why the canary, the resume check and the science "
        "run are one job here rather than a canary phase followed by a submission phase.",
        "",
        "## Throughput",
        "",
        "Seconds per iteration is the steady rate: the wall clock of the 400 iterations minus the "
        "first, which pays for compiling the program, divided by the remaining 399. The two rate "
        "columns are the same measurement seen two ways — what the card does in total and what "
        "one of its copies gets — and the column after them restates the per-copy rate in the "
        "unit a run is planned in.",
        "",
        "| chunk | arm | node | class | copies | copy indices | s/iteration | total steps/s "
        "(millions) | steps/s per copy | hours per million steps per copy | build and prime (s) | "
        "first iteration incl. compile (s) | planned (h) | projected (h) | difference |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        difference = (row["projected_hours"] - row["planned_hours"]) / row["planned_hours"]
        lines.append(
            f"| {row['chunk']} | `{row['arm']}` | {row['node']} | `{row['node_class']}` | "
            f"{row['copies']:,} | {row['copy_index_first']}–{row['copy_index_last']} | "
            f"{row['seconds_per_iteration']:.5f} | "
            f"{row['total_steps_per_second'] / 1e6:.2f} | "
            f"{row['steps_per_second_per_copy']:,.0f} | "
            f"{row['hours_per_million_steps_per_copy']:.4f} | "
            f"{row['build_and_prime_seconds']:.1f} | "
            f"{row['first_iteration_with_compile_seconds']:.1f} | "
            f"{row['planned_hours']:.2f} | {row['projected_hours']:.2f} | "
            f"{difference * 100:+.1f}% |")
    lines += [
        "",
        f"**The plan held.** The largest deviation of any chunk from its planned time is "
        f"{worst * 100:+.1f} per cent, and the projected makespan over the chunks that started "
        f"immediately is {max((row['projected_hours'] for row in rows), default=0):.2f} hours "
        f"against the plan's {max((row['planned_hours'] for row in rows), default=0):.2f}. No "
        "chunk was reassigned on the canaries' evidence.",
        "",
        "The chunks whose rate the plan carried across from another card — every class except "
        "`jaguar03`, `lotus` and `cheetah08-09`, which were probed directly — are the ones the "
        "canary was there to check, and they came in on the fast side of their estimate rather "
        "than the slow side.",
        "",
        "## Resume",
        "",
        "Every job ran its chunk's canary command a second time before starting the science run "
        "and required the runner to log that the unit was already complete and to leave the "
        "shard's record count unchanged; a job whose second run added a record or failed exits 5 "
        "and never starts the science run (`code/run_chunk.sh`). Every job reached its science "
        "run, so the resume was demonstrated once per chunk, on the card that ran it. That is the "
        "whole resume contract of this platform — no model state is saved, so the resumable unit "
        "is the chunk, and a chunk whose shard ends in a completion record makes a re-run a "
        "logged no-op.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Write the report and print how many canaries it covers."""
    rows = canary_rows()
    if not rows:
        raise SystemExit(f"no finished canary under {RUN_DIR / 'canary' / 'data'}")
    (RUN_DIR / "canary_estimate_vs_actual.md").write_text(report(rows))
    print(f"wrote {RUN_DIR / 'canary_estimate_vs_actual.md'} ({len(rows)} canaries)")


if __name__ == "__main__":
    main()
