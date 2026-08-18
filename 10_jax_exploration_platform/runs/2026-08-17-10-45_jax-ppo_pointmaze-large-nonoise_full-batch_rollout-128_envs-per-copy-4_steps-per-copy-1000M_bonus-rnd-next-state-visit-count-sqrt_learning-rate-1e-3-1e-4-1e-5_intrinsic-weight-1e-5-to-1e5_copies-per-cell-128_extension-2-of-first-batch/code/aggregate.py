"""Roll this run's shards up into run-level metrics, and score every configuration of every arm.

This is the run's own aggregation module. It reads only `data/*.jsonl` and `slurm/jobs/*/*.json`,
and everything downstream — the 20-minute status table, the development document's tables, the
training-curve figure — imports the functions here rather than re-deriving a number from the shards.

A shard here is one CHUNK of one arm, and a chunk carries ALL 33 configurations of that arm at a
slice of their copies. So a configuration's numbers are pooled twice over: within a shard, the
copies belonging to that configuration are selected by the shard's own `copy_cell` list; across
shards, the chunks' per-copy values are concatenated. Pooling is exact rather than an average of
averages, because every per-copy quantity is carried per copy all the way to the end.

**Everything here STREAMS.** This run's forty shards come to about 4 GB of JSON — 9,766 window
records per chunk, each carrying one reward per copy — so a module that read them into memory the
way the parent runs' did would need tens of gigabytes. Each pass therefore reads a shard line by
line and keeps only what it is accumulating: a running per-copy reward sum, an episode count, and
the last scored window. Memory is a few megabytes whatever the run's length.

Three rules the scoring obeys:

1. **Only phase-blocked windows are scored.** The task is continuing, so an episode ends only at the
   400-step truncation and every copy shares one episode clock; one iteration sees 128 of those 400
   steps, so its extrinsic reward is a sample of one window of the episode and can read zero for
   every copy while copies are solving. A record here covers 200 iterations, which is 8 whole turns
   of the 25-iteration clock, so its reward sum sees every part of the episode equally. A record
   whose `phase_blocked` flag is false is never scored.
2. **Only completed chunks feed a table number, and a configuration is scored only when ALL of its
   chunks are complete.** A chunk still running has windows on disk and they are fine for a
   truncated curve, never for a table; and half a configuration's copies are not the configuration.
3. **The chunks of an arm must partition its copies.** The union of their copy-index ranges is
   checked against 0..127 and a gap or an overlap is an error, not a silently smaller N.

Run:
  PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python code/aggregate.py
"""
import json
import math
import re
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent.parent

# the open cells of the large map; coverage is already reported as a fraction of them, and this is
# here so a reader of a coverage number can see what the denominator was
OPEN_CELLS = 46
# the copies one configuration of this run holds, across all of its chunks
COPIES_PER_CONFIGURATION = 128
# a window record's own iteration marker, pulled out without parsing the whole line
LAST_ITERATION = re.compile(rb'"last_iteration": (\d+)')


def live_lines(shard: Path) -> dict:
    """Which line of a shard is the live copy of each record, by a first cheap scan.

    A chunk that failed and was re-run appends its windows again from the start, so one shard can
    hold the same window twice; the later line is the live one. This scan finds the byte-cheap
    answer — a regular expression over the raw bytes, no JSON parsing — so the second pass can parse
    only the lines that matter.

    before: a shard whose lines 3..500 are a first attempt ending at iteration 100,000 and whose
            lines 501.. are the re-run from iteration 200 onward;
    after:  {"start": 1, "complete": 9999, "windows": {200: 501, 400: 502, ...}} — one line index
            per window, the later one wherever a window appears twice.
    """
    start, complete, windows = None, None, {}
    with open(shard, "rb") as handle:
        for index, line in enumerate(handle):
            if b'"record": "episode_window"' in line:
                match = LAST_ITERATION.search(line)
                windows[int(match.group(1))] = index
            elif b'"record": "unit_start"' in line:
                start = index
            elif b'"record": "unit_complete"' in line:
                complete = index
    return {"start": start, "complete": complete, "windows": windows}


def read_selected(shard: Path, indices: set):
    """Yield (line index, parsed record) for the given line indices of a shard, in file order."""
    with open(shard) as handle:
        for index, line in enumerate(handle):
            if index in indices:
                yield index, json.loads(line)


def cell_copies(start: dict) -> dict:
    """Which copy positions inside a chunk belong to each (learning rate, weight) cell.

    before: cell_settings [(1e-3, 1e-5), (1e-3, 1e-4), ...], copy_cell [0]*4 + [1]*4 + ...
    after:  {(0.001, 1e-05): [0, 1, 2, 3], (0.001, 0.0001): [4, 5, 6, 7], ...} — positions in the
            chunk's own per-copy lists, not seed indices
    """
    settings = [tuple(setting) for setting in start["cell_settings"]]
    members = {setting: [] for setting in settings}
    for position, cell in enumerate(start["copy_cell"]):
        members[settings[cell]].append(position)
    return members


def chunk_summary(shard: Path) -> dict:
    """One chunk's identity and its per-copy totals, accumulated in one streaming pass.

    before: data/unit-1_..._chunk-7-of-32_....jsonl, 9,766 window records of 132 rewards each;
    after:  {"start": {...}, "complete": {...} or None, "windows_scored": 9766,
             "episodes_per_copy": 2500096.0, "reward_sum_per_copy": [132 floats],
             "last_window": {"reward_per_copy": [...], "episodes": 256.0,
                             "coverage_per_copy": [...]}}
    """
    live = live_lines(shard)
    if live["start"] is None:
        return None
    wanted = set(live["windows"].values()) | {live["start"]}
    if live["complete"] is not None:
        wanted.add(live["complete"])
    summary = {"start": None, "complete": None, "windows_scored": 0, "episodes_per_copy": 0.0,
               "reward_sum_per_copy": None, "last_window": None, "windows_seen": 0}
    for index, record in read_selected(shard, wanted):
        kind = record["record"]
        if kind == "unit_start":
            summary["start"] = record
            continue
        if kind == "unit_complete":
            summary["complete"] = record
            continue
        summary["windows_seen"] += 1
        if not record["phase_blocked"]:
            continue
        rewards = record["reward_ext_sum_per_copy"]
        if summary["reward_sum_per_copy"] is None:
            summary["reward_sum_per_copy"] = [0.0] * len(rewards)
        for position, value in enumerate(rewards):
            summary["reward_sum_per_copy"][position] += value
        summary["episodes_per_copy"] += record["episodes_per_copy_in_window"]
        summary["windows_scored"] += 1
        summary["last_window"] = {"reward_per_copy": rewards,
                                  "episodes": record["episodes_per_copy_in_window"],
                                  "coverage_per_copy": record.get("coverage_per_copy", [])}
    return summary


def chunk_summaries(run_dir: Path = RUN_DIR) -> dict:
    """Every chunk's streamed summary, keyed by chunk id."""
    summaries = {}
    for shard in sorted((run_dir / "data").glob("*.jsonl")):
        summary = chunk_summary(shard)
        if summary is not None and summary["start"] is not None:
            summaries[summary["start"]["unit_id"]] = summary
    return summaries


def group_by_arm(summaries: dict) -> dict:
    """The chunk summaries of each arm, in copy-index order.

    before: forty chunk summaries, thirty-two of them chunks of the distillation arm;
    after:  {"rnd_next_state": [chunk with copy indices 0-3, chunk with 4-7, ...], ...}
    """
    groups = {}
    for chunk_id, summary in summaries.items():
        groups.setdefault(summary["start"]["bonus"], []).append(dict(summary, chunk_id=chunk_id))
    for group in groups.values():
        group.sort(key=lambda chunk: chunk["start"]["copy_seed_index_first"])
    return groups


def check_partition(arm: str, group: list, copies: int = None) -> None:
    """Fail unless the group's chunks cover copy indices 0..copies-1 exactly once between them.

    `copies` defaults to the module's configuration size, read at CALL time rather than baked into
    the signature, so a test can point it at a toy unit's smaller count.
    """
    copies = COPIES_PER_CONFIGURATION if copies is None else copies
    covered = []
    for chunk in group:
        start = chunk["start"]
        covered += list(range(start["copy_seed_index_first"], start["copy_seed_index_last"] + 1))
    if sorted(covered) != list(range(copies)):
        raise SystemExit(
            f"the {len(group)} chunks of {arm} do not partition {copies} copies per cell: "
            f"{len(covered)} indices covered, {len(set(covered))} of them distinct")


def mean_and_standard_error(values: list) -> tuple:
    """The mean of a list and the standard error of that mean; a single value has no error."""
    n = len(values)
    if n == 0:
        return None, None
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    return mean, math.sqrt(variance / n)


def cell_table(run_dir: Path = RUN_DIR, completed_only: bool = True) -> list:
    """One row per (arm, learning rate, intrinsic weight), pooled over that arm's chunks.

    `completed_only` keeps the table rule: an arm any of whose chunks has not finished contributes
    to curves but never to a table number.

    before: thirty-two chunk summaries of the distillation arm, each holding per-copy totals over
            33 cells x 4 copies;
    after:  33 rows, each pooling 32 x 4 = 128 copies of one configuration
    """
    rows = []
    for arm, group in sorted(group_by_arm(chunk_summaries(run_dir)).items()):
        if completed_only:
            if any(chunk["complete"] is None for chunk in group):
                continue
            check_partition(arm, group)
        pooled, episodes, windows_scored = {}, None, None
        for chunk in group:
            if chunk["reward_sum_per_copy"] is None:
                continue
            last = chunk["last_window"]
            # per copy of this chunk: its mean episode return over the whole run, its mean over the
            # final scored window, whether it ever collected reward, and its maze coverage
            whole = [value / chunk["episodes_per_copy"] for value in chunk["reward_sum_per_copy"]]
            final = [value / last["episodes"] for value in last["reward_per_copy"]]
            reached = [1.0 if value > 0 else 0.0 for value in chunk["reward_sum_per_copy"]]
            coverage = [100.0 * value for value in last["coverage_per_copy"]]
            values = {"whole_run_reward": whole, "last_window_reward": final,
                      "reached_goal": reached, "coverage_percent": coverage}
            for setting, positions in cell_copies(chunk["start"]).items():
                entry = pooled.setdefault(setting, {name: [] for name in values})
                for name, series in values.items():
                    entry[name] += [series[position] for position in positions] if series else []
            episodes = chunk["episodes_per_copy"]
            windows_scored = chunk["windows_scored"]
        for (rate, weight), entry in sorted(pooled.items()):
            row = {"bonus": arm, "learning_rate": rate, "intrinsic_weight": weight,
                   "chunk_ids": [chunk["chunk_id"] for chunk in group],
                   "chunks": len(group), "copies": len(entry["whole_run_reward"]),
                   "episodes_per_copy_scored": episodes, "windows_scored": windows_scored,
                   # steps to goal is not recorded by this trainer: the goal step inside an episode
                   # is never read back to the host, so the column has no number in this run
                   "steps_to_goal": None}
            for name in ("whole_run_reward", "last_window_reward", "coverage_percent"):
                mean, error = mean_and_standard_error(entry[name])
                row[name], row[f"{name}_standard_error"] = mean, error
            row["success_rate"] = (sum(entry["reached_goal"]) / len(entry["reached_goal"])
                                   if entry["reached_goal"] else None)
            rows.append(row)
    return rows


def best_per_arm(rows: list, key: str = "whole_run_reward") -> dict:
    """The highest-scoring configuration of each algorithm arm, by the column the table sorts on."""
    best = {}
    for row in rows:
        if row[key] is None:
            continue
        current = best.get(row["bonus"])
        if current is None or row[key] > current[key]:
            best[row["bonus"]] = row
    return best


def curve_of(bonus: str, rate: float, weight: float, run_dir: Path = RUN_DIR) -> dict:
    """The learning curve of one configuration, pooled over its arm's chunks window by window.

    Streams the arm's shards once and keeps three running numbers per window — how many chunks have
    reported it, the sum of this configuration's per-copy episode returns, and the sum of their
    squares — so the mean and the standard error over all 128 copies come out exactly without any
    window's per-copy values ever being held.

    A window that some chunk has not reached is dropped, so a curve drawn while the run is going
    stops at the slowest chunk.

    before: eight chunks, each with 9,766 windows of 528 per-copy reward sums;
    after:  9,766 points, each the mean over the 128 copies of this configuration with its standard
            error over them
    """
    setting = (rate, weight)
    per_window = {}
    chunks = 0
    for shard in sorted((run_dir / "data").glob("*.jsonl")):
        live = live_lines(shard)
        if live["start"] is None:
            continue
        _, start = next(read_selected(shard, {live["start"]}))
        if start["bonus"] != bonus:
            continue
        positions = cell_copies(start).get(setting)
        if positions is None:
            continue
        chunks += 1
        for _, record in read_selected(shard, set(live["windows"].values())):
            episodes = record["episodes_per_copy_in_window"]
            values = [record["reward_ext_sum_per_copy"][position] / episodes
                      for position in positions]
            entry = per_window.setdefault(record["last_iteration"],
                                          {"n": 0, "sum": 0.0, "square_sum": 0.0, "chunks": 0,
                                           "env_steps_per_copy": record["env_steps_per_copy"],
                                           "phase_blocked": True})
            entry["n"] += len(values)
            entry["sum"] += sum(values)
            entry["square_sum"] += sum(value * value for value in values)
            entry["chunks"] += 1
            entry["phase_blocked"] = entry["phase_blocked"] and record["phase_blocked"]
    steps, means, errors, blocked = [], [], [], []
    for iteration in sorted(per_window):
        entry = per_window[iteration]
        if entry["chunks"] != chunks:
            continue
        n = entry["n"]
        mean = entry["sum"] / n
        # the variance over the copies, from the running sums: (sum of squares - n x mean^2)/(n-1),
        # clamped at zero because rounding can make an exactly-zero variance read as -1e-18
        variance = max(0.0, (entry["square_sum"] - n * mean * mean) / (n - 1)) if n > 1 else 0.0
        steps.append(entry["env_steps_per_copy"])
        means.append(mean)
        errors.append(math.sqrt(variance / n))
        blocked.append(entry["phase_blocked"])
    return {"env_steps_per_copy": steps, "mean_episode_return": means,
            "standard_error": errors, "phase_blocked": blocked}


def write_metrics(run_dir: Path) -> int:
    """Copy every live window record of every shard into the run-level metrics.jsonl, streaming.

    The file is the same one every run of this platform writes — one line per (chunk, window), the
    later line kept where a re-attempt wrote a window twice. It is written by copying lines rather
    than by parsing and re-serialising them, because at this run's size that is the difference
    between a few megabytes of memory and several gigabytes.
    """
    written = 0
    with open(run_dir / "metrics.jsonl", "w") as out:
        for shard in sorted((run_dir / "data").glob("*.jsonl")):
            live = set(live_lines(shard)["windows"].values())
            with open(shard) as handle:
                for index, line in enumerate(handle):
                    if index in live:
                        out.write(line)
                        written += 1
    return written


def run_summary(run_dir: Path, summaries: dict) -> dict:
    """The run-level headline numbers, from the chunks' own completion records.

    Throughput is reported both ways the project's rule asks for — the aggregate rate over all
    copies and the rate one copy gets — and the per-copy rate is restated as hours per million
    steps per copy, which is the unit a run is planned in.
    """
    completions = [chunk["complete"] for chunk in summaries.values()
                   if chunk["complete"] is not None]
    copies = sum(chunk["start"]["copies"] for chunk in summaries.values()
                 if chunk["complete"] is not None)
    steady = (sum(r["seconds_per_iteration_steady"] for r in completions) / len(completions)
              if completions else None)
    env_steps = sum(r["env_steps"] for r in completions)
    iterations = sum(r["iterations"] for r in completions)
    steps_per_iteration = (env_steps / iterations) if iterations else 0.0
    total_rate = (steps_per_iteration / steady) if steady else 0.0
    per_copy_rate = (total_rate / copies) if copies else 0.0
    hardware = [json.loads(path.read_text())
                for path in sorted((run_dir / "slurm" / "jobs").glob("*/hardware.json"))]
    return {
        "run_id": run_dir.name,
        "units_complete": len(completions),
        "units_seen": len(summaries),
        "units": sorted(summaries),
        "throughput": {
            "copies": copies,
            "seconds_total_wall_clock": max((r["seconds_total"] for r in completions), default=0.0),
            "seconds_per_iteration_steady": steady,
            "total_env_steps": env_steps,
            "total_env_steps_per_second_steady": total_rate,
            "env_steps_per_second_per_copy_steady": per_copy_rate,
            "hours_per_million_steps_per_copy_steady": (
                1e6 / (3600 * per_copy_rate) if per_copy_rate else None),
        },
        "hardware": hardware,
    }


def aggregate(run_dir: Path = RUN_DIR) -> dict:
    """Write metrics.jsonl and summary.json, the summary carrying this run's own scores."""
    summaries = chunk_summaries(run_dir)
    summary = run_summary(run_dir, summaries)
    summary["records"] = write_metrics(run_dir)
    rows = cell_table(run_dir)
    summary["configurations_scored"] = len(rows)
    summary["cells"] = rows
    summary["best_per_arm"] = best_per_arm(rows)
    summary["scoring"] = {
        "windows": "phase-blocked episode windows only (200 iterations = 8 turns of the "
                   "25-iteration episode clock = 256 episodes per copy)",
        "open_cells": OPEN_CELLS,
        "copies_per_configuration": COPIES_PER_CONFIGURATION,
        "pooling": "a configuration's copies are selected inside each chunk by that chunk's own "
                   "copy_cell list and then concatenated across the arm's chunks, per copy, not "
                   "averaged over chunks; the chunks' copy-index ranges are checked to partition "
                   "the configuration's copies",
        "steps_to_goal": "not recorded in this run",
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    """Aggregate this run and print the best configuration of each arm."""
    summary = aggregate()
    print(f"{summary['units_complete']} chunks complete, {summary['records']} window records, "
          f"{summary['configurations_scored']} configurations scored")
    for bonus, row in sorted(summary["best_per_arm"].items()):
        print(f"  {bonus}: learning rate {row['learning_rate']:g}, intrinsic weight "
              f"{row['intrinsic_weight']:g}, {row['chunks']} chunk(s), N={row['copies']} -> "
              f"whole-run reward {row['whole_run_reward']:.4f} +- "
              f"{row['whole_run_reward_standard_error']:.4f}, "
              f"coverage {row['coverage_percent']:.2f}%, success {row['success_rate']:.3f}")


if __name__ == "__main__":
    main()
