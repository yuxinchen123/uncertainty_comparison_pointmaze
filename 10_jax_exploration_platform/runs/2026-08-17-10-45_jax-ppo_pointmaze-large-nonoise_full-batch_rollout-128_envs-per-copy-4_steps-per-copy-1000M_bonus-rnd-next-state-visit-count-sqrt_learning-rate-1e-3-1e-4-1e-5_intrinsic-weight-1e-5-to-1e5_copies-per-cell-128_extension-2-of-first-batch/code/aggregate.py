"""Roll this run's shards up into run-level metrics, and score every configuration of every arm.

This is the run's own aggregation module. It reads only `data/*.jsonl` and `slurm/jobs/*/*.json`,
and everything downstream — the 20-minute status table, the development document's tables, the
training-curve figure — imports the functions here rather than re-deriving a number from the shards.

A shard here is one CHUNK of one arm, and a chunk carries ALL 33 configurations of that arm at a
slice of their copies. So a configuration's numbers are pooled twice over: within a shard, the
copies belonging to that configuration are selected by the shard's own `copy_cell` list; across
shards, the chunks' per-copy lists are concatenated. Pooling is exact rather than an average of
averages, because every per-copy quantity is carried per copy all the way to the end.

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
import sys
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent.parent
PLATFORM_ROOT = RUN_DIR.parent.parent
sys.path.insert(0, str(PLATFORM_ROOT / "scripts"))

# the open cells of the large map; coverage is already reported as a fraction of them, and this is
# here so a reader of a coverage number can see what the denominator was
OPEN_CELLS = 46
# the copies one configuration of this run holds, across all of its chunks
COPIES_PER_CONFIGURATION = 128


def read_chunks(run_dir: Path = RUN_DIR) -> dict:
    """Every chunk's records, grouped: its identity card, its windows, and its completion record.

    before: data/unit-1_..._chunk-1-of-8_....jsonl with job / unit_start / 9,766 episode_window /
            unit_complete lines;
    after:  {"unit-1_..._chunk-1-of-8_...": {"start": {...}, "windows": [...], "complete": {...}}}
    A window recorded twice (a re-attempt of the same chunk) keeps the later line.
    """
    chunks = {}
    for shard in sorted((run_dir / "data").glob("*.jsonl")):
        for line in shard.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            kind = record.get("record")
            if kind not in ("unit_start", "episode_window", "unit_complete"):
                continue
            chunk = chunks.setdefault(record["unit_id"],
                                      {"start": None, "windows": {}, "complete": None})
            if kind == "unit_start":
                chunk["start"] = record
            elif kind == "unit_complete":
                chunk["complete"] = record
            else:
                chunk["windows"][record["last_iteration"]] = record
    for chunk in chunks.values():
        chunk["windows"] = [chunk["windows"][key] for key in sorted(chunk["windows"])]
    return chunks


def cell_copies(chunk: dict) -> dict:
    """Which copy positions inside this chunk belong to each (learning rate, weight) cell.

    before: cell_settings [(1e-3, 1e-5), (1e-3, 1e-4), ...], copy_cell [0]*16 + [1]*16 + ...
    after:  {(0.001, 1e-05): [0..15], (0.001, 0.0001): [16..31], ...} — positions in this chunk's
            own per-copy lists, not seed indices
    """
    settings = [tuple(setting) for setting in chunk["start"]["cell_settings"]]
    members = {setting: [] for setting in settings}
    for position, cell in enumerate(chunk["start"]["copy_cell"]):
        members[settings[cell]].append(position)
    return members


def group_by_arm(chunks: dict) -> dict:
    """The chunks of each arm, in copy-index order.

    before: sixteen shards, eight of them chunks of the distillation arm;
    after:  {"rnd_next_state": [chunk with copy indices 0-15, chunk with 16-31, ...], ...}
    """
    groups = {}
    for chunk_id, chunk in chunks.items():
        if chunk["start"] is None:
            continue
        groups.setdefault(chunk["start"]["bonus"], []).append(dict(chunk, chunk_id=chunk_id))
    for group in groups.values():
        group.sort(key=lambda chunk: chunk["start"].get("copy_seed_index_first", 0))
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
        first = start["copy_seed_index_first"]
        last = start["copy_seed_index_last"]
        covered += list(range(first, last + 1))
    if sorted(covered) != list(range(copies)):
        raise SystemExit(
            f"the {len(group)} chunks of {arm} do not partition {copies} copies per cell: "
            f"{len(covered)} indices covered, {len(set(covered))} of them distinct")


def scored_windows(chunk: dict) -> list:
    """The chunk's windows that may feed a score: the phase-blocked ones, in iteration order."""
    return [window for window in chunk["windows"] if window["phase_blocked"]]


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


def per_copy_scores(chunk: dict) -> dict:
    """The four per-copy numbers this run scores by, as lists over the chunk's copies.

    whole_run_reward   mean episode return over every phase-blocked window of the run
    last_window_reward mean episode return over the final phase-blocked window
    reached_goal       1.0 if the copy ever collected extrinsic reward, else 0.0
    coverage_percent   distinct open maze cells entered, as a percentage of the 46 open cells

    before: 9,766 windows, each holding a per-copy reward SUM over 256 episodes;
    after:  whole_run_reward[k] = sum of copy k's window sums / (9,766 x 256) episodes.
    """
    windows = scored_windows(chunk)
    if not windows:
        return {}
    copies = len(windows[0]["reward_ext_sum_per_copy"])
    total_reward = [0.0] * copies
    total_episodes = 0.0
    for window in windows:
        total_episodes += window["episodes_per_copy_in_window"]
        for index, value in enumerate(window["reward_ext_sum_per_copy"]):
            total_reward[index] += value
    last = windows[-1]
    return {
        "whole_run_reward": [value / total_episodes for value in total_reward],
        "last_window_reward": [value / last["episodes_per_copy_in_window"]
                               for value in last["reward_ext_sum_per_copy"]],
        "reached_goal": [1.0 if value > 0 else 0.0 for value in total_reward],
        "coverage_percent": [100.0 * value for value in last.get("coverage_per_copy", [])],
        "episodes_per_copy_scored": total_episodes,
        "windows_scored": len(windows),
        "last_window_episodes": last["episodes_per_copy_in_window"],
    }


def cell_table(run_dir: Path = RUN_DIR, completed_only: bool = True) -> list:
    """One row per (arm, learning rate, intrinsic weight), pooled over that arm's chunks.

    `completed_only` keeps the table rule: an arm any of whose chunks has not finished contributes
    to curves but never to a table number.

    before: eight chunk shards of the distillation arm, each with 33 cells x 16 copies;
    after:  33 rows, each pooling 8 x 16 = 128 copies of one configuration
    """
    rows = []
    for arm, group in sorted(group_by_arm(read_chunks(run_dir)).items()):
        if completed_only:
            if any(chunk["complete"] is None for chunk in group):
                continue
            check_partition(arm, group)
        pooled, episodes, windows_scored = {}, None, None
        for chunk in group:
            scores = per_copy_scores(chunk)
            if not scores:
                continue
            for setting, positions in cell_copies(chunk).items():
                entry = pooled.setdefault(setting, {name: [] for name in
                                                    ("whole_run_reward", "last_window_reward",
                                                     "reached_goal", "coverage_percent")})
                for name in entry:
                    entry[name] += [scores[name][position] for position in positions] \
                        if scores[name] else []
            episodes = scores["episodes_per_copy_scored"]
            windows_scored = scores["windows_scored"]
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


def configuration_curve(group: list, setting: tuple) -> dict:
    """The learning curve of one configuration, pooled over its chunks window by window.

    Every chunk of an arm runs the same iterations with the same window size, so a window is
    identified by the iteration it ends at and the chunks' per-copy values for that window are
    concatenated before the mean and the standard error are taken. A window that some chunk has not
    reached yet is dropped, so a curve drawn while the run is going stops at the slowest chunk.

    before: eight chunks, each with 9,766 windows of 528 per-copy reward sums;
    after:  9,766 points, each the mean over the 128 copies of this configuration
    """
    per_window = {}
    for chunk in group:
        positions = cell_copies(chunk)[setting]
        for window in chunk["windows"]:
            episodes = window["episodes_per_copy_in_window"]
            entry = per_window.setdefault(window["last_iteration"],
                                          {"values": [], "chunks": 0,
                                           "env_steps_per_copy": window["env_steps_per_copy"],
                                           "phase_blocked": True})
            entry["values"] += [window["reward_ext_sum_per_copy"][position] / episodes
                                for position in positions]
            entry["chunks"] += 1
            entry["phase_blocked"] = entry["phase_blocked"] and window["phase_blocked"]
    steps, means, errors, blocked = [], [], [], []
    for iteration in sorted(per_window):
        entry = per_window[iteration]
        if entry["chunks"] != len(group):
            continue
        mean, error = mean_and_standard_error(entry["values"])
        steps.append(entry["env_steps_per_copy"])
        means.append(mean)
        errors.append(error)
        blocked.append(entry["phase_blocked"])
    return {"env_steps_per_copy": steps, "mean_episode_return": means,
            "standard_error": errors, "phase_blocked": blocked}


def curve_of(bonus: str, rate: float, weight: float, run_dir: Path = RUN_DIR) -> dict:
    """The pooled curve of one configuration, named by its arm and its two swept values."""
    return configuration_curve(group_by_arm(read_chunks(run_dir))[bonus], (rate, weight))


def aggregate(run_dir: Path = RUN_DIR) -> dict:
    """Write metrics.jsonl and summary.json, the summary carrying this run's own scores."""
    from aggregate_run import aggregate as generic_aggregate
    summary = generic_aggregate(run_dir)
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
