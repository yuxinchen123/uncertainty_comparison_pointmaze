"""Learning check on the corrected game: original reference against the final configuration, same seeds.

Also reports rooms visited (union over updates, per seed) and the life-loss-bounded episode counts.
"""
import json
import statistics
import sys
from pathlib import Path

RUN = Path(sys.argv[1])
FIELDS = ["intrinsic_reward_mean", "predictor_loss", "value_loss", "policy_loss", "entropy",
          "approximate_kl", "episode_count", "episode_return_sum", "reward_ext_sum"]


def series(folder):
    """Per-copy list of per-update metric dicts (rooms_visited kept as a list of room numbers)."""
    rows = [json.loads(line) for line in (folder / "run" / "metrics.jsonl").read_text().splitlines()]
    copies = len(rows[0]["intrinsic_reward_mean"])
    return [[{**{field: row[field][copy] for field in FIELDS},
              "rooms": [room for room, seen in enumerate(row.get("rooms_visited", [[]] * copies)[copy]) if seen]}
             for row in rows] for copy in range(copies)]


def gather(pattern):
    runs = []
    for folder in sorted(RUN.glob(f"data/{pattern}")):
        if (folder / "run" / "metrics.jsonl").exists():
            runs.extend(series(folder))
    return runs


arms = {"A4500 original reference (4 seeds)": gather("a4500-jaguar03_learning-check_original-reference_seed-*"),
        "A4500 final configuration (4 seeds)": gather("a4500-jaguar03_learning-check_final-configuration_seed-*"),
        "H100 final configuration (4 copies, seeds 1-4)": gather("h100-serval02-card-1_learning-check_final-configuration_seeds-1-to-4")}
common = min(len(run) for runs in arms.values() if runs for run in runs)
windows = [(1, 1), (2, 5), (1, 20), (common - 49, common)]
report = {"common_updates": common, "windows": {}}
for first, last in windows:
    key = f"updates {first}-{last}"
    report["windows"][key] = {}
    for arm, runs in arms.items():
        if not runs:
            continue
        values = {}
        for field in FIELDS:
            per_seed = [statistics.mean(run[i][field] for i in range(first - 1, min(last, len(run)))) for run in runs]
            values[field] = {"mean": statistics.mean(per_seed),
                             "seed_minimum": min(per_seed), "seed_maximum": max(per_seed)}
        report["windows"][key][arm] = values
report["rooms_visited_per_seed"] = {arm: [sorted({room for update in run for room in update["rooms"]}) for run in runs]
                                    for arm, runs in arms.items() if runs}
(RUN / "analysis" / "learning_check.json").write_text(json.dumps(report, indent=2) + "\n")
for key, table in report["windows"].items():
    print(f"=== {key} (seed mean [minimum, maximum])")
    print("field".ljust(22) + "".join(arm[:30].ljust(34) for arm in table))
    for field in FIELDS:
        print(field.ljust(22) + "".join(
            f"{v[field]['mean']:.4g} [{v[field]['seed_minimum']:.3g},{v[field]['seed_maximum']:.3g}]".ljust(34)
            for v in table.values()))
print("rooms visited per seed (union over updates):")
for arm, rooms in report["rooms_visited_per_seed"].items():
    print(f"  {arm}: {rooms}")
