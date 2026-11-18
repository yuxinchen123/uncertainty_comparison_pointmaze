"""Rebuild RUN/early_signals.md: what the runs of a pixel sweep show so far, from their record files only.

    python report.py --run-dir RUN [--output FILE]
    python report.py --run-dir RUN --mark-read all            (or one section's title)
    python report.py --run-dir RUN --fold-archive ARCHIVED_DATA_FOLDER [--match TEXT]

The report reads, for every unit, the two record files train_final.py keeps beside its checkpoint
(game_first_times.json and game_windows.jsonl, written by scripts/atari_pixel_campaign/game_records.py)
and progress.json; it reads no metrics file and no checkpoint, starts nothing and stops nothing. A
unit a remote server holds is read from the copy of its record files that remote_fleet.py brings to
RUN/slurm/remote_live/ at every tick.

Runs of the archived baseline (trained before the game events were recorded) are shown beside the new
runs once their record files exist in RUN/archived_baseline/<unit>/: --fold-archive makes them from
the archived metrics files with the trainer's own fold. That reads every archived metric line (about
340 MB for the 12 archived seeds), so it runs inside a Slurm job, not on a login node.

Sections: fleet status and speed by card type; every configuration at one common mark; the mean score
over the run at every mark; medians of the first-time steps; keys, exits and lives lost; score
levels; standing checks. A contents table at the top says when each section first appeared, when it
last changed, and whether it was read (--mark-read; rule auto-research-report-contents.md); its
record is RUN/early_signals_manifest.json. Times are Pacific. Standard library only.
"""
import argparse
from datetime import datetime
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import shutil
import statistics
from zoneinfo import ZoneInfo

from campaign_imports import PLATFORM, common, load_file, unit_queue
import pixel_queue

ARCHIVE_FOLDER = "archived_baseline"
NOT_UNITS = ("packed",)        # queue folders that hold records of singles a pack replaced, not units
# the marks at which runs are compared, as shares of the stage's whole windows (200 windows: 1, 2, 5, ..., 200)
MARK_FRACTIONS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
EVENTS = (("key_pickup", "key"), ("door_opened", "door"), ("left_start_room", "exit from<br>the start<br>room"),
          ("enemy_killed", "enemy<br>kill"), ("sword_pickup", "sword"), ("torch_pickup", "torch"),
          ("gem_pickup", "gem"), ("amulet_pickup", "amulet"))
LIFE_CAUSES = (("life_lost_by_fall", "fall"), ("life_lost_by_enemy", "enemy"), ("life_lost_by_laser", "laser"),
               ("life_lost_by_other_cause", "other cause"))
LOCKED_EXIT = "left_start_room_with_both_doors_locked"
METHOD_WORDS = {"rnd": "RND", "fast-last-layer": "fast last layer"}
SETTINGS = ("step_size", "sharing_group", "intrinsic_weight")
DASH = "—"


def number(value):
    """Write a setting's value in the shortest plain form."""
    # before: 0.03 ; after: "0.03"      before: 1.0 ; after: "1"      before: 128 ; after: "128"
    return f"{value:g}"


def millions(steps):
    """Write a number of steps in millions with two decimals."""
    return f"{steps / 1e6:.2f}"


def read_records(folder):
    """Read the two record files of one run folder; None while the trainer has not written the first one."""
    first_path = Path(folder) / "game_first_times.json"
    if not first_path.exists():
        return None
    windows_path = Path(folder) / "game_windows.jsonl"
    # only whole lines count: the text after the last newline is a line still being written
    lines = windows_path.read_text().split("\n")[:-1] if windows_path.exists() else []
    return {"first_times": json.loads(first_path.read_text()),
            "windows": [json.loads(line) for line in lines if line.strip()]}


def copy_records(records, copy):
    """Take one copy's share of a unit's records: its first times, and its values in every finished window.

    before: a window line {"window": 3, "step": 2998272, "games": [4, 9], "score_sum": [400.0, 0.0],
            "mean_score": [100.0, 0.0], "games_with_key": [1, 0], "games_left_start_room": [0, 0],
            "event_counts": {"life_lost_by_fall": [7, 30], ...}}, copy 0
    after:  {"step": 2998272, "games": 4, "score_sum": 400.0, "mean_score": 100.0, "games_with_key": 1,
             "games_left_start_room": 0, "event_counts": {"life_lost_by_fall": 7, ...}}
    Lines without event fields (the archived baseline) keep None for the three event values.
    """
    def one(values):
        """This copy's entry of a per-copy list; None stays None."""
        return None if values is None else values[copy]

    windows = [{"step": line["step"], "games": line["games"][copy], "score_sum": line["score_sum"][copy],
                "mean_score": line["mean_score"][copy], "games_with_key": one(line["games_with_key"]),
                "games_left_start_room": one(line["games_left_start_room"]),
                "event_counts": (None if line["event_counts"] is None else
                                 {name: counts[copy] for name, counts in line["event_counts"].items()})}
               for line in records["windows"]]
    copies = records["first_times"]["copies"]
    return {"first": copies[copy] if copy < len(copies) else None, "windows": windows,
            "has_event_fields": records["first_times"]["has_event_fields"] is True,
            "updates_folded": records["first_times"]["updates_folded"]}


def unit_cards(unit):
    """The card types a unit's attempts ran on, in order, each named once per stretch."""
    # before: history on "NVIDIA RTX A4500", "NVIDIA RTX A4500", "NVIDIA GeForce RTX 5080", now running on that card
    # after:  ["RTX A4500", "GeForce RTX 5080"]
    cards = [attempt.get("card") for attempt in unit.get("history", [])] + [unit.get("card")]
    names = []
    for card in cards:
        name = str(card).replace("NVIDIA ", "").replace("_", " ") if card else None
        if name and (not names or names[-1] != name):
            names.append(name)
    return names


def unit_speeds(unit, progress, state):
    """The speed readings of one unit: one per attempt that trained, and one for the attempt that runs now."""
    # after: [{"card": "RTX A4500", "copies": 1, "seconds_per_update": 1.41, "running": False}, ...]
    readings, history = [], unit.get("history", [])
    for index, attempt in enumerate(history):
        seconds = attempt.get("seconds_per_update")
        # the attempt that completed a unit keeps its speed in the unit's result
        if seconds is None and attempt.get("outcome") == "complete" and index == len(history) - 1:
            seconds = unit.get("result", {}).get("seconds_per_update")
        if seconds and attempt.get("card"):
            readings.append({"card": attempt["card"], "seconds_per_update": seconds, "running": False})
    if state == "running" and progress["seconds_per_update"] and unit.get("card"):
        readings.append({"card": unit["card"], "seconds_per_update": progress["seconds_per_update"], "running": True})
    return [dict(reading, card=reading["card"].replace("NVIDIA ", "").replace("_", " "), copies=unit["copies"])
            for reading in readings]


def collect(run_dir):
    """Read every unit of a run folder and the archived runs beside it; returns units, runs and speed readings.

    A run is one policy. after (one run):
        {"method": "fast-last-layer", "settings": {"step_size": 0.03, "sharing_group": 16, "intrinsic_weight": 1.0},
         "seed": 13, "archived": False, "unit": "<unit name>", "state": "running", "target_updates": 12207,
         "update": 4120, "started": True, "first": {the copy's first times} or None, "windows": [...],
         "has_event_fields": True}
    """
    run_dir = Path(run_dir)
    units, runs, speeds = [], [], []
    states = [state for state in unit_queue.counts(run_dir) if state not in NOT_UNITS] if (run_dir / "queue").exists() else []
    for state in states:
        for unit in unit_queue.read_units(run_dir, state):
            specification = common.read_json(run_dir / unit["specification"])
            folder = pixel_queue.records_folder(run_dir, state, unit["name"])
            records, progress = read_records(folder), pixel_queue.read_progress(folder)
            update = unit["target_updates"] if state == "done" else progress["update"]
            units.append({"name": unit["name"], "state": state, "unit": unit, "copies": unit["copies"],
                          "update": update, "cards": unit_cards(unit), "has_records": records is not None,
                          "seconds_per_update": progress["seconds_per_update"]})
            speeds += unit_speeds(unit, progress, state)
            for copy, run in enumerate(specification["runs"]):
                share = copy_records(records, copy) if records else {"first": None, "windows": [], "has_event_fields": False}
                runs.append({"method": run["method"], "settings": {word: run[word] for word in SETTINGS if word in run},
                             "seed": run["seed"], "archived": False, "unit": unit["name"], "state": state,
                             "target_updates": unit["target_updates"], "update": update,
                             "started": records is not None or state != "pending", **share})
    # the archived runs: record files folded from their metrics, and the run.json that says what they were
    for folder in sorted((run_dir / ARCHIVE_FOLDER).glob("*")) if (run_dir / ARCHIVE_FOLDER).exists() else []:
        records = read_records(folder)
        if records is None:
            continue
        described = common.read_json(folder / "run.json")
        learner = described["learner"]
        for copy, seed in enumerate(described["copy_seeds"]):
            share = copy_records(records, copy)
            fast = learner.get("fast_last_layer_step", 0) > 0
            settings = {"intrinsic_weight": float(learner["intrinsic_coefficient"])}
            if fast:
                settings.update(step_size=learner["fast_last_layer_step"],
                                sharing_group=learner["fast_last_layer_sharing_group"])
            runs.append({"method": "fast-last-layer" if fast else "rnd", "settings": settings, "seed": seed,
                         "archived": True, "unit": folder.name, "state": "archived",
                         "target_updates": learner["total_timesteps"] // pixel_queue.STEPS_PER_UPDATE,
                         "update": share["updates_folded"], "started": True, **share})
    return {"units": units, "runs": runs, "speeds": speeds}


def configuration_key(run):
    """The place of a run's configuration in every table: RND first, then the sharing groups, weights and step sizes."""
    # after: (0, False, 0, 1.0, 0.0) for RND at weight 1; (0, True, 0, 1.0, 0.0) for the archived baseline;
    #        (1, False, -16, 1.0, 0.03) for the fast last layer with group 16, weight 1, step size 0.03
    settings = run["settings"]
    return (0 if run["method"] == "rnd" else 1, run["method"] if run["method"] != "rnd" else "", run["archived"],
            -settings.get("sharing_group", 0), settings["intrinsic_weight"], settings.get("step_size", 0.0))


def configurations(runs):
    """Group the runs by configuration, in table order; returns [{"key", "method", "settings", "archived", "runs"}]."""
    groups = {}
    for run in runs:
        group = groups.setdefault(configuration_key(run), {
            "method": run["method"], "settings": run["settings"], "archived": run["archived"], "runs": []})
        group["runs"].append(run)
    return [dict(groups[key], key=key) for key in sorted(groups)]


def method_words(method):
    """The method's name in words."""
    return METHOD_WORDS.get(method, method.replace("-", " "))


def full_label(configuration):
    """A configuration's name for a table of all methods, on short lines, so the column stays narrow."""
    # after: "RND<br>weight 1"       after: "RND, archived baseline<br>weight 1"
    # after: "Fast last layer<br>step size 0.03, group 16<br>weight 1"
    settings, words = configuration["settings"], method_words(configuration["method"])
    words = words[0].upper() + words[1:]
    if "step_size" in settings:
        return (f"{words}<br>step size {number(settings['step_size'])}, group {number(settings['sharing_group'])}"
                f"<br>weight {number(settings['intrinsic_weight'])}")
    return f"{words}{', archived baseline' if configuration['archived'] else ''}<br>weight {number(settings['intrinsic_weight'])}"


def blocks(groups):
    """Cut the configurations into the blocks that get a table each: RND, then one per method and sharing group."""
    # after: [("RND", [("weight 0.25", {...}), ..., ("weight 1<br>archived baseline", {...})]),
    #         ("Fast last layer, sharing group 128", [("weight 0.25<br>step size 0.01", {...}), ... 24 rows]), ...]
    result = {}
    for configuration in groups:
        settings, words = configuration["settings"], method_words(configuration["method"])
        title = (f"{words[0].upper() + words[1:]}, sharing group {number(settings['sharing_group'])}"
                 if "sharing_group" in settings else words[0].upper() + words[1:])
        # a row's name stands on short lines, so the column stays narrow
        label = f"weight {number(settings['intrinsic_weight'])}"
        label += f"<br>step size {number(settings['step_size'])}" if "step_size" in settings else ""
        label += "<br>archived baseline" if configuration["archived"] else ""
        result.setdefault(title, []).append((label, configuration))
    return list(result.items())


def score_up_to(run, mark):
    """The mean score over the run up to a mark: the mean of its first `mark` window values; None before that."""
    if len(run["windows"]) < mark:
        return None
    return statistics.fmean(window["mean_score"] for window in run["windows"][:mark])


def first_step(run, event):
    """The training step at which a run first had an event; None when it has not (or has no records yet)."""
    first = run["first"]["first_event"].get(event) if run["first"] else None
    return first["step"] if first else None


def capped_first(run, event, cap):
    """A run's first-time step of an event, capped: a run without the event by `cap` counts as `cap`."""
    step = first_step(run, event)
    return cap if step is None or step > cap else step


def rooms_by(run, cap):
    """How many rooms a run had visited by a step."""
    return sum(first["step"] <= cap for first in run["first"]["first_room_visit"].values()) if run["first"] else 0


def mean(values):
    """The mean of the values that exist; None when none does."""
    values = [value for value in values if value is not None]
    return statistics.fmean(values) if values else None


def emphasised(cells, larger_is_better):
    """Bold the best and underline the second best of a column of (text, value) cells; ties share the mark."""
    # before: [("412", 412.0), ("—", None), ("380", 380.0), ("412", 412.0)], larger is better
    # after:  ["**412**", "—", "<u>380</u>", "**412**"]
    ranked = sorted({value for _, value in cells if value is not None}, reverse=larger_is_better)
    marks = {ranked[0]: "**{}**"} if ranked else {}
    if len(ranked) > 1:
        marks[ranked[1]] = "<u>{}</u>"
    return [marks.get(value, "{}").format(text) for text, value in cells]


def table(headers, rows):
    """Write a Markdown table from its header cells and its rows of cells."""
    return ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] + [
        "| " + " | ".join(row) + " |" for row in rows]


def stage_marks(data, window_updates):
    """The marks runs are compared at, in whole windows, from the longest target of this run folder's units."""
    # before: units of 12,207 updates, windows of 61 updates (200 whole windows)
    # after:  [1, 2, 5, 10, 20, 50, 100, 200]
    targets = [run["target_updates"] for run in data["runs"] if not run["archived"]] or [
        run["target_updates"] for run in data["runs"]]
    windows = max(targets, default=0) // window_updates
    return sorted({max(1, round(fraction * windows)) for fraction in MARK_FRACTIONS}) if windows else []


def compared_mark(groups, marks):
    """The mark the configurations are compared at: the largest one that half of the started configurations reached."""
    started = [configuration for configuration in groups if any(run["started"] for run in configuration["runs"])]
    reached = [mark for mark in marks
               if 2 * sum(any(len(run["windows"]) >= mark for run in configuration["runs"])
                          for configuration in started) >= max(1, len(started))]
    return max(reached, default=None)


def definitions_section(window_steps, window_updates, steps_per_update):
    """The definitions every table uses, written once."""
    return [
        "1. **Score per game:** the raw game score of the training games that finished (not the clipped learning "
        "reward). This game's values differ from the Atari original, so scores are not comparable with published numbers.",
        f"2. **Window:** {window_updates} updates, {window_steps:,} training steps of one policy. Define $g_k$ as the "
        "number of games that finished in window $k$ and $s_k$ as the sum of their scores. The window's value is "
        "$v_k = \\dfrac{s_k}{g_k}$; a window with no finished game takes the previous window's value (0 before the first).",
        "3. **Mean score over the run up to a mark of $m$ windows:** $W(m) = \\dfrac{1}{m} \\sum_{k=1}^{m} v_k$. "
        "It is the measure that chooses each side's configuration.",
        f"4. **First-time step of an event:** for an event in update $u$ (counted from 1) at rollout step $t$ (counted "
        f"from 0), $T = (u - 1) \\times {steps_per_update} + (t + 1) \\times 128$: the training steps of the policy "
        "when it happened, exact to one step of its 128 environments.",
        "5. **Capped first exit at a mark of $S$ steps:** $\\min(T, S)$ for the first exit from the start room; a "
        "policy that has not left by $S$ counts as $S$. Smaller is better.",
        "6. **Fact about the start room:** it can only be left through one of its two locked doors, a door opens "
        "only for a player holding a key, and the room has one key. So the key is picked up before the first exit, "
        "and the number of exits with both doors locked is expected to be zero (standing checks).",
        "7. **Seeds at a mark:** a table cell at a mark uses the seeds that have finished that many windows; the "
        "count stands beside the value. No run is stopped, held or pruned on any number of this report.",
    ]


RUN_STATES = ("not started", "training now", "between two jobs", "on a remote server", "finished", "failed")


def run_state(run):
    """Name where a run stands, in the words of the fleet table."""
    # before: a run whose unit is pending and has record files ; after: "between two jobs"
    state = run["state"]
    if state in ("done", "failed"):
        return "finished" if state == "done" else "failed"
    if state == "running":
        return "training now"
    if state.startswith("remote_"):
        return "on a remote server"
    return "between two jobs" if run["started"] else "not started"


def fleet_section(data):
    """Units and runs by state, and the units that continued on another card type."""
    counts = dict.fromkeys(unit_queue.STATES, 0)
    for unit in data["units"]:
        counts[unit["state"]] = counts.get(unit["state"], 0) + 1
    lines = ["Units by queue folder: " + (", ".join(f"{state} {count}" for state, count in sorted(counts.items()))
                                           or "none") + ". A unit is one process on one card: one run, or four runs packed.", ""]
    # runs by method and state
    rows = {}
    for run in data["runs"]:
        if not run["archived"]:
            row = rows.setdefault(run["method"], dict.fromkeys(RUN_STATES, 0))
            row[run_state(run)] += 1
    lines += table(["method", "runs<br>planned", "not<br>started", "training<br>now", "between<br>two jobs",
                    "on a remote<br>server", "finished", "failed"],
                   [[method_words(method), str(sum(row.values()))] + [str(row[state]) for state in RUN_STATES]
                    for method, row in sorted(rows.items(), key=lambda item: (item[0] != "rnd", item[0]))])
    archived = sum(run["archived"] for run in data["runs"])
    if archived:
        lines += ["", f"Beside them: {archived} runs of the archived baseline (their records hold the score and the "
                  "rooms of every update, and no game events)."]
    # a unit may continue on another card type of the same execution profile; such units are named
    moved = [unit for unit in data["units"] if len(set(unit["cards"])) > 1]
    lines += ["", f"Units that continued on another card type: {len(moved)}."]
    lines += [f"- `{unit['name']}`: {', then '.join(unit['cards'])}" for unit in moved]
    return lines


def speed_section(data, steps_per_update):
    """Speed by card type and copies per process, with the five columns of every throughput table."""
    lines = [f"Define $t$ as the seconds per iteration: one iteration is one update, which advances every copy of a "
             f"process by {steps_per_update:,} steps. The steps per second per copy are $r = \\dfrac{{{steps_per_update}}}{{t}}$, "
             "the total is the copies times $r$, and the hours per million steps per copy are "
             "$\\dfrac{10^{6}}{3600 \\times r}$. A reading is one attempt of a unit: the median of the trainer's own "
             "one-minute rates over that attempt (for a unit that trains now, its last minute).", ""]
    rows = {}
    for reading in data["speeds"]:
        row = rows.setdefault((reading["card"], reading["copies"]), {"seconds": [], "running": 0})
        row["seconds"].append(reading["seconds_per_update"])
        row["running"] += reading["running"]
    if not rows:
        return lines + ["No unit has reported a speed yet."]
    body = []
    for (card, copies), row in sorted(rows.items(), key=lambda item: statistics.median(item[1]["seconds"]) / item[0][1]):
        seconds = statistics.median(row["seconds"])
        per_copy = steps_per_update / seconds
        spread = (f"{seconds:.2f}" if len(row["seconds"]) == 1 else
                  f"{seconds:.2f}<br>({min(row['seconds']):.2f} to {max(row['seconds']):.2f})")
        body.append([card, str(copies), spread, f"{copies * per_copy / 1e6:.4f}", f"{per_copy / 1e3:.1f}",
                     f"{1e6 / (3600 * per_copy):.4f}", f"{len(row['seconds'])}<br>({row['running']} now)"])
    return lines + table(["card type", "copies", "seconds per<br>iteration<br>median (range)",
                          "total steps<br>per second<br>(millions)", "steps per<br>second per<br>copy<br>(thousands)",
                          "hours per<br>million<br>steps per<br>copy", "readings<br>(training<br>now)"], body)


def compared_section(groups, mark, window_steps):
    """Every configuration at the one common mark: seeds, mean score over the run, capped first exit, rooms."""
    if mark is None:
        return ["No mark has been reached by half of the started configurations yet."]
    cap = mark * window_steps
    lines = [f"Compared at the mark of {mark} windows ({millions(cap)} million steps), the largest mark that at least "
             "half of the started configurations have reached. The mean score over the run is $W(m)$ and the capped "
             "first exit is $\\min(T, S)$ of the definitions, each a mean over the seeds at the mark. Rows are in two "
             "blocks, each sorted by the mean score; the best value of a column is bold and the second best "
             "underlined (first exit: smaller is better).", ""]
    rows = []
    for configuration in groups:
        runs = [run for run in configuration["runs"] if len(run["windows"]) >= mark]
        done = sum(run["state"] in ("done", "archived") for run in configuration["runs"])
        started = sum(run["started"] for run in configuration["runs"])
        exits = sum(capped_first(run, "left_start_room", cap) < cap for run in runs)
        rows.append({"configuration": configuration, "seeds": f"{done} / {started} / {len(configuration['runs'])}",
                     "at_mark": len(runs), "score": mean(score_up_to(run, mark) for run in runs),
                     "exit": mean(capped_first(run, "left_start_room", cap) for run in runs) if runs else None,
                     "exits": exits, "rooms": mean(rooms_by(run, cap) for run in runs) if runs else None})
    # two blocks: the method's configurations, then RND's weights and the archived baseline; each by the score
    rows.sort(key=lambda row: (row["configuration"]["method"] == "rnd", row["score"] is None, -(row["score"] or 0),
                               row["configuration"]["key"]))
    scores = emphasised([(f"{row['score']:,.0f}" if row["score"] is not None else DASH, row["score"]) for row in rows], True)
    exits = emphasised([(millions(row["exit"]) if row["exit"] is not None else DASH, row["exit"]) for row in rows], False)
    rooms = emphasised([(f"{row['rooms']:.1f}" if row["rooms"] is not None else DASH, row["rooms"]) for row in rows], True)
    body, in_baseline = [], False
    for row, score, capped, room in zip(rows, scores, exits, rooms):
        if row["configuration"]["method"] == "rnd" and not in_baseline:
            body.append(["**RND**"] + [""] * 6)
            in_baseline = True
        left = f"{row['exits']} of {row['at_mark']}" if row["at_mark"] else DASH
        body.append([full_label(row["configuration"]), row["seeds"], str(row["at_mark"]), score, capped, left, room])
    return lines + table(["configuration", "seeds:<br>done /<br>started /<br>planned", "seeds<br>at the<br>mark",
                          "mean score<br>over the<br>run ↑", "first exit,<br>capped<br>(million<br>steps)",
                          "seeds that<br>left the<br>start room", "rooms<br>reached"], body)


def marks_section(groups, marks, window_steps):
    """The mean score over the run at every mark, one table per block of configurations."""
    reached = [mark for mark in marks if any(len(run["windows"]) >= mark for group in groups for run in group["runs"])]
    if not reached:
        return ["No run has finished its first window yet."]
    lines = ["Each cell is $W(m)$ of the definitions at the mark, as the mean over the seeds that have reached it, "
             "and under it in parentheses the number of those seeds. A column's header is the mark in million steps. "
             "In each table the best value of a column is bold and the second best underlined.", ""]
    for title, members in blocks(groups):
        columns = []
        for mark in reached:
            cells = []
            for _, configuration in members:
                runs = [run for run in configuration["runs"] if len(run["windows"]) >= mark]
                value = mean(score_up_to(run, mark) for run in runs)
                cells.append((f"{value:,.0f}", value) if value is not None else (DASH, None))
            counts = [sum(len(run["windows"]) >= mark for run in configuration["runs"]) for _, configuration in members]
            columns.append([f"{text}<br>({count})" if count else text
                            for text, count in zip(emphasised(cells, True), counts)])
        lines += [f"**{title}**", ""] + table(
            ["configuration"] + [f"{mark * window_steps / 1e6:.0f}" for mark in reached],
            [[label] + [column[index] for column in columns] for index, (label, _) in enumerate(members)]) + [""]
    return lines[:-1]


def first_times_section(groups):
    """Medians of the first-time steps of the game events, one table per block of configurations."""
    lines = ["Each cell holds the median first-time step $T$ of the definitions, in million steps, over the started "
             "seeds of the configuration, and under it k/n: the k seeds that have had the event of the n seeds started. A seed "
             "without the event counts as later than every seed with it; a dash means the median seed has not had "
             "the event yet. The seeds are at different steps while they train, so a count can still rise. For the "
             "archived baseline only the key and the exit are known, each to the end of its update.", ""]
    for title, members in blocks(groups):
        body = []
        for label, configuration in members:
            runs = [run for run in configuration["runs"] if run["started"]]
            row = [label]
            for event, _ in EVENTS:
                steps = [first_step(run, event) for run in runs]
                known = [step for step in steps if step is not None]
                # the median over all started seeds, with a seed that has not had the event placed last
                median = statistics.median([math.inf if step is None else step for step in steps]) if steps else math.inf
                recorded = any(run["has_event_fields"] or event in ("key_pickup", "left_start_room") for run in runs)
                row.append(f"{millions(median) if math.isfinite(median) else DASH}<br>{len(known)}/{len(runs)}"
                           if runs and recorded else DASH)
            body.append(row)
        lines += [f"**{title}**", ""] + table(["configuration"] + [words for _, words in EVENTS], body) + [""]
    return lines[:-1]


def games_section(groups, marks, window_steps):
    """Shares of finished games with the key and out of the start room at each mark, and lives lost by cause."""
    reached = [mark for mark in marks if any(len(run["windows"]) >= mark for group in groups for run in group["runs"])]
    if not reached:
        return ["No run has finished its first window yet."]
    lines = ["Define $g$ as the games that finished in the one window ending at a mark, $a$ as those of them that "
             "had picked up the key, and $b$ as those that had left the start room. The first table's cells hold "
             "$\\dfrac{a}{g}$ and under it $\\dfrac{b}{g}$, each a mean over the seeds at the mark with a finished "
             "game in that window. A column's header is the mark in million steps.", ""]
    for title, members in blocks(groups):
        body = []
        for label, configuration in members:
            row = [label]
            for mark in reached:
                windows = [run["windows"][mark - 1] for run in configuration["runs"]
                           if len(run["windows"]) >= mark and run["has_event_fields"]]
                windows = [window for window in windows if window["games"]]
                with_key = mean(window["games_with_key"] / window["games"] for window in windows)
                left = mean(window["games_left_start_room"] / window["games"] for window in windows)
                row.append(DASH if with_key is None else f"{with_key:.2f}<br>{left:.2f}")
            body.append(row)
        lines += [f"**{title}**", ""] + table(
            ["configuration"] + [f"{mark * window_steps / 1e6:.0f}" for mark in reached], body) + [""]
    # lives lost by cause, per million steps, over every finished window of the configuration's seeds
    lines += ["Lives lost per million training steps, by cause, over all finished windows of the configuration's "
              "seeds so far (the count of all those windows divided by their steps in millions).", ""]
    for title, members in blocks(groups):
        body = []
        for label, configuration in members:
            windows = [window for run in configuration["runs"] if run["has_event_fields"] for window in run["windows"]]
            steps = len(windows) * window_steps / 1e6
            body.append([label] + [f"{sum(window['event_counts'][event] for window in windows) / steps:,.1f}"
                                   if windows else DASH for event, _ in LIFE_CAUSES])
        lines += [f"**{title}**", ""] + table(["configuration"] + [words for _, words in LIFE_CAUSES], body) + [""]
    return lines[:-1]


def levels_section(groups, levels):
    """When the window value first reached each score level, one table per block of configurations."""
    lines = ["A score level is reached by the first window whose value $v_k$ of the definitions is at or above it. "
             "Each cell holds the median step of that window, in million steps, over the started seeds (a seed that "
             "has not reached the level counts as later than every seed that has), and under it k/n: the k seeds that "
             "reached the level of the n seeds started.", ""]
    for title, members in blocks(groups):
        body = []
        for label, configuration in members:
            runs = [run for run in configuration["runs"] if run["started"] and run["first"]]
            row = [label]
            for level in levels:
                reached = [run["first"]["first_window_at_score_level"].get(str(level)) for run in runs]
                steps = [math.inf if entry is None else entry["step"] for entry in reached]
                median = statistics.median(steps) if steps else math.inf
                row.append(f"{millions(median) if math.isfinite(median) else DASH}<br>"
                           f"{sum(entry is not None for entry in reached)}/{len(runs)}" if runs else DASH)
            body.append(row)
        lines += [f"**{title}**", ""] + table(["configuration"] + [f"{level:,}" for level in levels], body) + [""]
    return lines[:-1]


def non_finite(value):
    """Count the numbers of a JSON value that are not finite (NaN or an infinity)."""
    if isinstance(value, float):
        return int(not math.isfinite(value))
    if isinstance(value, dict):
        return sum(non_finite(item) for item in value.values())
    if isinstance(value, list):
        return sum(non_finite(item) for item in value)
    return 0


def checks_section(data):
    """The standing checks: exits with both doors locked, key before exit, values that are not finite, records."""
    runs = [run for run in data["runs"] if run["first"] is not None]
    recorded = [run for run in runs if run["has_event_fields"]]

    def name(run):
        """A run's unit and seed, for a list item."""
        return f"`{run['unit']}`, seed {run['seed']}"

    # 1. the fact of the definitions: no exit from the start room while both of its doors are locked
    locked = [(run, run["first"]["event_totals"][LOCKED_EXIT]) for run in recorded
              if run["first"]["event_totals"][LOCKED_EXIT]]
    lines = [f"1. **Exits from the start room with both doors locked:** {sum(count for _, count in locked)} in "
             f"{len(recorded)} runs with recorded events (expected: 0)."]
    lines += [f"   - {name(run)}: {count}" for run, count in locked]
    # 2. the same fact read from the first times: the first key comes before the first exit
    early = [run for run in recorded if first_step(run, "left_start_room") is not None
             and (first_step(run, "key_pickup") is None
                  or first_step(run, "key_pickup") > first_step(run, "left_start_room"))]
    lines.append(f"2. **Runs whose first exit came before their first key:** {len(early)} (expected: 0).")
    lines += [f"   - {name(run)}" for run in early]
    # 3. numbers that are not finite, in the record files and in the speeds
    broken = [(run, non_finite(run["first"]) + non_finite(run["windows"])) for run in runs]
    broken = [(run, count) for run, count in broken if count]
    lines.append(f"3. **Values that are not finite in the record files:** {sum(count for _, count in broken)} "
                 f"in {len(runs)} runs with records (expected: 0; the trainer also stops on a loss that is not finite).")
    lines += [f"   - {name(run)}: {count}" for run, count in broken]
    # 4. units that train, or trained, and have no record file to read
    silent = [unit for unit in data["units"] if not unit["has_records"] and unit["update"] > 0]
    lines.append(f"4. **Units past their first update without record files:** {len(silent)} (expected: 0).")
    lines += [f"   - `{unit['name']}` ({unit['state']}, update {unit['update']:,})" for unit in silent]
    failed = [unit for unit in data["units"] if unit["state"] == "failed"]
    lines.append(f"5. **Failed units:** {len(failed)}.")
    lines += [f"   - `{unit['name']}`: {unit['unit'].get('failure_reason')}" for unit in failed]
    return lines


def anchor(title):
    """The link target a Markdown viewer gives a heading."""
    # before: "Mean score over the run, by mark" ; after: "mean-score-over-the-run-by-mark"
    return re.sub(r"[^a-z0-9 -]", "", title.lower()).replace(" ", "-")


def contents_section(sections, manifest, moment):
    """The contents table: when each section first appeared and last changed, and whether it was read."""
    # the record per section is updated here: its text's digest, and the time of the first and the last change
    rows = []
    for title, lines in sections:
        digest = sha256("\n".join(lines).encode()).hexdigest()
        entry = manifest.setdefault(title, {"first_written": moment, "last_changed": moment, "digest": digest,
                                            "read_digest": None})
        if entry["digest"] != digest:
            entry.update(digest=digest, last_changed=moment)
        status = ("$\\color{blue}{\\text{unread}}$" if entry["read_digest"] is None else
                  "read" if entry["read_digest"] == digest else "$\\color{#852D0F}{\\text{updated}}$")
        rows.append([f"[{title}](#{anchor(title)})", common.pacific(entry["first_written"]),
                     common.pacific(entry["last_changed"]), status])
    return table(["section", "first written", "last changed", "status"], rows)


def build(run_dir, moment=None):
    """Build the whole report's text and the updated contents record; returns (text, manifest)."""
    run_dir = Path(run_dir)
    moment = moment or common.now()
    data = collect(run_dir)
    # the sizes of an update and of a window come from the record files themselves
    described = next((common.read_json(path) for path in sorted(run_dir.glob("data/*/run/game_first_times.json"))), {})
    steps_per_update = described.get("steps_per_update", pixel_queue.STEPS_PER_UPDATE)
    window_updates = described.get("window_updates", 61)
    levels = described.get("score_levels", [400, 1000, 2500])
    window_steps = window_updates * steps_per_update
    groups = configurations(data["runs"])
    marks = stage_marks(data, window_updates)
    sections = [
        ("Definitions", definitions_section(window_steps, window_updates, steps_per_update)),
        ("Fleet status", fleet_section(data)),
        ("Speed by card type", speed_section(data, steps_per_update)),
        ("Configurations at the compared mark", compared_section(groups, compared_mark(groups, marks), window_steps)),
        ("Mean score over the run, by mark", marks_section(groups, marks, window_steps)),
        ("First-time steps of the game events", first_times_section(groups)),
        ("Keys, exits and lives lost", games_section(groups, marks, window_steps)),
        ("Score levels", levels_section(groups, levels)),
        ("Standing checks", checks_section(data)),
    ]
    manifest_path = run_dir / "early_signals_manifest.json"
    manifest = common.read_json(manifest_path) if manifest_path.exists() else {}
    lines = [f"# Early signals: {run_dir.name}", "",
             f"Written {common.pacific(moment)} from the record files of every run. Times are Pacific, converted from "
             "machines that run on Eastern. This page is rebuilt at every monitoring pass; it is a reading of runs "
             "that still train, and no run is stopped on it.", "",
             *contents_section(sections, manifest, moment), ""]
    for title, body in sections:
        lines += [f"## {title}", "", *body, ""]
    return "\n".join(lines), manifest


def mark_read(run_dir, title):
    """Record that the reader has read one section, or all of them, as the text stands now."""
    path = Path(run_dir) / "early_signals_manifest.json"
    manifest = common.read_json(path)
    names = list(manifest) if title == "all" else [title]
    for name in names:
        if name not in manifest:
            raise SystemExit(f"the report has no section {name!r}; its sections are: {', '.join(manifest)}")
        manifest[name]["read_digest"] = manifest[name]["digest"]
        manifest[name]["read_at"] = common.now()
    common.write_json(path, manifest)
    return names


def fold_archive(run_dir, archive_data, match="final-training"):
    """Make the record files of archived runs from their metrics, with the trainer's own fold; returns their names."""
    game_records = load_file("pixel_game_records", PLATFORM / "scripts/atari_pixel_campaign/game_records.py")
    folded = []
    for metrics in sorted(Path(archive_data).glob("*/run/metrics.jsonl")):
        name = metrics.parents[1].name
        if match not in name:
            continue
        target = Path(run_dir) / ARCHIVE_FOLDER / name
        target.mkdir(parents=True, exist_ok=True)
        game_records.rebuild(metrics, target)
        # run.json says which seeds and which settings the archived run had
        shutil.copy2(metrics.parent / "run.json", target / "run.json")
        folded.append(name)
        print(f"folded {name} into {target}", flush=True)
    return folded


def main(argv=None):
    """Rebuild the report, mark sections read, or fold the archived runs' metrics into record files."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="where the report is written; default: RUN/early_signals.md")
    parser.add_argument("--mark-read", help="mark one section (by its title) or all sections as read, then rebuild")
    parser.add_argument("--fold-archive", type=Path,
                        help="the data folder of an archived run folder: make RUN/archived_baseline/ from it "
                             "(reads every archived metric line; run it inside a Slurm job)")
    parser.add_argument("--match", default="final-training",
                        help="with --fold-archive: only archived units whose name holds this text")
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    if args.fold_archive:
        fold_archive(run_dir, args.fold_archive, args.match)
    if args.mark_read:
        print("marked as read: " + ", ".join(mark_read(run_dir, args.mark_read)))
    text, manifest = build(run_dir)
    output = args.output or run_dir / "early_signals.md"
    # the page is written under another name first, so a reader never sees half of it
    temporary = output.with_name(output.name + ".partial")
    temporary.write_text(text)
    temporary.replace(output)
    common.write_json(run_dir / "early_signals_manifest.json", manifest)
    print(f"wrote {output} at {datetime.now().astimezone(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d %H:%M PT')}")


if __name__ == "__main__":
    main()
