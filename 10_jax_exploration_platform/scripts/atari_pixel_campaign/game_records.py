"""Fold the per-update metric lines of a pixel Montezuma run into two small record files.

Standard library only. One fold reads the lines of metrics.jsonl in order (train_final.py writes
one line per update). It holds no state of its own on disk: the two files below are a function of
the metric lines alone, so after a resume they are rebuilt from the kept lines and cannot fall out
of step with them.

    python game_records.py --metrics RUN/metrics.jsonl --output-dir SOME_FOLDER

writes both files for an existing metrics file (also one written before the event fields existed).

game_first_times.json
    {"environments_per_policy": 128, "rollout_steps": 128, "steps_per_update": 16384,
     "window_updates": 61, "updates_folded": 200, "has_event_fields": true,
     "event_names": [...], "start_room": 4, "score_levels": [400, 1000, 2500],
     "copies": [one entry per copy of the process:
        {"first_event": {"key_pickup": FIRST or null, ... one entry per event name ...},
         "first_left_start_room_destinations": [5],
         "first_room_visit": {"4": FIRST, "5": FIRST},
         "first_window_at_score_level": {"400": {"window": 3, "update": 183, "step": 2998272,
                                                 "mean_score": 412.5} or null, ...},
         "event_totals": {"key_pickup": 12, ...} or null,
         "games": 1234, "score_sum": 56000.0,
         "games_with_key": 20 or null, "games_left_start_room": 3 or null}]}
    FIRST = {"update": 37, "rollout_step": 101, "step": 602880, "exact": true}

    step = (update - 1) * steps_per_update + (rollout_step + 1) * environments_per_policy: the
    number of training steps of the policy when the event happened, exact to one step of its
    parallel environments.

game_windows.jsonl: one line per completed window of `window_updates` updates
    {"window": 1, "first_update": 1, "last_update": 61, "step": 999424,
     "games": [per copy], "score_sum": [per copy], "mean_score": [per copy],
     "games_with_key": [per copy] or null, "games_left_start_room": [per copy] or null,
     "event_counts": {"key_pickup": [per copy], ...} or null}

    mean_score is score_sum / games of the window; a window with no finished game takes the
    previous window's value (0 before the first). A score level is reached by the first window
    whose mean_score is at or above it.

Metric lines without event fields (runs with record_game_events off, and the archived baseline)
give less: has_event_fields is false, the per-game and per-event sums are null, and first_event
holds only two entries, each with "exact": false, "rollout_step": null and the step at the END
of the update in which it was first seen:
- key_pickup: the first update with a positive extrinsic reward. In this game the first reward of
  a run is the start room's key: a door needs a key and a kill needs a sword from another room.
- left_start_room: the first update in which a game of the copy was in another room after a step.
first_room_visit is filled the same way from the rooms_visited field.
"""

import argparse
import importlib.util
import json
from pathlib import Path

EVENT_NAMES_FILE = (Path(__file__).resolve().parents[2]
                    / "src/exploration_platform/atari_pixels/game_event_names.py")
WINDOW_UPDATES = 61
SCORE_LEVELS = (400, 1000, 2500)
FIRST_TIMES_FILE = "game_first_times.json"
WINDOWS_FILE = "game_windows.jsonl"


def load_event_names():
    """Read the event names and the start room from the trainer's names file, by its path."""
    # Loading the file by path leaves the package (and with it jax) unimported.
    specification = importlib.util.spec_from_file_location("game_event_names", EVENT_NAMES_FILE)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return tuple(module.GAME_EVENT_NAMES), int(module.START_ROOM)


class GameRecords:
    """The fold: first-time steps and per-window sums of every copy, from metric lines in order."""

    def __init__(self, environments_per_policy=128, rollout_steps=128, window_updates=WINDOW_UPDATES):
        """Start with no line folded; the number of copies is read from the first line."""
        if min(environments_per_policy, rollout_steps, window_updates) < 1:
            raise ValueError("environments per policy, rollout steps and window updates must be positive")
        self.environments_per_policy = environments_per_policy
        self.rollout_steps = rollout_steps
        self.steps_per_update = environments_per_policy * rollout_steps
        self.window_updates = window_updates
        self.event_names, self.start_room = load_event_names()
        self.updates_folded = 0
        self.has_event_fields = None
        self.copies = None
        self.window = None
        self.mean_scores = None
        # True when a first time or a score level was recorded since the first-times file was written.
        self.changed = False

    def _first(self, update, rollout_step):
        """The record of one first time; without a rollout step it is the end of the update."""
        # before: update 3, rollout step 9, 128 environments, 16,384 steps per update
        # after: {"update": 3, "rollout_step": 9, "step": 2 * 16384 + 10 * 128, "exact": true}
        if rollout_step is None:
            return {"update": update, "rollout_step": None, "step": update * self.steps_per_update,
                    "exact": False}
        step = (update - 1) * self.steps_per_update + (rollout_step + 1) * self.environments_per_policy
        return {"update": update, "rollout_step": rollout_step, "step": step, "exact": True}

    def _empty_window(self, copies):
        """Zero sums for a new window; the event sums exist only when the lines carry events."""
        counted = self.has_event_fields
        return {"games": [0] * copies, "score_sum": [0.0] * copies,
                "games_with_key": [0] * copies if counted else None,
                "games_left_start_room": [0] * copies if counted else None,
                "event_counts": {name: [0] * copies for name in self.event_names} if counted else None}

    def _start(self, line):
        """Read the number of copies and whether events are recorded from the first line."""
        copies = len(line["episode_count"])
        self.has_event_fields = "event_counts" in line
        counted = self.has_event_fields
        names = self.event_names if counted else ()
        self.copies = [{"first_event": {name: None for name in names},
                        "first_left_start_room_destinations": [],
                        "first_room_visit": {},
                        "first_window_at_score_level": {str(level): None for level in SCORE_LEVELS},
                        "event_totals": {name: 0 for name in names} if counted else None,
                        "games": 0, "score_sum": 0.0,
                        "games_with_key": 0 if counted else None,
                        "games_left_start_room": 0 if counted else None} for _ in range(copies)]
        self.window = self._empty_window(copies)
        self.mean_scores = [0.0] * copies

    def _fold_events(self, line, copy):
        """Add one copy's exact events of one update to its totals, first times and window."""
        record, update = self.copies[copy], line["update"]
        room_steps = line["room_first_step"][copy]
        # before: event_counts [0, 2, ...], event_first_step [-1, 57, ...]
        # after: the second event's total is 2 higher and, when it was unseen, its first time is
        # (update, rollout step 57)
        for index, name in enumerate(self.event_names):
            count = line["event_counts"][copy][index]
            record["event_totals"][name] += count
            self.window["event_counts"][name][copy] += count
            if count > 0 and record["first_event"][name] is None:
                first_step = line["event_first_step"][copy][index]
                record["first_event"][name] = self._first(update, first_step)
                self.changed = True
                if name == "left_start_room":
                    # The rooms first entered at the step of the first exit are where it led.
                    record["first_left_start_room_destinations"] = [
                        room for room, step in enumerate(room_steps)
                        if step == first_step and room != self.start_room]
        # The first rollout step in each room, kept for rooms not seen before.
        for room, step in enumerate(room_steps):
            if step >= 0 and str(room) not in record["first_room_visit"]:
                record["first_room_visit"][str(room)] = self._first(update, step)
                self.changed = True
        for name in ("games_with_key", "games_left_start_room"):
            record[name] += line[name][copy]
            self.window[name][copy] += line[name][copy]

    def _fold_without_events(self, line, copy):
        """Take what a line without event fields can say: the update of the first reward and rooms."""
        record, update = self.copies[copy], line["update"]
        first = record["first_event"]
        if line["reward_ext_sum"][copy] > 0 and "key_pickup" not in first:
            first["key_pickup"] = self._first(update, None)
            self.changed = True
        # before: rooms_visited [False, False, False, False, True, True, False, ...] (33 rooms)
        # after: rooms 4 and 5 have a first visit; room 5 is another room than the start room
        visited = [room for room, seen in enumerate(line["rooms_visited"][copy]) if seen]
        others = [room for room in visited if room != self.start_room]
        if others and "left_start_room" not in first:
            first["left_start_room"] = self._first(update, None)
            record["first_left_start_room_destinations"] = others
            self.changed = True
        for room in visited:
            if str(room) not in record["first_room_visit"]:
                record["first_room_visit"][str(room)] = self._first(update, None)
                self.changed = True

    def _close_window(self):
        """Finish the current window: its line, the score levels it reaches, and a new window."""
        update = self.updates_folded
        number = update // self.window_updates
        for copy, record in enumerate(self.copies):
            games = self.window["games"][copy]
            if games:
                self.mean_scores[copy] = self.window["score_sum"][copy] / games
            # A level is reached by the first window whose mean score is at or above it.
            for level in SCORE_LEVELS:
                reached = record["first_window_at_score_level"]
                if reached[str(level)] is None and self.mean_scores[copy] >= level:
                    reached[str(level)] = {"window": number, "update": update,
                                           "step": update * self.steps_per_update,
                                           "mean_score": self.mean_scores[copy]}
                    self.changed = True
        closed = {"window": number, "first_update": update - self.window_updates + 1,
                  "last_update": update, "step": update * self.steps_per_update,
                  "games": self.window["games"], "score_sum": self.window["score_sum"],
                  "mean_score": list(self.mean_scores),
                  "games_with_key": self.window["games_with_key"],
                  "games_left_start_room": self.window["games_left_start_room"],
                  "event_counts": self.window["event_counts"]}
        self.window = self._empty_window(len(self.copies))
        return closed

    def fold(self, line):
        """Add one metric line; returns the window this line completes, or None."""
        if line["update"] != self.updates_folded + 1:
            raise ValueError(f"metric lines must be consecutive: expected update "
                             f"{self.updates_folded + 1}, got {line['update']}")
        if self.copies is None:
            self._start(line)
        if ("event_counts" in line) != self.has_event_fields:
            raise ValueError(f"update {line['update']}: lines with and without event fields are mixed")
        for copy, record in enumerate(self.copies):
            # Finished games and their raw scores, for the whole run and for the current window.
            record["games"] += line["episode_count"][copy]
            record["score_sum"] += line["episode_return_sum"][copy]
            self.window["games"][copy] += line["episode_count"][copy]
            self.window["score_sum"][copy] += line["episode_return_sum"][copy]
            if self.has_event_fields:
                self._fold_events(line, copy)
            else:
                self._fold_without_events(line, copy)
        self.updates_folded = line["update"]
        return self._close_window() if self.updates_folded % self.window_updates == 0 else None

    def first_times(self):
        """The contents of game_first_times.json for the lines folded so far."""
        return {"environments_per_policy": self.environments_per_policy,
                "rollout_steps": self.rollout_steps, "steps_per_update": self.steps_per_update,
                "window_updates": self.window_updates, "updates_folded": self.updates_folded,
                "has_event_fields": self.has_event_fields, "event_names": list(self.event_names),
                "start_room": self.start_room, "score_levels": list(SCORE_LEVELS),
                "copies": self.copies if self.copies is not None else []}


def write_first_times(path, records):
    """Replace game_first_times.json atomically with the fold's current contents."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(records.first_times(), indent=2) + "\n")
    temporary.replace(path)
    records.changed = False


def append_window(path, window):
    """Append one completed window to game_windows.jsonl and flush it."""
    with path.open("a") as stream:
        stream.write(json.dumps(window) + "\n")
        stream.flush()


def rebuild(metrics_path, output_dir, **settings):
    """Fold every line of a metrics file and write both record files anew; returns the fold.

    A missing metrics file counts as no line (a run that has not finished its first update).
    """
    records, windows = GameRecords(**settings), []
    if metrics_path.exists():
        with metrics_path.open() as stream:
            for text in stream:
                if text.strip():
                    window = records.fold(json.loads(text))
                    if window is not None:
                        windows.append(window)
    # The windows file is written whole under another name first, like the first-times file.
    windows_path = output_dir / WINDOWS_FILE
    temporary = windows_path.with_suffix(windows_path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(window) + "\n" for window in windows))
    temporary.replace(windows_path)
    write_first_times(output_dir / FIRST_TIMES_FILE, records)
    return records


def main(argv=None):
    """Write both record files for one metrics file."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics", type=Path, required=True, help="a run's metrics.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True, help="where both record files go")
    parser.add_argument("--environments-per-policy", type=int, default=128)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--window-updates", type=int, default=WINDOW_UPDATES)
    args = parser.parse_args(argv)
    if not args.metrics.exists():
        raise SystemExit(f"no metrics file at {args.metrics}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = rebuild(args.metrics, args.output_dir, environments_per_policy=args.environments_per_policy,
                      rollout_steps=args.rollout_steps, window_updates=args.window_updates)
    print(f"folded {records.updates_folded} updates into {args.output_dir}")


if __name__ == "__main__":
    main()
