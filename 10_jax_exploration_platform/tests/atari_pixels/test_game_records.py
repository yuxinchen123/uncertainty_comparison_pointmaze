"""The fold of metric lines into the two game record files. Standard library only.

These tests import no numerical library and start no trainer, so they may run on a login node:
`python3 -m unittest tests/atari_pixels/test_game_records.py` from the platform folder.
"""

import json
from pathlib import Path
import sys
import tempfile
import unittest

PLATFORM = Path(__file__).resolve().parents[2]
SCRIPTS = PLATFORM / "scripts/atari_pixel_campaign"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import game_records  # noqa: E402

NAMES, START_ROOM = game_records.load_event_names()
ROOMS = 33
ARCHIVED_METRICS = Path(
    "/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/runs/"
    "2026-11-06-01-14_pixel-montezuma_corrected-room-geometry_episode-limit-27000-steps_cleanrl-ppo-rnd-"
    "final-configuration_checks-then-steps-per-copy-2000000000_h100-nvl-serval02-card-1-copies-4-seeds-1-to-4_"
    "rtx-a4500-jaguar03-8-cards-copies-1-seeds-5-to-12/data/a4500-jaguar03_final-training_seed-5/run/metrics.jsonl")


def metric_line(update, games=(0,), scores=(0.0,), events=None, rooms=None, with_key=None, left=None,
                reward=None):
    """One metric line as train_final.py writes it, for len(games) copies.

    events: per copy {event name: (count, first rollout step)}; rooms: per copy {room: first rollout
    step}. Both None writes a line without event fields (the form of the archived baseline), whose
    rooms_visited then comes from `rooms` being a per-copy list of visited rooms.
    """
    copies = len(games)
    line = {"update": update, "unix": 1000.0 + update, "episode_count": list(games),
            "episode_return_sum": list(scores),
            "reward_ext_sum": list(reward) if reward is not None else [0.0] * copies}
    if events is None:
        visited = rooms if rooms is not None else [[START_ROOM]] * copies
        line["rooms_visited"] = [[room in visited[copy] for room in range(ROOMS)] for copy in range(copies)]
        return line
    rooms = rooms if rooms is not None else [{START_ROOM: 0}] * copies
    line["rooms_visited"] = [[room in rooms[copy] for room in range(ROOMS)] for copy in range(copies)]
    line["event_counts"] = [[events[copy].get(name, (0, -1))[0] for name in NAMES] for copy in range(copies)]
    line["event_first_step"] = [[events[copy].get(name, (0, -1))[1] for name in NAMES] for copy in range(copies)]
    line["room_first_step"] = [[rooms[copy].get(room, -1) for room in range(ROOMS)] for copy in range(copies)]
    line["games_with_key"] = list(with_key) if with_key is not None else [0] * copies
    line["games_left_start_room"] = list(left) if left is not None else [0] * copies
    return line


def scripted_run(updates):
    """A two-copy run with events: copy 0 finds the key, a door and an exit; copy 1 only loses lives."""
    lines = []
    for update in range(1, updates + 1):
        events = [{}, {"life_lost_by_fall": (2, 5)}]
        rooms = [{START_ROOM: 0}, {START_ROOM: 0}]
        if update >= 3:
            events[0]["key_pickup"] = (1 + update % 2, 40 + update)
        if update == 5:
            events[0]["door_opened"] = (1, 100)
            events[0]["left_start_room"] = (1, 110)
            rooms[0] = {START_ROOM: 0, START_ROOM + 1: 110}
        if update >= 6:
            rooms[0] = {START_ROOM: 0, START_ROOM + 1: 3}
        lines.append(metric_line(update, games=(update % 3, 1), scores=(400.0 * (update % 3), 0.0),
                                 events=events, rooms=rooms, with_key=(update % 3 if update >= 3 else 0, 0),
                                 left=(1 if update >= 5 and update % 3 else 0, 0), reward=(1.0, 0.0)))
    return lines


def train_like_run(folder, lines, window_updates, stop_after=None, lines_written_after_stop=0):
    """Write the record files the way train_final.py does, optionally stopping and resuming once.

    stop_after: the update of the last checkpoint before the stop. lines_written_after_stop: metric
    lines that reached the file after that checkpoint; the resume drops them and plays them again.
    """
    folder.mkdir(parents=True, exist_ok=True)
    metrics = folder / "metrics.jsonl"
    settings = {"window_updates": window_updates}

    def play(records, chosen):
        """Append the chosen metric lines and fold each one, as the training loop does."""
        with metrics.open("a") as stream:
            for line in chosen:
                stream.write(json.dumps(line) + "\n")
                stream.flush()
                window = records.fold(line)
                if window is not None:
                    game_records.append_window(folder / game_records.WINDOWS_FILE, window)
                if records.changed:
                    game_records.write_first_times(folder / game_records.FIRST_TIMES_FILE, records)

    records = game_records.rebuild(metrics, folder, **settings)
    if stop_after is None:
        play(records, lines)
    else:
        # The first process: it stops after some lines beyond its last checkpoint.
        play(records, lines[:stop_after + lines_written_after_stop])
        # The second process: keep the lines up to the checkpoint, rebuild, continue.
        kept = [json.loads(text) for text in metrics.read_text().splitlines() if json.loads(text)["update"] <= stop_after]
        metrics.write_text("".join(json.dumps(line) + "\n" for line in kept))
        records = game_records.rebuild(metrics, folder, **settings)
        play(records, lines[stop_after:])
    # The trainer writes the first times once more with its last checkpoint.
    game_records.write_first_times(folder / game_records.FIRST_TIMES_FILE, records)
    return records


class GameRecordsTest(unittest.TestCase):
    """The fold's first times, windows, resume behaviour and input checks."""

    def test_first_times_use_the_exact_step_of_the_event(self):
        """The first key, door and exit carry their update and rollout step; totals add up."""
        records = game_records.GameRecords(window_updates=61)
        for line in scripted_run(8):
            records.fold(line)
        first = records.first_times()
        self.assertEqual(first["updates_folded"], 8)
        self.assertTrue(first["has_event_fields"])
        self.assertEqual(first["steps_per_update"], 16384)
        copy = first["copies"][0]
        # update 3, rollout step 43: 2 full updates and 44 steps of 128 environments
        self.assertEqual(copy["first_event"]["key_pickup"],
                         {"update": 3, "rollout_step": 43, "step": 2 * 16384 + 44 * 128, "exact": True})
        self.assertEqual(copy["first_event"]["door_opened"]["step"], 4 * 16384 + 101 * 128)
        self.assertEqual(copy["first_event"]["left_start_room"]["step"], 4 * 16384 + 111 * 128)
        self.assertEqual(copy["first_left_start_room_destinations"], [START_ROOM + 1])
        self.assertIsNone(copy["first_event"]["enemy_killed"])
        self.assertEqual(copy["first_room_visit"][str(START_ROOM)]["step"], 128)
        self.assertEqual(copy["first_room_visit"][str(START_ROOM + 1)]["update"], 5)
        self.assertEqual(copy["event_totals"]["key_pickup"], sum(1 + update % 2 for update in range(3, 9)))
        # The other copy never has the key and loses two lives in every update.
        other = first["copies"][1]
        self.assertIsNone(other["first_event"]["key_pickup"])
        self.assertEqual(other["event_totals"]["life_lost_by_fall"], 16)
        self.assertEqual(other["first_event"]["life_lost_by_fall"]["step"], 6 * 128)
        self.assertEqual(other["games"], 8)

    def test_windows_carry_the_mean_score_forward_and_record_score_levels(self):
        """A window without a finished game keeps the previous mean; a level is recorded once."""
        records = game_records.GameRecords(window_updates=2)
        scores = [(0, 0.0), (0, 0.0), (2, 600.0), (2, 400.0), (0, 0.0), (0, 0.0), (1, 2600.0), (0, 0.0)]
        windows = []
        for update, (games, score) in enumerate(scores, start=1):
            window = records.fold(metric_line(update, games=(games,), scores=(score,)))
            if window is not None:
                windows.append(window)
        self.assertEqual([window["window"] for window in windows], [1, 2, 3, 4])
        self.assertEqual([window["mean_score"][0] for window in windows], [0.0, 250.0, 250.0, 2600.0])
        self.assertEqual(windows[1], {"window": 2, "first_update": 3, "last_update": 4, "step": 4 * 16384,
                                      "games": [4], "score_sum": [1000.0], "mean_score": [250.0],
                                      "games_with_key": None, "games_left_start_room": None,
                                      "event_counts": None})
        levels = records.first_times()["copies"][0]["first_window_at_score_level"]
        self.assertEqual(levels["400"], {"window": 4, "update": 8, "step": 8 * 16384, "mean_score": 2600.0})
        self.assertEqual(levels["1000"]["window"], 4)
        self.assertEqual(levels["2500"]["window"], 4)

    def test_window_sums_hold_events_and_game_facts(self):
        """With event fields a window holds the per-event counts and the games with each fact."""
        records = game_records.GameRecords(window_updates=4)
        windows = [window for line in scripted_run(8) if (window := records.fold(line)) is not None]
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0]["event_counts"]["life_lost_by_fall"], [0, 8])
        self.assertEqual(windows[0]["event_counts"]["key_pickup"], [2 + 1, 0])
        self.assertEqual(windows[1]["games_with_key"], [2 + 0 + 1 + 2, 0])
        self.assertEqual(windows[1]["games_left_start_room"], [1 + 0 + 1 + 1, 0])

    def test_files_after_a_resume_equal_the_uninterrupted_files(self):
        """Stopping at any update, with or without lines beyond the checkpoint, changes neither file."""
        lines = scripted_run(40)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_like_run(root / "uninterrupted", lines, window_updates=8)
            expected = {name: (root / "uninterrupted" / name).read_bytes()
                        for name in (game_records.FIRST_TIMES_FILE, game_records.WINDOWS_FILE)}
            self.assertEqual(len(expected[game_records.WINDOWS_FILE].splitlines()), 5)
            # Stops inside a window, at a window's end, before the first event, and at the first update.
            for case, (stop_after, extra) in enumerate([(20, 0), (20, 3), (16, 0), (16, 5), (2, 0), (1, 7), (39, 1)]):
                folder = root / f"resumed_{case}"
                train_like_run(folder, lines, window_updates=8, stop_after=stop_after,
                               lines_written_after_stop=extra)
                for name, content in expected.items():
                    self.assertEqual((folder / name).read_bytes(), content,
                                     f"{name} differs after a stop at update {stop_after} with {extra} extra lines")

    def test_lines_without_event_fields_give_the_update_of_the_first_reward_and_exit(self):
        """A baseline line has no events: the first key and exit are known to one update only."""
        records = game_records.GameRecords(window_updates=61)
        records.fold(metric_line(1))
        records.fold(metric_line(2, reward=(3.0,)))
        records.fold(metric_line(3, reward=(1.0,), rooms=[[START_ROOM, START_ROOM - 1]]))
        first = records.first_times()
        self.assertFalse(first["has_event_fields"])
        copy = first["copies"][0]
        self.assertEqual(copy["first_event"], {
            "key_pickup": {"update": 2, "rollout_step": None, "step": 2 * 16384, "exact": False},
            "left_start_room": {"update": 3, "rollout_step": None, "step": 3 * 16384, "exact": False}})
        self.assertEqual(copy["first_left_start_room_destinations"], [START_ROOM - 1])
        self.assertEqual(sorted(copy["first_room_visit"]), sorted([str(START_ROOM), str(START_ROOM - 1)]))
        self.assertIsNone(copy["event_totals"])
        self.assertIsNone(copy["games_with_key"])

    @unittest.skipUnless(ARCHIVED_METRICS.exists(), "the archived baseline is not on this machine")
    def test_the_archived_baseline_lines_are_accepted(self):
        """The first 200 lines of the archived seed 5 fold without error and name the start room."""
        records = game_records.GameRecords()
        windows = 0
        with ARCHIVED_METRICS.open() as stream:
            for _ in range(200):
                windows += records.fold(json.loads(stream.readline())) is not None
        first = records.first_times()
        self.assertEqual(first["updates_folded"], 200)
        self.assertEqual(windows, 3)
        self.assertFalse(first["has_event_fields"])
        self.assertEqual(first["copies"][0]["first_room_visit"][str(START_ROOM)]["update"], 1)
        self.assertGreater(first["copies"][0]["games"], 0)

    def test_a_missing_or_repeated_update_is_refused(self):
        """Lines must be consecutive from update 1, so a gap or a duplicate cannot go unseen."""
        records = game_records.GameRecords()
        records.fold(metric_line(1))
        with self.assertRaisesRegex(ValueError, "expected update 2, got 3"):
            records.fold(metric_line(3))
        with self.assertRaisesRegex(ValueError, "expected update 2, got 1"):
            records.fold(metric_line(1))
        with self.assertRaisesRegex(ValueError, "expected update 1, got 2"):
            game_records.GameRecords().fold(metric_line(2))

    def test_mixing_lines_with_and_without_event_fields_is_refused(self):
        """One run records events in every line or in none."""
        records = game_records.GameRecords()
        records.fold(metric_line(1))
        with self.assertRaisesRegex(ValueError, "mixed"):
            records.fold(scripted_run(2)[1])

    def test_rebuild_of_an_empty_run_writes_both_files(self):
        """A run with no finished update has an empty windows file and no copies yet."""
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            records = game_records.rebuild(folder / "metrics.jsonl", folder)
            self.assertEqual(records.updates_folded, 0)
            self.assertEqual((folder / game_records.WINDOWS_FILE).read_text(), "")
            first = json.loads((folder / game_records.FIRST_TIMES_FILE).read_text())
            self.assertEqual(first["copies"], [])
            self.assertEqual(first["event_names"], list(NAMES))

    def test_command_line_writes_both_files_into_another_folder(self):
        """The command reads a metrics file and leaves its folder untouched."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "run").mkdir()
            metrics = root / "run" / "metrics.jsonl"
            metrics.write_text("".join(json.dumps(line) + "\n" for line in scripted_run(9)))
            game_records.main(["--metrics", str(metrics), "--output-dir", str(root / "records"),
                               "--window-updates", "3"])
            self.assertEqual(sorted(path.name for path in (root / "run").iterdir()), ["metrics.jsonl"])
            windows = (root / "records" / game_records.WINDOWS_FILE).read_text().splitlines()
            self.assertEqual(len(windows), 3)
            self.assertEqual(json.loads((root / "records" / game_records.FIRST_TIMES_FILE).read_text())["updates_folded"], 9)


if __name__ == "__main__":
    unittest.main()
