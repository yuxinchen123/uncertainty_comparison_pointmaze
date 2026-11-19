"""The early-signals report on small synthetic record files, and on lines in the archived baseline's format."""
import contextlib
import io
import json
from pathlib import Path
import re
import tempfile
import unittest

from pixel_test_support import (ARCHIVE_DATA, CARD_ONE_COPY, CARD_SECOND_TYPE, build_units, claimant, common,
                                metric_lines, own, pixel_queue, queue_units, small_description, unit_queue,
                                write_run_records)

report = own("report")
PROFILES = pixel_queue.read_profiles()
WINDOW_STEPS = 61 * 16384
METHOD = "grid_fast-last-layer_step-size-0.03_sharing-group-16_intrinsic-weight-1_seed-13"
RND = "grid_rnd_intrinsic-weight-1_seed-13"
# the method's seed 13: key in update 37, door and exit in update 70, an enemy kill in update 130
METHOD_EVENTS = {37: {"key_pickup": 101}, 70: {"door_opened": 4, "left_start_room": 5}, 130: {"enemy_killed": 9}}


def section(text, title):
    """The lines of one section of a report."""
    lines = text.split("\n")
    start = lines.index(f"## {title}")
    end = next((index for index in range(start + 1, len(lines)) if lines[index].startswith("## ")), len(lines))
    return lines[start + 1:end]


def visible(cell):
    """The widest line a table cell shows, without its Markdown and formula markup."""
    # before: "**412** (3)" ; after: 7       before: "[Fleet status](#fleet-status)" ; after: 12
    cell = re.sub(r"\$\\color\{[^}]*\}\{\\text\{([^}]*)\}\}\$", r"\1", cell)
    cell = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", cell)
    cell = re.sub(r"\*\*|</?u>|`", "", cell)
    return max(len(part.strip()) for part in cell.split("<br>"))


def table_widths(text):
    """The printed width of every table of a report: the sum of its columns' widest cells, plus three per column."""
    widths, rows = [], []
    for line in text.split("\n") + [""]:
        if line.startswith("|"):
            rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
        elif rows:
            body = [row for row in rows if not all(set(cell) <= {"-"} for cell in row)]
            widths.append(sum(max(visible(row[column]) for row in body) + 3 for column in range(len(body[0]))))
            rows = []
    return widths


class ReportCase(unittest.TestCase):
    """A temporary run folder with the small study: two units done, one training, the rest pending."""

    def setUp(self):
        """Queue the eight units and give three of them record files, as their trainers would have written."""
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.run_dir = Path(folder.name) / "run"
        queue_units(self.run_dir, build_units.study_units(small_description()))
        # RND at weight 1, seed 13: done, 244 updates (4 windows), 100 points per game, no event ever
        self.start(RND, CARD_ONE_COPY)
        write_run_records(pixel_queue.run_folder(self.run_dir, RND), metric_lines(244), 244, completed=True)
        self.finish(RND, 1.4)
        # the method, seed 13: done on a second card type after a first stretch; 300 points per game, with events
        unit = self.start(METHOD, CARD_ONE_COPY)
        unit_queue.give_back(self.run_dir, unit, "stopped_with_state_saved", seconds_per_update=1.6)
        self.start(METHOD, CARD_SECOND_TYPE)
        write_run_records(pixel_queue.run_folder(self.run_dir, METHOD),
                          metric_lines(244, events=METHOD_EVENTS, score=300.0), 244, completed=True)
        self.finish(METHOD, 1.2)
        # RND at weight 2, seed 13: training, 130 updates so far (2 whole windows)
        self.training = "grid_rnd_intrinsic-weight-2_seed-13"
        self.start(self.training, CARD_ONE_COPY)
        write_run_records(pixel_queue.run_folder(self.run_dir, self.training), metric_lines(130, score=50.0), 244)

    def start(self, name, card):
        """Claim one named unit for a worker on a card (the other pending units are held back meanwhile)."""
        held = self.run_dir / "queue/held_by_the_test"
        held.mkdir(exist_ok=True)
        others = [path for path in (self.run_dir / "queue/pending").glob("*.json") if path.stem != name]
        for path in others:
            path.rename(held / path.name)
        unit = pixel_queue.claim(self.run_dir, card["memory_mb"], claimant(card, "71"), "one copy", PROFILES["one copy"])
        for path in others:
            (held / path.name).rename(path)
        held.rmdir()
        self.assertEqual(unit["name"], name)
        return unit

    def finish(self, name, seconds_per_update):
        """Move a running unit to done, with the speed its last attempt had."""
        unit = common.read_json(unit_queue.unit_path(self.run_dir, "running", name))
        unit_queue.finish(self.run_dir, unit, {"wall_seconds": 300.0, "seconds_per_update": seconds_per_update})

    def build(self):
        """Build the report's text."""
        return report.build(self.run_dir, moment="2026-11-18T03:30:00-05:00")[0]


class NewRuns(ReportCase):
    """The report on runs with recorded game events."""

    def test_every_section_is_there_and_times_are_pacific(self):
        """The header, the contents table and the nine sections; the written time is shown in Pacific."""
        text = self.build()
        self.assertIn("Written 2026-11-18 00:30 PT", text)
        self.assertIn("Times are Pacific", text)
        titles = [line[3:] for line in text.split("\n") if line.startswith("## ")]
        self.assertEqual(titles, ["Definitions", "Fleet status", "Speed by card type",
                                  "Configurations at the compared mark", "Mean score over the run, by mark",
                                  "First-time steps of the game events", "Keys, exits and lives lost", "Score levels",
                                  "Standing checks"])
        for title in titles:
            self.assertIn(f"](#{report.anchor(title)})", text)
        self.assertIn("| [Fleet status](#fleet-status) | 2026-11-18 00:30 PT | 2026-11-18 00:30 PT |", text)

    def test_fleet_status_counts_runs_by_state_and_names_a_unit_that_changed_cards(self):
        """Two methods, their runs by state, and the unit that went from an RTX A4500 to an RTX 5080."""
        fleet = "\n".join(section(self.build(), "Fleet status"))
        self.assertIn("Units by queue folder: done 2, failed 0, pending 5, running 1.", fleet)
        self.assertIn("| RND | 4 | 2 | 1 | 0 | 0 | 1 | 0 |", fleet)
        self.assertIn("| fast last layer | 4 | 3 | 0 | 0 | 0 | 1 | 0 |", fleet)
        self.assertIn("Units that continued on another card type: 1.", fleet)
        self.assertIn(f"- `{METHOD}`: RTX A4500, then GeForce RTX 5080", fleet)

    def test_speed_has_the_five_throughput_columns_per_card_type(self):
        """Copies, seconds per iteration, total and per-copy steps per second, hours per million steps per copy."""
        speed = section(self.build(), "Speed by card type")
        header = next(line for line in speed if line.startswith("| card type"))
        for words in ("copies", "seconds per<br>iteration", "total steps<br>per second<br>(millions)",
                      "steps per<br>second per<br>copy<br>(thousands)", "hours per<br>million<br>steps per<br>copy"):
            self.assertIn(words, header)
        # the RTX 5080: one reading of 1.2 s per update, so 16,384 / 1.2 = 13,653 steps per second per copy
        self.assertIn("| GeForce RTX 5080 | 1 | 1.20 | 0.0137 | 13.7 | 0.0203 | 1<br>(0 now) |", speed)
        for line in speed:
            self.assertNotIn("grid_", line)
        # the RTX A4500: 1.4 (done), 1.6 (stopped) and 1.4 (training now), median 1.40
        self.assertIn("| RTX A4500 | 1 | 1.40<br>(1.40 to 1.60) | 0.0117 | 11.7 | 0.0237 | 3<br>(1 now) |", speed)

    def test_the_compared_mark_table_has_the_plans_measures(self):
        """At the mark of 4 windows: the mean score over the run, the capped first exit, and the rooms reached."""
        compared = section(self.build(), "Configurations at the compared mark")
        self.assertIn("Compared at the mark of 4 windows (4.00 million steps)", compared[1])
        # the method's seed left the start room in update 70 at rollout step 5: 69 * 16,384 + 6 * 128 steps
        self.assertEqual((70 - 1) * 16384 + (5 + 1) * 128, 1131264)
        method = next(line for line in compared if line.startswith("| Fast last layer<br>step size 0.03, group 16"))
        self.assertEqual(method, "| Fast last layer<br>step size 0.03, group 16<br>weight 1 | 1 / 1 / 2 | 1 | "
                                 "**300** | **1.13** | 1 of 1 | **2.0** |")
        # RND never left: its first exit counts as the mark, 4 windows of 999,424 steps
        rnd = next(line for line in compared if line.startswith("| RND<br>weight 1 "))
        self.assertEqual(rnd, "| RND<br>weight 1 | 1 / 1 / 2 | 1 | <u>100</u> | <u>4.00</u> | 0 of 1 | <u>1.0</u> |")
        # a configuration without a seed at the mark has dashes, and the RND block starts with its own row
        self.assertIn("| RND<br>weight 2 | 0 / 1 / 2 | 0 | — | — | — | — |", compared)
        self.assertLess(compared.index(method), compared.index("| **RND** |  |  |  |  |  |  |"))
        self.assertIn("mean score<br>over the<br>run ↑", "\n".join(compared))
        self.assertEqual("\n".join(compared).count("↑"), 1)

    def test_the_mean_score_is_shown_at_every_mark_with_its_seeds(self):
        """Marks of 1, 2 and 4 windows; the training run has values up to the second mark only."""
        marks = section(self.build(), "Mean score over the run, by mark")
        self.assertIn("| configuration | 1 | 2 | 4 |", marks)
        self.assertIn("| weight 1 | **100**<br>(1) | **100**<br>(1) | **100**<br>(1) |", marks)
        self.assertIn("| weight 2 | <u>50</u><br>(1) | <u>50</u><br>(1) | — |", marks)
        self.assertIn("**Fast last layer, sharing group 16**", marks)
        self.assertIn("| weight 1<br>step size 0.03 | **300**<br>(1) | **300**<br>(1) | **300**<br>(1) |", marks)
        self.assertIn("| weight 1<br>step size 0.1 | — | — | — |", marks)

    def test_first_times_shares_lives_and_score_levels(self):
        """Medians of first-time steps with the seeds that had the event; shares at the marks; lives by cause; levels."""
        text = self.build()
        first = section(text, "First-time steps of the game events")
        # key in update 37 at rollout step 101: 36 * 16,384 + 102 * 128 = 602,880 steps
        self.assertIn("| weight 1<br>step size 0.03 | 0.60<br>1/1 | 1.13<br>1/1 | 1.13<br>1/1 | 2.11<br>1/1 | "
                      "—<br>0/1 | —<br>0/1 | —<br>0/1 | —<br>0/1 |", first)
        self.assertIn("| weight 1 | —<br>0/1 | —<br>0/1 | —<br>0/1 | —<br>0/1 | —<br>0/1 | —<br>0/1 | "
                      "—<br>0/1 | —<br>0/1 |", first)
        games = section(text, "Keys, exits and lives lost")
        # window 1 (updates 1 to 61): the key from update 37 on, 25 of 122 games; no exit yet
        self.assertIn("| weight 1<br>step size 0.03 | 0.20<br>0.00 | 0.50<br>0.43 | 0.50<br>0.50 |", games)
        # three lives lost by a fall in every update: 183 per window of 0.999424 million steps
        self.assertIn("| weight 1<br>step size 0.03 | 183.1 | 0.0 | 0.0 | 0.0 |", games)
        levels = section(text, "Score levels")
        self.assertIn("| configuration | 400 | 1,000 | 2,500 |", levels)
        self.assertIn("| weight 1<br>step size 0.03 | —<br>0/1 | —<br>0/1 | —<br>0/1 |", levels)

    def test_the_standing_checks_count_what_must_be_zero(self):
        """No exit with both doors locked, no exit before the key, no value that is not finite, no silent unit."""
        checks = "\n".join(section(self.build(), "Standing checks"))
        self.assertIn("**Exits from the start room with both doors locked:** 0 in 3 runs", checks)
        self.assertIn("**Runs whose first exit came before their first counted key:** 0.", checks)
        self.assertIn("**Runs with no such door:** 0 (expected: 0)", checks)
        self.assertIn("**Values that are not finite in the record files:** 0 in 3 runs", checks)
        self.assertIn("**Units past their first update without record files:** 0", checks)
        self.assertIn("**Failed units:** 0.", checks)

    def test_a_broken_fact_is_counted_and_named(self):
        """An exit with both doors locked and an exit before the key are both listed with the unit and the seed."""
        events = {20: {"left_start_room": 3, "left_start_room_with_both_doors_locked": 3}, 90: {"key_pickup": 2}}
        write_run_records(pixel_queue.run_folder(self.run_dir, RND), metric_lines(244, events=events), 244, completed=True)
        checks = "\n".join(section(self.build(), "Standing checks"))
        self.assertIn("**Exits from the start room with both doors locked:** 1 in 3 runs", checks)
        self.assertIn(f"   - `{RND}`, seed 13: 1", checks)
        self.assertIn("**Runs whose first exit came before their first counted key:** 1. Of them 0 opened", checks)
        self.assertIn("**Runs with no such door:** 1 (expected: 0)", checks)
        self.assertIn(f"   - `{RND}`, seed 13", checks)

    def test_a_key_carried_from_the_warm_up_is_counted_apart(self):
        """A door opened and an exit in update 1, the first counted key later: counted by seed, not listed."""
        # a door at rollout step 10 and the exit at step 20 of update 1; the first counted key in update 4
        events = {1: {"door_opened": 10, "left_start_room": 20}, 4: {"key_pickup": 5}}
        write_run_records(pixel_queue.run_folder(self.run_dir, RND), metric_lines(244, events=events), 244, completed=True)
        checks = "\n".join(section(self.build(), "Standing checks"))
        self.assertIn("**Runs whose first exit came before their first counted key:** 1. Of them 1 opened a door", checks)
        self.assertIn("(by seed: seed 13: 1)", checks)
        self.assertIn("**Runs with no such door:** 0 (expected: 0)", checks)
        self.assertNotIn(f"   - `{RND}`, seed 13\n", checks + "\n")

    def test_tables_fit_a_printed_page_and_hold_no_unit_name(self):
        """Every table is at most 100 characters wide when printed, with long names wrapped onto short lines."""
        text = self.build()
        self.assertLessEqual(max(table_widths(text)), 100, table_widths(text))
        for line in text.split("\n"):
            if line.startswith("|"):
                self.assertNotIn("grid_", line)

    def test_a_pack_gives_one_run_per_copy(self):
        """A four-copy unit's record files are split by copy: each of its runs gets its own row values."""
        run_dir = self.run_dir.parent / "packs"
        description = small_description(methods=[{"method": "fast-last-layer", "pack_over": "step_size", "grid": {
            "sharing_group": [16], "intrinsic_weight": [1], "step_size": [0.01, 0.03, 0.1, 0.3]}}])
        queue_units(run_dir, build_units.study_units(description))
        with contextlib.redirect_stdout(io.StringIO()):
            name = build_units.pack_pending(run_dir, 1)[0]
        pixel_queue.claim(run_dir, 95830, claimant(dict(CARD_ONE_COPY, gpu="NVIDIA H100 NVL", memory_mb=95830)),
                          "four copies", PROFILES["four copies"])
        # copy 0 (step size 0.01) has the events; all four copies score 300 per game
        write_run_records(pixel_queue.run_folder(run_dir, name),
                          metric_lines(130, copies=4, events=METHOD_EVENTS, score=300.0), 244)
        text = report.build(run_dir)[0]
        first = section(text, "First-time steps of the game events")
        self.assertIn("| weight 1<br>step size 0.01 | 0.60<br>1/1 |", "\n".join(first))
        self.assertIn("| weight 1<br>step size 0.3 | —<br>0/1 |", "\n".join(first))
        fleet = "\n".join(section(text, "Fleet status"))
        self.assertIn("| fast last layer | 8 | 4 | 4 | 0 | 0 | 0 | 0 |", fleet)
        # every run says how many copies shared its process: 4 runs in the pack, 4 singles
        self.assertIn("| method | 1 copy | 4 copies |\n|---|---|---|\n| fast last layer | 4 | 4 |", fleet)
        self.assertEqual({run["copies"] for run in report.collect(run_dir)["runs"]}, {1, 4})

    def test_the_contents_table_marks_read_and_updated_sections(self):
        """A section is unread until marked; it is read while its text stays the same, and updated when it changes."""
        with contextlib.redirect_stdout(io.StringIO()):
            report.main(["--run-dir", str(self.run_dir)])
            page = (self.run_dir / "early_signals.md").read_text()
            self.assertIn("| [Standing checks](#standing-checks) |", page)
            self.assertEqual(page.count("\\text{unread}"), 9)
            report.main(["--run-dir", str(self.run_dir), "--mark-read", "all"])
            page = (self.run_dir / "early_signals.md").read_text()
            self.assertEqual((page.count("\\text{unread}"), page.count("| read |")), (0, 9))
            # the training run finishes two more windows: the tables that show it change, the definitions do not
            write_run_records(pixel_queue.run_folder(self.run_dir, self.training), metric_lines(244, score=50.0), 244)
            report.main(["--run-dir", str(self.run_dir)])
        page = (self.run_dir / "early_signals.md").read_text()
        contents = {line.split("|")[1].strip(): line.split("|")[4].strip()
                    for line in page.split("\n") if line.startswith("| [")}
        self.assertEqual(contents["[Definitions](#definitions)"], "read")
        self.assertIn("updated", contents["[Mean score over the run, by mark](#mean-score-over-the-run-by-mark)"])

    def test_formulas_keep_to_the_rules_for_formulas_in_markdown(self):
        """No backslash before punctuation inside a formula, no brace-underscore-brace, no formula glued to a word."""
        text = self.build()
        self.assertIsNone(re.search(r"\\[,;!{}]", text))
        self.assertNotIn("}_{", text)
        self.assertIsNone(re.search(r"[^\s(|>]\$[A-Za-z\\(]", text))
        self.assertNotIn("$$", text)


class ArchivedBaseline(ReportCase):
    """Lines in the archived baseline's format (no event fields) beside the new runs."""

    def archive(self, folder):
        """Write an archived run folder of two copies (seeds 1 and 2) under an archived data folder."""
        run = Path(folder) / "h100-serval02-card-1_final-training_seeds-1-to-2" / "run"
        run.mkdir(parents=True)
        # seed 1 gets the key in update 37 and leaves the start room in update 70; seed 2 does neither
        lines = metric_lines(300, copies=2, events={37: {"key_pickup": 0}, 70: {"left_start_room": 0}}, score=200.0,
                             with_event_fields=False)
        (run / "metrics.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
        (run / "run.json").write_text(json.dumps({
            "learner": {"n_copies": 2, "intrinsic_coefficient": 1.0, "total_timesteps": 2000000000},
            "copy_seeds": [1, 2]}))
        return Path(folder)

    def test_archived_runs_are_folded_and_shown_beside_rnd(self):
        """--fold-archive makes the record files; the report shows the archived seeds as their own RND row."""
        with contextlib.redirect_stdout(io.StringIO()):
            folded = report.fold_archive(self.run_dir, self.archive(self.run_dir.parent / "archive"))
        self.assertEqual(folded, ["h100-serval02-card-1_final-training_seeds-1-to-2"])
        records = report.read_records(self.run_dir / "archived_baseline" / folded[0])
        self.assertIs(records["first_times"]["has_event_fields"], False)
        self.assertEqual(len(records["windows"]), 4)
        text = self.build()
        self.assertIn("Beside them: 2 runs of the archived baseline", text)
        compared = section(text, "Configurations at the compared mark")
        # two archived seeds at the mark of 4 windows; seed 1 left in update 70 (known to the end of that update)
        archived = next(line for line in compared if line.startswith("| RND, archived baseline<br>weight 1"))
        exit_step = 70 * 16384
        capped = (exit_step + 4 * WINDOW_STEPS) / 2 / 1e6
        self.assertEqual(archived, f"| RND, archived baseline<br>weight 1 | 2 / 2 / 2 | 2 | <u>200</u> | "
                                   f"<u>{capped:.2f}</u> | 1 of 2 | <u>1.5</u> |")
        marks = section(text, "Mean score over the run, by mark")
        self.assertIn("| weight 1<br>archived baseline | **200**<br>(2) | **200**<br>(2) | **200**<br>(2) |", marks)

    def test_events_the_archive_never_recorded_are_dashes(self):
        """Key and exit are known for the archive (to the end of their update); door, kill and items are not."""
        with contextlib.redirect_stdout(io.StringIO()):
            report.fold_archive(self.run_dir, self.archive(self.run_dir.parent / "archive"))
        text = self.build()
        first = section(text, "First-time steps of the game events")
        key_step, exit_step = 37 * 16384 / 1e6, 70 * 16384 / 1e6
        # two seeds, one with the event: the median seed has not had it
        self.assertIn("| weight 1<br>archived baseline | —<br>1/2 | — | —<br>1/2 | — | — | — | — | — |", first)
        self.assertEqual((f"{key_step:.2f}", f"{exit_step:.2f}"), ("0.61", "1.15"))
        games = section(text, "Keys, exits and lives lost")
        self.assertIn("| weight 1<br>archived baseline | — | — | — |", games)
        self.assertIn("| weight 1<br>archived baseline | — | — | — | — |", games)
        checks = "\n".join(section(text, "Standing checks"))
        self.assertIn("0 in 3 runs with recorded events", checks)
        self.assertIn("0 in 5 runs with records", checks)

    def test_the_real_archives_first_lines_are_accepted(self):
        """The first 122 lines of an archived metrics file fold into record files the report reads."""
        if ARCHIVE_DATA is None:
            self.skipTest("the archived baseline is not on this machine")
        source = ARCHIVE_DATA / "a4500-jaguar03_final-training_seed-5/run"
        copy = self.run_dir.parent / "real" / "a4500-jaguar03_final-training_seed-5" / "run"
        copy.mkdir(parents=True)
        with (source / "metrics.jsonl").open() as stream:
            (copy / "metrics.jsonl").write_text("".join(next(stream) for _ in range(122)))
        (copy / "run.json").write_text((source / "run.json").read_text())
        with contextlib.redirect_stdout(io.StringIO()):
            report.fold_archive(self.run_dir, copy.parents[1])
        text = self.build()
        self.assertIn("| weight 1<br>archived baseline |", text)
        self.assertIn("Beside them: 1 runs of the archived baseline", text)


if __name__ == "__main__":
    unittest.main()
