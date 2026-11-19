"""The remote fleet's local parts (no server is contacted), the check of a returned unit, and the smoke job's rules."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from pixel_test_support import (ARCHIVE_DATA, PIXEL, build_units, common, load_file, metric_lines, own, pixel_queue,
                                queue_units, small_description, unit_queue, write_run_records)
from campaign_imports import grid

remote = own("remote_fleet")
fleet = grid("remote_fleet")
smoke_check = load_file("pixel_smoke_check", PIXEL / "smoke_check.py")
PROFILES = pixel_queue.read_profiles()
ADAPTED = ("WORKER_TEMPLATE", "REMOTE_FOLDERS", "worker_script", "prepare", "verify_unit", "status_script",
           "parse_status", "result_block", "bring_back", "JOB_PREFIX", "RECOVERY_LIMIT")
FOUR_STEP_SIZES = [{"method": "fast-last-layer", "pack_over": "step_size", "grid": {
    "sharing_group": [16], "intrinsic_weight": [1], "step_size": [0.01, 0.03, 0.1, 0.3]}}]
TWO_WEIGHTS = [{"method": "fast-last-layer", "pack_over": "step_size", "grid": {
    "sharing_group": [16], "intrinsic_weight": [1, 2], "step_size": [0.01, 0.03, 0.1, 0.3]}}]


class FleetCase(unittest.TestCase):
    """A temporary run folder with a server table; the grid campaign's module is put back as it was afterwards."""

    def setUp(self):
        """Adapt the grid campaign's remote fleet for a temporary run folder and write its server table."""
        held = {name: getattr(fleet, name) for name in ADAPTED}
        self.addCleanup(lambda: [setattr(fleet, name, value) for name, value in held.items()])
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.run_dir = Path(folder.name) / "2026-11-18-02-00_pixel-montezuma_test-run"
        (self.run_dir / "slurm").mkdir(parents=True)
        remote.adapt(self.run_dir)
        with contextlib.redirect_stdout(io.StringIO()):
            remote.init_table(self.run_dir)
        self.table = common.read_json(fleet.table_path(self.run_dir))

    def pack(self, description=None):
        """Queue the four step sizes of one group, weight and seed and pack them; returns the pack's name."""
        queue_units(self.run_dir, build_units.study_units(description or small_description(
            seeds={"first": 13, "count": 1}, methods=FOUR_STEP_SIZES)))
        with contextlib.redirect_stdout(io.StringIO()):
            return build_units.pack_pending(self.run_dir, 1)[0]


class ServerTable(FleetCase):
    """The table a run folder starts with."""

    def test_every_server_starts_not_ready_with_its_request(self):
        """Gautschi, Gilbreth, ACES, DeltaAI and MIT, each not ready, each with the options the plan names."""
        self.assertEqual(list(self.table), ["gautschi", "gilbreth", "aces", "deltaai", "mit"])
        self.assertTrue(all(entry["ready"] is False and entry["not_ready_reason"] for entry in self.table.values()))
        gautschi = self.table["gautschi"]["pools"]["ai-h100"]
        self.assertEqual(gautschi["sbatch_options"], ["-p", "ai", "-q", "normal", "-A", "dsimchil", "-N1", "-n14",
                                                      "--gpus-per-node=1"])
        self.assertEqual((gautschi["time_limit"], gautschi["max_jobs"], gautschi["execution_profile"]),
                         ("1-00:00:00", 4, "four copies"))
        for pool in self.table["gilbreth"]["pools"].values():
            options = pool["sbatch_options"]
            self.assertEqual((options[options.index("-q") + 1], options[options.index("-A") + 1], pool["time_limit"]),
                             ("standby", "dsmichil", "04:00:00"))
        aces = self.table["aces"]
        for option in ("-p", "gpu", "--gres=gpu:h100:1", "--cpus-per-task=8", "--mem=39G"):
            self.assertIn(option, aces["pools"]["gpu-h100"]["sbatch_options"])
        self.assertEqual(aces["ssh"][:4] + aces["ssh"][-1:], ["ssh", "-o", "BatchMode=yes", "portal05", "aces"])
        self.assertEqual(aces["check"][-3:], ["-O", "check", "aces"])
        self.assertEqual((self.table["deltaai"]["ssh"][3], self.table["deltaai"]["ssh"][-1]), ("portal04", "deltaai"))
        self.assertIn("ARM", self.table["deltaai"]["not_ready_reason"])
        self.assertTrue(self.table["gautschi"]["remote_run"].endswith("/2026-11-18-02-00_pixel-montezuma_test-run"))
        self.assertEqual(fleet.ledger_path(self.run_dir, "gautschi").name, "submitted_jobids_gautschi.txt")

    def test_mit_asks_for_one_typed_h200_with_the_card_guard_and_never_logs_in(self):
        """Both MIT pools: one typed H200, at most 4 cores, no --gres, requeue, 2 days; only the shared connection."""
        mit = self.table["mit"]
        self.assertEqual(list(mit["pools"]), ["advanced-h200", "preemptable-h200"])
        for pool in mit["pools"].values():
            options = pool["sbatch_options"]
            self.assertTrue({"--gpus=h200:1", "--nodes=1", "--ntasks=1", "--cpus-per-task=4", "--requeue",
                             "--open-mode=append"} <= set(options))
            self.assertFalse(any(option.startswith("--gres") or option == "--exclusive" for option in options))
            self.assertIn("node3401-c1", options[options.index("-x") + 1])
            self.assertEqual((pool["time_limit"], pool["max_jobs"], pool["execution_profile"], pool["card_guard"]),
                             ("2-00:00:00", 4, "eight copies", True))
        advanced = mit["pools"]["advanced-h200"]["sbatch_options"]
        self.assertTrue({"--partition=mit_normal_gpu", "--account=mit_amf_advanced_gpu",
                         "--qos=mit_amf_advanced_gpu"} <= set(advanced))
        self.assertIn("--partition=mit_preemptable", mit["pools"]["preemptable-h200"]["sbatch_options"])
        # a command goes through the connection held on this machine or fails there; it never starts a login
        self.assertEqual(mit["ssh"], ["ssh", "-o", "BatchMode=yes", "-o", "ControlMaster=no", "-o",
                                      "ProxyCommand=false", "mit"])
        self.assertEqual(mit["check"], ["ssh", "-O", "check", "mit"])
        self.assertEqual({pool: limit["units_at_once"] for pool, limit in mit["hand_over"]["pools"].items()},
                         {"advanced-h200": 16, "preemptable-h200": 0})

    def test_the_mit_worker_job_holds_the_card_guard_before_any_unit(self):
        """The guard asks for one card, requeues on a wrong count, counts cards with jax, and stands before the worker."""
        entry = dict(self.table["mit"], code="/home/shuzel/rnd_pixel/code/source_abc/10_jax_exploration_platform")
        for pool_name, pool in entry["pools"].items():
            text = remote.worker_script("mit", entry, pool_name, pool)
            self.assertEqual(subprocess.run(["bash", "-n"], input=text, text=True, capture_output=True).returncode, 0)
            self.assertNotIn("@", text)
            for line in ("NEED_CARDS=1", "GUARD_ACTION=${GUARD_ACTION:-requeue}", 'import jax; print(len(jax.devices("gpu")))',
                         'scontrol requeue "$SLURM_JOB_ID"'):
                self.assertIn(line, text)
            self.assertNotIn("import torch", text)
            self.assertLess(text.index("[card guard] ok"), text.index("worker.py"))
        # Gautschi's jobs keep no guard
        gautschi = dict(self.table["gautschi"], code="/scratch/code")
        self.assertIn("# (no card guard on this server)",
                      remote.worker_script("gautschi", gautschi, "ai-h100", gautschi["pools"]["ai-h100"]))

    def test_a_table_is_not_written_over_and_a_tick_leaves_servers_that_are_not_ready(self):
        """init-table refuses a folder that has its table; hand-over-tick contacts no server that is not ready."""
        with self.assertRaises(SystemExit):
            remote.init_table(self.run_dir)
        common.write_json(self.run_dir / "source.json", {"commit": "abc"})
        self.pack()
        output = io.StringIO()
        with mock.patch.object(fleet, "call", side_effect=AssertionError("a server was contacted")), \
                contextlib.redirect_stdout(output):
            remote.hand_over_tick(self.run_dir, self.table, dry_run=False)
        self.assertEqual(output.getvalue().count("not part of this tick"), 5)

    def test_the_smoke_verdict_needs_every_pool(self):
        """A server is ready only when each of its pools has a passed smoke result."""
        gilbreth = self.table["gilbreth"]
        passed = {"passed": True, "checks": []}
        self.assertEqual(remote.smoke_verdict(gilbreth, {"standby-a100-80gb": passed}),
                         (False, "pool standby-a30 has no smoke result yet"))
        failed = {"passed": False, "checks": [{"name": "the asset pack equals UVA's, file by file", "passed": False}]}
        ready, reason = remote.smoke_verdict(gilbreth, {"standby-a100-80gb": passed, "standby-a30": failed,
                                                       "standby-a10": passed})
        self.assertEqual((ready, "the asset pack" in reason), (False, True))
        self.assertEqual(remote.smoke_verdict(gilbreth, {"standby-a100-80gb": passed, "standby-a30": passed,
                                                     "standby-a10": passed}),
                         (True, None))


class JobScripts(FleetCase):
    """The worker job and the smoke job of a pool."""

    def scripts(self, server, pool_name):
        """Both job scripts of one pool, as prepare writes them."""
        entry = dict(self.table[server], code="/scratch/code/source_abc/10_jax_exploration_platform")
        pool = entry["pools"][pool_name]
        return (remote.worker_script(server, entry, pool_name, pool),
                remote.job_script(remote.SMOKE_TEMPLATE, server, entry, pool_name, pool))

    def test_the_worker_job_starts_this_folders_worker_with_the_servers_assets(self):
        """bash accepts both scripts; every placeholder is filled; the environment is the design's."""
        worker_text, smoke_text = self.scripts("gautschi", "ai-h100")
        for text in (worker_text, smoke_text):
            self.assertEqual(subprocess.run(["bash", "-n"], input=text, text=True, capture_output=True).returncode, 0)
            self.assertNotIn("@", text)
            self.assertIn("XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true", text)
            self.assertIn("XDG_DATA_HOME=/home/liu4869/rnd_pixel/assets/"
                          "jaxtari_pixel_study_assets_montezuma_room_geometry_from_png", text)
            self.assertIn("unset LD_LIBRARY_PATH", text)
        self.assertIn('exec "$PY" "$CODE/scripts/pixel_sweep_campaign/worker.py" --run-dir "$RUN" --remote-records '
                      "--server gautschi", worker_text)
        self.assertIn("--minimum-seconds 1800 --margin-seconds 900", worker_text)
        self.assertIn("smoke_check.py", smoke_text)
        self.assertIn("--pool ai-h100 --profile four-copies --updates 20", smoke_text)
        self.assertNotIn("worker.py", smoke_text)

    def test_a_check_pool_sets_the_workers_least_training_time_and_margin(self):
        """A pool of 20-minute jobs gives its worker 240 s for each, so a short check trains; bash accepts the script."""
        entry = dict(self.table["mit"], code="/home/shuzel/code/source_abc/10_jax_exploration_platform")
        pool = dict(entry["pools"]["advanced-h200"], time_limit="00:20:00", worker_minimum_seconds=240,
                    worker_margin_seconds=240)
        text = remote.worker_script("mit", entry, "eight-copy-check", pool)
        self.assertEqual(subprocess.run(["bash", "-n"], input=text, text=True, capture_output=True).returncode, 0)
        self.assertIn("--minimum-seconds 240 --margin-seconds 240", text)
        self.assertNotIn("@", text)

    def test_two_run_folders_never_share_a_job_name_prefix(self):
        """The prefix comes from the folder's name and says the job belongs to the pixel sweep."""
        one, two = remote.job_prefix("/runs/folder-one"), remote.job_prefix("/elsewhere/folder-two")
        self.assertEqual(one, remote.job_prefix("/other/place/folder-one"))
        self.assertNotEqual(one, two)
        self.assertRegex(one, r"^pixelfleet-[0-9a-f]{8}$")
        self.assertEqual(fleet.JOB_PREFIX, remote.job_prefix(self.run_dir))

    def test_the_status_answer_gives_each_units_updates(self):
        """The status script reads progress.json on the server; the answer's progress lines are parsed."""
        entry = dict(self.table["gautschi"], code="/scratch/code")
        script = remote.status_script("gautschi", entry, ["unit-a", "unit-b"])
        self.assertEqual(subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True).returncode, 0)
        self.assertIn('folder="data/$unit/run"', script)
        answer = (f"{fleet.MARK}queue\nrunning unit-a\npending unit-b\n{fleet.MARK}squeue_exit 0\n"
                  f"{fleet.MARK}progress\nunit-a update=4120 completed=no checkpoint=yes\n"
                  f"unit-b update= completed=no checkpoint=no\n{fleet.MARK}estimates\n901 7200\n902 unknown\n{fleet.MARK}end\n")
        status = remote.parse_status(answer)
        self.assertEqual(status["progress"], {"unit-a": {"update": 4120, "completed": False, "checkpoint": True},
                                              "unit-b": {"update": 0, "completed": False, "checkpoint": False}})
        self.assertEqual(status["start_in_seconds"], {"901": 7200, "902": None})
        self.assertEqual(status["queue"]["running"], ["unit-a"])


class References(unittest.TestCase):
    """The archived lines a smoke job compares its first default updates with."""

    def test_the_lines_of_several_archived_runs_join_copy_by_copy(self):
        """Lists are joined in the runs' order; numbers shared by all copies stay; the write time is the first run's."""
        four = [{"update": 1, "unix": 5.0, "loss": [0.6, 0.5, 0.5, 0.6], "rooms_visited": [[True]] * 4,
                 "steps_per_policy": 16384}]
        one = [{"update": 1, "unix": 7.0, "loss": [0.7], "rooms_visited": [[False]], "steps_per_policy": 16384}]
        self.assertEqual(remote.joined_lines([four, one]),
                         [{"update": 1, "unix": 5.0, "loss": [0.6, 0.5, 0.5, 0.6, 0.7],
                           "rooms_visited": [[True]] * 4 + [[False]], "steps_per_policy": 16384}])
        with self.assertRaises(ValueError):
            remote.joined_lines([four, [dict(one[0], update=2)]])

    @unittest.skipIf(ARCHIVE_DATA is None, "the archived baseline is not on this machine")
    def test_the_eight_copy_reference_holds_seeds_1_to_8(self):
        """Seeds 1 to 4 from the archived four-copy run, then seeds 5 to 8 from the one-copy runs, 20 lines."""
        text, seeds = remote.archive_reference(PROFILES["eight copies"])
        lines = [json.loads(line) for line in text.splitlines()]
        self.assertEqual((seeds, len(lines), [line["update"] for line in lines][:3]), (list(range(1, 9)), 20, [1, 2, 3]))
        self.assertTrue(all(len(line["loss"]) == 8 for line in lines))
        four, _ = remote.archive_reference(PROFILES["four copies"])
        self.assertEqual([line["loss"] for line in map(json.loads, four.splitlines())],
                         [line["loss"][:4] for line in lines])


class HandOver(FleetCase):
    """Which units a pool gets."""

    def test_an_eight_copy_pool_gets_eight_copy_packs_only_and_a_check_pool_is_handed_over_by_hand(self):
        """MIT's pools take the eight-copy pack, not the four-copy one; `hand-over` hands a check pool one unit."""
        queue_units(self.run_dir, build_units.study_units(small_description(
            seeds={"first": 13, "count": 2}, methods=TWO_WEIGHTS)))
        with contextlib.redirect_stdout(io.StringIO()):
            four = build_units.pack_pending(self.run_dir, 1, match="seed-13")
            eight = build_units.pack_pending(self.run_dir, 1, match="seed-14", size=8)
            self.table["mit"]["ready"] = True
            self.assertEqual(remote.hand_over(self.run_dir, self.table, "mit", "advanced-h200", dry_run=True), eight)
            self.assertEqual(remote.hand_over(self.run_dir, self.table, "gautschi", "ai-h100", dry_run=True), four)
            # a pool outside hand_over.pools: the command names the limit and the text a unit's name holds
            self.table["mit"]["pools"]["eight-copy-check"] = dict(self.table["mit"]["pools"]["advanced-h200"],
                                                                  time_limit="00:20:00", max_jobs=1)
            self.assertEqual(remote.hand_over(self.run_dir, self.table, "mit", "eight-copy-check", dry_run=True,
                                              units_at_once=1, match="seed-14"), eight)
            self.assertEqual(remote.hand_over(self.run_dir, self.table, "mit", "eight-copy-check", dry_run=True,
                                              units_at_once=1, match="seed-15"), [])
            with self.assertRaises(KeyError):
                remote.hand_over(self.run_dir, self.table, "mit", "eight-copy-check", dry_run=True)

    def test_a_four_copy_pool_gets_packs_only_in_claim_order_up_to_its_limit(self):
        """Singles stay on UVA; a dry run names the pack and moves nothing."""
        description = small_description(methods=[
            {"method": "rnd", "pack_over": "intrinsic_weight", "grid": {"intrinsic_weight": [1, 2]}}] + FOUR_STEP_SIZES)
        queue_units(self.run_dir, build_units.study_units(description))
        with contextlib.redirect_stdout(io.StringIO()):
            packs = build_units.pack_pending(self.run_dir, 2)
            self.table["gautschi"]["hand_over"]["pools"]["ai-h100"]["units_at_once"] = 1
            chosen = remote.hand_over(self.run_dir, self.table, "gautschi", "ai-h100", dry_run=True)
            self.assertEqual(chosen, packs[:1])
            self.table["gautschi"]["hand_over"]["pools"]["ai-h100"]["units_at_once"] = 8
            self.assertEqual(remote.hand_over(self.run_dir, self.table, "gautschi", "ai-h100", dry_run=True), packs)
            # Gilbreth's one-copy pool takes singles, its four-copy pool packs
            self.table["gilbreth"]["hand_over"]["pools"]["standby-a30"]["units_at_once"] = 3
            singles = remote.hand_over(self.run_dir, self.table, "gilbreth", "standby-a30", dry_run=True)
        self.assertEqual(singles, ["grid_rnd_intrinsic-weight-1_seed-13", "grid_rnd_intrinsic-weight-2_seed-13",
                                   "grid_rnd_intrinsic-weight-1_seed-14"])
        self.assertEqual(unit_queue.counts(self.run_dir)["pending"], 6)
        self.assertFalse((self.run_dir / "queue/remote_gautschi").exists())

    def test_a_hand_over_sends_the_record_the_specification_and_a_begun_run_folder(self):
        """The unit leaves pending for queue/remote_<server>; the copy holds its files laid out as on the server."""
        name = self.pack()
        folder = pixel_queue.run_folder(self.run_dir, name)
        write_run_records(folder, metric_lines(61, copies=4), 244)
        (folder / "checkpoint.pkl").write_bytes(b"state")
        sent = {}

        def fake_rsync(run_dir, table, server, arguments):
            """Stand in for the copy to the server: note what the local tree holds."""
            tree = Path(arguments[0])
            sent.update({str(path.relative_to(tree)): path.stat().st_size for path in tree.rglob("*") if path.is_file()})

        with mock.patch.object(fleet, "rsync", side_effect=fake_rsync), \
                mock.patch.object(fleet, "remote", return_value=f"{fleet.MARK}moved place {name} no\n{fleet.MARK}end\n"), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(remote.hand_over(self.run_dir, self.table, "gautschi", "ai-h100", dry_run=False), [name])
        self.assertTrue((self.run_dir / "queue/remote_gautschi" / f"{name}.json").exists())
        self.assertEqual(unit_queue.counts(self.run_dir)["pending"], 0)
        assignment = fleet.load_assignment(self.run_dir)[name]
        self.assertEqual((assignment["server"], assignment["pool"], assignment["state"]), ("gautschi", "ai-h100", "staged"))
        self.assertEqual({path for path in sent if not path.startswith("data/")},
                         {f"specifications/{name}.json", f"queue/incoming/{name}.json"})
        self.assertTrue({f"data/{name}/run/{file_name}" for file_name in
                         ("checkpoint.pkl", "metrics.jsonl", "progress.json", "game_first_times.json")} <= set(sent))


class Ticks(FleetCase):
    """A whole tick and the smoke commands, with every call to a server replaced."""

    def test_a_tick_hands_packs_to_a_ready_server_and_submits_jobs_for_them(self):
        """Two packs pending and room for eight: both are handed over and two worker jobs are asked for."""
        description = small_description(methods=FOUR_STEP_SIZES)
        queue_units(self.run_dir, build_units.study_units(description))
        with contextlib.redirect_stdout(io.StringIO()):
            packs = build_units.pack_pending(self.run_dir, 2)
        common.write_json(self.run_dir / "source.json", {"commit": "abc"})
        self.table["gautschi"].update(ready=True, prepared_commit="abc", code="/scratch/code")
        status = remote.parse_status(f"{fleet.MARK}queue\n{fleet.MARK}squeue_exit 0\n{fleet.MARK}end\n")
        submitted, copied = [], []
        with mock.patch.object(fleet, "tick_server", return_value=(status, [])), \
                mock.patch.object(fleet, "rsync", side_effect=lambda run, table, name, arguments: copied.append(arguments)), \
                mock.patch.object(fleet, "remote", return_value=f"{fleet.MARK}end\n"), \
                mock.patch.object(fleet, "submit", side_effect=lambda run, table, args: submitted.append(args)), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            remote.hand_over_tick(self.run_dir, self.table, dry_run=False)
        self.assertEqual(sorted(path.stem for path in (self.run_dir / "queue/remote_gautschi").glob("*.json")), sorted(packs))
        self.assertEqual([(args.server, args.pool, args.jobs) for args in submitted], [("gautschi", "ai-h100", 2)])
        # the last copy of the tick brings the small record files of the units the server holds
        self.assertIn("--include=*/run/game_windows.jsonl", copied[-1])
        self.assertNotIn("--include=*/run/checkpoint.pkl", copied[-1])
        self.assertTrue(copied[-1][-1].endswith("slurm/remote_live/gautschi/"))
        self.assertIn("handed over at this tick", output.getvalue())
        self.assertEqual(output.getvalue().count("not part of this tick"), 4)

    def test_mit_holds_its_units_through_the_first_pool_and_both_pools_get_jobs(self):
        """Five eight-copy packs: the first pool takes all five (the server-wide limit 16), asks for its 4 jobs; the
        second pool takes no unit at this tick and asks for jobs only for units no job will take at the next."""
        queue_units(self.run_dir, build_units.study_units(small_description(
            seeds={"first": 13, "count": 5}, methods=TWO_WEIGHTS)))
        with contextlib.redirect_stdout(io.StringIO()):
            packs = build_units.pack_pending(self.run_dir, 5, size=8)
        common.write_json(self.run_dir / "source.json", {"commit": "abc"})
        self.table["mit"].update(ready=True, prepared_commit="abc", code="/home/shuzel/code")
        status = remote.parse_status(f"{fleet.MARK}queue\n{fleet.MARK}squeue_exit 0\n{fleet.MARK}end\n")
        submitted = []
        with mock.patch.object(fleet, "tick_server", return_value=(status, [])), \
                mock.patch.object(fleet, "rsync"), mock.patch.object(fleet, "remote", return_value=f"{fleet.MARK}end\n"), \
                mock.patch.object(fleet, "submit", side_effect=lambda run, table, args: submitted.append(args)), \
                contextlib.redirect_stdout(io.StringIO()):
            remote.hand_over_tick(self.run_dir, self.table, dry_run=False)
        self.assertEqual(sorted(path.stem for path in (self.run_dir / "queue/remote_mit").glob("*.json")), sorted(packs))
        self.assertEqual({unit["pool"] for unit in fleet.load_assignment(self.run_dir).values()}, {"advanced-h200"})
        self.assertEqual([(args.server, args.pool, args.jobs) for args in submitted], [("mit", "advanced-h200", 4)])
        # the next tick reads the five units waiting on the server and the first pool's four waiting jobs
        status = remote.parse_status(
            f"{fleet.MARK}queue\n" + "".join(f"pending {pack}\n" for pack in packs) + f"{fleet.MARK}jobs\n"
            + "".join(f"{900 + index}|PENDING|Priority|{fleet.JOB_PREFIX}-mit-advanced-h200-20261118-193000-{index}|0:00|"
                      f"mit_normal_gpu\n" for index in range(1, 5))
            + f"{fleet.MARK}squeue_exit 0\n{fleet.MARK}estimates\n" + "".join(f"{900 + index} 60\n" for index in range(1, 5))
            + f"{fleet.MARK}end\n")
        submitted.clear()
        with mock.patch.object(fleet, "tick_server", return_value=(status, sorted(packs))), \
                mock.patch.object(fleet, "rsync"), mock.patch.object(fleet, "remote", return_value=f"{fleet.MARK}end\n"), \
                mock.patch.object(fleet, "submit", side_effect=lambda run, table, args: submitted.append(args)), \
                contextlib.redirect_stdout(io.StringIO()):
            remote.hand_over_tick(self.run_dir, self.table, dry_run=False)
        self.assertEqual([(args.server, args.pool, args.jobs) for args in submitted], [("mit", "preemptable-h200", 4)])

    def test_an_unfinished_unit_comes_back_with_its_run_folder(self):
        """Checkpoint and metrics from the server replace UVA's run folder; the server's record is taken over."""
        name = self.pack()
        held = self.run_dir / "queue/remote_gautschi"
        held.mkdir()
        (self.run_dir / "queue/pending" / f"{name}.json").rename(held / f"{name}.json")
        common.write_json(fleet.assignment_path(self.run_dir), {name: {"server": "gautschi", "state": "staged"}})
        record = dict(common.read_json(held / f"{name}.json"), attempts=1, execution_profile="four copies",
                      minimum_copy_number=3)

        def fake_rsync(run_dir, table, server, arguments):
            """Stand in for the relative copy from the server: the unit's record and its run folder arrive."""
            self.assertIn("--relative", arguments)
            self.assertTrue(arguments[-2].endswith(f"/data/./{name}/run"))
            arrived = Path(arguments[-1])
            common.write_json(arrived / f"{name}.json", record)
            (arrived / name / "run").mkdir(parents=True)
            for file_name in ("checkpoint.pkl", "metrics.jsonl", "progress.json"):
                (arrived / name / "run" / file_name).write_text("from the server")

        with mock.patch.object(fleet, "rsync", side_effect=fake_rsync), contextlib.redirect_stdout(io.StringIO()):
            remote.bring_back(self.run_dir, self.table, "gautschi", [(name, True)], "take-back")
        back = common.read_json(unit_queue.unit_path(self.run_dir, "pending", name))
        self.assertEqual((back["attempts"], back["execution_profile"], back["minimum_copy_number"]),
                         (1, "four copies", 3))
        self.assertEqual((pixel_queue.run_folder(self.run_dir, name) / "checkpoint.pkl").read_text(), "from the server")
        self.assertEqual(fleet.load_assignment(self.run_dir)[name]["state"], "returned")

    def test_a_unit_that_never_started_on_the_server_comes_back_without_a_run_folder(self):
        """With no run folder and no record on the server, the unit returns to pending and UVA's state stays."""
        name = self.pack()
        held = self.run_dir / "queue/remote_gautschi"
        held.mkdir()
        (self.run_dir / "queue/pending" / f"{name}.json").rename(held / f"{name}.json")
        common.write_json(fleet.assignment_path(self.run_dir), {name: {"server": "gautschi", "state": "staged"}})
        with mock.patch.object(fleet, "rsync", return_value=None), contextlib.redirect_stdout(io.StringIO()) as out:
            remote.bring_back(self.run_dir, self.table, "gautschi", [(name, True)], "take-back")
        self.assertTrue(unit_queue.unit_path(self.run_dir, "pending", name).exists())
        self.assertFalse((pixel_queue.run_folder(self.run_dir, name) / "checkpoint.pkl").exists())
        self.assertEqual(fleet.load_assignment(self.run_dir)[name]["state"], "returned")

    def test_the_smoke_commands_submit_one_job_per_pool_and_read_the_verdict(self):
        """The submission script is valid bash and names every pool's script; a passed verdict makes the server ready."""
        entry = self.table["gilbreth"]
        script = remote.smoke_script(self.run_dir, "gilbreth", entry, list(entry["pools"]), "20261118-020000")
        self.assertEqual(subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True).returncode, 0)
        self.assertIn("-t 00:20:00 -p a100-80gb -q standby -A dsmichil", script)
        self.assertIn("/smoke_job_standby-a30.sh", script)
        self.assertIn("/smoke_job_standby-a10.sh", script)
        self.assertIn("slurm/submitted_jobids_gilbreth.txt", script)
        self.assertRegex(script, r"-J pixelfleet-[0-9a-f]{8}-gilbreth-standby-a100-80gb-smoke-20261118-020000")
        result = {"passed": True, "job_id": "77", "device": "NVIDIA H100 80GB HBM3",
                  "written_at": "2026-11-18T03:00:00-05:00", "speed": {"copies": 4},
                  "checks": [{"name": "jax computes on the card", "passed": True, "detail": "backend gpu"}],
                  "speed_lines": ["default: copies 4, 2.08 s per iteration"]}
        answer = f"\n{fleet.MARK}result ai-h100\n{json.dumps(result, indent=2)}\n\n{fleet.MARK}end\n"
        output = io.StringIO()
        with mock.patch.object(fleet, "remote", return_value=answer), contextlib.redirect_stdout(output):
            self.assertTrue(remote.smoke_result(self.run_dir, self.table, "gautschi"))
        table = common.read_json(fleet.table_path(self.run_dir))
        self.assertEqual((table["gautschi"]["ready"], "not_ready_reason" in table["gautschi"]), (True, False))
        self.assertEqual(table["gautschi"]["smoke"]["ai-h100"]["job_id"], "77")
        self.assertIn("gautschi, pool ai-h100: PASSED (job 77, NVIDIA H100 80GB HBM3, written 2026-11-18 00:00 PT)",
                      output.getvalue())
        # a server whose pool has no verdict stays not ready
        with mock.patch.object(fleet, "remote", return_value=f"{fleet.MARK}result gpu-h100\n\n{fleet.MARK}end\n"), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(remote.smoke_result(self.run_dir, self.table, "aces"))
        self.assertIn("no smoke result yet", common.read_json(fleet.table_path(self.run_dir))["aces"]["not_ready_reason"])

    def test_a_held_back_pool_takes_units_again_only_after_its_smoke_job_passed(self):
        """MIT ready on its Advanced pool; the preemptable pool comes back when its verdict passed, never on a failed one."""
        mit = self.table["mit"]
        mit["pools_held_back"] = {"preemptable-h200": dict(mit["pools"].pop("preemptable-h200"),
                                                           units_at_once_when_back=0)}
        del mit["hand_over"]["pools"]["preemptable-h200"]
        mit.update(ready=True)
        result = {"job_id": "29603828", "device": "NVIDIA H200 NVL", "written_at": "2026-11-18T23:40:00-05:00",
                  "speed": {"copies": 4}, "speed_lines": [],
                  "checks": [{"name": "jax computes on the card", "passed": False, "detail": "no card"}]}

        def answer(passed):
            """The server's answer for the held-back pool alone."""
            return f"{fleet.MARK}result preemptable-h200\n{json.dumps(dict(result, passed=passed))}\n{fleet.MARK}end\n"

        sent = []
        with mock.patch.object(fleet, "remote", side_effect=lambda run, table, name, script: sent.append(script)
                               or answer(False)), mock.patch.object(remote, "prepare") as prepare, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(remote.held_pool_result(self.run_dir, self.table, "mit", "preemptable-h200"))
        self.assertIn("smoke/result_preemptable-h200.json", sent[0])
        self.assertNotIn("result_advanced-h200", sent[0])
        table = common.read_json(fleet.table_path(self.run_dir))
        self.assertEqual((list(table["mit"]["pools"]), table["mit"]["ready"]), (["advanced-h200"], True))
        prepare.assert_not_called()
        with mock.patch.object(fleet, "remote", return_value=answer(True)), \
                mock.patch.object(remote, "prepare") as prepare, contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(remote.held_pool_result(self.run_dir, self.table, "mit", "preemptable-h200"))
        table = common.read_json(fleet.table_path(self.run_dir))
        self.assertEqual(sorted(table["mit"]["pools"]), ["advanced-h200", "preemptable-h200"])
        self.assertNotIn("units_at_once_when_back", table["mit"]["pools"]["preemptable-h200"])
        self.assertEqual((table["mit"]["hand_over"]["pools"]["preemptable-h200"], table["mit"]["pools_held_back"],
                          table["mit"]["ready"], table["mit"]["smoke"]["preemptable-h200"]["job_id"]),
                         ({"units_at_once": 0}, {}, True, "29603828"))
        prepare.assert_called_once()
        # a pool that is not held back is refused
        with self.assertRaises(SystemExit):
            remote.held_pool_result(self.run_dir, self.table, "mit", "preemptable-h200")

    def test_the_first_update_is_decided_again_from_the_jobs_own_lines(self):
        """A job that failed only the archive check by its frozen code passes when its lines meet the rule."""
        reference = [{"update": 1, "unix": 5.0, "policy_loss": [-1.5e-5], "episode_count": [25]},
                     {"update": 2, "unix": 6.0, "policy_loss": [-1.4e-5], "episode_count": [25]}]
        lines = [dict(line, unix=9.0, policy_loss=[line["policy_loss"][0] + 5e-8]) for line in reference]
        written = {"passed": False, "job_id": "12432866", "device": "NVIDIA A30", "written_at": "2026-11-18T16:01:00-05:00",
                   "speed": {"copies": 1}, "speed_lines": [],
                   "checks": [{"name": "jax computes on the card", "passed": True, "detail": "backend gpu"},
                              {"name": remote.ARCHIVE_CHECK, "passed": False, "detail": "3.33e-03 (limit 1e-03)"}]}
        self.table["gilbreth"]["pools"] = {"standby-a30": self.table["gilbreth"]["pools"]["standby-a30"]}

        def answer(default_lines):
            """The answer of the server: verdict, the job's default lines, the reference."""
            return (f"{fleet.MARK}result standby-a30\n{json.dumps(written, indent=2)}\n"
                    f"{fleet.MARK}default standby-a30\n" + "".join(json.dumps(line) + "\n" for line in default_lines)
                    + f"{fleet.MARK}reference standby-a30\n" + "".join(json.dumps(line) + "\n" for line in reference)
                    + f"{fleet.MARK}end\n")

        sent = []
        with mock.patch.object(fleet, "remote", side_effect=lambda run, table, name, script: sent.append(script)
                               or answer(lines)), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertTrue(remote.smoke_result(self.run_dir, self.table, "gilbreth"))
        self.assertIn('head -n 20 "smoke/standby-a30_$job/default/metrics.jsonl"', sent[0])
        self.assertIn('cat "smoke/reference_${profile:-one-copy}.jsonl"', sent[0])
        self.assertEqual(subprocess.run(["bash", "-n"], input=sent[0], text=True, capture_output=True).returncode, 0)
        self.assertIn("0 number(s) outside the rule", output.getvalue())
        self.assertIn("the job's own comparison, by its frozen code: NOT passed", output.getvalue())
        table = common.read_json(fleet.table_path(self.run_dir))
        self.assertEqual((table["gilbreth"]["ready"], table["gilbreth"]["smoke"]["standby-a30"]["first_update_rule"]),
                         (True, smoke_check.FIRST_LINE_RULE))
        # lines off by more than the rule keep the server not ready
        far = [dict(lines[0], policy_loss=[-2.5e-5])] + lines[1:]
        with mock.patch.object(fleet, "remote", return_value=answer(far)), contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(remote.smoke_result(self.run_dir, self.table, "gilbreth"))
        # a job that left no default lines keeps its own verdict
        with mock.patch.object(fleet, "remote", return_value=answer([])), contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(remote.smoke_result(self.run_dir, self.table, "gilbreth"))


class ReturnedUnit(FleetCase):
    """The check of a finished unit a server returns."""

    def setUp(self):
        """A pack that Gautschi holds, and the files its worker left for the tick in data/research/<unit>/."""
        super().setUp()
        self.name = self.pack()
        held = self.run_dir / "queue/remote_gautschi"
        held.mkdir()
        (self.run_dir / "queue/pending" / f"{self.name}.json").rename(held / f"{self.name}.json")
        common.write_json(fleet.assignment_path(self.run_dir), {self.name: {"server": "gautschi", "state": "staged"}})
        self.specification = common.read_json(self.run_dir / "specifications" / f"{self.name}.json")
        self.record = dict(common.read_json(held / f"{self.name}.json"), execution_profile="four copies",
                           execution_profile_settings={"learner": PROFILES["four copies"]["learner"],
                                                       "environment": PROFILES["four copies"]["environment"]})
        self.pulled = self.run_dir / "data/research" / self.name
        write_run_records(self.pulled, metric_lines(244, copies=4), 244, completed=True)
        (self.pulled / "checkpoint.pkl").write_bytes(b"state at the target")
        learner, _ = pixel_queue.trainer_settings(self.record, self.specification)
        # run.json as the trainer writes it: every learner field, the per-copy settings as sorted pairs
        described = {key: value for key, value in learner.items() if key != "per_copy_scalar_settings"}
        described.update(n_copies=4, per_copy_scalar_settings=[["fast_last_layer_step", [0.01, 0.03, 0.1, 0.3]]])
        (self.pulled / "run.json").write_text(json.dumps({"learner": described, "copy_seeds": [13, 13, 13, 13]}))
        (self.pulled / "events.jsonl").write_text(
            json.dumps({"event": "worker_start", "device": "NVIDIA H100 80GB HBM3"}) + "\n"
            + json.dumps({"event": "completed", "update": 244}) + "\n")
        common.write_json(self.pulled / "summary.json", {
            "completed": True, "unit_id": self.name, "target_updates": 244, "execution_profile": "four copies",
            "execution_profile_settings": self.record["execution_profile_settings"], "card": "NVIDIA H100 80GB HBM3",
            "attempt": {"seconds_per_update": 2.1, "update_at_end": 244, "wall_seconds": 600.0},
            "earlier_attempts": [{"job_id": "900", "started_at": "2026-11-18T01:00:00-05:00",
                                  "card": "NVIDIA H100 80GB HBM3", "outcome": "stopped_with_state_saved",
                                  "seconds_per_update": 2.2}]})

    def problem(self):
        """What the check finds wrong with the pulled files; None when nothing."""
        return remote.returned_unit_problem(self.pulled, self.record, self.specification)

    def test_complete_files_pass_and_move_into_the_units_run_folder(self):
        """The files are moved into data/<unit>/run/; UVA's record takes the server's profile and attempts."""
        self.assertIsNone(self.problem())
        self.assertIsNone(remote.verify_unit(self.run_dir, self.name))
        folder = pixel_queue.run_folder(self.run_dir, self.name)
        for file_name in pixel_queue.RUN_FILES:
            self.assertTrue((folder / file_name).exists(), file_name)
        self.assertTrue((self.pulled / "summary.json").exists())
        record = common.read_json(self.run_dir / "queue/remote_gautschi" / f"{self.name}.json")
        self.assertEqual(record["execution_profile"], "four copies")
        self.assertEqual([(attempt["outcome"], attempt["seconds_per_update"]) for attempt in record["history"]],
                         [("stopped_with_state_saved", 2.2), ("complete", 2.1)])
        self.assertEqual(record["history"][-1]["server"], "gautschi")
        # a second check, after the files moved, reads them where they are now
        self.assertIsNone(remote.verify_unit(self.run_dir, self.name))
        block = remote.result_block("gautschi", {"job": "901", "exit": "0", "wall_seconds": "600"},
                                    common.read_json(self.pulled / "summary.json"))
        self.assertEqual((block["server"], block["card"], block["seconds_per_update"], block["execution_profile"]),
                         ("gautschi", "NVIDIA H100 80GB HBM3", 2.1, "four copies"))

    def test_each_missing_piece_is_named(self):
        """A short metrics file, a progress record below the target, another seed list, other settings, no records."""
        lines = (self.pulled / "metrics.jsonl").read_text().splitlines(keepends=True)
        (self.pulled / "metrics.jsonl").write_text("".join(lines[:-1]))
        self.assertIn("metrics.jsonl holds 243 lines", self.problem())
        (self.pulled / "metrics.jsonl").write_text("".join(lines))
        progress = (self.pulled / "progress.json").read_text()
        (self.pulled / "progress.json").write_text(json.dumps({"update": 200, "target_updates": 244}))
        self.assertIn("progress.json does not say completed at update 244", self.problem())
        (self.pulled / "progress.json").write_text(progress)
        described = json.loads((self.pulled / "run.json").read_text())
        (self.pulled / "run.json").write_text(json.dumps(dict(described, copy_seeds=[1, 2, 3, 4])))
        self.assertIn("run.json is of the seeds [1, 2, 3, 4]", self.problem())
        other = dict(described, learner=dict(described["learner"], fast_last_layer_sharing_group=4))
        (self.pulled / "run.json").write_text(json.dumps(other))
        self.assertIn("run.json's learner differs from the unit's in ['fast_last_layer_sharing_group']", self.problem())
        (self.pulled / "run.json").write_text(json.dumps(described))
        events = (self.pulled / "events.jsonl").read_text()
        (self.pulled / "events.jsonl").write_text(json.dumps({"event": "checkpoint", "update": 244}) + "\n")
        self.assertIn("no completed event", self.problem())
        (self.pulled / "events.jsonl").write_text(events)
        windows = (self.pulled / "game_windows.jsonl").read_text().splitlines(keepends=True)
        (self.pulled / "game_windows.jsonl").write_text("".join(windows[:-1]))
        self.assertIn("game_windows.jsonl holds 3 windows; 4 are expected", self.problem())
        (self.pulled / "game_windows.jsonl").write_text("".join(windows))
        (self.pulled / "checkpoint.pkl").unlink()
        self.assertEqual(self.problem(), "missing or empty: checkpoint.pkl")
        self.assertEqual(remote.verify_unit(self.run_dir, self.name), "missing or empty: checkpoint.pkl")
        self.assertFalse(pixel_queue.run_folder(self.run_dir, self.name).exists())


class SmokeRules(unittest.TestCase):
    """The parts of the smoke job that need no card."""

    def test_the_first_update_decides_and_later_differences_are_counted(self):
        """Equal lines pass; a first line off by 1% fails; later lines that differ are reported and do not decide."""
        reference = [{"update": index + 1, "unix": 5.0 + index, "loss": [0.5 + index], "episode_count": [25],
                      "rooms_visited": [[True, False]]} for index in range(3)]
        same = [dict(line, unix=99.0, event_counts=[[0]]) for line in reference]
        check = smoke_check.compare_with_archive(reference, same)
        self.assertEqual((check["passed"], "every field equal" in check["detail"]), (True, True))
        later = [dict(same[0]), dict(same[1], loss=[1.6]), dict(same[2])]
        check = smoke_check.compare_with_archive(reference, later)
        self.assertEqual((check["passed"], "1 differing field(s)" in check["detail"]), (True, True))
        first = [dict(same[0], loss=[0.505])] + same[1:]
        self.assertFalse(smoke_check.compare_with_archive(reference, first)["passed"])
        count = [dict(same[0], episode_count=[26])] + same[1:]
        self.assertFalse(smoke_check.compare_with_archive(reference, count)["passed"])
        self.assertFalse(smoke_check.compare_with_archive(reference, same[:2])["passed"])

    def test_a_value_near_zero_may_differ_in_its_last_bits(self):
        """A policy loss of 1.5e-5 off by 5e-8 passes (relative 3.3e-3); off by 1e-5 it fails; whole numbers stay exact."""
        reference = [{"update": 1, "unix": 5.0, "policy_loss": [-1.5e-5], "loss": [0.595], "episode_count": [25],
                      "rooms_visited": [[True, False]]}]
        rounding = [dict(reference[0], unix=9.0, policy_loss=[-1.5e-5 + 5e-8], loss=[0.595 + 5e-4])]
        check = smoke_check.compare_with_archive(reference, rounding)
        self.assertEqual((check["passed"], "largest relative difference 3.33e-03, 0 number(s) outside" in check["detail"]),
                         (True, True))
        self.assertEqual(smoke_check.outside_first_line_rule([-1.5e-5], [-1.5e-5 - 5e-8]), 0)
        self.assertFalse(smoke_check.compare_with_archive(reference, [dict(rounding[0], policy_loss=[-2.5e-5])])["passed"])
        self.assertFalse(smoke_check.compare_with_archive(reference, [dict(rounding[0], loss=[0.5961])])["passed"])
        self.assertFalse(smoke_check.compare_with_archive(reference, [dict(rounding[0], episode_count=[26])])["passed"])
        self.assertFalse(smoke_check.compare_with_archive(reference, [dict(rounding[0], rooms_visited=[[True, True]])])
                         ["passed"])
        self.assertEqual((smoke_check.outside_first_line_rule([1.0, 2.0], [1.0]), smoke_check.outside_first_line_rule(
            float("nan"), float("nan")), smoke_check.outside_first_line_rule(None, None),
                          smoke_check.outside_first_line_rule(0.0, float("nan"))), (1, 0, 0, 1))

    def test_asset_hashes_name_a_changed_file(self):
        """Every file UVA lists must be there with the same bytes; a file only the server has does not decide."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "jaxtari/sprites").mkdir(parents=True)
            (root / "jaxtari/asset_source.json").write_text("{}")
            (root / "jaxtari/sprites/room.png").write_bytes(b"picture")
            expected = remote.asset_hashes(root)
            self.assertEqual(set(expected), {"jaxtari/asset_source.json", "jaxtari/sprites/room.png"})
            self.assertTrue(smoke_check.asset_check(root, expected)["passed"])
            (root / "jaxtari/cache.bin").write_bytes(b"left by a job")
            extra = smoke_check.asset_check(root, expected)
            self.assertEqual((extra["passed"], "1 file(s) only on this server" in extra["detail"]), (True, True))
            (root / "jaxtari/sprites/room.png").write_bytes(b"another picture")
            changed = smoke_check.asset_check(root, expected)
            self.assertEqual((changed["passed"], "jaxtari/sprites/room.png" in changed["detail"]), (False, True))

    def test_the_speed_line_has_the_five_throughput_figures(self):
        """Copies, seconds per iteration, total and per-copy steps per second, hours per million steps per copy."""
        line = smoke_check.speed_line("default", 4, {"seconds_per_update": 2.08, "peak_card_memory_mb": 61200})
        self.assertEqual(line, "default: copies 4, 2.08 s per iteration, total 0.0315 million steps per second, "
                               "7.9 thousand steps per second per copy, 0.0353 hours per million steps per copy, "
                               "peak card memory 61,200 MB")
        self.assertIn("too few updates", smoke_check.speed_line("default", 1, {"seconds_per_update": None,
                                                                                "peak_card_memory_mb": 0}))
        self.assertEqual((smoke_check.finite([1.0, {"a": [2.0]}]), smoke_check.finite([float("nan")])), (True, False))


if __name__ == "__main__":
    unittest.main()


class SmokeReferenceByJobProfile(unittest.TestCase):
    """The smoke result is compared with the reference of the profile its job ran, not the pool's present profile."""

    def read(self, result_profile, pool_profile):
        """Run the generated read script with bash in a folder standing for the server; return the reference lines used."""
        with tempfile.TemporaryDirectory() as folder:
            server = Path(folder)
            (server / "smoke/p_7/default").mkdir(parents=True)
            result = {"passed": True, "job_id": "7", "device": "NVIDIA H200", "written_at": "2026-11-19T15:00:40-05:00",
                      "checks": []}
            if result_profile:
                result["execution_profile"] = result_profile
            (server / "smoke/result_p.json").write_text(json.dumps(result, indent=2))
            (server / "smoke/p_7/default/metrics.jsonl").write_text('{"update": 1}\n')
            (server / "smoke/reference_four-copies.jsonl").write_text('{"reference": "four"}\n')
            (server / "smoke/reference_eight-copies.jsonl").write_text('{"reference": "eight"}\n')
            table = {"mit": {"remote_run": str(server), "pools": {},
                             "pools_held_back": {"p": {"execution_profile": pool_profile}}}}
            run = lambda run_dir, table, name, script: subprocess.run(["bash"], input=script, capture_output=True,
                                                                      text=True, check=True).stdout
            seen = []
            again = lambda result, reference, lines: seen.append(reference) or result
            with mock.patch.object(remote.fleet, "remote", side_effect=run), \
                    mock.patch.object(remote, "read_first_update_again", side_effect=again), \
                    contextlib.redirect_stdout(io.StringIO()):
                remote.read_smoke_results(server, table, "mit", ["p"])
            return seen[0]

    def test_the_profile_the_job_ran(self):
        """before: a four-copy job in a pool now listed as eight copies ; after: the four-copy reference."""
        self.assertEqual(self.read("four copies", "eight copies"), [{"reference": "four"}])

    def test_the_pool_profile_when_the_result_names_none(self):
        """A result without a profile falls back to the pool's present profile."""
        self.assertEqual(self.read(None, "eight copies"), [{"reference": "eight"}])
