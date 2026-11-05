"""Check durable monitor launch bookkeeping using private files and synthetic scheduler replies."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ACTUAL = Path(__file__).resolve().parent
ACTUAL_RUN = ACTUAL.parent.parent


class MonitorLauncherTests(unittest.TestCase):
    """Exercise the real prepared inputs, publication and ledger changes without actual scheduler actions."""
    def setUp(self):
        """Copy the reviewed small monitor packet into one clearly synthetic local repository."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.integration = Path(self.temporary.name) / "synthetic_integration"
        self.run = self.integration / "platform/benchmark_runs/synthetic_campaign"
        self.packet = self.run / "remote_setup" / ACTUAL.name
        self.unit = self.run / "data" / ACTUAL.name
        for path in (self.packet, self.unit, self.run / "queue", self.run / "slurm/jobs"):
            path.mkdir(parents=True, exist_ok=True)
        original = json.loads((ACTUAL / "static_source_freeze.json").read_text())
        for name, record in original["files"].items():
            old = Path(name)
            new = self.run / old.relative_to(ACTUAL_RUN)
            new.parent.mkdir(parents=True, exist_ok=True)
            raw = old.read_bytes()
            if "source_snapshot" not in old.parts and old.suffix not in {".py", ".tar"}:
                raw = raw.replace(str(ACTUAL_RUN).encode(), str(self.run).encode())
            new.write_bytes(raw)
            new.chmod(int(record["mode"], 8))
        for name in ("launch_monitor_once.py", "test_launch_monitor_once.py", "run_monitor_launcher_metadata_tests.py",
                     "monitor_launch_policy.json", "independent_prepared_packet_review.json"):
            raw = (ACTUAL / name).read_bytes()
            if not name.endswith(".py"):
                raw = raw.replace(str(ACTUAL_RUN).encode(), str(self.run).encode())
            (self.packet / name).write_bytes(raw)
        self.plan = json.loads((self.packet / "monitor_replacement_plan.json").read_text())
        self.plan["job_script"] = self.ref(self.packet / "monitor.sbatch")
        self.write(self.packet / "monitor_replacement_plan.json", self.plan)
        manifest = json.loads((self.packet / "manifest.yaml").read_text())
        manifest["plan"] = self.ref(self.packet / "monitor_replacement_plan.json")
        self.write(self.packet / "manifest.yaml", manifest)
        (self.unit / "manifest.yaml").write_bytes((self.packet / "manifest.yaml").read_bytes())
        frozen = {str(self.run / Path(name).relative_to(ACTUAL_RUN)): self.file_record(self.run / Path(name).relative_to(ACTUAL_RUN))
                  for name in original["files"]}
        self.write(self.packet / "static_source_freeze.json", {**original, "files": frozen})
        prepared = json.loads((self.packet / "independent_prepared_packet_review.json").read_text())
        prepared["source_freeze"] = self.ref(self.packet / "static_source_freeze.json")
        prepared["plan"] = self.ref(self.packet / "monitor_replacement_plan.json")
        prepared["candidate_full_report"] = self.ref(self.packet / "candidate_after_reporting_index_full_report_receipt.json")
        prepared["synthetic_fixture"] = True
        self.write(self.packet / "independent_prepared_packet_review.json", prepared)
        for name in ("submitted_jobids.txt", "submitted_jobids_uva.txt", "submissions.tsv"):
            (self.run / "slurm" / name).write_text("")
        (self.run / "queue/metadata.lock").write_text("")
        specification = importlib.util.spec_from_file_location("private_monitor_launcher", self.packet / "launch_monitor_once.py")
        self.module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(self.module)
        paths = list(frozen) + [str(self.packet / name) for name in (
            "static_source_freeze.json", "independent_prepared_packet_review.json", "launch_monitor_once.py",
            "test_launch_monitor_once.py", "run_monitor_launcher_metadata_tests.py", "monitor_launch_policy.json")]
        self.write(self.packet / "monitor_launcher_source_freeze.json", {"files": {name: self.ref(Path(name))["sha256"] for name in paths}})
        self.write(self.packet / "monitor_launcher_independent_review.json", {"passed": True, "synthetic_fixture": True,
            "source_freeze": self.ref(self.packet / "monitor_launcher_source_freeze.json")})
        self.git("init", "-q")
        self.publication = self.commit()
        self.job = "910002"
        self.calls = []
        self.fail_release = False
        self.altered_held = None

    def ref(self, path):
        """Hash one honest private metadata file."""
        import hashlib
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def file_record(self, path):
        """Retain the actual private file mode, size and digest."""
        return {**{key: value for key, value in self.ref(path).items() if key == "sha256"},
                "bytes": path.stat().st_size, "mode": oct(path.stat().st_mode & 0o777)}

    def write(self, path, value):
        """Write only private synthetic fixture metadata."""
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

    def git(self, *arguments):
        """Use only this temporary local Git repository with hooks and signing disabled."""
        return subprocess.check_output(["git", "-C", str(self.integration), "-c", "core.hooksPath=/dev/null",
            "-c", "commit.gpgSign=false", "-c", "user.name=Synthetic monitor fixture",
            "-c", "user.email=synthetic@example.invalid", *arguments], stderr=subprocess.PIPE)

    def commit(self):
        """Create a real private publication of the fixture's exact current inputs."""
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "Synthetic private monitor launcher fixture")
        return self.git("rev-parse", "HEAD").decode().strip()

    def held_text(self):
        """Describe the exact real-resource shape using a synthetic held identifier."""
        return ("JobId=" + self.job + " UserId=sl5nw(2618919) JobState=PENDING Reason=JobHeldUser "
            "Requeue=0 Restarts=0 TimeLimit=7-00:00:00 Partition=nolim Account=sl5nw_base QOS=normal "
            "NumNodes=1-1 NumCPUs=1 CPUs/Task=1 NumTasks=1 MinMemoryNode=2G "
            "ReqTRES=cpu=1,mem=2G,node=1,billing=309 TresPerNode=gres/gpu:0 Command=" +
            str(self.packet / "monitor.sbatch") + "\n")

    def fake_scheduler(self, command, stem):
        """Retain synthetic returns while replacing every actual scheduler invocation."""
        self.calls.append(command)
        if command[0] == "sbatch":
            stdout, returncode = (self.job + "\n").encode(), 0
        elif command[:3] == ["scontrol", "show", "job"]:
            self.assertEqual(command, ["scontrol", "show", "job", "-o", self.job])
            stdout, returncode = (self.altered_held or self.held_text()).encode(), 0
        else:
            self.assertEqual(command, ["scontrol", "release", self.job])
            stdout, returncode = b"", int(self.fail_release)
        for suffix, raw in (("_stdout.raw", stdout), ("_stderr.raw", b"")):
            self.module.publish(self.packet / (stem + suffix), raw)
        self.module.publish(self.packet / (stem + "_return.json"), {"command": command, "returncode": returncode,
            "synthetic_fixture": True, "started_at_utc": self.module.now(), "finished_at_utc": self.module.now()})
        if returncode:
            raise ValueError("synthetic native release failure")
        return SimpleNamespace(stdout=stdout, stderr=b"", returncode=returncode)

    def held(self):
        """Exercise real durable private ledger appends using a synthetic scheduler reply."""
        with mock.patch.object(self.module, "scheduler", side_effect=self.fake_scheduler):
            return self.module.submit_held_once(self.publication)

    def reviewed_release(self):
        """Publish an honest private fixed-identifier review after the actual private binding exists."""
        with mock.patch.object(self.module, "scheduler", side_effect=self.fake_scheduler):
            self.module.prepare_release(self.publication)
        self.write(self.packet / "monitor_release_independent_review.json", {"passed": True, "synthetic_fixture": True,
            "binding": self.ref(self.packet / "monitor_release_binding.json")})
        self.publication = self.commit()

    def test_default_real_publication_check_performs_no_action(self):
        """The actual private source and Git publication check never invokes a scheduler."""
        output = io.StringIO()
        with mock.patch.object(self.module, "scheduler", side_effect=AssertionError("no native action")), contextlib.redirect_stdout(output):
            self.module.main(["--publication-commit", self.publication])
        self.assertFalse(json.loads(output.getvalue())["action_performed"])
        self.assertFalse((self.packet / "monitor_submission_intent.json").exists())

    def test_changed_prepared_source_refuses_before_intent(self):
        """A changed original artifact cannot reach a native submission attempt."""
        (self.packet / "allocation_guard.py").write_text("# changed private guard\n")
        with mock.patch.object(self.module, "scheduler", side_effect=AssertionError("no native action")):
            with self.assertRaisesRegex(ValueError, "prepared monitor artifact changed"):
                self.module.submit_held_once(self.publication)
        self.assertFalse((self.packet / "monitor_submission_intent.json").exists())

    def test_reduced_freeze_and_missing_prepared_review_refuse(self):
        """A newly committed private roster cannot remove a required actual prepared proof."""
        freeze = json.loads((self.packet / "monitor_launcher_source_freeze.json").read_text())
        del freeze["files"][str(self.packet / "independent_prepared_packet_review.json")]
        self.write(self.packet / "monitor_launcher_source_freeze.json", freeze)
        self.write(self.packet / "monitor_launcher_independent_review.json", {"passed": True,
            "source_freeze": self.ref(self.packet / "monitor_launcher_source_freeze.json"), "synthetic_fixture": True})
        publication = self.commit()
        with self.assertRaisesRegex(ValueError, "mandatory prepared proof"):
            self.module.check_publication(publication)
        prepared = json.loads((self.packet / "independent_prepared_packet_review.json").read_text())
        prepared["passed"] = False
        self.write(self.packet / "independent_prepared_packet_review.json", prepared)
        with self.assertRaisesRegex(ValueError, "independent prepared monitor proof"):
            self.module.checked_packet()

    def test_one_held_submission_writes_three_real_ledgers_before_receipt(self):
        """The real private identifier and row are durable before the successful receipt publication."""
        original = self.module.publish
        def publish(path, value):
            """Check actual private ledger bytes when the receipt is about to be published."""
            if Path(path).name == "monitor_submission_receipt.json":
                self.assertEqual((self.run / "slurm/submitted_jobids.txt").read_text(), self.job + "\n")
                self.assertEqual((self.run / "slurm/submitted_jobids_uva.txt").read_text(), self.job + "\n")
                self.assertEqual((self.run / "slurm/submissions.tsv").read_text(),
                                 "\t".join(self.module.submission_row(self.job, self.plan)) + "\n")
                self.assertTrue((self.packet / "monitor_submission_ledgers_fsynced.json").is_file())
            return original(path, value)
        with mock.patch.object(self.module, "publish", side_effect=publish):
            receipt = self.held()
        self.assertEqual(receipt["job_id"], self.job)
        self.assertTrue((self.run / "slurm/jobs" / self.job / "assignment.json").is_file())
        with mock.patch.object(self.module, "scheduler", side_effect=self.fake_scheduler):
            with self.assertRaisesRegex(ValueError, "preserve the previous attempt"):
                self.module.submit_held_once(self.publication)
        self.assertEqual(sum(command[0] == "sbatch" for command in self.calls), 1)

    def test_partial_second_ledger_failure_preserves_id_and_refuses_retry(self):
        """A real first append remains saved when a private second-ledger error prevents a receipt."""
        original, counter = self.module.append_member, 0
        def append(path, value):
            """Inject one deterministic private synchronization error after the first real append."""
            nonlocal counter
            counter += 1
            if counter == 2:
                raise OSError("synthetic second-ledger failure")
            return original(path, value)
        with mock.patch.object(self.module, "scheduler", side_effect=self.fake_scheduler), mock.patch.object(self.module, "append_member", side_effect=append):
            with self.assertRaisesRegex(OSError, "second-ledger failure"):
                self.module.submit_held_once(self.publication)
        self.assertEqual((self.run / "slurm/submitted_jobids.txt").read_text(), self.job + "\n")
        self.assertFalse((self.packet / "monitor_submission_receipt.json").exists())
        with mock.patch.object(self.module, "scheduler", side_effect=self.fake_scheduler):
            with self.assertRaisesRegex(ValueError, "preserve the previous attempt"):
                self.module.submit_held_once(self.publication)
        self.assertEqual(sum(command[0] == "sbatch" for command in self.calls), 1)

    def test_inherited_batch_options_are_removed(self):
        """Inherited reservation, memory or partition options cannot alter the recorded request."""
        returned = SimpleNamespace(returncode=0, stdout=b"synthetic return\n", stderr=b"")
        with mock.patch.dict(self.module.os.environ, {"SBATCH_RESERVATION": "expired", "SBATCH_MEM_PER_NODE": "99G"}),              mock.patch.object(self.module.subprocess, "run", return_value=returned) as call:
            self.module.scheduler(["sbatch", "--hold"], "synthetic_environment")
        self.assertFalse(any(name.startswith("SBATCH_") for name in call.call_args.kwargs["env"]))
        self.assertEqual(call.call_args.kwargs["env"]["TZ"], "UTC")

    def test_foreign_query_owner_extra_gpu_and_wrong_memory_refuse(self):
        """Ownership is checked before a query; changed cached resources cannot become a binding."""
        self.held()
        with mock.patch.object(self.module, "scheduler", side_effect=AssertionError("foreign query forbidden")):
            with self.assertRaisesRegex(ValueError, "exact ledger-owned monitor"):
                self.module.query_held("910003", "foreign")
        for index, (before, after) in enumerate((("UserId=sl5nw(2618919)", "UserId=other(1)"),
                ("TresPerNode=gres/gpu:0", "TresPerNode=gres/gpu:1"), ("mem=2G", "mem=8G"))):
            self.altered_held = self.held_text().replace(before, after)
            with mock.patch.object(self.module, "scheduler", side_effect=self.fake_scheduler):
                with self.assertRaises(ValueError):
                    self.module.query_held(self.job, "invalid_" + str(index))
        self.assertFalse((self.packet / "monitor_release_binding.json").exists())

    def test_release_refuses_without_published_independent_binding_review(self):
        """A successful held job cannot be released without its exact approved fixed binding."""
        self.held()
        with mock.patch.object(self.module, "scheduler", side_effect=self.fake_scheduler):
            self.module.prepare_release(self.publication)
            before = len(self.calls)
            with self.assertRaises((ValueError, FileNotFoundError)):
                self.module.release_once(self.publication)
        self.assertEqual(before, len(self.calls))

    def test_release_once_saves_receipt_before_completion_and_never_changes_old_monitor(self):
        """The actual private release receipt precedes completion; no old monitor control path exists."""
        self.held()
        self.reviewed_release()
        original, events = self.module.publish, []
        def publish(path, value):
            """Retain the actual private publication order."""
            events.append(Path(path).name)
            if Path(path).name == "monitor_release_complete.json":
                self.assertTrue((self.packet / "monitor_release_receipt.json").is_file())
            return original(path, value)
        with mock.patch.object(self.module, "scheduler", side_effect=self.fake_scheduler), mock.patch.object(self.module, "publish", side_effect=publish):
            released = self.module.release_once(self.publication)
        self.assertFalse(released["old_monitor_stopped_or_adopted"])
        self.assertLess(events.index("monitor_release_receipt.json"), events.index("monitor_release_complete.json"))
        with mock.patch.object(self.module, "scheduler", side_effect=self.fake_scheduler):
            with self.assertRaisesRegex(ValueError, "preserve the previous attempt"):
                self.module.release_once(self.publication)
        self.assertEqual(sum(command[:2] == ["scontrol", "release"] for command in self.calls), 1)
        self.assertFalse((self.run / "next_actions.json").exists())

    def test_failed_release_preserves_attempt_without_false_success(self):
        """A failed native return cannot publish a successful release or be retried automatically."""
        self.held()
        self.reviewed_release()
        self.fail_release = True
        with mock.patch.object(self.module, "scheduler", side_effect=self.fake_scheduler):
            with self.assertRaisesRegex(ValueError, "synthetic native release failure"):
                self.module.release_once(self.publication)
            with self.assertRaisesRegex(ValueError, "preserve the previous attempt"):
                self.module.release_once(self.publication)
        self.assertFalse((self.packet / "monitor_release_receipt.json").exists())
        self.assertFalse((self.packet / "monitor_release_complete.json").exists())
        self.assertEqual(sum(command[:2] == ["scontrol", "release"] for command in self.calls), 1)

