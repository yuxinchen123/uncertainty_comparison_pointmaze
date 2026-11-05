"""Check one-shot launch bookkeeping with private files, local Git and a synthetic scheduler."""
import ast
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

ACTUAL_PACKET = Path(__file__).resolve().parent
ACTUAL_RUN = ACTUAL_PACKET.parent.parent
ACTUAL_UNIT = ACTUAL_RUN / "data" / ACTUAL_PACKET.name


class LauncherTests(unittest.TestCase):
    """Exercise real durable filesystem changes while replacing every scheduler invocation."""
    def setUp(self):
        """Copy only the small prepared metadata into a labelled synthetic local repository."""
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.integration = Path(self.temporary.name) / "synthetic_integration"
        self.run = self.integration / "platform/benchmark_runs/synthetic_campaign"
        self.packet = self.run / "remote_setup" / ACTUAL_PACKET.name
        self.unit = self.run / "data" / ACTUAL_PACKET.name
        for directory in (self.packet, self.unit / "code", self.run / "queue", self.run / "slurm/jobs"):
            directory.mkdir(parents=True, exist_ok=True)
        original = json.loads((ACTUAL_PACKET / "immutable_artifacts.json").read_text())["files"]
        for name in original:
            source = Path(name); destination = self.run / source.relative_to(ACTUAL_RUN)
            destination.parent.mkdir(parents=True, exist_ok=True)
            raw = source.read_bytes()
            if source.suffix != ".py":
                raw = raw.replace(str(ACTUAL_RUN).encode(), str(self.run).encode())
            destination.write_bytes(raw)
        # Before: original path-derived hashes; after: the private plan and its honest private checksum records.
        self.plan = json.loads((self.unit / "measurement_plan.json").read_text())
        self.write(self.unit / "measurement_prepared.json", {"plan_sha256": self.digest(self.unit / "measurement_plan.json")})
        checked = [*sorted((self.unit / "code").iterdir()), self.unit / "measurement_plan.json", self.unit / "measurement_prepared.json"]
        (self.packet / "source_checksums.sha256").write_text("".join(self.digest(path) + "  " + str(path) + "\n" for path in checked))
        rows = {str(self.run / Path(name).relative_to(ACTUAL_RUN)): self.file_record(self.run / Path(name).relative_to(ACTUAL_RUN))
                for name in original}
        self.write(self.packet / "immutable_artifacts.json", {"files": rows})
        self.write(self.packet / "packet_static_verification.json", {"passed": True, "synthetic_fixture": True})
        for name in ("launch_diagnostic_once.py", "test_launch_diagnostic_once.py"):
            (self.packet / name).write_bytes((ACTUAL_PACKET / name).read_bytes())
        specification = importlib.util.spec_from_file_location("private_nsight_launcher", self.packet / "launch_diagnostic_once.py")
        self.module = importlib.util.module_from_spec(specification); specification.loader.exec_module(self.module)
        for name in ("submitted_jobids.txt", "submitted_jobids_uva.txt", "submissions.tsv"):
            (self.run / "slurm" / name).write_text("")
        (self.run / "queue/metadata.lock").write_text("")
        frozen = [Path(name) for name in rows] + [self.packet / name for name in (
            "immutable_artifacts.json", "packet_static_verification.json", "launch_diagnostic_once.py", "test_launch_diagnostic_once.py")]
        self.write(self.packet / "launcher_source_freeze.json", {"files": {str(path): self.digest(path) for path in frozen}})
        self.write(self.packet / "launcher_independent_review.json", {"passed": True, "synthetic_fixture": True,
            "source_freeze": self.module.reference(self.packet / "launcher_source_freeze.json")})
        self.git("init", "-q"); self.publication = self.commit()
        self.job = "910001"; self.calls = []; self.fail_release = False

    def digest(self, path):
        """Hash a private ordinary file without loading any native library."""
        import hashlib
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def file_record(self, path):
        """Match the real formatter's size, mode and digest fields for a private artifact."""
        return {"sha256": self.digest(path), "bytes": path.stat().st_size, "mode": oct(path.stat().st_mode & 0o777)}

    def write(self, path, value):
        """Write clearly synthetic fixture metadata; never replace actual campaign artifacts."""
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

    def git(self, *arguments):
        """Use only this private local repository, with hooks and signing disabled."""
        return subprocess.check_output(["git", "-C", str(self.integration), "-c", "core.hooksPath=/dev/null",
            "-c", "commit.gpgSign=false", "-c", "user.name=Synthetic launcher fixture",
            "-c", "user.email=synthetic@example.invalid", *arguments], stderr=subprocess.PIPE)

    def commit(self):
        """Create a real local Git publication of the private fixture only."""
        self.git("add", "-A"); self.git("commit", "-q", "-m", "Synthetic diagnostic launcher fixture")
        return self.git("rev-parse", "HEAD").decode().strip()

    def held_text(self):
        """Describe a labelled synthetic held request with the exact diagnostic resource shape."""
        return ("JobId=" + self.job + " UserId=sl5nw(2618919) JobState=PENDING Reason=JobHeldUser "
            "Requeue=0 Restarts=0 TimeLimit=00:05:00 Partition=gpu Account=sl5nw_base QOS=csresnolim Reservation=sl5nw_168 "
            "ReqNodeList=jaguar03 NumNodes=1-1 NumCPUs=1 CPUs/Task=1 NumTasks=1 "
            "ReqTRES=cpu=1,mem=2G,node=1,gres/gpu=1 TresPerNode=gres/gpu:nvidia_rtx_a4500:1 "
            "Command=" + str(self.packet / "job.sbatch") + "\n")

    def fake_scheduler(self, command, stem):
        """Save synthetic returns in private files while never invoking a scheduler."""
        self.calls.append(command)
        if command[0] == "sbatch":
            stdout = (self.job + "\n").encode(); returncode = 0
        elif command[:3] == ["scontrol", "show", "job"]:
            self.assertEqual(command, ["scontrol", "show", "job", "-o", self.job])
            stdout = self.held_text().encode(); returncode = 0
        else:
            self.assertEqual(command, ["scontrol", "release", self.job])
            stdout = b""; returncode = 1 if self.fail_release else 0
        for suffix, raw in (("_stdout.raw", stdout), ("_stderr.raw", b"")):
            self.module.publish(self.packet / (stem + suffix), raw)
        self.module.publish(self.packet / (stem + "_return.json"), {"command": command, "returncode": returncode,
            "synthetic_fixture": True, "started_at_utc": self.module.now(), "finished_at_utc": self.module.now()})
        if returncode:
            raise ValueError("synthetic native command failure")
        return SimpleNamespace(stdout=stdout, stderr=b"", returncode=returncode)

    def make_held(self):
        """Run the actual ledger-writing implementation with a synthetic returned identifier."""
        with mock.patch.object(self.module, "scheduler", side_effect=self.fake_scheduler):
            return self.module.submit_held_once(self.publication)

    def prepare_fixed_release(self):
        """Bind the real private receipt and publish the private release review in local Git."""
        with mock.patch.object(self.module, "scheduler", side_effect=self.fake_scheduler):
            self.module.prepare_release(self.publication)
        self.write(self.packet / "release_independent_review.json", {"passed": True, "synthetic_fixture": True,
            "binding": self.module.reference(self.packet / "release_binding.json")})
        self.publication = self.commit()

    def test_default_publication_check_has_no_scheduler_action(self):
        """The real private Git check is useful without any submission or query."""
        output = io.StringIO()
        with mock.patch.object(self.module, "scheduler", side_effect=AssertionError("no native action")), contextlib.redirect_stdout(output):
            self.module.main(["--publication-commit", self.publication])
        self.assertFalse(json.loads(output.getvalue())["action_performed"])
        self.assertFalse((self.packet / "submission_intent.json").exists())

    def test_changed_initial_source_refuses_before_intent(self):
        """A changed staged source cannot reach the native submission call."""
        (self.unit / "code/nsight_process_identity.py").write_text("# altered private source\n")
        with mock.patch.object(self.module, "scheduler", side_effect=AssertionError("no native action")):
            with self.assertRaisesRegex(ValueError, "frozen diagnostic artifact"):
                self.module.submit_held_once(self.publication)
        self.assertFalse((self.packet / "submission_intent.json").exists())

    def test_reduced_publication_roster_refuses(self):
        """A second private Git commit cannot remove a mandatory helper from the freeze."""
        freeze = json.loads((self.packet / "launcher_source_freeze.json").read_text())
        del freeze["files"][str(self.unit / "code/nsight_process_identity.py")]
        self.write(self.packet / "launcher_source_freeze.json", freeze)
        self.write(self.packet / "launcher_independent_review.json", {"passed": True, "synthetic_fixture": True,
            "source_freeze": self.module.reference(self.packet / "launcher_source_freeze.json")})
        publication = self.commit()
        with self.assertRaisesRegex(ValueError, "mandatory source"):
            self.module.check_publication(publication)

    def test_one_submission_appends_and_fsyncs_all_three_ledgers_then_refuses_reuse(self):
        """The implementation writes actual private ledger files before making a held receipt."""
        receipt = self.make_held()
        row = self.module.submission_row(self.job, self.plan)
        self.assertEqual(receipt["kind"], "owned_nsight_process_diagnostic_submission")
        self.assertEqual((self.run / "slurm/submitted_jobids.txt").read_text(), self.job + "\n")
        self.assertEqual((self.run / "slurm/submitted_jobids_uva.txt").read_text(), self.job + "\n")
        self.assertEqual((self.run / "slurm/submissions.tsv").read_text(), "\t".join(row) + "\n")
        self.assertTrue((self.packet / "submission_ledgers_fsynced.json").is_file())
        with mock.patch.object(self.module, "scheduler", side_effect=self.fake_scheduler):
            with self.assertRaisesRegex(ValueError, "preserve the previous attempt"):
                self.module.submit_held_once(self.publication)
        self.assertEqual(sum(command[0] == "sbatch" for command in self.calls), 1)

    def test_partial_ledger_failure_preserves_identifier_and_cannot_resubmit(self):
        """A synchronization failure retains the first identifier and blocks another submission."""
        original = self.module.append_member; count = 0
        def append(path, member):
            """Inject one private second-ledger failure after the first real append."""
            nonlocal count
            count += 1
            if count == 2:
                raise OSError("synthetic ledger synchronization failure")
            return original(path, member)
        with mock.patch.object(self.module, "scheduler", side_effect=self.fake_scheduler), mock.patch.object(self.module, "append_member", side_effect=append):
            with self.assertRaisesRegex(OSError, "synchronization failure"):
                self.module.submit_held_once(self.publication)
        self.assertEqual((self.run / "slurm/submitted_jobids.txt").read_text(), self.job + "\n")
        self.assertFalse((self.packet / "submission_receipt.json").exists())
        with mock.patch.object(self.module, "scheduler", side_effect=self.fake_scheduler):
            with self.assertRaisesRegex(ValueError, "preserve the previous attempt"):
                self.module.submit_held_once(self.publication)
        self.assertEqual(sum(command[0] == "sbatch" for command in self.calls), 1)

    def test_scheduler_removes_inherited_batch_variables(self):
        """A synthetic inherited reservation cannot override the explicit unreserved request."""
        returned = SimpleNamespace(returncode=0, stdout=b"synthetic return\n", stderr=b"")
        with mock.patch.dict(self.module.os.environ, {"SBATCH_RESERVATION": "sl5nw_168", "SBATCH_MEM_PER_NODE": "99G"}), mock.patch.object(self.module.subprocess, "run", return_value=returned) as run:
            self.module.scheduler(["sbatch", "--hold"], "environment_fixture")
        self.assertFalse(any(key.startswith("SBATCH_") for key in run.call_args.kwargs["env"]))
        self.assertEqual(run.call_args.kwargs["env"]["TZ"], "UTC")

    def test_exact_held_owner_and_resources_refuse_substitution(self):
        """Other owners, extra devices or an unexpected reservation cannot be released."""
        for index, replacement in enumerate(("UserId=other(1)", "ReqTRES=cpu=1,mem=2G,node=1,gres/gpu=2", "Reservation=synthetic_other_reservation")):
            text = self.held_text()
            if index == 0: text = text.replace("UserId=sl5nw(2618919)", replacement)
            elif index == 1: text = text.replace("ReqTRES=cpu=1,mem=2G,node=1,gres/gpu=1", replacement)
            else: text = text.replace("Reservation=sl5nw_168", replacement)
            with mock.patch.object(self.module, "scheduler", return_value=SimpleNamespace(stdout=text.encode(), returncode=0)):
                with self.assertRaises(ValueError):
                    self.module.query_held(self.job, "rejected_private_query_" + str(index))

    def test_release_needs_actual_published_binding_review(self):
        """A held job is insufficient when its fixed release review does not exist."""
        self.make_held()
        with mock.patch.object(self.module, "scheduler", side_effect=self.fake_scheduler):
            self.module.prepare_release(self.publication)
            before = len(self.calls)
            with self.assertRaises((ValueError, FileNotFoundError)):
                self.module.release_once(self.publication)
        self.assertEqual(len(self.calls), before)

    def test_release_receipt_precedes_atomic_sidecar_and_matches_actual_runtime_reader(self):
        """The final pair-only ownership references satisfy the unchanged diagnostic function."""
        self.make_held(); self.prepare_fixed_release(); events = []; original = self.module.publish
        def publish(path, value):
            """Retain actual private publication order and check receipt durability at final publication."""
            events.append(Path(path).name)
            if Path(path).name == "owned_launch_inputs.json":
                self.assertTrue((self.packet / "release_receipt.json").is_file())
            return original(path, value)
        with mock.patch.object(self.module, "scheduler", side_effect=self.fake_scheduler), mock.patch.object(self.module, "publish", side_effect=publish):
            self.module.release_once(self.publication)
        self.assertLess(events.index("release_receipt.json"), events.index("owned_launch_inputs.json"))
        owned = self.module.document(self.unit / "owned_launch_inputs.json")
        for name in ("plan", "submission", "release"):
            self.assertEqual(set(owned[name]), {"path", "sha256"})
        # Execute only the unchanged metadata reader function, without importing its native or numerical module.
        source = ast.parse((ACTUAL_UNIT / "code/diagnose_nsight_process_tree.py").read_text())
        node = next(item for item in source.body if isinstance(item, ast.FunctionDef) and item.name == "owned_launch")
        environment = {"SLURM_JOB_ID": self.job, "SLURM_STEP_ID": "0", "SLURM_CPUS_PER_TASK": "1",
                       "SLURM_MEM_PER_NODE": "2048", "SLURM_STEP_GPUS": "0", "CUDA_VISIBLE_DEVICES": "0"}
        namespace = {"Path": Path, "KIND": self.module.KIND,
            "identity": SimpleNamespace(read_json=self.module.document, receipt=self.module.reference, require=self.module.require),
            "os": SimpleNamespace(environ=environment, uname=lambda: SimpleNamespace(nodename="jaguar03"))}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "unchanged_owned_launch_metadata_function", "exec"), namespace)
        self.assertEqual(namespace["owned_launch"](self.unit, self.plan), owned)
        with mock.patch.object(self.module, "scheduler", side_effect=self.fake_scheduler):
            with self.assertRaisesRegex(ValueError, "preserve the previous attempt"):
                self.module.release_once(self.publication)
        self.assertEqual(sum(command[:2] == ["scontrol", "release"] for command in self.calls), 1)

    def test_failed_release_retains_attempt_without_false_ownership(self):
        """An unsuccessful synthetic native release never produces a successful final sidecar."""
        self.make_held(); self.prepare_fixed_release(); self.fail_release = True
        with mock.patch.object(self.module, "scheduler", side_effect=self.fake_scheduler):
            with self.assertRaisesRegex(ValueError, "synthetic native command failure"):
                self.module.release_once(self.publication)
            with self.assertRaisesRegex(ValueError, "preserve the previous attempt"):
                self.module.release_once(self.publication)
        self.assertFalse((self.unit / "owned_launch_inputs.json").exists())
        self.assertFalse((self.packet / "release_receipt.json").exists())
        self.assertEqual(sum(command[:2] == ["scontrol", "release"] for command in self.calls), 1)

    def test_dangling_submission_temporary_refuses_before_native_action(self):
        """A prior partial intent remains preserved rather than being overwritten."""
        (self.packet / "submission_intent.json.prepared").symlink_to(self.packet / "absent")
        with mock.patch.object(self.module, "scheduler", side_effect=AssertionError("no native action")):
            with self.assertRaisesRegex(ValueError, "preserve the previous attempt"):
                self.module.submit_held_once(self.publication)


if __name__ == "__main__":
    unittest.main()
