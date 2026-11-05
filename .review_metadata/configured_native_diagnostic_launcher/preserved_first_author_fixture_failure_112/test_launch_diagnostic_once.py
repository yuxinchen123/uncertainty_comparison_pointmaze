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
        """Prepare only labelled synthetic small source/metadata in a private local Git repository."""
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.integration = Path(self.temporary.name) / "synthetic_integration"
        self.run = self.integration / "platform/benchmark_runs/synthetic_campaign"
        literal = "2026-11-05-06-39_native-process-control_jaguar03_python-3.12_gpu-count-1_processor-count-1_host-memory-gib-2_total-seconds-300_work-seconds-240_cleanup-seconds-60_environment-transitions-0_training-updates-0"
        self.packet = self.run / "remote_setup" / literal; self.unit = self.run / "data" / literal
        for directory in (self.packet, self.unit / "code", self.run / "queue", self.run / "slurm/jobs"):
            directory.mkdir(parents=True, exist_ok=True)
        for name in ("launch_diagnostic_once.py", "test_launch_diagnostic_once.py"):
            (self.packet / name).write_bytes((ACTUAL_PACKET / name).read_bytes())
        (self.packet / "schema").mkdir()
        (self.packet / "schema/diagnose_configured_native_process_approved.py").write_bytes((ACTUAL_PACKET / "schema/diagnose_configured_native_process_approved.py").read_bytes())
        specification = importlib.util.spec_from_file_location("private_configured_native_launcher", self.packet / "launch_diagnostic_once.py")
        self.module = importlib.util.module_from_spec(specification); specification.loader.exec_module(self.module)
        prefix = self.integration / self.module.PREFIX; prefix.mkdir(parents=True)
        for name in self.module.HELPERS:
            raw = ("# Clearly synthetic source fixture; no tool or native operation. " + name + "\nVALUE = 1\n").encode()
            (prefix / name).write_bytes(raw); (self.unit / "code" / name).write_bytes(raw)
        self.git("init", "-q"); source = self.commit()
        hashes = {self.module.PREFIX + name: self.digest(self.unit / "code" / name) for name in self.module.HELPERS}
        self.plan = {"schema_version": 1, "kind": self.module.KIND, "unit_id": literal, "campaign_run_dir": str(self.run),
            "packet": str(self.packet), "python": self.module.PYTHON, "hardware_class": "jaguar03_rtx_a4500", "gpu_uuid": None,
            "maximum_active_seconds": 240, "diagnostic_time_budget": dict(self.module.BUDGET), "diagnostic_boundary": dict(self.module.ZERO),
            "helpers": {"source_commit": source, "files": hashes},
            "profile_tool": {"path": "/explicitly-synthetic/no-tool", "version": "synthetic-only", "sha256": "0" * 64},
            "native_process_control": {"schema_version": 1, "backend": "linux-x86-64-libc-syscall-pidfd-version-1",
                "mode": "observe_configured_target_for_zero_credit_diagnostic", "expected_target_contract": None, "qualification": False,
                "python": self.module.PYTHON, "helper_hashes": {name: hashes[self.module.PREFIX + name]
                    for name in ("linux_pidfd_process_control.py", "native_process_evidence.py")}},
            "configured_native_process_control": {"schema_version": 1, "protocol": "configured-allocation-native-process-control-version-1",
                "hardware_class": "jaguar03_rtx_a4500", "standing_gpu_descriptor_held": False,
                "physical_gres_mapping_verified": False, "kernel_device_enforcement_verified": False, "configured_isolation_trust": True,
                "helper_hashes": {name: hashes[self.module.PREFIX + name] for name in self.module.HELPERS
                    if name != "diagnose_configured_native_process.py"}}}
        for name,value in (("measurement_plan.json",self.plan),("manifest.yaml",self.plan),
                ("config_resolved.yaml",{**self.module.ZERO,"budget":dict(self.module.BUDGET)})):
            self.write(self.unit / name,value)
        import shlex
        producer = [self.module.PYTHON,"-I","-S","-B",str(self.unit / "code/diagnose_configured_native_process.py"),"--prepared-unit",str(self.unit)]
        (self.unit / "command.txt").write_text(shlex.join(producer) + "\n")
        (self.unit / "experiment_background.md").write_text("Synthetic zero-environment launcher fixture only.\n")
        self.write(self.unit / "measurement_prepared.json", {"plan_sha256":self.digest(self.unit / "measurement_plan.json")})
        for name in ("manifest.yaml","config_resolved.yaml","command.txt","experiment_background.md"):
            (self.packet / name).write_bytes((self.unit / name).read_bytes())
        self.write(self.packet / "progress.json", {"kind":self.module.KIND,"state":"prepared","scientific_credit":0})
        (self.packet / "job.sbatch").write_bytes(self.module.expected_job_script())
        command=["sbatch","--hold","--parsable","--no-requeue","--partition=gpu","--account=sl5nw_base",
            "--qos=csresnolim","--reservation=sl5nw_168","--nodelist=jaguar03","--nodes=1","--ntasks=1","--cpus-per-task=1","--mem=2G",
            "--gres=gpu:nvidia_rtx_a4500:1","--time=00:05:00",str(self.packet / "job.sbatch")]
        self.write(self.packet / "held_submission_command.json",command)
        checks=[*(self.unit / "code" / name for name in self.module.HELPERS),self.unit / "measurement_plan.json",self.unit / "measurement_prepared.json"]
        (self.packet / "source_checksums.sha256").write_text("".join(self.digest(path) + "  " + str(path) + "\n" for path in checks))
        self.refreeze_initial()
        for name in ("submitted_jobids.txt","submitted_jobids_uva.txt","submissions.tsv"):(self.run / "slurm" / name).write_text("")
        (self.run / "queue/metadata.lock").write_text("")
        self.module.prepare_binding_once()
        self.write(self.packet / "packet_static_verification.json", {"passed":True,"synthetic_fixture":True,
            "prepared_binding":self.module.reference(self.packet / "prepared_launch_binding.json")})
        frozen = [*self.module.expected_initial_paths(),*(self.packet / name for name in (
            "immutable_artifacts.json","prepared_launch_binding.json","packet_static_verification.json","launch_diagnostic_once.py","test_launch_diagnostic_once.py","schema/diagnose_configured_native_process_approved.py"))]
        self.write(self.packet / "launcher_source_freeze.json", {"files":{str(path):self.digest(path) for path in frozen}})
        self.write(self.packet / "launcher_independent_review.json", {"kind":"independent_configured_native_diagnostic_prepared_launcher_review",
            "passed":True,"synthetic_fixture":True,"source_freeze":self.module.reference(self.packet / "launcher_source_freeze.json"),
            "prepared_binding":self.module.reference(self.packet / "prepared_launch_binding.json")})
        self.publication=self.commit();self.job="910001";self.calls=[];self.fail_release=False

    def refreeze_initial(self):
        """Freeze private small canonical artifacts; no actual study data is copied."""
        self.write(self.packet / "immutable_artifacts.json", {"files": {str(path):self.file_record(path)
            for path in self.module.expected_initial_paths()}})

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
            "kind":"independent_configured_native_diagnostic_prepared_launcher_review",
            "prepared_binding":self.module.reference(self.packet / "prepared_launch_binding.json"),
            "source_freeze": self.module.reference(self.packet / "launcher_source_freeze.json")})
        publication = self.commit()
        with self.assertRaisesRegex(ValueError, "mandatory source"):
            self.module.check_publication(publication)

    def test_one_submission_appends_and_fsyncs_all_three_ledgers_then_refuses_reuse(self):
        """The implementation writes actual private ledger files before making a held receipt."""
        receipt = self.make_held()
        row = self.module.submission_row(self.job, self.plan)
        self.assertEqual(receipt["kind"], "owned_configured_native_process_diagnostic_submission")
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
        self.make_held()
        for index, replacement in enumerate(("UserId=other(1)", "ReqTRES=cpu=1,mem=2G,node=1,gres/gpu=2", "Reservation=synthetic_other_reservation")):
            text = self.held_text()
            if index == 0: text = text.replace("UserId=sl5nw(2618919)", replacement)
            elif index == 1: text = text.replace("ReqTRES=cpu=1,mem=2G,node=1,gres/gpu=1", replacement)
            else: text = text.replace("Reservation=sl5nw_168", replacement)
            with mock.patch.object(self.module, "scheduler", return_value=SimpleNamespace(stdout=text.encode(), returncode=0)):
                with self.assertRaises(ValueError):
                    self.module.query_held(self.job, "rejected_private_query_" + str(index), self.plan)

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
        """The final durable pair-only receipts satisfy the actual new canonical metadata reader."""
        self.make_held(); self.prepare_fixed_release(); events = []; original = self.module.publish
        def publish(path,value):
            events.append(Path(path).name)
            if Path(path).name == "owned_launch_inputs.json":
                self.assertTrue((self.packet / "release_receipt.json").is_file())
            return original(path,value)
        with mock.patch.object(self.module,"scheduler",side_effect=self.fake_scheduler),mock.patch.object(self.module,"publish",side_effect=publish):
            self.module.release_once(self.publication)
        self.assertLess(events.index("release_receipt.json"),events.index("owned_launch_inputs.json"))
        owned=self.module.document(self.unit / "owned_launch_inputs.json")
        for name in ("plan","submission","release"):self.assertEqual(set(owned[name]),{"path","sha256"})
        source=ast.parse((ACTUAL_PACKET / "schema/diagnose_configured_native_process_approved.py").read_text())
        node=next(item for item in source.body if isinstance(item,ast.FunctionDef) and item.name=="owned_launch")
        identity=SimpleNamespace(read_json=self.module.document,receipt=self.module.reference,require=self.module.require)
        def reference_value(ref,metadata):
            self.assertEqual(ref,self.module.reference(ref["path"]))
            return self.module.document(ref["path"])
        namespace={"Path":Path,"identity":identity,"native":SimpleNamespace(DIAGNOSTIC=self.module.KIND,reference_value=reference_value),
            "os":SimpleNamespace(environ={"SLURM_JOB_ID":self.job,"SLURM_CPUS_PER_TASK":"1","SLURM_MEM_PER_NODE":"2048"})}
        exec(compile(ast.Module(body=[node],type_ignores=[]),"actual_new_owned_launch_metadata_only","exec"),namespace)
        self.assertEqual(namespace["owned_launch"](self.unit,self.plan,SimpleNamespace()),owned)
        with mock.patch.object(self.module,"scheduler",side_effect=self.fake_scheduler):
            with self.assertRaisesRegex(ValueError,"preserve the previous attempt"):self.module.release_once(self.publication)
        self.assertEqual(sum(command[:2]==["scontrol","release"] for command in self.calls),1)

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

    def test_late_binding_has_only_actual_source_and_plan_and_cannot_be_reused(self):
        binding=self.module.document(self.packet / "prepared_launch_binding.json")
        self.assertEqual(binding["plan"],self.module.reference(self.unit / "measurement_plan.json"))
        self.assertEqual(binding["source_commit"],self.plan["helpers"]["source_commit"])
        self.assertIsNone(binding["actual_job_id"])
        self.assertFalse(binding["actual_release_performed"])
        self.assertEqual(len(binding["canonical_inputs"]),21)
        with mock.patch.object(self.module,"scheduler",side_effect=AssertionError("no native action")):
            with self.assertRaisesRegex(ValueError,"preserve the previous attempt"):self.module.prepare_binding_once()

    def test_missing_binding_and_changed_binding_refuse_without_scheduler(self):
        path=self.packet / "prepared_launch_binding.json";saved=path.read_bytes();path.unlink()
        with mock.patch.object(self.module,"scheduler",side_effect=AssertionError("no native action")):
            with self.assertRaises(FileNotFoundError):self.module.submit_held_once(self.publication)
            path.write_bytes(saved);value=self.module.document(path);value["source_commit"]="f" * 40;self.write(path,value)
            with self.assertRaisesRegex(ValueError,"prepared binding changed"):self.module.submit_held_once(self.publication)
        self.assertFalse((self.packet / "submission_intent.json").exists())

    def test_typed_budget_native_and_wrapper_substitution_refuse(self):
        paths=[self.unit / "measurement_plan.json",self.unit / "manifest.yaml"]
        original=[path.read_bytes() for path in paths]
        for field,value in (("maximum_active_seconds",True),("gpu_uuid","synthetic"),("native_process_control",{})):
            for path in paths:
                plan=dict(self.plan);plan[field]=value;self.write(path,plan)
            self.write(self.unit / "measurement_prepared.json",{"plan_sha256":self.digest(paths[0])})
            (self.packet / "manifest.yaml").write_bytes(paths[1].read_bytes());self.refreeze_initial()
            with self.assertRaises(ValueError):self.module.prepared_records()
            for path,raw in zip(paths,original):path.write_bytes(raw)
        (self.packet / "job.sbatch").write_text("#!/bin/sh\nexit 0\n");self.refreeze_initial()
        with self.assertRaisesRegex(ValueError,"Slurm wrapper"):self.module.prepared_records()

    def test_owned_membership_checked_before_query_even_with_decimal_foreign_id(self):
        with mock.patch.object(self.module,"scheduler",side_effect=AssertionError("no foreign scheduler query")):
            with self.assertRaises(FileNotFoundError):self.module.query_held(self.job,"unknown",self.plan)
        self.make_held()
        with mock.patch.object(self.module,"scheduler",side_effect=AssertionError("no foreign scheduler query")):
            with self.assertRaisesRegex(ValueError,"not this exact owned submission"):
                self.module.query_held("910002","foreign",self.plan)

    def test_duplicate_scheduler_fields_and_tres_refuse(self):
        self.make_held()
        for addition in (" JobId=" + self.job," ReqTRES=cpu=1,mem=2G,node=1,gres/gpu=1"):
            with mock.patch.object(self.module,"scheduler",return_value=SimpleNamespace(stdout=(self.held_text().strip()+addition).encode(),returncode=0)):
                with self.assertRaisesRegex(ValueError,"duplicate"):self.module.query_held(self.job,"duplicates",self.plan)
        raw=self.held_text().replace("ReqTRES=cpu=1,mem=2G,node=1,gres/gpu=1","ReqTRES=cpu=1,cpu=1,mem=2G,node=1,gres/gpu=1")
        with mock.patch.object(self.module,"scheduler",return_value=SimpleNamespace(stdout=raw.encode(),returncode=0)):
            with self.assertRaisesRegex(ValueError,"duplicate"):self.module.query_held(self.job,"duplicated_tres",self.plan)

    def test_all_three_ledger_fsyncs_precede_submission_receipt(self):
        events=[];append=self.module.append_member;publish=self.module.publish
        def track_append(path,member):
            result=append(path,member);events.append("fsynced:"+Path(path).name);return result
        def track_publish(path,value):
            if Path(path).name=="submission_receipt.json":
                self.assertEqual(events,["fsynced:submitted_jobids.txt","fsynced:submitted_jobids_uva.txt","fsynced:submissions.tsv"])
                self.assertTrue((self.packet / "submission_ledgers_fsynced.json").is_file())
            return publish(path,value)
        with mock.patch.object(self.module,"append_member",side_effect=track_append),mock.patch.object(self.module,"publish",side_effect=track_publish):self.make_held()


if __name__ == "__main__":
    unittest.main()
