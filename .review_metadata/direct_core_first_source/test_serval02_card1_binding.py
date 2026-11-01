"""Use synthetic host/device records and a real temporary lock; grant no GPU qualification."""

import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import serval02_card1_binding as core


class DirectBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_temporary = tempfile.TemporaryDirectory(); cls.source = Path(cls.source_temporary.name)
        for name in core.HELPERS:
            target = cls.source / core.PREFIX / name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(SCRIPTS / name, target)
        for command in (["init", "-q"], ["add", "."], ["-c", "user.name=Metadata Test", "-c", "user.email=metadata@example.invalid", "commit", "-qm", "Record direct metadata fixture"]):
            subprocess.run(["git", "-C", str(cls.source), *command], check=True)
        cls.commit = subprocess.check_output(["git", "-C", str(cls.source), "rev-parse", "HEAD"], text=True).strip()

    @classmethod
    def tearDownClass(cls):
        cls.source_temporary.cleanup()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name); self.unit = self.run / "data/synthetic-direct-unit"; self.unit.mkdir(parents=True)
        self.lock_path = self.run / "existing_card_one.lock"; self.fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o644)
        os.fchmod(self.fd, 0o644); fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB); self.addCleanup(os.close, self.fd)
        info = os.fstat(self.fd)
        self.profile = {**core.PROFILE, "standing_lock": str(self.lock_path), "lock_device": info.st_dev, "lock_inode": info.st_ino, "lock_mode": 0o644}
        self.profile_patch = patch.object(core, "PROFILE", self.profile); self.profile_patch.start(); self.addCleanup(self.profile_patch.stop)
        self.process = core.current_process(); self.process["identity"]["hostname"] = "serval02"
        self.identity = self.process["identity"]
        self.plan = {"kind": "selected_configuration_measurement", "mode": "correctness", "hardware_class": "h100",
            "gpu_uuid": self.profile["gpu_uuid"], "source_commit": core.SOURCE_COMMIT, "source_root": "/synthetic/frozen/source",
            "campaign_run_dir": str(self.run), "unit_id": self.unit.name,
            "helpers": {"source_root": str(self.source), "source_commit": self.commit,
                "files": {core.PREFIX + name: core.receipt(SCRIPTS / name)["sha256"] for name in core.HELPERS}},
            "direct_binding": {"schema_version": 1, "profile": self.profile, "capability": core.CAPABILITY, "consumer": "selected_measurement",
                "physical_identity": core.pending_identity(), "frozen_inputs": {"path": str(self.unit / "direct_history/inputs.json"), "sha256": "0" * 64},
                "current_launch_inputs_path": str(self.unit / "direct_current_launch.json")}}
        self.write(self.unit / "measurement_plan.json", self.plan)
        self.row = {"pid": self.identity["pid"], "start_time": self.identity["process_start_time"], "hostname": "serval02",
            "kind": self.plan["kind"], "unit": self.unit.name, "source_commit": core.SOURCE_COMMIT,
            "source_root": self.plan["source_root"], "helper_commit": self.commit, "gpu_uuid": self.profile["gpu_uuid"]}
        self.write(self.run / "processes.json", [self.row]); self.write(self.unit / "direct_current_process_ledger_snapshot.json", [self.row])
        self.launch = {"schema_version": 1, "kind": "serval02_card1_current_launch", "plan": core.receipt(self.unit / "measurement_plan.json"),
            "process_row": self.row, "process_ledger": {"source_path": str(self.run / "processes.json"),
                "snapshot": core.receipt(self.unit / "direct_current_process_ledger_snapshot.json")}}
        self.write(self.unit / "direct_current_launch.json", self.launch)
        self.history = {"inputs": self.plan["direct_binding"]["frozen_inputs"], "comparisons": 120, "stop": {"jobs": 25}}
        self.history_patch = patch.object(core, "validate_history", return_value=self.history); self.history_patch.start(); self.addCleanup(self.history_patch.stop)
        lock = core.lock_metadata(self.fd)
        self.metadata = {"schema_version": 1, "kind": "serval02_card1_process_metadata", "profile": self.profile, "capability": core.CAPABILITY,
            "role": "owner", "source": self.plan["helpers"], "before": self.process, "after": copy.deepcopy(self.process),
            "environment": {"CUDA_VISIBLE_DEVICES": self.profile["gpu_uuid"]}, "lock_before": lock, "lock_after": copy.deepcopy(lock),
            "current_launch": self.launch, "history": self.history, **core.BOUNDARY}
        self.binding = self.physical_binding()

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(core.encoded(value)); return path

    def query(self, command, stdout):
        return {"command": command, "returncode": 0, "capture_status": "completed", "byte_limit": 32768,
            "stdout_scope": "complete_raw_utf8", "stdout": stdout, "stderr": "", "stdout_full_sha256": core.digest(stdout.encode()),
            "stdout_captured_bytes": len(stdout.encode()), "stderr_full_sha256": core.digest(b""), "stderr_captured_bytes": 0}

    def physical_binding(self):
        gpu, bus = self.profile["gpu_uuid"], "0000:65:00.0"
        driver = str(self.unit / "unavailable_node_driver/libcuda.so.590.48.01")
        target = self.unit / "captured_driver" / Path(driver).name; target.parent.mkdir(); target.write_bytes(b"synthetic direct driver")
        native = {"pid": self.identity["pid"], "jax_local_hardware_id": 0, "jax_device_kind": "NVIDIA H100 NVL", "cuda_device_count": 1,
            "cuda_device_handle": 0, "current_context": 1234, "current_context_device": 0,
            "jax_identity_basis": "sole local JAX GPU, sole CUDA-visible ordinal, and that device's current CUDA context",
            "api_return_codes": {name: 0 for name in ("cuDeviceGetCount", "cuDeviceGet", "cuCtxGetCurrent", "cuCtxGetDevice", "cuDeviceGetUuid_v2", "cuDeviceGetPCIBusId")},
            "gpu_uuid": gpu, "pci_address": bus, "driver_library": driver, "driver_library_sha256": core.receipt(target)["sha256"],
            "mapped_driver_bytes": "1-2 r-xp 0 0 1 " + driver + "\n", "mapped_driver_sha256": core.digest(("1-2 r-xp 0 0 1 " + driver + "\n").encode())}
        raw = f"GPU UUID: {gpu}\nModel: NVIDIA H100 NVL\nBus Location: {bus}\nDevice Minor: 3\n"
        custody = {"process": self.identity, "lock": self.metadata["lock_after"]}
        return {"schema_version": 1, "kind": "serval02_card1_current_process_resolved_device_binding", "profile": self.profile,
            "capability": core.CAPABILITY, "plan": core.receipt(self.unit / "measurement_plan.json"), "metadata_sha256": core.digest(core.encoded(self.metadata)),
            "process": self.identity, "original_environment": self.metadata["environment"], "gpu_uuid": gpu, "pci_address": bus,
            "physical_minor": 3, "physical_total_bytes": 95830 * 1024 * 1024, "native": native,
            "hardware": self.query(["nvidia-smi", "-i", gpu, "--query-gpu=uuid,name,pci.bus_id,memory.total,driver_version", "--format=csv,noheader,nounits"], f"{gpu}, NVIDIA H100 NVL, {bus}, 95830, 590.48.01\n"),
            "proc": {"path": "/proc/driver/nvidia/gpus/" + bus + "/information", "raw": raw, "sha256": core.digest(raw.encode())},
            "device": {"path": "/dev/nvidia3", "character_device": True, "symlink": False, "major": 195, "minor": 3},
            "driver_capture": {"schema_version": 1, "kind": "captured_mapped_cuda_driver_image", "pid": self.identity["pid"],
                "original": {"path": driver, "sha256": native["driver_library_sha256"]}, "captured": core.receipt(target), "bytes": target.stat().st_size},
            "custody_before": custody, "custody_after": copy.deepcopy(custody), "startup_memory_observed": False,
            "other_gpu_processes_observed": False, **core.BOUNDARY}

    def test_known_authorization_and_current_binding_are_distinct(self):
        core.validate_plan(self.plan); core.validate_metadata(self.plan, self.metadata); core.validate_binding(self.plan, self.metadata, self.binding)
        self.assertIsNone(self.plan["direct_binding"]["physical_identity"]["pci_address"])
        core.validate_metadata(self.plan, json.loads(core.encoded(self.metadata)))

    def test_wrong_profiles_devices_modes_and_consumers_refuse(self):
        changes = [lambda p: p.pop("direct_binding"), lambda p: p.update(gpu_uuid="GPU-other"),
            lambda p: p["direct_binding"]["profile"].update(schema_version=True), lambda p: p.update(hardware_class="jaguar03_rtx_a4500"),
            lambda p: p["direct_binding"]["physical_identity"].update(physical_minor=0), lambda p: p.update(configured_isolation={})]
        for mode in ("packing-point", "learning-check", "final", "scaling"):
            changes.append(lambda p, mode=mode: p.update(mode=mode))
        for change in changes:
            plan = copy.deepcopy(self.plan); change(plan)
            with self.subTest(change=change), self.assertRaises(ValueError): core.validate_plan(plan)

    def test_current_own_process_and_original_selector_are_mandatory(self):
        for change in (lambda v: v["before"]["identity"].update(uid=1), lambda v: v["after"]["identity"].update(process_start_time="1"),
                lambda v: v["environment"].update(SLURM_JOB_ID="123"), lambda v: v["environment"].update(CUDA_VISIBLE_DEVICES="0"),
                lambda v: v["before"]["identity"].update(hostname="serval05")):
            value = copy.deepcopy(self.metadata); change(value)
            with self.subTest(change=change), self.assertRaises(ValueError): core.validate_metadata(self.plan, value)

    def test_actual_temporary_lock_descriptor_proves_own_flock_without_reacquiring(self):
        value = core.lock_metadata(self.fd); core.validate_lock(value, os.getpid())
        other = os.open(self.lock_path, os.O_RDWR); self.addCleanup(os.close, other)
        # A different open file description sees the same inode but does not own this FLOCK.
        with self.assertRaises(ValueError): core.validate_lock(core.lock_metadata(other), os.getpid())

    def test_real_inherited_descriptor_retains_original_holder_pid(self):
        read, write = os.pipe(); parent = os.getpid(); child = os.fork()
        if child == 0:
            try:
                os.close(read); observed = core.lock_metadata(self.fd)
                core.validate_lock(observed, parent)
                try:
                    core.validate_lock(observed, os.getpid()); wrong_holder_rejected = False
                except ValueError:
                    wrong_holder_rejected = True
                os.write(write, core.encoded({"pid": os.getpid(), "parent": parent, "wrong_holder_rejected": wrong_holder_rejected}))
                os.close(write); os._exit(0)
            except BaseException:
                os._exit(1)
        os.close(write)
        try:
            value = json.loads(os.read(read, 8192)); observed, status = os.waitpid(child, 0)
        finally:
            os.close(read)
        self.assertEqual(observed, child); self.assertEqual(status, 0)
        self.assertEqual(value["parent"], parent); self.assertNotEqual(value["pid"], parent); self.assertTrue(value["wrong_holder_rejected"])

    def process_record(self, pid, parent, birth, group):
        value = copy.deepcopy(self.process); fields = value["stat"][value["stat"].rfind(")") + 2:].split()
        fields[1], fields[2], fields[19] = str(parent), str(group), str(birth)
        value["stat"] = str(pid) + " (synthetic_direct_process) " + " ".join(fields)
        value["stat_sha256"] = core.digest(value["stat"].encode())
        value["identity"].update(pid=pid, parent_pid=parent, process_start_time=str(birth), process_group=group)
        return value

    def test_inherited_profile_metadata_proves_holder_not_child_ownership(self):
        self.plan.update(kind="selected_configuration_profile", mode="profile")
        self.plan["direct_binding"]["consumer"] = "component_profile"
        self.write(self.unit / "measurement_plan.json", self.plan)
        self.row["kind"] = self.plan["kind"]
        self.write(self.run / "processes.json", [self.row]); self.write(self.unit / "direct_current_process_ledger_snapshot.json", [self.row])
        self.launch["plan"] = core.receipt(self.unit / "measurement_plan.json")
        self.launch["process_ledger"]["snapshot"] = core.receipt(self.unit / "direct_current_process_ledger_snapshot.json")
        self.write(self.unit / "direct_current_launch.json", self.launch)
        self.metadata["current_launch"] = self.launch
        parent_path = self.write(self.unit / "supervisor/direct_process_metadata.json", self.metadata)
        child_pid, profiler_pid = self.identity["pid"] + 1000, self.identity["pid"] + 500
        child = self.process_record(child_pid, profiler_pid, "9000", profiler_pid)
        owner = {"pid": profiler_pid, "process_start_time": "8000", "parent_pid": self.identity["pid"],
            "parent_start_time": self.identity["process_start_time"], "command": ["synthetic-nsight"]}
        value = copy.deepcopy(self.metadata)
        value.update(role="inherited", before=child, after=copy.deepcopy(child), parent_metadata=self.metadata,
            parent_lock_receipt=core.receipt(parent_path), profiler_process=owner,
            capture_release={**owner, "capture_identity": {"hostname": "serval02", "pid": child_pid, "process_start_time": "9000"}},
            holder_now=core.capture_owner_process(self.identity["pid"]))
        profiler = self.process_record(profiler_pid, self.identity["pid"], "8000", profiler_pid)
        value["profiler_now"] = {"pid": profiler_pid, **{key: profiler[key] for key in ("stat", "stat_sha256", "uid_bytes", "uid_sha256")}}
        core.validate_metadata(self.plan, value)
        for change in (lambda v: v["lock_before"].update(fd=999999), lambda v: v["profiler_process"].update(parent_start_time="1"),
                lambda v: v["before"]["identity"].update(process_group=1)):
            changed = copy.deepcopy(value); change(changed)
            with self.subTest(change=change), self.assertRaises(ValueError): core.validate_metadata(self.plan, changed)

    def exit_fixture(self, run=None):
        collector = core.adjacent("collect_worker_exit_receipts")
        source = core.SOURCE_COMMIT; stamp = "2026-11-01T05:00:00+00:00"
        launches = [{"kind": "queue_worker", "hostname": "serval02", "pid": 77, "start_time": "100", "time": stamp}]
        workers = [{"worker_id": "worker_" + str(i), "hostname": "jaguar03", "worker_pid": 100 + i,
            "worker_start_time": "110", "slurm_job_id": str(i), "time": stamp} for i in range(1, 26)]
        runner = {"unit_id": "synthetic_case", "attempt": 1, "hostname": "serval02", "runner_pid": 79,
            "runner_start_time": "101", "slurm_job_id": "", "finished_at": stamp}
        texts = {"processes.json": json.dumps(launches), "slurm/submissions.tsv": "\n".join(str(i) + "\tqueue-worker\t" + source + "\t/job/" + str(i) + ".sh" for i in range(1, 26)) + "\n",
            "slurm/submitted_jobids_uva.txt": "\n".join(map(str, range(1, 26))) + "\n", "queue/leases/synthetic.json": json.dumps(runner)}
        texts.update({"queue/workers/worker_" + row["slurm_job_id"] + ".json": json.dumps(row) for row in workers})
        workers = [json.loads(texts[key]) for key in sorted(texts) if key.startswith("queue/workers/")]
        jobs = collector.required_jobs(texts["slurm/submissions.tsv"], workers, [runner], texts["slurm/submitted_jobids_uva.txt"].splitlines())
        inputs = {"files": {key: {"text": raw, "sha256": core.digest(raw.encode())} for key, raw in texts.items()},
            "launches": launches, "workers": workers, "runners": [runner], "jobs": jobs}
        plan = {"schema_version": 1, "kind": "worker_exit_observation_plan", "owner": "sl5nw", "owner_uid": 2618919,
            "run_dir": str(self.run if run is None else run), "prepared_at": stamp, "collector_sha256": core.receipt(SCRIPTS / "collect_worker_exit_receipts.py")["sha256"], "inputs": inputs}
        observed = "2026-11-01T06:00:00+00:00"
        scheduler = {"command": ["sacct", "-X", "--jobs=" + ",".join(jobs), "--noheader", "--parsable2", "--format=JobIDRaw,User,State,ExitCode,NodeList,Start,End"],
            "stdout": "\n".join(job + "|sl5nw|COMPLETED|0:0|jaguar03|start|end" for job in jobs),
            "squeue_command": ["squeue", "--jobs=" + ",".join(jobs), "--noheader", "--format=%i|%u|%T"], "squeue_stdout": "", "observed_at": observed}
        raw = {"hostname": "serval02", "observed_at": observed, "observations": [{"pid": pid, "expected_birth": birth,
            "observed_birth": None, "observed_uid": None, "original_process_alive": False} for pid, birth in ((77, "100"), (79, "101"))]}
        observations = {"scheduler": scheduler, "direct_processes": [{"hostname": "serval02", "stdout": json.dumps(raw), "observed_at": observed}]}
        rebuilt_jobs = {identifier: {"job_id": identifier, "owner": "sl5nw", "state": "COMPLETED", "state_raw": "COMPLETED",
            "exit_code": "0:0", "node_list": "jaguar03", "start": "start", "end": "end", "submission_row": jobs[identifier], "observed_at": observed} for identifier in jobs}
        direct = {("serval02", row["pid"], row["expected_birth"]): {**row, "hostname": "serval02", "observed_at": observed} for row in raw["observations"]}
        stop, waiting = collector.make_receipt(plan, rebuilt_jobs, direct); self.assertEqual(waiting, [])
        status = {"schema_version": 1, "state": "complete", "waiting": [], "counts": {key: len(inputs[key]) for key in ("workers", "jobs", "launches", "runners")}}
        return plan, observations, stop, status

    def test_saved_exit_raw_evidence_replays_exactly_without_live_ledgers(self):
        plan, observations, stop, status = self.exit_fixture()
        value = core.snapshot_exit_history(plan, observations, stop, status)
        self.assertEqual(value["jobs"], 25); self.assertEqual(value["direct_processes"], 2)
        self.write(self.run / "processes.json", [{"new": "later launch"}])
        self.assertEqual(core.snapshot_exit_history(plan, observations, stop, status), value)

    def test_requeued_live_foreign_missing_or_stale_original_workers_refuse(self):
        plan, observations, stop, status = self.exit_fixture()
        for change in (lambda o: o["scheduler"].update(squeue_stdout="1|sl5nw|PENDING\n"),
                lambda o: o["scheduler"].update(stdout=o["scheduler"]["stdout"].replace("|sl5nw|", "|other|", 1)),
                lambda o: o["scheduler"].update(stdout="\n".join(o["scheduler"]["stdout"].splitlines()[1:])),
                lambda o: o["scheduler"].update(observed_at="2026-10-30T00:00:00+00:00")):
            value = copy.deepcopy(observations); change(value)
            with self.subTest(change=change), self.assertRaises(ValueError): core.snapshot_exit_history(plan, value, stop, status)
        raw = json.loads(observations["direct_processes"][0]["stdout"]); raw["observations"][0].update(original_process_alive=True, observed_birth="100")
        observations["direct_processes"][0]["stdout"] = json.dumps(raw)
        with self.assertRaises(ValueError): core.snapshot_exit_history(plan, observations, stop, status)

    def test_historical_receipt_cannot_omit_a_child_or_change_frozen_bytes(self):
        plan, observations, stop, status = self.exit_fixture()
        wrong = copy.deepcopy(stop); wrong["runners"] = []
        with self.assertRaises(ValueError): core.snapshot_exit_history(plan, observations, wrong, status)
        plan["inputs"]["files"]["processes.json"]["text"] = "[]"
        with self.assertRaises(ValueError): core.snapshot_exit_history(plan, observations, stop, status)

    def test_real_existing_reader_rederives_frozen_synthetic_120_case_history(self):
        """Use the unchanged raw cohort reader, never a supplied completion Boolean."""
        import test_post_comparison_validation as fixtures
        fixture = fixtures.PostComparisonTests("runTest"); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        registry_path = fixture.directory / "registry.json"
        registry = json.loads(Path(fixture.specification["study_registry"]).read_text())
        expected = sorted(core.adjacent("ideas").IDEAS_BY_NAME.values(), key=lambda entry: entry.position)[:30]
        registry["initial_ideas"] = registry["initial_ideas"][:30]
        for entry, mechanism in zip(registry["initial_ideas"], expected):
            entry.update(name=mechanism.name, position=mechanism.position)
            for rows in entry["results"].values():
                for row in rows:
                    result_path = Path(row["result_path"]); case_path = result_path.with_name("case.json")
                    case, result = json.loads(case_path.read_text()), json.loads(result_path.read_text())
                    case.update(hypothesis=mechanism.name, hypothesis_position=mechanism.position)
                    unsigned = {key: value for key, value in case.items() if key != "specification_digest"}
                    case["specification_digest"] = fixtures.VALIDATION.canonical_digest(unsigned)
                    result.update(idea=mechanism.name, specification_digest=case["specification_digest"])
                    self.write(case_path, case); self.write(result_path, result)
        self.write(registry_path, registry)
        exit_plan, observations, stop, status = self.exit_fixture(fixture.directory)
        for name, record in exit_plan["inputs"]["files"].items():
            path = fixture.directory / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(record["text"])
        unit = fixture.directory / "data/direct_history_test"; history = unit / "direct_history"; history.mkdir(parents=True)
        # Snapshot first, then allow live ledgers to grow. No source path is its own copy.
        candidates = [path for path in fixture.directory.rglob("*") if path.is_file() and not path.is_relative_to(history)]
        entries = []
        for index, original in enumerate(candidates):
            target = history / "file_snapshots" / str(index); target.parent.mkdir(exist_ok=True)
            target.write_bytes(original.read_bytes())
            entries.append({"original_path": str(original), "relative_path": str(target.relative_to(history)), "sha256": core.receipt(target)["sha256"]})
        clone = history / "source"
        subprocess.run(["git", "clone", "--shared", "--no-checkout", "--quiet", str(fixture.source), str(clone)], check=True)
        subprocess.run(["git", "-C", str(clone), "sparse-checkout", "set", "10_jax_exploration_platform/scripts/atari_pixel_campaign", "10_jax_exploration_platform/src/exploration_platform/atari_pixels"], check=True)
        subprocess.run(["git", "-C", str(clone), "checkout", "--quiet", "--detach", fixture.commit], check=True)
        manifest_path = self.write(history / "evidence_manifest.json", {"schema_version": 1, "kind": "portable_raw_evidence", "files": entries,
            "git_roots": [{"original_root": str(fixture.source), "relative_path": "source", "commit": fixture.commit}]})
        gates = {}
        for context in ("h100", "jaguar03_rtx_a4500"):
            original = fixture.directory / (context + "_original_gate.json")
            target = self.write(history / (context + "_gate.json"), {"passed": True, "source_commit": fixture.commit})
            gates[context] = str(original)
            entries.append({"original_path": str(original), "relative_path": target.name, "sha256": core.receipt(target)["sha256"]})
        self.write(manifest_path, {"schema_version": 1, "kind": "portable_raw_evidence", "files": entries,
            "git_roots": [{"original_root": str(fixture.source), "relative_path": "source", "commit": fixture.commit}]})
        admission = self.write(history / "admission.json", {"authorized": True, "source_commit": fixture.commit,
            "source_root": str(fixture.source), "contexts": list(gates), "gate_records": gates})
        registry_copy = history / "registry.json"; registry_copy.write_bytes(registry_path.read_bytes())
        exit_path = self.write(history / "exit_plan.json", exit_plan)
        status["prepared_plan_sha256"] = core.receipt(exit_path)["sha256"]
        documents = {"admission": core.receipt(admission), "registry": core.receipt(registry_copy), "evidence_manifest": core.receipt(manifest_path),
            "exit_plan": core.receipt(exit_path), "exit_observations": core.receipt(self.write(history / "observations.json", observations)),
            "worker_stop": core.receipt(self.write(history / "worker_stop.json", stop)), "exit_status": core.receipt(self.write(history / "status.json", status))}
        inputs_path = self.write(history / "inputs.json", {"schema_version": 1, "kind": "serval02_card1_frozen_history", **documents})
        plan = copy.deepcopy(self.plan); plan.update(campaign_run_dir=str(fixture.directory), unit_id=unit.name,
            source_root=str(fixture.source), source_commit=fixture.commit)
        plan["direct_binding"].update(frozen_inputs=core.receipt(inputs_path), current_launch_inputs_path=str(unit / "direct_current_launch.json"))
        # The fixture source is a metadata-only synthetic substitute, explicitly not actual D28 qualification.
        self.history_patch.stop()
        with patch.object(core, "SOURCE_COMMIT", fixture.commit), patch.object(core, "ORIGINAL_ADMISSION_SHA256", core.receipt(admission)["sha256"]):
            checked = core.validate_history(plan)
            self.assertEqual(checked["comparisons"], 120)
            (fixture.directory / "processes.json").write_text(json.dumps([{"new": "later direct launch"}]))
            self.assertEqual(core.validate_history(plan), checked)
            changed = copy.deepcopy(registry); changed["initial_ideas"][0]["results"]["h100"] = []
            self.write(registry_copy, changed); documents["registry"] = core.receipt(registry_copy)
            self.write(inputs_path, {"schema_version": 1, "kind": "serval02_card1_frozen_history", **documents})
            plan["direct_binding"]["frozen_inputs"] = core.receipt(inputs_path)
            with self.assertRaises(ValueError): core.validate_history(plan)

    def test_changed_lock_path_inode_alias_or_owner_refuses(self):
        for changes in ({"path_symlink": True}, {"fd_target": "/tmp/other"}, {"fdinfo": ""}):
            value = copy.deepcopy(self.metadata["lock_before"]); value.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError): core.validate_lock(value, os.getpid())
        value = copy.deepcopy(self.metadata["lock_before"]); value["fd_stat"]["inode"] += 1
        with self.assertRaises(ValueError): core.validate_lock(value, os.getpid())

    def test_historical_snapshot_and_current_registration_are_separate(self):
        self.assertEqual(core.launch_inputs(self.plan), self.launch)
        self.write(self.run / "processes.json", [self.row, {"hostname": "serval02", "pid": 999999, "start_time": "123"}])
        self.assertEqual(core.launch_inputs(self.plan), self.launch)
        changed = copy.deepcopy(self.launch); changed["process_ledger"]["snapshot"] = core.receipt(self.run / "processes.json")
        self.write(self.unit / "direct_current_launch.json", changed)
        with self.assertRaises(ValueError): core.launch_inputs(self.plan)

    def test_missing_or_duplicate_current_launch_refuses(self):
        for rows in ([], [self.row, self.row]):
            self.write(self.run / "processes.json", rows)
            with self.subTest(rows=rows), self.assertRaises(ValueError): core.launch_inputs(self.plan)

    def test_context_device_driver_total_and_claim_changes_refuse(self):
        changes = [lambda v: v["native"].update(current_context=None), lambda v: v["native"].update(current_context_device=1),
            lambda v: v["native"].update(cuda_device_count=2), lambda v: v["native"].update(gpu_uuid="GPU-other"),
            lambda v: v.update(physical_total_bytes=20470 * 1024 * 1024), lambda v: v.update(kernel_device_enforcement_verified=True),
            lambda v: v.update(kind="uva_current_process_resolved_device_binding"), lambda v: v["device"].update(minor=3.0)]
        for change in changes:
            value = copy.deepcopy(self.binding); change(value)
            with self.subTest(change=change), self.assertRaises(ValueError): core.validate_binding(self.plan, self.metadata, value)

    def test_global_or_foreign_device_queries_and_image_substitution_refuse(self):
        for command in (["nvidia-smi", "-q", "-x"], ["nvidia-smi", "--query-compute-apps=pid,used_memory"], ["nvidia-smi", "-i", "GPU-other"]):
            value = copy.deepcopy(self.binding); value["hardware"]["command"] = command
            with self.subTest(command=command), self.assertRaises(ValueError): core.validate_binding(self.plan, self.metadata, value)
        target = Path(self.binding["driver_capture"]["captured"]["path"]); target.write_bytes(b"changed")
        with self.assertRaises(ValueError): core.validate_binding(self.plan, self.metadata, self.binding)

    def test_missing_inherited_descriptor_fails_before_history_or_cuda(self):
        environment = {"CUDA_VISIBLE_DEVICES": self.profile["gpu_uuid"]}
        with patch.object(core, "current_process", return_value=self.process), patch.dict(os.environ, environment, clear=True),\
                patch.object(core, "validate_history", return_value=self.history) as history:
            with self.assertRaises(OSError): core.collect_metadata(self.plan, self.unit, 999999)
            history.assert_called_once()
        failure = json.loads((self.unit / "direct_process_metadata.json").read_text())
        self.assertIn("capture_error", failure)

    def test_native_failure_retains_attempt_and_never_guesses_a_hardware_query(self):
        device = object()
        def fail(device, observed): observed.update(cuda_device_count=2); raise ValueError("synthetic native failure")
        with patch.object(core, "current_process", return_value=self.process), patch.dict(os.environ, self.metadata["environment"], clear=True),\
                patch.dict(sys.modules, {"jax": SimpleNamespace(local_devices=lambda **kwargs: [device])}),\
                patch.object(core, "native_identity", side_effect=fail), patch.object(core.common, "bounded_query") as query:
            with self.assertRaisesRegex(ValueError, "synthetic native failure"): core.bind_process(self.plan, self.metadata, device, self.unit)
            query.assert_not_called()
        value = json.loads((self.unit / "direct_device_binding_attempt.json").read_text())
        self.assertFalse(value["passed"]); self.assertEqual(value["native"]["cuda_device_count"], 2)
        with self.assertRaises(ValueError): core.bind_process(self.plan, self.metadata, device, self.unit)

    def test_exact_source_closure_and_plan_bytes_are_bound(self):
        core.source_identity(self.plan)
        changed = copy.deepcopy(self.plan); changed["helpers"]["files"].pop(core.PREFIX + core.HELPERS[0])
        with self.assertRaises(ValueError): core.source_identity(changed)
        self.write(self.unit / "measurement_plan.json", {**self.plan, "unreviewed": True})
        with self.assertRaises(ValueError): core.launch_inputs(self.plan)


if __name__ == "__main__":
    unittest.main()
