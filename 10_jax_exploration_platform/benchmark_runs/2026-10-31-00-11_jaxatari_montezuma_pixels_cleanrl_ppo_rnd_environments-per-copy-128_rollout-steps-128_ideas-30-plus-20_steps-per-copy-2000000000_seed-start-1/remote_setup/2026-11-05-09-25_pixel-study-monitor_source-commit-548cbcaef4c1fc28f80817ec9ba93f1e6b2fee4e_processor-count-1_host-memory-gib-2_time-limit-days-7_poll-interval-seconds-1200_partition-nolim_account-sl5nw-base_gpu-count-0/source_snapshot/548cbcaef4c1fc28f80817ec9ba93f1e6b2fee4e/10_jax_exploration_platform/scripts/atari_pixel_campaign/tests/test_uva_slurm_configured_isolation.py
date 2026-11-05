"""Exercise synthetic single-card metadata without granting actual GPU qualification."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import uva_slurm_configured_isolation as core
import test_uva_cgroup_v2_ownership as structural_fixture


class ConfiguredIsolationTests(unittest.TestCase):
    """The real CPU record supplies only structure; all added GPU records are synthetic."""

    @classmethod
    def setUpClass(cls):
        cls.source_temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.source_temporary.name)
        for name in core.HELPERS:
            target = cls.root / core.PREFIX / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(SCRIPTS / name, target)
        subprocess.run(["git", "-C", str(cls.root), "init", "-q"], check=True)
        subprocess.run(["git", "-C", str(cls.root), "add", "."], check=True)
        subprocess.run(["git", "-C", str(cls.root), "-c", "user.name=Metadata Test", "-c", "user.email=metadata@example.invalid",
            "commit", "-qm", "Record synthetic qualifier source"], check=True)
        cls.commit = subprocess.check_output(["git", "-C", str(cls.root), "rev-parse", "HEAD"], text=True).strip()

    @classmethod
    def tearDownClass(cls):
        cls.source_temporary.cleanup()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name); self.unit = self.run / "data/synthetic-owned-unit"
        self.unit.mkdir(parents=True); (self.run / "slurm").mkdir()
        self.structural = structural_fixture.ExplicitUvaOwnershipTests("runTest")
        self.structural.setUp(); self.addCleanup(self.structural.doCleanups)
        self.job, self.step = self.structural.job, self.structural.step
        self.process = copy.deepcopy(self.structural.process)
        self.process.update(hostname="jaguar03", cpu_affinity=list(range(8)))
        self.plan = {"kind": "selected_configuration_measurement", "mode": "correctness", "gpu_uuid": None,
            "hardware_class": "jaguar03_rtx_a4500", "campaign_run_dir": str(self.run), "unit_id": self.unit.name,
            "helpers": {"source_root": str(self.root), "source_commit": self.commit,
                "files": {core.PREFIX + name: core.receipt(SCRIPTS / name)["sha256"] for name in core.HELPERS}},
            "configured_isolation": {"schema_version": 1, "profile": copy.deepcopy(core.PROFILE), "capability": core.CAPABILITY,
                "consumer": "selected_measurement", "resources": {"cpus": 8, "host_memory_gib": 64},
                "physical_identity": core.unresolved_identity(), "owned_inputs_path": str(self.unit / "owned_allocation_inputs.json")}}
        self.plan_path = self.unit / "measurement_plan.json"; self.write(self.plan_path, self.plan)
        script = self.run / "remote_setup" / self.unit.name / "job.sbatch"
        self.row = [self.job, self.plan["kind"], self.commit, str(script)]
        common = {"schema_version": 1, "job_id": self.job, "cluster": "uva", "plan_sha256": core.receipt(self.plan_path)["sha256"],
            "data_unit": str(self.unit), "returncode": 0}
        submission_path, release_path = self.unit / "submission.json", self.unit / "release.json"
        self.write(submission_path, {**common, "kind": "owned_uva_configured_isolation_submission",
            "held_command": ["sbatch", "--hold", "--no-requeue", str(script)], "submission_row": self.row})
        self.write(release_path, {**common, "kind": "owned_uva_configured_isolation_release",
            "command": ["scontrol", "release", self.job], "submission_receipt_sha256": core.receipt(submission_path)["sha256"]})
        self.inputs = {"schema_version": 1, "kind": "uva_configured_isolation_owned_inputs", "job_id": self.job,
            "plan": core.receipt(self.plan_path), "submission": core.receipt(submission_path), "release": core.receipt(release_path), "submission_row": self.row}
        for key, filename, raw in (("ledger", "submitted_jobids_uva.txt", self.job + "\n"),
                ("submissions", "submissions.tsv", "\t".join(self.row) + "\n")):
            source = self.run / "slurm" / filename; source.write_text(raw)
            snapshot = self.unit / ("configured_isolation_owned_job_ids.txt" if key == "ledger" else "configured_isolation_owned_submissions.tsv"); snapshot.write_text(raw)
            self.inputs[key] = {"snapshot": core.receipt(snapshot), "source_path": str(source)}
        self.write(Path(self.plan["configured_isolation"]["owned_inputs_path"]), self.inputs)
        chain = [{"path": path, "uid": uid, "mode": 0o755, "inode": index + 1, "device": 1,
            "directory": True, "symlink": False} for index, (path, uid) in enumerate((("/", 0), ("/etc", 0), ("/etc/slurm", 64030)))]
        # Explicit synthetic configuration bytes; production hashes are never changed by the test.
        self.configuration = {"service": {"name": "slurm", "uid": 64030, "gid": 64030}, "cache_absent": True,
            "files": {path: {"path": path, "uid": 64030, "mode": 0o644, "regular_file": True, "symlink": False,
                "sha256": core.digest(("synthetic " + path + "\n").encode()), "read_status": "read", "settings": [],
                "includes": [{"line": 121, "token": "/etc/slurm/nodes.conf"}, {"line": 122, "token": "/etc/slurm/partitions.conf"}] if path.endswith("/slurm.conf") else [],
                "read_trace": {"complete_file": True, "inode_changed_during_read": False, "path_replaced_during_read": False,
                    "bytes_read": len("synthetic " + path + "\n"),
                    "descriptor_before": {"path": path, "owner_uid": 64030, "mode": 0o644, "regular_file": True, "byte_count": len("synthetic " + path + "\n")},
                    "descriptor_after": {"path": path, "owner_uid": 64030, "mode": 0o644, "regular_file": True, "byte_count": len("synthetic " + path + "\n")}},
                "directory_chain": copy.deepcopy(chain), "directory_chain_after": copy.deepcopy(chain)} for path in core.CONFIG_FILES}}
        self.synthetic_hashes = {path: row["sha256"] for path, row in self.configuration["files"].items()}
        self.excerpt_patch = patch.object(core, "CONFIG_EXCERPT_HASHES", {path: core.digest(core.encoded(row["settings"])) for path, row in self.configuration["files"].items()})
        self.excerpt_patch.start(); self.addCleanup(self.excerpt_patch.stop)
        resource = "cpu=8,mem=64G,node=1,gres/gpu=1"
        self.metadata = {"schema_version": 1, "kind": "uva_configured_isolation_metadata", "profile": copy.deepcopy(core.PROFILE),
            "capability": core.CAPABILITY, "source": self.plan["helpers"], "inputs": self.inputs,
            "inputs_receipt": core.receipt(self.plan["configured_isolation"]["owned_inputs_path"]),
            "process": self.process, "environment": {"SLURM_JOB_ID": self.job, "SLURM_STEP_ID": self.step,
                "SLURM_CONF": "/etc/slurm/slurm.conf", "SLURM_JOB_GPUS": "6", "SLURM_STEP_GPUS": "6",
                "SLURM_GPUS_ON_NODE": "1", "SLURM_CPUS_PER_TASK": "8", "CUDA_VISIBLE_DEVICES": "0"},
            "cgroup": copy.deepcopy(self.structural.value), "configuration": self.configuration, **core.BOUNDARY}
        for key, stdout, command in (("job", f"JobId={self.job} UserId=sl5nw(2618919) JobState=RUNNING Account=sl5nw_base Partition=gpu QOS=csresnolim NodeList=jaguar03 NumNodes=1 NumCPUs=8 NumTasks=1 CPUs/Task=8 TresPerNode=gres/gpu:nvidia_rtx_a4500:1 Requeue=0 Restarts=0 AllocTRES={resource} ReqTRES={resource}", ["scontrol", "show", "job", "-d", self.job]),
                ("step", f"StepId={self.job}.{self.step} State=RUNNING UserId=2618919 NodeList=jaguar03 Tasks=1 TRES={resource}", ["scontrol", "show", "step", "-d", self.job + "." + self.step]),
                ("running_configuration", "SLURM_CONF = /etc/slurm/slurm.conf\nSlurmUser = slurm(64030)\nProctrackType = proctrack/cgroup\nGresTypes = gpu\nTaskPlugin = task/affinity,task/cgroup\nConstrainDevices = no\n", ["scontrol", "show", "config"])):
            self.metadata[key] = {"command": command, "capture_status": "completed", "returncode": 0, "stdout": stdout, "stderr": ""}
            self.rehash_query(self.metadata[key], configuration=key == "running_configuration")
        self.hash_patch = patch.object(core, "CONFIG_FILES", self.synthetic_hashes); self.hash_patch.start(); self.addCleanup(self.hash_patch.stop)
        gpu, bus, minor = "GPU-12345678-1234-1234-1234-123456789abc", "0000:65:00.0", 3
        driver = str(self.unit / "unavailable_node_driver/libcuda.so.580.159.04")
        target = self.unit / "captured_driver" / Path(driver).name; target.parent.mkdir(); target.write_bytes(b"synthetic driver")
        self.binding = {"schema_version": 1, "kind": "uva_current_process_resolved_device_binding", "profile": copy.deepcopy(core.PROFILE),
            "capability": core.CAPABILITY, "plan_sha256": core.digest(core.encoded(self.plan)), "metadata_sha256": core.digest(core.encoded(self.metadata)),
            "process": self.process, "process_after": copy.deepcopy(self.process), "original_environment": self.metadata["environment"],
            "environment_after": copy.deepcopy(self.metadata["environment"]), "gpu_uuid": gpu, "pci_address": bus, "physical_minor": minor,
            "native": {"pid": self.process["pid"], "cuda_device_count": 1, "jax_local_hardware_id": 0, "cuda_device_handle": 0,
                "jax_device_kind": core.PROFILE["model"], "api_return_codes": {k: 0 for k in ("cuDeviceGetCount", "cuDeviceGet", "cuDeviceGetUuid_v2", "cuDeviceGetPCIBusId", "cuCtxGetCurrent", "cuCtxGetDevice")},
                "current_context": 1234, "current_context_device": 0,
                "jax_identity_basis": "sole local JAX GPU, sole CUDA-visible ordinal, and that device's current CUDA context",
                "gpu_uuid": gpu, "pci_address": bus, "driver_library": driver, "driver_library_sha256": core.receipt(target)["sha256"],
                "mapped_driver_bytes": "1-2 r-xp 0 0 1 " + driver + "\n", "mapped_driver_sha256": core.digest(("1-2 r-xp 0 0 1 " + driver + "\n").encode())},
            "driver_capture": {"schema_version": 1, "kind": "captured_mapped_cuda_driver_image", "pid": self.process["pid"],
                "original": {"path": driver, "sha256": core.receipt(target)["sha256"]}, "captured": core.receipt(target), "bytes": target.stat().st_size},
            "hardware": {"command": ["nvidia-smi", "-i", gpu, "--query-gpu=uuid,name,pci.bus_id,memory.total,driver_version", "--format=csv,noheader,nounits"],
                "capture_status": "completed", "returncode": 0, "stdout": f"{gpu}, NVIDIA RTX A4500, {bus}, 20470, 580.159.04\n", "stderr": ""},
            "proc": {"path": "/proc/driver/nvidia/gpus/" + bus + "/information",
                "raw": f"GPU UUID: {gpu}\nModel: NVIDIA RTX A4500\nBus Location: {bus}\nDevice Minor: {minor}\n"},
            "device": {"path": "/dev/nvidia3", "character_device": True, "symlink": False, "major": 195, "minor": minor},
            "startup_memory_observed": False, "other_gpu_processes_observed": False, **core.BOUNDARY}
        self.binding["proc"]["sha256"] = core.digest(self.binding["proc"]["raw"].encode())
        self.rehash_query(self.binding["hardware"])

    def write(self, path, value):
        path.write_bytes(core.encoded(value))

    def rehash_query(self, value, configuration=False):
        value.update(byte_limit=32768, stdout_scope="filtered_controller_nonsecret_settings" if configuration else "complete_raw_utf8")
        for name in ("stdout", "stderr"):
            raw = value[name].encode(); value[name + "_full_sha256"] = core.digest(raw); value[name + "_captured_bytes"] = len(raw)
        if configuration:
            value.update(filtered_stdout_sha256=core.digest(value["stdout"].encode()), filtered_stdout_bytes=len(value["stdout"].encode()))

    def test_current_synthetic_binding_does_not_equate_logical_index_and_minor(self):
        self.assertEqual(core.validate_binding(self.plan, self.metadata, self.binding), self.binding)
        self.assertEqual(self.metadata["environment"]["SLURM_JOB_GPUS"], "6")
        self.assertEqual(self.binding["physical_minor"], 3)
        self.assertIsNone(self.plan["gpu_uuid"])

    def test_profile_and_plan_require_exact_opt_in_and_unresolved_identity(self):
        for change in (lambda p: p.pop("configured_isolation"), lambda p: p.update(gpu_uuid=self.binding["gpu_uuid"]),
                lambda p: p["configured_isolation"]["profile"].update(schema_version=True),
                lambda p: p["configured_isolation"]["physical_identity"].update(physical_minor=0),
                lambda p: p["configured_isolation"]["resources"].update(cpus=True), lambda p: p.update(unit_id="../other")):
            value = copy.deepcopy(self.plan); change(value)
            with self.subTest(change=change), self.assertRaises(ValueError): core.validate_plan(value)

    def test_unsupported_modes_and_protocols_refuse(self):
        for mode in ("packing-point", "learning-check", "final", "unknown"):
            with self.subTest(mode=mode), self.assertRaises(ValueError): core.validate_plan({**self.plan, "mode": mode})
        for key in ("device_memory_limit_protocol", "final_confirmation_protocol", "completed_study", "scaling_protocol"):
            with self.subTest(key=key), self.assertRaises(ValueError): core.validate_plan({**self.plan, key: {}})

    def test_committed_and_staged_dependency_bytes_are_checked(self):
        core.source_identity(self.plan)
        value = copy.deepcopy(self.plan); value["helpers"]["files"].pop(core.PREFIX + core.HELPERS[0])
        with self.assertRaises(ValueError): core.source_identity(value)

    def test_owned_launch_chain_and_canonical_row(self):
        self.assertEqual(core.owned_inputs(self.plan), self.inputs)
        self.assertEqual(self.row[1], self.plan["kind"])
        source = Path(self.inputs["submissions"]["source_path"])
        source.write_text(source.read_text() + self.job + "\tother\tother\tother\n")
        with self.assertRaises(ValueError): core.owned_inputs(self.plan)

    def test_missing_foreign_or_duplicate_ledger_refuses(self):
        source = Path(self.inputs["ledger"]["source_path"])
        for raw in ("999999\n", self.job + "\n" + self.job + "\n"):
            source.write_text(raw)
            with self.subTest(raw=raw), self.assertRaises(ValueError): core.owned_inputs(self.plan)

    def test_receipt_boolean_and_changed_release_refuse(self):
        path = Path(self.inputs["release"]["path"]); original = json.loads(path.read_text())
        for changes in ({"returncode": False}, {"command": ["scontrol", "release", "999999"]}, {"job_id": "999999"}):
            self.write(path, {**original, **changes}); value = copy.deepcopy(self.inputs); value["release"] = core.receipt(path)
            self.write(Path(self.plan["configured_isolation"]["owned_inputs_path"]), value)
            with self.subTest(changes=changes), self.assertRaises(ValueError): core.owned_inputs(self.plan)

    def test_config_owner_hash_alias_chain_or_cache_refuses(self):
        first = next(iter(self.configuration["files"]))
        changes = [lambda v: v["service"].update(uid=1), lambda v: v.update(cache_absent=False),
            lambda v: v["files"][first].update(uid=0), lambda v: v["files"][first].update(mode=0o666),
            lambda v: v["files"][first].update(symlink=True), lambda v: v["files"][first].update(sha256="changed"),
            lambda v: v["files"][first]["directory_chain"][0].update(uid=1), lambda v: v["files"].pop(first)]
        for change in changes:
            value = copy.deepcopy(self.configuration); change(value)
            with self.subTest(change=change), self.assertRaises(ValueError): core.validate_configuration(value)

    def test_own_process_uid_birth_membership_and_logical_selector_refuse(self):
        changes = [lambda v: v["process"].update(uid=1), lambda v: v["process"].update(process_start_time="1"),
            lambda v: v["environment"].update(SLURM_STEP_GPUS="5"), lambda v: v["environment"].update(SLURM_JOB_GPUS="6,7"),
            lambda v: v["environment"].update(SLURM_STEP_ID="batch"), lambda v: v["process"].update(hostname="other"),
            lambda v: v["process"].update(cpu_affinity=[False, 1, 2, 3, 4, 5, 6, 7]),
            lambda v: v["cgroup"]["memberships"][-1].update(raw="", sha256=core.digest(b""))]
        for change in changes:
            value = copy.deepcopy(self.metadata); change(value)
            with self.subTest(change=change), self.assertRaises(ValueError): core.validate_metadata(self.plan, value)

    def test_single_card_shape_and_plugin_duplicates_refuse(self):
        for key, old, new in (("job", "gres/gpu=1", "gres/gpu=2"), ("job", "NumTasks=1", "NumTasks=2"),
                ("job", "mem=64G", "mem=32G"), ("step", "mem=64G", "mem=128G"), ("step", "Tasks=1", "Tasks=2"),
                ("running_configuration", "task/affinity,task/cgroup", "task/cgroup,task/cgroup")):
            value = copy.deepcopy(self.metadata); value[key]["stdout"] = value[key]["stdout"].replace(old, new)
            self.rehash_query(value[key], configuration=key == "running_configuration")
            with self.subTest(key=key, old=old), self.assertRaises(ValueError): core.validate_metadata(self.plan, value)

    def test_changed_binding_process_selector_or_native_identity_refuses(self):
        changes = [lambda v: v["environment_after"].update(CUDA_VISIBLE_DEVICES=v["gpu_uuid"]),
            lambda v: v["process_after"].update(pid=1), lambda v: v["native"].update(pid=1),
            lambda v: v["native"].update(cuda_device_count=2), lambda v: v["native"].update(jax_local_hardware_id=1),
            lambda v: v["native"].update(current_context=None), lambda v: v["native"].update(current_context_device=1),
            lambda v: v["device"].update(minor=3.0), lambda v: v["device"].update(character_device=1),
            lambda v: v.update(physical_gres_mapping_verified=True), lambda v: v.update(startup_memory_observed=True),
            lambda v: v["native"].update(pci_address="0000:66:00.0")]
        for change in changes:
            value = copy.deepcopy(self.binding); change(value)
            with self.subTest(change=change), self.assertRaises(ValueError): core.validate_binding(self.plan, self.metadata, value)

    def test_full_xml_compute_process_query_and_physical_relabelling_refuse(self):
        for command in (["nvidia-smi", "-q", "-x"], ["nvidia-smi", "--query-compute-apps=pid,used_memory"]):
            value = copy.deepcopy(self.binding); value["hardware"]["command"] = command
            with self.subTest(command=command), self.assertRaises(ValueError): core.validate_binding(self.plan, self.metadata, value)
        value = copy.deepcopy(self.binding); value["proc"]["raw"] += "Device Minor: 4\n"; value["proc"]["sha256"] = core.digest(value["proc"]["raw"].encode())
        with self.assertRaises(ValueError): core.validate_binding(self.plan, self.metadata, value)

    def test_driver_copy_is_portable_and_symlink_or_changed_bytes_refuse(self):
        target = Path(self.binding["driver_capture"]["captured"]["path"])
        self.assertFalse(Path(self.binding["native"]["driver_library"]).exists())
        core.validate_binding(self.plan, self.metadata, self.binding)
        target.write_bytes(b"changed")
        with self.assertRaises(ValueError): core.validate_binding(self.plan, self.metadata, self.binding)

    def test_publication_is_immutable_and_plan_bytes_stay_identical(self):
        before = self.plan_path.read_bytes(); path = self.unit / "resolved_device_binding.json"
        core.publish(path, self.binding)
        with self.assertRaises(FileExistsError): core.publish(path, self.binding)
        self.assertEqual(self.plan_path.read_bytes(), before)

    def test_partial_metadata_capture_is_saved_before_validation_failure(self):
        with patch.object(core, "source_identity", side_effect=ValueError("synthetic source failure")):
            with self.assertRaisesRegex(ValueError, "synthetic source failure"): core.collect_metadata(self.plan, self.unit)
        result = json.loads((self.unit / "configured_isolation_metadata.json").read_text())
        self.assertEqual(result["capture_error"]["type"], "ValueError")
        self.assertFalse(result["physical_gres_mapping_verified"])

    def test_live_ledger_is_not_an_immutable_snapshot_and_later_growth_is_allowed(self):
        value = copy.deepcopy(self.inputs)
        value["ledger"]["snapshot"] = core.receipt(value["ledger"]["source_path"])
        self.write(Path(self.plan["configured_isolation"]["owned_inputs_path"]), value)
        with self.assertRaises(ValueError): core.owned_inputs(self.plan)
        self.write(Path(self.plan["configured_isolation"]["owned_inputs_path"]), self.inputs)
        source = Path(self.inputs["ledger"]["source_path"])
        source.write_text(source.read_text() + "999999\n")
        self.assertEqual(core.owned_inputs(self.plan), self.inputs)

    def test_binding_hashes_exact_original_plan_bytes(self):
        raw = json.dumps(self.plan, indent=4).encode() + b"\n"
        self.plan_path.write_bytes(raw)
        inputs = copy.deepcopy(self.inputs); inputs["plan"] = core.receipt(self.plan_path)
        for key in ("submission", "release"):
            path = Path(inputs[key]["path"]); document = json.loads(path.read_text())
            document["plan_sha256"] = inputs["plan"]["sha256"]
            if key == "release": document["submission_receipt_sha256"] = inputs["submission"]["sha256"]
            self.write(path, document); inputs[key] = core.receipt(path)
        self.write(Path(self.plan["configured_isolation"]["owned_inputs_path"]), inputs)
        metadata = copy.deepcopy(self.metadata); metadata.update(inputs=inputs,
            inputs_receipt=core.receipt(self.plan["configured_isolation"]["owned_inputs_path"]))
        value = copy.deepcopy(self.binding); value.update(plan_sha256=inputs["plan"]["sha256"], metadata_sha256=core.digest(core.encoded(metadata)))
        self.assertNotEqual(value["plan_sha256"], core.digest(core.encoded(self.plan)))
        core.validate_binding(self.plan, metadata, value)
        value["plan_sha256"] = core.digest(core.encoded(self.plan))
        with self.assertRaises(ValueError): core.validate_binding(self.plan, metadata, value)

    def test_configuration_private_change_is_never_retained(self):
        path = self.run / "private.conf"; private = b'AuthInfo="private value File=never disclose"\nTaskPlugin=task/cgroup\n'
        path.write_bytes(private); row = {}
        with patch.object(core, "CONFIG_FILES", {str(path): "0" * 64}), self.assertRaises(ValueError): core.config_file(path, row)
        output = core.encoded(row)
        self.assertNotIn(b"private value", output); self.assertNotIn(b"never disclose", output)
        self.assertNotIn("raw", row); self.assertEqual(row["sha256"], core.digest(private))
        self.assertEqual(row["settings"], [])

    def test_fifo_and_symlinked_parent_are_not_opened_for_reading(self):
        fifo = self.run / "fifo.conf"; os.mkfifo(fifo)
        seen, real_open = [], os.open
        def observe(path, flags, *args, **kwargs):
            seen.append((str(path), flags)); return real_open(path, flags, *args, **kwargs)
        with patch.object(core, "CONFIG_FILES", {str(fifo): "0" * 64}), patch.object(os, "open", side_effect=observe):
            with self.assertRaises(ValueError): core.config_file(fifo)
        self.assertFalse(any(path.startswith("/proc/self/fd/") for path, flags in seen))
        folder = self.run / "ordinary"; folder.mkdir(); (folder / "file.conf").write_text("synthetic")
        alias = self.run / "alias"; alias.symlink_to(folder, target_is_directory=True)
        with patch.object(core, "CONFIG_FILES", {str(alias / "file.conf"): "0" * 64}), self.assertRaises(OSError):
            core.config_file(alias / "file.conf")

    def test_foreign_uid_or_bad_four_uid_proof_cannot_query_scheduler(self):
        helper = core.adjacent("device_memory_limit"); helper.process_identity = lambda: {**self.process, "uid": 1}
        with patch.object(core, "source_identity", return_value=self.plan["helpers"]), patch.object(core, "owned_inputs", return_value=self.inputs),\
                patch.object(core, "adjacent", return_value=helper), patch.dict(os.environ, self.metadata["environment"], clear=True),\
                patch.object(core, "bounded_query") as query:
            with self.assertRaisesRegex(ValueError, "UID"): core.collect_metadata(self.plan, self.unit)
            query.assert_not_called()
        (self.unit / "configured_isolation_metadata.json").unlink()
        helper.process_identity = lambda: self.process
        helper.capture_v2_ownership = lambda *args: (_ for _ in ()).throw(ValueError("four UIDs rejected"))
        with patch.object(core, "source_identity", return_value=self.plan["helpers"]), patch.object(core, "owned_inputs", return_value=self.inputs),\
                patch.object(core, "adjacent", return_value=helper), patch.dict(os.environ, self.metadata["environment"], clear=True),\
                patch.object(core, "bounded_query") as query:
            with self.assertRaisesRegex(ValueError, "four UIDs"): core.collect_metadata(self.plan, self.unit)
            query.assert_not_called()

    def fake_query(self, stdout, stderr=b""):
        class Process:
            def __init__(self):
                self.stdout, self.stderr = self.pipe(stdout), self.pipe(stderr)
            def pipe(self, raw):
                read, write = os.pipe(); os.write(write, raw); os.close(write)
                return os.fdopen(read, "rb", buffering=0)
            def poll(self): return 0
            def wait(self, timeout=None): return 0
            def kill(self): raise AssertionError("completed synthetic pipe must not be killed")
        return Process()

    def test_bounded_queries_limit_output_and_redact_controller_private_values(self):
        with patch.object(subprocess, "Popen", return_value=self.fake_query(b"x" * 65)):
            record = core.bounded_query(["scontrol", "show", "job", "-d", self.job], byte_limit=64)
        self.assertEqual(record["capture_status"], "response_limit_exceeded")
        self.assertEqual(record["stdout_captured_bytes"], 65)
        private = b'AuthInfo="hidden File=private"\nTaskPlugin = task/cgroup,task/affinity\n'
        with patch.object(subprocess, "Popen", return_value=self.fake_query(private)):
            record = core.bounded_query(["scontrol", "show", "config"], configuration=True)
        self.assertNotIn("hidden", json.dumps(record)); self.assertNotIn("private", record["stdout"])
        self.assertIn("TaskPlugin", record["stdout"])

    def test_failed_native_binding_attempt_retains_partial_evidence_and_never_queries_a_guessed_gpu(self):
        device = object(); helper = SimpleNamespace(process_identity=lambda: self.process)
        def fail(device, value):
            value.update(pid=self.process["pid"], cuda_device_count=2)
            raise ValueError("synthetic native failure")
        with patch.object(core, "validate_metadata"), patch.object(core, "adjacent", return_value=helper),\
                patch.dict(os.environ, self.metadata["environment"], clear=True), patch.dict(sys.modules, {"jax": SimpleNamespace(local_devices=lambda **kwargs: [device])}),\
                patch.object(core, "native_identity", side_effect=fail), patch.object(core, "bounded_query") as query:
            with self.assertRaisesRegex(ValueError, "synthetic native failure"): core.bind_process(self.plan, self.metadata, device, self.unit)
            query.assert_not_called()
        attempt = json.loads((self.unit / "resolved_device_binding_attempt.json").read_text())
        self.assertFalse(attempt["passed"]); self.assertEqual(attempt["native"]["cuda_device_count"], 2)
        self.assertNotIn("gpu_uuid", attempt["native"])
        self.assertFalse((self.unit / "resolved_device_binding.json").exists())
        with self.assertRaises(ValueError): core.bind_process(self.plan, self.metadata, device, self.unit)

    def test_published_metadata_roundtrip_preserves_the_real_sorted_map(self):
        reloaded = core.adjacent("device_memory_limit").json_document(core.encoded(self.metadata))
        self.assertNotEqual(list(reloaded["configuration"]["files"]), list(self.configuration["files"]))
        self.assertEqual(core.validate_metadata(self.plan, reloaded), self.metadata)
        binding = json.loads(core.encoded(self.binding))
        core.validate_binding(self.plan, reloaded, binding)

    def test_allowed_excerpts_cannot_change_under_the_same_full_file_hash(self):
        value = copy.deepcopy(self.configuration)
        value["files"]["/etc/slurm/cgroup.conf"]["settings"] = [{"line": 12, "name": "ConstrainDevices", "value": "no"}]
        with self.assertRaisesRegex(ValueError, "excerpt list"): core.validate_configuration(value)

    def test_raw_query_hash_count_and_encoding_are_checked(self):
        command = self.metadata["job"]["command"]
        for key, replacement in (("stdout_full_sha256", "0" * 64), ("stdout_captured_bytes", 1),
                ("stderr_full_sha256", "0" * 64), ("byte_limit", True)):
            value = copy.deepcopy(self.metadata["job"]); value[key] = replacement
            with self.subTest(key=key), self.assertRaises(ValueError): core.validate_query(value, command)
        value = copy.deepcopy(self.metadata["running_configuration"]); value["filtered_stdout_sha256"] = "0" * 64
        with self.assertRaises(ValueError): core.validate_query(value, value["command"], configuration=True)
        with patch.object(subprocess, "Popen", return_value=self.fake_query(b"invalid\\xff".replace(b"\\xff", bytes([255])))):
            value = core.bounded_query(command)
        self.assertEqual(value["capture_status"], "invalid_utf8"); self.assertEqual(value["stdout"], "")
        with self.assertRaises(ValueError): core.validate_query(value, command)


if __name__ == "__main__":
    unittest.main()
