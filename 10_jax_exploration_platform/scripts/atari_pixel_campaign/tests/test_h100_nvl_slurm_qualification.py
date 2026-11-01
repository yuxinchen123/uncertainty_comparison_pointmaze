"""Test the opt-in configured-isolation profile with raw metadata and no numerical imports."""

import copy
import hashlib
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import h100_nvl_allocation_observation as observer
import h100_nvl_slurm_qualification as qualification
import h100_nvl_slurm_site_validation as site


def fixture():
    """Represent scheduler GPU one and physical minor zero without asserting they are the same index."""
    gpu, pci = "GPU-f3f58504-1234-5678-abcd-123456789abc", "0000:c1:00.0"
    path = "/slurm/uid_42/job_333/step_0/task_0"
    cgroup = "0::" + path + "\n"
    mount = observer.own_cgroup_mounts("33 22 0:28 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n")
    levels = [{"level": name, "cgroup_path": value,
        "directory": {"path": "/sys/fs/cgroup" + value, "status": "readable", "inode": index + 10}}
        for index, (name, value) in enumerate((("job", "/slurm/uid_42/job_333"), ("step", "/slurm/uid_42/job_333/step_0"), ("task", path)))]
    levels[-1]["effective_cpuset"] = {"status": "readable", "text": "0-7\n"}
    config = observer.selected_config("CgroupPlugin=autodetect\nConstrainDevices=yes\nConstrainCores=yes\n", qualification.CONFIG_KEYS)
    node = {**config, "status": "readable", "file": {"path": "/etc/slurm/cgroup.conf", "status": "readable", "owner_uid": 0, "mode": "0o644"}}
    plugins = {**observer.selected_config("TaskPlugin=task/cgroup,task/affinity\nProctrackType=proctrack/cgroup\n", qualification.PLUGIN_KEYS),
        "status": "readable", "command": ["scontrol", "show", "config"], "returncode": 0}
    placement = {"pid": 123, "uid": 42, "process_start_time": "456", "cpu_affinity": list(range(8)),
        "own_cgroup_file": {"text": cgroup, "sha256": hashlib.sha256(cgroup.encode()).hexdigest()}, "cgroup_mountinfo": mount,
        "cgroups": [{"hierarchy": "0", "controllers": [], "cgroup_path": path, "task_path_observed": True,
            "mounts": [{"filesystem": "cgroup2", "mountpoint": "/sys/fs/cgroup", "levels": levels}]}]}
    job = ("JobId=333 UserId=sl5nw(42) JobState=RUNNING NodeList=serval08 NumNodes=1 NumCPUs=8 "
        "JOB_GRES=gpu:nvidia_h100_nvl:1 TresPerNode=gres/gpu:nvidia_h100_nvl:1 AllocTRES=cpu=8,gres/gpu=1,gres/gpu:nvidia_h100_nvl=1\n"
        "   Nodes=serval08 CPU_IDs=0-7 Mem=65536 GRES=gpu:nvidia_h100_nvl:1(IDX:1)\n")
    step = "StepId=333.0 UserId=42 State=RUNNING NodeList=serval08 Nodes=1 CPUs=8 Tasks=1 TRES=cpu=8,gres/gpu=1\n"
    return {"schema_version": 1, "kind": "h100_nvl_slurm_configured_isolation_observation", "qualification_profile": qualification.PROFILE,
        "hostname": "serval08", "pid": 123, "process_start_time": "456",
        "allocation_environment": {"SLURM_JOB_ID": "333", "SLURM_STEP_ID": "0", "SLURM_JOB_GPUS": "1", "SLURM_STEP_GPUS": "1",
            "SLURM_GPUS_ON_NODE": "1", "SLURM_CPUS_PER_TASK": "8", "SLURM_LOCALID": "0", "CUDA_VISIBLE_DEVICES": "0"},
        "scheduler_job": {"command": ["scontrol", "show", "job", "-d", "333"], "returncode": 0, "stdout": job},
        "scheduler_step": {"command": ["scontrol", "show", "step", "-d", "333.0"], "returncode": 0, "stdout": step},
        "scheduler_node": {"command": ["scontrol", "show", "node", "-d", "serval08"], "returncode": 0, "stdout": "NodeName=serval08 Version=23.11.6\n"},
        "gpu_csv": f"0, {gpu}, NVIDIA H100 NVL, 95830, 580.159.04\n",
        "gpu_xml": f"<nvidia_smi_log><gpu><uuid>{gpu}</uuid><product_name>NVIDIA H100 NVL</product_name><minor_number>0</minor_number><pci><pci_bus_id>00000000:C1:00.0</pci_bus_id></pci></gpu></nvidia_smi_log>",
        "host_gpu_inventory": {"proc_gpu_inventory": {"status": "present", "cards": [{"status": "readable", "pci_directory": pci,
            "fields": {"Model": "NVIDIA H100 NVL", "GPU UUID": gpu, "Bus Location": pci, "Device Minor": "0"}}]},
            "pci_gpu_inventory": {"status": "present", "cards": [{"pci_address": pci, "fields": {
                "vendor": {"status": "readable", "text": "0x10de"}, "class": {"status": "readable", "text": "0x030200"}}}]},
            "device_files": [{"path": "/dev/nvidia0", "character_device": True, "major": 195, "minor": 0},
                {"path": "/dev/nvidia1", "character_device": True, "major": 195, "minor": 1}]},
        "device_isolation_observations": {"node_cgroup_configuration": node, "running_slurm_configuration": plugins, "own_placement": placement}}


class QualificationTests(unittest.TestCase):
    """Exercise independent scheduler and physical identities, strict missing evidence and unchanged defaults."""

    def test_scheduler_one_and_physical_zero_remain_distinct(self):
        """The explicit profile accepts consistent scheduler IDs without asserting minor or PCI rank equality."""
        result = qualification.validate_allocation(fixture())
        self.assertEqual(result["scheduler_gpu_id"], "1")
        self.assertEqual(result["physical_minor"], "0")
        self.assertEqual(result["pci_address"], "0000:c1:00.0")
        self.assertIs(result["numeric_gpu_mapping_reconstructed"], False)
        self.assertIs(result["isolation"]["device_bpf_rules_independently_inspected"], False)
        self.assertEqual(result["isolation"]["basis"], qualification.PROFILE["isolation_basis"])

    def test_conflicting_or_missing_scheduler_fields_reject(self):
        """A selected GPU cannot substitute for its exact owned typed job and current numerical step."""
        for key, value in (("SLURM_STEP_GPUS", "0"), ("SLURM_JOB_GPUS", None), ("SLURM_STEP_ID", "batch"),
                           ("SLURM_GPUS_ON_NODE", "2"), ("SLURM_LOCALID", "1")):
            raw = fixture()
            raw["allocation_environment"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                qualification.validate_allocation(raw)
        for kind, before, after in (("job", "IDX:1", "IDX:0"), ("step", "333.0", "334.0"),
                                    ("step", "gres/gpu=1", "gres/gpu=2"), ("node", "23.11.6", "24.11.0")):
            raw = fixture()
            raw["scheduler_" + kind]["stdout"] = raw["scheduler_" + kind]["stdout"].replace(before, after)
            with self.subTest(kind=kind, before=before), self.assertRaises(ValueError):
                qualification.validate_allocation(raw)

    def test_missing_or_disabled_configured_isolation_rejects(self):
        """An idle visible GPU does not establish the required configured scheduler device restrictions."""
        for field, value in (("ConstrainDevices", "no"), ("CgroupPlugin", "cgroup/v1"), ("IgnoreSystemd", "yes")):
            raw = fixture()
            text = "CgroupPlugin=autodetect\nConstrainDevices=yes\n" + field + "=" + value + "\n"
            if field != "IgnoreSystemd":
                text = "\n".join(line for line in text.splitlines()[:-1] if not line.startswith(field + "=")) + "\n" + field + "=" + value + "\n"
            node = raw["device_isolation_observations"]["node_cgroup_configuration"]
            node.update(observer.selected_config(text, qualification.CONFIG_KEYS))
            with self.subTest(field=field), self.assertRaises(ValueError):
                qualification.validate_allocation(raw)
        raw = fixture()
        raw["device_isolation_observations"]["node_cgroup_configuration"]["status"] = "permission denied"
        with self.assertRaisesRegex(ValueError, "unavailable"):
            qualification.validate_allocation(raw)

    def test_wrong_own_process_task_cpuset_or_config_bytes_reject(self):
        """Recorded membership must identify the same actual task, process and eight allowed CPUs."""
        for mutation in (lambda raw: raw["device_isolation_observations"]["own_placement"].update(pid=124),
            lambda raw: raw["device_isolation_observations"]["own_placement"].update(cpu_affinity=list(range(9))),
            lambda raw: raw["device_isolation_observations"]["own_placement"]["cgroups"][0].update(task_path_observed=False),
            lambda raw: raw["device_isolation_observations"]["own_placement"]["cgroups"][0]["mounts"][0]["levels"][-1]["effective_cpuset"].update(text="0-6"),
            lambda raw: raw["device_isolation_observations"]["node_cgroup_configuration"].update(filtered_bytes="changed"),
            lambda raw: raw["device_isolation_observations"]["node_cgroup_configuration"]["file"].update(mode="0o666")):
            raw = fixture()
            mutation(raw)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                qualification.validate_allocation(raw)

    def test_physical_uuid_pci_proc_sysfs_and_driver_must_agree(self):
        """PCI formatting normalization is allowed, but another physical card or driver cannot pass."""
        for mutation in (lambda raw: raw.update(gpu_csv=raw["gpu_csv"].replace("580.159.04", "590.48.01")),
            lambda raw: raw["host_gpu_inventory"]["proc_gpu_inventory"]["cards"][0]["fields"].update({"Device Minor": "1"}),
            lambda raw: raw["host_gpu_inventory"]["proc_gpu_inventory"]["cards"][0]["fields"].update({"Bus Location": "0000:61:00.0"}),
            lambda raw: raw["host_gpu_inventory"]["pci_gpu_inventory"]["cards"][0]["fields"]["vendor"].update(text="0x1002"),
            lambda raw: raw["host_gpu_inventory"]["device_files"][0].update(character_device=False),
            lambda raw: raw.update(gpu_csv=raw["gpu_csv"] + raw["gpu_csv"])):
            raw = fixture()
            mutation(raw)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                qualification.validate_allocation(raw)

    def test_cuda_uuid_comes_from_same_process_and_selected_jax_device(self):
        """A CVD declaration alone cannot replace actual JAX ordinal CUDA UUID and PCI proof."""
        allocation = qualification.validate_allocation(fixture())
        record = {"schema_version": 1, "kind": "same_process_jax_cuda_device_identity", "pid": 123,
            "jax_local_hardware_id": 0, "cuda_device_count": 1, "gpu_uuid": allocation["gpu_uuid"],
            "cuda_device_handle": 0, "api_return_codes": {name: 0 for name in ("cuDeviceGetCount", "cuDeviceGet", "cuDeviceGetUuid_v2", "cuDeviceGetPCIBusId")},
            "pci_address": allocation["pci_address"], "driver_library": "/lib/libcuda.so.580.159.04", "driver_library_sha256": "a" * 64}
        self.assertEqual(qualification.validate_cuda_identity(record, allocation, 123), record)
        for key, value in (("pid", 124), ("cuda_device_count", 2), ("gpu_uuid", "GPU-ffffffff-1234-5678-abcd-123456789abc"),
                           ("pci_address", "0000:61:00.0"), ("jax_local_hardware_id", 1), ("driver_library", "/lib/libcuda.so.590.48.01")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                qualification.validate_cuda_identity({**record, key: value}, allocation, 123)

    def test_version1_device_rules_reject_other_gpu_access(self):
        """Version1 can additionally check explicit device rules without querying another GPU."""
        raw = fixture()
        physical = qualification.physical_identity(raw)
        qualification.validate_device_rules("c 195:0 rwm\nc 195:255 rwm\n", raw["host_gpu_inventory"]["device_files"], physical)
        qualification.validate_device_rules("c 195:0 r\nc 195:0 w\n", raw["host_gpu_inventory"]["device_files"], physical)
        for text in ("a *:* rwm\n", "c 195:* rwm\n", "c 195:1 rwm\n", "c 195:0 rw\nc 195:1 r\n", "c 195:0 rw\nc 195:1 w\n"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                qualification.validate_device_rules(text, raw["host_gpu_inventory"]["device_files"], physical)

    def test_version1_rule_audit_requires_complete_gpu_device_inventory(self):
        """A missing physical device cannot disappear from a claim to have checked every host GPU rule."""
        inventory = fixture()["host_gpu_inventory"]
        inventory["device_files"] = inventory["device_files"][:1]
        self.assertEqual(qualification.complete_host_devices(inventory), inventory["device_files"])
        extra = copy.deepcopy(inventory["proc_gpu_inventory"]["cards"][0])
        extra.update(pci_directory="0000:61:00.0")
        extra["fields"].update({"GPU UUID": "GPU-ffffffff-1234-5678-abcd-123456789abc", "Device Minor": "1", "Bus Location": "0000:61:00.0"})
        inventory["proc_gpu_inventory"]["cards"].append(extra)
        with self.assertRaisesRegex(ValueError, "omits or substitutes"):
            qualification.complete_host_devices(inventory)

    def test_default_reference_and_closure_are_unchanged_and_opt_in_is_explicit(self):
        """New target evidence does not replace the original driver 590 reference or silently enable new helpers."""
        original = {"schema_version": 2, "reference": {"driver": "590.48.01"}}
        self.assertIsNone(site.qualification_profile(original))
        self.assertEqual(site.expected_driver(original), "590.48.01")
        self.assertEqual(site.required_helpers(None), site.HELPERS)
        selected = {**original, "schema_version": 3, "qualification_profile": qualification.PROFILE}
        self.assertEqual(site.expected_driver(selected), "580.159.04")
        self.assertEqual(selected["reference"]["driver"], "590.48.01")
        self.assertEqual(site.required_helpers(qualification.PROFILE), site.HELPERS + site.QUALIFICATION_HELPERS)
        for plan in ({**original, "qualification_profile": qualification.PROFILE},
                     {**selected, "qualification_profile": {**qualification.PROFILE, "expected_driver": "590.48.01"}}):
            with self.assertRaises(ValueError):
                site.qualification_profile(plan)
        with self.assertRaisesRegex(ValueError, "explicit configured-isolation profile"):
            site.qualification_profile({**selected, "qualification_profile": {**qualification.PROFILE, "workers_permitted": 0}})

    def test_capture_uses_only_exact_owned_scheduler_targets_and_allowlisted_environment(self):
        """Fresh evidence excludes arbitrary environment names and queries only this job, step and node."""
        raw = fixture()
        with patch.object(qualification.os, "getpid", return_value=123), patch.object(Path, "read_text", return_value="own cgroup"), \
                patch.object(observer, "host_inventory", return_value=raw["host_gpu_inventory"]), \
                patch.object(observer, "device_isolation_observations", return_value=raw["device_isolation_observations"]):
            queries = []
            def query(command):
                """Return the matching explicit raw scheduler fixture and retain the actual query."""
                queries.append(command)
                return raw["scheduler_" + command[2]]["stdout"]
            captured = qualification.capture(observer, {**raw["allocation_environment"], "PRIVATE_TOKEN": "do not record"},
                raw["hostname"], raw["gpu_csv"], raw["gpu_xml"], query, "456")
        self.assertEqual(queries, [raw["scheduler_" + name]["command"] for name in ("job", "step", "node")])
        self.assertNotIn("PRIVATE_TOKEN", json.dumps(captured))
        self.assertNotIn("do not record", json.dumps(captured))
        self.assertEqual(qualification.validate_allocation(captured)["scheduler_gpu_id"], "1")

    def test_cuda_query_uses_jax_ordinal_and_already_loaded_driver_only(self):
        """Mock the C calls to test exact UUID/PCI extraction, recorded statuses and no new driver load."""
        import ctypes
        import io
        import uuid
        from unittest.mock import Mock
        gpu = fixture()["gpu_csv"].split(",")[1].strip()
        driver = types.SimpleNamespace()
        def count(output):
            """Return the single visible CUDA device count."""
            ctypes.cast(output, ctypes.POINTER(ctypes.c_int))[0] = 1
            return 0
        def device(output, ordinal):
            """Preserve the requested JAX local hardware ordinal in the mocked CUDA query."""
            self.assertEqual(ordinal, 0)
            ctypes.cast(output, ctypes.POINTER(ctypes.c_int))[0] = 0
            return 0
        def identifier(output, ordinal):
            """Return the selected physical UUID bytes from the device query."""
            ctypes.memmove(output, uuid.UUID(gpu[4:]).bytes, 16)
            return 0
        def pci(output, length, ordinal):
            """Return the selected physical PCI address from that same device query."""
            output.value = b"0000:c1:00.0"
            return 0
        for name, function in (("cuDeviceGetCount", count), ("cuDeviceGet", device),
                ("cuDeviceGetUuid_v2", identifier), ("cuDeviceGetPCIBusId", pci)):
            setattr(driver, name, Mock(side_effect=function))
        maps = "1-2 r-xp 0 0:0 0 /lib/libcuda.so.580.159.04\n"
        with patch.object(Path, "read_text", return_value=maps), patch.object(Path, "open", return_value=io.BytesIO(b"actual driver bytes")), \
                patch.object(qualification.ctypes, "CDLL", return_value=driver) as loaded:
            record = qualification.cuda_device_identity(types.SimpleNamespace(platform="gpu", device_kind="NVIDIA H100 NVL", local_hardware_id=0))
        loaded.assert_called_once_with("/lib/libcuda.so.580.159.04", mode=qualification.os.RTLD_LOCAL | qualification.os.RTLD_NOLOAD)
        self.assertEqual(record["gpu_uuid"], gpu)
        self.assertEqual(record["driver_library_sha256"], hashlib.sha256(b"actual driver bytes").hexdigest())
        self.assertEqual(record["api_return_codes"], {name: 0 for name in vars(driver)})

    def test_unknown_or_missing_profile_observation_cannot_pass(self):
        """An optional plan flag or old allocation-only packet is not sufficient evidence."""
        for value in ({}, {"schema_version": 1, "kind": "h100_nvl_slurm_configured_isolation_observation"},
                {**fixture(), "qualification_profile": {**qualification.PROFILE, "numeric_gpu_mapping_reconstructed": True}}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                qualification.validate_allocation(value)

    def completed_profile(self):
        """Extend a complete original metadata fixture with actual-style explicit target-profile evidence."""
        import test_h100_nvl_slurm_site_validation as original_tests
        helper = original_tests.H100SiteTests(methodName="test_complete_site_receipt_verifies_original_children_without_admitting_workers")
        helper.setUp()
        self.addCleanup(helper.doCleanups)
        directory, plan, result, probe = helper.completed_site()
        raw = fixture()
        raw.update(pid=90, process_start_time="99")
        raw["device_isolation_observations"]["own_placement"].update(pid=90, process_start_time="99")
        proof = qualification.validate_allocation(raw)
        plan.update(schema_version=3, qualification_profile=qualification.PROFILE)
        helper.write(directory / "site_validation_plan.json", plan)
        result.update(plan=site.receipt(directory / "site_validation_plan.json"), qualification_profile=qualification.PROFILE)
        helper.write(directory / "qualification_observation.json", raw)
        assignment = site.read(directory / "site_assignment.json")
        environment = {**raw["allocation_environment"], "SLURM_CPU_BIND_TYPE": "threads"}
        assignment.update(allocation=proof, cpu_allocation=site.load("infrastructure_retry").cpu_allocation(environment, [proof["gpu_uuid"]], range(8)),
            qualification_observation=site.receipt(directory / "qualification_observation.json"))
        assignment["gpu_modes"]["gpu_uuid"] = proof["gpu_uuid"]
        helper.write(directory / "site_assignment.json", assignment)
        result["assignment"] = site.receipt(directory / "site_assignment.json")
        for kind in ("learner", "renderer", "probe"):
            path = directory / (kind + "_process.json")
            process = site.read(path)
            child_raw = copy.deepcopy(raw)
            child_raw.update(pid=process["pid"], process_start_time=process["process_start_time"])
            child_raw["device_isolation_observations"]["own_placement"].update(pid=process["pid"], process_start_time=process["process_start_time"])
            child_raw["allocation_environment"]["CUDA_VISIBLE_DEVICES"] = proof["gpu_uuid"]
            process.update(gpu_uuid=proof["gpu_uuid"], qualification_observation=child_raw)
            helper.write(path, process)
            result["children"][kind]["process"] = site.receipt(path)
        driver = helper.run / "libcuda.so.580.159.04"
        driver.write_bytes(b"synthetic qualified mapped CUDA driver")
        probe["loaded_libraries"]["loaded_shared_libraries"][-1] = str(driver)
        probe.update(gpu_uuid=proof["gpu_uuid"], qualification_cuda_identity={"schema_version": 1,
            "kind": "same_process_jax_cuda_device_identity", "pid": probe["pid"], "jax_local_hardware_id": 0,
            "cuda_device_count": 1, "cuda_device_handle": 0, "api_return_codes": {name: 0 for name in
                ("cuDeviceGetCount", "cuDeviceGet", "cuDeviceGetUuid_v2", "cuDeviceGetPCIBusId")},
            "gpu_uuid": proof["gpu_uuid"], "pci_address": proof["pci_address"],
            "driver_library": str(driver), "driver_library_sha256": site.sha256(driver)})
        helper.write(directory / "probe_result.json", probe)
        result["probe"] = site.receipt(directory / "probe_result.json")
        return helper, directory, plan, result, probe

    def test_complete_profile_requires_each_original_gate_child_and_actual_cuda_identity(self):
        """The new profile passes only after all unchanged original gates and the complete probe proof."""
        helper, directory, plan, result, probe = self.completed_profile()
        checked = site.verify_result(directory, plan, result)
        self.assertEqual(checked["qualification_profile"], qualification.PROFILE)
        self.assertFalse(checked["workers_permitted"])
        for kind in ("learner", "renderer", "probe"):
            path = directory / (kind + "_process.json")
            process = site.read(path)
            changed = {name: value for name, value in process.items() if name != "qualification_observation"}
            helper.write(path, changed)
            result["children"][kind]["process"] = site.receipt(path)
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "explicit configured-isolation"):
                site.verify_result(directory, plan, result)
            helper.write(path, process)
            result["children"][kind]["process"] = site.receipt(path)
        probe.pop("qualification_cuda_identity")
        helper.write(directory / "probe_result.json", probe)
        result["probe"] = site.receipt(directory / "probe_result.json")
        with self.assertRaisesRegex(ValueError, "same-process JAX CUDA"):
            site.verify_result(directory, plan, result)

    def test_rehashed_driver_bytes_and_nonzero_gate_cannot_pass_profile(self):
        """A valid plan/profile cannot conceal different mapped driver bytes or an unsuccessful original gate."""
        helper, directory, plan, result, probe = self.completed_profile()
        Path(probe["qualification_cuda_identity"]["driver_library"]).write_bytes(b"changed mapped image")
        with self.assertRaisesRegex(ValueError, "driver image bytes"):
            site.verify_result(directory, plan, result)
        terminal_path = directory / "learner_terminal.json"
        terminal = site.read(terminal_path)
        terminal["exit_code"] = 1
        helper.write(terminal_path, terminal)
        result["children"]["learner"]["terminal"] = site.receipt(terminal_path)
        with self.assertRaisesRegex(ValueError, "successful children"):
            site.verify_result(directory, plan, result)

    def test_profile_background_states_reference_target_and_isolation_limit(self):
        """Operational records distinguish the old reference from the new driver and limited isolation evidence."""
        prose = site.profile_background(qualification.PROFILE)
        for phrase in ("590.48.01", "580.159.04", "not independently inspected", "not reconstructed"):
            self.assertIn(phrase, prose)


if __name__ == "__main__":
    unittest.main()
