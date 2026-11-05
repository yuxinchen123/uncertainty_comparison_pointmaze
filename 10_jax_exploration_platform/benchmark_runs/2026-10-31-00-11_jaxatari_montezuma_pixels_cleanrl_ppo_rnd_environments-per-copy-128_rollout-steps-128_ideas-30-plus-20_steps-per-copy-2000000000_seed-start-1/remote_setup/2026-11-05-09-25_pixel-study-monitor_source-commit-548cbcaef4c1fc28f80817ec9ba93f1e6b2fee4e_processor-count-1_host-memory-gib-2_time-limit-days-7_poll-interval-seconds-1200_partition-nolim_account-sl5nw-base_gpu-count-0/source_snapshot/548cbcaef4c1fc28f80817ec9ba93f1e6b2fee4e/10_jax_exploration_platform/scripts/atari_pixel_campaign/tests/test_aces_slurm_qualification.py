"""Check ACES metadata rejection paths without importing JAX or touching any GPU."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1]
specification = importlib.util.spec_from_file_location("tested_aces_qualification", SCRIPT / "aces_slurm_qualification.py")
qualification = importlib.util.module_from_spec(specification)
specification.loader.exec_module(qualification)
PACKET_NAME = "2026-10-31-23-23_ACES_allocation-observation_gpu-count-1_processor-count-8_host-memory-gib-32_time-limit-minutes-5_partition-gpu_account-158590754315_helper-commit-11ceab5c640b484d18ca40e05785bbf55c6de86f"


def actual_observation():
    """Reuse committed raw job 2274380 bytes without granting credit to that metadata-only observation."""
    campaign, = (SCRIPT.parents[1] / "benchmark_runs").glob("2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_*")
    unit = campaign / "data" / PACKET_NAME
    record = json.loads((unit / "hardware_observations.json").read_text())
    return {"qualification_profile": copy.deepcopy(qualification.PROFILE),
        "owned_allocation": json.loads((unit / "owned_allocation.json").read_text()),
        "host_gpu_inventory": record["host_gpu_inventory"], "hardware_queries": record["hardware_queries"],
        "device_isolation_observations": record["device_isolation_observations"]}


def cached_record(path, kind, contents):
    """Construct explicit root-owned test metadata without representing it as an actual captured configuration."""
    digest = hashlib.sha256(contents.encode()).hexdigest()
    return {"path": path, "kind": kind, "status": "readable", "owner_uid": 0, "mode": "0o644",
        "regular_file": True, "symlink": False, "sha256": digest,
        "filtered_bytes": contents, "filtered_sha256": digest}


def valid_fixture():
    """Add synthetic configuration proof solely for testing the validator against real raw device observations."""
    record = actual_observation()
    root = qualification.PROFILE["configuration_directory"]
    contents = {"slurm": "TaskPlugin=task/cgroup,task/affinity\nProctrackType=proctrack/cgroup\nGresTypes=gpu\n",
        "gres": "AutoDetect=nvml\nNodeName=ac041 Name=gpu Type=h100 File=/dev/nvidia[0-7]\n",
        "cgroup": "ConstrainDevices=yes\nConstrainCores=yes\n"}
    candidates = [cached_record(root + "/" + name + ".conf", name, text) for name, text in contents.items()]
    candidates.extend({"path": "/etc/slurm/" + name + ".conf", "status": "absent"} for name in contents)
    record["cached_configuration"] = {"observed_paths": {"environment": root + "/slurm.conf",
        "running_controller": "/etc/slurm/slurm.conf"}, "candidates": candidates}
    return record


def cuda_record(allocation, pid=123):
    """Build a synthetic successful same-process CUDA receipt for negative metadata tests."""
    return {"schema_version": 1, "kind": "same_process_aces_jax_cuda_identity", "pid": pid,
        "cuda_device_count": 1, "jax_local_hardware_id": 0, "cuda_device_handle": 0,
        "api_return_codes": {name: 0 for name in ("cuDeviceGetCount", "cuDeviceGet", "cuDeviceGetUuid_v2", "cuDeviceGetPCIBusId")},
        "gpu_uuid": allocation["gpu_uuid"], "pci_address": allocation["pci_address"],
        "driver_library": "/node-only/libcuda.so.580.178.04", "driver_library_sha256": "a" * 64}


class QualificationTests(unittest.TestCase):
    """Exercise explicit device-file joins and refusal of unsupported or ambiguous configuration."""

    def test_real_observation_alone_cannot_qualify(self):
        """The completed metadata-only job has no additional active GRES/cache proof."""
        with self.assertRaises(KeyError):
            qualification.validate_allocation(actual_observation())

    def test_explicit_configured_mapping_not_numeric_equality(self):
        """The configured fourth index is joined through File/proc bytes to UUID and PCI."""
        record = valid_fixture()
        result = qualification.validate_allocation(record)
        self.assertEqual(result["gpu_uuid"], "GPU-578457c3-53f3-226f-9b16-12423da85ead")
        self.assertEqual(result["physical_device_file"], "/dev/nvidia4")
        self.assertEqual(result["cuda_selector_original"], "0")
        self.assertIs(result["kernel_device_enforcement_verified"], False)
        self.assertIs(result["workers_permitted"], False)
        self.assertEqual(result["scientific_hypothesis_credit"], 0)

    def test_reordered_configured_files_change_selected_identity(self):
        """A conflicting explicit File order cannot be hidden by Slurm ID/minor numeric equality."""
        record = valid_fixture()
        gres = next(row for row in record["cached_configuration"]["candidates"] if row.get("kind") == "gres")
        text = "\n".join("Name=gpu Type=h100 File=/dev/nvidia" + str(index) for index in reversed(range(8))) + "\n"
        gres.update(cached_record(gres["path"], "gres", text))
        with self.assertRaisesRegex(ValueError, "numeric order"):
            qualification.validate_allocation(record)

    def test_active_configuration_ambiguity_and_trust(self):
        """Other present candidates, writable files and includes each block numerical work."""
        for change in ("other_candidate", "writable", "include", "symlink", "wrong_running_plugin"):
            with self.subTest(change=change):
                record = valid_fixture()
                row = record["cached_configuration"]["candidates"][0]
                if change == "other_candidate":
                    record["cached_configuration"]["candidates"][-1]["status"] = "unreadable"
                elif change == "writable":
                    row["mode"] = "0o664"
                elif change == "include":
                    text = row["filtered_bytes"] + "Include additional.conf\n"
                    row.update(cached_record(row["path"], "slurm", text))
                elif change == "symlink":
                    row["symlink"] = True
                else:
                    record["device_isolation_observations"]["running_slurm_configuration"]["fields"]["TaskPlugin"] = "task/affinity"
                with self.assertRaises(ValueError):
                    qualification.validate_allocation(record)

    def test_missing_typed_file_mapping_is_not_autodetected(self):
        """AutoDetect alone and duplicate device files cannot become mapping proof."""
        for text in ("AutoDetect=nvml\n", "Name=gpu Type=h100 File=/dev/nvidia[0-7]\nName=gpu Type=h100 File=/dev/nvidia4\n"):
            with self.assertRaises(ValueError):
                qualification.gres_devices(text, "ac041")

    def test_strict_profile_and_cuda_identity_types(self):
        """Boolean substitutes and a foreign CUDA UUID must be rejected."""
        profile = copy.deepcopy(qualification.PROFILE)
        profile["schema_version"] = True
        with self.assertRaises(ValueError):
            qualification.require_profile(profile)
        allocation = qualification.validate_allocation(valid_fixture())
        identity = cuda_record(allocation)
        self.assertEqual(qualification.validate_cuda_identity(identity, allocation, 123), identity)
        for key, value in (("pid", True), ("cuda_device_count", True), ("gpu_uuid", "GPU-foreign"), ("pci_address", "0000:10:00.0")):
            changed = {**identity, key: value}
            with self.assertRaises(ValueError):
                qualification.validate_cuda_identity(changed, allocation, 123)

    def test_scope_and_resource_changes_fail(self):
        """A foreign owner, node, step or inflated allocation cannot enter numerical work."""
        for key, value in (("uid", 1), ("hostname", "alogin1"), ("step_id", "batch")):
            record = valid_fixture()
            record["owned_allocation"][key] = value
            with self.assertRaises(ValueError):
                qualification.validate_allocation(record)
        record = valid_fixture()
        record["owned_allocation"]["scheduler"]["job"]["stdout"] = record["owned_allocation"]["scheduler"]["job"]["stdout"].replace("mem=32G", "mem=64G")
        with self.assertRaises(ValueError):
            qualification.validate_allocation(record)

    def test_expand_only_explicit_bounded_lists(self):
        """Leading zeros are retained and unsupported or duplicate expressions fail."""
        self.assertEqual(qualification.expand_list("ac[039-041]"), ["ac039", "ac040", "ac041"])
        for text in ("/dev/nvidia[0-1000]", "/dev/nvidia[0-2,2]", "ac[1-2]x[3-4]"):
            with self.assertRaises(ValueError):
                qualification.expand_list(text)

    def test_cached_file_only_preserves_allowlisted_entries(self):
        """Unrelated values stay out of raw configuration while full bytes remain hashed."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "slurm.conf"
            path.write_text("TaskPlugin=task/cgroup,task/affinity\nUnrelatedPrivateValue=do-not-record\n")
            observed = qualification.cached_file(path, "slurm")
            self.assertNotIn("do-not-record", json.dumps(observed))
            self.assertEqual(observed["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            link = Path(folder) / "linked.conf"
            link.symlink_to(path)
            self.assertEqual(qualification.cached_file(link, "slurm")["status"], "not_regular_file")

    def test_case_insensitive_ambiguity_is_retained_and_rejected(self):
        """Lower-case Slurm entries cannot disappear from capture or override canonical values silently."""
        with self.assertRaises(ValueError):
            qualification.configuration_values("TaskPlugin=task/cgroup,task/affinity\ntaskplugin=task/affinity\n")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "gres.conf"
            path.write_text("name=gpu type=h100 file=/dev/nvidia[0-7]\n")
            observed = qualification.cached_file(path, "gres")
            self.assertEqual(observed["filtered_bytes"], path.read_text())
            with self.assertRaises(ValueError):
                qualification.gres_devices(observed["filtered_bytes"], "ac041")

    def test_raw_query_and_running_field_hash_bindings(self):
        """Changed query fields or selected settings must disagree with preserved raw evidence."""
        for change in ("command", "csv_fields", "running_fields", "running_hash"):
            record = valid_fixture()
            if change == "command":
                record["hardware_queries"]["gpu_csv"]["command"] = ["nvidia-smi", "-q"]
            elif change == "csv_fields":
                record["hardware_queries"]["gpu_csv"]["fields"] = ["uuid"]
            elif change == "running_fields":
                record["device_isolation_observations"]["running_slurm_configuration"]["fields"]["TaskPlugin"] = "task/affinity"
            else:
                record["device_isolation_observations"]["running_slurm_configuration"]["sha256"] = "0" * 64
            with self.subTest(change=change), self.assertRaises(ValueError):
                qualification.validate_allocation(record)

    def test_duplicate_paths_resources_and_foreign_cpuset_fail(self):
        """Duplicate rows never take the last value, and CPU paths must belong to this exact task."""
        for change in ("configuration", "device", "tres", "cpuset"):
            record = valid_fixture()
            if change == "configuration":
                record["cached_configuration"]["candidates"].append(copy.deepcopy(record["cached_configuration"]["candidates"][0]))
            elif change == "device":
                record["host_gpu_inventory"]["device_files"].append(copy.deepcopy(record["host_gpu_inventory"]["device_files"][0]))
            elif change == "tres":
                raw = record["owned_allocation"]["scheduler"]["job"]
                raw["stdout"] = raw["stdout"].replace("AllocTRES=cpu=8,", "AllocTRES=cpu=1,cpu=8,")
            else:
                group, = [row for row in record["device_isolation_observations"]["own_placement"]["cgroups"] if row["controllers"] == ["cpuset"]]
                group["mounts"][0]["levels"][-1]["directory"]["path"] = "/sys/fs/cgroup/cpuset/foreign/task_0"
            with self.subTest(change=change), self.assertRaises(ValueError):
                qualification.validate_allocation(record)


if __name__ == "__main__":
    unittest.main()
