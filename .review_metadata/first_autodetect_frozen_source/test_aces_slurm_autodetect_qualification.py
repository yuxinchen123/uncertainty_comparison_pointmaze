"""Check the separate AutoDetect trust boundary with metadata-only synthetic additions."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from test_aces_slurm_qualification import valid_fixture, cached_record, cuda_record
import test_aces_slurm_site_validation as legacy_site_tests


SCRIPT = Path(__file__).resolve().parents[1]
specification = importlib.util.spec_from_file_location("tested_aces_autodetect", SCRIPT / "aces_slurm_autodetect_qualification.py")
policy = importlib.util.module_from_spec(specification)
specification.loader.exec_module(policy)
site = legacy_site_tests.site


def fixture():
    """Add clearly synthetic AutoDetect and directory trust records to the committed metadata fixture."""
    value = valid_fixture()
    value["qualification_profile"] = copy.deepcopy(policy.PROFILE)
    candidates = value["cached_configuration"]["candidates"]
    contents = {"slurm": "TaskPlugin=task/affinity,task/cgroup\nProctrackType=proctrack/cgroup\nGresTypes=gpu\n",
        "gres": "NodeName=ac[041,045,049,055] AutoDetect=nvml\nNodeName=other Name=gpu File=/dev/other\n"}
    for row in candidates:
        if row.get("kind") in contents:
            row.update(cached_record(row["path"], row["kind"], contents[row["kind"]]))
    chain = [{"path": path, "owner_uid": 0, "owner_gid": 0, "mode": "0o755", "directory": True,
        "symlink": False, "inode": index + 100, "device": 1, "mtime_ns": 1} for index, path in enumerate(policy.DIRECTORIES)]
    value["cached_configuration"]["cache_directory_chain"] = {"before": chain, "after": copy.deepcopy(chain)}
    return value


def change_gres(value, text):
    """Replace only the synthetic GRES fixture while keeping its exact hash consistent."""
    row, = [row for row in value["cached_configuration"]["candidates"] if row.get("kind") == "gres"]
    row.update(cached_record(row["path"], "gres", text))


class AutoDetectTests(unittest.TestCase):
    """Require honest separate logical allocation and physical identity evidence."""

    def test_logical_index_does_not_select_a_minor_or_visible_ordinal(self):
        """Changing a consistent logical index never invents a different physical UUID."""
        value = fixture()
        first = policy.validate_allocation(value)
        owned = value["owned_allocation"]
        old_index = owned["allocation_environment"]["SLURM_JOB_GPUS"]
        other = "1" if old_index != "1" else "2"
        owned["scheduler"]["job"]["stdout"] = owned["scheduler"]["job"]["stdout"].replace("IDX:" + old_index + ")", "IDX:" + other + ")")
        owned["allocation_environment"].update(SLURM_JOB_GPUS=other, SLURM_STEP_GPUS=other)
        second = policy.validate_allocation(value)
        self.assertEqual(second["gpu_uuid"], first["gpu_uuid"])
        self.assertEqual(second["physical_minor"], first["physical_minor"])
        self.assertEqual(second["slurm_gres_index"], other)
        self.assertIs(second["logical_allocation_agreement_verified"], True)
        self.assertIs(second["physical_gres_mapping_verified"], False)
        self.assertIs(second["kernel_device_enforcement_verified"], False)
        self.assertFalse(second["workers_permitted"])

    def test_plugin_orders_canonicalize_but_duplicates_extras_and_omissions_fail(self):
        """Only the exact two names survive either written order."""
        expected = ("task/cgroup", "task/affinity")
        for text in ("task/affinity,task/cgroup", "task/cgroup,task/affinity"):
            self.assertEqual(policy.task_plugins(text), expected)
        for text in ("task/cgroup", "task/cgroup,task/cgroup", "task/cgroup,task/affinity,task/none", "affinity,cgroup", None):
            with self.assertRaises(ValueError):
                policy.task_plugins(text)

    def test_matching_autodetect_scope_is_exact_and_has_no_file_claim(self):
        """Unsupported matching overrides and duplicate declarations never become physical mapping proof."""
        accepted = policy.autodetect_configuration("NodeName=ac[041,045] AutoDetect=nvml\n", "ac045")
        self.assertIs(accepted["physical_gres_mapping_verified"], False)
        for text in ("AutoDetect=nvml\n", "NodeName=ac045 AutoDetect=nvidia\n",
                "NodeName=ac045 AutoDetect=nvml\nNodeName=ac045 AutoDetect=nvml\n",
                "AutoDetect=off\nNodeName=ac045 AutoDetect=nvml\n",
                "NodeName=ac045 AutoDetect=nvml Name=gpu File=/dev/nvidia0\n",
                "NodeName=ac045 AutoDetect=nvml\nNodeName=ac045 Name=shard Count=8\n",
                "NodeName=ac045 AutoDetect=nvml autodetect=nvml\n"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                policy.autodetect_configuration(text, "ac045")

    def test_cache_directory_trust_and_capture_stability_are_required(self):
        """A missing, replaced, writable or non-root ancestor blocks numerical initialization."""
        for change in ("missing", "changed", "writable", "owner", "boolean_owner", "symlink", "path"):
            with self.subTest(change=change):
                value = fixture()
                chain = value["cached_configuration"]["cache_directory_chain"]
                if change == "missing":
                    value["cached_configuration"].pop("cache_directory_chain")
                elif change == "changed":
                    chain["after"][-1]["inode"] += 1
                else:
                    field, contents = {"writable": ("mode", "0o775"), "owner": ("owner_uid", 5),
                        "boolean_owner": ("owner_uid", False), "symlink": ("symlink", True), "path": ("path", "/other")}[change]
                    for side in ("before", "after"):
                        chain[side][-1][field] = contents
                with self.assertRaises((ValueError, KeyError)):
                    policy.validate_allocation(value)

    def test_directory_capture_rejects_a_changed_chain(self):
        """The producer saves both observations without silently selecting either version."""
        before = fixture()["cached_configuration"]["cache_directory_chain"]["before"]
        after = copy.deepcopy(before)
        after[-1]["inode"] += 1
        captured = fixture()["cached_configuration"]
        captured.pop("cache_directory_chain")
        with mock.patch.object(policy, "directory_chain", side_effect=[before, after]), mock.patch.object(
                policy.legacy, "configuration_observations", return_value=captured):
            result = policy.configuration_observations({}, {})
        self.assertNotEqual(result["cache_directory_chain"]["before"], result["cache_directory_chain"]["after"])
        value = fixture()
        value["cached_configuration"] = result
        with self.assertRaisesRegex(ValueError, "unchanged"):
            policy.validate_allocation(value)

    def test_no_qualification_without_the_new_directory_evidence(self):
        """Earlier raw allocation observations cannot acquire the new profile retroactively."""
        value = fixture()
        value["cached_configuration"].pop("cache_directory_chain")
        with self.assertRaises(KeyError):
            policy.validate_allocation(value)

    def test_owned_logical_selectors_and_resources_must_agree(self):
        """Separate physical identity never excuses a foreign owner or different logical allocation."""
        for change in ("logical", "owner", "step", "count", "pid", "birth"):
            value = fixture()
            owned = value["owned_allocation"]
            if change == "logical":
                owned["allocation_environment"]["SLURM_STEP_GPUS"] = "999"
            elif change == "owner":
                owned["uid"] = 1
            elif change == "step":
                owned["step_id"] = "batch"
            elif change == "count":
                owned["scheduler"]["step"]["stdout"] = owned["scheduler"]["step"]["stdout"].replace("gres/gpu:h100=1", "gres/gpu:h100=2")
            elif change == "pid":
                owned["pid"] = True
            else:
                owned["process_start_time"] = "0"
            with self.subTest(change=change), self.assertRaises(ValueError):
                policy.validate_allocation(value)

    def test_physical_uuid_pci_and_complete_minor_roster_are_required(self):
        """No physical proof comes from a visible index or a coincidentally equal minor."""
        for change in ("uuid", "pci", "duplicate_minor", "missing_card"):
            value = fixture()
            if change in ("uuid", "pci"):
                hardware = value["hardware_queries"]["gpu_csv"]
                fields = hardware["stdout"].strip().split(",")
                fields[1 if change == "uuid" else 3] = "GPU-foreign" if change == "uuid" else "0000:ff:00.0"
                hardware["stdout"] = ",".join(fields) + "\n"
            elif change == "duplicate_minor":
                rows = value["host_gpu_inventory"]["device_files"]
                rows[1]["minor"] = rows[0]["minor"]
            else:
                value["host_gpu_inventory"]["proc_gpu_inventory"]["cards"].pop()
            with self.subTest(change=change), self.assertRaises(ValueError):
                policy.validate_allocation(value)

    def test_same_process_cuda_identity_and_strict_profile_remain_required(self):
        """A physical identity cannot turn configured trust into a proven scheduler mapping."""
        assignment = policy.validate_allocation(fixture())
        identity = cuda_record(assignment)
        self.assertEqual(policy.validate_cuda_identity(identity, assignment, 123), identity)
        for change in ("profile", "mapping", "kernel", "uuid", "pid"):
            changed, proof = copy.deepcopy(assignment), copy.deepcopy(identity)
            if change == "profile":
                changed["qualification_profile"]["schema_version"] = True
            elif change in ("mapping", "kernel"):
                changed["physical_gres_mapping_verified" if change == "mapping" else "kernel_device_enforcement_verified"] = True
            else:
                proof["gpu_uuid" if change == "uuid" else "pid"] = "GPU-foreign" if change == "uuid" else 124
            with self.subTest(change=change), self.assertRaises(ValueError):
                policy.validate_cuda_identity(proof, changed, 123)

    def test_profile_dispatch_and_helper_closure_preserve_old_default(self):
        """Only explicit selection loads the new helper and its distinct complete profile."""
        self.assertEqual(site.qualifier().PROFILE, policy.legacy.PROFILE)
        selected = site.qualifier(policy.PROFILE)
        self.assertEqual(selected.PROFILE, policy.PROFILE)
        self.assertEqual(set(site.helpers(policy.PROFILE)) - set(site.helpers()), {"aces_slurm_autodetect_qualification.py"})
        self.assertNotIn("aces_slurm_autodetect_qualification.py", site.helpers())
        for value in ({**policy.PROFILE, "physical_gres_mapping_verified": 0}, "unreviewed"):
            with self.assertRaises(ValueError):
                site.qualifier(value)

    def test_new_profile_keeps_the_complete_portable_site_result_chain(self):
        """All four stages and captured shared driver bytes remain mandatory under the new profile."""
        with tempfile.TemporaryDirectory() as folder:
            unit = Path(folder)
            with mock.patch.object(legacy_site_tests, "valid_fixture", side_effect=fixture), mock.patch.object(
                    site, "qualifier", return_value=policy):
                directory, plan, result = legacy_site_tests.full_result_fixture(unit)
            plan["qualification_profile"] = copy.deepcopy(policy.PROFILE)
            (directory / "site_validation_plan.json").write_text(json.dumps(plan))
            result["plan"] = site.receipt(directory / "site_validation_plan.json")
            verified = site.verify_result(directory, unit, plan, result)
            self.assertTrue(verified["passed"])
            self.assertIs(verified["physical_gres_mapping_verified"], False)
            self.assertIs(verified["kernel_device_enforcement_verified"], False)
            self.assertFalse(verified["workers_permitted"])
            driver_path = unit / "probe_driver/captured_driver/libcuda.so.580.178.04"
            driver_bytes = driver_path.read_bytes()
            for change in ("stage", "driver", "dtype"):
                modified = copy.deepcopy(result)
                if change == "stage":
                    modified["children"].pop("renderer")
                elif change == "driver":
                    driver_path.write_bytes(b"different bytes")
                else:
                    path = unit / "probe_result.json"
                    value = json.loads(path.read_text())
                    value["dtypes"] = {}
                    path.write_text(json.dumps(value))
                    modified["probe"] = site.receipt(path)
                with self.subTest(change=change), self.assertRaises(ValueError):
                    site.verify_result(directory, unit, plan, modified)
                driver_path.write_bytes(driver_bytes)


if __name__ == "__main__":
    unittest.main()
