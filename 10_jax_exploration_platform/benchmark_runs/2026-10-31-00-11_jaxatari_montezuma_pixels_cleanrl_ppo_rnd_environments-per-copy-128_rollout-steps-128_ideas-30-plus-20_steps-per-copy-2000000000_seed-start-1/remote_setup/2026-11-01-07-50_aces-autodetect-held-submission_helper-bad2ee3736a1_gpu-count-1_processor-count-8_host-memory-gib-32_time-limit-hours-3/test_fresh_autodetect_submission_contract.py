"""Check the fresh native packet and narrow wrapper pins without any cluster action."""

import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent


def load(path, name):
    """Load only reviewed standard-library metadata utilities."""
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


driver = load(ROOT / "submit_aces_full_site_held.py", "fresh_autodetect_submission")
adoption = load(ROOT / "adopt_aces_full_site_held.py", "fresh_autodetect_adoption")


class FreshAutoDetectContractTests(unittest.TestCase):
    """Keep exact native bytes, old-source refusal and local-only operations distinct."""

    def setUp(self):
        """Copy the real immutable packet into a private temporary directory for coherent negative tests."""
        self.plan = json.loads((ROOT / "aces_full_site_held_submission_plan.json").read_bytes())
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.packet, self.data = self.root / "packet", self.root / "data"
        shutil.copytree(self.plan["local_packet"], self.packet)
        source_data = Path(self.plan["local_campaign_run_dir"]) / "data" / Path(self.plan["local_packet"]).name
        shutil.copytree(source_data, self.data)

    def test_actual_new_packet_is_supported_and_original_wrapper_remains_unchanged(self):
        """The new pin accepts bad2/AutoDetect; the historical wrapper still refuses this new helper."""
        native = driver.validate_packet(self.plan, self.packet, self.data)
        self.assertEqual(native["helpers"]["source_commit"], "bad2ee3736a1451094d5d1644b36b8cfc2d91c38")
        self.assertEqual(len(native["helpers"]["files"]), 18)
        self.assertFalse(native["qualification_profile"]["physical_gres_mapping_verified"])
        original = Path(self.plan["local_campaign_run_dir"]) / "remote_setup/2026-11-01-02-55_aces-full-site-held-submission_helper-8e43cb1dc963_gpu-count-1_processor-count-8_host-memory-gib-32_time-limit-hours-3/submit_aces_full_site_held.py"
        old = load(original, "original_aces_submission_preserved")
        with self.assertRaisesRegex(ValueError, "changed its source"):
            old.validate_packet(self.plan, self.packet, self.data)

    def test_rehashed_profile_or_helper_changes_are_rejected(self):
        """A coherent rewritten test plan cannot change the fixed profile or enable mapping/enforcement claims."""
        original = (self.packet / "site_validation_plan.json").read_bytes()
        for kind in ("old_helper", "old_profile", "physical_mapping_claim", "kernel_enforcement_claim"):
            plan = copy.deepcopy(self.plan)
            native = json.loads(original)
            if kind == "old_helper":
                native["helpers"]["source_commit"] = "8e43cb1dc963045fc28f4e611411097efd93185a"
                plan["helpers"] = native["helpers"]
            elif kind == "old_profile":
                native["qualification_profile"]["name"] = "aces-h100-pcie-driver-580.178.04"
            elif kind == "physical_mapping_claim":
                native["qualification_profile"]["physical_gres_mapping_verified"] = True
            else:
                native["qualification_profile"]["kernel_device_enforcement_verified"] = True
            plan["qualification_profile"] = native["qualification_profile"]
            path = self.packet / "site_validation_plan.json"
            path.write_text(json.dumps(native) + "\n")
            for row in plan["native_packet_files"]:
                if row["relative_path"] == path.name:
                    row.update(sha256=driver.digest(path.read_bytes()), bytes=path.stat().st_size)
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "changed its source"):
                driver.validate_packet(plan, self.packet, self.data)

    def test_default_validation_has_no_intent_ledger_or_cluster_action(self):
        """Read all actual prepared inputs while every real process call is forbidden."""
        run = Path(self.plan["local_campaign_run_dir"])
        ledger_paths = [run / "slurm/submitted_jobids_aces.txt", run / "slurm/submissions_aces.tsv"]
        before = {path: path.read_bytes() for path in ledger_paths}
        with patch.object(driver.subprocess, "run", side_effect=AssertionError("no live commands")):
            payload, plan = driver.local_payload(ROOT / "aces_full_site_held_submission_plan.json")
        self.assertEqual(plan["helpers"]["source_commit"], "bad2ee3736a1451094d5d1644b36b8cfc2d91c38")
        self.assertEqual(plan["qualification_profile"]["name"], "aces-h100-pcie-driver-580.178.04-autodetect-nvml-configured-isolation")
        self.assertEqual({path: path.read_bytes() for path in ledger_paths}, before)
        self.assertFalse(Path(plan["receipt_directory"]).exists())
        for name in ("held_submission_intent.json", "submission_receipt.json", "release_receipt.json"):
            self.assertFalse((Path(plan["local_packet"]) / name).exists())
        self.assertLessEqual(len(json.dumps(payload).encode()) + 1, driver.MAX_BYTES)
        self.assertEqual(driver.PLAN_SHA, adoption.PLAN_SHA)
        self.assertEqual(driver.digest((ROOT / "submit_aces_full_site_held.py").read_bytes()), adoption.DRIVER_SHA)

    def test_all_frozen_local_inputs_fit_the_unchanged_adoption_read_bound(self):
        """Returning many small audit files does not silently exceed the original bounded local reader."""
        self.assertLessEqual((ROOT / "aces_full_site_held_submission_plan.json").stat().st_size, adoption.MAX_BYTES)
        self.assertTrue(all(row["bytes"] <= adoption.MAX_BYTES for row in self.plan["local_files"]))
        self.assertEqual(len(self.plan["native_packet_files"]), 11)
        self.assertEqual(len(self.plan["native_data_files"]), 4)
        self.assertTrue(any(Path(row["path"]).name == "independent_actual_native_preparation_verification.json"
            for row in self.plan["local_files"]))


if __name__ == "__main__":
    unittest.main()
