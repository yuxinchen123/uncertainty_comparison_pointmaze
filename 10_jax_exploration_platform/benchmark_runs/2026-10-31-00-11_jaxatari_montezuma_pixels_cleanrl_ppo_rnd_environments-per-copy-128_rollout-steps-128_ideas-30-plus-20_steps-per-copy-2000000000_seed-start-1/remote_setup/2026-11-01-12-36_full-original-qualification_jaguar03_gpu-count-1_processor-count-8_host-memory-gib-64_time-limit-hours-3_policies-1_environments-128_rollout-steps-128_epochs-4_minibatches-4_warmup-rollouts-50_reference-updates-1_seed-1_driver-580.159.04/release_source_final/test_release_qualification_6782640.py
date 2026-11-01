"""Exercise only synthetic metadata; every scheduler call is replaced before use."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

PATH = Path(__file__).with_name("release_qualification_6782640.py")
SPECIFICATION = importlib.util.spec_from_file_location("qualification_release_test_target", PATH)
release = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(release)


class QualificationReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name); self.packet = self.run / "remote_setup/synthetic_qualification"
        self.unit = self.run / "data/synthetic_qualification"
        self.packet.mkdir(parents=True); self.unit.mkdir(parents=True)
        for name, value in (("PACKET", self.packet), ("UNIT", self.unit), ("RUN", self.run)):
            control = patch.object(release, name, value); control.start(); self.addCleanup(control.stop)
        self.plan = {"kind": "selected_configuration_measurement", "synthetic": True}
        self.write(self.unit / "measurement_plan.json", self.plan)
        self.pre = {"schema_version": 1, "kind": "uva_configured_isolation_pre_release_inputs", "job_id": release.JOB,
            "plan": release.ref(self.unit / "measurement_plan.json"), "ledger": {}, "submissions": {},
            "submission": {}, "submission_row": [release.JOB, self.plan["kind"], release.SOURCE, str(self.packet / "job.sbatch")]}
        self.write(self.unit / "configured_isolation_pre_release_inputs.json", self.pre)
        control = patch.object(release, "PRE_RELEASE_SHA", release.sha(release.contents(self.unit / "configured_isolation_pre_release_inputs.json")))
        control.start(); self.addCleanup(control.stop)
        self.intent = {"publication_commit": "a" * 40, "source_sha256": "b" * 64,
            "held_monitor_review": {"path": "synthetic-only", "sha256": "c" * 64}}
        self.calls = []

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(release.encoded(value))

    def held(self):
        fields = {"JobId": release.JOB, "UserId": "sl5nw(2618919)", "Account": "sl5nw_base", "Partition": "gpu",
            "QOS": "csresnolim", "JobState": "PENDING", "Reason": "JobHeldUser", "ReqNodeList": "jaguar03",
            "Reservation": "sl5nw_168", "Requeue": "0", "Restarts": "0", "NumCPUs": "8", "NumTasks": "1",
            "CPUs/Task": "8", "MinMemoryNode": "64G", "TimeLimit": "03:00:00", "NumNodes": "1",
            "Command": str(self.packet / "job.sbatch"), "TresPerNode": "gres/gpu:nvidia_rtx_a4500:1",
            "ReqTRES": "cpu=8,mem=64G,node=1,billing=8,gres/gpu=1"}
        return " ".join(key + "=" + value for key, value in fields.items())

    def fake_run(self, command, **arguments):
        self.assertTrue((self.packet / "release_intent.json").is_file())
        self.calls.append(command)
        if command == ["scontrol", "show", "job", "-o", release.JOB]:
            return SimpleNamespace(returncode=0, stdout=self.held().encode(), stderr=b"")
        self.assertEqual(command, ["scontrol", "release", release.JOB])
        self.assertTrue((self.packet / "release_preflight_return.json").is_file())
        self.assertFalse((self.packet / "release_receipt.json").exists())
        self.assertFalse((self.unit / "owned_allocation_inputs.json").exists())
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    def fake_core(self, plan):
        self.assertEqual(plan, self.plan)
        def owned_inputs(observed):
            self.assertTrue((self.packet / "release_receipt.json").is_file())
            self.assertEqual(observed, self.plan)
            return release.document(self.unit / "owned_allocation_inputs.json")
        return SimpleNamespace(owned_inputs=owned_inputs)

    def test_exact_single_card_held_shape_and_changed_fields(self):
        raw = self.held(); release.validate_held(raw)
        for old, new in (("JobId=" + release.JOB, "JobId=999999"), ("Requeue=0", "Requeue=1"),
                ("Restarts=0", "Restarts=1"), ("NumCPUs=8", "NumCPUs=16"), ("Reservation=sl5nw_168", "Reservation=other"),
                ("JobHeldUser", "Priority"), ("gpu:nvidia_rtx_a4500:1", "gpu:nvidia_rtx_a4500:2"),
                ("cpu=8,", "cpu=8,cpu=8,"), ("gres/gpu=1", "gres/gpu=1,gres/gpu:other=1")):
            with self.subTest(old=old), self.assertRaises(ValueError): release.validate_held(raw.replace(old, new))
        release.validate_held(raw.replace("gres/gpu=1", "gres/gpu=1,gres/gpu:nvidia_rtx_a4500=1"))
        with self.assertRaises(ValueError):
            release.validate_held(raw.replace("gres/gpu=1", "gres/gpu=1,gres/gpu:nvidia_rtx_a4500=2"))

    def test_one_intent_raw_before_parse_real_receipt_before_atomic_inputs(self):
        original = release.validate_held
        def parsed(raw):
            self.assertEqual((self.packet / "release_preflight_stdout.raw").read_text(), raw)
            self.assertTrue((self.packet / "release_preflight_return.json").is_file())
            original(raw)
        with patch.object(release.subprocess, "run", side_effect=self.fake_run), patch.object(release, "core", side_effect=self.fake_core),\
                patch.object(release, "validate_held", side_effect=parsed):
            result = release.release_once(self.intent)
            self.assertTrue(result["actual_core_reader_passed"])
            self.assertEqual(len(self.calls), 2)
            final = release.document(self.unit / "owned_allocation_inputs.json")
            self.assertEqual({key: value for key, value in final.items() if key not in {"kind", "release"}},
                {key: value for key, value in self.pre.items() if key != "kind"})
            self.assertEqual(final["release"], release.ref(self.packet / "release_receipt.json"))
            self.assertEqual(release.document(final["release"]["path"])["kind"], "owned_uva_configured_isolation_release")
            with self.assertRaises(ValueError): release.release_once(self.intent)
            self.assertEqual(len(self.calls), 2)

    def test_bad_preflight_keeps_raw_and_never_releases(self):
        def bad(command, **arguments):
            self.calls.append(command)
            return SimpleNamespace(returncode=0, stdout=self.held().replace("JobHeldUser", "Priority").encode(), stderr=b"")
        with patch.object(release.subprocess, "run", side_effect=bad):
            with self.assertRaises(ValueError): release.release_once(self.intent)
        self.assertEqual(len(self.calls), 1)
        self.assertTrue((self.packet / "release_preflight_stdout.raw").exists())
        self.assertTrue((self.packet / "release_intent.json").exists())
        self.assertFalse((self.packet / "release_receipt.json").exists())
        self.assertFalse((self.unit / "owned_allocation_inputs.json").exists())

    def test_nonzero_release_retains_return_without_receipt_or_final_inputs(self):
        def failed(command, **arguments):
            if command[1] == "show": return self.fake_run(command, **arguments)
            self.calls.append(command)
            return SimpleNamespace(returncode=1, stdout=b"", stderr=b"synthetic scheduler failure")
        with patch.object(release.subprocess, "run", side_effect=failed):
            with self.assertRaises(ValueError): release.release_once(self.intent)
        self.assertEqual(release.document(self.packet / "release_return.json")["returncode"], 1)
        self.assertFalse((self.packet / "release_receipt.json").exists())
        self.assertFalse((self.unit / "owned_allocation_inputs.json").exists())

    def test_missing_publication_never_creates_intent_or_calls_a_command(self):
        with patch.object(release.subprocess, "run") as query:
            with self.assertRaises(ValueError): release.release_once({**self.intent, "publication_commit": None})
            query.assert_not_called()
        self.assertFalse((self.packet / "release_intent.json").exists())

    def test_timeout_keeps_partial_raw_and_prevents_a_second_attempt(self):
        with patch.object(release.subprocess, "run", side_effect=release.subprocess.TimeoutExpired(["synthetic"], 30, output=b"partial", stderr=b"error")):
            with self.assertRaises(release.subprocess.TimeoutExpired): release.release_once(self.intent)
        self.assertEqual((self.packet / "release_preflight_stdout.raw").read_bytes(), b"partial")
        self.assertIsNone(release.document(self.packet / "release_preflight_return.json")["returncode"])
        with self.assertRaises(ValueError): release.release_once(self.intent)

    def test_atomic_publication_never_overwrites_existing_bytes(self):
        target = self.unit / "owned_allocation_inputs.json"; target.write_bytes(b"existing")
        with self.assertRaises(ValueError): release.put(target, b"changed", atomic=True)
        self.assertEqual(target.read_bytes(), b"existing")

    def test_changed_pre_release_envelope_after_actual_release_refuses_final_inputs(self):
        def changed(command, **arguments):
            result = self.fake_run(command, **arguments)
            if command[1] == "release": self.write(self.unit / "configured_isolation_pre_release_inputs.json", {**self.pre, "job_id": "999999"})
            return result
        with patch.object(release.subprocess, "run", side_effect=changed):
            with self.assertRaises(ValueError): release.release_once(self.intent)
        self.assertTrue((self.packet / "release_receipt.json").exists())
        self.assertFalse((self.unit / "owned_allocation_inputs.json").exists())


if __name__ == "__main__":
    unittest.main()
