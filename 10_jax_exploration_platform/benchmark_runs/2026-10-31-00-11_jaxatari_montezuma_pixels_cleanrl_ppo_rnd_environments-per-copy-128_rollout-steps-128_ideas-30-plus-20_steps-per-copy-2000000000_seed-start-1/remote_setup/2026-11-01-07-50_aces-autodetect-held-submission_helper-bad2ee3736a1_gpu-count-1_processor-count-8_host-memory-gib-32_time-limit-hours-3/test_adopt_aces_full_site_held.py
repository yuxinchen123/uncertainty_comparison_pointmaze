"""Check local adoption ordering and exact bytes without any cluster or process action."""

import base64
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location("tested_" + name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adoption = load("adopt_aces_full_site_held")
wrapper = load("submit_aces_full_site_held")


class AdoptionTests(unittest.TestCase):
    """Use the real wrapper's receipt constructor and private synthetic returned files."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.plan = json.loads((ROOT / "aces_full_site_held_submission_plan.json").read_bytes())
        self.run = self.root / "run"
        self.packet = self.run / "remote_setup" / Path(self.plan["native_packet"]).name
        self.packet.mkdir(parents=True)
        (self.run / "slurm").mkdir()
        self.ledger = self.run / "slurm/submitted_jobids_aces.txt"
        self.table = self.run / "slurm/submissions_aces.tsv"
        self.ledger.write_text("101\n")
        self.table.write_text("101\tolder\tolder\tolder\n")
        self.received = self.root / "execution/received"
        self.received.mkdir(parents=True)
        self.plan.update(local_campaign_run_dir=str(self.run), local_packet=str(self.packet),
            receipt_directory=str(self.received.parent), local_files=[])
        self.receipt = wrapper.submission_receipt(self.plan, self.plan["native_packet"], "202", "b" * 40,
            adoption.DRIVER_SHA, b"202\n", b"synthetic account message\n")
        self.row = self.receipt["submission_row"]
        native_intent = {"schema_version": 1, "kind": "aces_full_site_held_submission_intent",
            "driver_sha256": adoption.DRIVER_SHA, "submission_plan_sha256": adoption.PLAN_SHA,
            "published_preparation_commit": "b" * 40, "held_command": self.plan["held_command"],
            "automatic_retry_permitted": False, **adoption.BOUNDARY}
        local_intent = {**native_intent, "kind": "aces_full_site_held_submission_local_intent", "native_packet": self.plan["native_packet"]}
        self.original_intent = adoption.json_bytes(local_intent)
        (self.packet / "held_submission_intent.json").write_bytes(self.original_intent)
        (self.received.parent / "local_intent.json").write_bytes(self.original_intent)
        self.values = {"submission_receipt.json": adoption.json_bytes(self.receipt),
            "held_submission_intent.json": adoption.json_bytes(native_intent), "held_submission_stdout.raw": b"202\n",
            "held_submission_stderr.raw": b"synthetic account message\n", "submitted_jobids_aces.txt": b"101\n202\n",
            "submissions_aces.tsv": ("\t".join(self.row) + "\n").encode()}
        self.values["held_submission_return.json"] = adoption.json_bytes({"returncode": 0, "automatic_retry_permitted": False,
            **{suffix: {"path": str(Path(self.plan["native_packet"]) / ("held_submission_" + suffix + ".raw")),
                "sha256": adoption.digest(self.values["held_submission_" + suffix + ".raw"]),
                "bytes": len(self.values["held_submission_" + suffix + ".raw"])} for suffix in ("stdout", "stderr")}})
        self.write_returned()

    def write_returned(self):
        files = []
        for name, raw in self.values.items():
            (self.received / name).write_bytes(raw)
            parent = Path(self.plan["native_run"]) / "slurm" if name in {"submitted_jobids_aces.txt", "submissions_aces.tsv"} else Path(self.plan["native_packet"])
            files.append({"path": str(parent / name), "name": name, "bytes": len(raw), "sha256": adoption.digest(raw),
                "contents_base64": base64.b64encode(raw).decode()})
        raw = (json.dumps({"receipt": self.receipt, "files": files}) + "\n").encode()
        (self.received.parent / "stdout.raw").write_bytes(raw)
        (self.received.parent / "stderr.raw").write_bytes(b"")
        transport = {"returncode": 0, "stdout": {"path": str(self.received.parent / "stdout.raw"), "sha256": adoption.digest(raw), "bytes": len(raw)},
            "stderr": {"path": str(self.received.parent / "stderr.raw"), "sha256": adoption.digest(b""), "bytes": 0}}
        (self.received.parent / "transport_result.json").write_text(json.dumps(transport))

    def test_exact_native_receipt_is_published_only_after_both_local_appends(self):
        publish = adoption.publish
        def ordered(path, raw):
            if Path(path).name in ("remote_held_submission_intent.json", "submission_receipt.json"):
                self.assertEqual(self.ledger.read_text(), "101\n202\n")
                self.assertEqual(self.table.read_text().splitlines()[-1], "\t".join(self.row))
            return publish(path, raw)
        with mock.patch.object(adoption, "publish", side_effect=ordered):
            result = adoption.adopt(self.plan, self.received)
        self.assertTrue(result["passed"])
        self.assertEqual((self.packet / "submission_receipt.json").read_bytes(), self.values["submission_receipt.json"])
        self.assertEqual((self.packet / "remote_held_submission_intent.json").read_bytes(), self.values["held_submission_intent.json"])
        self.assertEqual((self.packet / "held_submission_intent.json").read_bytes(), self.original_intent)
        with self.assertRaisesRegex(ValueError, "already attempted"):
            adoption.adopt(self.plan, self.received)
        self.assertEqual(self.ledger.read_text(), "101\n202\n")

    def test_changed_returned_hash_or_source_identity_never_appends(self):
        (self.received / "held_submission_stdout.raw").write_bytes(b"999\n")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            adoption.adopt(self.plan, self.received)
        self.assertEqual(self.ledger.read_text(), "101\n")
        self.write_returned()
        self.receipt["submission_driver_sha256"] = "f" * 64
        self.values["submission_receipt.json"] = adoption.json_bytes(self.receipt)
        self.write_returned()
        with self.assertRaisesRegex(ValueError, "another packet or wrapper"):
            adoption.adopt(self.plan, self.received)
        self.assertEqual(self.ledger.read_text(), "101\n")

    def test_missing_remote_ledger_membership_and_boolean_credit_reject(self):
        self.values["submitted_jobids_aces.txt"] = b"101\n"
        self.write_returned()
        with self.assertRaisesRegex(ValueError, "both returned native"):
            adoption.adopt(self.plan, self.received)
        self.values["submitted_jobids_aces.txt"] = b"101\n202\n"
        self.receipt["idea_credit"] = False
        self.values["submission_receipt.json"] = adoption.json_bytes(self.receipt)
        self.write_returned()
        with self.assertRaisesRegex(ValueError, "scientific credit"):
            adoption.adopt(self.plan, self.received)
        self.assertEqual(self.ledger.read_text(), "101\n")

    def test_existing_or_inconsistent_local_ids_reject_without_an_intent(self):
        for ids, rows in (("101\n202\n", "101\tolder\n"), ("101\n", "202\tforeign\n"), ("101", "101\tolder\n")):
            self.ledger.write_text(ids)
            self.table.write_text(rows)
            with self.assertRaises(ValueError):
                adoption.adopt(self.plan, self.received)
            self.assertFalse((self.packet / "submission_adoption_intent.json").exists())

    def test_busy_or_linked_local_ledgers_never_write(self):
        with mock.patch.object(adoption.fcntl, "flock", side_effect=BlockingIOError("busy own ledger")):
            with self.assertRaises(BlockingIOError):
                adoption.adopt(self.plan, self.received)
        self.assertFalse((self.packet / "submission_adoption_intent.json").exists())
        self.ledger.unlink()
        target = self.root / "kept"
        target.write_text("101\n")
        self.ledger.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "ordinary adoption"):
            adoption.adopt(self.plan, self.received)
        self.assertEqual(target.read_text(), "101\n")

    def test_publication_failure_keeps_durable_id_intent_and_forbids_retry(self):
        publish = adoption.publish
        def fail_receipt(path, raw):
            if Path(path).name == "submission_receipt.json":
                raise OSError("synthetic receipt publication failure")
            return publish(path, raw)
        with mock.patch.object(adoption, "publish", side_effect=fail_receipt):
            with self.assertRaisesRegex(OSError, "synthetic"):
                adoption.adopt(self.plan, self.received)
        self.assertEqual(self.ledger.read_text(), "101\n202\n")
        self.assertTrue((self.packet / "submission_adoption_failure.json").is_file())
        self.assertFalse((self.packet / "submission_receipt.json").exists())
        with self.assertRaisesRegex(ValueError, "already attempted"):
            adoption.adopt(self.plan, self.received)


if __name__ == "__main__":
    unittest.main()
