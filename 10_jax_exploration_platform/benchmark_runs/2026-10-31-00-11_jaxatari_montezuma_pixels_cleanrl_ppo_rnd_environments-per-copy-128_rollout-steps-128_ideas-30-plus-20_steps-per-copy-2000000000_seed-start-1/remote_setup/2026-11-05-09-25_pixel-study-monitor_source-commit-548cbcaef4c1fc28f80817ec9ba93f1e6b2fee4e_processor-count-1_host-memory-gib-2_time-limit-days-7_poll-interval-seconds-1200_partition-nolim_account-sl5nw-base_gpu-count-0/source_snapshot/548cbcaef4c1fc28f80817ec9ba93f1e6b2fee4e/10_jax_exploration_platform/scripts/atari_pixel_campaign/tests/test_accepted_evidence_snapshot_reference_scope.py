"""Check only the exact controller-output and historical-ledger reference scope correction."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("snapshot_reference_scope", SCRIPTS / "accepted_evidence_snapshot.py")
snapshot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(snapshot)


def fixture():
    campaign = "/fixture/project/10_jax_exploration_platform/benchmark_runs/campaign"
    source = "/fixture/original_source"
    rows = [{} for number in range(12)]
    helper = str(Path(campaign).parents[2] / snapshot.PREFIX / "diagnose_gpu_execution.py")
    for index, expected in snapshot.LEGACY_DIAGNOSTIC_ANNOTATIONS.items():
        rows[index] = {"kind": "execution_diagnostic", "context": "h100", "hostname": "serval02", "host": "serval02",
            "gpu_uuid": snapshot.CARD_ONE, "pid": expected["pid"], "start_time": expected["start_time"],
            "source_commit": snapshot.NUMERICAL_SOURCE, "source_root": source,
            "helper": {"git_commit": expected["git_commit"], "path": helper, "sha256": expected["sha256"]}}
    owner = {"hostname": "serval02", "gpu_uuid": snapshot.CARD_ONE, "kind": "selected_configuration_measurement",
        "unit": "accepted_unit", "source_commit": snapshot.NUMERICAL_SOURCE, "source_root": source,
        "helper_commit": snapshot.ORIGINAL_HELPER_SOURCE, "pid": 1405247, "start_time": "55826101"}
    rows.append(owner)
    return {"campaign_run_dir": campaign, "source_root": source, "owner_row": owner,
        "expected_owner_fields": deepcopy(owner)}, rows


class ExactReferenceScopeTests(unittest.TestCase):
    """Keep entire-ledger identity checks while refusing selected or unknown annotation scope."""

    def test_exact_known_annotations_preserve_owner_uniqueness(self):
        scope, rows = fixture()
        self.assertEqual(snapshot.validate_frozen_ledger_scope(scope, rows), {9, 10, 11})
        rows.append(deepcopy(scope["owner_row"]))
        with self.assertRaises(ValueError):
            snapshot.validate_frozen_ledger_scope(scope, rows)

    def test_selected_current_row_cannot_be_an_ignored_annotation(self):
        scope, rows = fixture()
        scope["owner_row"] = rows[9]
        scope["expected_owner_fields"] = deepcopy(rows[9])
        with self.assertRaises(ValueError):
            snapshot.validate_frozen_ledger_scope(scope, rows)

    def test_unknown_identity_role_or_source_annotation_is_refused(self):
        for field in ("kind", "pid", "source_commit", "helper"):
            scope, rows = fixture()
            if field == "helper":
                rows[10][field]["sha256"] = "0" * 64
            else:
                rows[10][field] = True
            with self.subTest(field=field), self.assertRaises(ValueError):
                snapshot.validate_frozen_ledger_scope(scope, rows)

    def test_wrong_frozen_ledger_hash_is_refused_before_scope(self):
        scope, rows = fixture()
        with tempfile.TemporaryDirectory() as temporary, snapshot.ValidationFiles() as files:
            path = Path(temporary) / "direct_current_process_ledger_snapshot.json"
            path.write_bytes(snapshot.encoded(rows))
            with self.assertRaises(ValueError):
                files.json({"path": str(path), "sha256": "0" * 64})
            self.assertTrue(files.failed)

    def test_controller_output_roles_require_exact_accepted_unit_and_field(self):
        with tempfile.TemporaryDirectory() as temporary, snapshot.ValidationFiles() as files:
            unit = Path(temporary)
            outputs = {"schema_version": 1, "kind": "actual_direct_child_output_hashes_after_wait"}
            for stream in ("stdout", "stderr"):
                path = unit / ("direct_controller_" + stream + ".raw"); path.write_bytes(stream.encode())
                outputs[stream] = {"path": str(path), "sha256": snapshot.digest(stream.encode()), "size_bytes": len(stream)}
            authority = unit / "direct_controller_output_hashes.json"; authority.write_bytes(snapshot.encoded(outputs))
            reference = {"path": str(authority), "sha256": snapshot.digest(authority.read_bytes())}
            inventory = snapshot.AcceptedInventory(files); inventory.accepted_units[unit] = {"plan": "fixture"}
            inventory.add(reference); inventory.process_metadata(); inventory.resolve_blobs()
            for stream in ("stdout", "stderr"):
                value = outputs[stream]
                self.assertEqual(inventory.shared_metadata[(value["path"], value["sha256"])]["role"], "accepted_controller_" + stream)
            with self.assertRaises(ValueError):
                inventory.add({key: outputs["stdout"][key] for key in ("path", "sha256")}, metadata=False, shared_metadata_role="accepted_controller_stdout",
                    shared_declaration={"authority": reference, "reference_field": ["stderr"]})
            substituted = {"path": str(unit / "other_stdout.raw"), "sha256": outputs["stdout"]["sha256"]}
            with self.assertRaises(ValueError):
                inventory.add(substituted, metadata=False, shared_metadata_role="accepted_controller_stdout",
                    shared_declaration={"authority": reference, "reference_field": ["stdout"]})


if __name__ == "__main__":
    unittest.main()
