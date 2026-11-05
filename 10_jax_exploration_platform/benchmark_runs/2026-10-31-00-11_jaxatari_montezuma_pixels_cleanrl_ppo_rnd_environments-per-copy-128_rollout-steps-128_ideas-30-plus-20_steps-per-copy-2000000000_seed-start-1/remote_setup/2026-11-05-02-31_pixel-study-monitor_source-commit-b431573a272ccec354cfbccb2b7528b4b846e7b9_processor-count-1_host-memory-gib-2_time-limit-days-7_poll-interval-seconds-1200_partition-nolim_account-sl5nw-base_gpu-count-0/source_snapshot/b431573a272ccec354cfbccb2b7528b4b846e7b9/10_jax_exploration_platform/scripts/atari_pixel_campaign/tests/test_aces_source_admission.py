"""Check ACES exact relocated admission with real Git and synthetic raw gates only."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import aces_source_admission as admission
import aces_slurm_site_validation as site
import h100_nvl_slurm_evidence as history
import post_comparison_validation as post


class PortableAdmissionTests(unittest.TestCase):
    """Keep all remote records synthetic and never initialize a numerical library."""

    def setUp(self):
        """Create one relocated real D28 checkout and explicitly copied synthetic gate files."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.stage = self.root / "stage"
        self.stage.mkdir()
        self.source = self.stage / "numerical_source"
        project = SCRIPTS.parents[2]
        subprocess.run(["git", "clone", "-q", "--shared", "--no-checkout", str(project), str(self.source)], check=True)
        subprocess.run(["git", "-C", str(self.source), "sparse-checkout", "init", "--cone"], check=True)
        subprocess.run(["git", "-C", str(self.source), "sparse-checkout", "set", "10_jax_exploration_platform/src",
            "10_jax_exploration_platform/scripts", "10_jax_exploration_platform/tests"], check=True)
        subprocess.run(["git", "-C", str(self.source), "checkout", "-q", "--detach", admission.SOURCE_COMMIT], check=True)
        self.logical_source = "/synthetic/original/source"
        self.run = "/synthetic/original/campaign"
        self.manifest = {"schema_version": 1, "kind": "portable_raw_evidence", "files": [], "git_roots": [
            {"original_root": self.logical_source, "relative_path": "numerical_source", "commit": admission.SOURCE_COMMIT}]}
        relatives = site.common().numerical_paths(self.source)
        for relative in relatives:
            self.map_file(self.logical_source + "/" + relative, "numerical_source/" + relative)
        self.configuration = history.configuration(site.common().source_identity(self.source, admission.SOURCE_COMMIT, relatives), post.Evidence())
        self.gates = {}
        for context in admission.CONTEXTS:
            hardware = {"hostname": "serval02" if context == "h100" else "jaguar03",
                "gpu_query": "NVIDIA H100 NVL" if context == "h100" else "NVIDIA RTX A4500",
                "runtime_environment": site.common().RUNTIME}
            checks = {}
            for kind in ("learner", "renderer"):
                command = (["/synthetic/original/bin/python", "-m", "pytest", "-q",
                    self.logical_source + "/10_jax_exploration_platform/tests/atari_pixels",
                    self.logical_source + "/10_jax_exploration_platform/tests/test_atari_pixel_environment.py"] if kind == "learner" else
                    ["/synthetic/original/bin/python", self.logical_source + "/" + site.PREFIX + "check_rendering.py", "--candidate", "all"])
                raw_receipt = self.document(self.run + "/" + context + "_" + kind + ".json",
                    {"passed": True, "exit_code": 0, "source_commit": admission.SOURCE_COMMIT, "command": command, "hardware": hardware})
                # Actual original gates include the checked command/hardware beside the raw receipt.
                checks[kind] = {**raw_receipt, "command": command, "hardware": hardware}
            self.gates[context] = self.document(self.run + "/" + context + ".json", {"passed": True,
                "source_commit": admission.SOURCE_COMMIT, "hardware_class": context, "checks": checks})
        reference = self.document(self.run + "/reference.json", self.configuration["learner"])
        self.ready = {"authorized": True, "source_root": self.logical_source, "source_commit": admission.SOURCE_COMMIT,
            "contexts": list(admission.CONTEXTS), "matched_copy_counts": [1, 2],
            "reference_configuration_path": reference["path"], "gate_records": {key: row["path"] for key, row in self.gates.items()}}
        self.document(self.run + "/SOURCE_AND_GATES_READY.json", self.ready)
        self.manifest_path = self.stage / "portable_evidence.json"
        self.binding = {"schema_version": 1, "kind": "aces_original_source_portable_admission", "campaign_run_dir": self.run,
            "evidence_manifest": {"path": str(self.manifest_path), "sha256": ""}}
        self.publish_manifest()

    def map_file(self, original, relative):
        """Bind an already written staged file to one exact original logical identity."""
        self.manifest["files"].append({"original_path": original, "relative_path": relative,
            "sha256": hashlib.sha256((self.stage / relative).read_bytes()).hexdigest()})

    def document(self, original, value):
        """Write an explicitly synthetic unchanged JSON copy and its original path receipt."""
        relative = "raw/" + Path(original).name
        target = self.stage / relative
        target.parent.mkdir(exist_ok=True)
        target.write_text(json.dumps(value, indent=2) + "\n")
        self.map_file(original, relative)
        return {"path": original, "sha256": self.manifest["files"][-1]["sha256"]}

    def publish_manifest(self):
        """Pin the exact current mapping bytes so intentional negative fixtures are explicit."""
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2) + "\n")
        self.binding["evidence_manifest"]["sha256"] = hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()

    def test_relocated_complete_admission_keeps_original_paths_and_native_source(self):
        """Nonexistent original paths pass only through all exact mappings and detached Git bytes."""
        expected, numerical, configuration = admission.snapshot(self.binding)
        self.assertEqual(expected["source_root"], self.logical_source)
        self.assertEqual(expected["files"][0]["path"], self.run + "/SOURCE_AND_GATES_READY.json")
        self.assertEqual(numerical["source_root"], str(self.source))
        self.assertEqual(configuration, self.configuration)
        plan = {"campaign_run_dir": "/synthetic/remote/campaign", "portable_admission": self.binding, "admission": expected, "numerical_source": numerical, "configuration": configuration}
        self.assertEqual(admission.check(plan), expected)
        packet = self.root / "packet"
        packet.mkdir()
        copied = admission.archive(packet, expected, self.binding)
        site.common().validate_admission_archive(packet, expected, copied)
        for row in copied["files"]:
            self.assertEqual(row["original"]["sha256"], row["captured"]["sha256"])
        self.assertNotIn("jax", sys.modules)
        self.assertNotIn("numpy", sys.modules)

    def test_omitted_mapping_never_uses_original_or_path_prefix(self):
        """A missing individual gate or source file fails even when related mappings remain."""
        original = copy.deepcopy(self.manifest)
        for name in ("h100_learner.json", "config.py"):
            self.manifest = copy.deepcopy(original)
            self.manifest["files"] = [row for row in self.manifest["files"] if Path(row["original_path"]).name != name]
            self.publish_manifest()
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "no exact file entry"):
                admission.snapshot(self.binding)

    def test_changed_manifest_byte_or_path_or_source_rejects(self):
        """The fixed receipt rejects byte changes and a source file cannot differ from Git."""
        initial = self.manifest_path.read_bytes()
        self.manifest_path.write_bytes(initial + b" ")
        with self.assertRaisesRegex(ValueError, "required SHA256"):
            admission.snapshot(self.binding)
        self.manifest_path.write_bytes(initial)
        changed = copy.deepcopy(self.binding)
        changed["evidence_manifest"]["path"] = str(self.stage / "absent.json")
        with self.assertRaises(FileNotFoundError):
            admission.snapshot(changed)
        target = self.source / "10_jax_exploration_platform/src/exploration_platform/atari_pixels/config.py"
        target.write_bytes(target.read_bytes() + b"\n# changed synthetic source\n")
        with self.assertRaisesRegex(ValueError, "changed|SHA256"):
            admission.snapshot(self.binding)

    def test_forged_mapping_hash_still_cannot_override_git_or_gate(self):
        """A newly hashed edited source remains invalid against its clean pinned Git root."""
        target = self.source / "10_jax_exploration_platform/src/exploration_platform/atari_pixels/config.py"
        target.write_bytes(target.read_bytes() + b"\n# deliberately changed source\n")
        for row in self.manifest["files"]:
            if row["relative_path"].endswith("/config.py"):
                row["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        self.publish_manifest()
        with self.assertRaisesRegex(ValueError, "uncommitted|changed|clean detached"):
            admission.snapshot(self.binding)

    def test_rehashed_partial_gate_command_and_wrong_hardware_reject(self):
        """Even coherent new hashes cannot replace a full original gate with a partial test."""
        initial = copy.deepcopy(self.manifest)
        check_path = self.stage / "raw/h100_learner.json"
        gate_path = self.stage / "raw/h100.json"
        original_check, original_gate = check_path.read_bytes(), gate_path.read_bytes()
        for change in ("command", "hardware"):
            self.manifest = copy.deepcopy(initial)
            check = json.loads(original_check)
            if change == "command":
                check["command"].extend(["-k", "one_test"])
            else:
                check["hardware"]["gpu_query"] = "synthetic unrelated GPU"
            check_path.write_text(json.dumps(check))
            gate = json.loads(original_gate)
            gate["checks"]["learner"].update(sha256=hashlib.sha256(check_path.read_bytes()).hexdigest(),
                command=check["command"], hardware=check["hardware"])
            gate_path.write_text(json.dumps(gate))
            for row in self.manifest["files"]:
                if row["relative_path"] in ("raw/h100_learner.json", "raw/h100.json"):
                    row["sha256"] = hashlib.sha256((self.stage / row["relative_path"]).read_bytes()).hexdigest()
            self.publish_manifest()
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "original gate|hardware context"):
                admission.snapshot(self.binding)

    def test_recorded_gate_metadata_must_equal_its_raw_record(self):
        """Original redundant command and hardware fields cannot disagree with the raw checked bytes."""
        path = self.stage / "raw/h100.json"
        original = path.read_bytes()
        for field in ("command", "hardware"):
            gate = json.loads(original)
            gate["checks"]["learner"][field] = ["changed"] if field == "command" else {"changed": True}
            path.write_text(json.dumps(gate))
            for row in self.manifest["files"]:
                if row["relative_path"] == "raw/h100.json":
                    row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            self.publish_manifest()
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "checked gate metadata"):
                admission.snapshot(self.binding)

    def test_changed_plan_identity_and_configuration_reject(self):
        """An internally valid mapping cannot change its native source or admitted configuration."""
        original, numerical, configuration = admission.snapshot(self.binding)
        plan = {"campaign_run_dir": "/synthetic/remote/campaign", "portable_admission": self.binding, "admission": original, "numerical_source": numerical, "configuration": configuration}
        for key in ("admission", "numerical_source", "configuration"):
            changed = copy.deepcopy(plan)
            changed[key]["changed"] = True
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "changed"):
                admission.check(changed)

    def test_strict_binding_schema_and_manifest_symlink_reject(self):
        """A numeric schema substitute, omitted field or linked manifest is not an explicit binding."""
        for change in ({"schema_version": True}, {"kind": "unrelated"}, {"campaign_run_dir": "/synthetic/../campaign"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                admission.snapshot({**self.binding, **change})
        linked = self.stage / "linked.json"
        linked.symlink_to(self.manifest_path)
        changed = copy.deepcopy(self.binding)
        changed["evidence_manifest"]["path"] = str(linked)
        with self.assertRaisesRegex(ValueError, "ordinary exact"):
            admission.snapshot(changed)

    def test_native_software_is_observed_and_cannot_be_substituted_by_mapping(self):
        """Original source mapping does not replace actual target software or asset bytes."""
        target = self.root / "software_at_prepare.json"
        original = {"installed_backend": {"packages": {"jax": "synthetic expected version"}},
            "asset_receipt": {"identity": "synthetic expected assets"}}
        target.write_text(json.dumps(original))
        plan = {"portable_admission": self.binding, "software_at_prepare": site.receipt(target)}
        with mock.patch.object(site, "software_observations", return_value=original) as observe:
            self.assertEqual(site.checked_software(plan), original)
            observe.assert_called_once_with(plan)
        for key in ("installed_backend", "asset_receipt"):
            changed = copy.deepcopy(original)
            changed[key]["changed"] = True
            with self.subTest(key=key), mock.patch.object(site, "software_observations", return_value=changed):
                with self.assertRaisesRegex(ValueError, "actual target"):
                    site.checked_software(plan)

    def test_ordinary_aces_and_uva_entry_points_keep_the_existing_path(self):
        """No explicit portable binding retains the previous native admission functions."""
        old = mock.Mock()
        old.snapshot_admission.return_value = {"source_root": "/ordinary/source"}
        old.source_identity.return_value = {"native": True}
        old.original_configuration.return_value = {"configuration": True}
        with mock.patch.object(site, "common", return_value=old):
            result = site.admission_inputs("/ordinary/run")
            self.assertEqual(result, ({"source_root": "/ordinary/source"}, {"native": True}, {"configuration": True}))
            site.check_admission({"campaign_run_dir": "/ordinary/run", "admission": result[0]})
            old.check_admission.assert_called_once_with("/ordinary/run", result[0])
        project = SCRIPTS.parents[2]
        before = subprocess.check_output(["git", "-C", str(project), "show", "e6fa1040236f4b219d003061963a24f7c5ce82d2:" + site.PREFIX + "h100_nvl_slurm_site_validation.py"])
        self.assertEqual((SCRIPTS / "h100_nvl_slurm_site_validation.py").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
