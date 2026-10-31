"""Verify exact staged evidence and historical Git reads using synthetic standard-library records."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import portable_evidence as portable
import post_comparison_validation as validation

SPECIFICATION = importlib.util.spec_from_file_location("portable_fixture_defaults", SCRIPTS / "diagnose_gpu_execution.py")
DEFAULTS = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(DEFAULTS)


class PortableEvidenceTests(unittest.TestCase):
    """Keep raw record identity independent of its current physical location."""

    def setUp(self):
        """Create separate original and staging directories without any live study data."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.original = self.directory / "original"
        self.staging = self.directory / "staging"
        self.original.mkdir()
        self.staging.mkdir()
        self.manifest_path = self.staging / "manifest.json"
        self.manifest = {"schema_version": 1, "kind": "portable_raw_evidence", "files": [], "git_roots": []}

    def manifest_file(self, manifest=None):
        """Write one explicit test manifest, leaving all evidence bytes unchanged."""
        self.manifest_path.write_text(json.dumps(self.manifest if manifest is None else manifest))
        return self.manifest_path

    def original_file(self, relative, contents):
        """Write an original artifact and return its exact absolute logical path."""
        path = self.original / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
        return path

    def staged_file(self, path, relative=None):
        """Copy exact original bytes under an explicit relative target and SHA256 entry."""
        contents = path.read_bytes()
        relative = relative or "files/" + hashlib.sha256(contents).hexdigest()
        target = self.staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(contents)
        self.manifest["files"].append({"original_path": str(path), "relative_path": relative,
                                       "sha256": hashlib.sha256(contents).hexdigest()})

    def git_fixture(self):
        """Create an independent detached Git checkout containing only the frozen defaults used by validation."""
        source = self.original / "source"
        repository = SCRIPTS.parents[2]
        relatives = ["10_jax_exploration_platform/" + relative for relative in (
            "src/exploration_platform/atari_pixels/config.py", "src/exploration_platform/atari_pixels/environment.py",
            "scripts/atari_pixel_campaign/paired_benchmark.py", "scripts/atari_pixel_campaign/worker_environment.sh")]
        for relative in relatives:
            path = source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((repository / relative).read_bytes())
        subprocess.run(["git", "init", "-q", str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "add", "."], check=True)
        subprocess.run(["git", "-C", str(source), "-c", "user.name=Synthetic evidence test",
            "-c", "user.email=synthetic@example.invalid", "-c", "core.hooksPath=/dev/null",
            "commit", "-qm", "Create synthetic source defaults"], check=True)
        commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        target = self.staging / "repositories/source"
        subprocess.run(["git", "clone", "-q", "--no-hardlinks", str(source), str(target)], check=True)
        subprocess.run(["git", "-C", str(target), "checkout", "-q", "--detach", commit], check=True)
        self.manifest["git_roots"].append({"original_root": str(source), "relative_path": "repositories/source", "commit": commit})
        for relative in relatives:
            self.staged_file(source / relative, "repositories/source/" + relative)
        return source, target, commit

    def legacy_case(self):
        """Build a complete synthetic legacy case whose gate, ledger and script all use original paths."""
        source, target, commit = self.git_fixture()
        defaults = DEFAULTS.default_configuration(source)
        reference = {name: defaults[name] for name in ("learner", "environment")}
        reference["execution"] = {"compilation_mode": "whole_iteration", "iterations_per_call": 1,
            "report_readout": "none", "report_every_iterations": 10}
        candidate = copy.deepcopy(reference)
        candidate["learner"]["rollout_unroll"] = 2
        runtime = {"JAX_ENABLE_X64": "true", "JAX_DEFAULT_MATMUL_PRECISION": "highest",
                   "XLA_FLAGS": "--xla_gpu_exclude_nondeterministic_ops=true"}
        run = self.original / "campaign"
        unit = run / "data/legacy_case/attempts/attempt-0001"
        case = {"source_root": str(source), "source_commit": commit, "campaign_run_dir": str(run),
            "hypothesis": "synthetic legacy mechanism", "unit_id": "legacy_case", "hardware_class": "h100",
            "configuration": reference["learner"], "candidate": {"learner": {"rollout_unroll": 2}, "environment": {}, "execution": {}},
            "reference_execution": None, "rounds": 7, "iterations": 1}
        case["specification_digest"] = validation.canonical_digest(case)
        row = {"iterations": 1, "copies": 1, "environments_per_copy": 128, "rollout_steps": 128,
            "training_transitions": 128 * 128, "seconds": 1., "memory": {"other_gpu_processes": [],
                "gpu_uuid": "GPU-fixture", "sampled_process_peak_bytes": 1024, "allocator": {"peak_bytes_in_use": 1024}}}
        checks = {"passed": True, "complete_iteration": {"leaves": 1}, "supplied_inputs": {
            key: {"leaves": 1} for key in ("preparation", "fixed_batch", "fixed_batch_update")}}
        record = {"source_commit": commit, "hardware_class": "h100", "idea": case["hypothesis"], "copies": 1,
            "unit_id": case["unit_id"], "specification_digest": case["specification_digest"], "case_completed": True,
            "reference": reference, "candidate": candidate, "implementation_distinct": True, "correctness_passed": True,
            "correctness_attempted": True, "correctness": checks, "measurement_valid": True, "contaminated": False,
            "rounds": [{"order": ["reference", "candidate"] if index % 2 == 0 else ["candidate", "reference"],
                        "reference": row, "candidate": row} for index in range(7)]}
        hardware = {"source_root": str(source), "source_commit": commit, "hardware_class": "h100", "pid": 10,
            "process_start_time": "100", "worker_pid": 9, "worker_start_time": "90"}
        worker = source / "10_jax_exploration_platform/scripts/atari_pixel_campaign/worker_environment.sh"
        script = run / "slurm/legacy_worker.sh"
        checks_by_kind = {}
        for kind in ("learner", "renderer"):
            check_path = run / ("legacy_" + kind + ".json")
            command = (["/original/python", "-m", "pytest", "-q", "/original/tests/one", "/original/tests/two"]
                       if kind == "learner" else ["/original/python", "/original/renderer.py", "--candidate", "all"])
            check = {"source_commit": commit, "passed": True, "exit_code": 0, "command": command,
                "hardware": {"hostname": "serval02", "gpu_query": "GPU-fixture, NVIDIA H100 NVL, 1000 MiB, driver",
                             "runtime_environment": runtime}}
            check_path.parent.mkdir(parents=True, exist_ok=True)
            check_path.write_text(json.dumps(check))
            checks_by_kind[kind] = {"path": str(check_path), "sha256": portable.digest(check_path)}
        artifacts = {unit / "case.json": case, unit / "result.json": record, unit / "hardware.json": hardware,
            run / "processes.json": [{"pid": 9, "start_time": "90", "source_commit": commit, "script": str(script)}],
            run / "data/source_checks" / commit / "h100.json": {"passed": True, "source_commit": commit,
                "hardware_class": "h100", "checks": checks_by_kind}}
        for path, value in artifacts.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text("source " + str(worker) + "\n")
        for path in sorted(run.rglob("*")):
            if path.is_file():
                self.staged_file(path)
        return unit / "result.json", case, target, commit

    def test_raw_bytes_and_original_receipts_survive_removal_of_original_tree(self):
        """Historical Git/default/gate/ledger/script reads all succeed only through their exact mapping."""
        path, case, target, commit = self.legacy_case()
        local = validation.Evidence()
        before = validation.validate_cohort_case(path, case["hypothesis"], "h100", 1, commit, local)
        resolver = portable.ExactPathResolver(self.manifest_file())
        staged = validation.Evidence(resolver)
        original_case = resolver.read_bytes(path.parent / "case.json")
        shutil.rmtree(self.original)
        after = validation.validate_cohort_case(path, case["hypothesis"], "h100", 1, commit, staged)
        self.assertEqual(after, before)
        self.assertEqual(staged.records(), local.records())
        self.assertEqual(resolver.read_bytes(path.parent / "case.json"), original_case)
        self.assertEqual(json.loads(original_case)["source_root"], case["source_root"])
        self.assertTrue((target / ".git").is_dir())
        self.assertNotIn("jax", sys.modules)
        self.assertNotIn("numpy", sys.modules)

    def test_no_fallback_to_existing_original_or_similar_path_prefix(self):
        """An original file or adjacent root is never an implicit fallback for a missing exact entry."""
        path = self.original_file("record.json", b"{}")
        resolver = portable.ExactPathResolver(self.manifest_file())
        with self.assertRaisesRegex(ValueError, "no exact file entry.*record.json"):
            validation.Evidence(resolver).read(path)
        with self.assertRaisesRegex(ValueError, "no exact Git-root entry"):
            resolver.git_root(str(self.original) + "_adjacent", "a" * 40)

    def test_duplicate_original_paths_and_json_keys_are_rejected(self):
        """Ambiguous entries cannot replace an earlier logical identity."""
        path = self.original_file("record.json", b"{}")
        self.staged_file(path)
        self.manifest["files"].append(dict(self.manifest["files"][0]))
        with self.assertRaisesRegex(ValueError, "duplicated"):
            portable.ExactPathResolver(self.manifest_file())
        self.manifest_path.write_text('{"schema_version":1,"kind":"portable_raw_evidence","files":[],"files":[],"git_roots":[]}')
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            portable.ExactPathResolver(self.manifest_path)

    def test_relative_targets_and_original_paths_cannot_escape_or_normalize(self):
        """Reject traversal, absolute targets, NUL and noncanonical original aliases."""
        path = self.original_file("record.json", b"{}")
        self.staged_file(path)
        original = copy.deepcopy(self.manifest)
        for bad in ("../outside", "/outside", "files/../outside", "files/./record", "files/\0record"):
            manifest = copy.deepcopy(original)
            manifest["files"][0]["relative_path"] = bad
            with self.subTest(target=bad), self.assertRaises(ValueError):
                portable.ExactPathResolver(self.manifest_file(manifest))
        for bad in ("relative.json", "/original/../record.json", "/original//record.json"):
            manifest = copy.deepcopy(original)
            manifest["files"][0]["original_path"] = bad
            with self.subTest(original=bad), self.assertRaises(ValueError):
                portable.ExactPathResolver(self.manifest_file(manifest))

    def test_symlink_escape_and_changed_bytes_fail_the_pinned_hash(self):
        """Identical basenames and harmless-looking JSON whitespace cannot bypass exact byte checks."""
        path = self.original_file("record.json", b"{\"value\":1}")
        self.staged_file(path, "files/record.json")
        resolver = portable.ExactPathResolver(self.manifest_file())
        target = self.staging / "files/record.json"
        target.write_bytes(b"{\"value\":1}\n")
        with self.assertRaisesRegex(ValueError, "pinned SHA256"):
            resolver.read_bytes(path)
        target.unlink()
        target.symlink_to(path)
        with self.assertRaisesRegex(ValueError, "outside"):
            portable.ExactPathResolver(self.manifest_path)

    def test_git_mapping_rejects_branch_wrong_commit_dirty_tree_and_ancestor(self):
        """A mapped archive under a repository cannot borrow that ancestor's Git identity."""
        source, target, commit = self.git_fixture()
        resolver = portable.ExactPathResolver(self.manifest_file())
        subprocess.run(["git", "-C", str(target), "checkout", "-q", "-b", "synthetic_branch"], check=True)
        with self.assertRaisesRegex(ValueError, "clean detached"):
            resolver.git_root(source, commit)
        subprocess.run(["git", "-C", str(target), "checkout", "-q", "--detach", commit], check=True)
        with self.assertRaisesRegex(ValueError, "requested original commit"):
            resolver.git_root(source, "a" * 40)
        (target / "untracked.txt").write_text("not part of the frozen source")
        with self.assertRaisesRegex(ValueError, "clean detached"):
            resolver.git_root(source, commit)
        (target / "untracked.txt").unlink()
        (target / "archive").mkdir()
        manifest = copy.deepcopy(self.manifest)
        manifest["files"] = []
        manifest["git_roots"][0]["relative_path"] += "/archive"
        with self.assertRaisesRegex(ValueError, "clean detached"):
            portable.ExactPathResolver(self.manifest_file(manifest))

    def test_missing_legacy_ledger_or_changed_script_cannot_be_skipped(self):
        """Indirect runtime evidence is mapped and hashed as strictly as the result itself."""
        path, case, target, commit = self.legacy_case()
        manifest = copy.deepcopy(self.manifest)
        manifest["files"] = [entry for entry in manifest["files"] if not entry["original_path"].endswith("/processes.json")]
        resolver = portable.ExactPathResolver(self.manifest_file(manifest))
        with self.assertRaisesRegex(ValueError, "no exact file entry.*processes.json"):
            validation.validate_cohort_case(path, case["hypothesis"], "h100", 1, commit, validation.Evidence(resolver))
        resolver = portable.ExactPathResolver(self.manifest_file())
        script = next(entry for entry in self.manifest["files"] if entry["original_path"].endswith("legacy_worker.sh"))
        (self.staging / script["relative_path"]).write_text("changed launch script")
        with self.assertRaisesRegex(ValueError, "pinned SHA256.*legacy_worker.sh"):
            validation.validate_cohort_case(path, case["hypothesis"], "h100", 1, commit, validation.Evidence(resolver))

    def test_manifest_hash_and_later_manifest_changes_are_rejected(self):
        """The mapping cannot change after its bytes have been pinned by the caller or resolver."""
        path = self.original_file("record.json", b"{}")
        self.staged_file(path)
        manifest = self.manifest_file()
        with self.assertRaisesRegex(ValueError, "required SHA256"):
            portable.ExactPathResolver(manifest, "a" * 64)
        resolver = portable.ExactPathResolver(manifest, portable.digest(manifest))
        self.assertEqual(resolver.receipt()["mapped_files"], 1)
        manifest.write_text(manifest.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            resolver.read_bytes(path)

    def test_full_final_admission_and_completion_rebuild_with_original_paths_unavailable(self):
        """Nested final-verification rebuilding shares the resolver and preserves every original receipt."""
        source, target, commit = self.git_fixture()
        specification = importlib.util.spec_from_file_location("portable_final_fixture", SCRIPTS / "tests/test_post_comparison_validation.py")
        fixtures = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(fixtures)
        fixture = fixtures.PostComparisonTests("test_all_five_gates_are_derived_from_complete_evidence")
        self.addCleanup(fixture.doCleanups)
        with patch.object(fixtures, "SCRIPTS", source / "10_jax_exploration_platform/scripts/atari_pixel_campaign"):
            fixture.setUp()
        original_spec = fixture.write("final_specification.json", fixture.specification)
        saved = validation.validate_final_verification(fixture.specification, specification_path=original_spec)
        final_path = fixture.write("final_verification.json", saved)
        training_path = fixture.write("training.json", {"schema_version": 1, "source_root": str(source),
            "source_commit": commit, "hardware_class": "h100", "configuration": fixture.configuration,
            "runtime_environment": fixture.runtime, "copy_seeds": [1], "final_verification": str(final_path)})
        admitted = validation.validate_training_admission(training_path)
        target_updates = 2_000_000_000 // (128 * 128)
        identity = {name: admitted[name] for name in ("source_root", "source_commit", "hardware_class",
                    "configuration", "runtime_environment", "copy_seeds")}
        identity.update(backend={"device_kind": "H100"}, helpers={})
        fixture.write("training_identity.json", identity)
        checkpoint = fixture.directory / "checkpoints/latest.pkl"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"synthetic full-state checkpoint, never unpickled")
        fixture.write("checkpoints/latest.metadata.json", {"kind": "optimized_training_checkpoint",
            "identity_digest": validation.canonical_digest(identity), "configuration_digest": validation.canonical_digest(fixture.configuration),
            "source_commit": commit, "iteration": target_updates, "target_updates": target_updates,
            "steps_per_copy": target_updates * 128 * 128, "copy_seeds": [1], "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": portable.digest(checkpoint)})
        fixture.write("result.json", {"completed": True, "update": target_updates, "target_updates": target_updates,
            "requested_steps_per_copy": 2_000_000_000, "steps_per_copy": target_updates * 128 * 128,
            "copy_seeds": [1], "copies": 1, "identity_digest": validation.canonical_digest(identity)})
        completed = validation.validate_training_completion(training_path, fixture.directory)
        for path in sorted(fixture.directory.rglob("*")):
            if path.is_file():
                self.staged_file(path)
        resolver = portable.ExactPathResolver(self.manifest_file())
        shutil.rmtree(self.original)
        shutil.rmtree(fixture.directory)
        rebuilt = validation.validate_final_verification(fixture.specification, validation.Evidence(resolver), original_spec)
        self.assertEqual(rebuilt, saved)
        self.assertEqual(validation.validate_training_admission(training_path, resolver=resolver), admitted)
        self.assertEqual(validation.validate_training_completion(training_path, fixture.directory, resolver=resolver), completed)
        # Before: a correctly pinned source file; after: another pinned file with the same fields but different Git bytes.
        entry = next(item for item in self.manifest["files"] if item["original_path"] == str(
            source / "10_jax_exploration_platform/src/exploration_platform/atari_pixels/config.py"))
        different = (self.staging / entry["relative_path"]).read_bytes() + b"\n"
        (self.staging / "files/different_config.py").write_bytes(different)
        entry.update(relative_path="files/different_config.py", sha256=hashlib.sha256(different).hexdigest())
        changed = portable.ExactPathResolver(self.manifest_file())
        with self.assertRaisesRegex(ValueError, "pinned Git blob"):
            validation.validate_training_admission(training_path, resolver=changed)
        self.assertNotIn("jax", sys.modules)
        self.assertNotIn("numpy", sys.modules)


if __name__ == "__main__":
    unittest.main()
