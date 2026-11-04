"""Metadata-only controls for future actual H correctness review; no actual result or approval."""

import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("raw_review_under_test", HERE / "review_actual_common_correctness.py")
review = importlib.util.module_from_spec(spec); spec.loader.exec_module(review)


class NamespaceTests(unittest.TestCase):
    """Bind synthetic source-directory declarations without issuing any Git command."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="correctness-audit-private-metadata-")
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        self.expected = {"/synthetic/original/" + str(index): format(index + 1, "040x") for index in range(8)}
        self.guard = types.SimpleNamespace(pinned_history_declarations=lambda plan: (dict(self.expected), set(self.expected)))
        self.plans = [self.make_plan("correctness"), self.make_plan("qualification")]

    def put(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(review.encoded(value)); return review.reference(path)

    def make_plan(self, name):
        unit = self.run / "data" / name
        history = unit / "direct_history"; history.mkdir(parents=True)
        root = self.run / ("source-" + name); root.mkdir()
        mappings = []
        for index, (original, commit) in enumerate(self.expected.items()):
            relative = "git_sources/source_" + str(index)
            (history / relative).mkdir(parents=True)
            mappings.append({"original_root": original, "commit": commit, "relative_path": relative})
        manifest = {"schema_version": 1, "kind": "portable_raw_evidence", "files": [], "git_roots": mappings}
        manifest_ref = self.put(history / "evidence_manifest.json", manifest)
        frozen = self.put(history / "inputs.json", {"evidence_manifest": manifest_ref})
        record = {"source_root": str(root), "source_commit": "a" * 40}
        return {"campaign_run_dir": str(self.run), "unit_id": name, "core": record,
            "paired_source": record, "helpers": record, "direct_binding": {"frozen_inputs": frozen}}

    def mutate_manifest(self, mutate):
        plan = self.plans[0]
        unit = self.run / "data" / plan["unit_id"]
        history = unit / "direct_history"
        value = review.read(history / "evidence_manifest.json"); mutate(value)
        record = self.put(history / "evidence_manifest.json", value)
        plan["direct_binding"]["frozen_inputs"] = self.put(history / "inputs.json", {"evidence_manifest": record})

    def roots(self):
        with mock.patch.object(review, "NUMERICAL", set()):
            return review.declared_roots(*self.plans, self.guard)

    def test_both_eight_root_namespaces_are_present(self):
        roots = self.roots()
        for plan in self.plans:
            for index in range(8):
                self.assertIn(str(self.run / "data" / plan["unit_id"] / "direct_history/git_sources" / ("source_" + str(index))), roots)

    def test_unlisted_original_name_refuses(self):
        self.mutate_manifest(lambda x: x["git_roots"][0].update(original_root="/not/declared"))
        with self.assertRaises(ValueError): self.roots()

    def test_changed_original_commit_refuses(self):
        self.mutate_manifest(lambda x: x["git_roots"][0].update(commit="f" * 40))
        with self.assertRaises(ValueError): self.roots()

    def test_missing_eighth_root_refuses(self):
        self.mutate_manifest(lambda x: x["git_roots"].pop())
        with self.assertRaises(ValueError): self.roots()

    def test_parent_directory_target_refuses(self):
        self.mutate_manifest(lambda x: x["git_roots"][0].update(relative_path="../outside"))
        with self.assertRaises(ValueError): self.roots()

    def test_duplicate_original_name_refuses(self):
        self.mutate_manifest(lambda x: x["git_roots"][1].update(original_root=x["git_roots"][0]["original_root"]))
        with self.assertRaises(ValueError): self.roots()

    def test_duplicate_target_refuses(self):
        self.mutate_manifest(lambda x: x["git_roots"][1].update(relative_path=x["git_roots"][0]["relative_path"]))
        with self.assertRaises(ValueError): self.roots()

    def test_symlinked_target_refuses(self):
        history = self.run / "data/correctness/direct_history"
        target = history / "git_sources/source_0"; target.rmdir()
        target.symlink_to(history / "git_sources/source_1", target_is_directory=True)
        with self.assertRaises(ValueError): self.roots()


class TerminalTests(unittest.TestCase):
    """Use private saved-record fixtures; never query a process or execute scientific consumers."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="correctness-terminal-private-metadata-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.directory = self.root / "data"; self.directory.mkdir()
        self.packet = self.root / "audit"; self.packet.mkdir()
        patch = mock.patch.object(review, "P", self.packet); patch.start(); self.addCleanup(patch.stop)
        self.metadata = {"after": {"identity": {"hostname": "serval02", "uid": 2618919,
            "pid": 101, "process_start_time": "202"}}}
        self.spawn = {"controller": {"pid": 303, "process_start_time": "404"}}
        self.wait = {"finished_at_utc": "2026-11-02T00:00:00+00:00"}
        self.observation = {"kind": "owned_prior_direct_process_exit_observation", "hostname": "serval02",
            "observer_uid": 2618919, "observer_pid": 505, "all_original_processes_exited": True,
            "gpu_or_standing_lock_operations": 0, "at_utc": "2026-11-02T00:00:01+00:00",
            "observations": [{"role": "controller", "expected_pid": 303, "expected_birth": "404",
                "observation": "absent", "original_process_exited": True},
                {"role": "numerical_child", "expected_pid": 101, "expected_birth": "202",
                    "observation": "absent", "original_process_exited": True}]}
        self.plan = {"private_synthetic_fixture": True}
        self.put(self.directory / "measurement_plan.json", self.plan)
        self.put(self.directory / "result.json", {"private_synthetic_fixture": True})
        self.put(self.directory / "direct_controller_outcome.json", {})
        self.put(self.packet / "source_freeze.json", {"private_synthetic_fixture": True})
        self.approval = {"kind": "independent_h100_common_correctness_raw_review_source_review",
            "approved": True, "passed": True, "independent_of_author": True,
            "source_freeze": review.reference(self.packet / "source_freeze.json")}
        self.save()
        self.guard = types.SimpleNamespace(controller_evidence=lambda *a: self.terminal_refs,
            read=lambda path: (review.read(path), b""))
        patch = mock.patch.object(review, "load", return_value=types.SimpleNamespace())
        patch.start(); self.addCleanup(patch.stop)

    def put(self, path, value):
        path.write_bytes(review.encoded(value)); return review.reference(path)

    def save(self):
        self.put(self.directory / "direct_process_metadata.json", self.metadata)
        self.put(self.directory / "direct_controller_spawn.json", self.spawn)
        wait_ref = self.put(self.directory / "wait.json", self.wait)
        terminal_ref = self.put(self.directory / "direct_terminal_process.json", {"return_record": wait_ref})
        self.terminal_refs = {"wait_record": wait_ref, "terminal_process": terminal_ref}
        observation_ref = self.put(self.root / "observation.json", self.observation)
        approval_ref = self.put(self.root / "approval.json", self.approval)
        self.ready = {"schema_version": 1, "kind": "actual_owned_direct_correctness_terminal_ready",
            "root_confirmed_result_normal_wait_controller_outcome_and_exits": True,
            "plan": review.reference(self.directory / "measurement_plan.json"),
            "result": review.reference(self.directory / "result.json"),
            "terminal": terminal_ref, "outcome": review.reference(self.directory / "direct_controller_outcome.json"),
            "exit_observation": observation_ref, "independent_source_review": approval_ref}
        self.ready_path = self.root / "ready.json"; self.put(self.ready_path, self.ready)

    def check(self):
        return review.terminal_ready(self.ready_path, self.directory, self.plan, self.guard)

    def test_exact_two_exits_and_reviewed_source_pass_without_scientific_work(self):
        self.assertEqual(self.check()[0], self.ready)

    def test_wrong_numerical_birth_refuses(self):
        self.observation["observations"][1]["expected_birth"] = "999"; self.save()
        with self.assertRaises(ValueError): self.check()

    def test_duplicate_role_refuses(self):
        self.observation["observations"][1] = dict(self.observation["observations"][0]); self.save()
        with self.assertRaises(ValueError): self.check()

    def test_running_original_process_refuses(self):
        self.observation["observations"][0]["observation"] = "present_same_birth"; self.save()
        with self.assertRaises(ValueError): self.check()

    def test_foreign_observer_uid_refuses(self):
        self.observation["observer_uid"] = 1; self.save()
        with self.assertRaises(ValueError): self.check()

    def test_boolean_gpu_operation_count_refuses(self):
        self.observation["gpu_or_standing_lock_operations"] = False; self.save()
        with self.assertRaises(ValueError): self.check()

    def test_exit_before_wait_refuses(self):
        self.observation["at_utc"] = "2026-11-01T23:59:59+00:00"; self.save()
        with self.assertRaises(ValueError): self.check()

    def test_source_not_independently_approved_refuses(self):
        self.approval["approved"] = False; self.save()
        with self.assertRaises(ValueError): self.check()

    def test_changed_source_freeze_refuses(self):
        self.approval["source_freeze"]["sha256"] = "0" * 64; self.save()
        with self.assertRaises(ValueError): self.check()

    def test_root_readiness_missing_refuses(self):
        self.ready["root_confirmed_result_normal_wait_controller_outcome_and_exits"] = False
        self.put(self.ready_path, self.ready)
        with self.assertRaises(ValueError): self.check()

    def test_mismatched_result_receipt_refuses(self):
        self.ready["result"]["sha256"] = "0" * 64; self.put(self.ready_path, self.ready)
        with self.assertRaises(ValueError): self.check()


class SourceOnlyTests(unittest.TestCase):
    """Check actual prepared small metadata without reading either actual correctness result."""

    def test_prepared_inputs_and_both_namespaces_need_no_correctness_result(self):
        real = review.reference
        def forbidden(path):
            if str(path).endswith("/result.json"): raise AssertionError("no actual correctness result read")
            return real(path)
        with mock.patch.object(review, "reference", side_effect=forbidden):
            for copies in (1, 2):
                unused, plan, qualified, guard = review.prepared(copies)
                self.assertGreaterEqual(len(review.declared_roots(plan, qualified, guard)), 16)

    def test_actual_review_context_refuses_a_login_or_other_job(self):
        with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "not-the-owned-job"}):
            with self.assertRaises(ValueError): review.owned_cpu({})

    def test_source_guard_retains_exact_published_raw_and_review_fields(self):
        unused, plan, qualified, guard = review.prepared(1)
        self.assertIn("backend_capture_begin", guard.RAW_FIELDS)
        self.assertIn("post_installed_package_sha256", guard.RAW_FIELDS)
        self.assertIn("qualification_source_contract", guard.RAW_FIELDS)
        self.assertIn("raw_audit", guard.REVIEW_FIELDS)


class RawFieldContractTests(unittest.TestCase):
    """Test only proof-field construction with explicit private mocked scientific validators."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="correctness-raw-fields-private-")
        self.addCleanup(self.temp.cleanup); self.directory = Path(self.temp.name)
        unused, original, qualified, actual_guard = review.prepared(1)
        self.plan = copy.deepcopy(original); self.events = []
        self.result = {"measurement_mode": "correctness", "measurement_valid": False,
            "correctness_attempted": True, "correctness_passed": True, "implementation_distinct": True,
            "execution_identity": {"hostname": "synthetic-metadata-host", "pid": 101,
                "process_start_time": "202", "finished_at_seconds": 1.0},
            "loaded_gpu_libraries": {"pid": 101, "loaded_shared_libraries": ["/synthetic/libcuda.so"],
                "verified_package_libraries": {}}, "direct_whole_work_memory": {"private_synthetic_fixture": True}}
        self.binding = {"gpu_uuid": "GPU-synthetic-private-fixture", "native": {"driver_library": "/synthetic/libcuda.so"}}
        self.metadata = {"after": {"identity": {"hostname": "synthetic-metadata-host", "pid": 101, "process_start_time": "202"}}}
        self.assets = {"private_synthetic_fixture": True}
        for name, value in [("result.json", self.result), ("measurement_plan.json", self.plan),
                ("direct_process_metadata.json", self.metadata), ("direct_device_binding.json", self.binding),
                ("direct_terminal_process.json", {}), ("asset_receipt.json", self.assets),
                ("installed_backend.json", {"gpu_libraries": {}}),
                ("direct_backend_capture.json", {"post_installed_package": {"private_synthetic_fixture": True}}),
                ("direct_backend_capture_begin.json", {}), ("source_contract.json", {})]:
            (self.directory / name).write_bytes(review.encoded(value))
        qreview = self.directory / "synthetic_qualification_review.json"
        qreview.write_bytes(review.encoded({"source_contract": review.reference(self.directory / "source_contract.json")}))
        self.plan["direct_qualification_inputs"]["review"] = review.reference(qreview)
        self.guard = types.SimpleNamespace(ZERO_FIELDS=actual_guard.ZERO_FIELDS, RAW_FIELDS=actual_guard.RAW_FIELDS,
            expected_review_identity=actual_guard.expected_review_identity,
            controller_evidence=lambda *a: {name: review.reference(self.directory / "direct_terminal_process.json")
                for name in ("terminal_process", "wait_record", "controller_spawn", "controller_outcome", "controller_output_hashes")})
        validators = types.SimpleNamespace(correctness=lambda x: self.events.append("mocked-correctness"),
            configuration=lambda x: None, execution_identity=lambda x: None)
        training = types.SimpleNamespace(validate_gpu_library_receipt=lambda *a: None, verify_assets=lambda *a: self.assets)
        def saved(*args):
            self.events.append("mocked-complete-saved-consumer"); return self.plan, self.binding
        self.support = types.SimpleNamespace(verify_saved_measurement=saved,
            adjacent=lambda name: validators if name == "post_comparison_validation" else training)

    def test_exact_raw_field_contract_follows_complete_consumer(self):
        result, identity, raw = review.actual_raw(self.directory, self.plan, {}, self.guard, self.support)
        self.assertEqual(set(raw), self.guard.RAW_FIELDS)
        self.assertEqual(self.events[:2], ["mocked-complete-saved-consumer", "mocked-correctness"])
        self.assertTrue(raw["unchanged_staged_consumer_passed"])
        self.assertFalse((self.directory / "independent_actual_review.json").exists())

    def test_failed_complete_consumer_cannot_construct_a_raw_pass(self):
        self.support.verify_saved_measurement = mock.Mock(side_effect=ValueError("private fixture consumer failure"))
        with self.assertRaises(ValueError):
            review.actual_raw(self.directory, self.plan, {}, self.guard, self.support)
        self.assertFalse((self.directory / "raw_validation.json").exists())
        self.assertFalse((self.directory / "independent_actual_review.json").exists())


if __name__ == "__main__":
    unittest.main()
