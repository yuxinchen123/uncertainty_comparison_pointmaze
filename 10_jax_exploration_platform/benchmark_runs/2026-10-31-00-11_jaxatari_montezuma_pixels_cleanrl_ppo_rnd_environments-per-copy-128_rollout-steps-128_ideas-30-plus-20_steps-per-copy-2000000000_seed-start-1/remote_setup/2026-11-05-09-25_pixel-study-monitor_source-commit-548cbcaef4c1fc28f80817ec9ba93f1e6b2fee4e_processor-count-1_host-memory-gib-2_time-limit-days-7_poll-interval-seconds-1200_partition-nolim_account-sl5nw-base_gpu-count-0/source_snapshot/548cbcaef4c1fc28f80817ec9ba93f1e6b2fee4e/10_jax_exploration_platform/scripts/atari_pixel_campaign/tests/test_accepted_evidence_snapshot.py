"""Check bounded file hashing and strict accepted-history metadata refusal with small fixtures."""
import ast
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
SPECIFICATION = importlib.util.spec_from_file_location("accepted_snapshot_under_test", SCRIPTS / "accepted_evidence_snapshot.py")
snapshot = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(snapshot)


class PhysicalFileTests(unittest.TestCase):
    """Use real small files so descriptor replacement and duplicate-content boundaries are exercised."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, name, raw=b"the same captured content"):
        path = self.root / name
        path.write_bytes(raw)
        return path, snapshot.digest(raw)

    def test_different_files_with_same_content_are_both_hashed(self):
        first, expected = self.write("first")
        second, unused = self.write("second")
        with snapshot.ValidationFiles() as files:
            files.file(first, expected); files.file(second, expected)
            proof = files.finish([str(first), str(second)])
        self.assertEqual(proof["distinct_physical_files"], 2)
        self.assertEqual(proof["bytes_hashed"], 2 * first.stat().st_size)

    def test_hard_links_share_one_hash_but_keep_both_paths(self):
        first, expected = self.write("first")
        second = self.root / "second"
        os.link(first, second)
        with snapshot.ValidationFiles() as files:
            files.file(first, expected); files.file(second, expected)
            proof = files.finish([str(first), str(second)])
        self.assertEqual(proof["distinct_paths"], 2)
        self.assertEqual(proof["distinct_physical_files"], 1)
        self.assertEqual(proof["bytes_hashed"], first.stat().st_size)

    def test_reuse_checks_identity_without_reading_file_bytes_again(self):
        path, expected = self.write("first")
        with snapshot.ValidationFiles() as files:
            files.file(path, expected)
            with patch.object(snapshot.os, "read", side_effect=AssertionError("a repeated byte read")):
                files.file(path, expected); files.finish([str(path)])

    def test_replaced_path_with_identical_content_is_refused(self):
        path, expected = self.write("first")
        with snapshot.ValidationFiles() as files:
            files.file(path, expected)
            replacement, unused = self.write("replacement")
            os.replace(replacement, path)
            with self.assertRaises(ValueError):
                files.finish([str(path)])

    def test_same_inode_changed_content_is_refused(self):
        path, expected = self.write("first")
        with snapshot.ValidationFiles() as files:
            files.file(path, expected); path.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                files.file(path, expected)

    def test_conflicting_expected_hash_is_refused(self):
        path, expected = self.write("first")
        with snapshot.ValidationFiles() as files:
            files.file(path, expected)
            with self.assertRaises(ValueError):
                files.file(path, "0" * 64)
            with self.assertRaises(ValueError):
                files.finish([str(path)])

    def test_partial_coverage_is_not_a_complete_proof(self):
        path, expected = self.write("first")
        with snapshot.ValidationFiles() as files:
            files.file(path, expected)
            with self.assertRaises(ValueError):
                files.finish([str(path), str(self.root / "missing")])
            with self.assertRaises(ValueError):
                files.finish([str(path)])

    def test_deadline_never_resets_for_raw_payload_cache(self):
        now = [0.0]
        path, expected = self.write("first")
        with snapshot.ValidationFiles(2, clock=lambda: now[0]) as files:
            files.file(path, expected); files.finish([str(path)])
            with self.assertRaises(ValueError):
                files.begin_metadata_reuse()
            now[0] = 2.0
            with self.assertRaises(ValueError):
                files.finish([str(path)])

    def test_small_complete_metadata_can_reuse_bounded_identity_checks(self):
        now = [0.0]
        path, expected = self.write("metadata", b"{}\n")
        with snapshot.ValidationFiles(2, clock=lambda: now[0], metadata_only=True) as files:
            files.file(path, expected, keep_bytes=True); files.finish([str(path)])
            now[0] = 100.0
            files.begin_metadata_reuse(); files.finish([str(path)])
            self.assertEqual(files.bytes_hashed, 3)

    def test_more_than_five_minutes_is_refused(self):
        with self.assertRaises(ValueError):
            snapshot.ValidationFiles(301)

    def test_descriptor_limit_refuses_incomplete_snapshot(self):
        first, expected = self.write("first")
        second, unused = self.write("second")
        with snapshot.ValidationFiles(maximum_open_files=1) as files:
            files.file(first, expected)
            with self.assertRaises(ValueError):
                files.file(second, expected)

    def test_symbolic_link_is_refused(self):
        path, expected = self.write("first")
        indirect = self.root / "indirect"
        indirect.symlink_to(path)
        with snapshot.ValidationFiles() as files:
            with self.assertRaises(ValueError):
                files.file(indirect, expected)

    def test_duplicate_json_key_and_nonfinite_metadata_are_refused(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                snapshot.document(raw)


class DeclaredCoverageTests(unittest.TestCase):
    """Separate historical logical names from the actual captured physical file universe."""

    setUp = PhysicalFileTests.setUp
    tearDown = PhysicalFileTests.tearDown
    write = PhysicalFileTests.write

    def test_same_original_name_different_hashes_resolve_to_different_files(self):
        first, one = self.write("first", b"old backend")
        second, two = self.write("second", b"current backend")
        with snapshot.ValidationFiles() as files:
            inventory = snapshot.AcceptedInventory(files)
            inventory.logical_mappings = [{"original_path": "/logical/backend.so", "sha256": one, "captured_path": str(first)},
                {"original_path": "/logical/backend.so", "sha256": two, "captured_path": str(second)}]
            inventory.add({"path": "/logical/backend.so", "sha256": one}, metadata=False)
            inventory.add({"path": "/logical/backend.so", "sha256": two}, metadata=False)
            inventory.resolve_blobs()
            self.assertEqual(inventory.declared, {str(first): one, str(second): two})

    def test_every_matching_capture_is_hashed_even_with_same_content(self):
        first, expected = self.write("first")
        second, unused = self.write("second")
        with snapshot.ValidationFiles() as files:
            inventory = snapshot.AcceptedInventory(files)
            inventory.logical_mappings = [{"original_path": "/logical/file", "sha256": expected, "captured_path": str(path)}
                for path in (first, second)]
            inventory.add({"path": "/logical/file", "sha256": expected}, metadata=False)
            proof = inventory.hash_complete()
            self.assertEqual(proof["distinct_physical_files"], 2)

    def test_historical_process_path_needs_inline_captured_bytes(self):
        with snapshot.ValidationFiles() as files:
            inventory = snapshot.AcceptedInventory(files)
            with self.assertRaises(ValueError):
                inventory.references({"path": "/proc/123/maps", "sha256": "0" * 64})
            inventory.references({"path": "/proc/123/maps", "raw": "old bytes", "sha256": snapshot.digest(b"old bytes")})
            self.assertEqual(inventory.pending_blobs, set())
            self.assertEqual(len(inventory.inline_historical_records), 1)

    def test_actual_driver_capture_resolves_logical_node_local_driver(self):
        captured, expected = self.write("captured_driver.so", b"accepted mapped driver bytes")
        with snapshot.ValidationFiles() as files:
            inventory = snapshot.AcceptedInventory(files)
            inventory.references({"schema_version": 1, "kind": "captured_mapped_cuda_driver_image",
                "original": {"path": "/unavailable/node/local/driver.so", "sha256": expected},
                "captured": {"path": str(captured), "sha256": expected}})
            proof = inventory.hash_complete()
            self.assertEqual(proof["distinct_physical_files"], 1)
            self.assertEqual(set(inventory.declared), {str(captured)})


class CurrentPlanTests(unittest.TestCase):
    """Keep new use narrower than ordinary common confirmation and preserve every old reader statement."""

    def plan(self):
        reference = {"learner": {"n_copies": 1}, "environment": {"cache_room_backgrounds": False, "render_final_two_only": False}, "execution": {}}
        candidate = deepcopy(reference)
        candidate["environment"] = {"cache_room_backgrounds": True, "render_final_two_only": True}
        return {"kind": "selected_configuration_measurement", "mode": "confirmation", "hardware_class": "h100",
            "gpu_uuid": snapshot.CARD_ONE, "source_commit": snapshot.NUMERICAL_SOURCE, "rounds": 7, "iterations": 10,
            "minimum_updates_per_round": 10, "direct_binding": {}, "configuration": candidate,
            "reference_configuration": reference, "copy_seeds": [1], "accepted_evidence_snapshot_protocol": dict(snapshot.PROTOCOL),
            "helpers": {"files": {snapshot.PREFIX + name: snapshot.digest((SCRIPTS / name).read_bytes())
                for name in snapshot.required_helper_names()}}, "accepted_evidence_snapshot_inputs": {
                    key: {"path": "/absolute/" + key + ".json", "sha256": "0" * 64} for key in ("proof", "review")}}

    def test_exact_source_union_contains_twenty_nine_helpers(self):
        names = snapshot.required_helper_names()
        self.assertEqual(len(names), 29)
        self.assertIn("direct_backend_image.py", names)
        self.assertIn("direct_historical_consumer.py", names)
        self.assertIn(snapshot.NAME, names)
        snapshot.require_plan(self.plan())

    def test_changed_protocol_and_extra_capabilities_are_refused(self):
        for field in ("native_process_control", "final_protocol", "device_memory_limit_protocol", "scaling_protocol", "direct_qualification"):
            plan = self.plan(); plan[field] = {}
            with self.subTest(field=field), self.assertRaises(ValueError):
                snapshot.require_plan(plan)
        plan = self.plan(); plan["accepted_evidence_snapshot_protocol"]["operational_admission"] = True
        with self.assertRaises(ValueError):
            snapshot.require_plan(plan)

    def test_same_count_substituted_helper_is_refused(self):
        plan = self.plan()
        plan["helpers"]["files"][snapshot.PREFIX + "unreviewed.py"] = plan["helpers"]["files"].pop(snapshot.PREFIX + "ideas.py")
        with self.assertRaises(ValueError):
            snapshot.require_plan(plan)

    def test_extra_configuration_change_is_refused(self):
        plan = self.plan(); plan["configuration"]["learner"]["n_copies"] = 2
        with self.assertRaises(ValueError):
            snapshot.require_plan(plan)

    def test_boolean_round_count_is_refused(self):
        plan = self.plan(); plan["rounds"] = True
        with self.assertRaises(ValueError):
            snapshot.require_plan(plan)

    def test_current_qualification_keeps_original_fifty_five_statements(self):
        tree = ast.parse((SCRIPTS / "direct_measurement.py").read_bytes())
        current = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "verify_qualification_inputs")
        self.assertEqual(len(current.body), 57)
        self.assertEqual(ast.unparse(current.body[1]), 'if \'accepted_evidence_snapshot_protocol\' in plan:\n    return adjacent(\'accepted_evidence_snapshot\').qualification(plan)')
        self.assertEqual(ast.unparse(current.body[2]), 'if \'common_correctness_inputs\' in plan:\n    return historical_support(plan).request(plan, \'qualification\')')
        self.assertEqual(len(current.body[:1] + current.body[3:]), 55)

    def test_history_dispatch_does_not_replace_old_body(self):
        tree = ast.parse((SCRIPTS / "serval02_card1_binding.py").read_bytes())
        current = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "validate_history")
        self.assertIsInstance(current.body[1], ast.If)
        self.assertEqual(ast.unparse(current.body[1].test), "'accepted_evidence_snapshot_protocol' in plan")
        self.assertIn("comparisons", ast.unparse(current.body[-1]))

    def test_complete_original_qualification_and_history_bodies_are_retained(self):
        repository = SCRIPTS.parents[2]
        prefix = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"
        def functions(raw):
            return {node.name: node for node in ast.parse(raw).body if isinstance(node, ast.FunctionDef)}
        def syntax(nodes):
            return [ast.dump(node, include_attributes=False) for node in nodes]
        original = functions(subprocess.check_output(["git", "-C", str(repository), "show",
            snapshot.ORIGINAL_HELPER_SOURCE + ":" + prefix + "direct_measurement.py"]))["verify_qualification_inputs"]
        current = functions((SCRIPTS / "direct_measurement.py").read_bytes())["verify_qualification_inputs"]
        self.assertEqual(len(original.body), 55)
        self.assertEqual(syntax(original.body), syntax(current.body[:1] + current.body[3:]))
        before = functions(subprocess.check_output(["git", "-C", str(repository), "show",
            "7fef164909b5eb8b31636da7ae0c653282388203:" + prefix + "serval02_card1_binding.py"]))
        after = functions((SCRIPTS / "serval02_card1_binding.py").read_bytes())
        self.assertEqual(set(before), set(after))
        for name in before:
            if name == "validate_history":
                self.assertEqual(syntax(before[name].body), syntax(after[name].body[:1] + after[name].body[2:]))
            else:
                self.assertEqual(ast.dump(before[name], include_attributes=False), ast.dump(after[name], include_attributes=False), name)

    def test_numerical_compute_function_is_byte_syntax_unchanged(self):
        relative = "10_jax_exploration_platform/scripts/atari_pixel_campaign/measure_selected.py"
        before = ast.parse(subprocess.check_output(["git", "-C", str(SCRIPTS.parents[2]), "show",
            "7fef164909b5eb8b31636da7ae0c653282388203:" + relative]))
        after = ast.parse((SCRIPTS / "measure_selected.py").read_bytes())
        selected = lambda tree: next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_compute")
        self.assertEqual(ast.dump(selected(before), include_attributes=False), ast.dump(selected(after), include_attributes=False))


class ReviewedMetadataTests(unittest.TestCase):
    """Use explicitly synthetic small reviewed proofs to exercise the complete consumer without a GPU."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.paths = {}
        self.plan = CurrentPlanTests().plan()
        self.plan.update({"campaign_run_dir": str(self.root), "unit_id": "new_common", "source_root": "/original/source",
            "core": {"sha256": "1" * 64}, "paired_source": {"sha256": "2" * 64}, "python": "/original/bin/python",
            "runtime_environment": {"matrix_precision": "highest"}, "asset_record": {"sha256": "3" * 64}, "inputs": {}})
        asset_manifest = self.write("assets_manifest.json", {"synthetic_fixture_only": True})
        self.plan["asset_record"] = {"root": "/accepted/assets", "manifest": asset_manifest["path"], "sha256": asset_manifest["sha256"]}
        asset_input = self.write("asset_input.json", self.plan["asset_record"])
        self.plan["inputs"]["asset_record"] = asset_input
        asset_value = {"root": "/accepted/assets", "manifest_sha256": asset_manifest["sha256"], "file_count": 4531}
        qualified_asset_receipt = self.write("qualified_asset_receipt.json", asset_value)
        self.plan["direct_binding"] = {"profile": {"gpu_uuid": snapshot.CARD_ONE}}
        backend = {key: value for key, value in (("selected_python", self.plan["python"]), ("site_packages", "/original/site"),
            ("image", {"sha256": "4" * 64}), ("distribution", {"name": "original"}), ("producer_source_sha256", "5" * 64))}
        preparation = self.write("backend_preparation.json", backend)
        current_backend = self.write("current_backend.json", backend)
        self.plan["inputs"]["direct_backend_image_record"] = preparation
        self.plan["direct_backend_image"] = {"expected": current_backend}
        roles = {}
        self.review_hashes = {}
        for name in ("qualification", "1", "2"):
            manifest_directory = self.root / name
            manifest_directory.mkdir()
            payload = manifest_directory / "captured.raw"
            payload.write_bytes(b"historically captured bytes")
            self.paths[str(payload)] = snapshot.digest(payload.read_bytes())
            self.payload = payload
            manifest = self.write(name + "/manifest.json", {"schema_version": 1, "kind": "portable_raw_evidence",
                "git_roots": [{} for unused in range(8)], "files": [{"original_path": "/old/source/" + str(number),
                    "relative_path": "captured.raw", "sha256": self.paths[str(payload)]} for number in range(693)]})
            history = self.write(name + "/history.json", {"evidence_manifest": manifest})
            result = self.write(name + "/result.json", {"execution_identity": {"hostname": "old_host", "pid": 123, "process_start_time": "987"}})
            actual_asset_receipt = self.write(name + "/asset_receipt.json", asset_value)
            raw_audit = self.write(name + "/raw_audit.json", {"asset_receipt": actual_asset_receipt})
            configuration = deepcopy(self.plan["configuration"])
            if name == "2":
                configuration["learner"]["n_copies"] = 2
            reference = deepcopy(configuration)
            reference["environment"] = deepcopy(self.plan["reference_configuration"]["environment"])
            old_plan = {key: deepcopy(self.plan[key]) for key in
                ("campaign_run_dir", "source_root", "source_commit", "core", "paired_source", "python", "runtime_environment", "asset_record")}
            old_plan.update({"configuration": configuration, "reference_configuration": reference,
                "copy_seeds": [1, 2] if name == "2" else [1], "helpers": {"source_commit": snapshot.ORIGINAL_HELPER_SOURCE},
                "direct_backend_image": {"expected": current_backend}, "inputs": {"direct_backend_image_record": preparation, "asset_record": asset_input},
                "qualification_software": {"asset_receipt": qualified_asset_receipt},
                "direct_binding": {"frozen_inputs": history, "profile": self.plan["direct_binding"]["profile"]}})
            plan_reference = self.write(name + "/plan.json", old_plan)
            review = self.write(name + "/review.json", {"synthetic_fixture_only": True, "role": name, "result": result, "raw_audit": raw_audit})
            self.review_hashes[name] = review["sha256"]
            roles[name] = {key: deepcopy(self.plan[key]) for key in
                ("campaign_run_dir", "source_root", "source_commit", "core", "paired_source", "python", "runtime_environment", "asset_record")}
            roles[name].update({"review": review, "result": result, "plan": plan_reference,
                "helper_commit": snapshot.ORIGINAL_HELPER_SOURCE,
                "configuration": configuration, "reference_configuration": reference,
                "copy_seeds": [1, 2] if name == "2" else [1], "direct_profile": self.plan["direct_binding"]["profile"],
                "backend_preparation": preparation, "backend_expected": current_backend,
                "historical_inputs": history, "historical_manifest": manifest,
                "asset_input": asset_input, "actual_asset_receipt": actual_asset_receipt, "qualified_asset_receipt": qualified_asset_receipt,
                "raw_validation_provenance": {"original_statements": 55, "retained_raw_statements": 53}})
        with snapshot.ValidationFiles() as files:
            for path, expected in self.paths.items():
                files.file(path, expected)
            physical = files.finish(self.paths)
        # These identities are scoped only to a different CPU host. The worker must not stat its payloads.
        physical["producer_hostname"] = "different_cpu_host"
        for row in physical["physical_files"]:
            row["identity"]["st_dev"] += 1000000
        self.proof = {"schema_version": 1, "kind": snapshot.PROOF_KIND, "complete": True, "protocol": dict(snapshot.PROTOCOL),
            "producer_source_sha256": self.plan["helpers"]["files"][snapshot.PREFIX + snapshot.NAME],
            "producer": {"hostname": "different_cpu_host"}, "created_at_seconds": 1.0,
            "file_snapshot": physical, "declared_files": [{"path": path, "sha256": expected} for path, expected in self.paths.items()],
            "historical_manifests": {role["historical_manifest"]["path"]: {} for role in roles.values()},
            "accepted": {"qualification": roles["qualification"], "correctness": {key: roles[key] for key in ("1", "2")}},
            "previous_gpu_or_raw_gate_cases_reexecuted": False, "live_training_inputs_validated_by_this_snapshot": False,
            "cross_host_descriptor_identity_asserted": False, "worker_admitted": False, "admission_granted": False,
            "scientific_credit": 0, "case_credit": 0, "idea_credit": 0, "site_qualification_credit": 0,
            "final_training_credit": 0, "throughput_credit": 0, "packing_point_credit": 0}
        proof_reference = self.write("proof.json", self.proof)
        self.review = {"schema_version": 1, "kind": snapshot.REVIEW_KIND, "approved": True, "passed": True,
            "independent_of_producer": True, "snapshot": proof_reference,
            "producer_source_sha256": self.proof["producer_source_sha256"], "complete_coverage_verified": True,
            "unresolved_anomalies": [], "reviewer_identity": "synthetic independent fixture", "reviewed_at_seconds": 2.0,
            "admission_granted": False, "worker_admitted": False, "scientific_credit": 0}
        review_reference = self.write("review.json", self.review)
        self.plan["accepted_evidence_snapshot_inputs"] = {"proof": proof_reference, "review": review_reference}
        self.plan["direct_qualification_inputs"] = {key: roles["qualification"][key] for key in ("review", "result")}
        self.plan["common_correctness_inputs"] = {key: roles["1"][key] for key in ("review", "result")}
        history_path = self.root / "data/new_common/direct_history/inputs.json"
        self.plan["direct_binding"]["frozen_inputs"] = snapshot.stage_history_inputs(self.plan, history_path)
        self.allowlist = patch.object(snapshot, "ACCEPTED_REVIEW_SHA256", self.review_hashes)
        self.allowlist.start()

    def tearDown(self):
        self.allowlist.stop()
        for value in snapshot._CONSUMERS.values():
            value.files.close()
        snapshot._CONSUMERS.clear()
        self.temporary.cleanup()

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = snapshot.encoded(value)
        path.write_bytes(raw)
        reference = {"path": str(path), "sha256": snapshot.digest(raw)}
        self.paths[str(path)] = reference["sha256"]
        return reference

    def replace_review(self):
        self.plan["accepted_evidence_snapshot_inputs"]["review"] = self.write("review.json", self.review)

    def test_reviewed_fact_consumes_without_opening_any_historical_payload(self):
        original = snapshot.ValidationFiles.file
        def refuse_payload(files, path, expected, **options):
            if Path(path).name == "captured.raw":
                raise AssertionError("the worker must not reopen historical payload bytes")
            return original(files, path, expected, **options)
        with patch.object(snapshot.ValidationFiles, "file", refuse_payload):
            result = snapshot.consume(self.plan)
            self.assertTrue(result.proof["complete"])
            self.assertEqual(result.proof["producer"]["hostname"], "different_cpu_host")
            self.assertEqual(snapshot.history(self.plan)["comparisons_recomputed"], 0)

    def test_old_payload_change_does_not_claim_or_require_gpu_time_equality(self):
        self.payload.write_bytes(b"later bytes; original CPU snapshot remains a historical fact")
        self.assertFalse(snapshot.history(self.plan)["cpu_descriptor_identity_asserted_on_gpu"])
        result = snapshot.common_validation(self.plan, snapshot.digest(snapshot.encoded(self.plan)))
        self.assertFalse(result["gpu_time_historical_byte_equality_asserted"])

    def test_partial_cpu_coverage_is_refused(self):
        self.proof["file_snapshot"]["physical_files"].pop()
        self.plan["accepted_evidence_snapshot_inputs"]["proof"] = self.write("proof.json", self.proof)
        self.review["snapshot"] = self.plan["accepted_evidence_snapshot_inputs"]["proof"]
        self.replace_review()
        with self.assertRaises(ValueError):
            snapshot.consume(self.plan)

    def test_unapproved_snapshot_review_is_refused(self):
        self.review["approved"] = False; self.replace_review()
        with self.assertRaises(ValueError):
            snapshot.consume(self.plan)

    def test_changed_snapshot_review_is_refused_on_cached_use(self):
        snapshot.consume(self.plan)
        Path(self.plan["accepted_evidence_snapshot_inputs"]["review"]["path"]).write_bytes(b"{}\n")
        with self.assertRaises(ValueError):
            snapshot.consume(self.plan)

    def test_changed_independent_seeds_are_refused(self):
        self.plan["copy_seeds"] = [2]
        with self.assertRaises(ValueError):
            snapshot.consume(self.plan)

    def test_changed_backend_record_is_refused(self):
        self.plan["inputs"]["direct_backend_image_record"] = self.write("other_backend_preparation.json", {"other": True})
        with self.assertRaises(ValueError):
            snapshot.consume(self.plan)

    def test_missing_captured_payload_coverage_is_refused(self):
        self.proof["historical_manifests"].pop(next(iter(self.proof["historical_manifests"])))
        self.plan["accepted_evidence_snapshot_inputs"]["proof"] = self.write("proof.json", self.proof)
        self.review["snapshot"] = self.plan["accepted_evidence_snapshot_inputs"]["proof"]
        self.replace_review()
        with self.assertRaises(ValueError):
            snapshot.consume(self.plan)


if __name__ == "__main__":
    unittest.main()
