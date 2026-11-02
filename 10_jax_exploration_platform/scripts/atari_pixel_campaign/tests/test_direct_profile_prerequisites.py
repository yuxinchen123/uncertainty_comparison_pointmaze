"""Metadata-only direct profile tests; no actual tool, binary, model, GPU or target qualification."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import profile_selected as profile
import direct_profile_prerequisites as prerequisite
import direct_backend_image as backend

spec = importlib.util.spec_from_file_location("synthetic_backend_test_fixture", SCRIPTS / "tests/test_direct_backend_image.py")
fixture_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture_module)


def original_configuration():
    """Read only original dataclass defaults through the existing metadata parser."""
    value = profile.diagnostic.default_configuration(SCRIPTS.parents[2])
    value["learner"]["n_copies"] = 1
    return value


def profile_plan():
    """Describe a private metadata protocol shape; its missing references cannot pass actual preparation."""
    configuration = original_configuration()
    names = profile.required_helpers(False, True, True)
    files = {backend.PREFIX + name: backend.digest((SCRIPTS / name).read_bytes()) for name in names}
    missing = {"result": {"path": "/private-synthetic-missing/result.json", "sha256": "a" * 64},
        "review": {"path": "/private-synthetic-missing/review.json", "sha256": "b" * 64}}
    return {"kind": "selected_configuration_profile", "mode": "profile", "hardware_class": "h100",
        "gpu_uuid": backend.CARD_ONE, "direct_binding": {"consumer": "component_profile"},
        "helpers": {"files": files}, "configuration": configuration, "copy_seeds": [1], "profile_updates": 1,
        "runtime_environment": {"JAX_ENABLE_X64": "true", "JAX_DEFAULT_MATMUL_PRECISION": "highest",
            "XLA_FLAGS": "--xla_gpu_exclude_nondeterministic_ops=true"},
        "direct_backend_image": {"schema_version": 2},
        "direct_profile_prerequisites": {"schema_version": 1, "capability": prerequisite.CAPABILITY,
            "mode": "qualification", "correctness": copy.deepcopy(missing), "timing": copy.deepcopy(missing),
            "lifecycle": copy.deepcopy(missing), "capture_qualification": None, "metadata_timeout_seconds": 30}}


class ProfileContractTests(unittest.TestCase):
    """Validate protocol shape while leaving all actual prerequisite files unresolved."""

    def setUp(self):
        self.plan = profile_plan()

    def refuse(self, mutate):
        mutate(self.plan)
        with self.assertRaises((ValueError, KeyError)): prerequisite.contract(self.plan)

    def test_explicit_profile_protocol_shape_and_current_twenty_eight_helpers(self):
        self.assertEqual(prerequisite.contract(self.plan)["mode"], "qualification")
        self.assertEqual(len(profile.required_helpers(False, True)), 28)
        self.assertEqual(len(profile.required_helpers(False, True, True)), 28)
        self.assertIn("direct_backend_image.py", profile.required_helpers(False, True))
        self.assertIn("direct_profile_prerequisites.py", profile.required_helpers(False, True))
        self.assertNotIn("direct_common_confirmation.py", profile.required_helpers(False, True))

    def test_default_and_a4500_profile_helper_rosters_remain_unchanged(self):
        self.assertEqual(len(profile.required_helpers()), 17)
        self.assertEqual(len(profile.required_helpers(True)), 21)
        self.assertEqual(len(profile.required_helpers(True, False, True)), 22)

    def test_mixed_hardware_helper_requests_refuse_before_staging(self):
        for common in (False, True):
            with self.subTest(common=common), self.assertRaises(ValueError): profile.required_helpers(True, True, common)

    def test_a_profile_cannot_be_relabelled_as_a_confirmation(self):
        self.refuse(lambda x: x.update(kind="selected_configuration_measurement", mode="confirmation"))

    def test_missing_backend_schema_refuses(self):
        self.refuse(lambda x: x.pop("direct_backend_image"))

    def test_missing_current_adapter_refuses(self):
        self.refuse(lambda x: x["helpers"]["files"].pop(backend.PREFIX + prerequisite.NAME))

    def test_current_optional_confirmation_helper_must_not_replace_historical_closures(self):
        self.refuse(lambda x: x["helpers"]["files"].update({backend.PREFIX + "direct_common_confirmation.py": "a" * 64}))

    def test_other_gpu_refuses(self):
        self.refuse(lambda x: x.update(gpu_uuid="GPU-other"))

    def test_changed_copy_seed_refuses(self):
        self.refuse(lambda x: x.update(copy_seeds=[2]))

    def test_partial_original_configuration_refuses(self):
        self.refuse(lambda x: x["configuration"]["learner"].update(num_steps=64))

    def test_boolean_timeout_refuses(self):
        self.refuse(lambda x: x["direct_profile_prerequisites"].update(metadata_timeout_seconds=True))

    def test_qualification_cannot_require_its_own_capture(self):
        self.refuse(lambda x: x["direct_profile_prerequisites"].update(capture_qualification={"result": {}, "review": {}}))

    def test_measurement_without_real_capture_qualification_refuses(self):
        self.refuse(lambda x: x["direct_profile_prerequisites"].update(mode="measurement"))

    def test_unsupported_final_group_and_instrumented_features_refuse(self):
        for name in ("final_protocol", "configured_group", "device_memory_limit_protocol"):
            with self.subTest(name=name):
                changed = copy.deepcopy(self.plan); changed[name] = {}
                with self.assertRaises(ValueError): prerequisite.contract(changed)

    def test_missing_real_records_cannot_become_preparation_approval(self):
        with self.assertRaises((ValueError, FileNotFoundError)):
            prerequisite.historical_metadata(self.plan)


class ProfileBackendTests(unittest.TestCase):
    """Reuse small synthetic package files to check the separate complete-profile receipt branch."""

    def setUp(self):
        self.fixture = fixture_module.DirectBackendImageTests("test_actual_unique_record_owner_is_discovered_without_assuming_package_name")
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        self.plan = self.fixture.plan
        shape = profile_plan()
        self.plan.update({key: value for key, value in shape.items() if key not in {"direct_backend_image"}})
        self.fixture.write(self.fixture.unit / "measurement_plan.json", self.plan)

    def result(self):
        value = self.fixture.result()
        value.update(capture_completed=True, profile_updates=1, profile_start_iteration=1, profile_end_iteration=2,
            normalization_warmup_iterations=50, compiled_signatures_before=[1], compiled_signatures_after=[1],
            configuration=self.plan["configuration"], copy_seeds=[1], runtime_environment=self.plan["runtime_environment"])
        return value

    def test_profile_plan_and_completed_full_capture_have_separate_labels(self):
        self.assertEqual(backend.plan_path(self.plan, self.fixture.unit), self.fixture.unit / "measurement_plan.json")
        self.assertEqual(backend.completed_work(self.plan, self.result()), "direct_profile_complete_update_capture")
        selected = {"kind": "selected_configuration_measurement", "mode": "correctness"}
        self.assertEqual(backend.plan_path(selected, self.fixture.unit), self.fixture.unit / "measurement_plan.json")

    def test_incomplete_count_cache_or_warmup_refuses(self):
        for fields in ({"profile_end_iteration": 1}, {"profile_start_iteration": True},
                {"compiled_signatures_after": [2]}, {"normalization_warmup_iterations": 49},
                {"capture_completed": False}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                backend.completed_work(self.plan, {**self.result(), **fields})

    def test_float_execution_pid_refuses(self):
        value = self.result(); value["execution_identity"]["pid"] = float(value["execution_identity"]["pid"])
        with self.assertRaises(ValueError): backend.completed_work(self.plan, value)

    def test_actual_profile_capture_order_precedes_trainer_window(self):
        self.fixture.begin()
        value = self.result()
        value["execution_identity"]["started_at_seconds"] = __import__("time").time()
        value["execution_identity"]["finished_at_seconds"] = __import__("time").time()
        with mock.patch.object(backend, "own_process", return_value=self.fixture.process), \
                mock.patch.object(backend, "mapped_snapshot", side_effect=self.fixture.maps):
            completed = backend.capture_complete(self.fixture.unit, self.plan, value)
        self.assertEqual(completed["completed_work_kind"], "direct_profile_complete_update_capture")

    def test_coherent_different_python_environment_cannot_reuse_old_package(self):
        wrong = copy.deepcopy(self.fixture.expected)
        python = self.fixture.root / "other-environment/bin/python"; python.parent.mkdir(parents=True)
        python.write_text("private fixture; never executed")
        wrong["selected_python"] = str(python)
        with self.assertRaises(ValueError): backend.validate_expected(wrong, str(python))

    def test_profile_update_count_requires_integer_type(self):
        result = self.result(); result["profile_updates"] = 1.0
        with self.assertRaises(ValueError): backend.completed_work(self.plan, result)


class ProfileSourceTests(unittest.TestCase):
    """Verify that wrapping metadata does not replace the numerical capture core."""

    def test_original_capture_core_has_no_new_backend_or_prerequisite_calls_inside_it(self):
        tree = ast.parse((SCRIPTS / "profile_selected.py").read_text())
        core = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_capture_core")
        calls = [node for node in ast.walk(core) if isinstance(node, ast.Call)]
        self.assertFalse(any(isinstance(node.func, ast.Attribute)
            and node.func.attr in {"after_capture", "before_compute", "capture_backend_completion"} for node in calls))

    def test_legacy_direct_prepare_requires_new_actual_backend_inputs(self):
        args = __import__("argparse").Namespace(direct_binding_profile=Path("/not/read"),
            configured_isolation_profile=None, direct_backend_image_record=None,
            hardware_class="h100", gpu_uuid=backend.CARD_ONE)
        with self.assertRaises(ValueError): profile.prepare(args)

    def test_legacy_default_cannot_bypass_explicit_card_one_protocol(self):
        args = __import__("argparse").Namespace(direct_binding_profile=None, configured_isolation_profile=None,
            direct_backend_image_record=None, hardware_class="h100", gpu_uuid=backend.CARD_ONE)
        with self.assertRaises(ValueError): profile.prepare(args)


class HistoricalPackageAndChildTests(unittest.TestCase):
    """Use complete small metadata records to test independent package and saved-child joins."""

    def setUp(self):
        self.fixture = fixture_module.DirectBackendImageTests("test_actual_unique_record_owner_is_discovered_without_assuming_package_name")
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        self.unit = self.fixture.unit
        self.plan = copy.deepcopy(self.fixture.plan)
        self.plan.update({key: value for key, value in profile_plan().items() if key != "direct_backend_image"})
        self.plan.update(source_root="/private/synthetic/numerical", source_commit="1" * 40,
            core={"source_root": "/private/synthetic/numerical", "source_commit": "1" * 40},
            paired_source={"source_root": "/private/synthetic/numerical", "source_commit": "1" * 40},
            asset_record={"root": "/private/synthetic/assets"}, accepted_baseline_specification={"synthetic_only": True})
        self.plan["helpers"].update(source_root="/private/synthetic/helpers", source_commit="2" * 40)
        self.plan = json.loads(prerequisite.encoded(self.plan))
        qualification = self.unit.parent / "qualification"; qualification.mkdir()
        self.historical_expected = copy.deepcopy(self.fixture.expected)
        self.historical_expected["producer_source_sha256"] = "a" * 64
        expected = self.put(qualification / "expected.json", self.historical_expected)
        self.put(qualification / "measurement_plan.json", {"direct_backend_image": {"expected": expected}})
        self.plan["direct_qualification_inputs"] = {"result": self.put(qualification / "direct_qualification_result.json",
            {"synthetic_only": True}), "review": {"path": "/private/synthetic/not-read", "sha256": "b" * 64}}
        self.directories = {}
        for number, phase in enumerate(("correctness", "timing"), 1):
            directory = self.unit.parent / phase; directory.mkdir(); self.directories[phase] = directory
            (directory / "code").mkdir()
            helpers = {"source_root": "/private/synthetic/historical-helpers", "source_commit": str(number + 2) * 40,
                "files": {}}
            if phase == "timing":
                guard = directory / "code/direct_common_confirmation.py"
                guard.write_bytes((SCRIPTS / guard.name).read_bytes())
                helpers["files"][backend.PREFIX + guard.name] = backend.digest(guard.read_bytes())
            old = {key: copy.deepcopy(self.plan[key]) for key in ("campaign_run_dir", "source_root", "source_commit", "core",
                "paired_source", "python", "hardware_class", "runtime_environment", "asset_record", "direct_qualification_inputs",
                "configuration", "copy_seeds")}
            old.update(kind="selected_configuration_measurement", mode="correctness" if phase == "correctness" else "confirmation",
                unit_id=phase, gpu_uuid=backend.CARD_ONE, direct_binding={}, helpers=helpers,
                direct_backend_image={"expected": self.put(directory / "expected.json", self.historical_expected)})
            old_ref = self.put(directory / "measurement_plan.json", old)
            result = {"candidate": self.plan["configuration"], "copy_seeds": [1], "helpers": helpers, "measurement_mode": old["mode"],
                "execution_identity": {"hostname": "serval02", "pid": number, "process_start_time": str(number),
                    "started_at_seconds": 1, "finished_at_seconds": 2},
                "correctness_attempted": True, "correctness_passed": True, "implementation_distinct": True,
                "measurement_valid": phase == "timing", "contaminated": False, "decision": "accepted", "rounds": [{}] * 7}
            result_ref = self.put(directory / "result.json", result)
            raw_ref = self.put(directory / "raw.json", {"passed": True, "result": result_ref, "unchanged_staged_consumer_passed": True})
            review = {"schema_version": 1, "kind": "independent_actual_direct_common_configuration_" + phase + "_review",
                "result": result_ref, "approved": True, "passed": True, "independent_of_producer": True,
                "unresolved_anomalies": [], "admission_granted": False, "worker_admitted": False,
                "reviewer_identity": "private synthetic metadata fixture; no actual scientific approval",
                "reviewed_at_seconds": 3, "raw_audit": raw_ref, "plan": old_ref,
                "configuration": old["configuration"], "copy_seeds": [1], "helper_commit": helpers["source_commit"],
                **{name: 0 for name in prerequisite.ZERO_FIELDS}}
            self.plan["direct_profile_prerequisites"][phase] = {"result": result_ref,
                "review": self.put(directory / "review.json", review)}
            self.put(directory / "direct_device_binding.json", {"synthetic_only": True})
        self.put(self.unit / "measurement_plan.json", self.plan)
        self.process = {"hostname": "serval02", "pid": 77, "process_start_time": "88", "uid": 2618919}
        (self.unit / "supervisor").mkdir()
        self.put(self.unit / "supervisor/direct_process_metadata.json",
            {"before": {"identity": self.process}, "after": {"identity": self.process}})

    def put(self, path, value):
        path.write_bytes(prerequisite.encoded(value)); return prerequisite.reference(path)

    def child(self, phase, number):
        pair = self.plan["direct_profile_prerequisites"][phase]; directory = self.directories[phase]
        old = prerequisite.read(directory / "measurement_plan.json")[0]
        bootstrap = {"module_path": "/usr/lib/python3.12/ctypes/__init__.py", "python_api_target": None,
            "own_process_python_api_initialized_before_validation_audit": True,
            "bootstrap_scope": "stdlib ctypes own-process Python API handle only", "later_library_loads_and_symbol_calls_refused": True}
        baseline = {"configuration": self.plan["configuration"], "hardware_class": "h100", "source_commit": self.plan["source_commit"],
            "runtime_environment": self.plan["runtime_environment"], "copies": 1, "paired_rounds": 7} if phase == "timing" else None
        return {"schema_version": 1, "kind": "exact_historical_direct_profile_prerequisite_validation", "passed": True,
            "phase": phase, "profile_plan_sha256": prerequisite.digest(prerequisite.encoded(self.plan)), "parent_pid": 77,
            "validation_pid": 100 + number, "validation_process": {"hostname": "serval02", "pid": 100 + number,
                "process_start_time": str(100 + number), "uid": 2618919}, "numerical_imports": [], "gpu_operations": 0,
            "scientific_credit": 0, "compute": {"hostname": "serval02", "uid": 2618919, "allocation_kind": "authorized_direct_compute"},
            "started_at_seconds": 10 + number, "finished_at_seconds": 10.5 + number, "historical_plan": prerequisite.reference(directory / "measurement_plan.json"),
            "result": pair["result"], "helpers": old["helpers"], "resolved_binding": prerequisite.reference(directory / "direct_device_binding.json"),
            "execution_identity": prerequisite.checked(pair["result"])["execution_identity"],
            "stdlib_bootstrap": bootstrap, "accepted_combination": baseline}

    def saved(self, mutate=None):
        rows = []
        for number, phase in enumerate(("correctness", "timing"), 1):
            proof = self.child(phase, number)
            if mutate is not None: mutate(proof)
            raw = prerequisite.encoded(proof); prefix = "direct_profile_" + phase + "_validation"
            (self.unit / (prefix + "_stdout.raw")).write_bytes(raw); (self.unit / (prefix + "_stderr.raw")).write_bytes(b"")
            command = ["/usr/bin/python3", "-I", "-S", "-B", str(self.unit / "code" / prerequisite.NAME), "--historical-phase", phase]
            returned = self.put(self.unit / (prefix + "_return.json"), {"command": command, "returncode": 0,
                "stdout_sha256": prerequisite.digest(raw), "stderr_sha256": prerequisite.digest(b"")})
            rows.append({"phase": phase, "proof": proof,
                "stdout": {"path": str(self.unit / (prefix + "_stdout.raw")), "sha256": prerequisite.digest(raw)}, "return_record": returned})
        self.put(self.unit / prerequisite.PROOF_NAME, {"schema_version": 1, "kind": "direct_profile_exact_historical_prerequisites",
            "passed": True, "plan_sha256": prerequisite.reference(self.unit / "measurement_plan.json")["sha256"],
            "inputs": self.plan["direct_profile_prerequisites"], "phases": rows, "parent_pid": 77, "parent_process": self.process,
            "compute": {"hostname": "serval02", "uid": 2618919, "allocation_kind": "authorized_direct_compute"},
            "started_at_seconds": 10, "finished_at_seconds": 13, "numerical_imports": [], "gpu_operations": 0, "scientific_credit": 0})
        with mock.patch.object(prerequisite, "lifecycle_metadata", return_value={}):
            return prerequisite.require_saved_validation(self.plan)

    def test_identical_package_can_keep_distinct_historical_producer_receipts(self):
        self.assertEqual(set(prerequisite.historical_metadata(self.plan)), {"correctness", "timing"})

    def test_new_package_image_hash_cannot_reuse_accepted_history(self):
        expected_path = Path(self.plan["direct_backend_image"]["expected"]["path"])
        changed = prerequisite.read(expected_path)[0]; changed["image"]["sha256"] = "c" * 64
        self.plan["direct_backend_image"]["expected"] = self.put(expected_path, changed)
        with self.assertRaises(ValueError): prerequisite.historical_metadata(self.plan)

    def test_new_owning_distribution_cannot_reuse_accepted_history(self):
        expected_path = Path(self.plan["direct_backend_image"]["expected"]["path"])
        changed = prerequisite.read(expected_path)[0]; changed["distribution"]["version"] = "2"
        self.plan["direct_backend_image"]["expected"] = self.put(expected_path, changed)
        with self.assertRaises(ValueError): prerequisite.historical_metadata(self.plan)

    def test_actual_qualification_package_is_checked_separately(self):
        directory = self.unit.parent / "qualification"; old = prerequisite.read(directory / "measurement_plan.json")[0]
        changed = copy.deepcopy(self.historical_expected); changed["image"]["sha256"] = "d" * 64
        old["direct_backend_image"]["expected"] = self.put(directory / "expected.json", changed)
        self.put(directory / "measurement_plan.json", old)
        with self.assertRaises(ValueError): prerequisite.historical_metadata(self.plan)

    def test_saved_complete_child_shapes_pass_without_scientific_validation(self):
        self.assertTrue(self.saved()["passed"])

    def test_coherently_rehashed_false_child_pass_refuses(self):
        with self.assertRaises(ValueError): self.saved(lambda p: p.update(passed=False))

    def test_coherently_rehashed_wrong_phase_refuses(self):
        with self.assertRaises(ValueError): self.saved(lambda p: p.update(phase="another_phase"))

    def test_coherently_rehashed_wrong_parent_refuses(self):
        with self.assertRaises(ValueError): self.saved(lambda p: p.update(parent_pid=78))

    def test_coherently_rehashed_float_child_pid_refuses(self):
        with self.assertRaises(ValueError): self.saved(lambda p: p.update(validation_pid=float(p["validation_pid"])))

    def test_coherently_rehashed_numerical_import_claim_refuses(self):
        with self.assertRaises(ValueError): self.saved(lambda p: p.update(numerical_imports=["jax"]))

    def test_coherently_rehashed_gpu_operation_claim_refuses(self):
        with self.assertRaises(ValueError): self.saved(lambda p: p.update(gpu_operations=1))

    def test_coherently_rehashed_child_interval_outside_parent_refuses(self):
        with self.assertRaises(ValueError): self.saved(lambda p: p.update(finished_at_seconds=14))

    def test_coherently_rehashed_numeric_context_uid_alias_refuses(self):
        with self.assertRaises(ValueError): self.saved(lambda p: p["compute"].update(uid=2618919.0))

    def test_coherently_rehashed_overlapping_phase_intervals_refuse(self):
        with self.assertRaises(ValueError): self.saved(lambda p: p.update(started_at_seconds=11, finished_at_seconds=12))

    def test_coherently_rehashed_changed_old_execution_refuses(self):
        with self.assertRaises(ValueError): self.saved(lambda p: p["execution_identity"].update(pid=999))

    def test_changed_supervisor_birth_cannot_reuse_saved_parent(self):
        self.process["process_start_time"] = "999"
        with self.assertRaises(ValueError): self.saved()

    def test_timing_nested_command_is_exact_and_retains_other_guard_checks(self):
        calls = []
        guard = types.SimpleNamespace(metadata_event_guard=lambda *args: calls.append(args))
        command = ["/usr/bin/python3", "-I", "-S", "-B", "/private/synthetic/timing/code/direct_common_confirmation.py", "--plan-stdin"]
        prerequisite.historical_event_guard(guard, "subprocess.Popen", (command[0], command, None, None), set(), command)
        self.assertEqual(calls, [])
        for values in ((command[0], command, "/another", None), (command[0], command, None, {})):
            with self.assertRaises(ValueError): prerequisite.historical_event_guard(guard, "subprocess.Popen", values, set(), command)
        prerequisite.historical_event_guard(guard, "ctypes.dlopen", (None,), set(), command)
        self.assertEqual(calls[0][0], "ctypes.dlopen")

    def test_other_python_command_still_uses_historical_refusal(self):
        guard = types.SimpleNamespace(metadata_event_guard=mock.Mock(side_effect=ValueError("only declared Git")))
        command = ["/usr/bin/python3", "-I", "-S", "-B", "/private/synthetic/other.py"]
        with self.assertRaises(ValueError): prerequisite.historical_event_guard(guard, "subprocess.Popen",
            (command[0], command, None, None), set(), None)


class CaptureFieldTests(unittest.TestCase):
    """Stop just after raw comparison, independently of the later real terminal/trace validators."""

    def setUp(self):
        import direct_measurement
        self.direct = direct_measurement
        self.raw = {"profile_updates": 1, "profile_start_iteration": 10, "profile_end_iteration": 11,
            "compiled_signatures_before": [1, 2], "compiled_signatures_after": [1, 2], "capture_completed": True}
        values = {name: {} for name in ("direct_profile_terminal.json", "profiler_process.json", "measurement_plan.json",
            "capture_identity.json", "capture_release.json")}
        values["capture_result.json"] = self.raw
        self.identity = types.SimpleNamespace(receipt=lambda path: {"path": str(path)},
            read_receipt=lambda ref: values[Path(ref["path"]).name],
            exact=lambda a, b: prerequisite.encoded(a) == prerequisite.encoded(b))

    def check(self, result):
        with mock.patch.object(self.direct, "core", return_value=self.identity), \
                mock.patch.object(self.direct, "validate_capture_wait", side_effect=RuntimeError("after exact raw comparison")):
            self.direct.validate_profile_terminal(Path("/private/synthetic/profile"), {"after": {"identity": {}}}, result)

    def test_unchanged_payload_reaches_the_independent_next_gate(self):
        with self.assertRaisesRegex(RuntimeError, "after exact raw comparison"): self.check(copy.deepcopy(self.raw))

    def test_different_equal_length_iteration_window_refuses(self):
        result = copy.deepcopy(self.raw); result.update(profile_start_iteration=20, profile_end_iteration=21)
        with self.assertRaises(ValueError): self.check(result)

    def test_different_equal_before_and_after_signature_lists_refuse(self):
        result = copy.deepcopy(self.raw); result.update(compiled_signatures_before=[3], compiled_signatures_after=[3])
        with self.assertRaises(ValueError): self.check(result)

    def test_numeric_alias_in_a_raw_field_refuses(self):
        result = copy.deepcopy(self.raw); result["profile_updates"] = 1.0
        with self.assertRaises(ValueError): self.check(result)


class SavedMarkerTests(unittest.TestCase):
    """Recheck the saved package-completion joins without invoking any real scientific consumer."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.unit = Path(self.temporary.name)
        self.protocol = {"mode": "qualification", "private_synthetic_fixture": True}
        self.plan = {"direct_profile_prerequisites": self.protocol}
        self.execution = {"hostname": "serval02", "pid": 100, "process_start_time": "200",
            "started_at_seconds": 20, "finished_at_seconds": 30}
        self.proof = {"started_at_seconds": 10, "finished_at_seconds": 15}
        self.capture = {"execution_identity": self.execution, "private_synthetic_fixture": True}
        self.backend = {"numerical_execution": copy.deepcopy(self.execution)}
        self.marker = {"schema_version": 1, "kind": "direct_profile_actual_package_capture_completion",
            "capture_result": {}, "backend_capture": {}, "prerequisite_validation": {}, "profile_protocol": self.protocol,
            "execution_identity": copy.deepcopy(self.execution), "scientific_credit": 0}

    def put(self, path, value):
        path.write_bytes(prerequisite.encoded(value)); return prerequisite.reference(path)

    def check(self):
        self.marker.update(capture_result=self.put(self.unit / "capture_result.json", self.capture),
            backend_capture=self.put(self.unit / "direct_backend_capture.json", self.backend),
            prerequisite_validation=self.put(self.unit / prerequisite.PROOF_NAME, self.proof))
        self.put(self.unit / "direct_profile_backend_completion.json", self.marker)
        with mock.patch.object(prerequisite, "contract", return_value=self.protocol), \
                mock.patch.object(prerequisite, "require_saved_validation", return_value=self.proof):
            return prerequisite.result_binding(self.unit, self.plan)

    def test_exact_raw_backend_marker_and_prior_validation_pass(self):
        self.assertTrue(self.check()["profile_qualification_only"])

    def test_changed_marker_execution_refuses_after_coherent_reference_update(self):
        self.marker["execution_identity"]["pid"] = 101
        with self.assertRaises(ValueError): self.check()

    def test_changed_backend_numerical_execution_refuses(self):
        self.backend["numerical_execution"]["pid"] = 101
        with self.assertRaises(ValueError): self.check()

    def test_prior_validation_finishing_after_capture_start_refuses(self):
        self.proof["finished_at_seconds"] = 21
        with self.assertRaises(ValueError): self.check()

    def test_boolean_saved_validation_time_refuses(self):
        self.proof["finished_at_seconds"] = True
        with self.assertRaises(ValueError): self.check()

    def test_boolean_marker_schema_refuses(self):
        self.marker["schema_version"] = True
        with self.assertRaises(ValueError): self.check()

    def test_numeric_zero_credit_alias_refuses(self):
        self.marker["scientific_credit"] = 0.0
        with self.assertRaises(ValueError): self.check()

    def test_unknown_marker_kind_refuses(self):
        self.marker["kind"] = "another_saved_marker"
        with self.assertRaises(ValueError): self.check()


if __name__ == "__main__":
    unittest.main()
