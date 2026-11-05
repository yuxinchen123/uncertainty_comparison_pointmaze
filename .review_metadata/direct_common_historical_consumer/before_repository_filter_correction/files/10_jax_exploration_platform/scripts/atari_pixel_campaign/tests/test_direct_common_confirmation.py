"""Exercise metadata prerequisite checks without numerical or device operations."""
import copy
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import shutil
import unittest
from unittest.mock import patch
from types import SimpleNamespace

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import direct_common_confirmation as common
import measure_selected as measurement


class DirectCommonConfirmationTests(unittest.TestCase):
    """Keep current common checks separate from historical and final protocols."""

    def setUp(self):
        """Make one minimal explicit intermediate configuration and immutable inputs."""
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.unit = Path(self.temporary.name)
        self.plan = {"kind": "selected_configuration_measurement", "mode": "confirmation",
            "hardware_class": "h100", "gpu_uuid": common.CARD_ONE, "direct_binding": {"profile": {}},
            "rounds": 7, "minimum_updates_per_round": 10, "iterations": 10, "copy_seeds": [1],
            "configuration": {"learner": {"n_copies": 1}, "environment": {"cache_room_backgrounds": True,
                "render_final_two_only": True}, "execution": {}},
            "reference_configuration": {"learner": {"n_copies": 1}, "environment": {"cache_room_backgrounds": False,
                "render_final_two_only": False}, "execution": {}},
            "helpers": {"source_commit": "a" * 40, "files": {common.PREFIX + name:
                hashlib.sha256((SCRIPTS / name).read_bytes()).hexdigest()
                for name in measurement.required_helpers(False, True, True, True)}},
            "direct_backend_image": {"schema_version": 2,
                "capability": "prepared_owning_package_image_and_same_process_mapping_version_2",
                "expected": {"path": str(self.unit / "synthetic_backend_expected.json"), "sha256": "e" * 64}},
            "direct_historical_consumer_protocol": copy.deepcopy(common.HISTORICAL_PROTOCOL),
            "common_correctness_inputs": {"result": {"path": str(self.unit / "prior.json"), "sha256": "b" * 64},
                "review": {"path": str(self.unit / "review.json"), "sha256": "c" * 64}}}
        (self.unit / "measurement_plan.json").write_bytes(common.encoded(self.plan))

    def proof(self, raw=None):
        """Return a synthetic successful child record with this actual test parent."""
        raw = common.encoded(self.plan) if raw is None else raw
        return {"schema_version": 1, "kind": "isolated_direct_common_correctness_validation", "passed": True,
            "plan_sha256": common.digest(raw), "common_correctness_inputs": self.plan["common_correctness_inputs"],
            "parent_pid": os.getpid(), "validation_pid": 12, "finished_at_seconds": 1., "numerical_imports": [], "gpu_operations": 0,
            "stdlib_ctypes_bootstrap": common.stdlib_ctypes_bootstrap()}

    def completed(self, proof=None):
        """Stand in only for the separately tested metadata child process."""
        return subprocess.CompletedProcess([], 0, json.dumps(proof or self.proof()).encode(), b"")

    def test_intermediate_contract_accepts_both_explicit_copy_counts(self):
        """The two accepted copy configurations retain their exact seed lists."""
        common.contract(self.plan)
        self.plan["configuration"]["learner"]["n_copies"] = 2
        self.plan["reference_configuration"]["learner"]["n_copies"] = 2
        self.plan["copy_seeds"] = [1, 2]
        common.contract(self.plan)

    def test_final_configured_capacity_and_wrong_mode_are_rejected(self):
        """No opt-in field can authorize a different or later protocol."""
        for key in ("configured_isolation", "direct_qualification", "target_qualification", "final_confirmation_protocol", "completed_study",
                    "device_memory_limit_protocol", "fixed_total_scaling", "scaling_protocol"):
            with self.subTest(key=key), self.assertRaises(ValueError):common.contract({**self.plan, key: {}})
        for mode in ("correctness", "packing-point", "learning-check"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):common.contract({**self.plan, "mode": mode})

    def test_window_types_and_incomplete_prerequisites_are_rejected(self):
        """Boolean counters, a reduced window and one missing record cannot pass."""
        for key, wrong in (("rounds", True), ("rounds", 11), ("iterations", 1), ("minimum_updates_per_round", 20),
                ("common_correctness_inputs", {"result": {}})):
            with self.subTest(key=key), self.assertRaises(ValueError):common.contract({**self.plan, key: wrong})

    def test_equal_or_extra_candidate_change_and_wrong_seeds_are_rejected(self):
        """Only the independently checked two environment changes are permitted."""
        equal = {**self.plan, "configuration": self.plan["reference_configuration"]}
        with self.assertRaises(ValueError):common.contract(equal)
        changed = copy.deepcopy(self.plan); changed["configuration"]["learner"]["learning_rate"] = 1
        changed["reference_configuration"]["learner"]["learning_rate"] = 2
        with self.assertRaises(ValueError):common.contract(changed)
        with self.assertRaises(ValueError):common.contract({**self.plan, "copy_seeds": [True]})
        changed = copy.deepcopy(self.plan); changed["configuration"]["environment"]["cache_room_backgrounds"] = 1
        with self.assertRaises(ValueError):common.contract(changed)

    def test_missing_backend_old_schema_and_helper_subset_are_rejected(self):
        """The new guard cannot make a historical direct closure operational."""
        for mutation in (lambda p:p.pop("direct_backend_image"),
                lambda p:p["direct_backend_image"].update(schema_version=1),
                lambda p:p["direct_backend_image"].update(schema_version=True),
                lambda p:p["helpers"]["files"].pop(common.PREFIX + "direct_backend_image.py")):
            plan = copy.deepcopy(self.plan); mutation(plan)
            with self.subTest(plan=plan.get("direct_backend_image")), self.assertRaises(ValueError):common.contract(plan)

    def test_staged_helper_bytes_and_reference_hashes_are_required(self):
        """A different helper or changed saved bytes do not substitute for the frozen records."""
        changed = copy.deepcopy(self.plan); changed["helpers"]["files"][common.PREFIX + common.NAME] = "0" * 64
        with self.assertRaises(ValueError):common.contract(changed)
        p = self.unit / "reference.json"; p.write_text('{"value":1}')
        record = common.reference(p); self.assertEqual(common.checked_reference(record), {"value": 1})
        p.write_text('{"value":2}')
        with self.assertRaises(ValueError):common.checked_reference(record)

    def test_child_request_leaves_parent_paths_and_public_modules_unchanged(self):
        """The parent dispatches only the declared no-site interpreter and changes no import paths."""
        before = list(sys.path); ideas = sys.modules.get("ideas")
        with patch.object(common.subprocess, "run", return_value=self.completed()) as run:
            common.request(self.plan)
        command = run.call_args.args[0]
        self.assertEqual(command, ["/usr/bin/python3", "-I", "-S", "-B", str(Path(common.__file__).absolute()), "--plan-stdin"])
        self.assertEqual(before, sys.path); self.assertIs(sys.modules.get("ideas"), ideas)

    def test_different_parent_raw_plan_and_boolean_gpu_claim_are_rejected(self):
        """A successful child response must describe this exact parent and plan."""
        for key, value in (("parent_pid", True), ("parent_pid", os.getpid() + 1), ("plan_sha256", "0" * 64), ("gpu_operations", False)):
            with self.subTest(key=key), patch.object(common.subprocess, "run", return_value=self.completed({**self.proof(), key: value})):
                with self.assertRaises(ValueError):common.request(self.plan)

    def test_success_captures_raw_child_before_publishing_proof_and_cannot_repeat(self):
        """Keep one durable actual validation attempt and its exact raw successful return."""
        with patch.object(common.subprocess, "run", return_value=self.completed()) as run:
            common.before_compute(self.unit, self.plan)
            binding = common.result_binding(self.unit, self.plan)
            self.assertEqual(binding["common_correctness_inputs"], self.plan["common_correctness_inputs"])
            self.assertTrue((self.unit / "direct_common_correctness_validation_stdout.raw").exists())
            with self.assertRaises(ValueError):common.before_compute(self.unit, self.plan)
            self.assertEqual(run.call_count, 1)

    def test_failed_and_malformed_child_return_preserves_raw_without_success_proof(self):
        """Retain both nonzero and invalid successful output before refusing the worker."""
        for value in (subprocess.CompletedProcess([], 1, b"partial", b"failed"), subprocess.CompletedProcess([], 0, b"invalid JSON", b"")):
            with tempfile.TemporaryDirectory() as tmp:
                unit = Path(tmp); (unit / "measurement_plan.json").write_bytes(common.encoded(self.plan))
                with patch.object(common.subprocess, "run", return_value=value):
                    with self.assertRaises((ValueError, json.JSONDecodeError)):common.before_compute(unit, self.plan)
                self.assertEqual((unit / "direct_common_correctness_validation_stdout.raw").read_bytes(), value.stdout)
                self.assertFalse((unit / common.PROOF_NAME).exists())
                with self.assertRaises(ValueError):common.before_compute(unit, self.plan)

    def test_timeout_preserves_partial_output_and_attempt(self):
        """A timed-out child never becomes a successful prerequisite or starts a repeat."""
        with patch.object(common.subprocess, "run", side_effect=subprocess.TimeoutExpired('synthetic', 1, output=b'partial', stderr=b'error')):
            with self.assertRaises(subprocess.TimeoutExpired):common.before_compute(self.unit, self.plan)
        self.assertEqual((self.unit / 'direct_common_correctness_validation_stdout.raw').read_bytes(), b'partial')
        self.assertIsNone(common.read(self.unit / 'direct_common_correctness_validation_return.json')[0]['returncode'])
        self.assertFalse((self.unit / common.PROOF_NAME).exists())

    def test_saved_proof_requires_same_actual_parent_and_precedes_numeric_work(self):
        """The saved result cannot claim another worker's or later validation."""
        with patch.object(common.subprocess, 'run', return_value=self.completed()):common.before_compute(self.unit, self.plan)
        row = {"iterations": 10, "copies": 1, "environments_per_copy": 128, "rollout_steps": 128,
            "training_transitions": 163840, "seconds": 1., "report_origin_iteration": 2}
        pairs = [{"round": number, "order": ["reference", "candidate"] if number % 2 else ["candidate", "reference"],
            **{arm: {**row, "report_origin_iteration": 2 + (number - 1) * 10} for arm in ("reference", "candidate")}}
            for number in range(1, 8)]
        result = {**common.result_binding(self.unit, self.plan), "rounds": pairs,
            'execution_identity': {'pid': os.getpid(), 'started_at_seconds': 2.}}
        with patch.object(common, 'request') as request:common.verify_saved(self.unit, self.plan, result)
        request.assert_called_once()
        for changed in ({'pid': os.getpid() + 1, 'started_at_seconds': 2.}, {'pid': os.getpid(), 'started_at_seconds': .5}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):common.verify_saved(self.unit, self.plan, {**result, 'execution_identity': changed})

    def test_default_direct_and_new_helper_closures_stay_separate(self):
        """Add only the new direct dependency while retaining default and A4500 closures."""
        historical = measurement.required_helpers(False, True)
        baseline = measurement.required_helpers(False, True, False, True)
        self.assertEqual(len(historical), 25)
        self.assertEqual(len(baseline), 26)
        self.assertEqual(set(measurement.required_helpers(False, True, True, True)),
            {*baseline, common.NAME, common.HISTORICAL_NAME})
        self.assertEqual(len(measurement.required_helpers(False, True, True, True)), 28)
        with self.assertRaises(ValueError):measurement.required_helpers(False, True, True)
        self.assertEqual(len(measurement.required_helpers()), 16)
        self.assertEqual(len(measurement.required_helpers(True, False, True)), 21)
        self.assertNotIn(common.NAME, measurement.required_helpers())
        self.assertNotIn(common.NAME, baseline)
        self.assertNotIn(common.NAME, measurement.required_helpers(True, False, True))

    def test_cli_requires_both_actual_records_and_refuses_final_or_prepared_overrides(self):
        """Preparation options cannot silently change the frozen protocol."""
        for extra in (["--common-correctness-record", "/missing"],
                ["--prepared-unit", "/missing", "--common-correctness-record", "/missing", "--common-correctness-review", "/missing"],
                ["--mode", "confirmation", "--final-confirmation", "--configured-isolation-profile", "/missing", "--common-correctness-record", "/missing", "--common-correctness-review", "/missing"]):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):measurement.parse_arguments(extra)

    def test_cold_actual_25_26_28_module_sets_and_unsupported_guard_without_backend(self):
        """Every admitted sibling set imports without numerical, network or device operations."""
        from test_configured_measurement import ISOLATED_IMPORT_PROBE
        rosters = (measurement.required_helpers(False, True),
            measurement.required_helpers(False, True, False, True), measurement.required_helpers(False, True, True, True))
        self.assertEqual([len(names) for names in rosters], [25, 26, 28])
        for names in rosters:
            with self.subTest(count=len(names)), tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary) / "data/unit/code"; folder.mkdir(parents=True)
                for name in names:shutil.copyfile(SCRIPTS / name, folder / name)
                outcome = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", ISOLATED_IMPORT_PROBE,
                    str(folder), json.dumps(names)], capture_output=True, text=True, timeout=30)
                self.assertEqual(outcome.returncode, 0, outcome.stderr)
                self.assertFalse(json.loads(outcome.stdout)["numerical_imports"])
        with self.assertRaises(ValueError):measurement.required_helpers(False, True, True, False)


class DirectActualSemanticTests(unittest.TestCase):
    """Check direct semantic joins with full unrelated raw consumers explicitly mocked."""
    def setUp(self):
        """Create synthetic private result/review files, never a campaign approval."""
        DirectCommonConfirmationTests.setUp(self)
        self.prior = self.unit / "synthetic_prior"
        self.prior.mkdir()
        self.plan["reference_configuration"]["execution"] = copy.deepcopy(common.DEFAULT_EXECUTION)
        self.plan["configuration"]["execution"] = copy.deepcopy(common.DEFAULT_EXECUTION)
        for key, value in {"campaign_run_dir": "/synthetic/run", "source_root": "/synthetic/source",
                "source_commit": "d"*40, "core": {"synthetic": True}, "paired_source": {"synthetic": True},
                "runtime_environment": {"JAX_ENABLE_X64": "true"}, "asset_record": {"synthetic": True},
                "direct_qualification_inputs": {"synthetic": True}}.items():
            self.plan[key] = value
        self.previous = copy.deepcopy(self.plan)
        self.previous.pop("common_correctness_inputs")
        self.previous["mode"] = "correctness"
        self.process = {"hostname": "serval02", "pid": 12, "process_start_time": "2", "parent_pid": 34}
        self.metadata = {"after": {"identity": self.process}}
        self.library = self.unit / "synthetic_gpu_library"
        self.library.write_bytes(b"synthetic library bytes")
        self.image = {"sha256": common.digest(self.library.read_bytes())}
        self.binding = {"gpu_uuid": common.CARD_ONE, "native": {"driver_library": str(self.library)}}
        loaded = {"pid": 12, "loaded_shared_libraries": [str(self.library)],
            "verified_package_libraries": {str(self.library): self.image}}
        self.result = {"measurement_mode": "correctness", "measurement_valid": False, "implementation_distinct": True,
            "execution_identity": {**self.process, "started_at_seconds": 1., "finished_at_seconds": 2.},
            "loaded_gpu_libraries": loaded, "direct_whole_work_memory": {"synthetic": True}}
        self.write(self.prior / "measurement_plan.json", self.previous)
        self.write(self.prior / "direct_process_metadata.json", self.metadata)
        self.write(self.prior / "direct_device_binding.json", self.binding)
        self.write(self.prior / "result.json", self.result)
        for filename in ("direct_terminal_process.json", "direct_process_wait_return.json",
                "direct_controller_spawn.json", "direct_controller_outcome.json", "direct_controller_output_hashes.json",
                "direct_backend_capture_begin.json", "direct_backend_capture.json"):
            self.write(self.prior / filename, {"synthetic": True})
        self.write(self.prior / "direct_backend_capture.json", {"post_installed_package": {"synthetic": True}})
        backend_record = self.unit / "synthetic_backend_preparation.json"
        self.write(backend_record, {"synthetic": True})
        self.plan["inputs"] = self.previous["inputs"] = {"direct_backend_image_record": common.reference(backend_record)}
        self.write(self.prior / "measurement_plan.json", self.previous)
        self.package = {"selected_python": "/synthetic/python", "site_packages": "/synthetic/site-packages",
            "image": {"synthetic": True}, "distribution": {"synthetic": True}, "producer_source_sha256": "f" * 64}
        self.reuse = {"source_contract": {"path": str(self.unit / "synthetic_qualification_source_contract.json"), "sha256": "e" * 64}}
        self.assets = {"synthetic": True}
        self.write(self.prior / "asset_receipt.json", self.assets)
        self.backend = {"gpu_libraries": {str(self.library): self.image}}
        self.write(self.prior / "installed_backend.json", self.backend)
        self.terminal = {name: common.reference(self.prior / filename) for name, filename in (
            ("terminal_process", "direct_terminal_process.json"), ("wait_record", "direct_process_wait_return.json"),
            ("controller_spawn", "direct_controller_spawn.json"), ("controller_outcome", "direct_controller_outcome.json"),
            ("controller_output_hashes", "direct_controller_output_hashes.json"))}
        expected = common.expected_review_identity(self.previous, self.result, self.binding, self.metadata, self.prior)
        self.raw = {**expected, **self.terminal, "schema_version": 1,
            "kind": "independent_actual_direct_common_correctness_raw_validation", "passed": True,
            "execution_identity": self.result["execution_identity"], "unchanged_staged_consumer_passed": True,
            "asset_receipt": common.reference(self.prior / "asset_receipt.json"),
            "installed_backend": common.reference(self.prior / "installed_backend.json"),
            "captured_backend": common.reference(self.prior / "direct_backend_capture.json"),
            "backend_expected": self.previous["direct_backend_image"]["expected"],
            "backend_preparation": self.previous["inputs"]["direct_backend_image_record"],
            "backend_capture_begin": common.reference(self.prior / "direct_backend_capture_begin.json"),
            "post_installed_package_sha256": common.digest(common.encoded({"synthetic": True})),
            "qualification_source_contract": self.reuse["source_contract"],
            "loaded_gpu_libraries_sha256": common.digest(common.encoded(loaded)),
            "direct_whole_work_memory_sha256": common.digest(common.encoded(self.result["direct_whole_work_memory"])),
            "admission_granted": False, "worker_admitted": False, **{key: 0 for key in common.ZERO_FIELDS}}
        self.review = {**expected, "schema_version": 1,
            "kind": "independent_actual_direct_common_configuration_correctness_review",
            "approved": True, "passed": True, "independent_of_producer": True,
            "reviewer_identity": "synthetic metadata fixture", "reviewed_at_seconds": 3., "unresolved_anomalies": [],
            "admission_granted": False, "worker_admitted": False, **{key: 0 for key in common.ZERO_FIELDS}}
        exact = lambda a, b: type(a) is type(b) and json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
        identity = SimpleNamespace(exact=exact, validate_plan=unittest.mock.Mock(),
            require_profile=unittest.mock.Mock(), receipt=lambda p: {"path": str(p),
                "sha256": common.digest(Path(p).read_bytes())})
        self.consumer = unittest.mock.Mock(side_effect=lambda plan, operation: {"proof": {
            "old_plan": common.reference(self.prior / "measurement_plan.json"),
            "old_inputs": plan["common_correctness_inputs"],
            "old_consumer_return_sha256": common.digest(common.encoded([self.previous, self.binding]))}})
        self.support = SimpleNamespace(core=lambda: identity,
            historical_support=lambda plan: SimpleNamespace(request=self.consumer),
            backend_support=lambda plan: SimpleNamespace(expected_for_plan=lambda p: self.package))
        self.validation = SimpleNamespace(correctness=unittest.mock.Mock(), configuration=unittest.mock.Mock(),
            execution_identity=unittest.mock.Mock())
        self.diagnostic = SimpleNamespace(default_configuration=lambda root: {
            "learner": copy.deepcopy(self.plan["reference_configuration"]["learner"]),
            "environment": copy.deepcopy(self.plan["reference_configuration"]["environment"])})
        self.training = SimpleNamespace(validate_gpu_library_receipt=unittest.mock.Mock(),
            verify_assets=unittest.mock.Mock(return_value=self.assets))
        self.refresh()

    def write(self, path, value):
        """Write only a private synthetic fixture file."""
        path.write_bytes(common.encoded(value))

    def refresh(self):
        """Rehash private changed fixture records without claiming an actual approval."""
        self.write(self.unit / "synthetic_raw.json", self.raw)
        self.review["raw_audit"] = common.reference(self.unit / "synthetic_raw.json")
        self.write(self.unit / "synthetic_review.json", self.review)
        self.plan["common_correctness_inputs"] = {"result": common.reference(self.prior / "result.json"),
            "review": common.reference(self.unit / "synthetic_review.json")}

    def validate(self):
        """Call the real semantic joins with unrelated full source/numerical gates mocked."""
        with patch.dict(sys.modules, {"direct_measurement": self.support,
                "post_comparison_validation": self.validation, "diagnose_gpu_execution": self.diagnostic,
                "train_optimized": self.training}), patch.object(common, "controller_evidence", return_value=self.terminal), \
                patch.object(common, "qualification_reuse_evidence", return_value=self.reuse):
            return common.validate_actual(self.plan, "f"*64)

    def test_complete_synthetic_direct_review_and_raw_joins(self):
        """Every actual-label join succeeds for matching explicit synthetic inputs."""
        value = self.validate()
        self.assertTrue(value["passed"])
        self.assertFalse(value["learning_equivalence_established"])
        self.consumer.assert_called_once()
        self.training.validate_gpu_library_receipt.assert_called_once()

    def test_coherently_rehashed_wrong_review_labels_types_and_unknowns(self):
        """Approved labels cannot disagree with real source, process, configuration or zero credit."""
        original = copy.deepcopy(self.review)
        for key, value in (("source_commit", "0"*40), ("copies", True), ("copy_seeds", [2]),
                ("gpu_uuid", "another device"), ("helper_commit", "0"*40), ("actual_returncode", False),
                ("scientific_credit", False), ("approved", 1), ("unresolved_anomalies", ["unknown"]),
                ("unknown_field", True)):
            self.review = {**copy.deepcopy(original), key: value}
            self.refresh()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate()

    def test_rehashed_raw_unknown_failed_or_changed_terminal_identity(self):
        """A real approval label cannot substitute for unchanged raw terminal gates."""
        original = copy.deepcopy(self.raw)
        for key, value in (("passed", False), ("controller_outcome", {"path": "/unknown", "sha256": "0"*64}),
                ("actual_returncode", 1), ("gpu_uuid", "another device"), ("unknown_field", True)):
            self.raw = {**copy.deepcopy(original), key: value}
            self.refresh()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate()

    def test_changed_runtime_or_loaded_library_bytes_are_refused(self):
        """The later plan and actual installed image must retain the checked bytes."""
        self.plan["runtime_environment"] = {"JAX_ENABLE_X64": "false"}
        with self.assertRaises(ValueError):
            self.validate()
        self.plan["runtime_environment"] = copy.deepcopy(self.previous["runtime_environment"])
        self.library.write_bytes(b"changed synthetic library")
        with self.assertRaises(ValueError):
            self.validate()


class QualificationReuseTests(unittest.TestCase):
    """Test exact source reuse with explicit synthetic files, never a campaign approval."""

    def setUp(self):
        """Copy small source files and create a private raw53 source contract."""
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name); self.qual = self.root / "qualification"; self.qual.mkdir()
        self.code = self.qual / "code"; self.code.mkdir()
        files = {}
        for name in measurement.required_helpers(False, True, False, True):
            shutil.copyfile(SCRIPTS / name, self.code / name)
            if name == "direct_measurement.py":
                # Before: current metadata dispatch plus original reader. After: synthetic old reader only.
                path = self.code / name
                raw = path.read_text()
                dispatch = '    if "common_correctness_inputs" in plan:\n        return historical_support(plan).request(plan, "qualification")\n'
                self.assertEqual(raw.count(dispatch), 1)
                path.write_text(raw.replace(dispatch, "", 1))
            files[common.PREFIX + name] = common.digest((self.code / name).read_bytes())
        numerical = self.root / "synthetic_numerical_source"; numerical.mkdir()
        core, paired = {}, {}
        for index in range(13):
            path = numerical / ("synthetic_source_" + str(index) + ".py")
            path.write_text("# synthetic source bytes; not executed\n")
            value = common.digest(path.read_bytes())
            if index < 11:core[path.name] = {"sha256": value}
            else:paired[path.name] = value
        self.plan = {"source_root": str(numerical), "source_commit": "d" * 40,
            "core": {"source_root": str(numerical), "source_commit": "d" * 40, "files": core},
            "paired_source": {"source_root": str(numerical), "source_commit": "d" * 40, "files": paired},
            "python": "/synthetic/python", "hardware_class": "h100", "runtime_environment": {"synthetic": True},
            "asset_record": {"synthetic": True}, "helpers": {"files": {common.PREFIX + name:
                common.digest((SCRIPTS / name).read_bytes()) for name in measurement.required_helpers(False, True, True, True)}},
            "inputs": {"direct_backend_image_record": {"path": str(self.root / "synthetic_backend.json"), "sha256": "e" * 64}}}
        self.qualified = copy.deepcopy(self.plan); self.qualified["helpers"] = {"files": files}
        self.write(self.qual / "measurement_plan.json", self.qualified)
        self.write(self.qual / "result.json", {"source_commit": "d" * 40})
        original = next(node for node in ast.parse((self.code / "direct_measurement.py").read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "verify_qualification_inputs")
        raw = copy.deepcopy(original); raw.name = "verify_unreviewed_actual_raw_qualification"
        raw.body = [node for index, node in enumerate(raw.body) if index not in (5, 50)]
        self.raw_path = self.root / "synthetic_raw53.py"; self.raw_path.write_text(ast.unparse(raw) + "\n")
        self.contract = {"kind": "exact_fresh_direct_qualification_raw_reader_two_review_statement_difference",
            "original_body_statement_count": 55, "retained_body_statement_count": 53,
            "removed_statement_indices": [5, 50], "removed_statements": [ast.unparse(original.body[index]) for index in (5, 50)],
            "retained_body_ast_sha256": common.digest("\n".join(ast.dump(node, include_attributes=False) for node in raw.body).encode()),
            "staged_source": self.file_ref(self.code / "direct_measurement.py"), "new_raw_reader": self.file_ref(self.raw_path)}
        self.contract_path = self.root / "synthetic_source_contract.json"; self.write(self.contract_path, self.contract)
        self.review_path = self.root / "synthetic_review.json"
        self.review = {"raw_validation_passed_before_review_existed": True, "placeholder_or_partial_approval_used": False,
            "result": common.reference(self.qual / "result.json"), "source_contract": common.reference(self.contract_path)}
        self.refresh()
        self.package = {"selected_python": "/synthetic/python", "site_packages": "/synthetic/site-packages",
            "image": {"synthetic": True}, "distribution": {"synthetic": True}, "producer_source_sha256": "e" * 64}
        exact = lambda a, b: type(a) is type(b) and common.encoded(a) == common.encoded(b)
        identity = SimpleNamespace(exact=exact, receipt=self.file_ref)
        self.support = SimpleNamespace(core=lambda: identity,
            backend_support=lambda plan: SimpleNamespace(expected_for_plan=lambda p: self.package))

    def write(self, path, value):
        """Write only private synthetic JSON."""
        path.write_bytes(common.encoded(value))

    def file_ref(self, path):
        """Hash small fixture source bytes without importing them."""
        return {"path": str(path), "sha256": common.digest(Path(path).read_bytes())}

    def refresh(self):
        """Keep a changed private fixture internally consistent for refusal controls."""
        self.write(self.contract_path, self.contract)
        self.review["source_contract"] = common.reference(self.contract_path)
        self.write(self.review_path, self.review)
        self.plan["direct_qualification_inputs"] = {"result": self.review["result"], "review": common.reference(self.review_path)}

    def test_complete_source_scope_retains_thirteen_files_and_original_execution(self):
        """Only unchanged code and a real-shaped raw53 contract can satisfy this source check."""
        proof = common.qualification_reuse_evidence(self.support, self.plan)
        self.assertEqual(len(proof["protected_numerical_files"]), 13)
        self.assertEqual(len(proof["protected_function_syntax"]), 12)
        self.assertTrue(proof["qualification_reader_metadata_extension"]["entire_original_body_retained"])
        self.assertFalse(proof["numerical_work_changed"])

    def test_old_raw50_or_extra_removed_gate_is_rejected_after_coherent_rehash(self):
        """A review label cannot weaken the current55/raw53 source contract."""
        original = copy.deepcopy(self.contract)
        for key, value in (("original_body_statement_count", 52), ("retained_body_statement_count", 50),
                ("removed_statement_indices", [5, 49, 50]), ("retained_body_ast_sha256", "0" * 64)):
            self.contract = {**copy.deepcopy(original), key: value}; self.refresh()
            with self.subTest(key=key), self.assertRaises(ValueError):common.qualification_reuse_evidence(self.support, self.plan)

    def test_changed_numerical_bytes_or_execution_function_are_rejected(self):
        """Rehashed helper receipts cannot authorize changed numerical execution."""
        source = Path(self.plan["core"]["source_root"]) / next(iter(self.plan["core"]["files"]))
        source.write_text("# changed synthetic numerical bytes\n")
        with self.assertRaises(ValueError):common.qualification_reuse_evidence(self.support, self.plan)
        source.write_text("# synthetic source bytes; not executed\n")
        old = self.code / "measure_selected.py"; raw = old.read_text().replace('def _compute(', 'def changed_compute(', 1); old.write_text(raw)
        self.qualified["helpers"]["files"][common.PREFIX + old.name] = common.digest(old.read_bytes())
        self.write(self.qual / "measurement_plan.json", self.qualified)
        with self.assertRaises(ValueError):common.qualification_reuse_evidence(self.support, self.plan)

    def test_placeholder_or_changed_prepared_backend_is_refused(self):
        """Reuse requires raw-first review and the same actual prepared package record."""
        self.review["placeholder_or_partial_approval_used"] = True; self.refresh()
        with self.assertRaises(ValueError):common.qualification_reuse_evidence(self.support, self.plan)
        self.review["placeholder_or_partial_approval_used"] = False; self.refresh()
        self.plan["inputs"]["direct_backend_image_record"] = {"path": "/different/prepin.json", "sha256": "e" * 64}
        with self.assertRaises(ValueError):common.qualification_reuse_evidence(self.support, self.plan)


class ExactHistoryNamespaceTests(unittest.TestCase):
    """Match all three per-unit maps to one pinned original declaration before any query."""

    def setUp(self):
        """Create only private ordinary metadata and directories; no Git repository is queried."""
        self.base = DirectCommonConfirmationTests(); self.base.setUp(); self.addCleanup(self.base.doCleanups)
        self.run = self.base.unit / "synthetic_campaign"; self.run.mkdir()
        self.sources = self.run / "synthetic_sources"; self.sources.mkdir()
        self.rows = [{"original_root": "/synthetic/original-source-" + str(index),
            "commit": str(index) * 40, "transport_repository": str(self.sources)} for index in range(8)]
        self.declarations_path = self.run / common.HISTORY_DECLARATIONS_RELATIVE_PATH
        self.declarations_path.parent.mkdir(parents=True)
        self.write(self.declarations_path, {"schema_version": 1, "kind": "exact_original_direct_history_preparation_inputs", "git_sources": self.rows})
        self.pin = common.digest(self.declarations_path.read_bytes())
        self.plans, self.units = {}, {}
        for name in ("current", "prior", "qualification"):
            unit = self.run / "data" / name; history = unit / "direct_history"; history.mkdir(parents=True)
            mapped = []
            for index, row in enumerate(self.rows):
                relative = "git_sources/original-source-" + str(index); (history / relative).mkdir(parents=True)
                mapped.append({"original_root": row["original_root"], "commit": row["commit"], "relative_path": relative})
            self.write(history / "evidence_manifest.json", {"schema_version": 1, "kind": "portable_raw_evidence", "files": [], "git_roots": mapped})
            self.write(history / "inputs.json", {"evidence_manifest": common.reference(history / "evidence_manifest.json")})
            plan = copy.deepcopy(self.base.plan)
            plan.update(campaign_run_dir=str(self.run), unit_id=name, source_root=str(self.sources), source_commit="d" * 40,
                core={"source_root": str(self.sources), "source_commit": "d" * 40},
                paired_source={"source_root": str(self.sources), "source_commit": "d" * 40})
            plan["helpers"]["source_root"] = str(self.sources)
            plan["direct_binding"]["frozen_inputs"] = common.reference(history / "inputs.json")
            self.write(unit / "result.json", {"synthetic": True})
            self.plans[name], self.units[name] = plan, unit
        self.plans["prior"]["direct_qualification_inputs"] = {"result": common.reference(self.units["qualification"] / "result.json")}
        self.plans["current"]["common_correctness_inputs"]["result"] = common.reference(self.units["prior"] / "result.json")
        for name in self.plans:self.write(self.units[name] / "measurement_plan.json", self.plans[name])

    def write(self, path, value):
        """Write only private synthetic metadata."""
        path.write_bytes(common.encoded(value))

    def evaluate(self):
        """Patch only the synthetic declaration hash and refuse every subprocess attempt."""
        with patch.object(common, "HISTORY_DECLARATIONS_SHA256", self.pin), \
                patch.object(common.subprocess, "Popen", side_effect=AssertionError("no query before exact namespace")):
            return common.declared_git_roots(self.plans["current"])

    def change_mapping(self, name, mutate):
        """Rehash all affected private references so only semantic mapping checks can reject."""
        history = self.units[name] / "direct_history"
        value = common.read(history / "evidence_manifest.json")[0]; mutate(value["git_roots"])
        self.write(history / "evidence_manifest.json", value)
        self.write(history / "inputs.json", {"evidence_manifest": common.reference(history / "evidence_manifest.json")})
        self.plans[name]["direct_binding"]["frozen_inputs"] = common.reference(history / "inputs.json")
        self.write(self.units[name] / "measurement_plan.json", self.plans[name])

    def test_all_current_prior_and_qualification_root_names_and_commits_match(self):
        roots = self.evaluate()
        for row in self.rows:self.assertIn(row["original_root"], roots)
        for unit in self.units.values():
            for index in range(8):self.assertIn(str(unit / "direct_history/git_sources" / ("original-source-" + str(index))), roots)

    def test_coherently_rehashed_unlisted_original_name_is_refused_in_each_unit(self):
        for name in self.units:
            original = copy.deepcopy(common.read(self.units[name] / "direct_history/evidence_manifest.json")[0]["git_roots"])
            self.change_mapping(name, lambda rows:rows[0].update(original_root="/unlisted/original"))
            with self.subTest(name=name), self.assertRaises(ValueError):self.evaluate()
            self.change_mapping(name, lambda rows:rows.__setitem__(slice(None), original))

    def test_wrong_commit_duplicate_and_traversal_are_refused_after_rehash(self):
        original = copy.deepcopy(common.read(self.units["current"] / "direct_history/evidence_manifest.json")[0]["git_roots"])
        for mutate in (lambda rows:rows[0].update(commit="f" * 40),
                lambda rows:rows[0].update(original_root=rows[1]["original_root"]),
                lambda rows:rows[0].update(relative_path="../outside")):
            self.change_mapping("current", mutate)
            with self.assertRaises(ValueError):self.evaluate()
            self.change_mapping("current", lambda rows:rows.__setitem__(slice(None), original))

    def test_changed_pinned_declaration_is_refused_before_mapping_or_query(self):
        self.write(self.declarations_path, {"schema_version": 1, "kind": "changed", "git_sources": self.rows})
        with self.assertRaises(ValueError):self.evaluate()

    def test_actual_published_declaration_contains_exactly_eight_original_sources(self):
        run = Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs") / (
            "2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1")
        expected, roots = common.pinned_history_declarations({"campaign_run_dir": str(run)})
        self.assertEqual(len(expected), 8)
        self.assertIn("/p/rlprojects/RND_worktrees/atari-cleanrl-source-d28e673", expected)
        self.assertIn("/p/rlprojects/RND_worktrees/merge-atari-cleanrl", roots)


class MetadataEventRefusalTests(unittest.TestCase):
    """Exercise audit predicates as data; perform no device, network or library operation."""

    def test_all_gpu_device_names_and_standing_lock_are_refused(self):
        for path in ("/dev/nvidia0", "/dev/dri/renderD128", b"/dev/dri/card0", "/tmp/sl5nw_serval02_card1.lock"):
            with self.subTest(path=path), self.assertRaises(RuntimeError):common.metadata_event_guard("open", (path, "rb"), set())

    def test_ctypes_process_handle_load_and_symbol_events_are_all_refused(self):
        for event, values in (("ctypes.dlopen", (None,)), ("ctypes.dlopen", ("libcuda.so",)),
                ("ctypes.dlsym", (None, "cuCtxGetCurrent")), ("socket.getaddrinfo", ("unqueried",))):
            with self.subTest(event=event), self.assertRaises(RuntimeError):common.metadata_event_guard(event, values, set())

    def test_missing_or_changed_stdlib_bootstrap_proof_is_rejected(self):
        value = common.stdlib_ctypes_bootstrap()
        self.assertEqual(common.validate_bootstrap_record(value), value)
        for wrong in (None, {**value, "python_api_target": "libcuda.so"},
                {**value, "later_library_loads_and_symbol_calls_refused": False},
                {**value, "bootstrap_scope": "general library exception"}):
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):common.validate_bootstrap_record(wrong)

    def test_only_exact_readonly_git_roots_can_pass_the_predicate(self):
        environment = common.metadata_environment({})
        common.metadata_event_guard("subprocess.Popen", ("git",
            ["git", "-C", "/known", "status", "--porcelain", "--untracked-files=all"], None, environment), {"/known"})
        for command in (["git", "-C", "/unlisted", "status"], ["git", "-C", "/known", "fetch"], ["nvidia-smi"]):
            with self.subTest(command=command), self.assertRaises(ValueError):
                common.metadata_event_guard("subprocess.Popen", (command[0], command, None, environment), {"/known"})

    def test_actual_python_or_git_executable_must_match_the_declared_arguments(self):
        """A different audit executable is refused even when its arguments are exact."""
        historical = SCRIPTS / common.HISTORICAL_NAME
        command = ["/usr/bin/python3", "-I", "-S", "-B", str(historical), "--read-only-stdin"]
        environment = common.metadata_environment({})
        with patch.object(common, "_ACTIVE_HISTORICAL_SOURCE_SHA256", common.digest(historical.read_bytes())):
            common.metadata_event_guard("subprocess.Popen", (command[0], command, None, environment), set())
            with self.assertRaises(ValueError):
                common.metadata_event_guard("subprocess.Popen", ("/another/executable", command, None, environment), set())
            with self.assertRaises(ValueError):
                common.metadata_event_guard("subprocess.Popen", (command[0], tuple(command), None, environment), set())
            with self.assertRaises(ValueError):
                common.metadata_event_guard("subprocess.Popen", (command[0], command, "/another/directory", environment), set())
        git = ["git", "-C", "/known", "rev-parse", "HEAD"]
        with self.assertRaises(ValueError):
            common.metadata_event_guard("subprocess.Popen", ("/another/executable", git, None, environment), {"/known"})

    def test_only_the_exact_actual_source_inspection_git_forms_are_allowed(self):
        """Source inspection needs no arbitrary options, external programs or output files."""
        valid = (["rev-parse", "HEAD"], ["rev-parse", "--show-toplevel"], ["symbolic-ref", "--quiet", "HEAD"],
            ["show", "a" * 40 + ":src/file.py"], ["cat-file", "blob", "b" * 40],
            ["ls-tree", "-rz", "a" * 40, "--", "src"],
            ["ls-tree", "-r", "--name-only", "a" * 40, "--", "src/"],
            ["status", "--porcelain", "--untracked-files=all"],
            ["status", "--porcelain", "--untracked-files=all", "--", "src/file.py"],
            ["status", "--porcelain", "--", "src/file.py"])
        for arguments in valid:
            with self.subTest(arguments=arguments):
                common.source_inspection_git(["git", "-C", "/known", *arguments], {"/known"})
        invalid = (["diff", "--ext-diff"], ["diff", "--output=/tmp/output"], ["status"],
            ["show", "a" * 40 + ":src/file.py", "--output=/tmp/output"], ["show", "--textconv", "a" * 40 + ":src/file.py"],
            ["show", "HEAD:src/file.py"], ["show", "a" * 40 + ":../private"],
            ["ls-tree", "-rz", "a" * 40, "--", ":(attr:secret)"],
            ["status", "--porcelain", "--untracked-files=all", "--", "../private"], ["cat-file", "--filters", "a" * 40])
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                common.source_inspection_git(["git", "-C", "/known", *arguments], {"/known"})

    def test_child_environment_removes_overrides_without_changing_the_parent(self):
        """Only isolated children receive the fixed Git/Python environment and two owned step fields."""
        original = {"SLURM_JOB_ID": "6783003", "SLURM_STEP_ID": "123", "PATH": "/another/bin",
            "GIT_CONFIG_PARAMETERS": "unsafe", "GIT_CONFIG_COUNT": "99", "GIT_EXTERNAL_DIFF": "/another/tool",
            "GIT_DIR": "/another/repository", "LD_PRELOAD": "/another/library.so", "PYTHONPATH": "/another/python",
            "JAX_ENABLE_X64": "true"}
        before = dict(original)
        environment = common.metadata_environment(original)
        self.assertEqual(original, before)
        self.assertEqual(environment["PATH"], "/usr/bin")
        self.assertEqual(environment["SLURM_JOB_ID"], "6783003")
        self.assertEqual(environment["SLURM_STEP_ID"], "123")
        self.assertFalse(set(original).intersection(environment) - {"PATH", "SLURM_JOB_ID", "SLURM_STEP_ID", "GIT_CONFIG_COUNT"})
        common.require_metadata_environment(environment)
        with patch.dict(os.environ, environment, clear=True):
            common.require_metadata_environment(None)
        for changed in ({**environment, "PATH": "/another/bin"}, {**environment, "LD_PRELOAD": "/another/library.so"},
                {**environment, "GIT_CONFIG_VALUE_0": "/another/program"}, {**environment, "GIT_ALLOW_PROTOCOL": "file"}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                common.require_metadata_environment(changed)

    def test_fresh_interpreter_real_main_import_order_bootstraps_only_stdlib_then_refuses_later_ctypes(self):
        """Start with no ctypes; reach real later imports and a private missing-result stop."""
        fixture = DirectCommonConfirmationTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        plan = copy.deepcopy(fixture.plan)
        plan["campaign_run_dir"] = "/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/" + (
            "2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1")
        probe = r'''
import builtins,importlib.util,io,json,pathlib,sys
assert "ctypes" not in sys.modules
initial=[]
def observe(event,args):
    if event in {"ctypes.dlopen","ctypes.dlsym"}:initial.append((event,args[0]))
sys.addaudithook(observe)
source=pathlib.Path(sys.argv[1])
spec=importlib.util.spec_from_file_location("cold_direct_common",source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
bootstrap=module.stdlib_ctypes_bootstrap()
assert initial and all(event=="ctypes.dlopen" and target is None for event,target in initial)
original_import=builtins.__import__
def source_import(name,*args,**kwargs):
    value=original_import(name,*args,**kwargs)
    if name=="direct_measurement":value.core().validate_plan=lambda plan:None
    return value
builtins.__import__=source_import
# Only raw namespace/plan validation is substituted. Real main, audit setup,
# validate_actual and all of its later metadata imports run in their actual order.
module.declared_git_roots=lambda plan:set()
raw=sys.stdin.buffer.read()
sys.stdin=io.TextIOWrapper(io.BytesIO(raw),encoding="utf-8")
sys.argv=[str(source),"--plan-stdin"]
try:module.main()
except FileNotFoundError:pass
else:raise AssertionError("private absent correctness result should end this import probe")
assert all(name in sys.modules for name in ("direct_measurement","post_comparison_validation","diagnose_gpu_execution","train_optimized"))
assert not (module.NUMERICAL & set(sys.modules))
ctypes=sys.modules["ctypes"]
for operation in (lambda:ctypes.CDLL(None),lambda:ctypes.CDLL("libcuda.so"),lambda:ctypes.pythonapi["Py_GetVersion"]):
    try:operation()
    except RuntimeError:pass
    else:raise AssertionError("every later ctypes load or symbol operation must be refused")
print(json.dumps({"fresh_interpreter":True,"stdlib_ctypes_bootstrap":bootstrap,"later_ctypes_operations_refused":3,
    "actual_h100_result_read":False,"numerical_imports":0,"gpu_operations":0,"qualification_established":False}))
'''
        result = subprocess.run(["/usr/bin/python3", "-I", "-S", "-B", "-c", probe,
            str(SCRIPTS / common.NAME)], input=common.encoded(plan), capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        value = json.loads(result.stdout)
        self.assertTrue(value["fresh_interpreter"])
        self.assertEqual(value["later_ctypes_operations_refused"], 3)
        self.assertFalse(value["actual_h100_result_read"])


class DirectPairAndContextTests(unittest.TestCase):
    """Check complete timing order and rejection of login-node raw recomputation."""
    def test_alternating_complete_pairs_and_changed_orders_or_work(self):
        """Keep actual compile-warm origins while requiring seven matched ten-update advances."""
        plan = {"configuration": {"learner": {"n_copies": 1}}}
        row = {"iterations": 10, "copies": 1, "environments_per_copy": 128, "rollout_steps": 128,
            "training_transitions": 163840, "seconds": 1., "report_origin_iteration": 2}
        result = {"rounds": [{"round": number,
            "order": ["reference", "candidate"] if number % 2 else ["candidate", "reference"],
            **{arm: {**row, "report_origin_iteration": 2 + (number - 1) * 10} for arm in ("reference", "candidate")}}
            for number in range(1, 8)]}
        common.validate_pair_protocol(plan, result)
        for mutate in (lambda v: v["rounds"][1].update(order=["reference", "candidate"]),
                lambda v: v["rounds"][0]["candidate"].update(iterations=1),
                lambda v: v["rounds"][0]["reference"].update(training_transitions=True),
                lambda v: v["rounds"][3]["reference"].update(report_origin_iteration=31)):
            value = copy.deepcopy(result)
            mutate(value)
            with self.assertRaises(ValueError):
                common.validate_pair_protocol(plan, value)

    def test_actual_current_owned_cpu_context_is_metadata_only(self):
        """The current numeric CPU step records its own context without a scheduler query."""
        run = Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs") / (
            "2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1")
        value = common.metadata_compute_context({"campaign_run_dir": str(run)})
        self.assertEqual(value["job_id"], "6783003")
        self.assertTrue(value["step_id"].isdecimal())

    def test_login_node_without_compute_allocation_is_refused(self):
        """An ordinary login process cannot recompute raw scientific prerequisites."""
        import socket
        with patch.object(socket, "gethostname", return_value="portal05"), patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "never a login node"):
                common.metadata_compute_context({"campaign_run_dir": "/unused"})


if __name__ == "__main__":
    unittest.main()
