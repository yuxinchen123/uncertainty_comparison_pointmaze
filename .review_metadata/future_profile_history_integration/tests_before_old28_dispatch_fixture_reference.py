"""Check synthetic direct-profile metadata without reading campaign science or devices."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import direct_profile_prerequisites as profile
import direct_profile_saved_consumers as saved
import direct_backend_image as backend
import profile_selected as selected
import direct_measurement as direct
import measure_selected as measurement
import profile_metadata_runtime as metadata_runtime
from test_direct_backend_image import DirectBackendImageTests


class DirectProfileMetadataTests(unittest.TestCase):
    """Keep missing real results unopened during metadata preparation and refusal."""
    def setUp(self):
        """Build only synthetic plan/review metadata and deliberately absent scientific results."""
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "campaign"; self.unit = self.run / "data/profile"
        self.unit.mkdir(parents=True)
        previous_unit = self.run / "data/confirmation"; previous_unit.mkdir()
        self.configuration = {"learner": {"n_copies": 1}, "environment": {"cache_room_backgrounds": True,
            "render_final_two_only": True}, "execution": {"iterations_per_call": 1, "report_readout": "none"}}
        expected = {"selected_python": {"path": "/synthetic/python"},
            "site_packages": "/synthetic/site", "image": {"sha256": "1" * 64}, "distribution": {"name": "synthetic"}}
        self.expected = self.write(self.unit / "direct_backend_image_expected.json", expected)
        previous_expected = self.write(previous_unit / "direct_backend_image_expected.json", expected)
        self.inputs = {name: {"path": str(self.run / (name + ".json")), "sha256": str(number) * 64}
            for name, number in (("correctness", 2), ("correctness_review", 3), ("qualification", 4), ("qualification_review", 5))}
        self.inputs["confirmation"] = {"path": str(previous_unit / "result.json"), "sha256": "6" * 64}
        self.previous = {"kind": "selected_configuration_measurement", "mode": "confirmation", "hardware_class": "h100",
            "gpu_uuid": profile.CARD_ONE, "direct_binding": {"profile": {"synthetic_only": True}},
            "rounds": 7, "iterations": 10, "minimum_updates_per_round": 10,
            "helpers": {"source_commit": "a" * 40, "files": {profile.PREFIX + name: hashlib.sha256((SCRIPTS / name).read_bytes()).hexdigest()
                for name in measurement.required_helpers(False, True, True, True) if name != "accepted_evidence_snapshot.py"}},
            "source_root": "/synthetic/frozen", "source_commit": "d" * 40, "core": {}, "paired_source": {},
            "python": "/synthetic/python", "campaign_run_dir": str(self.run), "unit_id": "confirmation",
            "configuration": self.configuration, "reference_configuration": {"synthetic_reference": True},
            "runtime_environment": {"synthetic_runtime": True}, "asset_record": {"synthetic_assets": True}, "copy_seeds": [1],
            "common_correctness_inputs": {"result": self.inputs["correctness"], "review": self.inputs["correctness_review"]},
            "direct_historical_consumer_protocol": ast.literal_eval(next(node.value for node in ast.parse(
                (SCRIPTS/'direct_common_confirmation.py').read_bytes()).body if isinstance(node,ast.Assign)
                and any(isinstance(target,ast.Name) and target.id=='HISTORICAL_PROTOCOL' for target in node.targets))),
            "direct_qualification_inputs": {"result": self.inputs["qualification"], "review": self.inputs["qualification_review"]},
            "direct_backend_image": {"expected": previous_expected}}
        previous_code = previous_unit / "code"; previous_code.mkdir()
        for relative in self.previous["helpers"]["files"]:
            (previous_code / Path(relative).name).write_bytes((SCRIPTS / Path(relative).name).read_bytes())
        self.inputs["confirmation_plan"] = self.write(previous_unit / "measurement_plan.json", self.previous)
        self.review = {"schema_version": 1, "kind": "independent_actual_direct_common_configuration_confirmation_review",
            "approved": True, "passed": True, "independent_of_producer": True, "unresolved_anomalies": [],
            "reviewer_identity": "synthetic reviewer; no actual review or credit", "decision": "accepted",
            "raw_window_recomputation_passed": True, "paired_rounds": 7, "updates_per_arm_per_round": 10,
            "reviewed_at_seconds": 5., "admission_granted": False, "worker_admitted": False,
            "result": self.inputs["confirmation"], "plan": self.inputs["confirmation_plan"],
            "source_commit": self.previous["source_commit"], "helper_commit": self.previous["helpers"]["source_commit"],
            "configuration": self.configuration, "reference_configuration": self.previous["reference_configuration"],
            "copies": 1, "copy_seeds": [1], "gpu_uuid": profile.CARD_ONE,
            "raw_audit": {"path": str(self.root / "absent_scientific_raw_audit.json"), "sha256": "8" * 64}}
        self.review_path = self.root / "synthetic_independent_review.json"
        self.inputs["confirmation_review"] = self.write(self.review_path, self.review)
        self.plan = {**{key: copy.deepcopy(self.previous[key]) for key in ("hardware_class", "gpu_uuid", "direct_binding",
            "source_root", "source_commit", "core", "paired_source", "python", "campaign_run_dir", "configuration",
            "runtime_environment", "asset_record", "copy_seeds", "direct_qualification_inputs")},
            "kind": "selected_configuration_profile", "unit_id": "profile", "profile_updates": 1,
            "direct_profile_protocol": profile.PROTOCOL, "capture_process_protocol": profile.PROCESS_PROTOCOL,
            "helpers": {"files": {profile.PREFIX + name: hashlib.sha256((SCRIPTS / name).read_bytes()).hexdigest()
                for name in (Path(relative).name for relative in profile.helper_roster())}},
            "direct_backend_image": {"schema_version": 2, "capability": backend.CAPABILITY, "expected": self.expected},
            "direct_profile_saved_consumer_protocol": copy.deepcopy(saved.PROTOCOL),
            "direct_profile_metadata_protocol": copy.deepcopy(metadata_runtime.PROTOCOL),
            "inputs": {"accepted_confirmation": self.inputs["confirmation"]},
            "accepted_baseline_specification": {"correctness_record": self.inputs["correctness"]["path"],
                "confirmation_record": self.inputs["confirmation"]["path"]}, "direct_profile_prerequisites": self.inputs}

    def write(self, path, value):
        """Write explicit synthetic metadata, never a campaign artifact."""
        path.write_bytes(profile.encoded(value))
        return profile.reference(path)

    def test_preparation_reads_only_plans_reviews_and_package_metadata(self):
        """Absent correctness, confirmation, qualification and raw audit files remain unopened."""
        self.assertEqual(profile.validate_metadata(self.plan), self.previous)
        self.assertFalse(Path(self.inputs["confirmation"]["path"]).exists())
        self.assertFalse(Path(self.review["raw_audit"]["path"]).exists())
        plan = copy.deepcopy(self.plan); plan.pop("direct_profile_prerequisites")
        profile.configure(plan, plan["accepted_baseline_specification"], self.review_path,
            self.inputs["qualification"]["path"], self.inputs["qualification_review"]["path"])
        self.assertEqual(plan["direct_profile_prerequisites"], self.inputs)

    def test_helper_union_includes_backend_and_separate_profile_observers(self):
        """The unchanged launcher remains closed while the new metadata adapter declares its extra helper."""
        names = selected.required_helpers(False, True, direct_profile=True)
        self.assertEqual(len(names), 34)
        self.assertIn("direct_backend_image.py", names)
        self.assertIn("direct_profile_prerequisites.py", names)
        self.assertIn("nsight_process_identity.py", names)
        self.assertNotIn("direct_common_confirmation.py", names)
        for arguments in ((True, True), (False, True, True), (False, True, False, True)):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):selected.required_helpers(*arguments)
        self.assertEqual(len(selected.required_helpers()), 17)
        self.assertIn("nsight_process_identity.py", selected.required_helpers(True, False, True, True))
        self.assertEqual(len(profile.helper_roster()), 34)
        self.assertEqual(profile.helper_roster(), set(self.plan["helpers"]["files"]))
        changed = copy.deepcopy(self.plan); changed["helpers"]["files"].pop(profile.PREFIX + "ideas.py")
        with self.assertRaises(ValueError):profile.contract(changed)
        changed = copy.deepcopy(self.plan); changed["helpers"]["files"][profile.PREFIX + "direct_common_confirmation.py"] = "a" * 64
        with self.assertRaises(ValueError):profile.contract(changed)

    def test_current_profile_does_not_use_the_a4500_common_dispatcher(self):
        """The direct metadata adapter reads no accepted timing file in either dispatcher."""
        self.assertFalse(selected.accepted_confirmation_common_reader(self.plan))
        self.assertFalse(selected.owned_processes(self.plan))

    def test_wrong_target_source_config_seeds_and_review_are_rejected(self):
        """No default or configured profile and no different common result can substitute."""
        for key, value in (("gpu_uuid", "GPU-other"), ("hardware_class", "jaguar03_rtx_a4500"),
            ("source_commit", "b" * 40), ("copy_seeds", [True]), ("capture_process_protocol", "other"),
            ("configuration", {**self.configuration, "learner": {"n_copies": 2}})):
            changed = copy.deepcopy(self.plan); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):profile.validate_metadata(changed)
        for key in ("configured_isolation", "completed_study", "common_correctness_inputs", "scaling_protocol", "device_memory_limit_protocol"):
            changed = copy.deepcopy(self.plan); changed[key] = {}
            with self.subTest(key=key), self.assertRaises(ValueError):profile.contract(changed)

    def test_changed_precomputed_review_and_package_bytes_are_rejected(self):
        """Review content and installed package description remain exact prerequisites."""
        for key, value in (("approved", False), ("paired_rounds", True), ("updates_per_arm_per_round", 20),
            ("raw_window_recomputation_passed", False), ("unresolved_anomalies", ["unreviewed"]), ("worker_admitted", True)):
            review = {**self.review, key: value}
            changed = copy.deepcopy(self.plan)
            changed["direct_profile_prerequisites"]["confirmation_review"] = self.write(self.review_path, review)
            with self.subTest(key=key), self.assertRaises(ValueError):profile.validate_metadata(changed)
        self.write(self.review_path, self.review)
        changed = copy.deepcopy(self.plan)
        changed["direct_backend_image"]["expected"] = self.write(self.unit / "direct_backend_image_expected.json",
            {**profile.checked_reference(self.expected), "image": {"sha256": "f" * 64}})
        with self.assertRaises(ValueError):profile.validate_metadata(changed)

    def test_actual_context_failure_precedes_child_creation_and_scientific_reads(self):
        """A login-side request cannot start a child or open saved scientific results."""
        with patch.object(profile, "compute_context", side_effect=ValueError("synthetic login refusal")), \
            patch.object(profile, "validate_metadata") as metadata, patch.object(profile.subprocess, "run") as child:
            with self.assertRaisesRegex(ValueError, "login refusal"):profile.before_compute(self.unit, self.plan)
            metadata.assert_not_called(); child.assert_not_called()
        with patch.object(profile, "compute_context", side_effect=ValueError("synthetic login refusal")), \
            patch.object(profile, "read") as reader:
            with self.assertRaises(ValueError):profile.checked_actual_reference(self.plan, self.inputs["confirmation"])
            reader.assert_not_called()

    def test_reserved_metadata_paths_are_checked_before_they_are_opened(self):
        """A mistaken plan or backend reference cannot cause preparation to open result.json."""
        forbidden = self.inputs["confirmation"]["path"]
        original = profile.read
        opened = []
        def checked_read(path, *arguments):
            """Refuse any attempted scientific read in this synthetic preparation fixture."""
            self.assertNotEqual(str(path), forbidden)
            opened.append(str(path))
            return original(path, *arguments)
        changed = copy.deepcopy(self.plan)
        changed["direct_profile_prerequisites"]["confirmation_plan"] = self.inputs["confirmation"]
        with patch.object(profile, "read", side_effect=checked_read):
            with self.assertRaises(ValueError):profile.validate_metadata(changed)
        review = {**self.review, "plan": self.inputs["confirmation"]}
        self.write(self.review_path, review)
        with patch.object(profile, "read", side_effect=checked_read):
            with self.assertRaises(ValueError):profile.configure(self.plan, self.plan["accepted_baseline_specification"],
                self.review_path, self.inputs["qualification"]["path"], self.inputs["qualification_review"]["path"])
        self.write(self.review_path, self.review)
        changed = copy.deepcopy(self.plan)
        changed["direct_backend_image"]["expected"] = self.inputs["confirmation"]
        with patch.object(profile, "read", side_effect=checked_read):
            with self.assertRaises(ValueError):profile.validate_metadata(changed)
        self.assertNotIn(forbidden, opened)

    def test_profile_binding_builder_never_invokes_raw_history_during_preparation(self):
        """The separate builder uses only plan metadata and the prepared backend sidecar."""
        history = self.unit / "direct_history"; history.mkdir()
        inputs = history / "inputs.json"; inputs.write_text('{"synthetic_history_metadata":true}')
        validated = []
        identity = SimpleNamespace(require_profile=lambda actual: None, CAPABILITY="synthetic metadata only",
            pending_identity=lambda: {"gpu_uuid": None}, validate_plan=lambda *arguments: validated.append(arguments))
        prepared = []
        module = SimpleNamespace(core=lambda: identity,
            adjacent=lambda name: SimpleNamespace(configure_plan=lambda *arguments: prepared.append(arguments)))
        with patch.object(profile.importlib, "import_module", return_value=module):
            result = profile.configure_binding(copy.deepcopy(self.plan), self.plan["direct_binding"]["profile"], inputs,
                Path(self.expected["path"]))
        self.assertEqual(len(validated), 1)
        self.assertEqual(len(prepared), 1)
        self.assertEqual(result["direct_binding"]["frozen_inputs"], profile.reference(inputs))
        tree = ast.parse((SCRIPTS / "profile_selected.py").read_text())
        prepare = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "prepare")
        blocks = [ast.Module(body=node.body, type_ignores=[]) for node in ast.walk(prepare)
            if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "direct"]
        calls = {node.func.attr for block in blocks for node in ast.walk(block)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "support"}
        self.assertNotIn("configure_plan", calls)

    def test_current_history_context_failure_precedes_import_or_raw_recomputation(self):
        """The unchanged scientific history reader is reached only on actual owned compute."""
        with patch.object(profile, "compute_context", side_effect=ValueError("synthetic unallocated process")), \
            patch.object(profile.importlib, "import_module") as imports:
            with self.assertRaises(ValueError):profile.validate_current_history(self.plan)
            imports.assert_not_called()

    def test_fresh_no_site_child_without_owned_context_refuses_missing_real_results(self):
        """The actual isolated entrypoint rejects an unallocated process before its absent result paths."""
        environment = {key: value for key, value in os.environ.items() if not key.startswith("SLURM_")}
        parent=saved.process(os.getpid())
        unsigned={"schema_version":2,"plan":self.plan,"parent":parent,
            "metadata_origin":metadata_runtime.origin(self.plan,parent)}
        envelope={**unsigned,"challenge":profile.digest(profile.encoded(unsigned))}
        completed = subprocess.run(["/usr/bin/python3", "-I", "-S", "-B", str(SCRIPTS / profile.NAME), "--plan-stdin"],
            input=profile.encoded(envelope), capture_output=True, env=environment, timeout=30, check=False)
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(completed.stdout, b"")
        self.assertIn(b"owned compute, never a login node", completed.stderr)
        self.assertNotIn(b"FileNotFoundError", completed.stderr)

    def test_parent_checks_actual_child_return_and_preserves_failure_bytes(self):
        """A failed child creates only exact failed stdout/stderr/return evidence."""
        completed = SimpleNamespace(returncode=1, stdout=b"failed metadata child", stderr=b"synthetic error")
        with patch.object(profile, "compute_context", return_value={}), patch.object(profile, "validate_metadata"), \
            patch.object(profile,"require_owned_stage",return_value=self.unit), \
            patch.object(profile, "saved_consumers", return_value=saved), \
            patch.object(saved,"metadata_runtime",return_value=metadata_runtime), \
            patch.object(metadata_runtime.MetadataScope,"run",return_value=completed):
            with self.assertRaises(ValueError):profile.before_compute(self.unit, self.plan)
        self.assertEqual((self.unit / "direct_profile_baseline_stdout.raw").read_bytes(), completed.stdout)
        self.assertEqual(profile.read(self.unit / "direct_profile_baseline_return.json")[0]["returncode"], 1)
        self.assertFalse((self.unit / profile.PROOF_NAME).exists())

    def test_parent_rejects_false_child_proof_even_with_zero_return(self):
        """A child claiming another plan cannot publish an accepted baseline proof."""
        completed = SimpleNamespace(returncode=0, stdout=b'{"passed":true}', stderr=b"")
        with patch.object(profile, "compute_context", return_value={}), patch.object(profile, "validate_metadata"), \
            patch.object(profile,"require_owned_stage",return_value=self.unit), \
            patch.object(profile, "saved_consumers", return_value=saved), \
            patch.object(saved,"metadata_runtime",return_value=metadata_runtime), \
            patch.object(metadata_runtime.MetadataScope,"run",return_value=completed):
            with self.assertRaises(ValueError):profile.before_compute(self.unit, self.plan)
        self.assertFalse((self.unit / profile.PROOF_NAME).exists())

    def test_saved_old_schema_cannot_bypass_current_metadata_closure(self):
        """A historical success label is refused before any absent result or outer proof is read."""
        for version in (None,1,True):
            proof={"kind":"actual_direct_profile_baseline_validation","passed":True}
            if version is not None:proof['schema_version']=version
            with self.subTest(version=version),patch.object(profile,'checked_reference',side_effect=AssertionError('saved payload read')):
                with self.assertRaisesRegex(ValueError,'old or changed baseline'):
                    profile.validate_completed_metadata_phase(self.plan,proof,{})

    def test_new29_history_pointer_cannot_open_a_result_path(self):
        """Reserved own-unit pointer shape precedes a corresponding history file read."""
        previous=copy.deepcopy(self.previous);previous['accepted_evidence_snapshot_protocol']={'synthetic_only':True}
        changed=copy.deepcopy(self.plan);changed['direct_binding']['frozen_inputs']=self.inputs['confirmation']
        with patch.object(profile,'compute_context',return_value={}),patch.object(profile,'validate_metadata',return_value=previous),\
                patch.object(profile,'checked_reference',side_effect=AssertionError('result file read')):
            with self.assertRaisesRegex(ValueError,'reserved unit path'):profile.validate_current_history(changed)

    def test_explicit_old28_profile_still_reaches_original_raw_gate(self):
        """The old protocol returns only a dispatch sentinel; it cannot manufacture history success."""
        with patch.object(profile,'compute_context',return_value={'synthetic_context_premise':True}),\
                patch.object(profile,'validate_current_history',side_effect=AssertionError('old raw gate was replaced')):
            self.assertIsNone(profile.validate_profile_history(self.plan))
        binding=direct.core();fake=SimpleNamespace(adjacent=lambda name:SimpleNamespace(validate_profile_history=lambda actual:None))
        with patch.object(binding,'source_identity',return_value={}),patch.object(binding,'validate_plan',return_value=self.plan['direct_binding']),\
                patch.object(binding,'adjacent',return_value=fake),patch.object(binding,'read_receipt',side_effect=ValueError('original raw gate reached')):
            with self.assertRaisesRegex(ValueError,'original raw gate reached'):binding.validate_history(self.plan)

    def test_post_core_backend_completion_preserves_the_raw_numerical_capture(self):
        """The separate wrapper adds actual backend evidence without editing the core result."""
        code = self.unit / "code"; code.mkdir()
        source = code / "direct_measurement.py"; source.write_text("synthetic metadata helper; never executed\n")
        capture = {"capture_completed": True, "execution_identity": {"pid": 1, "started_at_seconds": 1., "finished_at_seconds": 2.}}
        self.write(self.unit / "capture_result.json", capture)
        raw = (self.unit / "capture_result.json").read_bytes()
        def completed(directory, plan, actual):
            """Stand in only for the separately tested synthetic mapped-backend producer."""
            self.assertEqual(actual, capture)
            self.write(directory / "direct_backend_capture.json", {"synthetic_backend_complete": True})
        module = SimpleNamespace(__file__=str(source), capture_backend_completion=completed)
        with patch.object(profile, "compute_context", return_value={}), patch.object(profile.importlib, "import_module", return_value=module):
            receipt = profile.complete_capture(self.unit, self.plan)
        self.assertEqual((self.unit / "capture_result.json").read_bytes(), raw)
        self.assertTrue(receipt["original_capture_bytes_unchanged"])
        self.assertTrue((self.unit / "direct_profile_backend_completion.json").is_file())
        self.write(self.unit / "measurement_plan.json", self.plan)
        supervisor = self.unit / "supervisor"; supervisor.mkdir()
        self.write(supervisor / "direct_process_metadata.json", {"after": {"identity": {"pid": 42}}})
        proof = {"kind": "actual_direct_profile_baseline_validation", "passed": True,
            "plan_sha256": profile.digest((self.unit / "measurement_plan.json").read_bytes()),
            "prerequisites": self.inputs, "parent_pid": 42, "finished_at_seconds": .5,
            "exact_saved_consumer_bundle": {"synthetic_unrelated_completed_reader_prerequisite": True},
            "combination_metadata_dispatch_source_proof": {"all_other_statement_syntax_unchanged": True}}
        self.write(self.unit / profile.PROOF_NAME, proof)
        self.write(self.unit / "direct_profile_metadata_outer_scope.json", {"synthetic_upstream_closed_scope_fixture":True})
        output = profile.encoded(proof)
        (self.unit / "direct_profile_baseline_stdout.raw").write_bytes(output)
        (self.unit / "direct_profile_baseline_stderr.raw").write_bytes(b"")
        self.write(self.unit / "direct_profile_baseline_return.json", {"command": ["/usr/bin/python3", "-I", "-S", "-B",
            str(self.unit / "code" / profile.NAME), "--plan-stdin"], "returncode": 0, "error": None,
            "stdout_sha256": profile.digest(output), "stderr_sha256": profile.digest(b"")})
        # This wrapper-only fixture supplies a successful upstream proof; the new reader has separate source/proof tests.
        with patch.object(profile, "compute_context", return_value={}), patch.object(profile, "saved_consumers", return_value=saved), \
                patch.object(profile,"validate_completed_metadata_phase",return_value=proof), \
                patch.object(saved, "validate_bundle", return_value=proof["exact_saved_consumer_bundle"]):
            binding = profile.completed_capture_binding(self.unit, self.plan, capture)
        self.assertEqual(binding["direct_profile_backend_completion"], profile.reference(self.unit / "direct_profile_backend_completion.json"))
        with patch.object(profile, "compute_context", return_value={}):
            with self.assertRaises(ValueError):profile.completed_capture_binding(self.unit, self.plan, {**capture, "extra": True})

    def test_all_120_original_cases_use_the_actual_historical_case_validator(self):
        """The adapter requires both copy counts on both GPU types for every original idea."""
        registry = {"initial_matched_copy_counts": [1, 2], "reference_commit": "d" * 40,
            "initial_ideas": [{"name": "synthetic idea " + str(number), "results": {context: [
                {"copies": copies, "result_path": "/absent/synthetic/" + str(number) + "/" + context + "/" + str(copies)}
                for copies in (1, 2)] for context in ("h100", "jaguar03_rtx_a4500")}} for number in range(30)]}
        calls = []
        validator = SimpleNamespace(CONTEXTS=("h100", "jaguar03_rtx_a4500"),
            validate_cohort_case=lambda *arguments: calls.append(arguments))
        self.assertEqual(profile.validate_initial_cases(registry, validator, None), 120)
        self.assertEqual(len(calls), 120)
        registry["initial_ideas"][0]["results"]["h100"].pop()
        with self.assertRaises(ValueError):profile.validate_initial_cases(registry, validator, None)

    def test_new_and_old_registered_profiles_cannot_bypass_current_admission(self):
        """Current preparation/execution/readers refuse before any old process assumption."""
        with self.assertRaisesRegex(ValueError, "remain disabled"):profile.require_admission()
        with self.assertRaisesRegex(ValueError, "remain disabled"):selected.prepare(SimpleNamespace(
            configured_isolation_profile=None, direct_binding_profile=Path("/synthetic/profile.json")))
        with self.assertRaisesRegex(ValueError, "remain disabled"):selected.verify_prepared(self.unit, self.plan)
        with self.assertRaisesRegex(ValueError, "remain disabled"):selected.capture_and_summarize(self.unit, self.plan, 7)
        self.write(self.unit / "measurement_plan.json", self.plan)
        with self.assertRaisesRegex(ValueError, "remain disabled"):direct.verify_saved_measurement(self.unit, {})
        legacy = {**self.plan}; legacy.pop("direct_profile_protocol")
        self.write(self.unit / "measurement_plan.json", legacy)
        with self.assertRaisesRegex(ValueError, "explicit read-only"):direct.verify_saved_measurement(self.unit, {})

    def test_new_cli_requires_backend_and_genuine_common_review_inputs(self):
        """The direct-specific inputs cannot be supplied to unrelated operations."""
        with self.assertRaises(SystemExit):selected.parse_arguments(["--prepared-unit", str(self.unit),
            "--direct-backend-image-record", str(self.root / "fake.json")])
        with self.assertRaises(SystemExit):selected.parse_arguments(["--direct-confirmation-review", str(self.review_path)])

    def test_new_publication_preserves_prior_attempts_and_rejects_nonfinite_json(self):
        """No metadata utility can overwrite old evidence or accept duplicate keys."""
        path = self.root / "published.json"; profile.publish(path, b'{"value":1}')
        with self.assertRaises(FileExistsError):profile.publish(path, b'{"value":2}')
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}'):
            path.write_bytes(raw)
            with self.assertRaises(ValueError):profile.read(path)


class ProfileHistoryIntegrationTests(unittest.TestCase):
    """Synthetic preparation upstream reviews; genuine original admission bytes; no worker eligibility."""
    @classmethod
    def setUpClass(cls):
        root=SCRIPTS.parents[2]
        peer=json.loads((root/'.review_metadata/independent_full_profile_adapter_review/independent_dependency_6821437_50.json').read_bytes())
        cls.actual_common_ref=peer['profile_pointer']['common_plan']
        raw=Path(cls.actual_common_ref['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=cls.actual_common_ref['sha256']:raise ValueError('immutable common metadata changed')
        cls.actual_common=json.loads(raw)
        observed=json.loads((root/'.review_metadata/future_profile_history_integration/original_authority_6821437_52.json').read_bytes())
        cls.original_raw=Path(observed['provenance']['source_authority']['path']).read_bytes()
        if hashlib.sha256(cls.original_raw).hexdigest()!=profile.ORIGINAL_ADMISSION_SHA256:raise ValueError('original4a71bytes changed')
        cls.actual_provenance=observed['provenance']
        path=Path('/p/rlprojects/RND_worktrees/2026-11-05-04-02_common-timing-helper_source-69f1b507_helpers-29_numerical-source-d28e6735_gpu-count-0')/'10_jax_exploration_platform/scripts/atari_pixel_campaign/serval02_card1_binding.py'
        if hashlib.sha256(path.read_bytes()).hexdigest()!='af72d3de3e84f7b8be08a73f90f5becf143b1d32c90083c1773b10343751361b':raise ValueError('published69fsource changed')
        import importlib.util
        spec=importlib.util.spec_from_file_location('published_69f_source_only_compatibility',path)
        cls.published_binding=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.published_binding)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.run=self.root/'campaign';self.unit=self.run/'data/profile_fixture'
        self.unit.mkdir(parents=True);(self.run/'slurm').mkdir()
        (self.run/'slurm/submitted_jobids_uva.txt').write_text(os.environ['SLURM_JOB_ID']+'\n')
        (self.run/'SOURCE_AND_GATES_READY.json').write_bytes(self.original_raw)
        self.common=copy.deepcopy(self.actual_common);self.common.update(campaign_run_dir=str(self.run),unit_id='common_fixture')
        common_unit=self.run/'data/common_fixture';(common_unit/'code').mkdir(parents=True)
        for relative in self.common['helpers']['files']:
            (common_unit/'code'/Path(relative).name).write_bytes((Path(self.actual_common_ref['path']).parent/'code'/Path(relative).name).read_bytes())
        expected=json.loads(Path(self.actual_common['direct_backend_image']['expected']['path']).read_bytes())
        self.write(common_unit/'direct_backend_image_expected.json',expected)
        self.common['direct_backend_image']['expected']=profile.reference(common_unit/'direct_backend_image_expected.json')
        self.common_ref=self.write(common_unit/'measurement_plan.json',self.common)
        # This is a synthetic current-source prepin premise, not a new actual
        # target preparation or package observation. Keep accepted image bytes
        # identical while declaring the current exact producer source.
        current_expected=copy.deepcopy(expected)
        current_expected['producer_source_sha256']=hashlib.sha256((SCRIPTS/'direct_backend_image.py').read_bytes()).hexdigest()
        expected_path=self.unit/'direct_backend_image_expected.json';self.write(expected_path,current_expected)
        confirmation={'path':str(common_unit/'result.json'),'sha256':'6'*64}
        self.review={'schema_version':1,'kind':'independent_actual_direct_common_configuration_confirmation_review',
            'approved':True,'passed':True,'independent_of_producer':True,'unresolved_anomalies':[],
            'reviewer_identity':'synthetic upstream review premise ONLY; no actual timing/review/credit',
            'decision':'accepted','raw_window_recomputation_passed':True,'paired_rounds':7,'updates_per_arm_per_round':10,
            'reviewed_at_seconds':5.,'admission_granted':False,'worker_admitted':False,
            'result':confirmation,'plan':self.common_ref,'source_commit':self.common['source_commit'],
            'helper_commit':self.common['helpers']['source_commit'],'configuration':self.common['configuration'],
            'reference_configuration':self.common['reference_configuration'],'copies':1,'copy_seeds':[1],'gpu_uuid':profile.CARD_ONE,
            'raw_audit':{'path':str(self.root/'ABSENT_synthetic_scientific_audit.json'),'sha256':'7'*64}}
        review_path=self.root/'synthetic_review.json';self.write(review_path,self.review)
        profile_path=self.root/'direct_profile.json';self.write(profile_path,direct.core().PROFILE)
        baseline={'source_commit':self.common['source_commit'],'hardware_class':'h100','runtime_environment':self.common['runtime_environment'],
            'direct_binding_profile':direct.core().PROFILE,'correctness_record':self.common['common_correctness_inputs']['result']['path'],
            'confirmation_record':confirmation['path']}
        registry={'initial_matched_copy_counts':[1,2],'initial_ideas':[{'name':'synthetic upstream summary '+str(index),
            'results':{context:[{'copies':copies,'implementation_distinct':True,'result_path':'/synthetic/absent/result.json',
                'measurement_valid':True,'correctness_passed':True} for copies in (1,2)] for context in ('h100','jaguar03_rtx_a4500')}} for index in range(30)]}
        paths={}
        for name,value in (('configuration',self.common['configuration']),('runtime_environment',self.common['runtime_environment']),
                ('asset_record',self.common['asset_record']),('study_registry',registry),('accepted_baseline_specification',baseline),
                ('profile_tool_record',{'path':'/synthetic/Nsight_NOT_EXECUTED','version':'synthetic_fixture_only','sha256':'8'*64})):
            paths[name]=self.root/(name+'.json');self.write(paths[name],value)
        self.args=SimpleNamespace(**paths,configured_isolation_profile=None,direct_binding_profile=profile_path,
            hardware_class='h100',gpu_uuid=profile.CARD_ONE,seed_start=1,maximum_active_seconds=30.,
            source_root=Path(self.common['source_root']),source_commit=self.common['source_commit'],run_dir=self.run,
            direct_confirmation_review=review_path,direct_history_inputs=self.unit/'direct_history/inputs.json',
            direct_backend_image_record=expected_path,direct_qualification_record=Path(self.common['direct_qualification_inputs']['result']['path']),
            direct_qualification_review=Path(self.common['direct_qualification_inputs']['review']['path']),
            minimum_updates=1,unit_id='profile_fixture',python=Path(self.common['python']),jaxtari_source_root=None)
        self.provenance=copy.deepcopy(self.actual_provenance)
        self.provenance['canonical_admission']={'path':str(self.run/'SOURCE_AND_GATES_READY.json'),'sha256':profile.ORIGINAL_ADMISSION_SHA256}
        self.provenance['synthetic_upstream_fact_fixture_only']=True

    def write(self,path,value):
        path.write_bytes(profile.encoded(value));return profile.reference(path)

    def prepared_fixture(self):
        """Call real prepare/configure/stage functions; only unavailable upstream eligibility is synthetic."""
        roster=selected.required_helpers(False,True,direct_profile=True)
        helpers={'source_root':str(SCRIPTS.parents[2]),'source_commit':'27f9b5c9276de2405c9aa7225b2c4c6b99e5c50d',
            'files':{profile.PREFIX+name:hashlib.sha256((SCRIPTS/name).read_bytes()).hexdigest() for name in roster}}
        original=profile.configure_binding;events=[]
        def bound(plan,target,history,backend_record):
            events.append('binding')
            self.assertTrue(Path(history).is_file())
            self.assertEqual((Path(history).parent/'admission.json').read_bytes(),self.original_raw)
            return original(plan,target,history,backend_record)
        def authority(plan,common,*,live=False):
            events.append('authority-live' if live else 'authority-frozen')
            return self.original_raw,copy.deepcopy(self.provenance)
        with patch.object(profile,'require_admission',return_value=None),patch.object(selected,'helper_receipt',return_value=helpers),\
                patch.object(profile,'original_admission_metadata',side_effect=authority),patch.object(profile,'configure_binding',side_effect=bound):
            directory=selected.prepare(self.args)
        self.assertEqual(directory,self.unit)
        self.assertLess(events.index('authority-live'),events.index('binding'))
        plan=json.loads((self.unit/'measurement_plan.json').read_bytes())
        self.assertEqual(len(plan['helpers']['files']),34)
        self.assertEqual({p.name for p in (self.unit/'code').iterdir()},set(roster))
        self.assertFalse(Path(self.review['result']['path']).exists())
        self.assertFalse(Path(self.review['raw_audit']['path']).exists())
        return plan

    def test_actual_prepare_caller_history_admission_and_published_claim(self):
        plan=self.prepared_fixture()
        pointer=profile.checked_reference(plan['direct_binding']['frozen_inputs'])
        self.assertEqual(pointer['admission']['path'],str(self.unit/'direct_history/admission.json'))
        self.assertFalse(pointer['admission_provenance']['profile_worker_admitted'])
        claim=self.published_binding.current_admission_claim(plan)
        self.assertEqual(claim['frozen_admission'],pointer['admission'])
        self.published_binding.check_live_admission(claim)
        with self.assertRaises(ValueError):profile.require_admission()

    def test_shared_profile_dispatch_and_original_live_admission_failures(self):
        plan=self.prepared_fixture();binding=direct.core()
        snap=SimpleNamespace(qual={'review':self.common['direct_qualification_inputs']['review']},
            correctness={'review':self.common['common_correctness_inputs']['review'],'result':self.common['common_correctness_inputs']['result']})
        reader=profile.saved_consumers(plan)
        with patch.object(profile,'original_admission_metadata',return_value=(self.original_raw,self.provenance)),\
                patch.object(reader,'reviewed_snapshot',return_value=snap),patch.object(binding,'source_identity',return_value={'synthetic_upstream_source_premise_only':True}):
            history=binding.validate_history(plan)
        self.assertEqual(history['kind'],'reviewed_profile_original_source_history_fact')
        self.assertFalse(history['profile_worker_admitted']);self.assertFalse(history['original_cases_reexecuted'])
        claim=self.published_binding.current_admission_claim(plan)
        canonical=self.run/'SOURCE_AND_GATES_READY.json';canonical.write_text('{}\n')
        with self.assertRaisesRegex(ValueError,'canonical'):self.published_binding.check_live_admission(claim)
        canonical.write_bytes(self.original_raw)
        admission=self.unit/'direct_history/admission.json';admission.write_text('{}\n')
        with self.assertRaises(ValueError):self.published_binding.check_live_admission(claim)

    def test_closed_prepare_has_no_history_side_effect(self):
        with self.assertRaisesRegex(ValueError,'remain disabled'):selected.prepare(self.args)
        self.assertFalse(self.args.direct_history_inputs.exists())
        self.assertFalse((self.unit/'measurement_plan.json').exists())


class DirectProfileBackendTests(DirectBackendImageTests):
    """Add the profile chronology while retaining all original backend metadata checks."""
    def profile_result(self):
        """Build an explicit complete synthetic full-update window after mapped capture."""
        self.plan.update(kind="selected_configuration_profile", direct_profile_protocol=profile.PROTOCOL,
            profile_updates=1, configuration={"learner": {"n_copies": 1}}, runtime_environment={}, copy_seeds=[1])
        self.plan.pop("mode")
        self.write(self.unit / "measurement_plan.json", self.plan)
        self.begin()
        memory_start, execution_start, execution_finish, memory_finish = (time.time() + offset for offset in (0., .000001, .000002, .000003))
        while time.time() <= memory_finish:
            pass
        result = self.result()
        result.update(capture_completed=True, profile_updates=1, profile_start_iteration=4, profile_end_iteration=5,
            normalization_warmup_iterations=50, compiled_signatures_before=[1], compiled_signatures_after=[1],
            configuration=self.plan["configuration"], copy_seeds=[1], actual_runtime_environment={},
            profile_memory={"assigned_gpu_process_observation": {"window_started_at_seconds": memory_start,
                "window_finished_at_seconds": memory_finish}})
        result["execution_identity"].update(started_at_seconds=execution_start, finished_at_seconds=execution_finish)
        return result

    def test_source_preserves_numerical_compute_original_gates_and_cuda_collectors(self):
        """Use this task's approved base after later historical-consumer corrections."""
        root = SCRIPTS.parents[2]
        before = json.loads((root / ".review_metadata/direct_h100_profile_protected_source_before.json").read_text())
        for path, expected in before["files"].items():
            self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), expected)
        tree = ast.parse((SCRIPTS / "profile_selected.py").read_text())
        core = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_capture_core")
        self.assertEqual(ast.dump(core, include_attributes=False), before["capture_core_ast"])
        for name in ("train_optimized.py", "configured_measurement.py", "common_confirmation.py",
            "direct_common_confirmation.py", "measure_selected.py", "nsight_process_identity.py"):
            raw = subprocess.check_output(["git", "-C", str(root), "show", before["base"] + ":" + backend.PREFIX + name])
            self.assertEqual(raw, (SCRIPTS / name).read_bytes())
        old = subprocess.check_output(["git", "-C", str(root), "show", before["base"] + ":" + backend.PREFIX + "diagnose_gpu_execution.py"], text=True)
        original = {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(old).body if isinstance(node, ast.FunctionDef)}
        current = {node.name: ast.dump(node, include_attributes=False) for node in ast.parse((SCRIPTS / "diagnose_gpu_execution.py").read_text()).body if isinstance(node, ast.FunctionDef)}
        for name in ("core_identity", "default_configuration", "cuda_profiler", "loaded_gpu_libraries"):
            self.assertEqual(current[name], original[name], name)

    def test_profile_completion_preserves_original_timestamp_meaning(self):
        """Mapped begin precedes core execution and post-copy follows its whole memory window."""
        result = self.profile_result()
        with patch.object(backend, "own_process", return_value=self.process), patch.object(backend, "mapped_snapshot", side_effect=self.maps):
            value = backend.capture_complete(self.unit, self.plan, result)
        self.assertEqual(value["completed_work_kind"], "complete_selected_profile_window")
        self.assertEqual(value["numerical_execution"], result["execution_identity"])
        backend.validate_capture(self.unit, self.plan, self.binding, result["loaded_gpu_libraries"], result)

    def test_profile_partial_work_boolean_counts_and_changed_compilation_are_rejected(self):
        """A partial or changed profile never becomes a seven-round backend completion."""
        result = self.profile_result()
        for key, value in (("profile_end_iteration", 4), ("profile_start_iteration", True), ("profile_updates", True),
            ("normalization_warmup_iterations", 1), ("compiled_signatures_after", [2]), ("capture_completed", False)):
            with self.subTest(key=key), self.assertRaises(ValueError):backend.completed_work(self.plan, {**result, key: value})
        with self.assertRaises(ValueError):backend.plan_path({**self.plan, "direct_profile_protocol": "old"}, self.unit)

    def test_wrong_profile_memory_chronology_is_rejected(self):
        """A window starting before mapped capture cannot replace the actual full window."""
        result = self.profile_result()
        result["profile_memory"]["assigned_gpu_process_observation"]["window_started_at_seconds"] = self.start
        with patch.object(backend, "own_process", return_value=self.process), patch.object(backend, "mapped_snapshot", side_effect=self.maps):
            with self.assertRaisesRegex(ValueError, "surround"):backend.capture_complete(self.unit, self.plan, result)


if __name__ == "__main__":
    unittest.main()
