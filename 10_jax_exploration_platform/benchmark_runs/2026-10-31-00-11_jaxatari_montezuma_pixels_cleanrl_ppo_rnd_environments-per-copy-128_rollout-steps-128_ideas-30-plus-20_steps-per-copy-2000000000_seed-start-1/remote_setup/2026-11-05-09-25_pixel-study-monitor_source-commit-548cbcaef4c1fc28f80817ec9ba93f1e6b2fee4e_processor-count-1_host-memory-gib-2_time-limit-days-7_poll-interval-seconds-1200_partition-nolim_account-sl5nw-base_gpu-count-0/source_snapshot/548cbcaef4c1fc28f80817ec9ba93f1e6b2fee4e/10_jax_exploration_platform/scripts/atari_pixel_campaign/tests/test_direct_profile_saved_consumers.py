"""Check saved-reader metadata, source dispatch and unchanged combination decisions.

All source/device prerequisites in the combination fixtures are explicit synthetic
success records. No fixture is actual qualification, timing or profiling evidence.
"""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import direct_profile_saved_consumers as saved
import direct_profile_prerequisites as prerequisite
import post_comparison_validation as validation
import diagnose_gpu_execution as diagnostic


def configurations():
    """Read original source defaults and change only the two accepted rendering switches."""
    values = diagnostic.default_configuration(SCRIPTS.parents[2])
    reference = {key: values[key] for key in ("learner", "environment")}
    reference["learner"]["n_copies"] = 1
    reference["execution"] = {"compilation_mode": "whole_iteration", "iterations_per_call": 1,
        "report_readout": "none", "report_every_iterations": 10}
    reference["environment"].update(cache_room_backgrounds=False, render_final_two_only=False)
    candidate = copy.deepcopy(reference)
    candidate["environment"].update(cache_room_backgrounds=True, render_final_two_only=True)
    return candidate, reference


def records():
    """Construct full-size metadata windows with synthetic successful source/device prerequisites."""
    candidate, reference = configurations()
    runtime = {"JAX_ENABLE_X64": "true", "JAX_DEFAULT_MATMUL_PRECISION": "highest",
        "XLA_FLAGS": "--xla_gpu_exclude_nondeterministic_ops=true"}
    checks = {"passed": True, "supplied_inputs": {name: {"leaves": 1} for name in (
        "preparation", "fixed_batch", "fixed_batch_update")}, "complete_iteration": {"leaves": 1}}
    first = {"hardware_class": "h100", "source_commit": "0" * 40, "runtime_environment": runtime,
        "direct_binding_record": True, "candidate": candidate, "reference": reference,
        "correctness_attempted": True, "correctness_passed": True, "correctness": checks,
        "implementation_distinct": True, "copies": 1,
        "execution_identity": {"hostname": "synthetic-host", "pid": 101, "process_start_time": "1001",
            "started_at_seconds": 1.0, "finished_at_seconds": 2.0}}
    second = {**copy.deepcopy(first), "measurement_valid": True, "contaminated": False, "decision": "accepted",
        "execution_identity": {"hostname": "synthetic-host", "pid": 102, "process_start_time": "1002",
            "started_at_seconds": 3.0, "finished_at_seconds": 4.0}, "rounds": []}
    for number in range(7):
        pair = {"order": ["reference", "candidate"] if number % 2 == 0 else ["candidate", "reference"]}
        for arm, seconds in (("reference", 2.0), ("candidate", 1.0)):
            pair[arm] = {"iterations": 10, "copies": 1, "environments_per_copy": 128, "rollout_steps": 128,
                "training_transitions": 163840, "seconds": seconds,
                "memory": {"other_gpu_processes": [], "gpu_uuid": "GPU-synthetic-fixture",
                    "sampled_process_peak_bytes": 1000, "allocator": {"peak_bytes_in_use": 900}}}
        second["rounds"].append(pair)
    specification = {"correctness_record": "/synthetic/correctness/result.json", "confirmation_record": "/synthetic/confirmation/result.json",
        "hardware_class": "h100", "source_commit": "0" * 40, "runtime_environment": runtime,
        "direct_binding_profile": {"synthetic_saved_profile": True}}
    return specification, first, second


class CombinationFixture:
    """Represent already successful source/device readers only for the unchanged statistic tests."""
    resolver = None

    def __init__(self, specification, first, second):
        """Keep two separate explicit saved records and the expected fixture profile."""
        self.specification = specification
        self.records = {specification["correctness_record"]: first, specification["confirmation_record"]: second}
        self.dispatched = []

    def path_identity(self, path):
        """Retain the supplied independent source locations without creating actual units."""
        return path

    def read(self, path):
        """Return the selected synthetic data record without filesystem or GPU work."""
        return self.records[path]

    def require_direct_profile(self, profile):
        """Represent a successful exact-profile prerequisite for this statistic-only fixture."""
        if profile != self.specification["direct_binding_profile"]:
            raise ValueError("synthetic profile differs")

    def verified_direct_plan(self, path, record):
        """Represent completed independent raw readers, without running or qualifying one."""
        if record != self.records[path]:
            raise ValueError("synthetic record differs")
        self.dispatched.append(path)
        return {"direct_binding": {"profile": self.specification["direct_binding_profile"]}}, {}


class CombinationChecks(unittest.TestCase):
    """Retain the actual original field, work, ordering and statistic predicates."""
    def setUp(self):
        """Create only source-independent metric fixtures, without learner transitions."""
        self.specification, self.first, self.second = records()

    def verify(self):
        """Run the real copied source-proved statistic function with explicit successful reader prerequisites."""
        evidence = CombinationFixture(self.specification, self.first, self.second)
        return saved.combine_records_with_verified_readers(self.specification, evidence, validation), evidence

    def test_exact_original_statement_source_proof(self):
        """Only the metadata import and two explicitly declared reader calls may differ."""
        proof = saved.statistic_source_proof(SCRIPTS / "post_comparison_validation.py")
        self.assertTrue(proof["all_other_statement_syntax_unchanged"])
        self.assertFalse(proof["tolerances_or_statistics_changed"])
        self.assertEqual(len(proof["replaced_metadata_dispatch"]), 2)

    def test_positive_full_windows_dispatch_both_records(self):
        """Seven complete alternating pairs preserve the original speedup and sign calculation."""
        result, evidence = self.verify()
        self.assertEqual(result["paired_rounds"], 7)
        self.assertEqual(result["median_speedup"], 2.0)
        self.assertEqual(result["one_sided_sign_probability"], 0.0078125)
        self.assertEqual(evidence.dispatched, [self.specification["correctness_record"], self.specification["confirmation_record"]])

    def test_original_source_runtime_and_correctness_fields_refuse(self):
        """A failed earlier gate cannot be hidden behind a successful saved-reader dispatch."""
        changes = [lambda first, second: first.update(source_commit="f" * 40),
            lambda first, second: second["runtime_environment"].update(JAX_ENABLE_X64="false"),
            lambda first, second: first.update(correctness_passed=False),
            lambda first, second: first["correctness"].update(supplied_inputs={}),
            lambda first, second: second.update(implementation_distinct=False)]
        for index, mutate in enumerate(changes):
            self.specification, self.first, self.second = records(); mutate(self.first, self.second)
            with self.subTest(change=index), self.assertRaises(ValueError):
                self.verify()

    def test_original_configuration_work_and_decision_fields_refuse(self):
        """Changed original updates, contamination or an unaccepted decision retain their old refusal."""
        changes = [lambda first, second: second["candidate"]["learner"].update(num_steps=64),
            lambda first, second: second["rounds"][0]["candidate"].update(iterations=9),
            lambda first, second: second["rounds"][0].update(order=["candidate", "reference"]),
            lambda first, second: second.update(contaminated=True),
            lambda first, second: second.update(decision="inconclusive"),
            lambda first, second: second["rounds"][0]["candidate"]["memory"].update(other_gpu_processes=[999])]
        for index, mutate in enumerate(changes):
            self.specification, self.first, self.second = records(); mutate(self.first, self.second)
            with self.subTest(change=index), self.assertRaises(ValueError):
                self.verify()

    def test_original_insufficient_rounds_and_no_speedup_refuse(self):
        """Six pairs or no repeatable improvement cannot satisfy the original decision rule."""
        self.second["rounds"].pop()
        with self.assertRaises(ValueError):
            self.verify()
        self.specification, self.first, self.second = records()
        for pair in self.second["rounds"]:
            pair["candidate"]["seconds"] = pair["reference"]["seconds"]
        with self.assertRaises(ValueError):
            self.verify()

    def test_original_process_independence_and_no_final_bypass(self):
        """The same numerical execution and a final-mode request still fail."""
        self.second["execution_identity"] = self.first["execution_identity"]
        with self.assertRaises(ValueError):
            self.verify()
        self.specification, self.first, self.second = records()
        evidence = CombinationFixture(self.specification, self.first, self.second)
        with self.assertRaises(ValueError):
            saved.combine_records_with_verified_readers(self.specification, evidence, validation, final=True)


class SourceAndProtocolChecks(unittest.TestCase):
    """Check exact helper sets and metadata-only namespaces without old scientific artifacts."""
    def source_fixture(self, common):
        """Before: source declarations; after: their exact required adjacent-name set."""
        names = set(saved.literal_helpers((SCRIPTS / "measure_selected.py").read_bytes()))
        names.update(saved.literal_helpers((SCRIPTS / "serval02_card1_binding.py").read_bytes()))
        names.update(("direct_measurement.py", "configured_measurement.py", "direct_backend_image.py"))
        if common:
            names.update(("direct_common_confirmation.py", "direct_historical_consumer.py"))
        return {Path(name).stem: {"raw": (SCRIPTS / name).read_bytes()} for name in names}

    def test_exact_original_and_common_rosters_derive_from_sources(self):
        """Complete names pass; losing or adding one helper fails regardless of total count."""
        for common in (False, True):
            values = self.source_fixture(common)
            self.assertEqual(set(saved.source_roster(values, common)), {name + ".py" for name in values})
            removed = dict(values); removed.pop("direct_backend_image")
            with self.assertRaises(ValueError):
                saved.source_roster(removed, common)
            substituted = dict(values); substituted["unlisted_same_count"] = substituted.pop("direct_backend_image")
            with self.assertRaises(ValueError):
                saved.source_roster(substituted, common)

    def test_new_profile_roster_is_exact_and_native_diagnostic_refuses(self):
        """One saved-consumer helper is added; diagnostic-only native evidence stays unsupported."""
        roster = prerequisite.helper_roster()
        self.assertEqual(len(roster), 34)
        self.assertIn(saved.PREFIX + "profile_metadata_runtime.py",roster)
        self.assertIn(saved.PREFIX + saved.NAME, roster)
        self.assertNotIn(saved.PREFIX + "native_process_evidence.py", roster)
        with self.assertRaises(ValueError):
            prerequisite.helper_roster({"native_process_control": {"mode": "observe_target_contract_for_zero_credit_diagnostic"}})
        with self.assertRaises(ValueError):
            prerequisite.require_admission()

    def test_pinned_metadata_module_bytes_and_aliases_refuse(self):
        """Read-only metadata can execute only its exact ordinary source bytes."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.py"; path.write_text("answer = 7\n")
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            module = saved.source_module("synthetic_metadata_fixture", path, expected)
            self.assertEqual(module.answer, 7)
            with self.assertRaises(ValueError):
                saved.source_module("synthetic_metadata_fixture", path, "0" * 64)
            alias = path.with_name("alias.py"); alias.symlink_to(path)
            with self.assertRaises(ValueError):
                saved.source_module("synthetic_metadata_fixture", alias, expected)

    def test_metadata_module_change_during_import_refuses(self):
        """Changing the source pathname cannot replace the already verified executed buffer."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.py"
            path.write_text("from pathlib import Path\nPath(__file__).write_text('answer = 99\\n')\nanswer = 7\n")
            with self.assertRaises(ValueError):
                saved.source_module("synthetic_metadata_fixture", path, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_lazy_saved_namespace_records_real_source_and_refuses_numerical(self):
        """Only two private fixture modules run; neither represents a scientific consumer."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); sources = {}
            for name, raw in (("synthetic_saved_first", b"import synthetic_saved_second\nanswer = synthetic_saved_second.answer\n"),
                    ("synthetic_saved_second", b"answer = 7\n")):
                path = root / (name + ".py"); path.write_bytes(raw)
                sources[name] = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "raw": raw}
            loader = saved.install_sources(sources)
            try:
                module = __import__("synthetic_saved_first")
                self.assertEqual(module.answer, 7)
                self.assertEqual(set(loader.loaded), set(sources))
                with self.assertRaises(ValueError):
                    loader.find_spec("jax")
            finally:
                sys.meta_path.remove(loader)
                for name in sources:
                    sys.modules.pop(name, None)

    def test_existing_module_namespace_cannot_be_replaced(self):
        """An ambient helper is refused instead of silently changing its source."""
        with patch.dict(sys.modules, {"synthetic_existing_helper": SimpleNamespace()}):
            with self.assertRaises(ValueError):
                saved.install_sources({"synthetic_existing_helper": {"path": "/synthetic", "raw": b"", "sha256": "0" * 64}})

    def test_duplicate_keys_nonfinite_and_metadata_writes_refuse(self):
        """Malformed data and write operations cannot enter the read-only protocol."""
        for raw in (b'{"same":1,"same":2}', b'{"number":NaN}'):
            with self.assertRaises(ValueError):
                saved.document(raw)
        guard = SimpleNamespace(metadata_event_guard=lambda *args: None)
        with self.assertRaises(ValueError):
            saved.readonly_audit(guard, set(), Path("/synthetic"), {}, "open", ("/synthetic/file", "w", os.O_WRONLY))
        for event in ("os.mkdir", "os.rename", "os.symlink"):
            with self.assertRaises(ValueError):
                saved.readonly_audit(guard, set(), Path("/synthetic"), {}, event, ())

    def test_typed_saved_process_identity_refuses_substitutions(self):
        """All four UIDs, real birth and typed positive process numbers remain mandatory."""
        value = saved.process(os.getpid())
        self.assertEqual(saved.validate_process_record(value), value)
        for key, replacement in (("pid", True), ("parent_pid", value["pid"]), ("process_start_time", "0"),
                ("uids", [2618919] * 3 + [0]), ("argv", []), ("hostname", "")):
            changed = copy.deepcopy(value); changed[key] = replacement
            with self.subTest(key=key), self.assertRaises(ValueError):
                saved.validate_process_record(changed)

    def test_actual_kernel_context_refuses_changed_claim(self):
        """A synthetic job claim is rejected before any ledger or scheduler lookup."""
        with patch.dict(os.environ, {"SLURM_JOB_ID": "987654", "SLURM_STEP_ID": "123"}):
            with self.assertRaisesRegex(ValueError, "actual kernel"):
                saved.compute_context({"campaign_run_dir": "/synthetic/unopened"})




class NewTwentyNineCommonMetadataChecks(unittest.TestCase):
    """Inspect actual bound timing metadata only, with no invented timing result or profile admission."""
    @classmethod
    def setUpClass(cls):
        """Read the exact existing one-copy timing plan and its hash-bound staged helper bytes."""
        run = Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1')
        cls.unit = next((run / 'data').glob('2026-11-05-02-24_common-timing*copies-1_*'))
        raw = (cls.unit / 'measurement_plan.json').read_bytes()
        if hashlib.sha256(raw).hexdigest() != 'c486981267066f96fc9e440b131f845ffa702e2afff368c0f7741e7d2f015f0c':
            raise ValueError('actual bound timing metadata changed')
        cls.plan = json.loads(raw)
        cls.sources = {Path(relative).stem: {'raw': (cls.unit / 'code' / Path(relative).name).read_bytes()}
            for relative in cls.plan['helpers']['files']}

    def test_actual_new_twenty_nine_metadata_and_exact_union(self):
        """The actual new snapshot declaration admits metadata recognition, never a timing result."""
        actual = saved.validate_common_metadata(self.plan, self.unit)
        self.assertEqual(len(actual), 29)
        self.assertIn('accepted_evidence_snapshot', actual)
        self.assertEqual(set(saved.source_roster(self.sources, True, accepted_snapshot=True)),
            {name + '.py' for name in self.sources})
        self.assertNotIn('native_process_evidence', actual)

    def test_missing_or_mixed_historical_protocol_refused(self):
        """The old28 and new29 declarations cannot be absent, mixed or inferred from file count."""
        changed = copy.deepcopy(self.plan);del changed['accepted_evidence_snapshot_protocol']
        with self.assertRaises(ValueError):saved.validate_common_metadata(changed, self.unit)
        changed = copy.deepcopy(self.plan);changed['direct_historical_consumer_protocol'] = {}
        with self.assertRaises(ValueError):saved.validate_common_metadata(changed, self.unit)
        with self.assertRaises(ValueError):saved.source_roster(self.sources, True)
        with self.assertRaises(ValueError):saved.source_roster(self.sources, False, accepted_snapshot=True)

    def test_protocol_types_and_values_must_match_staged_literal(self):
        """A Boolean version or weakened operation scope differs from actual committed metadata source."""
        for key, value in [('schema_version', True), ('maximum_check_seconds', 600),
                ('operational_admission', True), ('reuse_scope', 'current_runtime')]:
            changed = copy.deepcopy(self.plan);changed['accepted_evidence_snapshot_protocol'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):saved.validate_common_metadata(changed, self.unit)

    def test_same_count_missing_or_substituted_helper_refused(self):
        """Removing a required source or replacing it with another name fails despite matching counts."""
        for removed in ('accepted_evidence_snapshot', 'direct_historical_consumer', 'direct_backend_image'):
            changed = dict(self.sources);changed.pop(removed)
            with self.subTest(removed=removed), self.assertRaises(ValueError):
                saved.source_roster(changed, True, accepted_snapshot=True)
        changed = dict(self.sources);changed['other_name'] = changed.pop('accepted_evidence_snapshot')
        with self.assertRaises(ValueError):saved.source_roster(changed, True, accepted_snapshot=True)

    def test_profile_admission_and_native_scope_remain_closed(self):
        """Metadata recognition cannot enable formal profiling or promote diagnostic native control."""
        with self.assertRaises(ValueError):prerequisite.require_admission()
        with self.assertRaises(ValueError):prerequisite.helper_roster({'native_process_control': {}})
        self.assertIs(saved.PROTOCOL['operational_admission'], False)


if __name__ == "__main__":
    unittest.main()
