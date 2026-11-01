"""Check final-only protocols and complete raw evidence without numerical imports or live commands."""

import copy
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import controller
import launch_final_training as final_launcher
import measure_selected as selected
import portable_evidence
import post_comparison_validation as validation
import short_learning
import paired_benchmark
from final_phase_fixtures import FinalPhaseFixture


class FinalProtocolTests(unittest.TestCase):
    """Keep the old intermediate protocol and require explicit longer final measurements."""

    def test_intermediate_defaults_and_explicit_final_protocol(self):
        """Seven by ten remains the default; final confirmation uses eleven by twenty."""
        arguments = SimpleNamespace(mode="confirmation", rounds=None, minimum_updates_per_round=None)
        self.assertEqual(selected.requested_protocol(arguments), (7, 10, False))
        arguments.final_confirmation = True
        self.assertEqual(selected.requested_protocol(arguments), (11, 20, True))
        for name, value in (("rounds", 7), ("minimum_updates_per_round", 10), ("mode", "packing-point")):
            bad = copy.deepcopy(arguments)
            setattr(bad, name, value)
            with self.subTest(name=name), self.assertRaises(ValueError):
                selected.requested_protocol(bad)

    def test_learning_budget_preserves_complete_groups_and_full_horizon(self):
        """Round the predeclared minimum upward without changing original updates or annealing."""
        first = {"execution": {"iterations_per_call": 3, "report_readout": "none", "report_every_iterations": 10}}
        second = {"execution": {"iterations_per_call": 1, "report_readout": "scheduled", "report_every_iterations": 10}}
        contract = short_learning.protocol([first, second])
        self.assertEqual((contract["updates"], contract["complete_window_updates"]), (150, 30))
        self.assertEqual(contract["full_training_horizon"], 2_000_000_000)
        self.assertIs(contract["learning_equivalence_established"], False)
        self.assertIs(contract["automatic_tolerance_adjustment"], False)

    def test_learning_scheduled_report_bytes_and_windows_are_required(self):
        """The reader retains every scheduled report and rejects omitted bytes or changed counters."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            configurations = [{"execution": {"report_readout": "scheduled", "report_every_iterations": 2}}] * 2
            curves = [{"variant": arm, "iteration": index, "metrics": {"steps_per_policy": index * 16384, "loss": [.1]}}
                      for arm in ("reference", "candidate") for index in range(1, 5)]
            reports = [{"variant": arm, "iteration": index, "window_updates": 2,
                        "metrics": {"steps_per_policy": index * 16384, "loss": [.1]}}
                       for index in (2, 4) for arm in (("reference", "candidate") if index == 2 else ("candidate", "reference"))]
            path = directory / "learning_reports.jsonl"

            def publish(values):
                """Bind explicitly changed synthetic report bytes without hiding the changed schedule."""
                path.write_text(''.join(json.dumps(row) + '\n' for row in values))
                return {"reports": {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}

            result = publish(reports)
            self.assertEqual(validation.validate_learning_reports(result, directory / "result.json", curves,
                configurations, validation.Evidence()), result["reports"])
            for changed in (reports[:-1], [{**row, "iteration": 1} for row in reports]):
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    validation.validate_learning_reports(publish(changed), directory / "result.json", curves,
                        configurations, validation.Evidence())
            result = publish(reports)
            path.write_bytes(path.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, "reports changed"):
                validation.validate_learning_reports(result, directory / "result.json", curves, configurations, validation.Evidence())

    def test_single_update_latency_retains_scheduled_windows_and_includes_readout_time(self):
        """A production-used fake-clock loop includes every selected report while synchronizing every update."""
        for mode in ("scheduled", "per_update", "none"):
            with self.subTest(mode=mode):
                now, synchronizations, readouts = [0.], [], []

                class Arrays:
                    """Provide scalar host/device addition for the unchanged frozen reporting functions."""

                    @staticmethod
                    def array(value, copy=True):
                        """Retain a scalar copy just as the reporting function expects."""
                        return value

                    @staticmethod
                    def add(first, second):
                        """Add successive original metric values without changing the report window."""
                        return first + second

                def one(state):
                    """Spend one second on exactly one original complete update."""
                    now[0] += 1.
                    iteration = state.iteration + 1
                    return SimpleNamespace(iteration=iteration), {"loss": float(iteration), "reward_ext_sum": 1., "steps_per_policy": iteration * 16384}

                def read(value):
                    """Spend two seconds on each actual selected host readout."""
                    now[0] += 2.
                    readouts.append(value)
                    return copy.deepcopy(value)

                def synchronize(value):
                    """Record every original update boundary without adding synthetic work."""
                    synchronizations.append(value)

                executor = SimpleNamespace(group_size=1, one=one, options={"report_readout": mode, "report_every_iterations": 2})
                state, metrics, reports, durations = short_learning.synchronized_report_window(executor,
                    SimpleNamespace(iteration=0), 6, paired_benchmark, device_get=read, synchronize=synchronize,
                    device_arrays=Arrays, host_arrays=Arrays, clock=lambda: now[0])
                self.assertEqual((state.iteration, len(synchronizations)), (6, 7))
                self.assertEqual(durations, [1., 3., 1., 3., 1., 3.] if mode == "scheduled" else [3.] * 6 if mode == "per_update" else [1.] * 6)
                self.assertEqual([row["iteration"] for row in reports], [2, 4, 6] if mode != "none" else [])
                if reports:
                    self.assertEqual([row["metrics"]["loss"] for row in reports], [1.5, 3.5, 5.5])
                    self.assertEqual([row["metrics"]["reward_ext_sum"] for row in reports], [2.] * 3)
                self.assertEqual(len(readouts), 3 if mode == "scheduled" else 6 if mode == "per_update" else 0)
                self.assertEqual(metrics["steps_per_policy"], 6 * 16384)
                with self.assertRaisesRegex(ValueError, "complete selected report windows"):
                    short_learning.synchronized_report_window(SimpleNamespace(group_size=1,
                        options={"report_readout": "scheduled", "report_every_iterations": 2}),
                        SimpleNamespace(iteration=0), 3, paired_benchmark, device_get=read, synchronize=synchronize,
                        device_arrays=Arrays, host_arrays=Arrays, clock=lambda: now[0])

    def test_single_update_latency_waits_for_deferred_accumulation_on_that_update(self):
        """Pending device accumulation belongs to its own update rather than the next report readout."""
        now, boundaries = [0.], []

        class Pending:
            """Represent device work that remains incomplete until a caller waits for it."""

            def __init__(self, value, seconds, dependencies=()):
                """Keep a result and its exact pending predecessors without doing their work early."""
                self.value, self.seconds, self.dependencies, self.ready = value, seconds, dependencies, False

        class Arrays:
            """Make the original reporting functions dispatch deferred scalar work."""

            @staticmethod
            def array(value, copy=True):
                """Dispatch two seconds of initial accumulator work."""
                return Pending(value, 2.)

            @staticmethod
            def add(first, second):
                """Dispatch three seconds of addition after the previous accumulator completes."""
                return Pending(first.value + second, 3., (first,))

        def complete(value):
            """Wait only for the explicitly supplied device-work tree and its dependencies."""
            if isinstance(value, Pending):
                if not value.ready:
                    for dependency in value.dependencies:
                        complete(dependency)
                    now[0] += value.seconds
                    value.ready = True
                return value.value
            if isinstance(value, dict):
                return {name: complete(item) for name, item in value.items()}
            if isinstance(value, tuple):
                return tuple(complete(item) for item in value)
            return value

        def one(state):
            """Spend one second executing one original update before dispatching report work."""
            now[0] += 1.
            return SimpleNamespace(iteration=state.iteration + 1), {"loss": float(state.iteration + 1)}

        def read(value):
            """Wait for the accumulated device work, then spend five seconds on its selected readout."""
            result = complete(value)
            now[0] += 5.
            return result

        def synchronize(value):
            """Complete exactly the timing boundary that production asks to wait for."""
            boundaries.append(value)
            complete(value)

        executor = SimpleNamespace(group_size=1, one=one, options={"report_readout": "scheduled", "report_every_iterations": 2})
        state, unused, reports, durations = short_learning.synchronized_report_window(executor,
            SimpleNamespace(iteration=0), 4, paired_benchmark, device_get=read, synchronize=synchronize,
            device_arrays=Arrays, host_arrays=Arrays, clock=lambda: now[0])
        self.assertEqual(durations, [3., 9., 3., 9.])
        self.assertEqual(now[0], 24.)
        self.assertEqual(state.iteration, 4)
        self.assertEqual([row["metrics"]["loss"] for row in reports], [1.5, 3.5])
        self.assertEqual(len(boundaries), 5)
        self.assertIsInstance(boundaries[1][2]["loss"], Pending)
        self.assertTrue(boundaries[1][2]["loss"].ready)
        self.assertIsNone(boundaries[2][2])

        def omit_accumulator(value):
            """Reproduce the former incomplete boundary without changing the production source."""
            return synchronize(value[:2] if isinstance(value, tuple) else value)

        now[0] = 0.
        unused, unused_metrics, repeated_reports, former_durations = short_learning.synchronized_report_window(executor,
            SimpleNamespace(iteration=0), 4, paired_benchmark, device_get=read, synchronize=omit_accumulator,
            device_arrays=Arrays, host_arrays=Arrays, clock=lambda: now[0])
        self.assertEqual(former_durations, [1., 11., 1., 11.])
        self.assertEqual(repeated_reports, reports)
        self.assertEqual(now[0], 24.)

    def test_legacy_final_receipts_cannot_admit_new_training(self):
        """Literal current-version gates are required even when all five old flags are true."""
        record = {**{name: True for name in validation.FINAL_FLAGS},
                  "schema_version": 2, "short_learning_checks_passed": True, "fixed_total_scaling_complete": True,
                  "specification": {"final_protocol": copy.deepcopy(validation.FINAL_PROTOCOL)}}
        validation.require_current_final(record)
        for key, value in (("schema_version", 1), ("schema_version", True), ("short_learning_checks_passed", False), (validation.FINAL_FLAGS[0], False)):
            wrong = copy.deepcopy(record)
            wrong[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validation.require_current_final(wrong)
        record["specification"]["final_protocol"]["minimum_learning_updates"] = True
        with self.assertRaises(ValueError):
            validation.require_current_final(record)
        self.assertFalse(any(name in sys.modules for name in ("jax", "numpy", "torch")))

    def test_staged_learning_helper_ignores_foreign_module_and_rejects_changed_bytes(self):
        """Execute exact staged bytes even when the module cache names a foreign numerical-root copy."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "code/short_learning.py"
            path.parent.mkdir()
            shutil.copyfile(SCRIPTS / "short_learning.py", path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            plan = {"helpers": {"files": {"10_jax_exploration_platform/scripts/atari_pixel_campaign/short_learning.py": digest}}}
            with patch.dict(sys.modules, {"short_learning": SimpleNamespace(__file__="/foreign/numerical/short_learning.py")}):
                actual = selected.staged_short_learning(directory, plan)
                self.assertEqual(Path(actual.__file__), path)
                self.assertEqual(actual.MINIMUM_UPDATES, 128)
            path.write_bytes(path.read_bytes() + b'\n# changed staged bytes\n')
            with self.assertRaisesRegex(ValueError, "staged short-learning"):
                selected.staged_short_learning(directory, plan)

    def test_shared_reader_dependencies_cover_both_each_and_neither(self):
        """Shared portable evidence remains required when either committed reader needs it."""
        prefix = final_launcher.PREFIX
        base = prefix + 'unchanged_reader.py'
        expected = {(False, False): set(),
            (False, True): {'device_memory_limit.py', 'validate_training_deployment.py', 'portable_evidence.py'},
            (True, False): {'short_learning.py', 'portable_evidence.py'},
            (True, True): {'short_learning.py', 'device_memory_limit.py', 'validate_training_deployment.py', 'portable_evidence.py'}}
        for flags, needed in expected.items():
            source = ('SHORT_LEARNING_MODULE = "short_learning.py"\n' if flags[0] else '')
            source += ('DEVICE_MEMORY_LIMIT_MODULE = "device_memory_limit.py"\n' if flags[1] else '')
            trees = [ast.parse(source)]
            actual, universe = validation.reader_feature_dependencies(trees)
            with self.subTest(flags=flags):
                self.assertEqual(actual, needed)
                self.assertEqual(final_launcher.reader_feature_dependencies(trees), (actual, universe))
                fields = validation.helper_requirements({base} | {prefix + name for name in universe},
                    {prefix + 'post_comparison_validation.py': source.encode()})
                self.assertEqual(fields, {base} | {prefix + name for name in needed})
        old = {prefix + 'post_comparison_validation.py': b'VALUE = 1\n', prefix + 'portable_evidence.py': b'# originally recorded reader\n'}
        self.assertIn(prefix + 'portable_evidence.py', validation.helper_requirements({prefix + 'portable_evidence.py'}, old))
        with self.assertRaises(ValueError):
            validation.reader_feature_dependencies([ast.parse('SHORT_LEARNING_MODULE = "another.py"')])

    def test_frozen_worker_rosters_use_the_union_and_preserve_older_callers(self):
        """A detached old source excludes unused new files even when those files are present in its tree."""
        for learning, memory in ((False, False), (False, True), (True, False), (True, True)):
            with self.subTest(learning=learning, memory=memory), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                for name in final_launcher.TRAIN_HELPERS:
                    path = root / final_launcher.PREFIX / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text('VALUE = 1\n')
                source = ('SHORT_LEARNING_MODULE = "short_learning.py"\n' if learning else '')
                source += ('DEVICE_MEMORY_LIMIT_MODULE = "device_memory_limit.py"\n' if memory else '')
                (root / final_launcher.PREFIX / 'post_comparison_validation.py').write_text(source)
                subprocess.run(['git', 'init', '-q', str(root)], check=True)
                subprocess.run(['git', '-C', str(root), 'add', '.'], check=True)
                subprocess.run(['git', '-C', str(root), '-c', 'user.name=Metadata dependency fixture',
                    '-c', 'user.email=dependency@example.invalid', '-c', 'core.hooksPath=/dev/null',
                    'commit', '-qm', 'Record exact reader declarations'], check=True)
                commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
                subprocess.run(['git', '-C', str(root), 'checkout', '-q', '--detach', commit], check=True)
                expected, universe = validation.reader_feature_dependencies([ast.parse(source)])
                record = final_launcher.frozen_helpers(root, commit)
                self.assertEqual({Path(name).name for name in record['files']} & universe, expected)


class FinalRawEvidenceTests(unittest.TestCase):
    """Use detached synthetic Git sources and full prepared metadata for final-only checks."""

    @classmethod
    def setUpClass(cls):
        """Build one immutable fixture and restore changed evidence after every test."""
        cls.fixture = FinalPhaseFixture()
        cls.original = {path: path.read_bytes() for path in cls.fixture.root.rglob('*') if path.is_file() and '.git' not in path.parts}

    @classmethod
    def tearDownClass(cls):
        """Delete only the synthetic test directories."""
        cls.fixture.close()

    def tearDown(self):
        """Return every fixture byte to its exact original value after an adversarial edit."""
        for path, raw in self.original.items():
            if not path.exists() or path.read_bytes() != raw:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)

    def common(self, context="h100"):
        """Supply the current source/context/runtime alongside its exact prepared paths."""
        return {**self.fixture.specification["contexts"][context], "hardware_class": context,
                "source_commit": self.fixture.commit, "runtime_environment": self.fixture.runtime}

    def write(self, path, value):
        """Replace only synthetic fixture bytes to exercise strict changed-evidence rejection."""
        return self.fixture.write(Path(path), value)

    def test_complete_current_final_evidence_passes_and_keeps_no_equivalence_claim(self):
        """Derive all new gates from complete eleven-by-twenty, learning and exact-eight evidence."""
        result = validation.validate_final_verification(self.fixture.specification)
        validation.require_current_final(result)
        self.assertEqual(result["schema_version"], 2)
        for context in validation.CONTEXTS:
            self.assertEqual(result["selected"][context]["combination"]["paired_rounds"], 11)
            self.assertEqual(result["learning_checks"][context]["updates_per_arm"], 128)
            self.assertIs(result["learning_checks"][context]["learning_equivalence_established"], False)
        self.assertEqual(result["fixed_total_scaling"]["exact_total_copies"], 8)
        self.assertEqual(json.loads(json.dumps(result)), validation.validate_final_verification(self.fixture.specification))
        self.assertFalse(any(name in sys.modules for name in ("jax", "numpy", "torch")))

    def test_real_metadata_prepare_freezes_final_latency_and_learning_without_module_fallback(self):
        """The actual producer preparation binds reviewed proof, full configuration and exact helper contracts."""
        context = self.common()
        record = json.loads(Path(context["confirmation_record"]).read_bytes())
        inputs = {name: self.write(self.fixture.root / ("preparation_" + name + ".json"), value)
                  for name, value in (("configuration", record["candidate"]), ("reference", record["reference"]),
                      ("runtime", self.fixture.runtime), ("assets", self.fixture.assets))}
        for mode in ("confirmation", "learning-check"):
            command = ["--prepare-only", "--mode", mode, "--source-root", str(self.fixture.source),
                "--source-commit", self.fixture.commit, "--configuration", str(inputs["configuration"]),
                "--reference-configuration", str(inputs["reference"]), "--runtime-environment", str(inputs["runtime"]),
                "--asset-record", str(inputs["assets"]), "--run-dir", str(self.fixture.run), "--unit-id", "fresh_prepared_" + mode,
                "--gpu-uuid", "GPU-one", "--hardware-class", "h100",
                "--completed-study-proof", self.fixture.completed_study["proof"]["path"],
                "--completed-study-review", self.fixture.completed_study["review"]["path"]]
            if mode == "confirmation":
                command.append("--final-confirmation")
            with patch.object(selected, "SCRIPTS", self.fixture.scripts), patch.object(selected, "helper_receipt", return_value=self.fixture.helpers), \
                    patch.dict(sys.modules, {"short_learning": SimpleNamespace(protocol=lambda unused: {"wrong": "foreign cached helper"})}):
                directory = selected.prepare(selected.parse_arguments(command))
            plan = json.loads((directory / "measurement_plan.json").read_bytes())
            self.assertEqual(plan["completed_study"], self.fixture.completed_study)
            self.assertEqual(plan["configuration"], record["candidate"])
            if mode == "confirmation":
                self.assertEqual((plan["rounds"], plan["minimum_updates_per_round"]), (11, 20))
                self.assertEqual(plan["single_update_latency_protocol"],
                    {arm: short_learning.latency_protocol(record[arm]) for arm in ("reference", "candidate")})
            else:
                self.assertEqual(plan["learning_protocol"], short_learning.protocol([record["reference"], record["candidate"]]))
                self.assertIs(plan["learning_protocol"]["learning_equivalence_established"], False)
            self.assertFalse((directory / "result.json").exists())

    def test_final_rounds_and_latency_are_required_while_intermediate_reader_stays_unchanged(self):
        """Reject short final timings and grouped amortized latency without changing old combination checks."""
        common = self.common()
        path = Path(common["confirmation_record"])
        original = json.loads(path.read_bytes())
        short = copy.deepcopy(original)
        short["rounds"] = short["rounds"][:7]
        self.write(path, short)
        validation.validate_combination(common, validation.Evidence())
        with self.assertRaisesRegex(ValueError, "eleven"):
            validation.validate_combination(common, validation.Evidence(), final=True)
        self.write(path, original)
        original["synchronized_per_update_latency"]["candidate"]["updates_per_call"] = 3
        self.write(path, original)
        with self.assertRaisesRegex(ValueError, "single-update"):
            validation.validate_final_confirmation(common, validation.Evidence())

    def test_learning_zero_reward_only_or_missing_review_cannot_pass(self):
        """A separate review must discuss learning dynamics and must not claim equivalence from reward."""
        common = self.common()
        confirmation = validation.validate_final_confirmation(common, validation.Evidence())
        analysis_path = Path(common["learning_analysis_record"])
        analysis = json.loads(analysis_path.read_bytes())
        for change in ({"zero_reward_only_evidence": True}, {"unresolved_anomalies": ["unexplained policy collapse"]},
                       {"comparative_findings": {"episodes_and_exploration": "Both returned zero reward."}}):
            self.write(analysis_path, {**analysis, **change})
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "separate complete comparative"):
                validation.validate_short_learning(common, confirmation, validation.Evidence())

    def test_learning_collapse_and_changed_schedule_are_rejected_from_raw_curves(self):
        """Rehashing altered curves and analysis cannot hide absent learning or a shorter horizon."""
        common = self.common()
        confirmation = validation.validate_final_confirmation(common, validation.Evidence())
        result_path = Path(common["learning_record"])
        original_result = json.loads(result_path.read_bytes())
        curves_path = Path(original_result["curves"]["path"])
        original_rows = [json.loads(line) for line in curves_path.read_bytes().splitlines()]
        for field, value in (("floating_state_nonfinite_count", 1), ("optimizer_step", 1)):
            rows = copy.deepcopy(original_rows)
            for row in rows:
                row[field] = value
            curves_path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            result = copy.deepcopy(original_result)
            result["curves"]["sha256"] = self.fixture.sha(curves_path)
            self.write(result_path, result)
            with self.subTest(field=field), self.assertRaises(ValueError):
                validation.validate_short_learning(common, confirmation, validation.Evidence())
        result = copy.deepcopy(original_result)
        result["arms"]["candidate"]["agent_change_squared"] = [0.]
        self.write(result_path, result)
        curves_path.write_bytes(self.original[curves_path])
        with self.assertRaisesRegex(ValueError, "missing actual per-copy"):
            validation.validate_short_learning(common, confirmation, validation.Evidence())

    def test_observer_fidelity_and_complete_boundary_namespace_are_required(self):
        """Reject an unchecked observer and a room omitted from the thirty-three indexed bins."""
        common = self.common()
        confirmation = validation.validate_final_confirmation(common, validation.Evidence())
        path = Path(common["learning_record"])
        result = json.loads(path.read_bytes())
        result["arms"]["candidate"].pop("observer_agreement")
        self.write(path, result)
        with self.assertRaisesRegex(ValueError, "actual selected Executor"):
            validation.validate_short_learning(common, confirmation, validation.Evidence())
        path.write_bytes(self.original[path])
        result = json.loads(path.read_bytes())
        curves = Path(result["curves"]["path"])
        rows = [json.loads(line) for line in curves.read_bytes().splitlines()]
        rows[0]["boundary_room_bins"] = list(range(1, 33))
        rows[0]["boundary_room_histogram"] = [[128] + [0] * 31]
        curves.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        result["curves"]["sha256"] = self.fixture.sha(curves)
        self.write(path, result)
        with self.assertRaisesRegex(ValueError, "cover every observed"):
            validation.validate_short_learning(common, confirmation, validation.Evidence())

    def test_completed_study_review_must_precede_measurement_and_bind_every_raw_file(self):
        """A later review or changed raw case cannot relabel earlier work as post-study confirmation."""
        common = self.common()
        plan_path = Path(common["confirmation_record"]).parent / "measurement_plan.json"
        plan = json.loads(plan_path.read_bytes())
        binding = plan["completed_study"]
        with self.assertRaisesRegex(ValueError, "earlier independent review"):
            validation.verify_completed_study(binding, self.fixture.commit, self.fixture.runtime,
                validation.Evidence(), before_seconds=11.)
        proof = json.loads(Path(binding["proof"]["path"]).read_bytes())
        raw_path = Path(proof["evidence"][0]["path"])
        original = raw_path.read_bytes()
        try:
            raw_path.write_bytes(original + b'\n')
            with self.assertRaisesRegex(ValueError, "raw evidence changed"):
                validation.verify_completed_study(binding, self.fixture.commit, self.fixture.runtime, validation.Evidence())
        finally:
            raw_path.write_bytes(original)

    def test_completed_study_producer_is_owned_compute_only_and_preserves_previous_artifacts(self):
        """Run real metadata recomputation with only the synthetic allocation boundary mocked."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        output_directory = Path(temporary.name)
        registry = self.write(self.fixture.run / "study_registry.json", json.loads(Path(self.fixture.raw.specification["study_registry"]).read_bytes()))
        slurm = self.fixture.run / "slurm"
        slurm.mkdir(exist_ok=True)
        script = slurm / "synthetic_metadata_only.sbatch"
        script.write_text('#!/bin/sh\n# Synthetic metadata, never submitted.\n')
        (slurm / "submitted_jobids_uva.txt").write_text('123\n')
        (slurm / "submissions.tsv").write_text('123\tsynthetic_monitor\t' + self.fixture.commit + '\t' + str(script) + '\n')
        specification = {"source_commit": self.fixture.commit, "runtime_environment": self.fixture.runtime,
            "study_registry": str(registry), "helpers": self.fixture.receipt((*selected.HELPERS, "controller.py", "ideas.py")),
            "accepted_combinations": self.fixture.accepted_combination_inputs}
        with patch.object(controller, 'final_validation_allocation') as owned, patch.dict(os.environ, {"SLURM_JOB_ID": "123"}):
            proof = validation.produce_completed_study_proof(specification, output_directory / 'proof.json')
            owned.assert_called_once_with(self.fixture.run)
            with self.assertRaisesRegex(ValueError, "fresh exact"):
                validation.produce_completed_study_proof(specification, output_directory / 'proof.json')
        self.assertEqual(proof['investigated_hypotheses'], 50)
        self.assertEqual(proof['owner']['job_id'], '123')
        self.assertIs(proof['scientific_recomputation_on_owned_compute'], True)

    def test_differing_initial_source_can_keep_the_actual_last_accepted_source(self):
        """Retain a separately committed initial cohort while final source follows actual last-batch combinations."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        earlier = root / 'initial_source'
        subprocess.run(['git', 'clone', '-q', '--no-hardlinks', str(self.fixture.source), str(earlier)], check=True)
        subprocess.run(['git', '-C', str(earlier), '-c', 'user.name=Earlier metadata source',
            '-c', 'user.email=earlier@example.invalid', '-c', 'core.hooksPath=/dev/null',
            'commit', '--allow-empty', '-qm', 'Separate initial source identity'], check=True)
        earlier_commit = subprocess.check_output(['git', '-C', str(earlier), 'rev-parse', 'HEAD'], text=True).strip()
        subprocess.run(['git', '-C', str(earlier), 'checkout', '-q', '--detach', earlier_commit], check=True)
        registry = json.loads(Path(self.fixture.raw.specification['study_registry']).read_bytes())
        registry['reference_commit'] = earlier_commit
        changed = {}
        try:
            for entry in registry['initial_ideas']:
                for rows in entry['results'].values():
                    for summary in rows:
                        path = Path(summary['result_path'])
                        case_path, hardware_path = path.parent / 'case.json', path.parent / 'hardware.json'
                        for current in (path, case_path, hardware_path):
                            changed[current] = current.read_bytes()
                        case = json.loads(case_path.read_bytes())
                        case.update(source_root=str(earlier), source_commit=earlier_commit)
                        case.pop('specification_digest')
                        case['specification_digest'] = validation.canonical_digest(case)
                        result = json.loads(path.read_bytes())
                        result.update(source_commit=earlier_commit, specification_digest=case['specification_digest'])
                        hardware = json.loads(hardware_path.read_bytes())
                        hardware.update(source_root=str(earlier), source_commit=earlier_commit)
                        self.write(case_path, case); self.write(path, result); self.write(hardware_path, hardware)
            run = root / 'study'
            registry_path = self.write(run / 'study_registry.json', registry)
            slurm = run / 'slurm'
            slurm.mkdir()
            script = slurm / 'synthetic_owned_metadata.sbatch'
            script.write_text('#!/bin/sh\n# Synthetic metadata fixture; never submitted.\n')
            (slurm / 'submitted_jobids_uva.txt').write_text('123\n')
            (slurm / 'submissions.tsv').write_text('123\tsynthetic_monitor\t' + self.fixture.commit + '\t' + str(script) + '\n')
            output = root / 'proof'
            output.mkdir()
            specification = {'source_commit': self.fixture.commit, 'runtime_environment': self.fixture.runtime,
                'study_registry': str(registry_path), 'helpers': self.fixture.receipt((*selected.HELPERS, 'controller.py', 'ideas.py')),
                'accepted_combinations': self.fixture.accepted_combination_inputs}
            with patch.object(controller, 'final_validation_allocation'), patch.dict(os.environ, {'SLURM_JOB_ID': '123'}):
                proof = validation.produce_completed_study_proof(specification, output / 'proof.json')
            self.assertNotEqual(earlier_commit, self.fixture.commit)
            self.assertEqual(proof['source_commit'], self.fixture.commit)
            self.assertTrue(all(value['source_commit'] == self.fixture.commit for value in proof['accepted_combinations'].values()))
            bad = copy.deepcopy(specification)
            bad['source_commit'] = earlier_commit
            with patch.object(controller, 'final_validation_allocation'), patch.dict(os.environ, {'SLURM_JOB_ID': '123'}):
                with self.assertRaisesRegex(ValueError, 'last completed batch'):
                    validation.produce_completed_study_proof(bad, output / 'never_published.json')
            accepted_path = Path(specification['accepted_combinations']['h100']['confirmation_record'])
            accepted = json.loads(accepted_path.read_bytes())
            changed[accepted_path] = accepted_path.read_bytes()
            accepted['source_commit'] = earlier_commit
            self.write(accepted_path, accepted)
            with patch.object(controller, 'final_validation_allocation'), patch.dict(os.environ, {'SLURM_JOB_ID': '123'}):
                with self.assertRaisesRegex(ValueError, 'source or hardware context'):
                    validation.produce_completed_study_proof(specification, output / 'never_published.json')
        finally:
            for path, raw in changed.items():
                path.write_bytes(raw)

    def test_exact_eight_memory_evidence_is_required_for_not_applicable(self):
        """A smaller packing failure or throughput-selected count does not establish eight-copy capacity."""
        specification = copy.deepcopy(self.fixture.specification)
        selected_configuration = json.loads(Path(self.common("jaguar03_rtx_a4500")["confirmation_record"]).read_bytes())["candidate"]
        point = specification["fixed_total_scaling"]["eligibility_record"]
        record = json.loads(Path(point).read_bytes())
        record["configuration"]["learner"]["n_copies"] = 4
        self.write(point, record)
        with self.assertRaises(ValueError):
            validation.validate_fixed_total_scaling(specification, selected_configuration, validation.Evidence())

    def test_exact_eight_eligibility_cannot_use_an_old_or_different_physical_group(self):
        """The exact-eight result must follow proof review and use the selected physical/backend/asset group."""
        specification = self.fixture.specification
        selected_configuration = self.fixture.raw.configuration
        path = Path(specification["fixed_total_scaling"]["eligibility_record"])
        original = json.loads(path.read_bytes())
        changed = copy.deepcopy(original)
        changed["execution_identity"]["started_at_seconds"] = 60000.
        self.write(path, changed)
        with self.assertRaisesRegex(ValueError, "fresh after fifty"):
            validation.validate_fixed_total_scaling(specification, selected_configuration, validation.Evidence())
        self.write(path, original)
        backend = path.parent / "installed_backend.json"
        self.write(backend, {"different": "self-consistent but another installed backend"})
        changed = copy.deepcopy(original)
        changed["installed_backend"] = json.loads(backend.read_bytes())
        self.write(path, changed)
        with self.assertRaisesRegex(ValueError, "physical/backend/asset"):
            validation.validate_fixed_total_scaling(specification, selected_configuration, validation.Evidence())

    def test_measured_fixed_total_groups_match_global_seeds_total_work_and_actual_memory(self):
        """Compare eight policies on one card with the same eight on eight concurrent cards."""
        specification = {**self.fixture.specification, "fixed_total_scaling": {"status": "measured", "total_copies": 8,
            "single_gpu_record": str(self.fixture.fixed_groups[0]), "eight_gpu_record": str(self.fixture.fixed_groups[1])}}
        selected_configuration = self.fixture.raw.configuration
        result = validation.validate_fixed_total_scaling(specification, selected_configuration, validation.Evidence())
        self.assertEqual(result["median_fixed_total_speedup"], 2.)
        self.assertEqual(result["global_copy_seeds"], list(range(1, 9)))
        self.assertEqual(result["total_copies"], 8)
        path = self.fixture.fixed_groups[1]
        original = json.loads(path.read_bytes())
        for change in (lambda value: value["scaling_protocol"]["global_copy_seeds"].__setitem__(7, 1),
                       lambda value: value["configuration"]["learner"].__setitem__("rollout_unroll", 4),
                       lambda value: value["rounds"][0].__setitem__("seconds", .5)):
            changed = copy.deepcopy(original)
            change(changed)
            self.write(path, changed)
            with self.assertRaises(ValueError):
                validation.validate_fixed_total_scaling(specification, selected_configuration, validation.Evidence())
        specification["fixed_total_scaling"] = {"status": "not_applicable", "selected_copies": 1}
        with self.assertRaisesRegex(ValueError, "exact eight-copy"):
            validation.validate_fixed_total_scaling(specification, selected_configuration, validation.Evidence())

    def test_both_fixed_total_groups_cannot_replace_the_selected_backend(self):
        """Two internally matching layouts still need the full backend used by the selected confirmation."""
        specification = {**self.fixture.specification, "fixed_total_scaling": {"status": "measured", "total_copies": 8,
            "single_gpu_record": str(self.fixture.fixed_groups[0]), "eight_gpu_record": str(self.fixture.fixed_groups[1])}}
        for result_path in self.fixture.fixed_groups:
            for path in result_path.parent.glob("workers/*/installed_backend.json"):
                self.write(path, {"self_consistent": "a different installed backend"})
        with self.assertRaisesRegex(ValueError, "full backend"):
            validation.validate_fixed_total_scaling(specification, self.fixture.raw.configuration, validation.Evidence())

    def test_fixed_total_rounds_cannot_relabel_the_worker_uuid_or_physical_total(self):
        """A consistently edited round remains bound to its actual assigned card and raw memory.total query."""
        specification = {**self.fixture.specification, "fixed_total_scaling": {"status": "measured", "total_copies": 8,
            "single_gpu_record": str(self.fixture.fixed_groups[0]), "eight_gpu_record": str(self.fixture.fixed_groups[1])}}
        path = self.fixture.fixed_groups[1]
        worker_path = path.parent / "workers/physical_gpu_1/result.json"
        original, worker_original = json.loads(path.read_bytes()), json.loads(worker_path.read_bytes())
        for change_uuid in (True, False):
            result, worker = copy.deepcopy(original), copy.deepcopy(worker_original)
            for index, row in enumerate(worker["rounds"]):
                if change_uuid:
                    row["gpu_uuid"] = row["memory"]["gpu_uuid"] = "GPU-foreign"
                    for sample in row["memory"]["device_memory_samples"]:
                        sample["gpu_uuid"] = "GPU-foreign"
                else:
                    for sample in row["memory"]["device_memory_samples"]:
                        sample["total_bytes"] *= 2
                result["rounds"][index]["workers"][0] = copy.deepcopy(row)
            self.write(worker_path, worker)
            result["workers"][0]["worker_result_sha256"] = self.fixture.sha(worker_path)
            self.write(path, result)
            with self.subTest(change_uuid=change_uuid), self.assertRaisesRegex(ValueError, "round/sample UUID or physical memory.total"):
                validation.validate_fixed_total_scaling(specification, self.fixture.raw.configuration, validation.Evidence())

    def test_changed_prepared_helper_source_or_physical_gpu_is_rejected(self):
        """Reject changed staged bytes and internally inconsistent physical-GPU sidecars."""
        common = self.common()
        directory = Path(common["confirmation_record"]).parent
        helper = directory / "code/short_learning.py"
        helper.write_bytes(helper.read_bytes() + b'\n# changed synthetic staged helper\n')
        with self.assertRaisesRegex(ValueError, "staged final producer"):
            validation.validate_final_confirmation(common, validation.Evidence())
        helper.write_bytes(self.original[helper])
        hardware_path = directory / "hardware.json"
        hardware = json.loads(hardware_path.read_bytes())
        hardware["gpu_uuid"] = "GPU-foreign"
        self.write(hardware_path, hardware)
        with self.assertRaisesRegex(ValueError, "physical process"):
            validation.validate_final_confirmation(common, validation.Evidence())

    def test_complete_portable_confirmation_passes_without_original_paths(self):
        """Use exact copied byte mappings and a detached relocated Git root, with original paths unchanged."""
        common = self.common()
        original_evidence = validation.Evidence()
        expected = validation.validate_final_confirmation(common, original_evidence)
        with tempfile.TemporaryDirectory() as temporary:
            staging = Path(temporary)
            git_root = staging / "relocated_source"
            subprocess.run(["git", "clone", "-q", "--no-hardlinks", str(self.fixture.source), str(git_root)], check=True)
            subprocess.run(["git", "-C", str(git_root), "checkout", "-q", "--detach", self.fixture.commit], check=True)
            paths = {Path(item["path"]) for item in original_evidence.records()}
            paths |= {path for path in self.fixture.source.rglob('*') if path.is_file() and '.git' not in path.parts}
            entries = []
            for index, path in enumerate(sorted(paths)):
                relative = Path("copied") / (str(index) + '_' + path.name)
                destination = staging / relative
                destination.parent.mkdir(exist_ok=True)
                shutil.copyfile(path, destination)
                entries.append({"original_path": str(path), "relative_path": str(relative), "sha256": self.fixture.sha(path)})
            manifest = self.write(staging / "portable.json", {"schema_version": 1, "kind": "portable_raw_evidence", "files": entries,
                "git_roots": [{"original_root": str(self.fixture.source), "relative_path": "relocated_source", "commit": self.fixture.commit}]})
            resolver = portable_evidence.ExactPathResolver(manifest)
            self.assertEqual(validation.validate_final_confirmation(common, validation.Evidence(resolver)), expected)
            omitted = json.loads(manifest.read_bytes())
            omitted["files"] = [item for item in omitted["files"] if not item["original_path"].endswith('/code/short_learning.py')]
            self.write(manifest, omitted)
            with self.assertRaises(ValueError):
                validation.validate_final_confirmation(common, validation.Evidence(portable_evidence.ExactPathResolver(manifest)))


if __name__ == '__main__':
    unittest.main()
