"""Validate final evidence and training admission with synthetic standard-library records."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1]
SPECIFICATION = importlib.util.spec_from_file_location("post_comparison_records", SCRIPTS / "post_comparison_validation.py")
VALIDATION = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(VALIDATION)
SPECIFICATION = importlib.util.spec_from_file_location("post_comparison_defaults", SCRIPTS / "diagnose_gpu_execution.py")
DEFAULTS = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(DEFAULTS)


class PostComparisonTests(unittest.TestCase):
    """Exercise the actual final gate boundary without training or choosing study ideas."""

    def setUp(self):
        """Build complete synthetic evidence with explicit source, runtime and configuration identity."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.source = SCRIPTS.parents[2]
        self.commit = subprocess.check_output(["git", "-C", str(self.source), "rev-parse", "HEAD"], text=True).strip()
        self.execution_counter = 0
        self.runtime = {"JAX_ENABLE_X64": "true", "JAX_DEFAULT_MATMUL_PRECISION": "highest",
                        "XLA_FLAGS": "--xla_gpu_exclude_nondeterministic_ops=true"}
        defaults = DEFAULTS.default_configuration(self.source)
        self.configuration = {name: defaults[name] for name in ("learner", "environment")}
        self.configuration["learner"]["n_copies"] = 1
        self.configuration["learner"]["rollout_unroll"] = 2
        self.configuration["execution"] = {"compilation_mode": "whole_iteration", "iterations_per_call": 1,
                                            "report_readout": "none", "report_every_iterations": 10}
        self.checks = {"passed": True, "complete_iteration": {"leaves": 10},
            "supplied_inputs": {name: {"leaves": 10} for name in ("preparation", "fixed_batch", "fixed_batch_update")}}
        self.specification = {"source_commit": self.commit, "runtime_environment": self.runtime,
                              "study_registry": str(self.make_registry()), "contexts": {}}
        for context in VALIDATION.CONTEXTS:
            correctness = self.write(context + "_correctness.json", self.comparison(context))
            confirmation = self.write(context + "_confirmation.json", self.comparison(context))
            packing = [self.write(f"{context}_packing_{count}.json", self.packing(context, count, seconds, 100 + count))
                       for count, seconds in ((1, 1.), (2, 1.9))]
            repeated = [self.write(f"{context}_packing_repeat_{count}.json", self.packing(context, count, seconds, 200 + count))
                        for count, seconds in ((1, 1.), (2, 1.9))]
            self.specification["contexts"][context] = {"correctness_record": str(correctness),
                "confirmation_record": str(confirmation), "packing_records": list(map(str, packing)),
                "packing_confirmation_records": list(map(str, repeated)), "selected_copies": 1}
        first = self.write("single_gpu.json", self.scaling(1))
        eight = self.write("eight_gpu.json", self.scaling(8))
        self.specification["scaling"] = {"single_gpu_record": str(first), "eight_gpu_record": str(eight)}

    def write(self, name, value):
        """Store one independently hashed fixture as a complete JSON document."""
        path = self.directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n")
        return path

    def row(self, copies=1, seconds=1., gpu="GPU-one", iterations=10):
        """Represent ten complete updates and actual-style GPU memory fields."""
        return {"iterations": iterations, "copies": copies, "environments_per_copy": 128, "rollout_steps": 128,
                "training_transitions": copies * 128 * 128 * iterations, "seconds": seconds,
                "memory": {"other_gpu_processes": [], "gpu_uuid": gpu,
                    "device_memory_samples": [{"gpu_uuid": gpu, "total_bytes": 1000 * 2 ** 20,
                        "used_bytes": 10 * 2 ** 20, "free_bytes": 990 * 2 ** 20} for _ in range(2)],
                    "sampled_process_peak_bytes": 1024, "allocator": {"peak_bytes_in_use": 1024}}}

    def comparison(self, context):
        """Represent separate successful complete-combination paired measurements."""
        reference = copy.deepcopy(self.configuration)
        reference["learner"]["rollout_unroll"] = 1
        self.execution_counter += 1
        result = {"source_commit": self.commit, "hardware_class": context, "runtime_environment": self.runtime,
            "copies": 1, "candidate": copy.deepcopy(self.configuration), "reference": reference,
            "implementation_distinct": True, "correctness_attempted": True, "correctness_passed": True,
            "correctness": self.checks, "measurement_valid": True, "contaminated": False, "decision": "accepted"}
        result["execution_identity"] = {"hostname": context, "pid": 1000 + self.execution_counter,
            "process_start_time": str(self.execution_counter), "started_at_seconds": 1000. + 10 * self.execution_counter,
            "finished_at_seconds": 1002. + 10 * self.execution_counter}
        result["rounds"] = [{"order": ["reference", "candidate"] if index % 2 == 0 else ["candidate", "reference"],
                             "reference": self.row(seconds=2.), "candidate": self.row()} for index in range(7)]
        return result

    def packing(self, context, copies, seconds, pid):
        """Give each packing point a new process and otherwise identical implementation."""
        configuration = copy.deepcopy(self.configuration)
        configuration["learner"]["n_copies"] = copies
        hostname, device = ("serval02", "NVIDIA H100 NVL") if context == "h100" else ("jaguar03", "NVIDIA RTX A4500")
        return {"source_commit": self.commit, "hardware_class": context, "runtime_environment": self.runtime,
            "configuration": configuration, "completed": True, "isolated_variant": True,
            "normalization_warmup_iterations": 50, "copy_seeds": list(range(1, copies + 1)),
            "hardware": {"hostname": hostname, "pid": pid, "process_start_time": str(pid),
                         "gpu_uuid": "GPU-one", "gpu_query": f"GPU-one, {device}, 1000 MiB, driver-version"},
            "rounds": [self.row(copies, seconds) for _ in range(7)]}

    def scaling(self, cards):
        """Use synchronized wall windows and distinct per-GPU policies, not summed worker rates."""
        rounds = []
        for _ in range(7):
            workers = [{**self.row(gpu=f"GPU-{index}"), "gpu_uuid": f"GPU-{index}",
                        "copy_seeds": [index + 1],
                        "start_time_seconds": 100., "end_time_seconds": 101.} for index in range(cards)]
            rounds.append({"measurement_scope": "synchronized_gpu_group_wall_time", "seconds": 1.,
                           "start_time_seconds": 100., "end_time_seconds": 101., "workers": workers})
        return {"source_commit": self.commit, "hardware_class": "jaguar03_rtx_a4500",
                "runtime_environment": self.runtime, "configuration": self.configuration,
                "gpu_count": cards, "rounds": rounds}

    def make_registry(self):
        """Represent fifty fixture mechanisms completed at both counts on both contexts."""
        entries = []
        for index in range(50):
            name = f"synthetic_mechanism_{index}"
            results = {}
            for context in VALIDATION.CONTEXTS:
                results[context] = []
                for copies in (1, 2):
                    record = self.comparison(context)
                    window = (1, 5, 10)[index % 3]
                    for variant in ("reference", "candidate"):
                        record[variant]["learner"]["n_copies"] = copies
                    case = {"source_root": str(self.source), "source_commit": self.commit,
                        "campaign_run_dir": str(self.directory), "hypothesis": name, "unit_id": f"result_{index}_{context}_{copies}",
                        "hardware_class": context, "configuration": record["reference"]["learner"],
                        "candidate": {"learner": {"rollout_unroll": 2}, "environment": {}, "execution": {}},
                        "reference_execution": None, "rounds": 7, "iterations": window}
                    case["specification_digest"] = VALIDATION.canonical_digest(case)
                    record.update(idea=name, copies=copies, case_completed=True, unit_id=case["unit_id"],
                                  specification_digest=case["specification_digest"])
                    record["rounds"] = [{"order": ["reference", "candidate"] if step % 2 == 0 else ["candidate", "reference"],
                        "reference": self.row(copies, 2., iterations=window),
                        "candidate": self.row(copies, 1., iterations=window)} for step in range(7)]
                    path = self.write(case["unit_id"] + "/result.json", record)
                    self.write(case["unit_id"] + "/case.json", case)
                    self.write(case["unit_id"] + "/hardware.json", {"source_root": str(self.source), "source_commit": self.commit,
                        "hardware_class": context, "pid": record["execution_identity"]["pid"],
                        "process_start_time": record["execution_identity"]["process_start_time"]})
                    results[context].append({"copies": copies, "result_path": str(path)})
            entries.append({"name": name, "results": results})
        registry = {"reference_commit": self.commit, "initial_matched_copy_counts": [1, 2],
            "initial_ideas": entries[:30], "adaptive_batches": [{"reference_commit": self.commit,
                "matched_copy_counts": [1, 2], "hypotheses": entries[30 + 5 * index:35 + 5 * index]} for index in range(4)]}
        return self.write("registry.json", registry)

    def test_all_five_gates_are_derived_from_complete_evidence(self):
        """Require original comparison, packing, scaling and all matched study records."""
        result = VALIDATION.validate_final_verification(self.specification)
        self.assertTrue(all(result[name] for name in VALIDATION.FINAL_FLAGS))
        self.assertEqual(result["selected"]["h100"]["copies"], 1)
        self.assertEqual(result["scaling"]["median_aggregate_speedup"], 8.)
        self.assertGreater(len(result["evidence"]), 200)
        self.assertNotIn("jax", sys.modules)

    def test_combination_needs_separate_longer_confirmation(self):
        """A reused correctness file or one-update comparison cannot satisfy final confirmation."""
        context = self.specification["contexts"]["h100"]
        context["confirmation_record"] = context["correctness_record"]
        with self.assertRaisesRegex(ValueError, "separate"):
            VALIDATION.validate_final_verification(self.specification)

        path = self.write("short_confirmation.json", self.comparison("h100"))
        record = json.loads(path.read_text())
        record["rounds"][0]["candidate"]["iterations"] = 1
        record["rounds"][0]["reference"]["iterations"] = 1
        path.write_text(json.dumps(record))
        context["confirmation_record"] = str(path)
        with self.assertRaisesRegex(ValueError, "ten complete"):
            VALIDATION.validate_final_verification(self.specification)

    def test_copied_combination_file_is_not_an_independent_execution(self):
        """Different filenames cannot disguise an identical PID, birth counter and recorded run."""
        context = self.specification["contexts"]["h100"]
        Path(context["confirmation_record"]).write_bytes(Path(context["correctness_record"]).read_bytes())
        with self.assertRaisesRegex(ValueError, "same process"):
            VALIDATION.validate_final_verification(self.specification)

    def test_success_flags_and_empty_rounds_cannot_count_as_investigated(self):
        """A registry marker cannot replace actual correctness reports and measured per-case work."""
        registry = json.loads(Path(self.specification["study_registry"]).read_text())
        path = Path(registry["initial_ideas"][0]["results"]["h100"][0]["result_path"])
        original = json.loads(path.read_text())
        for changed in ({**original, "rounds": [{} for _ in range(7)]},
                        {**original, "correctness": {"passed": True}}, {**original, "case_completed": False}):
            path.write_text(json.dumps(changed))
            with self.subTest(keys=list(changed)), self.assertRaises(ValueError):
                VALIDATION.validate_final_verification(self.specification)

    def test_cohort_execution_reconstruction_matches_the_frozen_harness(self):
        """Full candidate options override reference controls; partial options inherit as D28 actually does."""
        registry = json.loads(Path(self.specification["study_registry"]).read_text())
        path = Path(registry["initial_ideas"][0]["results"]["h100"][0]["result_path"])
        original_case = json.loads((path.parent / "case.json").read_text())
        original_result = json.loads(path.read_text())
        reference = {"compilation_mode": "separate_phases", "iterations_per_call": 1,
                     "report_readout": "none", "report_every_iterations": 10}
        cases = [{"compilation_mode": "whole_iteration", "iterations_per_call": 5,
                  "report_readout": "none", "report_every_iterations": 10},
                 {"report_readout": "scheduled", "report_every_iterations": 5}]
        for changes in cases:
            case, result = copy.deepcopy(original_case), copy.deepcopy(original_result)
            case.update(reference_execution=reference, iterations=5)
            case["candidate"]["execution"] = changes
            case["specification_digest"] = VALIDATION.canonical_digest({key: value for key, value in case.items()
                                                                       if key != "specification_digest"})
            result["reference"]["execution"] = reference
            result["candidate"]["execution"] = {**reference, **changes}
            result["specification_digest"] = case["specification_digest"]
            result["rounds"] = [{"order": ["reference", "candidate"] if step % 2 == 0 else ["candidate", "reference"],
                "reference": self.row(1, 2., iterations=5), "candidate": self.row(1, 1., iterations=5)} for step in range(7)]
            path.write_text(json.dumps(result))
            (path.parent / "case.json").write_text(json.dumps(case))
            with self.subTest(changes=changes):
                proof = VALIDATION.validate_cohort_case(path, result["idea"], "h100", 1, self.commit, VALIDATION.Evidence())
                self.assertEqual(proof["scope"], "actual_runtime")

    def test_legacy_runtime_proof_preserves_old_records_and_allows_ledger_appends(self):
        """Tag uncaptured environments as inferred only after immutable source/gate/launch checks."""
        registry = json.loads(Path(self.specification["study_registry"]).read_text())
        path = Path(registry["initial_ideas"][0]["results"]["h100"][0]["result_path"])
        record = json.loads(path.read_text())
        record.pop("runtime_environment")
        path.write_text(json.dumps(record))
        original = path.read_bytes()
        hardware_path = path.parent / "hardware.json"
        hardware = json.loads(hardware_path.read_text())
        hardware.update(worker_pid=7000, worker_start_time="7000")
        hardware_path.write_text(json.dumps(hardware))
        script = self.directory / "legacy_worker.sh"
        worker = self.source / "10_jax_exploration_platform/scripts/atari_pixel_campaign/worker_environment.sh"
        script.write_text("source " + str(worker) + "\n")
        parent = {"pid": 7000, "start_time": "7000", "source_commit": self.commit, "script": str(script)}
        ledger = self.write("processes.json", [parent])
        checks = {}
        platform = self.source / "10_jax_exploration_platform"
        for kind in ("learner", "renderer"):
            command = (["/usr/bin/python3", "-m", "pytest", "-q", str(platform / "tests/atari_pixels"),
                        str(platform / "tests/test_atari_pixel_environment.py")] if kind == "learner" else
                       ["/usr/bin/python3", str(platform / "scripts/atari_pixel_campaign/check_rendering.py"), "--candidate", "all"])
            check = self.write("legacy_" + kind + ".json", {"passed": True, "exit_code": 0,
                "source_commit": self.commit, "command": command, "hardware": {"hostname": "serval02",
                    "gpu_query": "GPU-one, NVIDIA H100 NVL, 1000 MiB, driver", "runtime_environment": self.runtime}})
            checks[kind] = {"path": str(check), "sha256": hashlib.sha256(check.read_bytes()).hexdigest()}
        self.write("data/source_checks/" + self.commit + "/h100.json", {"passed": True,
            "source_commit": self.commit, "hardware_class": "h100", "checks": checks})
        proof = VALIDATION.validate_cohort_case(path, record["idea"], "h100", 1, self.commit, VALIDATION.Evidence())
        self.assertEqual(proof["scope"], "inferred_source_protocol")
        self.assertIn("unknown", proof["limitation"])
        ledger.write_text(json.dumps([parent, {"pid": 7001, "start_time": "7001"}]))
        repeated = VALIDATION.validate_cohort_case(path, record["idea"], "h100", 1, self.commit, VALIDATION.Evidence())
        self.assertEqual(proof, repeated)
        self.assertEqual(path.read_bytes(), original)

    def test_packing_reports_next_doubling_until_a_limit_exists(self):
        """Twofold throughput growth requires another measured copy count."""
        context = self.specification["contexts"]["h100"]
        path = Path(context["packing_records"][1])
        record = self.packing("h100", 2, 1., 102)
        path.write_text(json.dumps(record))
        result = VALIDATION.validate_packing({**context, "hardware_class": "h100", "source_commit": self.commit,
            "runtime_environment": self.runtime}, VALIDATION.Evidence())
        self.assertFalse(result["complete"])
        self.assertEqual(result["next_copies"], 4)

    def test_missing_matched_count_cannot_be_hidden_by_registry_flags(self):
        """Every count is checked against an original result rather than summary success fields."""
        registry = json.loads(Path(self.specification["study_registry"]).read_text())
        registry["adaptive_batches"][-1]["hypotheses"][-1]["results"]["h100"] = []
        self.write("registry.json", registry)
        with self.assertRaisesRegex(ValueError, "unfinished mechanism"):
            VALIDATION.validate_final_verification(self.specification)

    def test_packing_requires_independent_repetition_and_the_smaller_count(self):
        """A first low gain does not complete packing, and repetition cannot select the larger count."""
        context = self.specification["contexts"]["h100"]
        common = {**context, "hardware_class": "h100", "source_commit": self.commit,
                  "runtime_environment": self.runtime}
        common.pop("packing_confirmation_records")
        result = VALIDATION.validate_packing(common, VALIDATION.Evidence())
        self.assertFalse(result["complete"])
        self.assertEqual(result["required_confirmation_counts"], [1, 2])
        common["packing_confirmation_records"] = context["packing_confirmation_records"]
        common["selected_copies"] = 2
        with self.assertRaisesRegex(ValueError, "smaller"):
            VALIDATION.validate_packing(common, VALIDATION.Evidence())

    def test_disagreeing_packing_repeat_stays_incomplete(self):
        """An independently repeated large gain cannot be labeled a confirmed packing limit."""
        context = self.specification["contexts"]["h100"]
        path = Path(context["packing_confirmation_records"][1])
        path.write_text(json.dumps(self.packing("h100", 2, 1., 202)))
        result = VALIDATION.validate_packing({**context, "hardware_class": "h100", "source_commit": self.commit,
            "runtime_environment": self.runtime}, VALIDATION.Evidence())
        self.assertFalse(result["complete"])
        self.assertIn("disagreeing", result["next_action"])

    def test_packing_requires_actual_memory_headroom(self):
        """The observed process peak must leave fifteen percent of queried physical GPU memory free."""
        context = self.specification["contexts"]["h100"]
        path = Path(context["packing_records"][1])
        record = json.loads(path.read_text())
        for sample in record["rounds"][0]["memory"]["device_memory_samples"]:
            sample.update(used_bytes=900 * 2 ** 20, free_bytes=100 * 2 ** 20)
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "fifteen percent"):
            VALIDATION.validate_final_verification(self.specification)

    def test_duplicate_scaling_gpu_or_rate_sum_is_rejected(self):
        """Eight workers on one GPU or missing synchronized wall time cannot satisfy scaling."""
        path = Path(self.specification["scaling"]["eight_gpu_record"])
        for kind in ("duplicate", "summed"):
            record = self.scaling(8)
            if kind == "duplicate":
                record["rounds"][0]["workers"][1]["gpu_uuid"] = "GPU-0"
            else:
                record["rounds"][0]["measurement_scope"] = "sum_of_worker_rates"
            path.write_text(json.dumps(record))
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                VALIDATION.validate_final_verification(self.specification)

    def test_scaling_requires_finite_consistent_worker_times(self):
        """NaN group times and a worker duration inconsistent with its bounds cannot pass scaling."""
        path = Path(self.specification["scaling"]["eight_gpu_record"])
        for kind in ("nan", "duration"):
            record = self.scaling(8)
            if kind == "nan":
                record["rounds"][0]["start_time_seconds"] = float("nan")
            else:
                record["rounds"][0]["workers"][0]["end_time_seconds"] = 100.5
            path.write_text(json.dumps(record))
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                VALIDATION.validate_final_verification(self.specification)

    def test_admission_revalidates_hashes_and_binds_exact_runtime(self):
        """Changed raw evidence and a different compiler runtime cannot reuse an admitted final receipt."""
        receipt = VALIDATION.validate_final_verification(self.specification)
        final = self.write("final_verification.json", receipt)
        training = self.write("training.json", {"schema_version": 1, "source_root": str(self.source),
            "source_commit": self.commit, "hardware_class": "h100", "configuration": self.configuration,
            "runtime_environment": self.runtime, "copy_seeds": [1], "final_verification": str(final)})
        original_query = VALIDATION.subprocess.check_output
        def source_query(command, **options):
            """Retain actual source/Git reads while suppressing unrelated test-tree status changes."""
            return "" if "status" in command else original_query(command, **options)
        with patch.object(VALIDATION.subprocess, "check_output", side_effect=source_query):
            admitted = VALIDATION.validate_training_admission(training)
        self.assertEqual(admitted["validation"]["selected_copies"], 1)
        path = Path(self.specification["contexts"]["h100"]["confirmation_record"])
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            VALIDATION.validate_training_admission(training)

    def test_completion_requires_exact_budget_and_checkpoint_bytes(self):
        """Verify the finished checkpoint through its readable sidecar without unpickling arrays."""
        admitted = {"source_root": str(self.source), "source_commit": self.commit, "hardware_class": "h100",
                    "configuration": self.configuration, "runtime_environment": self.runtime, "copy_seeds": [1]}
        admitted = json.loads(json.dumps(admitted))
        identity = {**admitted, "backend": {"device_kind": "H100"}, "helpers": {}}
        self.write("training_identity.json", identity)
        checkpoint = self.directory / "checkpoints/latest.pkl"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"complete-state-checkpoint-fixture")
        target = 2_000_000_000 // (128 * 128)
        sidecar = {"kind": "optimized_training_checkpoint", "identity_digest": VALIDATION.canonical_digest(identity),
            "configuration_digest": VALIDATION.canonical_digest(self.configuration), "source_commit": self.commit,
            "iteration": target, "target_updates": target, "steps_per_copy": target * 128 * 128, "copy_seeds": [1],
            "checkpoint_path": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
        (checkpoint.parent / "latest.metadata.json").write_text(json.dumps(sidecar))
        self.write("result.json", {"completed": True, "update": target, "target_updates": target,
            "requested_steps_per_copy": 2_000_000_000, "steps_per_copy": target * 128 * 128,
            "copy_seeds": [1], "copies": 1, "identity_digest": VALIDATION.canonical_digest(identity)})
        with patch.object(VALIDATION, "validate_training_admission", return_value=admitted):
            result = VALIDATION.validate_training_completion(self.directory / "training.json", self.directory)
            self.assertEqual(result["steps_per_copy"], 1_999_994_880)
            checkpoint.write_bytes(b"different checkpoint")
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                VALIDATION.validate_training_completion(self.directory / "training.json", self.directory)
        self.assertNotIn("jax", sys.modules)


if __name__ == "__main__":
    unittest.main()
