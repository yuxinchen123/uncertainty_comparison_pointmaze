"""Exercise future admission revisions using temporary metadata and no numerical imports."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import advance_adaptive_research as advance
import controller
import diagnose_gpu_execution as diagnostic
import ideas


class AdaptiveRevisionTests(unittest.TestCase):
    """Test immutable preparation, explicit registration, evidence and interrupted publication."""

    def setUp(self):
        """Build synthetic full-size records; mock only Git, existing source gates and prior results."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name)
        self.root = SCRIPTS.parents[2]
        self.commit = subprocess.check_output(["git", "-C", str(self.root), "rev-parse", "HEAD"], text=True).strip()
        self.runtime = {"JAX_ENABLE_X64": "true", "JAX_DEFAULT_MATMUL_PRECISION": "highest",
                        "XLA_FLAGS": "--xla_gpu_exclude_nondeterministic_ops=true"}
        defaults = diagnostic.default_configuration(self.root)
        self.selected = json.loads(json.dumps({"learner": defaults["learner"], "environment": defaults["environment"],
            "execution": {"compilation_mode": "whole_iteration", "iterations_per_call": 1,
                          "report_readout": "none", "report_every_iterations": 10}}))
        self.selected["learner"]["rollout_unroll"] = 2
        self.reference = copy.deepcopy(self.selected)
        self.reference["learner"]["rollout_unroll"] = 1
        self.core = {"source_root": str(self.root), "source_commit": self.commit, "files": {}}
        self.counter = 100
        self.helper_bytes = {name: ("# synthetic committed " + name + "\n").encode() for name in (
            "measure_selected.py", "diagnose_gpu_execution.py", "post_comparison_validation.py", "infrastructure_retry.py", "pregather_async_retry.py", "train_optimized.py")}
        self.helpers = {"source_root": str(self.root), "source_commit": self.commit,
                        "files": {advance.SCRIPT_PREFIX + name: advance.sha256(contents)
                                  for name, contents in self.helper_bytes.items()}}
        self.paired = {"source_root": str(self.root), "source_commit": self.commit,
                       "files": {advance.SCRIPT_PREFIX + name: "a" * 64 for name in ("paired_benchmark.py", "ideas.py")}}
        self.write("slurm/submissions.tsv", "")
        self.write("processes.json", [])
        self.write("worker_stop.json", {"schema_version": 1, "workers": [], "jobs": [], "processes": [], "runners": []})
        controller.layout(self.run)
        self.registry = ideas.new_registry(self.commit, self.root / "10_jax_exploration_platform")
        self.registry["initial_matched_copy_counts"] = [1, 2]
        for entry in self.registry["initial_ideas"]:
            entry["results"] = {context: [{"copies": count, "result_path": str(self.run / f"old/{entry['name']}_{context}_{count}.json")}
                                        for count in (1, 2)] for context in advance.validation.CONTEXTS}
        self.write("study_registry.json", self.registry)
        old_gates = {context: str(self.write(f"old/{context}_gate.json", {"passed": True, "source_commit": "b" * 40}))
                     for context in advance.validation.CONTEXTS}
        old_configuration = self.write("old/configuration.json", self.reference["learner"])
        self.old_ready = {"authorized": True, "source_root": str(self.root), "source_commit": "b" * 40,
            "contexts": list(advance.validation.CONTEXTS), "matched_copy_counts": [1, 2], "seed_start": 1,
            "gate_records": old_gates, "reference_configuration_path": str(old_configuration)}
        self.write(controller.SENTINEL_NAME, self.old_ready)
        # Preserve deliberately different whitespace to check byte copies rather than reserialization.
        (self.run / controller.SENTINEL_NAME).write_text(json.dumps(self.old_ready, separators=(",", ":")) + "\n\n")
        self.specification = {"schema_version": 1, "source_root": str(self.root), "source_commit": self.commit,
            "matched_copy_counts": [1, 2], "runtime_environment": self.runtime,
            "worker_stop_record": str(self.run / "worker_stop.json"), "contexts": {},
            "hypotheses": [self.hypothesis(index) for index in range(5)]}
        for context in advance.validation.CONTEXTS:
            first = self.measurement(context, "correctness", 1)
            second = self.measurement(context, "confirmation", 1)
            profile = self.profile(context)
            capacities = [str(self.measurement(context, "packing-point", copies)) for copies in (1, 2)]
            checks = {kind + "_record": str(self.write(f"checks/{context}_{kind}.json", {"passed": True}))
                      for kind in ("learner", "renderer")}
            self.specification["contexts"][context] = {**checks, "correctness_record": str(first),
                "confirmation_record": str(second), "profile_record": str(profile), "capacity_records": capacities}
        self.specification_path = self.write("proposed_revision.json", self.specification)
        self.gate = mock.patch.object(advance.admission, "verified_check", side_effect=lambda path, *args:
            {"path": str(path), "sha256": advance.sha256(Path(path).read_bytes())}).start()
        self.prior = mock.patch.object(advance.validation, "validate_cohort_case", return_value={}).start()
        mock.patch.object(advance, "source_identity", return_value=self.core).start()
        mock.patch.object(advance, "committed_files").start()
        mock.patch.object(advance, "launcher_capability").start()
        mock.patch.object(advance, "require_validation_allocation").start()
        self.addCleanup(mock.patch.stopall)

    def write(self, relative, value):
        """Store text or JSON fixture bytes below a temporary campaign."""
        path = self.run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if isinstance(value, str) else json.dumps(value, indent=2) + "\n")
        return path

    def hypothesis(self, index):
        """Give each synthetic caller proposal a distinct implemented combination."""
        return {"name": f"synthetic_distinct_mechanism_{index}", "hypothesis": f"Synthetic hypothesis {index}",
            "mechanism": f"Synthetic independent mechanism number {index}",
            "learner": {"rollout_unroll": index + 3, "update_unroll": 2}, "environment": {}, "execution": {},
            "correctness_checks": ["compare all supplied-input leaves"],
            "measurement_settings": {"paired_rounds": 7, "minimum_updates_per_round": 1},
            "acceptance_criteria": dict(advance.ACCEPTANCE_CRITERIA)}

    def row(self, copies, seconds, gpu):
        """Represent ten complete updates and independently observed physical-device memory."""
        return {"iterations": 10, "copies": copies, "environments_per_copy": 128, "rollout_steps": 128,
            "training_transitions": copies * 128 * 128 * 10, "seconds": seconds,
            "memory": {"gpu_uuid": gpu, "other_gpu_processes": [], "sampled_process_peak_bytes": 1024,
                "allocator": {"peak_bytes_in_use": 1024}, "device_memory_samples": [
                    {"gpu_uuid": gpu, "total_bytes": 1000 * 2 ** 20, "used_bytes": 10 * 2 ** 20,
                     "free_bytes": 990 * 2 ** 20} for _ in range(2)]}}

    def unit(self, context, kind, copies):
        """Prepare one unique original process, GPU identity and full immutable plan."""
        self.counter += 1
        unit = f"synthetic_{context}_{kind}_{copies}_{self.counter}"
        directory = self.run / "data" / unit
        gpu = "GPU-synthetic-" + context
        hardware = {"hostname": "jaguar03" if context == "jaguar03_rtx_a4500" else "serval02",
            "pid": self.counter, "process_start_time": str(self.counter * 100), "gpu_uuid": gpu,
            "runtime_environment": {**self.runtime, "JAX_ENABLE_COMPILATION_CACHE": "false",
                                    "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": ""},
            "gpu_query": gpu + (", NVIDIA RTX A4500, 1000 MiB" if context == "jaguar03_rtx_a4500"
                                  else ", NVIDIA H100 NVL, 1000 MiB")}
        selected = copy.deepcopy(self.selected)
        selected["learner"]["n_copies"] = copies
        plan = {"schema_version": 1, "kind": "selected_configuration_measurement", "mode": kind,
            "source_root": str(self.root), "source_commit": self.commit, "core": self.core,
            "paired_source": self.paired, "helpers": self.helpers, "configuration": selected,
            "reference_configuration": self.reference if kind != "packing-point" else None,
            "runtime_environment": self.runtime, "hardware_class": context, "gpu_uuid": gpu,
            "copy_seeds": list(range(1, copies + 1)), "rounds": 7, "iterations": 10,
            "minimum_updates_per_round": 10,
            "campaign_run_dir": str(self.run), "unit_id": unit, "compilation_cache_policy": "disabled"}
        execution = {key: hardware[key] for key in ("hostname", "pid", "process_start_time")}
        execution.update(started_at_seconds=1000 + self.counter * 10, finished_at_seconds=1005 + self.counter * 10)
        self.write(directory / "hardware.json", hardware)
        for name, contents in self.helper_bytes.items():
            self.write(directory / "code" / name, contents.decode())
        return directory, hardware, plan, execution

    def finish(self, directory, plan, result):
        """Bind a fixture result to the exact prepared-plan bytes."""
        path = self.write(directory / "measurement_plan.json", plan)
        self.write(directory / "measurement_prepared.json", {"plan_sha256": advance.sha256(path.read_bytes())})
        return self.write(directory / "result.json", result)

    def measurement(self, context, mode, copies):
        """Create the producer's separate correctness, confirmation or isolated-capacity shape."""
        directory, hardware, plan, execution = self.unit(context, mode, copies)
        common = {"source_commit": self.commit, "hardware_class": context, "runtime_environment": self.runtime,
            "measurement_mode": mode, "helpers": self.helpers, "copy_seeds": plan["copy_seeds"],
            "protocol": {"rounds": 7, "requested_minimum_updates_per_round": 10, "actual_updates_per_round": 10},
            "execution_identity": execution, "copies": copies}
        if mode == "packing-point":
            result = {**common, "completed": True, "isolated_variant": True, "configuration": plan["configuration"],
                "normalization_warmup_iterations": 50, "hardware": hardware,
                "rounds": [self.row(copies, 1., plan["gpu_uuid"]) for _ in range(7)]}
        else:
            checks = {"passed": True, "complete_iteration": {"leaves": 10}, "supplied_inputs": {
                name: {"leaves": 10} for name in ("preparation", "fixed_batch", "fixed_batch_update")}}
            result = {**common, "reference": self.reference, "candidate": plan["configuration"],
                "implementation_distinct": True, "correctness_passed": True, "correctness_attempted": True,
                "correctness": checks, "measurement_valid": True, "contaminated": False, "decision": "accepted",
                "rounds": [{"order": ["reference", "candidate"] if index % 2 == 0 else ["candidate", "reference"],
                    "reference": self.row(copies, 2., plan["gpu_uuid"]), "candidate": self.row(copies, 1., plan["gpu_uuid"])}
                    for index in range(7)] if mode == "confirmation" else []}
        return self.finish(directory, plan, result)

    def profile(self, context):
        """Create a dedicated selected-baseline profile with real nonempty synthetic CSV bytes."""
        directory, hardware, plan, execution = self.unit(context, "profile", 1)
        plan["kind"] = "selected_configuration_profile"
        plan["profile_updates"] = 1
        report = self.write(directory / "trace/capture.nsys-rep", "synthetic raw capture")
        tool = self.write("synthetic_nsys", "synthetic tool bytes")
        plan["profile_tool"] = {"path": str(tool), "version": "synthetic version", "sha256": advance.sha256(tool.read_bytes())}
        summaries = {}
        for name, column in (("cuda_gpu_kern_sum", "Instances"), ("cuda_api_sum", "Num Calls")):
            path = self.write(directory / "trace" / (name + ".csv"), f"Name,{column},Total Time (ns)\nsynthetic,{1},{200}\n")
            summaries[name] = {"path": str(path), **diagnostic.summary_csv_evidence(path, column)}
        analysis = self.write(directory / "trace/analysis.json", {"completed": True, "exit_code": 0,
            "report": str(report), "summaries": summaries})
        result = {**plan, "completed": True, "hardware": hardware, "execution_identity": execution,
            "profile_updates": 1, "profile_start_iteration": 52, "profile_end_iteration": 53,
            "capture_completed": True, "normalization_warmup_iterations": 50,
            "compiled_signatures_before": [1], "compiled_signatures_after": [1],
            "compilation_warmup": [{"call": index, "seconds": 1., "updates": 1,
                                    "compiled_signatures": [1]} for index in (1, 2)],
            "trace_analysis_record": str(analysis), "trace_report_sha256": advance.sha256(report.read_bytes()),
            "profile_tool": {"path": str(tool), "version": "synthetic version", "sha256": advance.sha256(tool.read_bytes())}}
        return self.finish(directory, plan, result)

    def save_specification(self):
        """Persist an explicit changed caller specification for a rejection test."""
        self.write(self.specification_path, self.specification)

    def test_prepare_is_idempotent_preserves_original_bytes_and_stages_all_five(self):
        """Create twenty complete proposed cases while changing no live admission or registry bytes."""
        sentinel = (self.run / controller.SENTINEL_NAME).read_bytes()
        registry = (self.run / "study_registry.json").read_bytes()
        revision = advance.prepare(self.run, self.specification_path)
        self.assertEqual(advance.prepare(self.run, self.specification_path), revision)
        self.assertEqual((self.run / controller.SENTINEL_NAME).read_bytes(), sentinel)
        self.assertEqual((self.run / "study_registry.json").read_bytes(), registry)
        self.assertEqual((revision / "history/SOURCE_AND_GATES_READY.json").read_bytes(), sentinel)
        self.assertEqual(len(list((revision / "staged/cases").glob("*.json"))), 20)
        self.assertFalse(list((self.run / "queue/pending").glob("*.json")))
        self.assertEqual(self.prior.call_count, 120)
        for path in self.old_ready["gate_records"].values():
            contents = Path(path).read_bytes()
            self.assertEqual((revision / "history/evidence" / advance.sha256(contents)).read_bytes(), contents)

    def test_publication_recovers_after_registry_write_and_allows_later_results(self):
        """Recover an interrupted transaction without reverting its history or later indexed work."""
        revision = advance.prepare(self.run, self.specification_path)
        original_replace = advance.replace_bytes
        def fail_sentinel(path, contents):
            """Simulate loss of the parent after the registry is safely written."""
            if Path(path).name == controller.SENTINEL_NAME:
                raise OSError("synthetic publication interruption")
            original_replace(path, contents)
        with mock.patch.object(advance, "replace_bytes", side_effect=fail_sentinel):
            with self.assertRaisesRegex(OSError, "publication interruption"):
                advance.activate(revision)
        self.assertEqual(controller.read_json(self.run / controller.SENTINEL_NAME), self.old_ready)
        self.assertEqual(controller.read_json(revision / "transaction.json")["state"], "publishing")
        self.assertFalse(advance.activate(revision)["already_active"])
        self.assertEqual(advance.prepare(self.run, self.specification_path), revision)
        live = controller.read_json(self.run / "study_registry.json")
        live["adaptive_batches"][0]["hypotheses"][0]["results"]["h100"].append({"new_result": True})
        self.write("study_registry.json", live)
        self.write("queue/pending/synthetic_new_case.json", {})
        self.assertTrue(advance.activate(revision)["already_active"])
        live["adaptive_batches"][0]["hypotheses"][0]["mechanism"] = "unauthorized changed mechanism"
        self.write("study_registry.json", live)
        with self.assertRaisesRegex(ValueError, "endpoints"):
            advance.activate(revision)

    def test_every_original_count_and_prior_batch_count_is_required(self):
        """A result at one copy count cannot satisfy a different saved cohort count."""
        registry = copy.deepcopy(self.registry)
        registry["initial_ideas"][0]["results"]["h100"].pop()
        with self.assertRaisesRegex(ValueError, "unfinished prior"):
            advance.completed_cohorts(registry, advance.validation.Evidence())
        registry = copy.deepcopy(self.registry)
        batch = registry["adaptive_batches"][0]
        batch.update(reference_commit=self.commit, matched_copy_counts=[1, 4])
        for index, slot in enumerate(batch["hypotheses"]):
            slot.update(self.hypothesis(index), state="specified", results={context: [{"copies": 1, "result_path": "/synthetic"}]
                for context in advance.validation.CONTEXTS})
        with self.assertRaisesRegex(ValueError, "unfinished prior"):
            advance.completed_cohorts(registry, advance.validation.Evidence())

    def test_duplicates_cover_initial_trials_prior_batches_and_mechanism_text(self):
        """Reject renamed settings and mechanisms from every earlier adaptive batch."""
        for duplicate in ("name", "mechanism", "initial_signature", "prior_signature"):
            registry, proposals = copy.deepcopy(self.registry), [self.hypothesis(index) for index in range(5)]
            if duplicate == "name":
                proposals[0]["name"] = ideas.INITIAL_IDEAS[0].name.upper().replace("_", "-")
            elif duplicate == "mechanism":
                proposals[0]["mechanism"] = "  " + ideas.INITIAL_IDEAS[0].mechanism.upper() + "  "
            elif duplicate == "initial_signature":
                proposals[0].update(ideas.trial_changes(ideas.INITIAL_IDEAS[2], 0))
            else:
                previous = copy.deepcopy(self.hypothesis(10))
                previous.update(position=31, results={})
                registry["adaptive_batches"][0]["hypotheses"][0] = previous
                proposals[0].update({scope: previous[scope] for scope in ("learner", "environment", "execution")})
            with self.subTest(duplicate=duplicate), self.assertRaisesRegex(ValueError, "repeat"):
                advance.register_five(registry, proposals, 2 if duplicate == "prior_signature" else 1,
                                      self.commit, [1], {}, self.root)

    def test_measurement_settings_are_applied_and_unsupported_criteria_reject(self):
        """Use the approved common minimum and preserve the fixed statistical decision rule."""
        for entry in self.specification["hypotheses"]:
            entry["measurement_settings"]["minimum_updates_per_round"] = 3
        self.save_specification()
        revision = advance.prepare(self.run, self.specification_path)
        ready = controller.read_json(revision / "staged/SOURCE_AND_GATES_READY.json")
        self.assertEqual((ready["rounds"], ready["iterations"]), (7, 3))
        self.assertTrue(all(controller.read_json(path)["iterations"] == 3 for path in (revision / "staged/cases").glob("*.json")))
        self.specification["hypotheses"][0]["acceptance_criteria"]["maximum_one_sided_sign_probability"] = .5
        self.save_specification()
        with self.assertRaisesRegex(ValueError, "acceptance criteria"):
            advance.prepare(self.run, self.specification_path)

    def test_empty_capacity_wrong_baseline_reused_process_and_empty_profile_reject(self):
        """Refuse missing counts, shared processes, changed full baselines and rc-zero empty traces."""
        context = "jaguar03_rtx_a4500"
        supplied = self.specification["contexts"][context]
        supplied["capacity_records"].pop()
        self.save_specification()
        with self.assertRaisesRegex(ValueError, "every requested count"):
            advance.prepare(self.run, self.specification_path)
        supplied["capacity_records"].append(str(self.measurement(context, "packing-point", 2)))
        profile = Path(supplied["profile_record"])
        summary = profile.parent / "trace/cuda_gpu_kern_sum.csv"
        summary.write_text("Name,Instances,Total Time (ns)\n")
        self.save_specification()
        with self.assertRaisesRegex(ValueError, "no data rows"):
            advance.prepare(self.run, self.specification_path)

    def test_different_gpu_baseline_and_reused_capacity_process_reject(self):
        """Require one shared baseline and a fresh process for each isolated capacity point."""
        context = "jaguar03_rtx_a4500"
        supplied = self.specification["contexts"][context]
        saved = {}
        for key in ("correctness_record", "confirmation_record"):
            path = Path(supplied[key])
            result, plan = controller.read_json(path), controller.read_json(path.parent / "measurement_plan.json")
            saved[key] = (copy.deepcopy(result), copy.deepcopy(plan))
            result["candidate"]["learner"]["update_unroll"] = 2
            plan["configuration"]["learner"]["update_unroll"] = 2
            self.finish(path.parent, plan, result)
        with self.assertRaisesRegex(ValueError, "same complete accepted baseline"):
            advance.prepare(self.run, self.specification_path)
        for key, (result, plan) in saved.items():
            self.finish(Path(supplied[key]).parent, plan, result)
        path = Path(supplied["capacity_records"][0])
        capacity = controller.read_json(path)
        original = controller.read_json(supplied["correctness_record"])["execution_identity"]
        capacity["execution_identity"] = original
        for key in ("hostname", "pid", "process_start_time"):
            capacity["hardware"][key] = original[key]
        self.write(path.parent / "hardware.json", capacity["hardware"])
        self.write(path, capacity)
        with self.assertRaisesRegex(ValueError, "reused another measurement process"):
            advance.prepare(self.run, self.specification_path)

    def test_evidence_hash_change_after_preparation_blocks_activation(self):
        """A previously validated record cannot be changed between prepare and activate."""
        revision = advance.prepare(self.run, self.specification_path)
        path = Path(self.specification["contexts"]["h100"]["confirmation_record"])
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "evidence changed before activation"):
            advance.activate(revision)

    def test_active_queue_changed_evidence_and_third_registry_state_reject(self):
        """Leave all live bytes intact when late publication preconditions change."""
        revision = advance.prepare(self.run, self.specification_path)
        self.write("queue/pending/synthetic_case.json", {})
        with self.assertRaisesRegex(ValueError, "pending, running or failed"):
            advance.activate(revision)
        (self.run / "queue/pending/synthetic_case.json").unlink()
        self.write("study_registry.json", {"unrelated": "controller change"})
        with self.assertRaisesRegex(ValueError, "endpoints"):
            advance.activate(revision)

    def test_staged_tampering_and_custom_or_mixed_settings_reject(self):
        """Reject a changed prepared case and settings that the existing benchmark does not consume."""
        revision = advance.prepare(self.run, self.specification_path)
        case = next((revision / "staged/cases").glob("*.json"))
        case.write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            advance.activate(revision)
        self.specification["hypotheses"][0]["measurement_settings"]["minimum_updates_per_round"] = 2
        self.save_specification()
        with self.assertRaisesRegex(ValueError, "common measurement"):
            advance.prepare(self.run, self.specification_path)

    def test_worker_receipts_cover_pending_launches_stale_records_and_children(self):
        """Require exact observations for a worker job that has no worker metadata yet."""
        row = "123\tqueue-worker\t" + self.commit + "\t/synthetic_worker.sh"
        self.write("slurm/submissions.tsv", row + "\n")
        receipt = controller.read_json(self.run / "worker_stop.json")
        with self.assertRaisesRegex(ValueError, "including unstarted"):
            advance.stopped_workers(self.run, self.run / "worker_stop.json", advance.validation.Evidence())
        receipt["jobs"] = [{"job_id": "123", "submission_row": row, "state": "COMPLETED",
                            "observed_at": "2026-10-31T12:00:00+00:00"}]
        worker = {"worker_id": "synthetic_worker", "hostname": "jaguar03", "worker_pid": 77,
            "worker_start_time": "100", "slurm_job_id": "123", "state": "running", "time": "2026-10-31T11:00:00+00:00"}
        self.write("queue/workers/synthetic.json", worker)
        self.write("slurm/submitted_jobids_uva.txt", "123\n")
        receipt["workers"] = [{**worker, "kind": "terminal_slurm_job", "state": "COMPLETED",
            "ledger_path": str(self.run / "slurm/submitted_jobids_uva.txt"), "observed_at": "2026-10-31T12:00:00+00:00"}]
        self.write("worker_stop.json", receipt)
        self.assertEqual(advance.stopped_workers(self.run, self.run / "worker_stop.json", advance.validation.Evidence()), [worker])
        receipt["workers"][0]["worker_start_time"] = "other original process"
        self.write("worker_stop.json", receipt)
        with self.assertRaisesRegex(ValueError, "original PID"):
            advance.stopped_workers(self.run, self.run / "worker_stop.json", advance.validation.Evidence())
        receipt["workers"][0]["worker_start_time"] = "100"
        lease = {"unit_id": "synthetic_case", "attempt": 1, "hostname": "jaguar03", "runner_pid": 78,
            "runner_start_time": "101", "slurm_job_id": "123", "state": "completed"}
        self.write("queue/leases/synthetic.json", lease)
        self.write("worker_stop.json", receipt)
        with self.assertRaisesRegex(ValueError, "original numerical child"):
            advance.stopped_workers(self.run, self.run / "worker_stop.json", advance.validation.Evidence())
        receipt["runners"] = [{**lease, "kind": "terminal_slurm_job", "state": "COMPLETED",
                               "observed_at": "2026-10-31T12:00:00+00:00"}]
        self.write("worker_stop.json", receipt)
        self.assertEqual(advance.stopped_workers(self.run, self.run / "worker_stop.json", advance.validation.Evidence()), [worker])

    def test_fixed_protocol_exact_types_and_unimplemented_fields_reject(self):
        """Preserve algorithm/game/precision while rejecting Boolean scan counts and unknown settings."""
        for scope, name, value in (("learner", "n_envs", 64), ("environment", "frame_skip", 8),
                                   ("learner", "rollout_unroll", True), ("environment", "num_envs", 128.0)):
            changed = copy.deepcopy(self.selected)
            changed[scope][name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                advance.checked_configuration(changed, self.root)
        self.specification["runtime_environment"]["JAX_DEFAULT_MATMUL_PRECISION"] = "tensorfloat32"
        self.save_specification()
        with self.assertRaisesRegex(ValueError, "highest float32"):
            advance.prepare(self.run, self.specification_path)

    def test_original_default_diagnostic_and_incomplete_profile_window_reject(self):
        """Require the selected-baseline profile schema and its actual complete work count."""
        path = Path(self.specification["contexts"]["h100"]["profile_record"])
        result, plan = controller.read_json(path), controller.read_json(path.parent / "measurement_plan.json")
        result["kind"] = "original_default_diagnostic"
        self.write(path, result)
        arguments = (path, self.run, self.root, self.commit, self.runtime, "h100",
                     advance.validation.configuration_without_copies(self.selected), self.core, advance.validation.Evidence())
        with self.assertRaisesRegex(ValueError, "selected-configuration schema"):
            advance.profile_record(*arguments)
        result["kind"] = plan["kind"]
        result["profile_end_iteration"] = result["profile_start_iteration"] + 2
        self.write(path, result)
        with self.assertRaisesRegex(ValueError, "complete execution groups"):
            advance.profile_record(*arguments[:-1], advance.validation.Evidence())

    def test_profile_count_seeds_and_compilation_stability_match_the_prepared_plan(self):
        """A different policy count, random identity or compile state cannot reuse a valid profile plan."""
        path = Path(self.specification["contexts"]["h100"]["profile_record"])
        original = controller.read_json(path)
        for mutation in ("copies", "seeds", "updates", "compilation", "warmup"):
            result = copy.deepcopy(original)
            if mutation == "copies":
                result["configuration"]["learner"]["n_copies"] = 2
            elif mutation == "seeds":
                result["copy_seeds"] = [55]
            elif mutation == "updates":
                result["profile_updates"] = 2
            elif mutation == "compilation":
                result["compiled_signatures_after"] = [2]
            else:
                result["normalization_warmup_iterations"] = 49
            self.write(path, result)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                advance.profile_record(path, self.run, self.root, self.commit, self.runtime, "h100",
                    advance.validation.configuration_without_copies(self.selected), self.core, advance.validation.Evidence())

    def test_imports_remain_standard_library_only(self):
        """Preparation and tests must not create a numerical context on the login host."""
        self.assertFalse(any(name in sys.modules for name in ("jax", "numpy", "torch")))


class MetadataBoundaryTests(unittest.TestCase):
    """Verify actual filesystem/Git and compute-allocation boundaries without mocks of those checks."""

    def setUp(self):
        """Keep every test repository and ledger isolated from the live study."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_write_once_retains_identical_bytes_and_rejects_changed_input(self):
        """Do not overwrite an existing revision artifact, even if JSON could be reformatted."""
        path = self.root / "revision/original.json"
        advance.write_once(path, b'{"original": true}\n\n')
        advance.write_once(path, b'{"original": true}\n\n')
        with self.assertRaisesRegex(ValueError, "immutable revision artifact"):
            advance.write_once(path, b'{"original":true}\n')
        self.assertEqual(path.read_bytes(), b'{"original": true}\n\n')

    def test_activation_requires_reviewed_committed_launcher_serialization(self):
        """An uncommitted marker or source check outside the shared lock cannot authorize a revision."""
        contents = (SCRIPTS / "launch_queue.py").read_bytes()
        with mock.patch.object(advance.subprocess, "check_output", return_value=contents):
            advance.launcher_capability(advance.validation.Evidence())
        with mock.patch.object(advance.subprocess, "check_output", return_value=b"old committed launcher"):
            with self.assertRaisesRegex(ValueError, "must be committed"):
                advance.launcher_capability(advance.validation.Evidence())
        invalid = contents.replace(b"with source_admission_lock(args.run_dir):", b"if True:")
        with mock.patch.object(advance.validation.Evidence, "file", return_value=invalid):
            with self.assertRaisesRegex(ValueError, "must share the admission lock"):
                advance.launcher_capability(advance.validation.Evidence())

    def test_actual_process_precision_and_cache_evidence_is_required(self):
        """An immutable plan alone cannot substitute for the numerical child's observed runtime."""
        expected = {"JAX_ENABLE_X64": "true", "JAX_DEFAULT_MATMUL_PRECISION": "highest",
                    "XLA_FLAGS": "--xla_gpu_exclude_nondeterministic_ops=true"}
        actual = {**expected, "JAX_ENABLE_COMPILATION_CACHE": "false",
                  "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": ""}
        advance.actual_runtime(actual, expected)
        for name, value in (("JAX_ENABLE_COMPILATION_CACHE", "true"), ("JAX_COMPILATION_CACHE_DIR", "/old/cache"),
                            ("JAX_DEFAULT_MATMUL_PRECISION", "tensorfloat32")):
            with self.subTest(name=name), self.assertRaises(ValueError):
                advance.actual_runtime({**actual, name: value}, expected)
        with self.assertRaises(ValueError):
            advance.actual_runtime(expected, expected)

    def test_matched_copy_counts_reject_boolean_duplicates_or_missing_one(self):
        """Retain exact positive integer copy identities rather than Python's Boolean/integer equality."""
        self.assertEqual(advance.copy_counts([4, 1, 2]), [1, 2, 4])
        for counts in ([True, 2], [1, 1], [2], [1, 0], []):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                advance.copy_counts(counts)

    def test_correctness_window_rejects_incompatible_group_pairs(self):
        """A valid timing least common multiple cannot repair the pinned correctness function's shorter window."""
        reference = {"iterations_per_call": 5, "report_readout": "none", "report_every_iterations": 10}
        advance.correctness_window_supported(reference, {**reference, "iterations_per_call": 10})
        with self.assertRaisesRegex(ValueError, "cannot compare"):
            advance.correctness_window_supported(reference, {**reference, "iterations_per_call": 2})
        advance.correctness_window_supported(reference, {**reference, "iterations_per_call": 1, "report_readout": "scheduled"})

    def test_source_requires_detached_clean_commit_and_helper_receipt_uses_git(self):
        """Reject a branch/uncommitted source and bind historical helpers to their committed bytes."""
        relative = advance.SCRIPT_PREFIX + "synthetic_helper.py"
        helper = self.root / relative
        helper.parent.mkdir(parents=True)
        helper.write_text("# synthetic source\n")
        core = self.root / "10_jax_exploration_platform/src/exploration_platform/atari_pixels/config.py"
        core.parent.mkdir(parents=True)
        core.write_text("# synthetic numerical source, never imported\n")
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        subprocess.run(["git", "-C", str(self.root), "add", "."], check=True)
        subprocess.run(["git", "-C", str(self.root), "-c", "user.name=Synthetic metadata test",
            "-c", "user.email=synthetic@example.invalid", "-c", "core.hooksPath=/dev/null",
            "commit", "-qm", "Create synthetic metadata source"], check=True)
        commit = subprocess.check_output(["git", "-C", str(self.root), "rev-parse", "HEAD"], text=True).strip()
        with self.assertRaisesRegex(ValueError, "detached"):
            advance.source_identity(self.root, commit)
        subprocess.run(["git", "-C", str(self.root), "checkout", "-q", "--detach"], check=True)
        self.assertEqual(advance.source_identity(self.root, commit)["source_commit"], commit)
        receipt = {"source_root": str(self.root), "source_commit": commit,
                   "files": {relative: advance.sha256(helper.read_bytes())}}
        helper.write_text("# later helper edits do not change the original Git blob\n")
        advance.committed_files(receipt, advance.validation.Evidence(), [relative])
        with self.assertRaisesRegex(ValueError, "differs"):
            advance.committed_files(receipt, advance.validation.Evidence(), [relative], require_worktree=True)
        with self.assertRaisesRegex(ValueError, "uncommitted"):
            advance.source_identity(self.root, commit)

    def test_prepare_requires_owned_compute_cgroup_and_rejects_login_hosts(self):
        """A forged or inherited job variable alone cannot authorize real-data statistics on login."""
        ledger = self.root / "slurm/submitted_jobids_uva.txt"
        ledger.parent.mkdir()
        ledger.write_text("333\n")
        original_read = Path.read_text
        def read(path, *args, **kwargs):
            """Provide a synthetic owned compute cgroup while keeping real fixture ledger reads."""
            if path == Path("/proc/self/cgroup"):
                return "0::/system.slice/slurmstepd.scope/job_333/step_batch/task_0\n"
            return original_read(path, *args, **kwargs)
        with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "333"}), mock.patch.object(Path, "read_text", read):
            with mock.patch.object(advance.socket, "gethostname", return_value="heartpiece"):
                advance.require_validation_allocation(self.root)
            with mock.patch.object(advance.socket, "gethostname", return_value="portal05"):
                with self.assertRaisesRegex(ValueError, "owned Slurm compute"):
                    advance.require_validation_allocation(self.root)
        with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "999"}):
            with self.assertRaisesRegex(ValueError, "owned Slurm compute"):
                advance.require_validation_allocation(self.root)


if __name__ == "__main__":
    unittest.main()
