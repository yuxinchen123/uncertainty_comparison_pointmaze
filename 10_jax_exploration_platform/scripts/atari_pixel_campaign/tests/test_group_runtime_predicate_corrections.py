"""Check the four refused metadata cases without scientific or native operations."""
import copy
from contextlib import ExitStack
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import group_runtime as runtime
import group_runtime_evidence as evidence
import test_group_runtime as fixtures


def configurations():
    """Use complete original source-derived options with one distinct rendering implementation."""
    base = runtime.producer.selected.diagnostic.default_configuration(fixtures.SCRIPTS.parents[2])
    reference = json.loads(json.dumps({key: base[key] for key in ("learner", "environment")}))
    reference["learner"]["n_copies"] = 1
    reference["execution"] = {"compilation_mode": "whole_iteration", "iterations_per_call": 1,
        "report_readout": "none", "report_every_iterations": 10}
    reference["environment"]["cache_room_backgrounds"] = False
    candidate = copy.deepcopy(reference)
    candidate["environment"]["cache_room_backgrounds"] = True
    return candidate, reference


def agreement():
    """Represent a successful unchanged comparator report, without executing comparisons."""
    return {"leaves": 1, "largest_float32_absolute_error": 0.0,
        "largest_float64_absolute_error": 0.0, "float32_derived_reward_moment_fields": []}


def correctness_result():
    """Retain initialization, all supplied-input reports and one complete original update."""
    return {"correctness_attempted": True, "correctness_passed": True, "implementation_distinct": True,
        "measurement_valid": False, "correctness": {"initialization": agreement(), "passed": True,
            "supplied_inputs": {name: agreement() for name in ("preparation", "fixed_batch", "fixed_batch_update")},
            "complete_iterations": 1, "complete_iteration": agreement()}}


class CorrectnessReportTests(unittest.TestCase):
    """Use the actual unchanged standard correctness predicate inside the saved-member reader."""
    def setUp(self):
        """Create full original configurations and positive nonempty metadata reports."""
        candidate, reference = configurations()
        self.plan = {"configuration": candidate, "reference_configuration": reference}
        self.result = correctness_result()

    def test_positive_original_report_and_complete_length(self):
        """A full report passes both standard and group predicates without relaxed tolerances."""
        evidence.selected.validation.correctness(self.result)
        self.assertEqual(evidence.validate_member_correctness(self.plan, self.result), self.result["correctness"])

    def test_empty_reports_and_invalid_comparison_fields_refuse(self):
        """Empty reports, changed update counts and wider field selection cannot pass."""
        changes = [lambda x: x["correctness"].update(supplied_inputs={}),
            lambda x: x["correctness"].update(complete_iteration={}),
            lambda x: x["correctness"]["supplied_inputs"]["fixed_batch"].update(leaves=True),
            lambda x: x["correctness"].update(complete_iterations=10),
            lambda x: x["correctness"]["initialization"].update(largest_float64_absolute_error=float("nan")),
            lambda x: x["correctness"]["supplied_inputs"]["preparation"].update(
                float32_derived_reward_moment_fields=["copy_1.reward_statistics.mean"]),
            lambda x: x["correctness"]["complete_iteration"].update(
                float32_derived_reward_moment_fields=["copy_1.policy_parameters"])]
        for mutate in changes:
            result = copy.deepcopy(self.result); mutate(result)
            with self.subTest(change=changes.index(mutate)), self.assertRaises((ValueError, KeyError)):
                evidence.validate_member_correctness(self.plan, result)

    def test_actual_saved_member_calls_standard_report_gate(self):
        """Successful unrelated source/device placeholders cannot bypass the actual report predicate."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); unit = root / "data/group"; worker = unit / "workers/worker_1"
            worker.mkdir(parents=True)
            plan = {**self.plan, "campaign_run_dir": str(root), "unit_id": "group", "mode": "correctness",
                "gpu_count": 1, "copy_seeds": [[1]], "source_commit": "0" * 40, "helpers": {},
                "runtime_environment": {}, "asset_record": {}}
            metadata = {"process": {"pid": 101}}
            binding = {"gpu_uuid": "GPU-synthetic", "process": metadata["process"]}
            for path, value in ((unit / "scaling_plan.json", plan), (worker / "group_step_metadata.json", metadata),
                    (worker / "resolved_device_binding.json", binding), (worker / "group_process_identity.json", {})):
                path.write_text(json.dumps(value))
            result = {**copy.deepcopy(self.result), "schema_version": 2, "kind": evidence.package.MODES["correctness"],
                "completed": True, **{key: plan[key] for key in ("configuration", "reference_configuration",
                    "source_commit", "helpers", "runtime_environment", "asset_record")}, "copy_seeds": [1],
                "worker_index": 0, "group_metadata": evidence.group.receipt(worker / "group_step_metadata.json"),
                "resolved_device_binding": binding, "scientific_credit": 0, "admission_granted": False}
            result_path = worker / "result.json"
            def adjacent(name):
                """Replace only declared unrelated successful source checks in this isolated predicate test."""
                return {"group_runtime": runtime, "configured_scaling_evidence": SimpleNamespace(verify_sources=lambda *args: None)}[name]
            with ExitStack() as stack:
                for target, name, value in ((evidence, "require_owned_compute", None), (runtime, "validate_plan", None),
                        (evidence.configured, "verify_qualification_source_receipts", None),
                        (evidence.package, "worker_index", 0), (evidence.group, "source_identity", None),
                        (evidence.group, "validate_binding", None), (runtime, "numeric_command", []),
                        (runtime, "validate_process", None), (evidence.package, "completed_work", None),
                        (evidence, "verify_library_evidence", None), (evidence, "validate_controlled_window", None)):
                    stack.enter_context(patch.object(target, name, return_value=value))
                stack.enter_context(patch.object(evidence.group, "adjacent", side_effect=adjacent))
                result_path.write_text(json.dumps(result))
                self.assertEqual(evidence.verify_saved_member(worker, terminal=False)[0], result)
                result["correctness"].update(supplied_inputs={}, complete_iteration={})
                result_path.write_text(json.dumps(result))
                with self.assertRaises(ValueError):
                    evidence.verify_saved_member(worker, terminal=False)


class PrerequisiteSeedTests(unittest.TestCase):
    """Join exact global seeds to the physical member for both prerequisite kinds."""
    def test_same_physical_member_requires_exact_seed_types_and_values(self):
        """Previously verified seed99, float1 and Boolean1 cannot qualify current seed1."""
        configuration, reference = configurations()
        for mode in ("qualification", "correctness"):
            for seed in (1, 99, 1.0, True):
                with self.subTest(mode=mode, seed=repr(seed)), tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    plan = {"gpu_count": 1, "configuration": configuration, "reference_configuration": reference,
                        "copy_seeds": [[1]], "source_commit": "0" * 40, "helpers": {}, "runtime_environment": {},
                        "asset_record": {}, "qualification_software": {}}
                    previous = {**copy.deepcopy(plan), "mode": mode, "copy_seeds": [[seed]]}
                    result = {"copy_seeds": [seed]}; binding = {"gpu_uuid": "GPU-synthetic"}
                    (root / "result.json").write_text(json.dumps(result))
                    item = {"result": evidence.group.receipt(root / "result.json")}
                    review = {"schema_version": 2, "kind": evidence.REVIEW_KIND, "approved": True,
                        "result": item["result"], "mode": mode, "job_id": "123456", "actual_terminal_state": "COMPLETED",
                        "actual_exit_code": "0:0", "admission_granted": False, "scientific_credit": 0}
                    (root / "review.json").write_text(json.dumps(review)); item["review"] = evidence.group.receipt(root / "review.json")
                    expected = {name: {} for name in ("selected_python", "site_packages", "image", "distribution")}
                    with patch.object(evidence, "verify_saved_member", return_value=(result, previous, binding)), \
                            patch.object(evidence.package, "expected_for_plan", return_value=expected), \
                            patch.object(evidence.group, "owned_inputs", return_value={"job_id": "123456"}):
                        if type(seed) is int and seed == 1:
                            self.assertEqual(evidence.reviewed_member(item, mode, plan, {"index": 0, **binding}), result)
                        else:
                            with self.assertRaises(ValueError):
                                evidence.reviewed_member(item, mode, plan, {"index": 0, **binding})


class CurrentKernelOwnershipTests(unittest.TestCase):
    """Read only actual own-process metadata in the admitted CPU allocation."""
    def test_real_reader_allocation_can_differ_from_historical_gpu_job(self):
        """The actual current numeric cgroup, not the historical plan job, determines reader ownership."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root / "slurm").mkdir()
            (root / "slurm/submitted_jobids_uva.txt").write_text(os.environ["SLURM_JOB_ID"] + "\n")
            current = evidence.require_owned_compute({"campaign_run_dir": str(root), "job_id": "123456"})
            self.assertEqual(current["job_id"], "6783003")
            self.assertEqual(current["step_id"], os.environ["SLURM_STEP_ID"])

    def test_changed_claim_refuses_before_any_native_query(self):
        """A private ledger naming987654 cannot substitute the real kernel cgroup of6783003."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root / "slurm").mkdir()
            (root / "slurm/submitted_jobids_uva.txt").write_text("987654\n")
            with patch.dict(os.environ, {"SLURM_JOB_ID": "987654", "SLURM_STEP_ID": "123"}), \
                    patch.object(runtime.group.single, "bounded_query") as query:
                with self.assertRaises(ValueError):
                    evidence.require_owned_compute({"campaign_run_dir": str(root)})
            query.assert_not_called()

    def test_gpu_producer_checks_actual_kernel_before_query(self):
        """Successful source placeholders do not let a producer query a forged environment job."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            real_adjacent = runtime.group.adjacent
            helper = real_adjacent("device_memory_limit")
            observed = helper.process_identity()
            observed["hostname"] = "jaguar03"
            def identity():
                """Keep actual cgroup and all process identities while substituting only this host fixture."""
                return copy.deepcopy(observed)
            def adjacent(name):
                """Supply own-process metadata without reading any GPU or native library."""
                return SimpleNamespace(process_identity=identity) if name == "device_memory_limit" else real_adjacent(name)
            plan = {"configured_group": {"owned_inputs_path": str(directory / "owned.json")}}
            (directory / "owned.json").write_text("{}")
            with patch.dict(os.environ, {"SLURM_JOB_ID": "987654", "SLURM_STEP_ID": "123"}), \
                    patch.object(runtime.group, "validate_plan"), patch.object(runtime.group, "source_identity", return_value={}), \
                    patch.object(runtime.group, "owned_inputs", return_value={"job_id": "987654"}), \
                    patch.object(runtime.group, "adjacent", side_effect=adjacent), \
                    patch.object(runtime.group.single, "bounded_query") as query:
                with self.assertRaisesRegex(ValueError, "kernel cgroup"):
                    runtime.group.collect_metadata(plan, directory, "worker")
            query.assert_not_called()
            partial = json.loads((directory / "group_step_metadata.json").read_text())
            self.assertIn("kernel cgroup", partial["capture_error"]["error"])


class SavedProcessCustodyTests(unittest.TestCase):
    """Check complete batch-controller, srun-wrapper and numerical-step records."""
    @classmethod
    def setUpClass(cls):
        """Use the original full source fixture without inheriting unrelated tests."""
        fixtures.RuntimePlanTests.setUpClass()

    @classmethod
    def tearDownClass(cls):
        """Remove only this private synthetic source fixture."""
        fixtures.RuntimePlanTests.tearDownClass()

    def setUp(self):
        """Write distinct complete controller/wrapper/numerical records for an eight-GPU metadata fixture."""
        fixture = fixtures.RuntimePlanTests("runTest"); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        self.plan, self.unit, self.write = fixture.plan, fixture.unit, fixture.write
        self.job = "123456"
        (self.unit / "wrapper_records").mkdir()
        self.controller = {"schema_version": 2, "kind": "owned_configured_group_batch_controller", "job_id": self.job,
            "plan": runtime.group.receipt(self.unit / "scaling_plan.json"), "helpers": self.plan["helpers"],
            "source_commit": self.plan["source_commit"], "process": self.process(5000, 4999, "batch", runtime.controller_command(self.plan)),
            "started_at_seconds": 1.0, "gpu_operations": 0}
        self.write(self.unit / "group_job_controller.json", self.controller)
        self.rows = []
        with patch.object(runtime.group, "owned_inputs", return_value={"job_id": self.job}):
            for number, (role, index) in enumerate([("coordinator", None), *[("worker", x) for x in range(8)]]):
                name = "coordinator" if role == "coordinator" else "worker_" + str(index + 1)
                directory = self.unit if role == "coordinator" else runtime.group.worker_directory(self.plan, index)
                directory.mkdir(parents=True, exist_ok=True)
                command = runtime.producer.step_command(self.plan, self.job, role, runtime.numeric_command(self.plan, role, index))
                common = {"schema_version": 2, "role": role, "index": index, "job_id": self.job,
                    "plan": runtime.group.receipt(self.unit / "scaling_plan.json"), "helpers": self.plan["helpers"],
                    "controller_record": runtime.group.receipt(self.unit / "group_job_controller.json"), "command": command}
                process = self.process(6000 + number, 5000, "batch", command)
                intent = {**common, "controller": self.controller["process"], "at_seconds": 2.0, "automatic_retry_permitted": False}
                start = {**common, "process": process, "started_at_seconds": 3.0, "is_numerical_process": False}
                self.write(self.unit / "wrapper_records" / (name + "_intent.json"), intent)
                self.write(self.unit / "wrapper_records" / (name + "_started.json"), start)
                wait = {"schema_version": 2, "job_id": self.job, "plan": common["plan"], "helpers": common["helpers"],
                    "controller_record": common["controller_record"], "started": runtime.group.receipt(self.unit / "wrapper_records" / (name + "_started.json")),
                    "startup_registration_complete": True, "process": process, "returncode": 0,
                    "waited_at_seconds": 4.0, "terminated_for_failure": False, "is_numerical_wait": False}
                self.write(self.unit / "wrapper_records" / (name + "_wait.json"), wait)
                numeric = self.process(7000 + number, 6900 + number, str(number), runtime.numeric_command(self.plan, role, index))
                metadata = {"role": role, "process": {key: numeric[key] for key in ("hostname", "pid", "parent_pid", "process_start_time", "cgroup")},
                    "environment": {"SLURM_JOB_ID": self.job, "SLURM_STEP_ID": str(number)}, "inputs": {"job_id": self.job}}
                self.write(directory / "group_step_metadata.json", metadata)
                self.write(directory / "group_process_identity.json", numeric)
                self.rows.append({"name": name, "directory": directory, "start": start, "wait": wait, "numeric": numeric, "metadata": metadata})

    def process(self, pid, parent, step, command):
        """Keep scheduler numeric steps separate from the batch wrapper's parent relation."""
        return {"hostname": "jaguar03", "pid": pid, "parent_pid": parent, "process_start_time": str(pid + 100),
            "uids": [2618919] * 4, "argv": command,
            "cgroup": "0::/system.slice/slurmstepd.scope/job_" + self.job + "/step_" + step + "/user/task_0\n"}

    def verify(self):
        """Mock unrelated saved allocation validation, retaining the actual complete process predicates."""
        with patch.object(runtime.group, "owned_inputs", return_value={"job_id": self.job}), \
                patch.object(runtime.group, "validate_metadata"):
            return runtime.verify_wrapper_waits(self.plan)

    def test_complete_distinct_process_records_pass(self):
        """Full normal waits join actual role-specific commands and source-bound records."""
        self.assertEqual(self.verify(), self.controller)

    def test_controller_ownership_source_plan_birth_and_cgroup_refuse(self):
        """The previously accepted unrelated controller fails each actual ownership/source gate."""
        changes = [("uids", [0] * 4), ("process_start_time", "not-a-birth"), ("parent_pid", 5000),
            ("argv", ["/unrelated/controller"]), ("cgroup", "unrelated"), ("hostname", "other")]
        for key, value in changes:
            altered = copy.deepcopy(self.controller); altered["process"][key] = value
            with patch.object(runtime.group, "owned_inputs", return_value={"job_id": self.job}), self.subTest(key=key), self.assertRaises(ValueError):
                runtime.validate_saved_controller(self.plan, altered)
        for key, value in (("helpers", {}), ("source_commit", "f" * 40), ("plan", {"path": "/unrelated"}), ("job_id", "999")):
            altered = copy.deepcopy(self.controller); altered[key] = value
            with patch.object(runtime.group, "owned_inputs", return_value={"job_id": self.job}), self.subTest(key=key), self.assertRaises(ValueError):
                runtime.validate_saved_controller(self.plan, altered)

    def test_coherent_wrong_wrapper_parent_cgroup_uid_command_and_birth_refuse(self):
        """Updating both start and wait does not conceal a different wrapper identity."""
        row = self.rows[1]
        for key, value in (("parent_pid", 123), ("cgroup", "unrelated"), ("uids", [2618919] * 3 + [0]),
                ("argv", ["/unrelated/wrapper"]), ("process_start_time", "not-a-birth")):
            start, wait = copy.deepcopy(row["start"]), copy.deepcopy(row["wait"])
            start["process"][key] = value; wait["process"] = start["process"]
            self.write(self.unit / "wrapper_records" / (row["name"] + "_started.json"), start)
            wait["started"] = runtime.group.receipt(self.unit / "wrapper_records" / (row["name"] + "_started.json"))
            self.write(self.unit / "wrapper_records" / (row["name"] + "_wait.json"), wait)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.verify()
        self.write(self.unit / "wrapper_records" / (row["name"] + "_started.json"), row["start"])
        self.write(self.unit / "wrapper_records" / (row["name"] + "_wait.json"), row["wait"])

    def test_numeric_identity_and_exact_wait_join_refuse(self):
        """Numeric commands/UIDs and typed normal waits remain separate from wrapper success."""
        row = self.rows[-1]
        for key, value in (("uids", [0] * 4), ("process_start_time", "0"), ("cgroup", "unrelated"), ("argv", ["wrong"]), ("pid", 6000)):
            value_record = copy.deepcopy(row["numeric"]); value_record[key] = value
            self.write(row["directory"] / "group_process_identity.json", value_record)
            with self.subTest(numeric=key), self.assertRaises(ValueError):
                self.verify()
        self.write(row["directory"] / "group_process_identity.json", row["numeric"])
        for key, value in (("returncode", True), ("returncode", 1), ("startup_registration_complete", False),
                ("started", {"path": "/unrelated"}), ("waited_at_seconds", 2.5), ("helpers", {})):
            wait = copy.deepcopy(row["wait"]); wait[key] = value
            self.write(self.unit / "wrapper_records" / (row["name"] + "_wait.json"), wait)
            with self.subTest(wait=key), self.assertRaises(ValueError):
                self.verify()
