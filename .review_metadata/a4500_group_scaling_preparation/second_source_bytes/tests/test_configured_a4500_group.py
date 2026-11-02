"""Exercise only synthetic group ownership, control and common-wall evidence."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "tests"))
import uva_group_binding as group
import configured_gpu_scaling as producer
import configured_scaling_evidence as evidence
import test_uva_slurm_configured_isolation as single_fixture


class GroupMetadataTests(unittest.TestCase):
    """Reuse saved CPU structure only; all added group/device data are synthetic."""
    @classmethod
    def setUpClass(cls):
        """Create only private temporary Git metadata used by the existing fixture."""
        single_fixture.ConfiguredIsolationTests.setUpClass()

    @classmethod
    def tearDownClass(cls):
        """Remove only the test's private temporary Git repository."""
        single_fixture.ConfiguredIsolationTests.tearDownClass()

    def setUp(self):
        """Construct a separate whole-job eight/step-one metadata fixture."""
        self.old = single_fixture.ConfiguredIsolationTests("runTest")
        self.old.setUp()
        self.addCleanup(self.old.doCleanups)
        self.unit = self.old.unit
        self.worker = self.unit / "workers/worker_1"
        self.worker.mkdir(parents=True)
        base = producer.selected.diagnostic.default_configuration(SCRIPTS.parents[2])
        configuration = {name: base[name] for name in ("learner", "environment")}
        configuration["learner"]["n_copies"] = 1
        configuration["execution"] = {"compilation_mode": "whole_iteration", "iterations_per_call": 1,
            "report_readout": "none", "report_every_iterations": 10}
        self.plan = {"schema_version": 1, "kind": "configured_a4500_scaling_group",
            "hardware_class": "jaguar03_rtx_a4500", "campaign_run_dir": str(self.old.run),
            "unit_id": self.unit.name, "gpu_count": 8, "gpu_uuids": [None] * 8,
            "configuration": configuration, "runtime_environment": {"JAX_ENABLE_X64": "true",
                "JAX_DEFAULT_MATMUL_PRECISION": "highest", "XLA_FLAGS": "--xla_gpu_exclude_nondeterministic_ops=true"},
            "helpers": copy.deepcopy(self.old.plan["helpers"]), "rounds": 7, "iterations": 10,
            "minimum_updates_per_round": 10, "copy_seeds": [[index + 1] for index in range(8)],
            "scaling_protocol": {"kind": "same_copies_per_card", "global_copy_seeds": list(range(1, 9)),
                "gradient_averaging": False, "measurement_scope": evidence.SCOPE},
            "compilation_cache_policy": "disabled",
            "configured_group": {"schema_version": 1, "profile": copy.deepcopy(group.GROUP_PROFILE),
                "capability": group.CAPABILITY, "physical_identities": [group.single.unresolved_identity() for _ in range(8)],
                "owned_inputs_path": str(self.unit / "owned_group_inputs.json"),
                "resources": {"job": {"requested_cpus": 65, "allocated_cpus": 66, "host_memory_mib": 526336, "gpus": 8},
                    "coordinator": {"cpus": 1, "host_memory_mib": 2048, "gpus": 0},
                    "worker": {"cpus": 8, "host_memory_mib": 65536, "gpus": 1}}}}
        # Before: source-default tuples; after: the same JSON lists used by a real frozen plan.
        self.plan = json.loads(json.dumps(self.plan))
        self.write(self.unit / "scaling_plan.json", self.plan)
        row = [self.old.job, self.plan["kind"], self.plan["helpers"]["source_commit"],
            str(self.old.run / "remote_setup" / self.unit.name / "job.sbatch")]
        common = {"schema_version": 1, "job_id": self.old.job, "cluster": "uva",
            "plan_sha256": group.receipt(self.unit / "scaling_plan.json")["sha256"],
            "data_unit": str(self.unit), "returncode": 0}
        submission = {**common, "kind": "owned_uva_configured_group_submission",
            "held_command": ["sbatch", "--hold", "--no-requeue", row[-1]], "submission_row": row}
        self.write(self.unit / "group_submission.json", submission)
        release = {**common, "kind": "owned_uva_configured_group_release",
            "command": ["scontrol", "release", self.old.job],
            "submission_receipt_sha256": group.receipt(self.unit / "group_submission.json")["sha256"]}
        self.write(self.unit / "group_release.json", release)
        owned = {**{key: common[key] for key in ("schema_version", "job_id")}, "kind": "uva_configured_group_owned_inputs",
            "plan": group.receipt(self.unit / "scaling_plan.json"), "submission_row": row,
            "submission": group.receipt(self.unit / "group_submission.json"),
            "release": group.receipt(self.unit / "group_release.json")}
        for key, filename, text, snapshot in (("ledger", "submitted_jobids_uva.txt", self.old.job + "\n", "group_owned_job_ids.txt"),
                ("submissions", "submissions.tsv", "\t".join(row) + "\n", "group_owned_submissions.tsv")):
            source = self.old.run / "slurm" / filename
            source.write_text(text)
            (self.unit / snapshot).write_text(text)
            owned[key] = {"source_path": str(source), "snapshot": group.receipt(self.unit / snapshot)}
        self.write(self.unit / "owned_group_inputs.json", owned)
        self.metadata = copy.deepcopy(self.old.metadata)
        self.metadata.update(kind="uva_owned_group_step_metadata", role="worker", directory=str(self.worker),
            profile=copy.deepcopy(group.GROUP_PROFILE), capability=group.CAPABILITY, source=self.plan["helpers"],
            inputs=owned, inputs_receipt=group.receipt(self.unit / "owned_group_inputs.json"), **group.BOUNDARY)
        self.metadata["environment"].update(SLURM_JOB_GPUS="0,1,2,3,4,5,6,7")
        job = self.old.job
        request = "cpu=65,mem=514G,node=1,gres/gpu=8"
        actual = "cpu=66,mem=514G,node=1,gres/gpu=8"
        self.metadata["job"]["stdout"] = (f"JobId={job} UserId=sl5nw(2618919) JobState=RUNNING "
            "Account=sl5nw_base Partition=gpu QOS=csresnolim NodeList=jaguar03 NumNodes=1 NumCPUs=66 "
            "TresPerNode=gres/gpu:nvidia_rtx_a4500:8 Requeue=0 Restarts=0 "
            f"AllocTRES={actual} ReqTRES={request}")
        self.old.rehash_query(self.metadata["job"])
        self.source_patch = patch.object(group, "source_identity", return_value=self.plan["helpers"])
        self.source_patch.start()
        self.addCleanup(self.source_patch.stop)
        for name, value in (("CONFIG_FILES", self.old.synthetic_hashes),
                ("CONFIG_EXCERPT_HASHES", {path: group.single.digest(group.single.encoded(row["settings"]))
                    for path, row in self.old.configuration["files"].items()})):
            current = patch.object(group.single, name, value)
            current.start()
            self.addCleanup(current.stop)
        self.binding = copy.deepcopy(self.old.binding)
        self.binding.update(kind="uva_group_current_process_resolved_device_binding", capability=group.CAPABILITY,
            plan_sha256=group.receipt(self.unit / "scaling_plan.json")["sha256"],
            metadata_sha256=group.digest(group.encoded(self.metadata)), **group.BOUNDARY)
        target = self.worker / "captured_driver" / Path(self.binding["native"]["driver_library"]).name
        target.parent.mkdir()
        target.write_bytes(b"synthetic driver")
        self.binding["driver_capture"]["captured"] = group.receipt(target)
        self.binding["original_environment"] = copy.deepcopy(self.metadata["environment"])
        self.binding["environment_after"] = copy.deepcopy(self.metadata["environment"])

    def write(self, path, value):
        """Write only private fixture data with the production stable encoding."""
        path.write_bytes(group.encoded(value))

    def test_eight_job_one_worker_and_current_context(self):
        """Eight logical job devices and one step selector bind physical minor three."""
        self.assertEqual(group.validate_metadata(self.plan, self.metadata), self.metadata)
        self.assertEqual(group.validate_binding(self.plan, self.metadata, self.binding, self.worker), self.binding)
        self.assertEqual(self.metadata["environment"]["SLURM_STEP_GPUS"], "6")
        self.assertEqual(self.binding["physical_minor"], 3)
        self.assertEqual(self.binding["native"]["jax_local_hardware_id"], 0)

    def test_zero_gpu_numeric_coordinator(self):
        """Actual coordinator step TRES proves zero GPUs independently of the eight-GPU job."""
        value = copy.deepcopy(self.metadata)
        value.update(role="coordinator", directory=str(self.unit))
        value["process"]["cpu_affinity"] = [0]
        value["environment"].update(SLURM_CPUS_PER_TASK="1", SLURM_STEP_GPUS="", CUDA_VISIBLE_DEVICES="")
        value["step"]["stdout"] = (f"StepId={self.old.job}.{self.old.step} State=RUNNING "
            "UserId=2618919 NodeList=jaguar03 Tasks=1 TRES=cpu=1,mem=2G,node=1")
        self.old.rehash_query(value["step"])
        self.assertEqual(group.validate_metadata(self.plan, value), value)
        value["step"]["stdout"] += ",gres/gpu=1"
        self.old.rehash_query(value["step"])
        with self.assertRaises(ValueError):
            group.validate_metadata(self.plan, value)

    def test_ownership_resources_and_selector_negatives(self):
        """Foreign ownership, changed counts and unproved selectors are refused."""
        mutations = [
            lambda v: v["environment"].update(SLURM_JOB_GPUS="0"),
            lambda v: v["environment"].update(SLURM_STEP_GPUS="8"),
            lambda v: v["environment"].update(CUDA_VISIBLE_DEVICES="0,1"),
            lambda v: v["process"].update(uid=0),
            lambda v: v["process"].update(cpu_affinity=[0]),
            lambda v: v.update(directory="/another/unit"),
            lambda v: v["job"].update(stdout=v["job"]["stdout"].replace("NumCPUs=66", "NumCPUs=65")),
            lambda v: v["step"].update(stdout=v["step"]["stdout"].replace("mem=64G", "mem=65G")),
            lambda v: v["job"].update(stdout=v["job"]["stdout"].replace("sl5nw(2618919)", "foreign(1)")),
        ]
        for mutate in mutations:
            value = copy.deepcopy(self.metadata)
            mutate(value)
            for name in ("job", "step"):
                self.old.rehash_query(value[name])
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                group.validate_metadata(self.plan, value)

    def test_missing_context_mapping_and_driver_capture_negatives(self):
        """No context fallback, invented mapping or changed actual driver bytes are accepted."""
        mutations = [lambda v: v["native"].update(current_context=None),
            lambda v: v.update(physical_gres_mapping_verified=True),
            lambda v: v["native"].update(current_context_device=7),
            lambda v: v["driver_capture"]["captured"].update(sha256="0"*64)]
        for mutate in mutations:
            value = copy.deepcopy(self.binding)
            mutate(value)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                group.validate_binding(self.plan, self.metadata, value, self.worker)

    def test_live_ledger_required_before_any_query(self):
        """Removing the actual own row fails without issuing a scheduler or GPU command."""
        source = self.old.run / "slurm/submitted_jobids_uva.txt"
        source.write_text("")
        with patch.object(group.single, "bounded_query") as query, patch.object(group, "source_identity") as numerical:
            with self.assertRaises(ValueError):
                group.owned_inputs(self.plan)
            query.assert_not_called()
            numerical.assert_not_called()

    def test_explicit_step_commands_do_not_launch(self):
        """Coordinator uses documented --gres=none; worker requests one typed GPU."""
        with patch.object(group.single, "bounded_query") as query:
            coordinator = producer.step_command(self.plan, self.old.job, "coordinator", ["/private/program"])
            worker = producer.step_command(self.plan, self.old.job, "worker", ["/private/program"])
        self.assertIn("--gres=none", coordinator)
        self.assertIn("--gres=gpu:nvidia_rtx_a4500:1", worker)
        self.assertNotIn("--gpus=0", coordinator)
        query.assert_not_called()

    def test_global_seed_protocol_and_exact_eight_gate(self):
        """Fixed total retains eight seeds; a smaller-count eligibility claim is refused."""
        self.assertEqual(evidence.validate_protocol(self.plan), self.plan["configuration"])
        fixed = copy.deepcopy(self.plan)
        fixed["scaling_protocol"]["kind"] = "fixed_total_eight"
        evidence.validate_protocol(fixed)
        fixed["scaling_protocol"]["kind"] = "exact_eight_eligibility"
        with self.assertRaises(ValueError):
            evidence.validate_protocol(fixed)
        changed = copy.deepcopy(self.plan)
        changed["copy_seeds"][7] = [1]
        with self.assertRaises(ValueError):
            evidence.validate_protocol(changed)

    def test_coordinator_never_has_native_device_binding(self):
        """The physical binder refuses a zero-GPU coordinator even with a synthetic device."""
        value = copy.deepcopy(self.metadata)
        value.update(role="coordinator", directory=str(self.unit))
        with patch.object(group, "validate_metadata", return_value=value), patch.object(group.single, "native_identity") as native:
            with self.assertRaises(ValueError):
                group.bind_process_attempt(self.plan, value, object(), self.worker, {"native": {}})
            native.assert_not_called()


class CommonWallTests(unittest.TestCase):
    """Exercise rates and bounded transport without a numerical library."""
    def test_total_work_over_one_interval_not_sum_of_rates(self):
        """Different worker durations still use the coordinator's common wall denominator."""
        plan = {"gpu_count": 8, "configuration": {"learner": {"n_copies": 1}},
            "iterations": 10, "scaling_protocol": {"kind": "same_copies_per_card"}}
        rows = [{"training_transitions": 163840, "start_time_seconds": 10., "end_time_seconds": 12. + i}
            for i in range(8)]
        resolved = {"members": [{"index": index} for index in range(8)]}
        with patch.object(evidence, "validate_protocol"), patch.object(evidence, "worker_round", return_value=.2):
            value = evidence.aggregate_round(plan, resolved, rows, 9., 20., 1)
        self.assertEqual(value["training_transitions"], 8 * 163840)
        self.assertEqual(value["transitions_per_second_total"], 8 * 163840 / 11)

    def test_nonoverlap_incomplete_or_nonfinite_group(self):
        """Missing workers, reversed/future windows and no concurrency are refused."""
        plan = {"gpu_count": 8, "configuration": {"learner": {"n_copies": 1}}, "iterations": 10,
            "scaling_protocol": {"kind": "same_copies_per_card"}}
        resolved = {"members": [{"index": index} for index in range(8)]}
        rows = [{"training_transitions": 163840, "start_time_seconds": float(i+1), "end_time_seconds": float(i+2)}
            for i in range(8)]
        with patch.object(evidence, "validate_protocol"), patch.object(evidence, "worker_round", return_value=.2):
            for data, start, end in ((rows, 0., 10.), (rows[:-1], 1., 10.),
                    (rows, 2., 1.), (rows, float("nan"), 10.), (rows, 1., 10.)):
                with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                    evidence.aggregate_round(plan, resolved, data, start, end, 1)

    def test_one_deadline_for_entire_multiworker_barrier(self):
        """The second worker receives only the remaining common timeout."""
        values = []
        def receive(plan, attempt, index, direction, phase, number, remaining):
            values.append(remaining)
            return index
        with patch.object(producer.time, "monotonic", side_effect=[1., 2., 4.]), \
                patch.object(producer, "read_message", side_effect=receive):
            result = producer.collect_phase({"gpu_count": 2}, {}, "ready", 1, 5.)
        self.assertEqual(result, [0, 1])
        self.assertEqual(values, [4., 2.])

    def test_default_entrypoint_refuses_unqualified_execution(self):
        """New source does not silently enable existing/default/final workers."""
        with self.assertRaisesRegex(ValueError, "separately reviewed"):
            producer.refuse_unqualified_entrypoint()

    def test_no_existing_numerical_or_single_card_source_edits(self):
        """This feature owns only its three new modules and new tests."""
        import subprocess
        result = subprocess.check_output(["git", "-C", str(SCRIPTS.parents[2]), "diff", "--name-only"], text=True)
        self.assertEqual(result.strip(), "")


if __name__ == "__main__":
    unittest.main()
