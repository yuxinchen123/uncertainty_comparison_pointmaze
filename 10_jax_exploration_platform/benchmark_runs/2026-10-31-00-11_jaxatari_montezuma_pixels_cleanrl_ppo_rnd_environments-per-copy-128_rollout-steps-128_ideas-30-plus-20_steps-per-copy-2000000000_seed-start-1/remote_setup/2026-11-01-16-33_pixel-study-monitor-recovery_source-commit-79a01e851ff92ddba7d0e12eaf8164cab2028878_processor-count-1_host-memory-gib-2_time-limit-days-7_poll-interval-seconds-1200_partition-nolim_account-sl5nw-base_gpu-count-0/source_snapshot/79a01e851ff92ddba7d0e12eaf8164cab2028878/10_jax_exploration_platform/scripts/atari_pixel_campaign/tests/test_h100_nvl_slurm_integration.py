"""Verify strict inactive H100 dispatch, portable reads and unchanged legacy metadata paths."""

import ast
import copy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import controller
import post_comparison_validation as post
import h100_nvl_slurm_queue as queue
import h100_nvl_slurm_evidence as history
import h100_nvl_slurm_site_validation as site
import test_h100_nvl_slurm_historical_evidence as historical_tests
from portable_evidence import ExactPathResolver


REAL_COHORT = post.validate_cohort_case
DEPENDENCIES = set(post.H100_NVL_SLURM_DEPENDENCIES)


class DispatchTests(unittest.TestCase):
    """Replace only numerical proof readers to inspect routing decisions and exact adapter arguments."""

    def setUp(self):
        """Create one complete raw numerical-rejection fixture with original algorithm settings."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.attempt = self.root / "data/unit/attempts/attempt-0001"
        self.attempt.mkdir(parents=True)
        self.path = self.attempt / "result.json"
        self.execution = {"compilation_mode": "whole_iteration", "iterations_per_call": 1, "report_readout": "none", "report_every_iterations": 10}
        self.environment = dict(post.ENVIRONMENT)
        self.case = {"unit_id": "unit", "campaign_run_dir": str(self.root), "source_root": str(self.root / "source"),
            "source_commit": "a" * 40, "maximum_attempts": 3, "hardware_class": "h100", "hypothesis": "synthetic comparison",
            "configuration": {**post.ALGORITHM, "n_copies": 1, "rollout_unroll": 1}, "reference_execution": {},
            "candidate": {"learner": {"rollout_unroll": 2}, "environment": {}, "execution": {}}, "rounds": 7, "iterations": 1}
        self.case["specification_digest"] = post.canonical_digest(self.case)
        self.hardware = {"hostname": "serval06", "slurm_job_id": "333", "source_root": self.case["source_root"],
            "source_commit": self.case["source_commit"], "hardware_class": "h100", "pid": 10, "process_start_time": "20"}
        self.lease = {"attempt": 1, "hostname": "serval06", "slurm_job_id": "333"}
        reference = {"learner": self.case["configuration"], "environment": self.environment, "execution": self.execution}
        candidate = copy.deepcopy(reference)
        candidate["learner"]["rollout_unroll"] = 2
        self.result = {"case_completed": True, "idea": self.case["hypothesis"], "unit_id": "unit", "copies": 1,
            "source_commit": self.case["source_commit"], "specification_digest": self.case["specification_digest"], "hardware_class": "h100",
            "reference": reference, "candidate": candidate, "implementation_distinct": True, "attempt": 1,
            "failure_stage": "correctness", "correctness_attempted": True, "correctness_passed": False,
            "decision": "rejected", "error": "synthetic comparison mismatch"}
        self.proof = {"runtime_environment": site.RUNTIME, "placement_policy": {"path": "/approved/policy", "sha256": "a" * 64},
            "allocation_validation": {"path": "/checked/site/result.json", "sha256": "b" * 64}}
        self.save()

    def save(self):
        """Serialize exact current fixture bytes without invoking a numerical component."""
        for name, value in (("case.json", self.case), ("hardware.json", self.hardware), ("result.json", self.result), ("slurm_execution.json", self.proof)):
            controller.atomic_json(self.attempt / name, value)

    def cohort(self):
        """Exercise the production cohort adapter while keeping source defaults explicitly synthetic."""
        evidence = post.Evidence()
        with patch.object(post, "case_source_defaults", return_value=(self.environment, self.execution, "source")):
            result = REAL_COHORT(self.path, self.case["hypothesis"], self.case["hardware_class"], 1, self.case["source_commit"], evidence)
        return result, evidence

    def test_controller_routes_job_host_and_explicit_marker_before_any_legacy_retry(self):
        """Removing one optional marker never turns an H100 Slurm claim into an ordinary legacy claim."""
        variants = [({"attempt": 1}, {**self.hardware, "slurm_job_id": ""}),
            ({"attempt": 1, "slurm_job_id": "333", "hostname": "other-host"}, {**self.hardware, "hostname": "other-host", "slurm_job_id": ""}),
            ({"attempt": 5, "execution_kind": queue.EXECUTION_KIND}, {})]
        validator = types.SimpleNamespace(validate_execution=Mock(side_effect=ValueError("required strict proof")))
        for lease, hardware in variants:
            controller.atomic_json(self.attempt / "hardware.json", hardware)
            with self.subTest(lease=lease), patch.object(controller, "h100_nvl_slurm_validation_module", return_value=validator):
                with self.assertRaisesRegex(ValueError, "strict proof"):
                    controller.validate_attempt_completion(self.root, self.case, lease, self.path)
            self.assertEqual(validator.validate_execution.call_args.kwargs, {"require_completed": True})

    def test_controller_preserves_direct_h100_and_a4500_ordinary_dispatch(self):
        """Existing direct H100 and Slurm A4500 ordinary cases do not require a new H100 receipt."""
        with patch.object(controller, "h100_nvl_slurm_validation_module") as loader:
            for context, host, job in (("h100", "serval02", ""), ("jaguar03_rtx_a4500", "jaguar03", "333")):
                controller.atomic_json(self.attempt / "hardware.json", {"hostname": host, "slurm_job_id": job})
                self.assertIsNone(controller.validate_attempt_completion(self.root, {**self.case, "hardware_class": context},
                    {"attempt": 1, "hostname": host, "slurm_job_id": job}, self.path))
        loader.assert_not_called()

    def test_controller_preserves_approved_a4500_fourth_and_fifth_validators(self):
        """New placement dispatch never redirects existing exact approved A4500 retry records."""
        controller.atomic_json(self.attempt / "hardware.json", {"hostname": "jaguar03", "slurm_job_id": "333"})
        for attempt, filename in ((4, "infrastructure_retry.py"), (5, "pregather_async_retry.py")):
            module = types.SimpleNamespace(validate_execution=Mock(return_value="legacy retry proof"))
            specification = types.SimpleNamespace(loader=types.SimpleNamespace(exec_module=Mock()))
            with self.subTest(attempt=attempt), patch.object(controller.importlib.util, "spec_from_file_location", return_value=specification) as spec, \
                    patch.object(controller.importlib.util, "module_from_spec", return_value=module):
                self.assertEqual(controller.validate_attempt_completion(self.root, {**self.case, "hardware_class": "jaguar03_rtx_a4500"},
                    {"attempt": attempt}, self.path), "legacy retry proof")
                self.assertEqual(Path(spec.call_args.args[1]).name, filename)

    def test_post_passes_same_evidence_and_requires_exact_completed_original_result(self):
        """The historical caller preserves its resolver and rejects another result returned by the proof."""
        module = types.SimpleNamespace(validate_execution=Mock(return_value=(self.proof, self.result)))
        with patch.object(post, "h100_nvl_slurm_validation_module", return_value=module), patch.object(post, "cohort_runtime_evidence") as legacy:
            proof, evidence = self.cohort()
        legacy.assert_not_called()
        self.assertEqual(proof["scope"], "actual_h100_nvl_slurm_runtime")
        self.assertIs(module.validate_execution.call_args.kwargs["evidence"], evidence)
        self.assertTrue(module.validate_execution.call_args.kwargs["require_completed"])
        module.validate_execution.return_value = (self.proof, {**self.result, "copies": 2})
        with patch.object(post, "h100_nvl_slurm_validation_module", return_value=module):
            with self.assertRaisesRegex(ValueError, "differs from.*cohort result"):
                self.cohort()

    def test_post_stripped_job_marker_with_slurm_only_host_cannot_use_legacy(self):
        """A source-only runtime inference cannot replace a missing physical Slurm execution proof."""
        self.hardware.pop("slurm_job_id")
        self.save()
        module = types.SimpleNamespace(validate_execution=Mock(side_effect=ValueError("missing execution proof")))
        with patch.object(post, "h100_nvl_slurm_validation_module", return_value=module), patch.object(post, "cohort_runtime_evidence") as legacy:
            with self.assertRaisesRegex(ValueError, "missing execution proof"):
                self.cohort()
        legacy.assert_not_called()

    def test_post_legacy_direct_h100_and_a4500_use_original_dispatch(self):
        """Actual ordinary legacy data remains readable without a Slurm site policy."""
        for context, host, job in (("h100", "serval02", ""), ("jaguar03_rtx_a4500", "jaguar03", "333")):
            self.case["hardware_class"] = context
            self.case["specification_digest"] = post.canonical_digest({key: value for key, value in self.case.items() if key != "specification_digest"})
            self.result.update(hardware_class=context, specification_digest=self.case["specification_digest"])
            self.hardware.update(hardware_class=context, hostname=host, slurm_job_id=job)
            self.save()
            with self.subTest(context=context), patch.object(post, "cohort_runtime_evidence", return_value={"scope": "legacy"}), \
                    patch.object(post, "h100_nvl_slurm_validation_module") as loader:
                self.assertEqual(self.cohort()[0], {"scope": "legacy"})
            loader.assert_not_called()

    def test_recovery_rejects_foreign_cluster_id_and_nonterminal_owned_job(self):
        """Same-number foreign ledgers or a still-running owned job cannot authorize recovery."""
        controller.layout(self.root)
        self.lease.update(token="token", hostname="serval06", worker_pid=10, worker_start_time="20", runner_pid=30, runner_start_time="40")
        controller.atomic_json(self.root / "queue/running/unit.json", self.case)
        controller.atomic_json(self.root / "queue/leases/unit.json", self.lease)
        (self.root / "slurm").mkdir()
        (self.root / "slurm/submitted_jobids_aces.txt").write_text("333\n")
        with patch.object(controller.socket, "gethostname", return_value="serval06"), patch.object(controller, "process_matches", return_value=False), \
                patch.object(controller.subprocess, "check_output") as query:
            with self.assertRaisesRegex(ValueError, "campaign-owned UVA"):
                controller.recover_local_claim(self.root, "unit", "token")
            query.assert_not_called()
            (self.root / "slurm/submitted_jobids_uva.txt").write_text("333\n")
            query.return_value = "333|RUNNING|0:0\n"
            with self.assertRaisesRegex(ValueError, "not uniquely terminal"):
                controller.recover_local_claim(self.root, "unit", "token")
        self.assertTrue((self.root / "queue/running/unit.json").exists())
        self.assertFalse((self.attempt / "slurm_recovery.json").exists())


class FullProofIntegrationTests(unittest.TestCase):
    """Use the complete existing synthetic source/site/proof fixture and its actual byte validators."""

    @classmethod
    def setUpClass(cls):
        """Read committed source bytes once without importing numerical files."""
        historical_tests.HistoricalEvidenceTests.setUpClass()

    def setUp(self):
        """Make a full isolated proof and connect production dispatch to its explicitly synthetic constants."""
        self.fixture = historical_tests.HistoricalEvidenceTests("test_complete_history_uses_captured_admission_and_never_reader_packages")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        fixture = self.fixture
        hardware_path = fixture.attempt / "hardware.json"
        hardware = json.loads(hardware_path.read_bytes())
        hardware["hardware_class"] = "h100"
        fixture.execution["hardware"] = fixture.write(hardware_path, hardware)
        fixture.write(fixture.attempt / "slurm_execution.json", fixture.execution)
        fixture.write_intent()
        original_queue_load = queue.load
        patch.object(queue, "load", side_effect=lambda name: history if name == "h100_nvl_slurm_evidence" else original_queue_load(name)).start()
        patch.object(post, "h100_nvl_slurm_validation_module", return_value=queue).start()
        patch.object(controller, "h100_nvl_slurm_validation_module", return_value=queue).start()

    def cohort(self, evidence=None):
        """Call the real production cohort function rather than the prior-packet fixture's mock."""
        f = self.fixture
        evidence = post.Evidence() if evidence is None else evidence
        return REAL_COHORT(f.attempt / "result.json", f.case["hypothesis"], "h100", 1, f.commit, evidence), evidence

    def test_completed_proof_passes_controller_and_actual_cohort_validation(self):
        """The adapted ordinary H100 receipt preserves raw successful comparisons through both integration points."""
        f = self.fixture
        controller.validate_attempt_completion(f.run, f.case, f.lease, f.attempt / "result.json")
        self.assertEqual(self.cohort()[0]["scope"], "actual_h100_nvl_slurm_runtime")

    def test_complete_raw_result_cannot_override_actual_nonzero_child_exit(self):
        """A complete canonical result and rehashed publication intent still reject a failed child."""
        f = self.fixture
        path = f.attempt / "slurm_child_terminal.json"
        terminal = json.loads(path.read_bytes())
        terminal["exit_code"] = 1
        f.write(path, terminal)
        f.write_intent()
        with self.assertRaises(ValueError):
            controller.validate_attempt_completion(f.run, f.case, f.lease, f.attempt / "result.json")
        with self.assertRaises(ValueError):
            self.cohort()

    def test_rehashed_execution_from_another_job_cannot_complete_this_case(self):
        """A different owned job's execution cannot replace this lease merely by changing transport hashes."""
        f = self.fixture
        changed = {**f.execution, "job_id": "444"}
        f.write(f.attempt / "slurm_execution.json", changed)
        f.write_intent()
        with self.assertRaises(ValueError):
            controller.validate_attempt_completion(f.run, f.case, f.lease, f.attempt / "result.json")
        with self.assertRaises(ValueError):
            self.cohort()

    def test_actual_cohort_dispatch_survives_exact_relocation_and_changed_live_admission(self):
        """Portable completion uses only captured historical bytes after all original source/evidence roots are removed."""
        f = self.fixture
        f.write(f.run / "SOURCE_AND_GATES_READY.json", {"authorized": False, "source_commit": "later"})
        before, evidence = self.cohort()
        manifest = f.portable_package(evidence.records())
        for path in list(f.base.iterdir()):
            if path != manifest.parent:
                shutil.rmtree(path) if path.is_dir() else path.unlink()
        relocated, mapped = self.cohort(post.Evidence(ExactPathResolver(manifest)))
        self.assertEqual(relocated, before)
        self.assertEqual(mapped.records(), evidence.records())

    def recovery_fixture(self, *, completed=False):
        """Retain a complete child proof while simulating an interrupted parent publication."""
        f = self.fixture
        controller.layout(f.run)
        f.write(f.run / "queue/specifications" / (f.case["unit_id"] + ".json"), f.case)
        f.write(f.run / "queue/running" / (f.case["unit_id"] + ".json"), f.case)
        lease = {**f.lease, "hostname": "serval06", "state": "completed" if completed else "running"}
        f.write(f.attempt / "lease.json", lease)
        f.write(f.run / "queue/leases" / (f.case["unit_id"] + ".json"), lease)
        f.write(f.run / "slurm/submitted_jobids_uva.txt", b"333\n")
        if not completed:
            for path in queue.terminal_paths(f.unit, f.attempt).values():
                path.unlink()
            (f.attempt / "slurm_publication_intent.json").unlink()
        return lease

    def recovery_patches(self):
        """Return only exact owned accounting metadata; keep all Git/source checks real."""
        original = controller.subprocess.check_output
        def query(command, *args, **kwargs):
            """Substitute the single owned sacct command and inspect its full-width state format."""
            if command[0] == "sacct":
                self.assertEqual(command[command.index("--jobs") + 1], "333")
                self.assertIn("State%64", command[-1])
                return "333|COMPLETED|0:0\n"
            return original(command, *args, **kwargs)
        return (patch.object(controller.socket, "gethostname", return_value="serval06"),
            patch.object(controller, "process_matches", return_value=False),
            patch.object(controller.subprocess, "check_output", side_effect=query))

    def test_staged_zero_exit_recovery_keeps_original_bytes_after_ledger_append(self):
        """A second recovery reuses its original immutable ledger prefix and completes a recorded partial publication."""
        f = self.fixture
        self.recovery_fixture()
        a, b, c = self.recovery_patches()
        with a, b, c, patch.object(queue, "publish_bytes", side_effect=OSError("synthetic publication interruption")):
            with self.assertRaisesRegex(OSError, "publication interruption"):
                controller.recover_local_claim(f.run, f.case["unit_id"], f.lease["token"])
        before = (f.attempt / "slurm_recovery.json").read_bytes()
        f.write(f.run / "slurm/submitted_jobids_uva.txt", b"333\n444\n")
        a, b, c = self.recovery_patches()
        with a, b, c:
            controller.recover_local_claim(f.run, f.case["unit_id"], f.lease["token"])
        self.assertEqual((f.attempt / "slurm_recovery.json").read_bytes(), before)
        self.assertTrue((f.run / "queue/completed" / (f.case["unit_id"] + ".json")).exists())
        for name, path in queue.terminal_paths(f.unit, f.attempt).items():
            self.assertEqual(path.read_bytes(), Path(f.execution["terminal_files"][name]["path"]).read_bytes())
        self.assertEqual(self.cohort()[0]["scope"], "actual_h100_nvl_slurm_runtime")

    def test_completed_leases_with_pending_final_rename_recover_without_progress_rewrite(self):
        """Recovery of the final queue transition preserves all original result and completed-progress bytes."""
        f = self.fixture
        self.recovery_fixture(completed=True)
        before = {name: path.read_bytes() for name, path in queue.terminal_paths(f.unit, f.attempt).items()}
        a, b, c = self.recovery_patches()
        with a, b, c:
            controller.recover_local_claim(f.run, f.case["unit_id"], f.lease["token"])
        self.assertTrue((f.run / "queue/completed" / (f.case["unit_id"] + ".json")).exists())
        self.assertEqual({name: path.read_bytes() for name, path in queue.terminal_paths(f.unit, f.attempt).items()}, before)
        self.cohort()


class DependencyIntegrationTests(unittest.TestCase):
    """Preserve historical helper identities while requiring every new reachable validator dependency."""

    def test_future_helper_lists_include_four_files_and_legacy_operational_lists_stay_unchanged(self):
        """Every producer/training path that can read new cohort proof stages its explicit lazy imports."""
        for name in ("measure_selected", "profile_selected", "measure_gpu_scaling", "diagnose_correctness_cache", "diagnose_allocator_region", "diagnose_pregather_allocator", "train_deployed"):
            module = importlib.import_module(name)
            with self.subTest(module=name):
                self.assertTrue(DEPENDENCIES <= set(module.HELPERS))
                self.assertEqual(len(module.HELPERS), len(set(module.HELPERS)))
        launcher = importlib.import_module("launch_final_training")
        self.assertTrue(DEPENDENCIES <= set(launcher.TRAIN_HELPERS))
        self.assertTrue({site.PREFIX + name for name in DEPENDENCIES} <= importlib.import_module("validate_training_deployment").REQUIRED_HELPERS)
        self.assertNotIn("h100_nvl_slurm_queue.py", importlib.import_module("infrastructure_retry").HELPERS)
        self.assertNotIn("h100_nvl_slurm_queue.py", site.HELPERS)

    def test_source_sensitive_requirements_preserve_old_pinned_callers_and_require_new_complete_closure(self):
        """Only already verified source declarations add new files; old receipts are never rewritten."""
        prefix = site.PREFIX
        required = {prefix + name for name in (*DEPENDENCIES, "post_comparison_validation.py", "train_optimized.py")}
        old = {prefix + "post_comparison_validation.py": b"# original caller without new dispatch\n"}
        self.assertEqual(post.helper_requirements(required, old), {prefix + "post_comparison_validation.py", prefix + "train_optimized.py"})
        current = {prefix + "post_comparison_validation.py": b"H100_NVL_SLURM_QUEUE_MODULE='h100_nvl_slurm_queue.py'\n"}
        self.assertTrue({prefix + name for name in DEPENDENCIES} <= post.helper_requirements(set(), current))

    def test_real_committed_old_and_new_helper_receipts_use_source_sensitive_minima(self):
        """Actual adaptive/deployment readers preserve old source identities and reject incomplete new callers."""
        adaptive = importlib.import_module("advance_adaptive_research")
        deployment = importlib.import_module("validate_training_deployment")
        prefix = site.PREFIX
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            names = ("post_comparison_validation.py", "train_optimized.py")
            for name in names:
                path = root / prefix / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# old committed metadata caller\n")
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            def committed(selected):
                """Commit only isolated synthetic metadata and return its exact receipt."""
                subprocess.run(["git", "-C", str(root), "add", "."], check=True)
                subprocess.run(["git", "-C", str(root), "-c", "user.name=Metadata test", "-c", "user.email=test@example.invalid",
                    "-c", "core.hooksPath=/dev/null", "commit", "-qm", "Record synthetic metadata source"], check=True)
                commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
                subprocess.run(["git", "-C", str(root), "checkout", "--detach", "-q", commit], check=True)
                return {"source_root": str(root), "source_commit": commit,
                    "files": {prefix + name: hashlib.sha256((root / prefix / name).read_bytes()).hexdigest() for name in selected}}
            required = {prefix + name for name in (*names, *DEPENDENCIES)}
            original = committed(names)
            before = copy.deepcopy(original)
            adaptive.committed_files(original, post.Evidence(), required)
            deployment.committed_files(original, required, post.Evidence())
            self.assertEqual(original, before)
            (root / prefix / "post_comparison_validation.py").write_text("H100_NVL_SLURM_QUEUE_MODULE='h100_nvl_slurm_queue.py'\n")
            incomplete = committed(names)
            with self.assertRaisesRegex(ValueError, "required"):
                adaptive.committed_files(incomplete, post.Evidence(), required)
            with self.assertRaisesRegex(ValueError, "required"):
                deployment.committed_files(incomplete, required, post.Evidence())
            for name in DEPENDENCIES:
                (root / prefix / name).write_text("# explicit new historical metadata dependency\n")
            complete = committed((*names, *DEPENDENCIES))
            adaptive.committed_files(complete, post.Evidence(), required)
            deployment.committed_files(complete, required, post.Evidence())

    def test_alias_loaded_post_uses_adjacent_staged_modules_and_rejects_missing_dependency(self):
        """A wrong ambient module cannot replace a pinned validator imported from an explicit staged path."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            for name in ("post_comparison_validation.py", *DEPENDENCIES):
                (root / name).write_bytes((SCRIPTS / name).read_bytes())
            program = """import importlib.util,pathlib,sys,types
root=pathlib.Path(sys.argv[1]);sys.modules['h100_nvl_slurm_queue']=types.SimpleNamespace(__file__='/wrong/global.py')
spec=importlib.util.spec_from_file_location('explicit_post_alias',root/'post_comparison_validation.py');post=importlib.util.module_from_spec(spec);spec.loader.exec_module(post)
queue=post.h100_nvl_slurm_validation_module();assert pathlib.Path(queue.__file__)==root/'h100_nvl_slurm_queue.py'
history=queue.load('h100_nvl_slurm_evidence');assert pathlib.Path(history.__file__)==root/'h100_nvl_slurm_evidence.py'
assert not any(name in sys.modules for name in ('jax','numpy','torch','controller','ideas'))
(root/'h100_nvl_slurm_queue.py').unlink()
try:post.h100_nvl_slurm_validation_module()
except FileNotFoundError:pass
else:raise AssertionError('missing adjacent validator accepted')
"""
            subprocess.run([sys.executable, "-I", "-c", program, str(root)], check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
