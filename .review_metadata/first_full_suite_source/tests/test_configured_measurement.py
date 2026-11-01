"""Reject unsupported or incomplete single-card evidence without any GPU or numerical import."""

import ast
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import configured_measurement as support
import measure_selected as selected
import profile_selected as profile
import post_comparison_validation as validation

GPU = "GPU-12345678-1234-1234-1234-123456789abc"
PID = 1234
ISOLATED_IMPORT_PROBE = r'''
import builtins,ctypes,hashlib,importlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]).resolve(strict=True)
names=json.loads(sys.argv[2])
expected={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in {'jax','jaxlib','numpy','torch','jaxtari','exploration_platform','pynvml','cupy','numba'}:
        raise AssertionError('numerical import forbidden')
    return original(name,*args,**kwargs)
builtins.__import__=guarded
def audit(event,args):
    if event in {'subprocess.Popen','socket.connect','ctypes.dlopen'}:
        raise AssertionError('live operation forbidden:'+event)
    if event=='open' and isinstance(args[0],str) and args[0].startswith('/dev/'):
        raise AssertionError('device open forbidden')
sys.addaudithook(audit)
sys.path.insert(0,str(root))
seen={}
for filename in names:
    module=importlib.import_module(filename[:-3])
    path=pathlib.Path(module.__file__).resolve(strict=True)
    assert path==root/filename
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected[filename]
    seen[filename]=str(path)
if 'uva_slurm_configured_isolation.py' in names:
    core=sys.modules['uva_slurm_configured_isolation']
    for filename in core.HELPERS:
        if filename=='uva_slurm_configured_isolation.py':continue
        module=core.adjacent(filename[:-3])
        path=pathlib.Path(module.__file__).resolve(strict=True)
        assert path==root/filename and hashlib.sha256(path.read_bytes()).hexdigest()==expected[filename]
for name,module in list(sys.modules.items()):
    filename=getattr(module,'__file__',None)
    if filename and pathlib.Path(filename).name in expected:
        assert pathlib.Path(filename).resolve(strict=True)==root/pathlib.Path(filename).name
print(json.dumps({'loaded':seen,'numerical_imports':False,'live_operations':0},sort_keys=True))
'''


def raw(stdout, *, returncode=0, stderr=b"", complete=True, error=None):
    """Create bounded-query bytes without a subprocess or GPU command."""
    return {"stdout": stdout, "stderr": stderr, "returncode": returncode, "complete": complete, "capture_error": error}


def own_sample(pid=PID):
    """Make one complete successful raw observation with a real typed own-row conversion."""
    with patch.object(support.time, "time", side_effect=[1., 1.1]):
        return support.process_sample(GPU, pid, lambda command: raw(f"{GPU}, {pid}, 512\n".encode()))


def memory_window():
    """Make exact boundary/raw/device/allocator evidence that the reader independently rebuilds."""
    process = own_sample()
    text = f"{GPU}, 20470, 512, 19958\n"
    sample = {"timestamp_seconds": 1.2, "gpu_uuid": GPU, "total_bytes": 20470 * 2 ** 20,
        "used_bytes": 512 * 2 ** 20, "free_bytes": 19958 * 2 ** 20, "process_query": process,
        "device_query": {"command": ["nvidia-smi", "-i", GPU, "--query-gpu=uuid,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits"], "stdout": text, "sha256": support.sha(text.encode()),
            "returncode": 0, "stderr": "", "complete": True, "capture_error": None,
            "started_at_seconds": .8, "finished_at_seconds": .9}}
    last = copy.deepcopy(sample); last["timestamp_seconds"] = 3.2
    last["process_query"]["started_at_seconds"] = 3.
    last["process_query"]["finished_at_seconds"] = 3.1
    last["device_query"]["started_at_seconds"] = 2.8
    last["device_query"]["finished_at_seconds"] = 2.9
    return {"schema_version": 1, "kind": support.CAPABILITY, "gpu_uuid": GPU, "pid": PID,
        "window_started_at_seconds": 1.5, "window_finished_at_seconds": 2.5,
        "interval_seconds": .2, "samples": [sample, last], "errors": [], "alive_at_requested_stop": True,
        "normal_requested_stop": True, "joined": True, "allocator": {"peak_bytes_in_use": 123456}}


class AssignedProcessTests(unittest.TestCase):
    """Keep every identifying foreign row out of persisted records while requiring real own-only evidence."""

    def test_success_preserves_exact_command_bytes_and_positive_peak(self):
        """Rebuild the exact assigned-UUID command and own process bytes from saved raw CSV."""
        value = own_sample()
        self.assertEqual(value["command"], ["nvidia-smi", "-i", GPU,
            "--query-compute-apps=gpu_uuid,pid,used_gpu_memory", "--format=csv,noheader,nounits"])
        self.assertEqual(support.validate_process_sample(value, GPU, PID), 512 * 2 ** 20)
        self.assertEqual(value["other_active_compute_rows"], 0)

    def test_other_process_is_anonymous_even_at_zero_memory(self):
        """A foreign row invalidates the sample and saves only a count and byte hashes."""
        value = support.process_sample(GPU, PID, lambda command: raw(f"{GPU}, {PID}, 512\n{GPU}, 987654, 0\n".encode()))
        self.assertFalse(value["valid"])
        self.assertEqual(value["other_active_compute_rows"], 1)
        self.assertNotIn("987654", json.dumps(value))
        self.assertNotIn("stdout", value)
        with self.assertRaises(ValueError):
            support.validate_process_sample(value, GPU, PID)

    def test_missing_own_process_never_becomes_empty_success(self):
        """An empty query has unknown occupancy/memory and cannot supply an observed empty list."""
        value = support.process_sample(GPU, PID, lambda command: raw(b""))
        self.assertFalse(value["valid"])
        self.assertNotIn("own", value)

    def test_invalid_rows_and_duplicate_rows_refuse(self):
        """Wrong UUID, repeated PID, unavailable/negative memory and malformed CSV cannot receive credit."""
        bad = [f"{GPU}, {PID}, N/A\n", f"{GPU}, {PID}, -1\n", f"{GPU}, {PID}, nan\n",
            f"{GPU}, {PID}, 0\n", f"{GPU}, 0, 2\n", f"{GPU}, {PID}, 1\n{GPU}, {PID}, 1\n",
            f"GPU-ffffffff-ffff-ffff-ffff-ffffffffffff, {PID}, 1\n", f"{GPU}, {PID}\n", '"unfinished']
        for text in bad:
            with self.subTest(text=text):
                value = support.process_sample(GPU, PID, lambda command: raw(text.encode()))
                self.assertFalse(value["valid"])
                self.assertNotIn("stdout", value)

    def test_failed_or_partial_query_cannot_keep_an_early_positive_row(self):
        """Return code, stderr, timeout and truncated capture invalidate otherwise valid own rows."""
        text = f"{GPU}, {PID}, 512\n".encode()
        for kwargs in ({"returncode": 1}, {"stderr": b"private text"}, {"complete": False}, {"error": "TimeoutError"}):
            with self.subTest(kwargs=kwargs):
                value = support.process_sample(GPU, PID, lambda command: raw(text, **kwargs))
                self.assertFalse(value["valid"])
                self.assertNotIn("stdout", value)
                self.assertNotIn("private text", json.dumps(value))

    def test_saved_proof_changes_are_rejected(self):
        """Changing the command, raw row/hash, PID, UUID, typed memory or timestamp breaks the reader."""
        original = own_sample()
        for key, changed in (("command", ["nvidia-smi"]), ("stdout", "different"), ("stdout_sha256", "0" * 64),
                ("own_pid", PID + 1), ("gpu_uuid", GPU + "x"), ("own", {"pid": PID, "bytes": 1}),
                ("started_at_seconds", float("nan"))):
            with self.subTest(key=key), self.assertRaises((ValueError, KeyError)):
                value = copy.deepcopy(original); value[key] = changed
                support.validate_process_sample(value, GPU, PID)

    def test_no_query_before_actual_current_binding(self):
        """An unresolved, remembered or another-process binding refuses before any subprocess."""
        for binding in ({}, {"kind": "uva_current_process_resolved_device_binding", "process": {"pid": PID}, "gpu_uuid": GPU}):
            with self.subTest(binding=binding), patch.object(support, "bounded_query") as query:
                with self.assertRaises((ValueError, KeyError)):
                    support.MemoryObservation(None, binding, Path("/unused"))
                query.assert_not_called()


class CompleteWindowTests(unittest.TestCase):
    """Validate observed memory and thread lifecycle independently of numerical timing summaries."""

    def test_complete_window_rebuilds_actual_peak_and_empty_observed_list(self):
        """Only complete successful own-only samples produce the original round's required empty list."""
        value = support.validate_window(memory_window())
        self.assertEqual(value["sampled_process_peak_bytes"], 512 * 2 ** 20)
        self.assertEqual(value["other_gpu_processes"], [])
        self.assertIs(value["other_gpu_processes_observed"], True)
        self.assertEqual(value["allocator"]["peak_bytes_in_use"], 123456)

    def test_late_sampler_failure_and_abnormal_shutdown_refuse(self):
        """An early successful sample never hides errors, dead threads or incomplete joins."""
        for key, value in (("errors", ["late failure"]), ("alive_at_requested_stop", False),
                ("normal_requested_stop", False), ("joined", False), ("schema_version", True)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                proof = memory_window(); proof[key] = value
                support.validate_window(proof)

    def test_missing_boundaries_unknown_peak_and_changed_device_refuse(self):
        """The actual window must have both boundaries, an allocator peak and matching raw physical bytes."""
        for mutation in (lambda v: v.update(window_started_at_seconds=.5),
                lambda v: v.update(window_finished_at_seconds=4.), lambda v: v.update(samples=[]),
                lambda v: v.update(allocator={}), lambda v: v["samples"][0].update(gpu_uuid=GPU + "x"),
                lambda v: v["samples"][0].update(used_bytes=2),
                lambda v: v["samples"][1]["process_query"].update(valid=False)):
            with self.subTest(mutation=mutation), self.assertRaises((ValueError, KeyError)):
                proof = memory_window(); mutation(proof); support.validate_window(proof)

    def test_pair_adapter_restores_original_on_success_and_failure(self):
        """The frozen paired global observer is never called and is restored for every exit."""
        original = object()
        paired = types.SimpleNamespace(MemoryObservation=original)
        for fail in (False, True):
            try:
                with support.paired_observer(paired, {}, Path("/unused")):
                    self.assertIsNot(paired.MemoryObservation, original)
                    if fail:
                        raise RuntimeError("synthetic numerical failure")
            except RuntimeError:
                pass
            self.assertIs(paired.MemoryObservation, original)

    def test_formal_old_round_gate_stays_strict(self):
        """Neither unknown occupancy nor an invented zero process peak becomes a measured round."""
        row = {"iterations": 10, "copies": 1, "environments_per_copy": 128, "rollout_steps": 128,
            "training_transitions": 163840, "seconds": 2., "memory": support.validate_window(memory_window())}
        self.assertEqual(validation.measured_round(row, 1), 81920.)
        for value in (None, [1]):
            bad = copy.deepcopy(row); bad["memory"]["other_gpu_processes"] = value
            with self.assertRaises(ValueError):
                validation.measured_round(bad, 1)


class ProducerBoundaryTests(unittest.TestCase):
    """Keep opt-in CLI/source boundaries separate from every existing consumer and default."""

    def test_prepared_measurement_rejects_new_preparation_flags(self):
        """Execution must not silently ignore a profile, qualification or source override."""
        for flags in (["--configured-isolation-profile", "/x"], ["--target-qualification"],
                ["--target-qualification-record", "/x"], ["--target-qualification-review", "/x"],
                ["--cleanrl-source-root", "/x"], ["--qualification-software-result", "/x"]):
            with self.subTest(flags=flags), patch.object(sys, "stderr", io.StringIO()), self.assertRaises(SystemExit):
                selected.parse_arguments(["--prepared-unit", "/unused", *flags])

    def test_prepared_profile_rejects_new_preparation_flags(self):
        """The profiling child cannot change a recorded identity or qualification input."""
        for flags in (["--configured-isolation-profile", "/x"], ["--target-qualification-record", "/x"],
                      ["--target-qualification-review", "/x"]):
            with self.subTest(flags=flags), patch.object(sys, "stderr", io.StringIO()), self.assertRaises(SystemExit):
                profile.parse_arguments(["--capture-unit", "/unused", *flags])

    def test_default_helper_rosters_are_unchanged(self):
        """Historical preparation keeps its original sixteen measurement helpers and profile addition."""
        self.assertEqual(selected.required_helpers(False), selected.HELPERS)
        self.assertEqual(profile.required_helpers(False), profile.HELPERS)
        self.assertNotIn("configured_measurement.py", selected.HELPERS)
        self.assertTrue(set(support.core().HELPERS) <= set(selected.required_helpers(True)))

    def test_default_and_unsupported_readers_refuse_new_profile(self):
        """Packing/final/scaling callers cannot accept a new record through generic identity validation."""
        value = {"configured_isolation": {}, "hardware_class": "jaguar03_rtx_a4500",
            "source_commit": "a" * 40, "runtime_environment": {}}
        with self.assertRaisesRegex(ValueError, "does not support"):
            validation.identity(value, "jaguar03_rtx_a4500", "a" * 40, {})
        del value["configured_isolation"]
        value["helpers"] = {"files": {"uva_slurm_configured_isolation.py": "a" * 64}}
        with self.assertRaisesRegex(ValueError, "does not support"):
            validation.identity(value, "jaguar03_rtx_a4500", "a" * 40, {})

    def test_original_qualification_has_no_completed_study_or_baseline_dependency(self):
        """The explicit same-original path can prepare before baseline/final admission exists."""
        root = SCRIPTS.parents[2]
        defaults = selected.diagnostic.default_configuration(root)
        configuration = {name: defaults[name] for name in ("learner", "environment")}
        configuration["execution"] = {"compilation_mode": "whole_iteration", "iterations_per_call": 1,
            "report_readout": "none", "report_every_iterations": 10}
        plan = {"source_root": str(root), "source_commit": "d28e6735e79788f618acb0ebfde8609db1f776af",
            "mode": "correctness", "configuration": json.loads(json.dumps(configuration)),
            "reference_configuration": json.loads(json.dumps(configuration))}
        support.validate_original_qualification(plan)
        changed = copy.deepcopy(plan); changed["configuration"]["learner"]["n_envs"] = 1
        with self.assertRaises(ValueError):
            support.validate_original_qualification(changed)
        self.assertEqual(support.QUALIFICATION["scientific_credit"], 0)

    def test_ordinary_equal_comparison_rejection_is_still_present(self):
        """The equality exception is restricted to the explicit target qualification branch."""
        source = (SCRIPTS / "measure_selected.py").read_text()
        self.assertIn('reference == selected and not qualification', source)
        self.assertNotIn("configured_isolation", (SCRIPTS / "paired_benchmark.py").read_text())

    def test_fresh_isolated_process_resolves_only_complete_staged_helper_rosters(self):
        """Import real staged/default/combined closures with isolated Python and no ambient helper fallback."""
        rosters = {"default": selected.required_helpers(False), "profile_only": profile.required_helpers(True),
            "existing_memory_and_learning": selected.HELPERS, "combined": selected.required_helpers(True)}
        for label, names in rosters.items():
            with self.subTest(roster=label), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                for name in names:
                    (directory / name).write_bytes((SCRIPTS / name).read_bytes())
                command = [sys.executable, "-I", "-B", "-c", ISOLATED_IMPORT_PROBE, str(directory), json.dumps(names)]
                completed = subprocess.run(command, text=True, capture_output=True, timeout=30)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertEqual(set(json.loads(completed.stdout)["loaded"]), set(names))

    def test_isolated_optional_import_refuses_missing_sibling(self):
        """A missing ownership sibling cannot be replaced from the checkout or an ambient module cache."""
        names = selected.required_helpers(True)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in names:
                if name != "uva_active_configuration_observation.py":
                    (directory / name).write_bytes((SCRIPTS / name).read_bytes())
            command = [sys.executable, "-I", "-B", "-c", ISOLATED_IMPORT_PROBE, str(directory),
                json.dumps([name for name in names if name != "uva_active_configuration_observation.py"])]
            completed = subprocess.run(command, text=True, capture_output=True, timeout=30)
            self.assertNotEqual(completed.returncode, 0)


class ConfiguredPreparationTests(unittest.TestCase):
    """Exercise actual producer preparation/execution branches with only the physical operations replaced."""

    def setUp(self):
        """Reuse the existing complete metadata fixture without repeating its inherited test cases."""
        from test_measure_selected import SelectedMeasurementTests
        self.fixture = SelectedMeasurementTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.profile_path = self.fixture.run_dir / "profile.json"
        self.profile_path.write_text(json.dumps(support.core().PROFILE))
        for name in selected.required_helpers(True):
            self.fixture.helpers["files"]["10_jax_exploration_platform/scripts/atari_pixel_campaign/" + name] = support.sha((SCRIPTS / name).read_bytes())

    def arguments(self):
        """Keep a null UUID and explicit new profile while retaining every full learner/runtime input."""
        arguments = self.fixture.arguments()
        arguments.gpu_uuid, arguments.hardware_class = None, "jaguar03_rtx_a4500"
        arguments.configured_isolation_profile = self.profile_path
        arguments.target_qualification_record = self.fixture.run_dir / "actual_qualification.json"
        arguments.target_qualification_review = self.fixture.run_dir / "actual_review.json"
        return arguments

    def prepare(self, arguments):
        """Replace only prior actual qualification checking; no scheduler, process or GPU query is permitted."""
        def previous(plan, *unused):
            """Represent the separate prior qualification binding without inventing a job identity."""
            plan["target_qualification_inputs"] = {"synthetic_metadata_fixture": True}
        with patch.object(support, "prepare_qualification_inputs", side_effect=previous), \
                patch.object(support, "bounded_query") as query:
            directory = self.fixture.prepare(arguments)
            query.assert_not_called()
        return directory

    def test_actual_preparation_keeps_null_plan_and_complete_optional_union(self):
        """Prepare through the real entry and stage every optional dependency without resolving an identity."""
        directory = self.prepare(self.arguments())
        raw_plan = (directory / "measurement_plan.json").read_bytes()
        plan = json.loads(raw_plan)
        self.assertIsNone(plan["gpu_uuid"])
        self.assertEqual(plan["configured_isolation"]["physical_identity"], support.core().unresolved_identity())
        self.assertEqual(plan["configured_isolation"]["owned_inputs_path"], str(directory / "owned_allocation_inputs.json"))
        self.assertFalse((directory / "owned_allocation_inputs.json").exists())
        self.assertEqual((directory / "measurement_prepared.json").read_text(), selected.json_bytes({"plan_sha256": support.sha(raw_plan)}).decode())
        for name in selected.required_helpers(True):
            self.assertEqual((directory / "code" / name).read_bytes(), (SCRIPTS / name).read_bytes())

    def test_wrong_context_or_instrumented_modes_refuse_before_output(self):
        """This single-card profile cannot enter packing, final learning, H100 or memory instrumentation."""
        for key, value in (("hardware_class", "h100"), ("gpu_uuid", GPU), ("mode", "packing-point"),
                ("mode", "learning-check"), ("final_confirmation", True), ("record_device_memory_limit", True)):
            arguments = self.arguments(); setattr(arguments, key, value)
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.prepare(arguments)
        self.assertFalse((self.fixture.run_dir / "data").exists())

    def test_unresolved_execution_collects_metadata_and_never_uses_old_resolver(self):
        """The real metadata execution branch leaves Slurm's original CUDA selector untouched."""
        arguments = self.arguments(); arguments.python = Path(sys.executable)
        directory = self.prepare(arguments)
        identity = support.core()
        identity.collect_metadata = unittest.mock.Mock(return_value={"synthetic_current_process": True})
        seen = []
        def compute(output, plan, hardware, backend):
            """Observe the numerical boundary without importing any numerical module."""
            seen.append(hardware)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0")
            self.assertIsNone(plan["gpu_uuid"])
            self.assertEqual(hardware["configured_isolation_metadata"], {"synthetic_current_process": True})
        with patch.object(selected, "__file__", str(directory / "code/measure_selected.py")), \
                patch.object(selected, "verify_prepared"), patch.object(support, "core", return_value=identity), \
                patch.object(support, "verify_qualification_inputs"), \
                patch.object(selected, "card_admission", return_value=selected.nullcontext()), \
                patch.object(selected, "preflight") as old, \
                patch.object(selected.training, "verify_assets", return_value={}), \
                patch.object(selected.training, "jaxtari_python_receipt", return_value={}), \
                patch.object(selected.training, "installed_gpu_library_receipts", return_value={}), \
                patch.object(selected, "installed_backend_receipt", return_value={}), \
                patch.object(selected, "_compute", side_effect=compute), \
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=True):
            selected.run_prepared(directory)
        old.assert_not_called()
        identity.collect_metadata.assert_called_once()
        self.assertEqual(len(seen), 1)

    def test_target_preparation_accepts_only_full_equal_original_and_distinct_result_kind(self):
        """A qualification prepares without a prior common baseline, study proof or target review."""
        arguments = self.arguments()
        arguments.mode, arguments.target_qualification = "correctness", True
        arguments.source_commit = "d28e6735e79788f618acb0ebfde8609db1f776af"
        arguments.configuration = arguments.reference_configuration
        arguments.target_qualification_record = arguments.target_qualification_review = None
        arguments.cleanrl_source_root = Path("/synthetic/cleanrl")
        arguments.qualification_software_result = Path("/synthetic/result.json")
        with patch.object(support, "qualification_sources") as sources:
            directory = self.fixture.prepare(arguments)
        sources.assert_called_once()
        plan = json.loads((directory / "measurement_plan.json").read_text())
        self.assertEqual(plan["target_qualification"], support.QUALIFICATION)
        self.assertNotIn("completed_study", plan)
        self.assertNotIn("target_qualification_inputs", plan)
        selected.save_measurement_result(directory, plan, {"completed": True})
        self.assertFalse((directory / "result.json").exists())
        result = json.loads((directory / "target_qualification_result.json").read_text())
        self.assertFalse(result["measurement_valid"])
        for name in ("scientific_credit", "case_credit", "idea_credit", "packing_point_credit", "throughput_credit"):
            self.assertEqual(result[name], 0)


if __name__ == "__main__":
    unittest.main()
