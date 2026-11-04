"""Reject unsupported or incomplete single-card evidence without any GPU or numerical import."""

import ast
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import threading
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
        for kwargs in ({"returncode": 1}, {"stderr": b"private text"}, {"complete": False}, {"complete": 1}, {"error": "TimeoutError"}):
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

    def test_observer_requires_the_exact_saved_binding_before_any_query(self):
        """An isolated ordinary-file fixture checks immutable binding substitution separately from physical validation."""
        with tempfile.TemporaryDirectory() as temporary:
            unit = Path(temporary)
            binding = {"kind": "uva_current_process_resolved_device_binding", "process": {"pid": os.getpid()},
                "gpu_uuid": GPU, "native": {"gpu_uuid": GPU}}
            for name, value in (("measurement_plan.json", {}), ("configured_isolation_metadata.json", {}),
                                ("resolved_device_binding.json", binding)):
                (unit / name).write_text(json.dumps(value))
            identity = support.core()
            with patch.object(support, "core", return_value=identity), patch.object(identity, "validate_binding") as validate, \
                    patch.object(support, "bounded_query") as query:
                observer = support.MemoryObservation(None, binding, unit / "paired_memory")
                self.assertEqual(observer.uuid, GPU)
                validate.assert_called_once_with({}, {}, binding, unit)
                different = copy.deepcopy(binding)
                different["gpu_uuid"] = different["native"]["gpu_uuid"] = "GPU-ffffffff-ffff-ffff-ffff-ffffffffffff"
                with self.assertRaises(ValueError):
                    support.MemoryObservation(None, different, unit / "paired_memory")
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

    def failed_allocator(self, directory):
        """Isolate shutdown accounting without opening a device or constructing a real GPU binding."""
        value = support.MemoryObservation.__new__(support.MemoryObservation)
        value.uuid, value.pid, value.directory = GPU, PID, directory
        value.window_started, value.samples, value.errors = 1.5, [memory_window()["samples"][0]], []
        value.normal_stop, value.counter, value.stop = True, 0, threading.Event()
        value.thread = types.SimpleNamespace(is_alive=unittest.mock.Mock(side_effect=[True, False]), join=lambda timeout: None)
        value.sample = lambda: memory_window()["samples"][-1]
        value.device = types.SimpleNamespace(memory_stats=unittest.mock.Mock(side_effect=RuntimeError("synthetic allocator failure")))
        return value

    def test_allocator_failure_saves_raw_lifecycle_before_raising(self):
        """Missing allocator statistics cannot discard already captured raw boundaries and shutdown facts."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "paired_memory"
            value = self.failed_allocator(directory)
            with patch.object(support.time, "time", return_value=2.5), self.assertRaises(RuntimeError):
                value.finish()
            proof = json.loads((directory / "memory_window_0.json").read_text())
            self.assertIsNone(proof["allocator"])
            self.assertEqual(proof["allocator_error"]["type"], "RuntimeError")
            self.assertEqual(len(proof["samples"]), 2)
            self.assertIs(proof["joined"], True)

    def test_allocator_failure_does_not_replace_an_original_update_error(self):
        """The frozen round's finally block retains its original numerical failure and explicit unknown memory."""
        with tempfile.TemporaryDirectory() as temporary:
            value = self.failed_allocator(Path(temporary) / "paired_memory")
            saved = []
            with patch.object(support.time, "time", return_value=2.5), self.assertRaisesRegex(KeyError, "original update"):
                try:
                    raise KeyError("original update")
                finally:
                    saved.append(value.finish())
            self.assertIsNone(saved[0]["other_gpu_processes"])
            self.assertIsNone(saved[0]["sampled_process_peak_bytes"])
            self.assertIs(saved[0]["measurement_valid"], False)

    def test_every_invalid_successful_device_capture_is_retained_anonymously(self):
        """Wrong UUID, shape, UTF8 and unavailable counters preserve hashes but no identifying raw output."""
        rows = [b"private-other-device, 20470, 1, 20469\n", b"private-malformed\n",
            f"{GPU}, 20470, N/A, 20469\n".encode(), b"\xff\n"]
        for contents in rows:
            with self.subTest(contents=contents), tempfile.TemporaryDirectory() as temporary:
                value = support.MemoryObservation.__new__(support.MemoryObservation)
                value.uuid, value.pid, value.directory = GPU, os.getpid(), Path(temporary) / "paired_memory"
                birth = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]
                value.binding = {"process": {"process_start_time": birth}, "original_environment": {}}
                with patch.object(support, "bounded_query", return_value=raw(contents)), self.assertRaises(ValueError):
                    value.sample()
                paths = list(value.directory.glob("device_query_failure_*.json"))
                self.assertEqual(len(paths), 1)
                proof = json.loads(paths[0].read_text())
                self.assertEqual(proof["stdout_sha256"], support.sha(contents))
                self.assertNotIn("stdout", proof)
                self.assertNotIn("private-", json.dumps(proof))

    def test_device_counters_do_not_invent_a_reserved_memory_equality(self):
        """Keep bounded actual total/used/free values without pretending separately reserved bytes are measured."""
        with tempfile.TemporaryDirectory() as temporary:
            value = support.MemoryObservation.__new__(support.MemoryObservation)
            value.uuid, value.pid, value.directory = GPU, os.getpid(), Path(temporary) / "paired_memory"
            birth = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]
            value.binding = {"process": {"process_start_time": birth}, "original_environment": {}}
            with patch.object(support, "bounded_query", return_value=raw(f"{GPU}, 20470, 512, 19000\n".encode())), \
                    patch.object(support, "process_sample", return_value=own_sample(os.getpid())):
                sample = value.sample()
            self.assertEqual(sample["free_bytes"], 19000 * 2 ** 20)
            self.assertNotEqual(sample["total_bytes"], sample["free_bytes"] + sample["used_bytes"])


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

    def test_actual_numerical_module_files_must_match_the_frozen_source(self):
        """Synthetic module objects exercise source/namespace/hash checks without importing numerical code."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "10_jax_exploration_platform/src/exploration_platform/atari_pixels"
            folder.mkdir(parents=True)
            modules, files = {}, {}
            for name in ("config", "environment", "training"):
                path = folder / (name + ".py"); path.write_text("# synthetic source fixture\n")
                modules["exploration_platform.atari_pixels." + name] = types.SimpleNamespace(__file__=str(path))
                files[str(path.relative_to(root))] = {"sha256": support.sha(path.read_bytes())}
            plan = {"source_root": str(root), "core": {"files": files}}
            with patch.dict(sys.modules, modules):
                self.assertEqual(set(support.numerical_module_paths(plan)), set(modules))
                (folder / "training.py").write_text("# changed fixture\n")
                with self.assertRaises(ValueError):
                    support.numerical_module_paths(plan)

    def test_ordinary_equal_comparison_rejection_is_still_present(self):
        """The equality exception is restricted to the explicit target qualification branch."""
        from test_measure_selected import SelectedMeasurementTests
        fixture = SelectedMeasurementTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        arguments = fixture.arguments(); arguments.configuration = fixture.files["reference"]
        with self.assertRaisesRegex(ValueError, "distinct implemented setting"):
            fixture.prepare(arguments)
        self.assertFalse((fixture.run_dir / "data").exists())
        self.assertNotIn("configured_isolation", (SCRIPTS / "paired_benchmark.py").read_text())

    def test_fresh_isolated_process_resolves_only_complete_staged_helper_rosters(self):
        """Import real staged/default/combined closures with isolated Python and no ambient helper fallback."""
        rosters = {"default": selected.required_helpers(False), "profile_only": profile.required_helpers(True),
            "existing_memory_and_learning": selected.HELPERS, "combined": selected.required_helpers(True)}
        for label, names in rosters.items():
            with self.subTest(roster=label), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary) / "data/unit/code"
                directory.mkdir(parents=True)
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
            directory = Path(temporary) / "data/unit/code"
            directory.mkdir(parents=True)
            for name in names:
                if name != "uva_active_configuration_observation.py":
                    (directory / name).write_bytes((SCRIPTS / name).read_bytes())
            command = [sys.executable, "-I", "-B", "-c", ISOLATED_IMPORT_PROBE, str(directory),
                json.dumps([name for name in names if name != "uva_active_configuration_observation.py"])]
            completed = subprocess.run(command, text=True, capture_output=True, timeout=30)
            self.assertNotEqual(completed.returncode, 0)


class QualificationReaderTests(unittest.TestCase):
    """Rebuild saved semantic gates while replacing only unrelated actual source/allocation validation."""

    def setUp(self):
        """Create ordinary local files with complete synthetic test, renderer, library and memory evidence."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.unit = self.root / "data/unit"
        self.unit.mkdir(parents=True)
        self.identity = support.core()
        source = self.root / "source"
        test = source / "10_jax_exploration_platform/tests/atari_pixels/test_fixture.py"
        test.parent.mkdir(parents=True)
        test.write_text("def test_plain(): pass\n@pytest.mark.parametrize('value', [False, True])\ndef test_pair(value): pass\n")
        self.nodes = ["tests/atari_pixels/test_fixture.py::test_plain", "tests/atari_pixels/test_fixture.py::test_pair[False]",
                      "tests/atari_pixels/test_fixture.py::test_pair[True]"]
        self.tests = {"command": [sys.executable, "-m", "pytest", "--rootdir", str(source / "10_jax_exploration_platform"), "-q", str(source / "10_jax_exploration_platform/tests/atari_pixels"),
            str(source / "10_jax_exploration_platform/tests/test_atari_pixel_environment.py")], "returncode": 0,
            "collection_rootdir": str(source / "10_jax_exploration_platform"), "collected_nodeids": self.nodes,
            "deselected_nodeids": [], "collection_reports": [{"nodeid": "atari_pixels/test_fixture.py", "outcome": "passed"}],
            "reports": [{"nodeid": node, "phase": phase, "outcome": "passed", "duration_seconds": .01}
                for node in self.nodes for phase in ("setup", "call", "teardown")]}
        self.renderer = {"rooms": 33, "cases": 132, "synthetic_pixel_cases": 11,
            "checks": {"cache": "passed", "palette": "passed", "fused": "passed"}}
        self.binding = {"gpu_uuid": GPU, "process": {"pid": PID}, "native": {"driver_library": "/node/libcuda.so.580.159.04"}}
        self.backend = {"gpu_libraries": {}, "jaxtari_source": {"synthetic": True}, "packages": {"synthetic": "1"}}
        for name in ("libcudnn.so.9", "libcudart.so.12"):
            path = self.root / name; path.write_bytes(b"synthetic ordinary image bytes")
            self.backend["gpu_libraries"][str(path)] = {"sha256": support.sha(path.read_bytes()), "package": "synthetic", "version": "1"}
        constants_root = self.root / "jaxtari_source"
        constants = constants_root / "src/jaxtari/games/montezuma_revenge/core.py"
        constants.parent.mkdir(parents=True)
        constants.write_text("class MontezumaRevengeConstants:\n    MAX_ROOMS: int = struct.field(pytree_node=False, default=33)\n")
        self.backend["jaxtari_source"] = {"source_root": str(constants_root),
            "source_commit": "04ab9aec318c72162e2e53503a594a489d7e162b", "python_files": {
                "jaxtari/games/montezuma_revenge/core.py": {"sha256": support.sha(constants.read_bytes())}}}
        renderer_source = source / "10_jax_exploration_platform/scripts/atari_pixel_campaign/check_rendering.py"
        renderer_source.parent.mkdir(parents=True)
        renderer_source.write_text("def fixture_states(environment):\n    with jax.enable_x64(False):\n        room_ids = jnp.repeat(None, 4)\n"
            "def check_synthetic_pixels(baseline, candidate):\n    colors = jnp.asarray(" + repr([[0, 0, 0]] * 9) + ")\n"
            "    patterns = jnp.stack((first, second))\n")
        software = self.root / "reference_installed.json"
        software.write_text(json.dumps(self.backend))
        assets = self.root / "reference_assets.json"; assets.write_text('{"synthetic_assets": true}')
        self.plan = {"source_root": str(source), "source_commit": "d28e6735e79788f618acb0ebfde8609db1f776af", "hardware_class": "jaguar03_rtx_a4500",
            "runtime_environment": {}, "configured_isolation": {"profile": {}}, "asset_record": {}, "helpers": {},
            "target_qualification": support.QUALIFICATION, "python": sys.executable,
            "qualification_sources": {"files": {"10_jax_exploration_platform/tests/atari_pixels/test_fixture.py": support.sha(test.read_bytes()),
                "10_jax_exploration_platform/scripts/atari_pixel_campaign/check_rendering.py": support.sha(renderer_source.read_bytes())}},
            "qualification_software": {"installed_backend": self.identity.receipt(software), "asset_receipt": self.identity.receipt(assets)}}
        loaded = {"pid": PID, "platform": "cuda", "platform_version": "synthetic CUDA backend",
            "cudnnGetVersion": 90000, "cudaRuntimeGetVersion": 12000,
            "cudnn_library": str(self.root / "libcudnn.so.9"), "cuda_runtime_library": str(self.root / "libcudart.so.12"),
            "verified_package_libraries": self.backend["gpu_libraries"],
            "loaded_shared_libraries": [*self.backend["gpu_libraries"], self.binding["native"]["driver_library"]]}
        self.plan["renderer_contract"] = support.renderer_contract(self.plan, self.backend)
        self.result = {"kind": "uva_configured_isolation_full_original_qualification", "completed": True,
            "qualification_protocol": support.QUALIFICATION, "source_commit": self.plan["source_commit"],
            "hardware_class": self.plan["hardware_class"], "runtime_environment": {}, "configured_isolation": {"profile": {}},
            "asset_record": {}, "all_original_gates_passed": True, "resolved_device_binding": self.binding, "helpers": {},
            "reference_iteration": 1, "reference_optimizer_step": 16, "frozen_target_unchanged": True, "policy_parameters_changed": True,
            "executed_work": {"full_reference_warmup_transitions": 819200, "full_reference_training_transitions": 16384,
                "full_reference_updates": 1, "full_reference_optimizer_minibatch_updates": 16},
            "qualification_memory": support.validate_window(memory_window()), "original_test_reports": self.tests["reports"],
            "original_collected_test_nodeids": self.nodes, "renderer": self.renderer, "renderer_contract": self.plan["renderer_contract"], "dtypes": copy.deepcopy(support.EXPECTED_DTYPES),
            "finite_state_leaves": 100, "installed_backend": self.backend, "loaded_gpu_libraries": loaded}
        self.result.update(scientific_credit=0, case_credit=0, idea_credit=0, packing_point_credit=0,
            throughput_credit=0, site_qualification_credit=0, worker_admitted=False, admission_granted=False)

    def publish(self):
        """Rehash all changed synthetic files coherently so each negative reaches semantic validation."""
        documents = {"measurement_plan.json": self.plan, "configured_isolation_metadata.json": {"inputs": {"job_id": "123"}, "process": {"pid": PID}},
            "resolved_device_binding.json": self.binding, "original_tests.json": self.tests, "original_renderer.json": self.renderer,
            "installed_backend.json": self.backend, "asset_receipt.json": {"synthetic_assets": True}}
        for name, value in documents.items():
            (self.unit / name).write_text(json.dumps(value))
        (self.unit / "original_tests.stdout").write_text(f"{len(self.tests['collected_nodeids'])} passed in 0.1s\n")
        (self.unit / "original_tests.stderr").write_text("")
        self.result["gate_artifacts"] = {name: self.identity.receipt(self.unit / name) for name in (
            "original_tests.json", "original_tests.stdout", "original_tests.stderr", "original_renderer.json")}
        path = self.unit / "target_qualification_result.json"; path.write_text(json.dumps(self.result))
        review = self.unit / "review.json"
        review.write_text(json.dumps({"schema_version": 1, "kind": "independent_full_original_target_qualification_review",
            "result": self.identity.receipt(path), "approved": True, "admission_granted": False, "scientific_credit": 0,
            "job_id": "123", "actual_terminal_state": "COMPLETED", "actual_exit_code": "0:0"}))
        return {**self.plan, "target_qualification_inputs": {"result": self.identity.receipt(path), "review": self.identity.receipt(review)}}

    def validate(self):
        """Run the real saved semantic reader while forbidding any fake physical/source fixture from claiming qualification."""
        outer = self.publish()
        with patch.object(support, "core", return_value=self.identity), patch.object(self.identity, "validate_binding"), \
                patch.object(support, "verify_saved_sources"), patch.object(support, "validate_original_qualification"), \
                patch.object(support, "verify_qualification_source_receipts"):
            return support.verify_qualification_inputs(outer)

    def test_complete_saved_semantics_pass_without_numerical_imports(self):
        """A complete isolated semantic fixture passes all newly reconstructed fields and exact source test cases."""
        self.assertEqual(self.validate(), self.result)
        self.assertEqual(sum(support.test_collection_contract(self.plan).values()), 3)

    def test_full_renderer_counts_derive_from_exact_source_and_reduced_matrices_refuse(self):
        """Require all33 indexed rooms,132 variants and11 synthetic cases declared by these pinned source fixtures."""
        contract = support.renderer_contract(self.plan, self.backend)
        self.assertEqual([contract[key] for key in ("rooms", "cases", "synthetic_pixel_cases")], [33, 132, 11])
        self.renderer.update(rooms=2, cases=8)
        with self.assertRaises(ValueError):
            self.validate()
        self.renderer.update(rooms=33, cases=132, synthetic_pixel_cases=1)
        with self.assertRaises(ValueError):
            self.validate()

    def test_renderer_backend_constants_cannot_change_after_their_recorded_hash(self):
        """The metadata derivation refuses a changed game constant rather than inferring a smaller valid matrix."""
        path = Path(self.backend["jaxtari_source"]["source_root"]) / "src/jaxtari/games/montezuma_revenge/core.py"
        path.write_text(path.read_text().replace("default=33", "default=2"))
        with self.assertRaises(ValueError):
            support.renderer_contract(self.plan, self.backend)

    def test_actual_pinned_backend_and_original_renderer_metadata_contract(self):
        """Exercise the real source with its nested x64 context without importing or running numerical code."""
        source = Path("/p/rlprojects/RND_worktrees/atari-cleanrl-source-d28e673")
        campaign = Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/"
            "2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1")
        backend_path = campaign / ("data/idea-position-16__fuse_network_bias_and_activation__hardware-jaguar03_rtx_a4500__copies-2__seed-start-1__updates-per-round-1__"
            "source-commit-d28e6735e79788f618acb0ebfde8609db1f776af/attempts/attempt-0004/installed_backend.json")
        backend = json.loads(backend_path.read_text())
        relative = "10_jax_exploration_platform/scripts/atari_pixel_campaign/check_rendering.py"
        plan = {"source_root": str(source), "source_commit": "d28e6735e79788f618acb0ebfde8609db1f776af",
            "qualification_sources": {"files": {relative: support.sha((source / relative).read_bytes())}}}
        matrix = support.renderer_contract(plan, backend)
        self.assertEqual([matrix[key] for key in ("rooms", "cases", "synthetic_pixel_cases")], [33, 132, 11])

    def test_renderer_scope_traversal_rejects_duplicates_and_ignores_nested_functions(self):
        """A nested function cannot supply or duplicate the current function's room matrix assignment."""
        relative = "10_jax_exploration_platform/scripts/atari_pixel_campaign/check_rendering.py"
        path = Path(self.plan["source_root"]) / relative
        original = path.read_text()
        changed = original.replace("    with jax.enable_x64(False):", "    def other():\n        room_ids = jnp.repeat(None, 2)\n    with jax.enable_x64(False):")
        path.write_text(changed)
        self.plan["qualification_sources"]["files"][relative] = support.sha(path.read_bytes())
        self.assertEqual(support.renderer_contract(self.plan, self.backend)["cases"], 132)
        path.write_text(changed.replace("        room_ids = jnp.repeat(None, 4)",
            "        room_ids = jnp.repeat(None, 4)\n        room_ids = jnp.repeat(None, 2)"))
        self.plan["qualification_sources"]["files"][relative] = support.sha(path.read_bytes())
        with self.assertRaises(ValueError):
            support.renderer_contract(self.plan, self.backend)
        path.write_text(changed.replace("        room_ids = jnp.repeat(None, 4)", "        pass"))
        self.plan["qualification_sources"]["files"][relative] = support.sha(path.read_bytes())
        with self.assertRaises(ValueError):
            support.renderer_contract(self.plan, self.backend)

    def test_wrong_dtype_or_nonpositive_finite_state_refuses(self):
        """A Boolean or bare successful gate cannot replace the exact six real dtype groups and finite leaves."""
        for field, value in (("dtypes", {"parameters": ["float16"]}), ("finite_state_leaves", 0), ("finite_state_leaves", True)):
            before = self.result[field]; self.result[field] = value
            with self.subTest(field=field), self.assertRaises((ValueError, KeyError)):
                self.validate()
            self.result[field] = before

    def test_setup_only_or_truncated_collection_refuses(self):
        """Every declared original case needs complete phases; coherently rehashed subsets remain insufficient."""
        reports = self.tests["reports"][:]
        self.tests["reports"][:] = [row for row in reports if row["phase"] == "setup"]
        with self.assertRaises(ValueError):
            self.validate()
        self.tests["reports"][:] = reports
        self.nodes.pop()
        self.tests["reports"][:] = [row for row in reports if row["nodeid"] in self.nodes]
        with self.assertRaises(ValueError):
            self.validate()

    def test_failed_renderer_and_changed_raw_result_agreement_refuse(self):
        """Failed full-room/synthetic checks cannot remain accepted after their result hashes are updated."""
        self.renderer["checks"]["cache"] = "failed"
        with self.assertRaises(ValueError):
            self.validate()
        self.renderer["checks"]["cache"] = "passed"
        self.renderer["cases"] = 7
        with self.assertRaises(ValueError):
            self.validate()

    def test_missing_libraries_wrong_library_process_and_backend_refuse(self):
        """The actual loaded image/API evidence must match both original package bytes and the resolved PID."""
        loaded = self.result.pop("loaded_gpu_libraries")
        with self.assertRaises((ValueError, KeyError)):
            self.validate()
        self.result["loaded_gpu_libraries"] = loaded
        loaded["pid"] = PID + 1
        with self.assertRaises(ValueError):
            self.validate()
        loaded["pid"] = PID
        self.backend["packages"]["synthetic"] = "changed"
        with self.assertRaises(ValueError):
            self.validate()

    def test_valid_other_uuid_memory_and_changed_summary_refuse(self):
        """An internally valid memory proof for another card cannot be joined to this physical qualification."""
        memory = self.result["qualification_memory"]
        proof = json.loads(json.dumps(memory["assigned_gpu_process_observation"]).replace(GPU, "GPU-ffffffff-ffff-ffff-ffff-ffffffffffff"))
        for sample in proof["samples"]:
            sample["device_query"]["sha256"] = support.sha(sample["device_query"]["stdout"].encode())
            sample["process_query"]["stdout_sha256"] = support.sha(sample["process_query"]["stdout"].encode())
        self.result["qualification_memory"] = support.validate_window(proof)
        with self.assertRaises(ValueError):
            self.validate()
        self.result["qualification_memory"] = memory
        memory["sampled_process_peak_bytes"] += 1
        with self.assertRaises(ValueError):
            self.validate()

    def test_boolean_controlled_counts_and_deselection_refuse(self):
        """Exact counters retain integer types and external test deselection is never accepted."""
        self.result["executed_work"]["full_reference_updates"] = True
        with self.assertRaises(ValueError):
            self.validate()

        self.result["executed_work"]["full_reference_updates"] = 1
        self.tests["deselected_nodeids"] = [self.nodes[0]]
        with self.assertRaises(ValueError):
            self.validate()

    def test_dynamic_collection_and_duplicate_phases_refuse(self):
        """Unsupported dynamic cases and repeated reports cannot replace the committed full test matrix."""
        test = Path(self.plan["source_root"]) / next(iter(self.plan["qualification_sources"]["files"]))
        original = test.read_text()
        test.write_text("@pytest.mark.parametrize('value', load_dynamic_cases())\ndef test_pair(value): pass\n")
        with self.assertRaises(ValueError):
            support.test_collection_contract(self.plan)
        test.write_text(original)
        self.tests["reports"].append(copy.deepcopy(self.tests["reports"][0]))
        with self.assertRaises(ValueError):
            self.validate()

    def test_read_only_source_receipts_do_not_apply_execution_environment_controls(self):
        """The historical reader validates the complete roster while leaving the current environment untouched."""
        plan = {"source_root": str(self.root), "source_commit": "a" * 40,
            "qualification_sources": {"files": {"a.py": "x", "b.py": "y"}},
            "cleanrl_reference": {"source_root": "/synthetic/original"},
            "qualification_software": {"result": {"path": "/r"}, "installed_backend": {"path": "/i"}, "asset_receipt": {"path": "/a"}}}
        site = types.SimpleNamespace(cleanrl_identity=unittest.mock.Mock(return_value=plan["cleanrl_reference"]),
            software_reference=unittest.mock.Mock(return_value=plan["qualification_software"]))
        with patch.object(support, "core", return_value=self.identity), patch.object(self.identity, "adjacent", return_value=site), \
                patch.object(support, "qualification_source_paths", return_value=["a.py", "b.py"]), \
                patch.object(selected.training, "source_receipt", return_value=plan["qualification_sources"]) as receipt, \
                patch.dict(os.environ, {"CLEANRL_REFERENCE_FILE": "current-process-value"}):
            before = dict(os.environ)
            support.verify_qualification_source_receipts(plan)
            self.assertEqual(before, dict(os.environ))
            receipt.assert_called_once_with(self.root, "a" * 40, ["a.py", "b.py"])


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


class ConfiguredProfileTests(unittest.TestCase):
    """Keep the supervisor and trace child's real producer paths separate before numerical imports."""

    def setUp(self):
        """Build complete profile metadata using the existing tested fixture and explicit new trust profile."""
        from test_profile_selected import SelectedProfileTests
        self.fixture = SelectedProfileTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.profile_path = self.fixture.run / "configured_profile.json"
        self.profile_path.write_text(json.dumps(support.core().PROFILE))
        self.fixture.confirmation.update(hardware_class="jaguar03_rtx_a4500", configured_isolation={})
        self.fixture.files["confirmation"].write_text(json.dumps(self.fixture.confirmation))
        self.fixture.documents["baseline"]["configured_isolation_profile"] = support.core().PROFILE
        self.fixture.files["baseline"].write_text(json.dumps(self.fixture.documents["baseline"]))
        for name in profile.required_helpers(True, False, False, True):
            self.fixture.helpers["files"]["10_jax_exploration_platform/scripts/atari_pixel_campaign/" + name] = support.sha((SCRIPTS / name).read_bytes())

    def prepared(self):
        """Stage the real new profile plan while replacing only already reviewed previous numerical evidence."""
        arguments = self.fixture.arguments()
        arguments.gpu_uuid, arguments.hardware_class = None, "jaguar03_rtx_a4500"
        arguments.configured_isolation_profile = self.profile_path
        arguments.target_qualification_record = self.fixture.run / "actual_target.json"
        arguments.target_qualification_review = self.fixture.run / "actual_review.json"
        arguments.python = Path(sys.executable)
        def previous(plan, *unused):
            """Mark a synthetic previous actual qualification only inside this metadata fixture."""
            plan["target_qualification_inputs"] = {"synthetic_metadata_fixture": True}
        with patch.object(support, "prepare_qualification_inputs", side_effect=previous), \
                patch.object(profile, "require_reviewed_profile_process_admission"), \
                patch.object(support, "verify_saved_measurement"), patch.object(support, "bounded_query") as query:
            directory = self.fixture.prepare(arguments)
            query.assert_not_called()
        return directory

    def test_profile_plan_null_identity_and_optional_dependencies_are_preserved(self):
        """Both supervisor and child use measurement_plan.json and the original unit launch-binding path."""
        directory = self.prepared()
        plan = json.loads((directory / "measurement_plan.json").read_text())
        self.assertIsNone(plan["gpu_uuid"])
        self.assertEqual(plan["configured_isolation"]["consumer"], "component_profile")
        self.assertEqual(plan["configured_isolation"]["owned_inputs_path"], str(directory / "owned_allocation_inputs.json"))
        self.assertFalse((directory / "profile_plan.json").exists())
        self.assertFalse((directory / "owned_allocation_inputs.json").exists())
        for name in profile.required_helpers(True):
            self.assertEqual((directory / "code" / name).read_bytes(), (SCRIPTS / name).read_bytes())

    def test_profile_parent_uses_reserved_supervisor_metadata_without_old_gpu_query(self):
        """The actual metadata-only supervisor branch records its own separate directory and leaves CUDA unchanged."""
        directory = self.prepared()
        plan = json.loads((directory / "measurement_plan.json").read_text())
        identity = support.core()
        identity.collect_metadata = unittest.mock.Mock(return_value={"synthetic_supervisor_process": True})
        with patch.object(profile, "verify_prepared"), patch.object(support, "core", return_value=identity), \
                patch.object(support, "verify_qualification_inputs"), \
                patch.object(profile.measurement, "preflight") as old, \
                patch.object(profile.measurement, "card_admission", return_value=selected.nullcontext()), \
                patch.object(profile, "tool_record", return_value=plan["profile_tool"]), \
                patch.object(profile, "validate_measured_baseline"), \
                patch.object(profile.training, "verify_assets", return_value={}), \
                patch.object(profile.training, "jaxtari_python_receipt", return_value={}), \
                patch.object(profile.training, "installed_gpu_library_receipts", return_value={}), \
                patch.object(profile, "capture_and_summarize") as capture, \
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=True):
            profile.run_prepared(directory)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0")
        identity.collect_metadata.assert_called_once_with(plan, directory / "supervisor")
        old.assert_not_called()
        capture.assert_called_once()
        self.assertFalse((directory / "resolved_device_binding.json").exists())

    def test_trace_child_collects_its_own_unit_metadata_without_parent_substitution(self):
        """The historical release format still collects the actual child's metadata, separately from the new protocol."""
        import socket
        directory = self.prepared()
        plan = json.loads((directory / "measurement_plan.json").read_text())
        plan.pop("capture_process_protocol")
        plan["helpers"]["files"].pop("10_jax_exploration_platform/scripts/atari_pixel_campaign/nsight_process_identity.py")
        (directory / "measurement_plan.json").write_text(json.dumps(plan))
        owner = {"parent_pid": 456, "parent_start_time": "4560", "pid": 123, "process_start_time": "1230"}
        child_identity = {"hostname": socket.gethostname().split(".")[0], "pid": os.getpid(),
            "process_start_time": str(os.getpid() * 10)}
        (directory / "profiler_process.json").write_text(json.dumps(owner))
        (directory / "capture_release.json").write_text(json.dumps({**owner, "capture_identity": child_identity}))
        (directory / "installed_backend.json").write_text("{}")
        identity = support.core()
        identity.collect_metadata = unittest.mock.Mock(return_value={"synthetic_actual_child_process": True})
        seen = []
        def compute(output, prepared_plan, hardware, installed):
            """Inspect the released child's exact numerical boundary without importing JAX."""
            seen.append(hardware["configured_isolation_metadata"])
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0")
        with patch.object(profile, "verify_prepared"), patch.object(support, "core", return_value=identity), \
                patch.object(profile, "process_birth", side_effect=lambda pid: str(pid * 10)), \
                patch.object(profile, "process_parent", return_value=456), patch.object(profile.os, "getpgrp", return_value=123), \
                patch.object(profile.measurement, "preflight") as old, patch.object(profile, "_capture_core", side_effect=compute), \
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=True):
            profile.capture(directory)
        identity.collect_metadata.assert_called_once_with(plan, directory)
        old.assert_not_called()
        self.assertEqual(seen, [{"synthetic_actual_child_process": True}])


if __name__ == "__main__":
    unittest.main()
