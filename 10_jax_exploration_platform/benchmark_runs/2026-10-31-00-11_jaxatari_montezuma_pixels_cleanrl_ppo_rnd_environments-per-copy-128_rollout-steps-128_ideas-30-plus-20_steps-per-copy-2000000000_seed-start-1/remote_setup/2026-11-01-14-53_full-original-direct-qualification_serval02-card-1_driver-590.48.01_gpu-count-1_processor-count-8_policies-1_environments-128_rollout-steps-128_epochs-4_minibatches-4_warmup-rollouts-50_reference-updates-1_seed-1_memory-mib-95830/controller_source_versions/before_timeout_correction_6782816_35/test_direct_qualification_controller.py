"""Real CPU-process checks for the separate direct qualification controller.

Every child and ledger is under TemporaryDirectory. Host/profile labels used by
the protected launch-input reader are explicitly synthetic. No test contacts
serval02 or opens its standing lock, NVIDIA devices, or a numerical library.
"""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

HERE = Path(__file__).absolute().parent
spec = importlib.util.spec_from_file_location("direct_controller_under_test", HERE / "direct_qualification_controller.py")
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)
CORE_PATH = Path("/p/rlprojects/RND_worktrees/2026-11-01-14-53_full-original-direct-qualification-source_commit-44f7d99db22c01bfd8beefefc5f548f5517d5348/10_jax_exploration_platform/scripts/atari_pixel_campaign/serval02_card1_binding.py")
REAL_POPEN = subprocess.Popen

CHILD_SOURCE = r'''
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import ctypes

blocked = {"jax", "jaxlib", "numpy", "torch", "cupy", "flax", "optax", "scipy", "jaxtari", "exploration_platform"}
class NoNumericalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in blocked:
            raise AssertionError("CPU metadata fixture attempted a numerical import")
sys.meta_path.insert(0, NoNumericalImports())
def guard(event, args):
    if event.startswith("socket.") or event == "ctypes.dlsym" or (event == "ctypes.dlopen" and args[0] is not None):
        raise AssertionError("CPU metadata fixture attempted network/shared-library access")
    if event == "open":
        name = str(args[0])
        if name.startswith(("/dev/nvidia", "/dev/dri")) or name == "/tmp/sl5nw_serval02_card1.lock":
            raise AssertionError("CPU metadata fixture attempted a device or standing-lock open")
sys.addaudithook(guard)
assert not (blocked & {name.split(".", 1)[0] for name in sys.modules})
unit = Path(sys.argv[2])
fixture = json.loads((unit / "fixture.json").read_text())
plan = json.loads((unit / "measurement_plan.json").read_text())
spec = importlib.util.spec_from_file_location("private_cpu_launch_core", fixture["core_path"])
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
try:
    # There is deliberately no absent-only barrier or sleep before this real guard.
    launch = core.launch_inputs(plan)
    assert launch["process_row"]["pid"] == os.getpid()
    raw = Path("/proc/self/stat").read_text()
    assert launch["process_row"]["start_time"] == raw[raw.rfind(")") + 2:].split()[19]
    (unit / "synthetic_compute_allowed.json").write_text(json.dumps({
        "actual_pid": os.getpid(), "actual_affinity": sorted(os.sched_getaffinity(0)),
        "protected_launch_reader_passed": True, "numerical_work": False,
        "fixture_host_label_is_synthetic": True}))
except BaseException as error:
    (unit / "synthetic_guard_failure.json").write_text(json.dumps({"type": type(error).__name__, "error": str(error)}))
    raise SystemExit(13)
assert not (blocked & {name.split(".", 1)[0] for name in sys.modules})
time.sleep(fixture.get("after_guard_seconds", .05))
raise SystemExit(fixture.get("exit_code", 0))
'''

class ControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture_results = []

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="direct-controller-private-cpu-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "campaign"
        self.unit = self.run / "data" / "private-cpu-fixture"
        (self.unit / "code").mkdir(parents=True)
        (self.run / "queue").mkdir()
        (self.run / "processes.json").write_text("[]\n")
        self.plan = {
            "python": sys.executable,
            "campaign_run_dir": str(self.run), "unit_id": self.unit.name,
            "kind": "selected_configuration_measurement", "gpu_uuid": controller.GPU_UUID,
            "source_commit": controller.SOURCE_COMMIT, "source_root": str(self.root / "synthetic-source"),
            "helpers": {"source_commit": controller.HELPER_COMMIT},
            "direct_binding": {"profile": {"hostname": "serval02"},
                "current_launch_inputs_path": str(self.unit / "direct_current_launch.json")}}
        (self.unit / "measurement_plan.json").write_bytes(controller.encoded(self.plan))
        (self.unit / "code/measure_selected.py").write_text(CHILD_SOURCE)
        self.command = [sys.executable, str(self.unit / "code/measure_selected.py"), "--prepared-unit", str(self.unit)]
        (self.unit / "command.txt").write_text(__import__("shlex").join(self.command) + "\n")
        self.fixture = {"core_path": str(CORE_PATH), "after_guard_seconds": .05, "exit_code": 0}
        self.environment = {"PATH": "/usr/bin:/bin", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        self.original_affinity = sorted(os.sched_getaffinity(0))
        self.cpu_count = min(2, len(self.original_affinity))

    def run_fixture(self, **kwargs):
        (self.unit / "fixture.json").write_bytes(controller.encoded(self.fixture))
        result = controller.supervise(self.plan, self.command, self.unit, self.environment,
            expected_uid=os.getuid(), cpu_count=self.cpu_count, registration_hostname="serval02", **kwargs)
        retained = {"test": self.id(), "outcome": result,
            "actual_test_host": __import__("socket").gethostname(),
            "fixture_serval02_label_is_synthetic": True, "no_actual_gpu_qualification": True,
            "files": {path.name: self.fixture_file(path) for path in self.unit.glob("*.json")
                      if path.name not in {"measurement_plan.json", "fixture.json"}},
            "stdout": (self.unit / "direct_controller_stdout.raw").read_text() if (self.unit / "direct_controller_stdout.raw").exists() else None,
            "stderr": (self.unit / "direct_controller_stderr.raw").read_text() if (self.unit / "direct_controller_stderr.raw").exists() else None}
        type(self).fixture_results.append(retained)
        self.assertEqual(sorted(os.sched_getaffinity(0)), self.original_affinity)
        return result

    def fixture_file(self, path):
        raw = path.read_bytes()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"deliberately_non_json_raw_hex": raw.hex(), "sha256": hashlib.sha256(raw).hexdigest()}

    def test_real_spawn_exact_pid_argv_affinity_actual_wait_and_launch_guard(self):
        observed = []
        original_publish = controller.publish
        def capture(path, value):
            if Path(path).name == "direct_current_launch.json":
                self.assertTrue((self.unit / "direct_controller_spawn.json").is_file())
                self.assertTrue((self.unit / "direct_current_process_ledger_snapshot.json").is_file())
                ledger = json.loads((self.run / "processes.json").read_text())
                self.assertEqual(ledger, [value["process_row"]])
                observed.append("journal-and-ledger-before-complete-launch-publication")
            return original_publish(path, value)
        with mock.patch.object(controller, "publish", capture):
            result = self.run_fixture()
        self.assertTrue(result["passed"], type(self).fixture_results[-1])
        self.assertEqual(observed, ["journal-and-ledger-before-complete-launch-publication"])
        spawn = controller.document(self.unit / "direct_controller_spawn.json")
        wait = controller.document(self.unit / "direct_process_wait_return.json")
        terminal = controller.document(self.unit / "direct_terminal_process.json")
        marker = controller.document(self.unit / "synthetic_compute_allowed.json")
        self.assertEqual(spawn["popen_pid"], wait["pid"])
        self.assertEqual(wait["pid"], marker["actual_pid"])
        self.assertEqual(spawn["observed_child"]["command"], self.command)
        self.assertEqual(spawn["observed_child"]["affinity"], marker["actual_affinity"])
        self.assertEqual(len(marker["actual_affinity"]), self.cpu_count)
        self.assertEqual(wait["returncode"], 0)
        self.assertIs(wait["exited_normally"], True)
        self.assertEqual(wait["waited_pid"], wait["pid"])
        self.assertEqual(terminal["return_record"], controller.receipt(self.unit / "direct_process_wait_return.json"))
        metadata = {"after": {"identity": {"pid": wait["pid"], "hostname": "serval02",
            "process_start_time": wait["process_start_time"], "parent_pid": os.getpid()}}}
        direct_path = CORE_PATH.with_name("direct_measurement.py")
        spec = importlib.util.spec_from_file_location("private_direct_terminal_reader", direct_path)
        direct = importlib.util.module_from_spec(spec); spec.loader.exec_module(direct)
        self.assertEqual(direct.terminal_process(self.unit, metadata), terminal)

    def core(self):
        spec = importlib.util.spec_from_file_location("cpu_test_core", CORE_PATH)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module

    def test_missing_launch_record_is_rejected_before_synthetic_compute_marker(self):
        def deliberate_delay(child, identity):
            time.sleep(.6)
        result = self.run_fixture(after_spawn=deliberate_delay)
        self.assertFalse(result["passed"])
        self.assertFalse((self.unit / "synthetic_compute_allowed.json").exists())
        self.assertTrue((self.unit / "synthetic_guard_failure.json").is_file())
        wait = controller.document(self.unit / "direct_process_wait_return.json")
        self.assertEqual(wait["returncode"], 13)
        self.assertFalse((self.unit / "direct_terminal_process.json").exists())
        self.assertFalse((self.unit / "direct_current_launch.json").exists())

    def test_nonzero_ordinary_child_exit_retains_wait_and_never_terminal_success(self):
        self.fixture["exit_code"] = 7
        result = self.run_fixture()
        self.assertFalse(result["passed"])
        wait = controller.document(self.unit / "direct_process_wait_return.json")
        self.assertEqual(wait["returncode"], 7)
        self.assertIs(wait["exited_normally"], True)
        self.assertFalse((self.unit / "direct_terminal_process.json").exists())

    def test_own_timeout_signals_only_observed_child_and_retains_signal_exit(self):
        self.fixture["after_guard_seconds"] = 10
        real_kill = os.kill
        signaled = []
        def own_kill(pid, signum):
            spawn = controller.document(self.unit / "direct_controller_spawn.json")
            self.assertEqual(pid, spawn["popen_pid"])
            self.assertEqual(controller.read_process(pid, os.getuid())["process_start_time"],
                             spawn["observed_child"]["process_start_time"])
            signaled.append(pid)
            return real_kill(pid, signum)
        with mock.patch.object(controller.os, "kill", own_kill):
            result = self.run_fixture(seconds=.5)
        self.assertFalse(result["passed"])
        self.assertEqual(len(signaled), 1)
        wait = controller.document(self.unit / "direct_process_wait_return.json")
        self.assertEqual(wait["returncode"], -signal.SIGTERM)
        self.assertIs(wait["exited_normally"], False)
        self.assertFalse((self.unit / "direct_terminal_process.json").exists())

    def test_any_previous_intent_prevents_another_popen(self):
        (self.unit / "direct_controller_intent.json").write_text("{}")
        with mock.patch.object(controller.subprocess, "Popen", side_effect=AssertionError("no second spawn")):
            with self.assertRaisesRegex(ValueError, "earlier or partial"):
                self.run_fixture()
        self.assertFalse((self.unit / "direct_controller_stdout.raw").exists())

    def test_partial_launch_prepared_file_prevents_any_spawn(self):
        (self.unit / "direct_current_launch.json.prepared").write_text("{")
        with mock.patch.object(controller.subprocess, "Popen", side_effect=AssertionError("no partial retry")):
            with self.assertRaisesRegex(ValueError, "earlier or partial"):
                self.run_fixture()

    def test_popen_failure_keeps_attempt_failure_and_restores_affinity(self):
        with mock.patch.object(controller.subprocess, "Popen", side_effect=OSError("private synthetic spawn failure")):
            result = self.run_fixture()
        self.assertFalse(result["passed"])
        self.assertFalse(result["actual_child_started"])
        self.assertFalse((self.unit / "direct_process_wait_return.json").exists())
        self.assertTrue((self.unit / "direct_controller_intent.json").exists())

    def test_incomplete_actual_child_identity_still_gets_true_wait_without_terminal(self):
        original = controller.read_process
        def identity(pid, uid):
            if pid != os.getpid():
                raise ValueError("synthetic missing kernel observation")
            return original(pid, uid)
        with mock.patch.object(controller, "read_process", identity):
            result = self.run_fixture()
        self.assertFalse(result["passed"])
        raw = controller.document(self.unit / "direct_controller_unidentified_wait.json")
        self.assertIsInstance(raw["pid"], int)
        self.assertEqual(raw["returncode"], 13)
        self.assertFalse((self.unit / "direct_current_launch.json").exists())
        self.assertFalse((self.unit / "direct_terminal_process.json").exists())

    def test_changed_observed_child_argv_never_publishes_launch(self):
        original = controller.read_process
        def identity(pid, uid):
            result = original(pid, uid)
            if pid != os.getpid():
                result["command"] = ["changed-synthetic-command"]
            return result
        with mock.patch.object(controller, "read_process", identity):
            result = self.run_fixture()
        self.assertFalse(result["passed"])
        self.assertFalse((self.unit / "direct_current_launch.json").exists())
        self.assertTrue((self.unit / "direct_process_wait_return.json").exists())

    def test_late_launch_collision_preserves_existing_bytes_and_no_terminal(self):
        def collision(child, identity):
            (self.unit / "direct_current_launch.json").write_text("foreign synthetic bytes")
        result = self.run_fixture(after_spawn=collision)
        self.assertFalse(result["passed"])
        self.assertEqual((self.unit / "direct_current_launch.json").read_text(), "foreign synthetic bytes")
        self.assertFalse((self.unit / "direct_terminal_process.json").exists())

    def test_observed_cpu_selection_has_no_guessed_or_boolean_indices(self):
        self.assertEqual(controller.chosen_affinity([21, 8, 40, 33, 2, 9, 16, 50, 60], 8), [2, 8, 9, 16, 21, 33, 40, 50])
        for available in ([0, 1], [True] + list(range(1, 9)), list(range(7)) + [6]):
            with self.assertRaises(ValueError):
                controller.chosen_affinity(available, 8)

    def test_exact_safe_environment_removes_slurm_cache_and_python_injection(self):
        plan = {"runtime_environment": {"JAX_ENABLE_X64": "true", "JAX_DEFAULT_MATMUL_PRECISION": "highest",
            "XLA_FLAGS": "--xla_gpu_exclude_nondeterministic_ops=true"}}
        value = controller.child_environment(plan, {"SLURM_JOB_ID": "123", "JAX_COMPILATION_CACHE_DIR": "/old",
            "XLA_FLAGS": "wrong", "PYTHONPATH": "/bad", "LD_PRELOAD": "/bad.so", "SYNTHETIC_SECRET": "must-not-save",
            "LD_LIBRARY_PATH": "/bad-libraries", "HOME": "/private-fixture"})
        self.assertFalse(any(key.startswith("SLURM_") for key in value))
        for key in ("JAX_COMPILATION_CACHE_DIR", "PYTHONPATH", "LD_PRELOAD", "SYNTHETIC_SECRET"):
            self.assertNotIn(key, value)
        self.assertEqual(value["LD_LIBRARY_PATH"], "/usr/lib/nvidia")
        self.assertEqual(value["CUDA_VISIBLE_DEVICES"], controller.GPU_UUID)
        self.assertEqual(value["OMP_NUM_THREADS"], "8")
        self.assertEqual(value["JAX_DEFAULT_MATMUL_PRECISION"], "highest")
        self.assertEqual(value["JAX_ENABLE_X64"], "true")

    def test_ordinary_reader_rejects_symlink_and_fifo_before_opening_for_read(self):
        source = self.root / "ordinary"; source.write_text("safe")
        alias = self.root / "alias"; alias.symlink_to(source)
        with self.assertRaises((ValueError, OSError)):
            controller.ordinary_bytes(alias)
        fifo = self.root / "fifo"; os.mkfifo(fifo)
        with self.assertRaisesRegex(ValueError, "regular file"):
            controller.ordinary_bytes(fifo)
        self.assertEqual(controller.ordinary_bytes(source), b"safe")

    def test_prepublication_metadata_launch_refuses_without_any_popen(self):
        with mock.patch.object(controller, "checked_inputs", return_value=(self.plan, self.command, {})), \
             mock.patch.object(controller.subprocess, "Popen", side_effect=AssertionError("no unreviewed action")):
            with self.assertRaises((ValueError, FileNotFoundError)):
                controller.actual_launch(None)

    def test_publication_freeze_must_include_the_running_controller_and_test(self):
        (self.root / "controller_source_freeze.json").write_bytes(controller.encoded({
            "schema_version": 1, "kind": "direct_qualification_controller_source_freeze", "files": {}}))
        with mock.patch.object(controller, "PACKET", self.root), \
             mock.patch.object(controller.subprocess, "check_output", side_effect=AssertionError("no incomplete-publication query")):
            with self.assertRaisesRegex(ValueError, "actual controller and regression source"):
                controller.check_publication("a" * 40)

    def test_output_hashing_streams_a_log_larger_than_metadata_bound(self):
        path = self.root / "private-output.raw"
        block = b"private CPU metadata fixture\n" * 2048
        expected = hashlib.sha256()
        with path.open("xb") as stream:
            for index in range(320):
                stream.write(block)
                expected.update(block)
        result = controller.log_receipt(path)
        self.assertGreater(result["size_bytes"], 16 * 1024 * 1024)
        self.assertEqual(result["sha256"], expected.hexdigest())
        alias = self.root / "aliased-output.raw"; alias.symlink_to(path)
        with self.assertRaises((ValueError, OSError)):
            controller.log_receipt(alias)

    def test_actual_fixed_packet_default_is_read_only_without_commands(self):
        with mock.patch.object(controller.subprocess, "Popen", side_effect=AssertionError("default must not spawn")), \
             mock.patch.object(controller.subprocess, "check_output", side_effect=AssertionError("default must not query")):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(controller.main([]), 0)
        result = json.loads(output.getvalue())
        self.assertIs(result["action_performed"], False)
        self.assertIsNone(result["actual_child_pid"])
        self.assertFalse((controller.UNIT / "direct_controller_intent.json").exists())

    def test_frozen_source_order_proves_guard_before_numerical_compute_and_producer_lock_scope(self):
        tree = ast.parse((CORE_PATH.parent / "measure_selected.py").read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_prepared")
        selected_scope = next(node for node in ast.walk(function) if isinstance(node, ast.With)
            and any(isinstance(item.context_expr, ast.Call) and isinstance(item.context_expr.func, ast.Name)
                    and item.context_expr.func.id == "card_admission" for item in node.items))
        metadata = [node.lineno for node in ast.walk(selected_scope)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "collect_metadata"]
        compute = [node.lineno for node in ast.walk(selected_scope)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_compute"]
        self.assertTrue(metadata and compute and max(metadata) < min(compute))
        core_tree = ast.parse(CORE_PATH.read_text())
        collect = next(node for node in core_tree.body if isinstance(node, ast.FunctionDef) and node.name == "collect_metadata")
        launch = [node for node in ast.walk(collect)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "launch_inputs"]
        self.assertEqual(len(launch), 1)
        self.assertFalse(any(isinstance(node, (ast.Import, ast.ImportFrom)) and
            any(alias.name.split(".")[0] in controller.NUMERICAL_MODULES for alias in node.names)
            for node in ast.walk(collect)))

if __name__ == "__main__":
    unittest.main(verbosity=2)

