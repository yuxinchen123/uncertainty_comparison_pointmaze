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
CORE_PATH = Path("/p/rlprojects/RND_worktrees/atari-direct-backend-package-helpers-0a880984b4a0/10_jax_exploration_platform/scripts/atari_pixel_campaign/serval02_card1_binding.py")
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

    def test_timeout_with_handler_returning_zero_keeps_true_zero_and_refuses_terminal_success(self):
        self.fixture["after_guard_seconds"] = 10
        path = self.unit / "code/measure_selected.py"
        source = path.read_text().replace('time.sleep(fixture.get("after_guard_seconds", .05))',
            'import signal\nsignal.signal(signal.SIGTERM, lambda *unused: sys.exit(0))\n'
            'time.sleep(fixture.get("after_guard_seconds", .05))')
        path.write_text(source)
        result = self.run_fixture(seconds=.5)
        wait = controller.document(self.unit / "direct_process_wait_return.json")
        self.assertEqual(wait["returncode"], 0)
        self.assertIs(wait["exited_normally"], True)
        self.assertTrue(wait["timeouts"])
        self.assertEqual(wait["kind"], "owned_direct_process_wait_return_after_controller_intervention")
        self.assertFalse(result["passed"])
        self.assertEqual(result["failure"]["type"], "ControllerTimeoutOrWaitFailure")
        self.assertFalse((self.unit / "direct_terminal_process.json").exists())
        observation = controller.document(self.unit / "direct_controller_timeout_observation.json")
        self.assertEqual(observation["actual_returncode"], 0)
        self.assertIs(observation["ordinary_successful_terminal_proof_allowed"], False)

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

    def test_timed_out_wait_is_rejected_by_the_unchanged_terminal_reader(self):
        self.fixture["after_guard_seconds"] = 10
        path = self.unit / "code/measure_selected.py"
        path.write_text(path.read_text().replace('time.sleep(fixture.get("after_guard_seconds", .05))',
            'import signal\nsignal.signal(signal.SIGTERM, lambda *unused: sys.exit(0))\n'
            'time.sleep(fixture.get("after_guard_seconds", .05))'))
        result = self.run_fixture(seconds=.5)
        self.assertFalse(result["passed"])
        self.assertFalse((self.unit / "direct_terminal_process.json").exists())
        wait = controller.document(self.unit / "direct_process_wait_return.json")
        self.assertEqual(wait["returncode"], 0)
        forged = {"schema_version": 1, "kind": "direct_serval02_card1_terminal_process",
            "launch": controller.receipt(self.unit / "direct_current_launch.json"),
            "pid": wait["pid"], "hostname": "serval02", "process_start_time": wait["process_start_time"],
            "uid": os.getuid(), "return_record": controller.receipt(self.unit / "direct_process_wait_return.json")}
        (self.unit / "direct_terminal_process.json").write_bytes(controller.encoded(forged))
        spec = importlib.util.spec_from_file_location("private_unchanged_intervention_terminal_reader", CORE_PATH.with_name("direct_measurement.py"))
        direct = importlib.util.module_from_spec(spec); spec.loader.exec_module(direct)
        metadata = {"after": {"identity": {"pid": wait["pid"], "hostname": "serval02",
            "process_start_time": wait["process_start_time"], "parent_pid": os.getpid()}}}
        with self.assertRaisesRegex(ValueError, "did not return normally"):
            direct.terminal_process(self.unit, metadata)



class BackendInputTests(unittest.TestCase):
    """Check real prepared metadata only; never start or qualify the target GPU."""

    def setUp(self):
        """Read the actual new unit's small immutable plan, without numerical packages."""
        self.plan = controller.document(controller.UNIT / "measurement_plan.json")
        self.expected = controller.document(controller.UNIT / "direct_backend_image_expected.json")

    def changed(self):
        """Return a private metadata copy, never a changed real prepared file."""
        return json.loads(json.dumps(self.plan))

    def test_actual_prepared_schema_two_and_real_copies_pass_without_gpu_action(self):
        self.assertEqual(controller.check_backend_inputs(self.plan), self.plan["direct_backend_image"])

    def test_legacy_missing_schema_and_wrong_capability_refuse_before_file_reads(self):
        for mutation in (lambda p:p.pop("direct_backend_image"),
                lambda p:p["direct_backend_image"].update(schema_version=1),
                lambda p:p["direct_backend_image"].update(schema_version=True),
                lambda p:p["direct_backend_image"].update(capability="unsupported")):
            plan = self.changed(); mutation(plan)
            with self.subTest(plan=plan.get("direct_backend_image")), mock.patch.object(controller, "receipt", side_effect=AssertionError("must refuse first")):
                with self.assertRaises(ValueError):controller.check_backend_inputs(plan)

    def test_missing_new_helper_or_circular_scope_is_rejected(self):
        for key in ("direct_qualification", "target_qualification", "completed_study", "final_confirmation_protocol", "common_correctness_inputs", "device_memory_limit_protocol"):
            with self.subTest(key=key), self.assertRaises(ValueError):controller.check_backend_inputs({**self.plan, key: {}})
        plan = self.changed(); plan["helpers"]["files"].pop("10_jax_exploration_platform/scripts/atari_pixel_campaign/direct_backend_image.py")
        with self.assertRaises(ValueError):controller.check_backend_inputs(plan)

    def test_copied_and_original_pin_paths_hashes_and_bytes_are_required(self):
        for mutation in (lambda p:p["direct_backend_image"]["expected"].update(path="/unlisted/expected.json"),
                lambda p:p["direct_backend_image"]["expected"].update(sha256="0" * 64),
                lambda p:p["inputs"]["direct_backend_image_record"].update(path="/unlisted/prepared.json"),
                lambda p:p["inputs"]["direct_backend_image_record"].update(sha256="0" * 64)):
            plan = self.changed(); mutation(plan)
            with self.subTest(plan=plan["direct_backend_image"]), self.assertRaises(ValueError):controller.check_backend_inputs(plan)
        actual = controller.receipt
        def changed_receipt(path):
            value = actual(path)
            return {**value, "sha256": "0" * 64} if Path(path) == controller.UNIT / "direct_backend_image_expected.json" else value
        with mock.patch.object(controller, "receipt", side_effect=changed_receipt), self.assertRaises(ValueError):controller.check_backend_inputs(self.plan)

    def test_staged_producer_or_owning_package_version_change_is_rejected(self):
        plan = self.changed(); plan["helpers"]["files"]["10_jax_exploration_platform/scripts/atari_pixel_campaign/direct_backend_image.py"] = "0" * 64
        with self.assertRaises(ValueError):controller.check_backend_inputs(plan)
        actual = controller.document
        request = actual(controller.PACKET / "preparation_request.json")
        qualified = json.loads(json.dumps(actual(request["qualification_plan"]["path"])))
        owner = self.expected["distribution"]["name"]
        for key in list(qualified["qualification_software"]["packages"]):
            if controller.re.sub(r"[-_.]+", "-", key).lower() == controller.re.sub(r"[-_.]+", "-", owner).lower():
                qualified["qualification_software"]["packages"][key] = "unqualified-version"
        def changed_qualification(path):
            return qualified if str(path) == request["qualification_plan"]["path"] else actual(path)
        with mock.patch.object(controller, "document", side_effect=changed_qualification), self.assertRaises(ValueError):
            controller.check_backend_inputs(self.plan)

    def test_actual_prepared_image_and_metadata_only_observation_cannot_change(self):
        actual = controller.document
        for field, value in (("image", {**self.expected["image"], "size_bytes": 1}), ("gpu_operations", True), ("numerical_imports", ["jax"])):
            wrong = {**self.expected, field: value}
            def changed_document(path):
                return wrong if Path(path) in (controller.UNIT / "direct_backend_image_expected.json", controller.BACKEND_PREPARATION_RECORD) else actual(path)
            with self.subTest(field=field), mock.patch.object(controller, "document", side_effect=changed_document):
                with self.assertRaises(ValueError):controller.check_backend_inputs(self.plan)

    def test_new_backend_partial_attempt_markers_and_dangling_aliases_are_rejected(self):
        for name in ("direct_backend_capture_attempt.json", "direct_backend_capture_begin.json", "direct_backend_capture_failure.json", "direct_backend_capture.json", "direct_backend"):
            with self.subTest(name=name), tempfile.TemporaryDirectory(prefix="backend-controller-marker-") as temporary:
                folder = Path(temporary); (folder / name).symlink_to(folder / "absent")
                with self.assertRaises(ValueError):controller.check_unattempted(folder)


class SerialPredecessorTests(unittest.TestCase):
    """Use only private synthetic saved records; never inspect any actual process."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="direct-serial-private-metadata-")
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name) / "campaign"
        self.previous = self.run / "data/prior-unit"
        self.packet = self.run / "remote_setup/current-unit"
        self.old_packet = self.run / "remote_setup/prior-unit"
        self.previous.mkdir(parents=True); self.packet.mkdir(parents=True); self.old_packet.mkdir()
        self.patches = [mock.patch.object(controller, "RUN", self.run), mock.patch.object(controller, "PACKET", self.packet)]
        for patch in self.patches:
            patch.start(); self.addCleanup(patch.stop)
        self.identity = {"hostname": "serval02", "pid": 101, "process_start_time": "202"}
        self.metadata = {"after": {"identity": {**self.identity, "parent_pid": 303, "uid": controller.UID}}}
        self.plan = {"unit_id": self.previous.name, "campaign_run_dir": str(self.run),
            "source_commit": controller.SOURCE_COMMIT, "source_root": str(self.run / "synthetic-source"),
            "helpers": {"source_commit": controller.HELPER_COMMIT, "source_root": str(self.run / "synthetic-helpers")},
            "runtime_environment": {"JAX_ENABLE_X64": "true"},
            "asset_record": {"root": str(self.run / "synthetic-assets"), "manifest": str(self.run / "synthetic-assets.json"), "sha256": "a" * 64},
            "gpu_uuid": controller.GPU_UUID, "python": sys.executable}
        self.result = {"completed": True, "kind": "direct_serval02_card1_full_original_qualification",
            "execution_identity": self.identity, "all_original_gates_passed": True}
        self.spawn = {"actual_pid": 303, "actual_birth_counter": "404",
            "kind": "actual_controller_transport_popen_return", "host": "serval02",
            "uid": controller.UID, "automatic_retry_permitted": False}
        fields = ["S"] + ["0"] * 19; fields[19] = "404"
        owner = {"pid": 303, "stat": "303 (synthetic-metadata) " + " ".join(fields)}
        self.wait = {"schema_version": 1, "kind": "owned_direct_process_wait_return", **self.identity,
            "uid": controller.UID, "waited_pid": 101,
            "command": [sys.executable, str(self.previous / "code/measure_selected.py"), "--prepared-unit", str(self.previous)],
            "wait_semantics": "Popen.wait on this exact spawned child", "returncode": 0, "exited_normally": True,
            "timeouts": [], "waiter_before": owner, "waiter_after": owner, "finished_at_utc": "2026-11-02T00:00:00+00:00"}
        self.outcome = {"kind": "direct_qualification_controller_outcome", "passed": True,
            "actual_child_started": True, "actual_wait_saved": True, "failure": None,
            "controller_owns_gpu_standing_descriptor": False}
        self.exit = {"kind": "owned_prior_direct_process_exit_observation", "hostname": "serval02",
            "observer_pid": 505, "observer_uid": controller.UID, "all_original_processes_exited": True,
            "gpu_or_standing_lock_operations": 0, "at_utc": "2026-11-02T00:00:01+00:00",
            "observations": [
                {"role": "controller", "expected_pid": 303, "expected_birth": "404",
                    "observation": "absent", "original_process_exited": True},
                {"role": "numerical_child", "expected_pid": 101, "expected_birth": "202",
                    "observation": "absent", "original_process_exited": True}]}
        self.request = {"serial_predecessor_unit": str(self.previous), "serial_order": 1,
            "numerical_source_root": self.plan["source_root"], "helper_source_root": self.plan["helpers"]["source_root"],
            "python": self.plan["python"], "runtime_environment": self.plan["runtime_environment"],
            "asset_record": self.plan["asset_record"]}
        self.save()

    def put(self, path, value):
        path.write_bytes(controller.encoded(value))
        return controller.receipt(path)

    def save(self):
        """Publish coherent private hashes after each deliberate fixture mutation."""
        records = {
            "plan": self.put(self.previous / "measurement_plan.json", self.plan),
            "result": self.put(self.previous / "direct_qualification_result.json", self.result),
            "metadata": self.put(self.previous / "direct_process_metadata.json", self.metadata),
            "wait": self.put(self.previous / "direct_process_wait_return.json", self.wait),
            "outcome": self.put(self.previous / "direct_controller_outcome.json", self.outcome),
            "controller_spawn": self.put(self.old_packet / "root_actual_controller_spawn.json", self.spawn),
            "exit_observation": self.put(self.old_packet / "exit_observation.json", self.exit)}
        launch = self.put(self.previous / "direct_current_launch.json", {"private_synthetic_fixture": True})
        terminal = {"kind": "direct_serval02_card1_terminal_process", **self.identity, "uid": controller.UID,
            "return_record": records["wait"], "launch": launch}
        records["terminal"] = self.put(self.previous / "direct_terminal_process.json", terminal)
        self.request.update(qualification_plan=records["plan"], qualification_terminal=records["terminal"],
            qualification_controller_outcome=records["outcome"], direct_qualification_inputs={"result": records["result"]})
        self.inputs = {"schema_version": 1, "kind": "actual_owned_direct_serial_predecessor_exit_inputs",
            "predecessor_unit": str(self.previous), **records}
        self.reference = self.put(self.packet / "serial_predecessor_exit_inputs.json", self.inputs)

    def check(self):
        return controller.check_serial_predecessor_inputs(self.request, self.reference)

    def test_boolean_wait_schema_refuses(self):
        self.wait["schema_version"] = True; self.save()
        with self.assertRaises(ValueError): self.check()

    def test_boolean_operation_count_refuses(self):
        self.exit["gpu_or_standing_lock_operations"] = False; self.save()
        with self.assertRaises(ValueError): self.check()

    def test_changed_numerical_source_root_refuses(self):
        self.plan["source_root"] = str(self.run / "different-source"); self.save()
        with self.assertRaises(ValueError): self.check()

    def test_changed_helper_root_refuses(self):
        self.plan["helpers"]["source_root"] = str(self.run / "different-helpers"); self.save()
        with self.assertRaises(ValueError): self.check()

    def test_changed_python_refuses(self):
        self.plan["python"] = "/unqualified/python"; self.save()
        with self.assertRaises(ValueError): self.check()

    def test_changed_runtime_refuses(self):
        self.plan["runtime_environment"] = {"JAX_ENABLE_X64": "false"}; self.save()
        with self.assertRaises(ValueError): self.check()

    def test_changed_assets_refuses(self):
        self.plan["asset_record"] = {"root": "/unqualified/assets"}; self.save()
        with self.assertRaises(ValueError): self.check()

    def test_exact_private_saved_normal_wait_and_both_exits(self):
        self.assertEqual(self.check(), self.inputs)

    def test_controller_still_present(self):
        self.exit["observations"][0]["observation"] = "present_same_birth"
        self.exit["observations"][0]["original_process_exited"] = False
        self.save()
        with self.assertRaises(ValueError): self.check()

    def test_wrong_numerical_birth_coherently_rehashed(self):
        self.exit["observations"][1]["expected_birth"] = "999"; self.save()
        with self.assertRaises(ValueError): self.check()

    def test_wrong_controller_birth(self):
        self.spawn["actual_birth_counter"] = "999"; self.save()
        with self.assertRaises(ValueError): self.check()

    def test_intervened_wait_cannot_pass(self):
        self.wait["kind"] = "owned_direct_process_wait_return_after_controller_intervention"; self.save()
        with self.assertRaises(ValueError): self.check()

    def test_timeout_with_success_exit_refuses(self):
        self.wait["timeouts"] = [{"kind": "timeout"}]; self.save()
        with self.assertRaises(ValueError): self.check()

    def test_failed_controller_outcome(self):
        self.outcome["passed"] = False; self.save()
        with self.assertRaises(ValueError): self.check()

    def test_observation_before_normal_wait(self):
        self.exit["at_utc"] = "2026-11-01T23:59:59+00:00"; self.save()
        with self.assertRaises(ValueError): self.check()

    def test_duplicate_role_and_missing_child(self):
        self.exit["observations"][1] = dict(self.exit["observations"][0]); self.save()
        with self.assertRaises(ValueError): self.check()

    def test_foreign_observer_uid(self):
        self.exit["observer_uid"] = 1; self.save()
        with self.assertRaises(ValueError): self.check()

    def test_result_wrong_process(self):
        self.result["execution_identity"] = {**self.identity, "pid": 999}; self.save()
        with self.assertRaises(ValueError): self.check()

    def test_serial_proof_substitution(self):
        wrong = dict(self.reference); wrong["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            controller.check_serial_predecessor_inputs(self.request, wrong)

    def test_changed_saved_record_after_binding(self):
        (self.previous / "direct_process_wait_return.json").write_text("{}")
        with self.assertRaises(ValueError): self.check()

    def make_second_copy_fixture(self):
        """Describe the completed first-copy result as the second unit's synthetic predecessor."""
        original = controller.document(HERE / "preparation_request.json")
        reference = json.loads(json.dumps(original["reference_configuration"]))
        candidate = json.loads(json.dumps(original["configuration"]))
        reference["learner"]["n_copies"] = candidate["learner"]["n_copies"] = 2
        qualification = {"result": {"path": str(self.run / "data/synthetic-original/result.json"), "sha256": "a" * 64}}
        self.request.update(serial_order=2, reference_configuration=reference, configuration=candidate,
            direct_qualification_inputs=qualification)
        previous_reference = json.loads(json.dumps(reference)); previous_reference["learner"]["n_copies"] = 1
        previous_candidate = json.loads(json.dumps(candidate)); previous_candidate["learner"]["n_copies"] = 1
        self.plan.update(mode="correctness", copy_seeds=[1], reference_configuration=previous_reference,
            configuration=previous_candidate, direct_qualification_inputs=qualification)
        self.inputs["plan"] = self.put(self.previous / "measurement_plan.json", self.plan)
        self.result.update(correctness_attempted=True, correctness_passed=True,
            implementation_distinct=True, measurement_valid=False)
        self.inputs["result"] = self.put(self.previous / "result.json", self.result)
        self.reference = self.put(self.packet / "serial_predecessor_exit_inputs.json", self.inputs)

    def test_second_unit_requires_actual_successful_first_copy_and_exact_exit(self):
        self.make_second_copy_fixture()
        self.assertEqual(self.check(), self.inputs)

    def test_second_unit_refuses_different_first_copy_seed(self):
        self.make_second_copy_fixture()
        self.plan["copy_seeds"] = [2]
        self.inputs["plan"] = self.put(self.previous / "measurement_plan.json", self.plan)
        self.reference = self.put(self.packet / "serial_predecessor_exit_inputs.json", self.inputs)
        with self.assertRaises(ValueError): self.check()

    def test_other_predecessor_unit(self):
        self.request["serial_predecessor_unit"] = str(self.run / "data/other-unit")
        with self.assertRaises(ValueError): self.check()


if __name__ == "__main__":
    unittest.main(verbosity=2)

