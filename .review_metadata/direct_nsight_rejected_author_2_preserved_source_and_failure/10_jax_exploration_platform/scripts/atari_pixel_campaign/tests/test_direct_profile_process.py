"""Use temporary CPU locks and labelled proc fixtures; never run Nsight or touch the GPU lock."""
import ast
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import direct_profile_process as direct
import diagnose_direct_nsight_process as diagnostic
import profile_selected as profile
import direct_profile_prerequisites as prerequisites
import direct_backend_image as backend
from test_nsight_process_identity import M, ProcessObservationTests


class ActualTemporaryDescriptorTests(unittest.TestCase):
    """Check real CPU descriptor custody on a private temporary ordinary file."""
    def setUp(self):
        """Pin this test's temporary lock identity, never the actual serval02 lock."""
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "temporary_cpu_test_lock"
        self.path.write_text("temporary CPU descriptor fixture; no GPU operation\n")
        info = self.path.stat()
        self.identity = {"device": info.st_dev, "inode": info.st_ino, "uid": os.getuid(),
            "mode": stat.S_IMODE(info.st_mode), "regular_file": True}
        self.plan = {"helpers": {"files": {direct.PREFIX + "nsight_process_identity.py":
            hashlib.sha256((SCRIPTS / "nsight_process_identity.py").read_bytes()).hexdigest()}}}
        self.patches = (patch.object(direct, "LOCK", str(self.path)), patch.object(direct, "LOCK_IDENTITY", self.identity),
            patch.object(direct, "OWNER", os.getuid()))
        for item in self.patches:
            item.start(); self.addCleanup(item.stop)
        self.observer = direct.generic(self.plan)

    def test_real_temporary_owner_fdinfo_is_valid_only_while_exclusive_flock_is_held(self):
        """The production reader validates actual local kernel FLOCK bytes on a CPU fixture."""
        with self.path.open("r+") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            holder = self.observer.ProcReader().snapshot(os.getpid(), os.getuid())
            before = direct.descriptor_record(self.plan, stream.fileno(), holder, local=True)
            self.assertTrue(before["local_to_producer"])
            self.assertEqual(before["producer"]["pid"], os.getpid())
            with self.assertRaises(ValueError):direct.no_standing_descriptor()
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            with self.assertRaisesRegex(ValueError, "exclusive standing FLOCK"):
                direct.descriptor_record(self.plan, stream.fileno(), holder, local=True)

    def test_real_child_does_not_inherit_the_temporary_producer_descriptor(self):
        """The actual Popen close_fds rule keeps a parent's private lock out of its child."""
        with self.path.open("r+") as held:
            fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            script = ("import json,sys;sys.path.insert(0,sys.argv[1]);import direct_profile_process as p;"
                "p.LOCK_IDENTITY=json.loads(sys.argv[2]);print(json.dumps(p.no_standing_descriptor()))")
            child = subprocess.run(["/usr/bin/python3", "-I", "-S", "-B", "-c", script, str(SCRIPTS), json.dumps(self.identity)],
                capture_output=True, close_fds=True, pass_fds=(), timeout=30, check=False)
            self.assertEqual(child.returncode, 0, child.stderr.decode())
            self.assertEqual(json.loads(child.stdout)["standing_descriptor_count"], 0)
            holder = self.observer.ProcReader().snapshot(os.getpid(), os.getuid())
            direct.descriptor_record(self.plan, held.fileno(), holder, local=True)

    def test_real_readonly_producer_inspection_does_not_duplicate_its_descriptor(self):
        """Inspect only this test's known owned child and wait on its exact Popen return."""
        script = ("import fcntl,json,os,sys;f=open(sys.argv[1],'r+');fcntl.flock(f.fileno(),fcntl.LOCK_EX);"
            "print(json.dumps({'pid':os.getpid(),'fd':f.fileno()}),flush=True);sys.stdin.readline();f.close()")
        child = subprocess.Popen(["/usr/bin/python3", "-I", "-S", "-B", "-c", script, str(self.path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, close_fds=True, pass_fds=())
        try:
            claimed = json.loads(child.stdout.readline())
            self.assertEqual(claimed["pid"], child.pid)
            holder = self.observer.ProcReader().snapshot(child.pid, os.getuid())
            value = direct.descriptor_record(self.plan, claimed["fd"], holder, local=False)
            self.assertFalse(value["local_to_producer"])
            self.assertEqual(value["read_by_pid"], os.getpid())
            self.assertEqual(direct.no_standing_descriptor()["standing_descriptor_count"], 0)
            child.stdin.write("finish\n"); child.stdin.flush()
            self.assertEqual(child.wait(timeout=30), 0)
        finally:
            if child.poll() is None:
                child.terminate(); child.wait(timeout=30)
            child.stdin.close(); child.stdout.close(); child.stderr.close()


class DirectSavedProcessTests(ProcessObservationTests):
    """Build an explicit synthetic direct history with independent process groups and sessions."""
    def setUp(self):
        """Reuse only generic positive ancestry fixtures and mark every direct record synthetic."""
        super().setUp()
        self.original = direct.generic
        self.patches_direct = (patch.object(direct, "generic", return_value=M), patch.object(direct, "OWNER", 888),
            patch.object(direct, "LOCK_IDENTITY", {**direct.LOCK_IDENTITY, "uid": 888}))
        for item in self.patches_direct:
            item.start(); self.addCleanup(item.stop)

    def descriptor(self, *, reader=100, local=True):
        """Describe a labelled fixture FLOCK, never claim an actual GPU descriptor."""
        identifier = direct.LOCK_IDENTITY
        raw = ("pos:\t0\nflags:\t02100002\nlock:\t1: FLOCK ADVISORY WRITE 100 "
            + format(os.major(identifier["device"]), "x") + ":" + format(os.minor(identifier["device"]), "x")
            + ":" + str(identifier["inode"]) + " 0 EOF\n")
        return {"path": direct.LOCK, "fd": 7, "fd_stat": identifier, "path_stat": identifier,
            "path_symlink": False, "fd_target": direct.LOCK, "fdinfo": raw, "fdinfo_sha256": M.digest(raw.encode()),
            "producer": self.parent, "producer_before": self.parent, "producer_after": self.parent,
            "read_by_pid": reader, "local_to_producer": local}

    def saved(self):
        """Publish complete synthetic direct actual-wait records under one private test unit."""
        unit = self.root / "data/unit"; (unit / "code").mkdir(parents=True)
        for name in diagnostic.HELPERS:
            shutil.copyfile(SCRIPTS / name, unit / "code" / name)
        plan = {"kind": direct.DIAGNOSTIC, "gpu_uuid": direct.GPU, "hardware_class": "h100",
            "capture_process_protocol": direct.PROTOCOL, "campaign_run_dir": str(self.root), "unit_id": "unit",
            "python": "/synthetic/python", "profile_tool": {"path": "/synthetic/nsys"}, "maximum_active_seconds": 30,
            "helpers": {"files": {direct.PREFIX + name: M.receipt(SCRIPTS / name)["sha256"] for name in diagnostic.HELPERS}}}
        M.publish(unit / "measurement_plan.json", plan)
        waiter, capture = direct.commands(unit, plan)
        for pid, command in ((102, waiter), (103, capture)):
            (self.root / str(pid) / "cmdline").write_bytes(b"\0".join(part.encode() for part in command) + b"\0")
        owner = M.OwnedProcesses(101, "20", self.parent, self.reader); self.addCleanup(owner.close)
        observed = owner.observe(103, "40", capture)
        observed = json.loads(json.dumps(observed))
        intent = {"schema_version": 1, "protocol": direct.PROTOCOL, "challenge": "a" * 64,
            "hostname": "serval02", "original_environment": {"CUDA_VISIBLE_DEVICES": direct.GPU},
            "plan": M.receipt(unit / "measurement_plan.json"), "source": M.receipt(unit / "code" / diagnostic.NAME),
            "producer": self.parent, "producer_lock_before": self.descriptor(), "nsight_command": direct.tool_command(unit, plan),
            "waiter_command": waiter, "capture_command": capture, "standing_descriptor_passed_to_children": False,
            "metadata_controller_owns_standing_descriptor": False, "root_transport_owns_standing_descriptor": False}
        M.publish(unit / "direct_profile_process_intent.json", intent)
        child = observed["nodes"]["103"]
        claim = {"schema_version": 1, "protocol": direct.PROTOCOL, "challenge": intent["challenge"], "plan": intent["plan"],
            "source": intent["source"], "intent": M.receipt(unit / "direct_profile_process_intent.json"),
            "pid": 103, "process_start_time": "40", "process": child, "standing_descriptor_inherited": False,
            "no_standing_descriptor": {"pid": 103, "standing_descriptor_count": 0, "own_descriptors_only": True,
                "unrelated_descriptor_names_or_contents_retained": False},
            "observed_producer_lock": self.descriptor(reader=103, local=False)}
        M.publish(unit / "capture_identity.json", claim)
        M.publish(unit / "capture_process_observation.json", observed)
        launch = {"schema_version": 1, "protocol": direct.PROTOCOL, "challenge": intent["challenge"],
            "waiter": observed["nodes"]["102"], "child": child, "command": capture,
            "claim": M.receipt(unit / "capture_identity.json"), "close_fds": True, "pass_fds": [],
            "producer_lock_before_wait": self.descriptor(reader=102, local=False)}
        M.publish(unit / "capture_actual_launch.json", launch)
        M.publish(unit / "capture_actual_wait.json", {"schema_version": 1, "protocol": direct.PROTOCOL,
            "challenge": intent["challenge"], "command": capture, "waited_pid": 103, "returncode": 0,
            "exited_normally": True, "interventions": [], "launch": M.receipt(unit / "capture_actual_launch.json"),
            "producer_lock_after_wait": self.descriptor(reader=102, local=False), "standing_descriptor_inherited": False})
        tool = {"pid": 101, "process_start_time": "20", "parent_pid": 100, "parent_start_time": "10",
            "command": intent["nsight_command"], "close_fds": True, "pass_fds": []}
        M.publish(unit / "profiler_process.json", tool)
        M.publish(unit / "profiler_actual_wait.json", {"schema_version": 1, "protocol": direct.PROTOCOL,
            "command": intent["nsight_command"], "waited_pid": 101, "process_start_time": "20", "returncode": 0,
            "exited_normally": True, "interventions": [], "launch": M.receipt(unit / "profiler_process.json"),
            "producer_lock_after_wait": self.descriptor()})
        return unit, plan

    def change(self, path, key, value):
        """Mutate only an explicitly synthetic record for one refusal check."""
        actual = M.read_json(path); actual[key] = value; path.unlink(); M.publish(path, actual)

    def test_saved_direct_waits_accept_positive_edges_without_process_group_equality(self):
        """The complete reader replays after process exit without querying historical PIDs."""
        unit, plan = self.saved()
        with patch.object(M, "ProcReader", side_effect=AssertionError("historical PID query")):
            proof = direct.validate_saved(unit, plan)
        self.assertTrue(proof["passed"])
        self.assertFalse(proof["standing_descriptor_passed_to_children"])

    def test_actual_root_and_payload_waits_are_required_independently(self):
        """A successful Nsight primary never replaces a failed, signalled or different payload."""
        for filename in ("capture_actual_wait.json", "profiler_actual_wait.json"):
            for key, value in (("returncode", 7), ("returncode", True), ("waited_pid", 999), ("interventions", ["timeout"]),
                ("protocol", "configured")):
                unit, plan = self.saved()
                self.change(unit / filename, key, value)
                with self.subTest(filename=filename, key=key), self.assertRaises(ValueError):direct.validate_saved(unit, plan)
                shutil.rmtree(unit)

    def test_standing_descriptor_passage_and_controller_ownership_are_rejected(self):
        """Only the actual producer may hold the GPU descriptor, even in a synthetic fixture."""
        for key in ("standing_descriptor_passed_to_children", "metadata_controller_owns_standing_descriptor", "root_transport_owns_standing_descriptor"):
            unit, plan = self.saved()
            self.change(unit / "direct_profile_process_intent.json", key, True)
            with self.assertRaises(ValueError):direct.validate_saved(unit, plan)
            shutil.rmtree(unit)

    def test_changed_standing_inode_or_holder_and_wrong_claim_source_are_rejected(self):
        """The descriptor and source-child evidence remain exact after normal waits."""
        unit, plan = self.saved()
        value = self.descriptor()
        for key, wrong in (("fd_target", "/tmp/unrelated"), ("local_to_producer", "true"), ("fd", True),
            ("fdinfo_sha256", "0" * 64)):
            with self.subTest(key=key), self.assertRaises(ValueError):direct.validate_descriptor(plan, {**value, key: wrong}, self.parent)
        claim = M.read_json(unit / "capture_identity.json"); claim["standing_descriptor_inherited"] = True
        with self.assertRaises(ValueError):direct.validate_claim(unit, plan, M.read_json(unit / "direct_profile_process_intent.json"), claim)

    def run_waiter_fixture(self, returned, *, missing_claim=False):
        """Exercise the production waiter with private proc records and a mocked exact Popen child."""
        unit, plan = self.saved()
        for name in ("capture_actual_launch.json", "capture_actual_wait.json"):
            (unit / name).unlink()
        if missing_claim:
            (unit / "capture_identity.json").unlink()
        current = self.reader.snapshot(102, 888)
        handle = os.open(self.root / "private_cpu_parent_handle_fixture", os.O_RDONLY | os.O_CREAT, 0o600)
        child = SimpleNamespace(pid=103, poll=lambda: returned, wait=lambda timeout=None: returned,
            terminate=lambda: None, kill=lambda: None)
        absence = {"pid": 102, "standing_descriptor_count": 0, "own_descriptors_only": True,
            "unrelated_descriptor_names_or_contents_retained": False}
        with patch.object(direct, "context", return_value=current), patch.object(direct, "no_standing_descriptor", return_value=absence), \
            patch.object(direct, "producer_handle", return_value=handle), patch.object(M, "ProcReader", return_value=self.reader), \
            patch.object(direct.subprocess, "Popen", return_value=child) as popen, \
            patch.object(direct, "descriptor_record", return_value=self.descriptor(reader=102, local=False)):
            if missing_claim:
                with self.assertRaises(ValueError):direct.run_waiter(unit, plan)
            else:
                self.assertEqual(direct.run_waiter(unit, plan), returned)
            self.assertTrue(popen.call_args.kwargs["close_fds"])
            self.assertEqual(popen.call_args.kwargs["pass_fds"], ())
        return unit

    def test_actual_waiter_publishes_its_own_zero_and_failed_payload_returns(self):
        """The actual source child return remains distinct from the outer tool's status."""
        for returned in (0, 7):
            unit = self.run_waiter_fixture(returned)
            self.assertEqual(M.read_json(unit / "capture_actual_wait.json")["returncode"], returned)
            shutil.rmtree(unit)

    def test_early_payload_exit_preserves_actual_failure_wait_without_a_success_claim(self):
        """A child ending before its source claim still has a saved real Popen return."""
        unit = self.run_waiter_fixture(7, missing_claim=True)
        failed = M.read_json(unit / "capture_actual_wait_failure.json")
        self.assertEqual(failed["known_actual_popen_pid"], 103)
        self.assertEqual(failed["returncode"], 7)
        self.assertFalse(failed["successful_capture_claimed"])
        self.assertFalse((unit / "capture_actual_wait.json").exists())


class DirectDiagnosticSourceTests(unittest.TestCase):
    """Keep source roles, current admission and protected numerical implementation distinct."""
    def test_new_diagnostic_is_rejected_by_formal_profile_and_backend_contracts(self):
        """A zero-transition tool fixture cannot acquire ordinary measurement or profile credit."""
        plan = {"kind": direct.DIAGNOSTIC, "capture_process_protocol": direct.PROTOCOL}
        with self.assertRaises(ValueError):prerequisites.contract(plan)
        with self.assertRaises(ValueError):backend.plan_path(plan, Path("/unused"))
        self.assertEqual(diagnostic.ZERO["environment_transitions"], 0)
        self.assertFalse(diagnostic.ZERO["cuda_qualification_established"])
        self.assertFalse(diagnostic.ZERO["whole_tool_library_or_device_activity_quantified"])

    def test_all_new_tool_and_payload_launches_close_the_standing_descriptor(self):
        """The controller, producer and waiter pass no GPU descriptor to another process."""
        for module in (direct, diagnostic):
            tree = ast.parse(Path(module.__file__).read_text())
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "Popen"]
            self.assertTrue(calls)
            for call in calls:
                keywords = {item.arg: item.value for item in call.keywords}
                self.assertTrue(ast.literal_eval(keywords["close_fds"]))
                self.assertEqual(ast.literal_eval(keywords["pass_fds"]), ())
        tree = ast.parse(Path(diagnostic.__file__).read_text())
        controller = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "controller")
        self.assertFalse(any(isinstance(node, ast.Attribute) and node.attr in {"flock", "kill"} for node in ast.walk(controller)))

    def test_current_profile_admission_stays_closed_with_real_runner_source_present(self):
        """The new source route does not replace actual process and CUDA qualification."""
        with self.assertRaisesRegex(ValueError, "remain disabled"):prerequisites.require_admission()
        names = profile.required_helpers(False, True, direct_profile=True)
        self.assertEqual(len(names), 30)
        self.assertIn("direct_profile_process.py", names)
        self.assertNotIn("diagnose_direct_nsight_process.py", names)
        self.assertIn("direct-owned-capture-unit", Path(profile.__file__).read_text())

    def test_numerical_files_and_native_cuda_functions_remain_unchanged(self):
        """Only declared process/custody branches differ from the frozen disabled-source base."""
        root = SCRIPTS.parents[2]
        before = json.loads((root / ".review_metadata/direct_nsight_owned_producer_source_before.json").read_text())
        for path, expected in before["files"].items():
            self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), expected)
        tree = ast.parse((SCRIPTS / "profile_selected.py").read_text())
        core = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_capture_core")
        self.assertEqual(ast.dump(core, include_attributes=False), before["capture_core_ast"])
        old = subprocess.check_output(["git", "-C", str(root), "show", before["base"] + ":" + direct.PREFIX + "serval02_card1_binding.py"], text=True)
        original = {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(old).body if isinstance(node, ast.FunctionDef)}
        current = {node.name: ast.dump(node, include_attributes=False) for node in ast.parse((SCRIPTS / "serval02_card1_binding.py").read_text()).body if isinstance(node, ast.FunctionDef)}
        for name in ("validate_history", "snapshot_exit_history", "native_identity", "bind_process", "collect_metadata", "lock_metadata", "validate_lock"):
            self.assertEqual(current[name], original[name], name)


if __name__ == "__main__":
    unittest.main()
