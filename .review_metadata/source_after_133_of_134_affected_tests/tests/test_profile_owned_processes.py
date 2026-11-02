"""Check the new profile process boundary with private metadata and mocked processes only."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from test_nsight_process_identity import M, ProcessObservationTests

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("owned_profile_metadata_test", SCRIPTS / "profile_selected.py")
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)


class SavedProcessTests(ProcessObservationTests):
    """Construct all source, actual-parent-wait and release artifacts in a private unit."""
    def saved(self):
        """Exercise the complete actual saved reader on a labelled synthetic process history."""
        unit = self.root / "data/unit"; (unit / "code").mkdir(parents=True)
        source = unit / "code/profile_selected.py"; source.write_text("# synthetic source only\n")
        plan = {"python": "/synthetic/python", "profile_tool": {"path": "/synthetic/nsys"}, "helpers": {"files": {
            "10_jax_exploration_platform/scripts/atari_pixel_campaign/profile_selected.py": M.receipt(source)["sha256"]}}}
        M.publish(unit / "measurement_plan.json", plan)
        waiter_command, capture_command = M.commands(unit, plan)
        for pid, command in ((102, waiter_command), (103, capture_command)):
            (self.root / str(pid) / "cmdline").write_bytes(b"\0".join(part.encode() for part in command) + b"\0")
        observer = M.OwnedProcesses(101, "20", self.parent, self.reader); self.addCleanup(observer.close)
        observed = observer.observe(103, "40", capture_command)
        intent = {"schema_version": 1, "protocol": M.PROTOCOL, "challenge": "a" * 64,
            "plan": M.receipt(unit / "measurement_plan.json"), "source": M.receipt(source), "parent": self.parent,
            "job_id": "123456789", "step_id": "0", "nsight_command": M.nsight_command(unit, plan),
            "waiter_command": waiter_command, "capture_command": capture_command}
        M.publish(unit / "profile_process_intent.json", intent)
        child = observed["nodes"][103]
        claim = {"schema_version": 1, "protocol": M.PROTOCOL, "pid": 103, "process_start_time": "40",
            "process": child, "challenge": intent["challenge"], "intent": M.receipt(unit / "profile_process_intent.json"),
            "source": intent["source"], "plan": intent["plan"]}
        M.publish(unit / "capture_identity.json", claim)
        M.publish(unit / "capture_process_observation.json", observed)
        M.publish(unit / "executed_command.json", intent["nsight_command"])
        owner = {"pid": 101, "process_start_time": "20", "parent_pid": 100,
            "parent_start_time": "10", "command": intent["nsight_command"]}
        M.publish(unit / "profiler_process.json", owner)
        launch = {"schema_version": 1, "challenge": intent["challenge"], "waiter": observed["nodes"][102],
            "child": child, "command": capture_command, "claim": M.receipt(unit / "capture_identity.json")}
        M.publish(unit / "capture_actual_launch.json", launch)
        waited = {"schema_version": 1, "challenge": intent["challenge"], "command": capture_command,
            "waited_pid": 103, "returncode": 0, "exited_normally": True, "interventions": [],
            "launch": M.receipt(unit / "capture_actual_launch.json")}
        M.publish(unit / "capture_actual_wait.json", waited)
        M.publish(unit / "capture_release.json", {**owner, "capture_identity": claim,
            "process_observation": M.receipt(unit / "capture_process_observation.json")})
        result = {"execution_identity": {"pid": 103, "process_start_time": "40"},
            "capture_process_protocol": M.PROTOCOL,
            "capture_process_observation": M.receipt(unit / "capture_process_observation.json"),
            "capture_actual_wait": M.receipt(unit / "capture_actual_wait.json")}
        return unit, plan, result

    def replace(self, path, change):
        """Change a private synthetic record, never a real execution artifact."""
        value = M.read_json(path); change(value); path.unlink(); M.publish(path, value)

    def test_full_saved_history_never_reads_historical_pids(self):
        """A valid actual-parent wait can be replayed after its process has ended."""
        unit, plan, result = self.saved()
        with patch.object(M, "ProcReader", side_effect=AssertionError("historical PID read")):
            rebuilt = M.validate_saved(unit, plan, result)
        self.assertEqual(rebuilt["capture_pid"], 103)

    def test_saved_intent_schema_challenge_and_step_are_strict(self):
        """A source hash alone cannot replace the predeclared one-attempt and numeric-step protocol."""
        for field, changed in (("schema_version", True), ("challenge", "old"), ("job_id", True), ("step_id", "batch")):
            unit, plan, unused = self.saved()
            intent = M.read_json(unit / "profile_process_intent.json"); intent[field] = changed
            with self.assertRaises(ValueError): M.validate_saved_intent(unit, plan, intent)
            shutil.rmtree(unit)

    def test_nsight_exit_does_not_replace_actual_child_wait(self):
        """A failed, signalled, intervened or different Python child never passes as a successful capture."""
        for field, changed in (("returncode", 7), ("returncode", True), ("waited_pid", 999),
                               ("exited_normally", False), ("interventions", ["timeout"])):
            unit, plan, result = self.saved()
            self.replace(unit / "capture_actual_wait.json", lambda value: value.update({field: changed}))
            result["capture_actual_wait"] = M.receipt(unit / "capture_actual_wait.json")
            with self.assertRaises(ValueError): M.validate_saved(unit, plan, result)
            shutil.rmtree(unit)

    def test_saved_numerical_result_must_match_the_actual_child(self):
        """Another result PID or an omitted observation cannot receive profile credit."""
        unit, plan, result = self.saved()
        result["execution_identity"]["pid"] = 101
        with self.assertRaisesRegex(ValueError, "another process"):
            M.validate_saved(unit, plan, result)


class ProfileProtocolTests(unittest.TestCase):
    """Keep the new configured protocol separate from historical and direct profile paths."""
    def test_new_helper_is_conditional_and_rejects_unsupported_paths(self):
        """Only explicitly new configured profiles receive the observation dependency."""
        for configured, direct, common in ((False, False, False), (True, False, False),
                                           (True, False, True), (False, True, False)):
            old = P.required_helpers(configured, direct, common)
            self.assertNotIn("nsight_process_identity.py", old)
            if configured and not direct:
                new = P.required_helpers(configured, direct, common, True)
                self.assertEqual(new, (*old, "nsight_process_identity.py"))
            else:
                with self.assertRaises(ValueError): P.required_helpers(configured, direct, common, True)

    def test_protocol_marker_refuses_wrong_hardware_or_direct_plan(self):
        """An unchanged direct/default plan cannot select the A4500 process protocol."""
        plan = {"capture_process_protocol": M.PROTOCOL, "configured_isolation": {}, "hardware_class": "jaguar03_rtx_a4500",
            "helpers": {"files": {"10_jax_exploration_platform/scripts/atari_pixel_campaign/nsight_process_identity.py": "a" * 64}}}
        self.assertTrue(P.owned_processes(plan)); self.assertFalse(P.owned_processes({}))
        for changed in ({"direct_binding": {}}, {"hardware_class": "h100"}, {"capture_process_protocol": "unknown"}):
            with self.assertRaises(ValueError): P.owned_processes({**plan, **changed})
        missing = copy.deepcopy(plan); missing.pop("capture_process_protocol")
        with self.assertRaisesRegex(ValueError, "lacks its explicit"): P.owned_processes(missing)
        missing = copy.deepcopy(plan); missing["helpers"]["files"].clear()
        with self.assertRaises(ValueError): P.owned_processes(missing)

    def test_child_observation_precedes_every_numerical_operation(self):
        """A denied release must not reach metadata-dependent CUDA/JAX initialization or the learner."""
        reader = Mock(); reader.read_json.return_value = {"parent": {}}
        reader.claim_self.side_effect = ValueError("no actual source child")
        with patch.object(P, "process_reader", return_value=reader), \
                patch.object(P.measurement, "producer_support") as metadata, patch.object(P, "_capture_core") as core:
            with self.assertRaisesRegex(ValueError, "actual source child"):
                P.capture_owned_child(Path("/private/synthetic"), {})
            metadata.assert_not_called(); core.assert_not_called()

    def test_capture_core_matches_preserved_original_ast(self):
        """The observer does not change numerical configuration, warmup, kernels or capture contents."""
        previous = SCRIPTS.parents[2] / ".review_metadata/source_after_first_11_process_tests/profile_selected.py"
        def selected(path):
            """Compare exact syntax for the complete numerical capture function."""
            return ast.dump(next(node for node in ast.parse(path.read_text()).body
                                 if isinstance(node, ast.FunctionDef) and node.name == "_capture_core"), include_attributes=False)
        self.assertEqual(selected(previous), selected(SCRIPTS / "profile_selected.py"))

    def test_waiter_records_the_actual_python_wait_status(self):
        """A successful Nsight process cannot change a failing actual Python return to zero."""
        for returned in (0, 7, -15):
            with tempfile.TemporaryDirectory() as temporary:
                unit = Path(temporary)
                (unit / "measurement_plan.json").write_text("{}")
                (unit / "capture_identity.json").write_text("{}")
                process = {"pid": 103, "parent_pid": 102}
                reader = Mock()
                reader.read_json.return_value = {}
                reader.validate_intent.return_value = {"parent": {"cgroup": "owned"}, "waiter_command": ["waiter"],
                    "capture_command": ["capture"], "challenge": "a" * 64}
                reader.ProcReader.return_value.snapshot.side_effect = [
                    {"pid": 102, "cgroup": "owned", "command": ["waiter"]}, process]
                reader.validate_claim.return_value = {"pid": 103, "process": process}
                reader.stable.side_effect = lambda value: value
                reader.parent_handle.return_value = 999999
                child = Mock(pid=103); child.poll.return_value = returned; child.wait.return_value = returned
                with patch.object(P, "verify_prepared"), patch.object(P, "process_reader", return_value=reader), \
                        patch.object(P.subprocess, "Popen", return_value=child), patch.object(P.os, "close") as closed:
                    status = P.run_owned_waiter(unit)
                child.wait.assert_called_once_with(); child.terminate.assert_not_called()
                waited = next(call.args[1] for call in reader.publish.call_args_list if call.args[0].name == "capture_actual_wait.json")
                self.assertEqual(waited["returncode"], returned); self.assertEqual(waited["waited_pid"], 103)
                self.assertEqual(waited["exited_normally"], returned >= 0)
                self.assertEqual(status, returned if returned >= 0 else 128 - returned)
                closed.assert_called_once_with(999999)

    def test_cold_new_helper_union_imports_without_site_or_numerical_modules(self):
        """Use an actual isolated metadata child from only the complete staged helper roster."""
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary) / "source/10_jax_exploration_platform/scripts/atari_pixel_campaign"
            stage.mkdir(parents=True)
            for name in P.required_helpers(True, False, True, True): shutil.copyfile(SCRIPTS / name, stage / name)
            code = ("import importlib.util,pathlib,sys; p=pathlib.Path(sys.argv[1]);sys.path.insert(0,str(p));"
                "s=importlib.util.spec_from_file_location('cold_profile',p/'profile_selected.py');"
                "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
                "assert not any(k.split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax'} for k in sys.modules);"
                "assert len(m.required_helpers(True,False,True,True))==23;print('metadata-only staged profile import passed')")
            returned = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", code, str(stage)],
                                      capture_output=True, text=True, timeout=30)
            self.assertEqual(returned.returncode, 0, returned.stderr)
            self.assertEqual(returned.stdout.strip(), "metadata-only staged profile import passed")


if __name__ == "__main__":
    unittest.main()
