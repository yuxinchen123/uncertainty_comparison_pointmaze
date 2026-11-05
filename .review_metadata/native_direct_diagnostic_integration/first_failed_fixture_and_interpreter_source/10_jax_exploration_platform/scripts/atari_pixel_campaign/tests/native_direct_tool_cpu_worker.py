"""Exercise the real cleanup path with private CPU processes and a temporary lock only."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import time
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import direct_profile_process as direct
import diagnose_direct_nsight_process as diagnostic
import native_process_evidence as native_evidence


def run(root, mode):
    """Keep exact CPU root/payload evidence until cleanup proves real kernel closure."""
    root = Path(root)
    unit = root / "data/unit"; (unit / "code").mkdir(parents=True)
    for name in diagnostic.required_helpers(True):
        shutil.copyfile(SCRIPTS / name, unit / "code" / name)
    binary = sys.executable
    plan = {"kind": direct.DIAGNOSTIC, "gpu_uuid": direct.GPU, "hardware_class": "h100",
        "capture_process_protocol": direct.PROTOCOL, "campaign_run_dir": str(root), "unit_id": "unit",
        "python": binary, "profile_tool": {"path": binary}, "maximum_active_seconds": .25,
        "fixture_kind": "actual_temporary_cpu_startup_failure_not_a_tool_or_gpu_qualification",
        "helpers": {"files": {direct.PREFIX + name: hashlib.sha256((SCRIPTS / name).read_bytes()).hexdigest()
            for name in diagnostic.required_helpers(True)}}}
    plan["native_process_control"] = native_evidence.declaration(plan)
    (unit / "measurement_plan.json").write_text(json.dumps(plan))
    lock = root / "temporary_cpu_lock"
    lock.write_text("temporary CPU fixture; not the standing GPU lock\n")
    info = lock.stat()
    lock_identity = {"device": info.st_dev, "inode": info.st_ino, "uid": info.st_uid,
        "mode": stat.S_IMODE(info.st_mode), "regular_file": True}
    actual_file = root / "actual_cpu_processes.json"
    leaf = "import os,time;os.setsid();time.sleep(6)"
    middle = ("import json,os,pathlib,subprocess,sys;os.setsid();p=subprocess.Popen([sys.argv[1],'-I','-S','-B','-c',sys.argv[3]],"
        "close_fds=True,pass_fds=());pathlib.Path(sys.argv[2]).write_text(json.dumps({'middle_pid':os.getpid(),'payload_pid':p.pid}));os._exit(0)")
    if mode == "orphan_root_failure":
        script = ("import os,pathlib,subprocess,sys,time;p=subprocess.Popen([sys.argv[1],'-I','-S','-B','-c',sys.argv[3],"
            "sys.argv[1],sys.argv[2],sys.argv[4]],close_fds=True,pass_fds=());"
            "deadline=time.monotonic()+3\nwhile not pathlib.Path(sys.argv[2]).exists() and time.monotonic()<deadline:time.sleep(.001)\nos._exit(7)")
        command = [binary, "-I", "-S", "-B", "-c", script, binary, str(actual_file), middle, leaf]
    else:
        script = ("import json,os,pathlib,subprocess,sys,time;p=subprocess.Popen([sys.argv[1],'-I','-S','-B','-c',sys.argv[3]],"
            "close_fds=True,pass_fds=());pathlib.Path(sys.argv[2]).write_text(json.dumps({'root_pid':os.getpid(),'payload_pid':p.pid}));time.sleep(6)")
        command = [binary, "-I", "-S", "-B", "-c", script, binary, str(actual_file), leaf]
    observer = direct.generic(plan)
    owner = observer.ProcReader().snapshot(os.getpid(), os.getuid())
    runtime = native_evidence.bootstrap(unit, plan, "producer", observer, owner)
    observer = runtime.observer
    scopes, independent_handles, holder_checks = [], {}, []
    original_scope, original_descriptor = direct.tool_scope, direct.descriptor_record
    original_reader = observer.ProcReader
    class DelayedRootReader(original_reader):
        """Represent a real scheduling delay after fast tool-root exit, before full observation."""
        def snapshot(self, pid, uid, cgroup=None):
            """Keep actual kernel records; delay only the first non-producer root observation."""
            if mode == "orphan_root_failure" and pid != os.getpid():
                time.sleep(.15)
            return super().snapshot(pid, uid, cgroup)
    def recorded_scope(directory, actual_plan, descriptor, **keywords):
        """Keep the actual source scope available for an independent test pidfd."""
        scope = original_scope(directory, actual_plan, descriptor, **keywords)
        scope.grace_seconds = .2
        scopes.append(scope)
        return scope
    def held_descriptor(actual_plan, descriptor, holder, *, local):
        """Verify real temporary custody while separately retaining the exact CPU payload handle."""
        value = original_descriptor(actual_plan, descriptor, holder, local=local)
        with lock.open("r+") as second:
            try:
                fcntl.flock(second.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                holder_checks.append(True)
            else:
                holder_checks.append(False)
                fcntl.flock(second.fileno(), fcntl.LOCK_UN)
                raise AssertionError("the temporary lock was released during cleanup")
        if scopes and actual_file.exists():
            scope = scopes[0]
            scope.scan()
            pid = json.loads(actual_file.read_text())["payload_pid"]
            if pid in scope.handles and pid not in independent_handles:
                independent_handles[pid] = os.dup(scope.handles[pid])
        return value
    result = {"kind": "actual_cpu_temporary_lock_descendant_cleanup", "mode": mode,
        "nsight_executed": False, "standing_gpu_lock_opened": False, "gpu_operations": 0,
        "environment_transitions": 0, "scientific_credit": 0, "admission_granted": False}
    try:
        with lock.open("r+") as held, patch.object(direct, "LOCK", str(lock)), patch.object(direct, "LOCK_IDENTITY", lock_identity), \
            patch.object(direct, "context", return_value=owner), patch.object(direct, "tool_command", return_value=command), \
            patch.object(direct, "tool_scope", side_effect=recorded_scope), patch.object(direct, "descriptor_record", side_effect=held_descriptor), \
            patch.object(observer, "ProcReader", DelayedRootReader):
            fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                direct.run_tool(unit, plan, held.fileno(), observer=observer)
            except (ValueError, RuntimeError, ProcessLookupError) as error:
                result["expected_startup_failure"] = {"type": type(error).__name__, "message": str(error)}
            else:
                raise AssertionError("the deliberately incomplete CPU payload was accepted")
            assert scopes and scopes[0].closed
            assert actual_file.exists()
            actual = json.loads(actual_file.read_text())
            assert independent_handles and actual["payload_pid"] in independent_handles
            result["payload_exit_observed_by_independent_pidfd"] = observer.handle_exited(independent_handles[actual["payload_pid"]])
            assert result["payload_exit_observed_by_independent_pidfd"] is True
            with lock.open("r+") as second:
                try:
                    fcntl.flock(second.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    result["temporary_lock_retained_until_all_descendants_exited"] = True
                else:
                    raise AssertionError("the producer lock disappeared before the caller finished")
            terminal = observer.read_json(unit / "tool_descendant_scope_terminal.json")
            assert terminal["terminal"]["actual_echild"] is True
            assert terminal["terminal"]["fresh_own_child_lists"][0]["raw"].strip() == ""
            result["kernel_terminal"] = terminal["terminal"]
            result["actual_root_waits"] = terminal["actual_root_waits"]
            result["actual_processes"] = actual
            result["holder_checks"] = len(holder_checks)
        with lock.open("r+") as released:
            fcntl.flock(released.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            result["temporary_lock_reacquired_only_after_verified_descendant_exit"] = True
        result["native_provider"] = runtime.provider.evidence()
        assert result["native_provider"]["open_handle_count"] == 0
        result["native_process_control_only"] = True
        result["passed"] = True
    finally:
        for handle in independent_handles.values():
            os.close(handle)
        (root / "cpu_regression_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
