"""Review only corrected root metadata custody and its exact one-shot transport."""
import argparse
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time
from unittest.mock import patch

P = Path(__file__).resolve().parent
R = P.parent.parent
UID = 2618919
PUB = "d3208df3e8588b29544835fc6e6d242c4be44153"
ORIGINAL_POPEN = subprocess.Popen
ORIGINAL_SLEEP = time.sleep
CHECKS = []


def require(condition, statement):
    """Record a required assertion and stop at the first failure."""
    assert condition, statement
    CHECKS.append(statement)


def sha(path):
    """Bind one ordinary small source or metadata file to its bytes."""
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "ordinary source or metadata: " + str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path):
    """Name one exact saved document."""
    return {"path": str(path), "sha256": sha(path)}


def load(name, path):
    """Import exact metadata source without invoking its launch entry."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def lock_is_held(path):
    """Positively observe the private metadata lock on a separately opened descriptor."""
    import fcntl
    with path.open("r+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
        return False


def record_failure_case(starter, control):
    """Fail the actual spawn-record writer and prove exact child cleanup precedes unlock."""
    with tempfile.TemporaryDirectory(prefix="record-failure-cpu-") as temporary:
        run = Path(temporary) / "run"
        packet = run / "remote_setup/cpu-controller"
        packet.mkdir(parents=True)
        logs = run / "slurm/direct_jobs" / packet.name
        logs.mkdir(parents=True)
        (run / "queue").mkdir()
        lock = run / "queue/metadata.lock"
        lock.write_text("")
        lock.chmod(0o600)
        stub = packet / "direct_qualification_controller.py"
        stub.write_text("import fcntl,pathlib\np=pathlib.Path(__file__).resolve().parent;r=p.parent.parent\n(p/'before_lock').write_text('actual CPU child')\nwith (r/'queue/metadata.lock').open('r+') as f:\n fcntl.flock(f,fcntl.LOCK_EX)\n (p/'after_lock').write_text('should never occur after record failure')\n")
        command = ["/usr/bin/env", "-i", "PATH=/usr/local/bin:/usr/bin:/bin", "LANG=C.UTF-8", "TZ=UTC", "CUDA_VISIBLE_DEVICES=GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3", "PYTHONDONTWRITEBYTECODE=1", "PYTHONNOUSERSITE=1", "/usr/bin/python3", "-I", "-S", "-B", str(stub), "--launch", "--publication-commit", PUB]
        (packet / "actual_launch_command_preparation.json").write_text(json.dumps({"host": "serval02", "automatic_retry_permitted": False, "publication": PUB, "command": command, "stdout": str(logs / "controller.stdout"), "stderr": str(logs / "controller.stderr")}))
        (packet / "root_remote_launch_intent.json").write_text("{}")
        original_write = starter.write
        spawned = []
        injected = []

        def track_popen(*args, **kwargs):
            """Retain only the real Popen child created by the frozen starter."""
            child = ORIGINAL_POPEN(*args, **kwargs)
            spawned.append(child)
            return child

        def fail_record(path, value):
            """Inject failure only after the real child is observed behind the held barrier."""
            if path.name == "root_actual_controller_spawn.json":
                deadline = time.monotonic() + 5
                while not (packet / "before_lock").exists() and time.monotonic() < deadline:
                    ORIGINAL_SLEEP(0.005)
                require((packet / "before_lock").exists() and lock_is_held(lock), "real child waits behind held metadata barrier before injected record failure")
                injected.append("actual root spawn writer failed")
                raise OSError("declared independent CPU fixture: spawn record failure")
            return original_write(path, value)

        observed_error = None
        try:
            with patch.object(starter, "P", packet), patch.object(starter, "R", run), patch.object(starter, "controller_metadata", return_value=control), patch.object(starter.socket, "gethostname", return_value="serval02"), patch.object(starter, "write", side_effect=fail_record), patch.object(starter.subprocess, "Popen", side_effect=track_popen), contextlib.redirect_stdout(io.StringIO()):
                try:
                    starter.main()
                except OSError as error:
                    observed_error = str(error)
            require(len(spawned) == 1 and injected == ["actual root spawn writer failed"] and observed_error is not None, "one actual CPU Popen and one declared record failure")
            child = spawned[0]
            require(type(child.returncode) is int and child.returncode < 0, "exact positively verified child was waited after owned termination")
            failure = json.loads((packet / "root_transport_failure.json").read_text())
            require(failure["ordinary_launch_success"] is False and failure["cleanup"]["kernel_child_exit_observed"] is True and failure["cleanup"]["actual_popen_pid"] == child.pid, "record failure retains explicit actual cleanup and no ordinary success")
            require(not (packet / "after_lock").exists() and not (packet / "root_actual_controller_spawn.json").exists() and not lock_is_held(lock), "failed record never lets child past barrier; unlock follows exact wait")
            return {"passed": True, "actual_cpu_child_pid": child.pid, "actual_wait_returncode": child.returncode, "cleanup": failure["cleanup"], "real_target_operation": False}
        finally:
            for child in spawned:
                if child.poll() is None:
                    identity = control.read_process(child.pid, UID)
                    control.wait_exact(child, identity, UID, 0)


def ordinary_error_case(starter, control):
    """Keep custody through ordinary poll/wait/report/print/sleep errors until actual child exit."""
    with tempfile.TemporaryDirectory(prefix="ordinary-custody-cpu-") as temporary:
        root = Path(temporary)
        lock = root / "metadata.lock"
        lock.write_text("")
        lock.chmod(0o600)
        child_script = root / "cpu_exit_after_flag.py"
        child_script.write_text("import pathlib,sys,time\np=pathlib.Path(sys.argv[1]);(p/'started').write_text(str(__import__('os').getpid()))\nwhile not (p/'allow_exit').exists():time.sleep(0.005)\n")
        child = ORIGINAL_POPEN(["/usr/bin/python3", "-I", "-S", "-B", str(child_script), str(root)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        identity = control.read_process(child.pid, UID)
        counters = {"poll_error": 0, "wait_error": 0, "report_error": 0, "print_error": 0, "sleep_error": 0, "wait_calls": 0}
        original_write = starter.write

        class ActualChildWithDeclaredErrors:
            """Keep the genuine Popen identity while injecting only declared ordinary API failures."""
            pid = child.pid

            def poll(self):
                """Inject the one initial cleanup poll failure while the actual child is alive."""
                require(lock_is_held(lock), "metadata barrier held at each custody poll")
                if counters["poll_error"] == 0:
                    require(child.poll() is None, "actual child remains alive at initial poll failure")
                    counters["poll_error"] += 1
                    raise OSError("declared independent CPU fixture: initial poll failure")
                return child.poll()

            def wait(self, *args, **kwargs):
                """Inject one wait failure, then permit this same child to exit while custody stays held."""
                require(lock_is_held(lock), "metadata barrier remains held at every custody wait")
                counters["wait_calls"] += 1
                if counters["wait_error"] == 0:
                    counters["wait_error"] += 1
                    raise OSError("declared independent CPU fixture: wait failure")
                (root / "allow_exit").write_text("actual child may now exit without entering any launch scope")
                return child.wait(*args, **kwargs)

            def __getattr__(self, name):
                """Expose only the original actual Popen object's remaining attributes."""
                return getattr(child, name)

        def fail_report(path, value):
            """Exercise a real ordinary reporting failure without changing current packet files."""
            if counters["report_error"] == 0:
                counters["report_error"] += 1
                raise OSError("declared independent CPU fixture: custody report failure")
            return original_write(path, value)

        def fail_print(*args, **kwargs):
            """Exercise a disconnected-output error without allowing it to release custody."""
            counters["print_error"] += 1
            raise OSError("declared independent CPU fixture: custody output failure")

        def fail_sleep(seconds):
            """Exercise one ordinary error in the custody delay, then retain normal CPU delay."""
            if counters["sleep_error"] == 0:
                counters["sleep_error"] += 1
                raise OSError("declared independent CPU fixture: custody sleep failure")
            return ORIGINAL_SLEEP(min(seconds, 0.005))

        handlers_before = {number: signal.getsignal(number) for number in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)}
        try:
            with control.metadata_lock(lock, UID):
                with patch.object(starter, "P", root), patch.object(starter, "write", side_effect=fail_report), patch.object(starter, "print", side_effect=fail_print, create=True), patch.object(starter.time, "sleep", side_effect=fail_sleep), patch.object(control, "wait_exact", side_effect=AssertionError("unknown identity must never be signaled")):
                    result = starter.cleanup_under_barrier(control, ActualChildWithDeclaredErrors(), None, RuntimeError("declared unknown-identity CPU fixture"))
                require(lock_is_held(lock) and child.returncode == 0, "barrier retained until same actual child has completed its exact wait")
            require(result["kernel_child_exit_observed"] is True and result["actual_popen_pid"] == child.pid and type(result["returncode"]) is int and result["returncode"] == 0, "positive exact Popen exit and integer status precede release")
            require(counters["poll_error"] == counters["wait_error"] == counters["report_error"] == counters["sleep_error"] == 1 and counters["print_error"] >= 1 and counters["wait_calls"] >= 2, "all five declared ordinary error paths exercised")
            require(handlers_before == {number: signal.getsignal(number) for number in handlers_before} and not lock_is_held(lock), "termination handlers restored only after positive child exit and barrier release")
            return {"passed": True, "actual_cpu_child_pid": child.pid, "actual_cpu_child_birth": identity["process_start_time"], "actual_wait_returncode": child.returncode, "counters": counters, "result": result, "real_target_operation": False}
        finally:
            (root / "allow_exit").write_text("fixture cleanup permits only its own child to exit")
            child.wait(timeout=5)


# Verify only the new frozen transport and unchanged small source metadata.
parser = argparse.ArgumentParser()
parser.add_argument("--freeze", required=True)
parser.add_argument("--sha256", required=True)
options = parser.parse_args()
start = time.monotonic()
require(os.environ.get("SLURM_JOB_ID") == "6821437" and os.environ.get("SLURM_STEP_ID", "").isdigit() and os.getresuid() == (UID, UID, UID), "current owned CPU step and exact owner")
require((R / "slurm/submitted_jobids_uva.txt").read_text().splitlines().count("6821437") == 1, "only current owned monitor allocation")
freeze_path = Path(options.freeze)
require(sha(freeze_path) == options.sha256, "exact newly frozen corrected source")
freeze = json.loads(freeze_path.read_text())
for name, expected in freeze["files"].items():
    require(sha(name) == expected, "unchanged corrected frozen transport file: " + Path(name).name)
old42 = json.loads((P / "controller_source_freeze.json").read_text())
for name, expected in old42["files"].items():
    require(sha(name) == expected, "unchanged accepted 42-file controller packet: " + Path(name).name)
packet_freeze = json.loads((P / "packet_metadata_freeze.json").read_text())
for name, expected in packet_freeze["files"].items():
    require(sha(name) == expected, "unchanged prepared metadata/helper closure: " + Path(name).name)
require(ref(freeze["preserved_rejected_freeze"]["path"]) == freeze["preserved_rejected_freeze"], "rejected freeze preserved exactly")
require(ref(freeze["preserved_rejected_source"]["path"]) == freeze["preserved_rejected_source"], "rejected source preserved exactly")
prior_freeze = json.loads(Path(freeze["preserved_rejected_freeze"]["path"]).read_text())
old_source = Path(prior_freeze["accepted_old_transport"]["path"])
prior_functions = {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(Path(freeze["preserved_rejected_source"]["path"]).read_text()).body if isinstance(node, ast.FunctionDef)}
new_functions = {node.name: ast.dump(node, include_attributes=False) for node in ast.parse((P / "root_start_controller_once.py").read_text()).body if isinstance(node, ast.FunctionDef)}
require(set(prior_functions) == set(new_functions) and {name for name in new_functions if prior_functions[name] != new_functions[name]} == {"retain_custody", "cleanup_under_barrier"}, "only the two required custody functions changed")
old_ast = ast.parse(old_source.read_text())
new_ast = ast.parse((P / "root_start_controller_once.py").read_text())
old_write = next(node for node in old_ast.body if isinstance(node, ast.FunctionDef) and node.name == "write")
new_write = next(node for node in new_ast.body if isinstance(node, ast.FunctionDef) and node.name == "write")
require(ast.dump(old_write, include_attributes=False) == ast.dump(new_write, include_attributes=False), "original exclusive write/filefsync/directoryfsync function retained exactly")
proposal = json.loads((P / "corrected_root_transport_proposal.json").read_text())
expected_remote = "/usr/bin/python3 -I -S -B " + str(P / "root_start_controller_once.py")
expected_ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ControlMaster=no", "-o", "ConnectTimeout=15", "serval02", expected_remote]
require(proposal["ssh_argv"] == expected_ssh and proposal["remote_command"] == expected_remote and "ProxyCommand" not in " ".join(expected_ssh), "exact one normal-route SSH to new root starter")
require(proposal["controller_publication"] == PUB and proposal["transport_source_publication"] is None and proposal["automatic_retry_permitted"] is False, "real controller publication and unknown future transport publication without retries")
command = json.loads((P / "actual_launch_command_preparation.json").read_text())
require(command["command"][8:12] == ["/usr/bin/python3", "-I", "-S", "-B"] and command["producer_command"][0] == "/p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python", "system metadata Python and unchanged scientific Python")
require(command["actual_controller_pid"] is None and command["actual_numerical_child_pid"] is None and command["transport_executed"] is False, "no fabricated actual process or executed transport")
require(not any((P / name).exists() for name in ("root_actual_controller_spawn.json", "root_transport_popen_attempt.json", "root_transport_failure.json")), "real target transport remains unattempted")

# Run only the normal causal case and the two affected failure cases.
starter = load("independent_corrected_transport", P / "root_start_controller_once.py")
control = load("independent_unchanged_controller_metadata", P / "direct_qualification_controller.py")
author_case = load("independent_execution_of_frozen_causal_case", P / "test_root_transport_barrier.py")
normal_output = io.StringIO()
with contextlib.redirect_stdout(normal_output):
    author_case.main()
normal = json.loads(normal_output.getvalue())
require(normal["passed"] is True and normal["actual_wait_returncode"] == 0, "normal real-Popen causal write/lock order")
known_failure = record_failure_case(starter, control)
ordinary_errors = ordinary_error_case(starter, control)
for name, expected in freeze["files"].items():
    require(sha(name) == expected, "corrected frozen transport remains exact after CPU cases: " + Path(name).name)
require(not {"jax", "jaxlib", "numpy", "torch", "flax", "optax", "cupy", "scipy", "jaxtari", "exploration_platform"}.intersection(sys.modules), "no numerical modules imported")
review = {"schema_version": 1, "kind": "independent_corrected_root_transport_metadata_custody_review", "passed": True, "launch_approved_after_exact_publication": True, "reviewer": "/root/review_bound_h100_timing_launch", "source_freeze": ref(freeze_path), "starter": ref(P / "root_start_controller_once.py"), "actual_controller_publication": PUB, "prospective_ssh_argv": expected_ssh, "transport_publication": None, "actual_owned_cpu_job_id": os.environ["SLURM_JOB_ID"], "actual_cpu_step_id": os.environ["SLURM_STEP_ID"], "hostname": socket.gethostname(), "normal_case": normal, "known_record_failure_case": known_failure, "ordinary_error_retention_case": ordinary_errors, "checks": CHECKS, "elapsed_seconds": time.monotonic() - start, "controller_and_29_helper_source_changed": False, "large_payloads_or_numerical_gates_reexecuted": False, "requires_before_actual_launch": ["Publish exact transport source, freeze and this review.", "Root verifies each proposed SSH/source byte against the actual new published commit.", "Root protects only own metadata locks and restores same inode/mode/bytes after real registration.", "Root executes exactly one reviewed SSH; never retry this attempted packet.", "Only the numerical producer holds the standing GPU descriptor; copy two remains unbound until actual timing exit proof."], "actual_target_or_gpu_or_remote_action": False, "scientific_credit": 0, "worker_admission": False}
out = P / "independent_corrected_root_transport_retention_review.json"
with out.open("x") as handle:
    json.dump(review, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
print(json.dumps({"passed": True, "review": ref(out), "actual_step_id": os.environ["SLURM_STEP_ID"], "checks": len(CHECKS), "elapsed_seconds": review["elapsed_seconds"]}))
