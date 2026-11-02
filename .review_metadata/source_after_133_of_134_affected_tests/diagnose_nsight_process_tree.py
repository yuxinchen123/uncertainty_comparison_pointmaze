"""Observe one actual Nsight process tree with standard-library payloads and zero scientific credit."""
import argparse
import ctypes
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import nsight_process_identity as identity

KIND = "nsight_process_identity_diagnostic"
NUMERICAL = {"jax", "jaxlib", "numpy", "torch", "flax", "optax", "cupy", "scipy", "jaxtari", "exploration_platform"}
BOUNDARY = {"scientific_credit": 0, "profile_credit": 0, "case_credit": 0, "idea_credit": 0,
            "admission_granted": False, "worker_admitted": False, "physical_gres_mapping_verified": False}


def metadata_guard():
    """Reject array-runtime imports and Python library/device work in every diagnostic role."""
    identity.require(not any(name.split(".")[0] in NUMERICAL for name in sys.modules),
                     "a numerical module was loaded before the diagnostic")
    class NoNumerical(importlib.abc.MetaPathFinder):
        """Keep this payload standard-library only even after release."""
        def find_spec(self, name, path=None, target=None):
            """Refuse any scientific runtime import."""
            if name.split(".")[0] in NUMERICAL:
                raise RuntimeError("numerical import forbidden in the Nsight process diagnostic")
    sys.meta_path.insert(0, NoNumerical())
    ctypes.CDLL = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Python shared-library loading is forbidden"))
    def audit(event, values):
        """Do not perform network calls or open a CUDA device from Python."""
        if event in {"socket.connect", "socket.bind", "socket.getaddrinfo", "os.system"}:
            raise RuntimeError("network or shell work forbidden in the process diagnostic")
        if event == "open" and isinstance(values[0], str) and values[0].startswith("/dev/nvidia"):
            raise RuntimeError("Python device opening forbidden in the process diagnostic")
    sys.addaudithook(audit)


def read_plan(directory):
    """Require the exact staged diagnostic source and its immutable original location."""
    directory = Path(directory).resolve(strict=True)
    plan = identity.read_json(directory / "measurement_plan.json")
    identity.require(plan["kind"] == KIND and plan["unit_id"] == directory.name
        and directory == Path(plan["campaign_run_dir"]) / "data" / plan["unit_id"]
        and identity.read_json(directory / "measurement_prepared.json") == {
            "plan_sha256": identity.receipt(directory / "measurement_plan.json")["sha256"]}
        and Path(__file__).resolve() == directory / "code/diagnose_nsight_process_tree.py"
        and str(Path(sys.executable).absolute()) == plan["python"], "execute only the exact staged process diagnostic")
    for name in ("diagnose_nsight_process_tree.py", "nsight_process_identity.py"):
        relative = "10_jax_exploration_platform/scripts/atari_pixel_campaign/" + name
        identity.require(identity.receipt(directory / "code" / name)["sha256"] == plan["helpers"]["files"][relative],
                         "a staged process-diagnostic helper changed")
    return directory, plan


def owned_launch(directory, plan):
    """Use only an actual released ID already present in this campaign's own ledgers."""
    value = identity.read_json(directory / "owned_launch_inputs.json")
    job = os.environ.get("SLURM_JOB_ID")
    identity.require(type(job) is str and job.isdigit() and value["job_id"] == job
        and value["plan"] == identity.receipt(directory / "measurement_plan.json")
        and job != "6783003" and os.environ.get("SLURM_STEP_ID", "").isdigit()
        and os.environ.get("SLURM_CPUS_PER_TASK") == "1"
        and os.environ.get("SLURM_MEM_PER_NODE") == "2048"
        and os.uname().nodename.split(".")[0] == "jaguar03"
        and len(os.environ.get("SLURM_STEP_GPUS", "").split(",")) == 1
        and os.environ.get("SLURM_STEP_GPUS")
        and len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) == 1
        and os.environ.get("CUDA_VISIBLE_DEVICES"),
        "the tool experiment requires its own one-GPU numeric allocation, never the CPU monitor")
    row = [job, KIND, plan["helpers"]["source_commit"], str(Path(plan["packet"]) / "job.sbatch")]
    identity.require(value["submission_row"] == row, "the exact diagnostic submission row changed")
    run = Path(plan["campaign_run_dir"])
    for path, member in ((run / "slurm/submitted_jobids.txt", job),
                         (run / "slurm/submitted_jobids_uva.txt", job),
                         (run / "slurm/submissions.tsv", "\t".join(row))):
        identity.require(path.read_text().splitlines().count(member) == 1, "the actual diagnostic is not uniquely ledger-owned")
    rows = [line.split("\t") for line in (run / "slurm/submissions.tsv").read_text().splitlines() if line.split("\t")[0] == job]
    identity.require(rows == [row], "another command is recorded for this actual diagnostic job")
    for name, kind in (("submission", "owned_nsight_process_diagnostic_submission"),
                       ("release", "owned_nsight_process_diagnostic_release")):
        record = value[name]
        identity.require(identity.receipt(record["path"]) == record, "an actual diagnostic launch receipt changed")
        actual = identity.read_json(record["path"])
        identity.require(actual["kind"] == kind and actual["job_id"] == job
            and type(actual["returncode"]) is int and actual["returncode"] == 0
            and actual["plan_sha256"] == value["plan"]["sha256"] and actual["data_unit"] == str(directory),
            "the diagnostic has no actual successful owned submission/release")
    return value


def wait_for_owned_launch(directory, plan):
    """Wait only for an absent final real-release sidecar, and reject any present invalid file."""
    path = Path(directory) / "owned_launch_inputs.json"
    started = time.monotonic()
    while not os.path.lexists(path):
        identity.require(time.monotonic() - started < 180, "the actual released diagnostic inputs did not arrive")
        time.sleep(.1)
    return owned_launch(directory, plan)


def probe(directory):
    """Save one real Python identity, wait without numerical work, then exit normally."""
    directory, plan = read_plan(directory)
    claim = identity.claim_self(directory, plan)
    intent = identity.read_json(directory / "profile_process_intent.json")
    parent = identity.parent_handle(intent)
    try:
        started = time.monotonic()
        while not (directory / "probe_finish.json").exists():
            if identity.handle_exited(parent) or time.monotonic() - started > 120:
                raise TimeoutError("the actual process observer disappeared or did not finish the probe")
            time.sleep(.05)
        release = identity.read_json(directory / "probe_finish.json")
        identity.require(release == {"challenge": intent["challenge"], "claim": identity.receipt(directory / "capture_identity.json")},
                         "the probe received another attempt's finish request")
        identity.publish(directory / "probe_result.json", {"completed": True, "pid": claim["pid"],
            "process_start_time": claim["process_start_time"], "numerical_modules": sorted(NUMERICAL.intersection(sys.modules)),
            "cuda_profiler_api_called": False, **BOUNDARY})
    finally:
        os.close(parent)


def waiter(directory):
    """Popen and wait on one exact standard-library probe child."""
    directory, plan = read_plan(directory)
    intent = identity.validate_intent(directory, plan, identity.read_json(directory / "profile_process_intent.json"))
    current = identity.ProcReader().snapshot(os.getpid(), os.getuid(), intent["parent"]["cgroup"])
    identity.require(current["command"] == intent["waiter_command"], "the actual probe waiter command changed")
    parent = identity.parent_handle(intent)
    child = subprocess.Popen(intent["capture_command"])
    try:
        started = time.monotonic()
        while not (directory / "capture_identity.json").exists():
            if child.poll() is not None or identity.handle_exited(parent) or time.monotonic() - started > 120:
                raise RuntimeError("the actual standard-library child did not publish its identity")
            time.sleep(.05)
        claim = identity.validate_claim(directory, plan, intent, identity.read_json(directory / "capture_identity.json"))
        actual = identity.ProcReader().snapshot(child.pid, os.getuid(), current["cgroup"])
        identity.require(child.pid == claim["pid"] and actual["parent_pid"] == current["pid"]
            and identity.stable(actual) == identity.stable(claim["process"]), "the actual probe Popen child changed")
        identity.publish(directory / "capture_actual_launch.json", {"schema_version": 1, "challenge": intent["challenge"],
            "waiter": current, "child": actual, "command": intent["capture_command"],
            "claim": identity.receipt(directory / "capture_identity.json")})
        while child.poll() is None:
            if identity.handle_exited(parent):
                raise RuntimeError("the original diagnostic observer disappeared")
            time.sleep(.05)
        returned = child.wait()
        identity.publish(directory / "capture_actual_wait.json", {"schema_version": 1, "challenge": intent["challenge"],
            "command": intent["capture_command"], "waited_pid": child.pid, "returncode": returned,
            "exited_normally": returned >= 0, "interventions": [],
            "launch": identity.receipt(directory / "capture_actual_launch.json")})
        return returned if returned >= 0 else 128 - returned
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()
        os.close(parent)


def run(directory):
    """Launch the exact real tool once and observe its actual owned process relationship."""
    directory, plan = read_plan(directory)
    wait_for_owned_launch(directory, plan)
    identity.require(not any(os.path.lexists(directory / name) for name in (
        "profile_process_intent.json", "capture_identity.json", "capture_actual_launch.json", "result.json", "failure.json", "trace")),
        "preserve the previous diagnostic and prepare a new unit")
    tool = plan["profile_tool"]
    actual_version = subprocess.run([tool["path"], "--version"], capture_output=True, timeout=15)
    identity.publish(directory / "tool_version_return.json", {"command": [tool["path"], "--version"],
        "returncode": actual_version.returncode, "stdout": actual_version.stdout.decode("utf-8", errors="strict"),
        "stderr": actual_version.stderr.decode("utf-8", errors="strict")})
    identity.require(actual_version.returncode == 0 and actual_version.stderr == b""
        and actual_version.stdout.decode().strip() == tool["version"]
        and identity.receipt(tool["path"])["sha256"] == tool["sha256"], "the actual installed Nsight tool changed")
    (directory / "trace").mkdir()
    command = identity.nsight_command(directory, plan)
    identity.publish(directory / "executed_command.json", command)
    intent = identity.make_intent(directory, plan, command)
    processes = None
    with (directory / "capture.log").open("x") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            born = identity.parsed_stat(Path("/proc", str(child.pid), "stat").read_text())["process_start_time"]
            identity.publish(directory / "profiler_process.json", {"pid": child.pid, "process_start_time": born,
                "parent_pid": os.getpid(), "parent_start_time": intent["parent"]["process_start_time"], "command": command})
            processes = identity.OwnedProcesses(child.pid, born, intent["parent"])
            started = time.monotonic()
            while not all((directory / name).exists() for name in ("capture_identity.json", "capture_actual_launch.json")):
                if child.poll() is not None or time.monotonic() - started > 120:
                    raise RuntimeError("the real Nsight probe did not reach its blocked identity boundary")
                time.sleep(.05)
            claim = identity.validate_claim(directory, plan, intent, identity.read_json(directory / "capture_identity.json"))
            observation = processes.observe(claim["pid"], claim["process_start_time"], intent["capture_command"])
            identity.publish(directory / "capture_process_observation.json", observation)
            rebuilt = identity.validate_observation(identity.read_json(directory / "capture_process_observation.json"),
                claim, intent["capture_command"], intent["parent"])
            identity.publish(directory / "probe_finish.json", {"challenge": intent["challenge"],
                "claim": identity.receipt(directory / "capture_identity.json")})
            returned = child.wait(timeout=60)
            actual_wait = identity.read_json(directory / "capture_actual_wait.json")
            identity.require(returned == 0 and actual_wait["waited_pid"] == claim["pid"]
                and type(actual_wait["returncode"]) is int and actual_wait["returncode"] == 0
                and actual_wait["interventions"] == [], "the exact real probe did not complete through its actual parent wait")
            identity.publish(directory / "result.json", {"schema_version": 1, "kind": KIND, "completed": True,
                "source": plan["helpers"], "plan": identity.receipt(directory / "measurement_plan.json"),
                "tool": tool, "command": command, "observation": identity.receipt(directory / "capture_process_observation.json"),
                "actual_capture_wait": identity.receipt(directory / "capture_actual_wait.json"),
                "nsight_returncode": returned, "capture_pid": claim["pid"], "capture_start_time": claim["process_start_time"],
                "old_process_group_equality_observed": rebuilt["capture"]["process_group"] == rebuilt["root"]["pid"],
                "same_numerical_profile_not_executed": True, "numerical_modules": sorted(NUMERICAL.intersection(sys.modules)),
                "cuda_profiler_api_called_by_payload": False, **BOUNDARY})
        except BaseException as error:
            cleanup = processes.stop(child) if processes is not None else None
            if processes is None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait()
            identity.publish(directory / "failure.json", {"error": repr(error), "traceback": traceback.format_exc(),
                "cleanup": cleanup, "actual_numerical_work": False, **BOUNDARY})
            raise
        finally:
            if processes is not None:
                processes.close()


def main(argv=None):
    """Require a distinct explicit diagnostic role and never run from the default command."""
    parser = argparse.ArgumentParser(description=__doc__)
    role = parser.add_mutually_exclusive_group(required=True)
    role.add_argument("--prepared-unit", type=Path)
    role.add_argument("--waiter-unit", type=Path)
    role.add_argument("--probe-unit", type=Path)
    args = parser.parse_args(argv)
    metadata_guard()
    if args.prepared_unit:
        run(args.prepared_unit)
    elif args.waiter_unit:
        raise SystemExit(waiter(args.waiter_unit))
    else:
        probe(args.probe_unit)


if __name__ == "__main__":
    main()
