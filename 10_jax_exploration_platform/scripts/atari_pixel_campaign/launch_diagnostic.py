"""Prepare, then optionally queue one recorded H100 execution diagnostic.

The default action writes metadata only. --launch starts a standard-library
supervisor which waits for existing probes/checks before acquiring the card lock.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time


GPU_UUID = "GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3"
PYTHON = "/p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python"
CORE_COMMIT = "d28e6735e79788f618acb0ebfde8609db1f776af"


def load_helper(path):
    """Load metadata helpers without importing numerical libraries."""
    specification = importlib.util.spec_from_file_location("atari_execution_diagnostic", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def write_once(path, contents):
    """Retain an identical launch artifact and reject any attempt to replace its contents."""
    path = Path(path)
    if path.exists():
        if path.read_bytes() != contents:
            raise ValueError("an immutable diagnostic artifact differs: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(contents)
        stream.flush()
        os.fsync(stream.fileno())


def encoded_json(value):
    """Use JSON syntax, also valid YAML, for complete configuration and manifest records."""
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def helper_identity():
    """Require both diagnostic helpers to be committed and record their exact original bytes."""
    scripts = Path(__file__).resolve().parent
    root = Path(subprocess.check_output(["git", "-C", str(scripts), "rev-parse", "--show-toplevel"], text=True).strip())
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    files = {}
    for name in ("diagnose_gpu_execution.py", "launch_diagnostic.py"):
        path = scripts / name
        relative = str(path.relative_to(root))
        committed = subprocess.check_output(["git", "-C", str(root), "show", commit + ":" + relative])
        contents = path.read_bytes()
        if contents != committed:
            raise ValueError("commit the diagnostic helper before preparation: " + relative)
        files[name] = {"path": str(path), "git_commit": commit,
                       "sha256": hashlib.sha256(contents).hexdigest()}
    return files


def append_process(run_dir, record):
    """Append one exact remote PID and birth counter to the existing locked process ledger."""
    ledger = run_dir / "processes.json"
    with ledger.with_suffix(".lock").open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        records = json.loads(ledger.read_text()) if ledger.exists() else []
        records.append(record)
        temporary = ledger.with_suffix(f".tmp-{os.getpid()}")
        with temporary.open("x") as target:
            target.write(json.dumps(records, indent=2) + "\n")
            target.flush()
            os.fsync(target.fileno())
        temporary.replace(ledger)


def predecessor_records(paths):
    """Return complete predecessor evidence, or None while any owned predecessor still runs."""
    records = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            return None
        contents = path.read_bytes()
        record = json.loads(contents)
        finished = record.get("completed") is True or (
            type(record.get("passed")) is bool and type(record.get("exit_code")) is int)
        if not finished:
            return None
        records.append({"path": str(path), "sha256": hashlib.sha256(contents).hexdigest(),
                        "completed": record.get("completed"), "passed": record.get("passed"),
                        "exit_code": record.get("exit_code")})
    return records


def free_selected_card():
    """Check the selected card's mode, free memory and processes after acquiring its standing lock."""
    settings = subprocess.check_output(["nvidia-smi", "-i", GPU_UUID,
        "--query-gpu=mig.mode.current,memory.used,name", "--format=csv,noheader,nounits"], text=True)
    mode, memory, name = [part.strip() for part in settings.strip().split(",")]
    processes = subprocess.check_output(["nvidia-smi", "-i", GPU_UUID,
        "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    confidential = subprocess.check_output(["nvidia-smi", "conf-compute", "-f", "-i", GPU_UUID], text=True)
    if mode != "Disabled" or int(memory) >= 1024 or processes or "H100" not in name or "CC status: OFF" not in confidential:
        raise ValueError("serval02 card 1 is not free in its required GPU mode")
    return {"gpu_uuid": GPU_UUID, "settings": settings.strip(), "processes": processes,
            "confidential_compute": confidential.strip()}


def bounded_command(command, seconds):
    """Bound only this supervisor's new process group, without signaling an existing process."""
    child = subprocess.Popen(command, start_new_session=True)
    try:
        return child.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        raise TimeoutError("the diagnostic exceeded its recorded active time limit")


def supervise(directory):
    """Wait without a GPU context, acquire the existing card lock, then run the fixed diagnostic."""
    directory = Path(directory).resolve(strict=True)
    plan = json.loads((directory / "diagnostic_plan.json").read_text())
    for name, identity in (("launch_diagnostic.py", plan["launcher"]),
                           ("diagnose_gpu_execution.py", plan["helper"])):
        if hashlib.sha256((directory / "code" / name).read_bytes()).hexdigest() != identity["sha256"]:
            raise ValueError("a staged diagnostic helper changed after preparation: " + name)
    helper = load_helper(directory / "code/diagnose_gpu_execution.py")
    deadline = time.monotonic() + 12 * 3600
    helper.record_progress(directory, "waiting_for_predecessors", predecessors=plan["after_results"])
    while (predecessors := predecessor_records(plan["after_results"])) is None:
        if time.monotonic() > deadline:
            raise TimeoutError("recorded predecessor checks did not finish within twelve hours")
        time.sleep(10)
    write_once(directory / "predecessor_results.json", encoded_json(predecessors))
    helper.record_progress(directory, "waiting_for_standing_lock")
    with Path("/tmp/sl5nw_serval02_card1.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        write_once(directory / "card_preflight.json", encoded_json(free_selected_card()))
        helper.core_identity(Path(plan["core"]["source_root"]), plan["core"]["source_commit"])
        worker = [plan["python"], str(directory / "code/diagnose_gpu_execution.py"), "--unit-dir", str(directory)]
        nsys = Path("/usr/local/bin/nsys")
        if nsys.is_file() and os.access(nsys, os.X_OK):
            command = [str(nsys), "profile", "--trace=cuda,cudnn", "--sample=none", "--cpuctxsw=none",
                "--cuda-graph-trace=node", "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
                "--stop-on-exit=true", "--force-overwrite=false", "--output=" + str(directory / "trace/capture"),
                *worker, "--trace-enabled"]
        else:
            command = worker
        write_once(directory / "executed_command.json", encoded_json(command))
        helper.record_progress(directory, "starting_diagnostic", command=command)
        returncode = bounded_command(command, plan["maximum_active_seconds"])
        if returncode != 0:
            raise RuntimeError(f"the GPU diagnostic exited with status {returncode}")
        if command != worker:
            report = directory / "trace/capture.nsys-rep"
            if not report.is_file():
                raise ValueError("Nsight did not produce the requested one-update capture")
            stats = [str(nsys), "stats", "--report=cuda_gpu_kern_sum,cuda_api_sum", "--format=csv",
                     "--force-export=true", "--output=" + str(directory / "trace/summary"), str(report)]
            stats_code = bounded_command(stats, 180)
            helper.atomic_json(directory / "trace/analysis.json", {"command": stats, "exit_code": stats_code,
                "report": str(report), "completed": stats_code == 0})
            if stats_code != 0:
                raise RuntimeError(f"Nsight summary generation exited with status {stats_code}")
        helper.record_progress(directory, "completed", completed=True, diagnostic_only=True)


def prepare(args):
    """Stage immutable helper bytes and all launch records before optionally starting a supervisor."""
    source_root, run_dir = args.source_root.resolve(strict=True), args.run_dir.resolve(strict=True)
    helper = load_helper(Path(__file__).with_name("diagnose_gpu_execution.py"))
    if args.source_commit != CORE_COMMIT:
        raise ValueError("this diagnostic must use the unchanged frozen reference commit")
    core = helper.core_identity(source_root, args.source_commit)
    helpers = helper_identity()
    configuration = helper.default_configuration(source_root)
    if configuration["learner"]["n_copies"] != 1 or configuration["learner"]["observation_warmup_iterations"] != 50:
        raise ValueError("the frozen defaults do not match the requested one-policy diagnostic")
    identifier = ("serval02_reference_execution_diagnostic_copies-1_environments-per-copy-128_rollout-steps-128"
        "_epochs-4_minibatches-4_warmup-rollouts-50_timed-rounds-2_updates-per-round-2_traced-updates-1_seed-1"
        "_source-" + args.source_commit[:12] + "_helper-" + helpers["diagnose_gpu_execution.py"]["git_commit"][:12])
    directory = run_dir / "data" / identifier
    after = [
        run_dir / "data/serval02_pixel_benchmark_copies_2_source_1d488ab23e11/result.json",
        run_dir / ("data/serval02_pixel_gates_copies_1_source_" + args.source_commit[:12] + "/result.json"),
        run_dir / ("data/serval02_pixel_rendering_copies_1_source_" + args.source_commit[:12] + "/result.json")]
    after += [path for path in (args.after_result or []) if path.resolve() not in after]
    plan = {"schema_version": 1, "diagnostic_only": True, "formal_benchmark": False,
            "capacity_evidence": False, "unit_id": identifier, "core": core,
            "helper": helpers["diagnose_gpu_execution.py"], "launcher": helpers["launch_diagnostic.py"],
            "configuration": configuration, "python": str(args.python), "gpu_uuid": GPU_UUID,
            "standing_lock": "/tmp/sl5nw_serval02_card1.lock", "maximum_active_seconds": 2700,
            "after_results": [str(path.resolve()) for path in after]}
    plan_bytes = encoded_json(plan)
    artifacts = {directory / "diagnostic_plan.json": plan_bytes,
                 directory / "manifest.yaml": encoded_json({"schema_version": 1, "run_id": identifier,
                     "description": "Separate full-size one-policy deterministic execution diagnostic; no formal result admission.",
                     "git": {"commit": args.source_commit, "dirty": False},
                     "helper": plan["helper"], "configuration": configuration}),
                 directory / "config_resolved.yaml": encoded_json(configuration)}
    for name in helpers:
        artifacts[directory / "code" / name] = Path(helpers[name]["path"]).read_bytes()
    command = [str(args.python), str(directory / "code/diagnose_gpu_execution.py"), "--unit-dir", str(directory), "--trace-enabled"]
    artifacts[directory / "command.txt"] = (shlex.join(command) + "\n").encode()
    background = ("## Purpose\n\nMeasure full-size one-policy training with deterministic operations to distinguish "
        "the runtime setting from the change to two independent policies. Trace one later update for CUDA kernel durations. "
        "These are diagnostic measurements, not formal benchmark or capacity evidence.\n\n"
        "## Key hyperparameters\n\nOne policy, 128 environments, 128 rollout steps, four epochs, four minibatches, "
        "50 normalization warmup rollouts, seed one. Two compile updates, two timing rounds of two updates, "
        "then one separately traced update. Policy and optimizer use float32; running statistics use float64; "
        "matrix precision is highest and deterministic GPU operations remain required.\n\n"
        "## Code and config changes\n\nThe learner and environment use the unchanged frozen core. "
        "A separately committed helper controls timing and CUDAProfilerApi capture. No checkpoint is saved. "
        "The supervisor waits for all recorded predecessors and acquires the standing lock for serval02 card 1.\n\n"
        f"## Git state\n\nFrozen core commit: `{args.source_commit}`. "
        f"Diagnostic helper commit: `{plan['helper']['git_commit']}`. "
        "All core bytes are checked against Git and recorded with SHA-256 hashes.\n")
    artifacts[directory / "experiment_background.md"] = background.encode()
    for path, contents in artifacts.items():
        if path.exists() and path.read_bytes() != contents:
            raise ValueError("an immutable diagnostic artifact differs: " + str(path))
    for path, contents in artifacts.items():
        write_once(path, contents)
    (directory / "trace").mkdir(exist_ok=True)
    if args.launch:
        return launch(run_dir, directory, plan)
    return {"prepared": True, "launched": False, "unit_dir": str(directory), "plan": str(directory / "diagnostic_plan.json")}


def launch(run_dir, directory, plan):
    """Serialize launch attempts and reserve this unit before any remote process can start."""
    with (directory / "launch.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        attempt = directory / "launch_attempt.json"
        if attempt.exists():
            raise ValueError("this diagnostic already has a recorded launch attempt; inspect its exact ownership before retrying")
        write_once(attempt, encoded_json({"pid": os.getpid(), "time": datetime.now(timezone.utc).isoformat(),
            "source_commit": plan["core"]["source_commit"], "helper": plan["helper"]}))
        return launch_supervisor(run_dir, directory, plan)


def launch_supervisor(run_dir, directory, plan):
    """Record the remote supervisor before releasing it to wait for predecessors and the card lock."""
    release = directory / "supervisor.released"
    pid_file = directory / "supervisor.pid"
    if pid_file.exists() or release.exists():
        raise ValueError("this diagnostic already has a recorded supervisor")
    environment = Path(plan["core"]["source_root"]) / "10_jax_exploration_platform/scripts/atari_pixel_campaign/worker_environment.sh"
    script = directory / "supervisor.sh"
    log = directory / "supervisor.log"
    lines = ["#!/bin/bash", "set -euo pipefail",
        f"for attempt in {{1..180}}; do test -r {shlex.quote(str(release))} && break; sleep 1; done",
        f"test -r {shlex.quote(str(release))}", f"source {shlex.quote(str(environment))}",
        f"export CUDA_VISIBLE_DEVICES={GPU_UUID}", "export PYTHONDONTWRITEBYTECODE=1",
        "exec /usr/bin/python3 " + shlex.join([str(directory / "code/launch_diagnostic.py"), "--supervise", str(directory)])]
    write_once(script, ("\n".join(lines) + "\n").encode())
    remote = ("nohup bash " + shlex.quote(str(script)) + " >" + shlex.quote(str(log))
        + " 2>&1 </dev/null & diagnostic_pid=$!; /usr/bin/python3 - \"$diagnostic_pid\" <<'PY'\n"
        + "import json,pathlib,socket,sys\n"
        + "pid=int(sys.argv[1]); text=pathlib.Path(f'/proc/{pid}/stat').read_text()\n"
        + "print(json.dumps({'pid':pid,'start_time':text.rsplit(')',1)[1].split()[19],"
        + "'hostname':socket.gethostname()}),flush=True)\nPY")
    result = subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
                             "serval02", remote], capture_output=True, text=True, check=True)
    identity = json.loads(result.stdout.strip().splitlines()[-1])
    if type(identity.get("pid")) is not int or identity["pid"] < 1 or not identity.get("start_time"):
        raise ValueError("the remote diagnostic did not return a valid PID and birth counter")
    record = {**identity, "host": "serval02", "unit": plan["unit_id"], "kind": "execution_diagnostic",
              "context": "h100", "gpu_uuid": GPU_UUID, "source_root": plan["core"]["source_root"],
              "source_commit": plan["core"]["source_commit"], "helper": plan["helper"],
              "script": str(script), "log": str(log), "time": datetime.now(timezone.utc).isoformat()}
    append_process(run_dir, record)
    write_once(pid_file, (str(identity["pid"]) + "\n").encode())
    write_once(release, encoded_json(record))
    return record


def main():
    """Keep preparation metadata-only and require an explicit launch flag for a remote supervisor."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--source-commit", default=CORE_COMMIT)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--after-result", type=Path, action="append")
    parser.add_argument("--python", type=Path, default=Path(PYTHON))
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--supervise", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.supervise:
        try:
            supervise(args.supervise)
        except BaseException as error:
            helper = load_helper(args.supervise / "code/diagnose_gpu_execution.py")
            helper.atomic_json(args.supervise / "supervisor_failure.json", {"error": repr(error),
                "time": datetime.now(timezone.utc).isoformat(), "diagnostic_only": True})
            raise
        return
    if args.source_root is None or args.run_dir is None:
        parser.error("--source-root and --run-dir are required for preparation or launch")
    if not args.source_root.is_absolute() or not args.run_dir.is_absolute() or not args.python.is_absolute():
        raise ValueError("source, run and Python paths must be absolute")
    print(json.dumps(prepare(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
