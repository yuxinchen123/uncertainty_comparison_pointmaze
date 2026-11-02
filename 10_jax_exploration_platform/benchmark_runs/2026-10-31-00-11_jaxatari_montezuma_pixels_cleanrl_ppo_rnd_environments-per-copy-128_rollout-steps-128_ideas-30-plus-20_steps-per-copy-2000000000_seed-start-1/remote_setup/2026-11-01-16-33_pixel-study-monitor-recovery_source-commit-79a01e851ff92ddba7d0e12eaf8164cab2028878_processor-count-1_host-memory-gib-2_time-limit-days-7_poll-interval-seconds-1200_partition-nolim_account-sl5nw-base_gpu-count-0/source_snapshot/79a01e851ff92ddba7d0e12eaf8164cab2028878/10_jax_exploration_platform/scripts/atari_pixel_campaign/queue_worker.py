"""Claim and execute one pixel-benchmark case at a time on one explicitly named GPU."""

import argparse
from contextlib import contextmanager
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from controller import (attempt_paths, atomic_json, case_states, layout,
                        process_start_time, queue_lock, read_json, utc_now,
                        ready_specification, scientific_case_complete, verify_source)
from ideas import REQUIRED_HARDWARE


def gpu_identity(gpu_uuid, context):
    """Verify the specified physical GPU and refuse an ambiguous CUDA visibility setting."""
    if not gpu_uuid.startswith("GPU-"):
        raise ValueError("--gpu-uuid must name the exact physical card")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible != gpu_uuid:
        raise ValueError("set CUDA_VISIBLE_DEVICES to the same single GPU UUID before starting this worker")
    result = subprocess.run(["nvidia-smi", "-i", gpu_uuid, "--query-gpu=uuid,name",
                             "--format=csv,noheader"], capture_output=True, text=True, check=True, timeout=10)
    rows = list(csv.reader(result.stdout.splitlines()))
    if len(rows) != 1 or rows[0][0].strip() != gpu_uuid:
        raise ValueError("nvidia-smi did not resolve the requested single GPU")
    name = rows[0][1].strip()
    if context == "h100" and "H100" not in name:
        raise ValueError("the H100 worker does not hold an H100")
    if context == "jaguar03_rtx_a4500" and ("A4500" not in name or socket.gethostname() != "jaguar03"):
        raise ValueError("the A4500 context requires an RTX A4500 on jaguar03")
    return {"gpu_uuid": gpu_uuid, "gpu_name": name}


def worker_identity(context, gpu):
    """Record exactly this worker's process and scheduler identity, without submitting jobs."""
    return {"worker_id": uuid.uuid4().hex, "hostname": socket.gethostname(),
            "worker_pid": os.getpid(), "worker_start_time": process_start_time(os.getpid()),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            "hardware_class": context, **gpu}


@contextmanager
def standing_case_lock(path):
    """Hold a standing host lock for one case only, never while waiting for another batch."""
    if path is None:
        yield True, None
        return
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("the standing GPU lock path must be absolute")
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False, None
            return
        try:
            yield True, stream.fileno()
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def gpu_processes(gpu_uuid):
    """Wait for any existing compute process instead of starting a competing measurement."""
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
                            capture_output=True, text=True, check=True, timeout=10)
    processes = []
    for row in csv.reader(result.stdout.splitlines()):
        if not row:
            continue
        if len(row) != 2:
            raise RuntimeError(f"unexpected nvidia-smi process row: {row!r}")
        if row[1].strip() == gpu_uuid:
            processes.append(int(row[0].strip()))
    return processes


def gpu_readiness(gpu_uuid, context):
    """Recheck processes, free memory and H100 modes after taking the per-case host lock."""
    processes = gpu_processes(gpu_uuid)
    if processes:
        return {"ready": False, "reason": "gpu_has_compute_processes", "processes": processes}
    if context == "h100":
        result = subprocess.run(["nvidia-smi", "-i", gpu_uuid,
                                 "--query-gpu=mig.mode.current,memory.used", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, check=True, timeout=10)
        rows = list(csv.reader(result.stdout.splitlines()))
        if len(rows) != 1 or len(rows[0]) != 2:
            raise RuntimeError("cannot read the selected H100's mode and memory")
        mode, memory = (value.strip() for value in rows[0])
        if mode != "Disabled" or int(memory) >= 1024:
            return {"ready": False, "reason": "h100_mode_or_memory_changed", "mig_mode": mode,
                    "memory_used_mib": int(memory)}
        # NVIDIA documents UUID selection as a conf-compute subcommand option:
        # https://docs.nvidia.com/deploy/nvidia-smi/index.html#confidential-compute
        confidential = subprocess.run(["nvidia-smi", "conf-compute", "-i", gpu_uuid, "-f"],
                                       capture_output=True, text=True, check=True, timeout=10).stdout
        statuses = [line.split(":", 1)[1].strip() for line in confidential.splitlines()
                    if "CC status:" in line]
        if statuses != ["OFF"]:
            return {"ready": False, "reason": "confidential_computing_not_off", "status": confidential.strip()}
    return {"ready": True}


def eligible_pending_exists(run_dir, context):
    """Look for authorized work before taking a standing host lock or retaining a Slurm allocation."""
    with queue_lock(run_dir):
        ready = ready_specification(run_dir)
        if ready is None:
            return False
        return any(case["hardware_class"] == context
                   and (case["source_root"], case["source_commit"]) == (ready["source_root"], ready["source_commit"])
                   for case in (read_json(path) for path in (Path(run_dir) / "queue/pending").glob("*.json")))


def exit_when_idle(*, once=False, idle_exit=False, slurm_job_id=None):
    """Release empty Slurm allocations; a direct-host metadata supervisor may wait for later work."""
    job = os.environ.get("SLURM_JOB_ID", "") if slurm_job_id is None else slurm_job_id
    return bool(once or idle_exit or job)


def claim_next(run_dir, context, identity):
    """Atomically rename one eligible pending file, then persist its lease before any child starts."""
    run_dir = Path(run_dir)
    queue = run_dir / "queue"
    with queue_lock(run_dir):
        ready = ready_specification(run_dir)
        if ready is None:
            return None
        for path in sorted((queue / "pending").glob("*.json")):
            case = read_json(path)
            if case["hardware_class"] != context:
                continue
            if (case["source_root"], case["source_commit"]) != (ready["source_root"], ready["source_commit"]):
                continue
            verify_source(case["source_root"], case["source_commit"])
            if case_states(run_dir, case["unit_id"]) != ["pending"]:
                raise ValueError("a pending case occupies another queue state")
            previous_path = queue / "leases" / path.name
            previous = read_json(previous_path) if previous_path.exists() else None
            attempt = previous["attempt"] + 1 if previous is not None else 1
            if attempt > case["maximum_attempts"]:
                raise ValueError("a pending case already exhausted its explicit retry limit")
            path.rename(queue / "running" / path.name)
            lease = {**identity, "token": uuid.uuid4().hex, "unit_id": case["unit_id"],
                     "attempt": attempt, "state": "claimed", "claimed_at": utc_now(),
                     "runner_pid": None, "runner_start_time": None}
            atomic_json(previous_path, lease)
            _, attempt_dir = attempt_paths(run_dir, case["unit_id"], attempt)
            attempt_dir.mkdir(parents=True, exist_ok=False)
            atomic_json(attempt_dir / "lease.json", lease)
            return case, lease
    return None


def finish_claim(run_dir, case, lease, exit_code):
    """Keep scientific rejections as completed work and infrastructure failures as retry candidates."""
    run_dir = Path(run_dir)
    queue = run_dir / "queue"
    unit, attempt = attempt_paths(run_dir, case["unit_id"], lease["attempt"])
    result_path = attempt / "result.json"
    result = read_json(result_path) if result_path.exists() else None
    complete = scientific_case_complete(result)
    with queue_lock(run_dir):
        current_path = queue / "leases" / f"{case['unit_id']}.json"
        current = read_json(current_path)
        if current["token"] != lease["token"]:
            raise ValueError("lease ownership changed while the case was running")
        running = queue / "running" / f"{case['unit_id']}.json"
        if complete:
            atomic_json(unit / "result.json", result)
            destination = "completed"
        else:
            failure_path = attempt / "failure.json"
            if failure_path.exists():
                failure = read_json(failure_path)
            else:
                failure = {"failure_stage": "invalid_measurement" if result else "runner_exit",
                           "retryable": True, "case_completed": False,
                           "exit_code": exit_code, "time": utc_now()}
                if result and result.get("contaminated"):
                    failure["failure_stage"] = "unrelated_gpu_use"
                atomic_json(failure_path, failure)
            atomic_json(unit / "failure.json", failure)
            destination = "failed"
        current.update(state=destination, finished_at=utc_now(), runner_exit_code=exit_code)
        atomic_json(current_path, current)
        atomic_json(attempt / "lease.json", current)
        running.rename(queue / destination / running.name)
        atomic_json(unit / "progress.json", {"state": destination, "attempt": lease["attempt"],
                                            "unit_id": case["unit_id"], "time": utc_now(),
                                            "decision": result.get("decision") if result else None})
    return destination


def run_one(run_dir, case, lease, python, gpu_lock_fd, standing_lock_fd=None):
    """Start the immutable case runner, recording its PID before allowing GPU initialization."""
    unit, attempt = attempt_paths(run_dir, case["unit_id"], lease["attempt"])
    source = Path(case["source_root"]) / "10_jax_exploration_platform/scripts/atari_pixel_campaign/run_case.py"
    case_path = Path(run_dir) / "queue" / "specifications" / f"{case['unit_id']}.json"
    lease_path = Path(run_dir) / "queue" / "leases" / f"{case['unit_id']}.json"
    command = [python, str(source), "--run-dir", str(Path(run_dir).resolve()), "--case", str(case_path.resolve()),
               "--attempt", str(lease["attempt"]), "--lease-token", lease["token"]]
    with (attempt / "runner.log").open("x") as log:
        descriptors = ((gpu_lock_fd,) if standing_lock_fd is None else (gpu_lock_fd, standing_lock_fd))
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=case["source_root"],
                                 pass_fds=descriptors)
        # The child waits for this release record before importing JAX. If the worker dies
        # before writing it, the waiting child cannot start an unrecorded GPU measurement.
        lease.update(runner_pid=child.pid, runner_start_time=process_start_time(child.pid), state="running")
        with queue_lock(run_dir):
            current = read_json(lease_path)
            if current["token"] != lease["token"]:
                raise ValueError("lease changed before child release")
            atomic_json(lease_path, lease)
            atomic_json(attempt / "lease.json", lease)
        exit_code = child.wait()
    return finish_claim(run_dir, case, lease, exit_code)


def perform_next_case(run_dir, context, identity, python, gpu_lock_fd, standing_lock=None):
    """Take and release per-case admission around exactly one recorded numerical child."""
    if not eligible_pending_exists(run_dir, context):
        return {"state": "idle", "ran_case": False}
    with standing_case_lock(standing_lock) as (acquired, standing_fd):
        if not acquired:
            return {"state": "waiting_for_standing_lock", "ran_case": False}
        availability = gpu_readiness(identity["gpu_uuid"], context)
        if not availability["ready"]:
            return {"state": "waiting_for_gpu", "ran_case": False, "gpu_readiness": availability}
        claimed = claim_next(run_dir, context, identity)
        if claimed is None:
            return {"state": "idle", "ran_case": False}
        case, lease = claimed
        atomic_json(Path(run_dir) / "queue/workers" / f"{identity['worker_id']}.json",
                    {**identity, "state": "running", "unit_id": case["unit_id"],
                     "attempt": lease["attempt"], "time": utc_now()})
        destination = run_one(run_dir, case, lease, python, gpu_lock_fd, standing_fd)
        return {"state": destination, "ran_case": True, "unit_id": case["unit_id"],
                "attempt": lease["attempt"]}


def main():
    """Run a standing worker without creating jobs, cancelling work or reclaiming old live claims."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--context", choices=REQUIRED_HARDWARE, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--idle-exit", action="store_true")
    parser.add_argument("--standing-lock", type=Path)
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    if args.poll_seconds <= 0:
        raise ValueError("poll interval must be positive")
    required_locks = {"serval02": "/tmp/sl5nw_serval02_card1.lock", "serval05": "/tmp/jr_efficiency_profile.lock"}
    hostname = socket.gethostname().split(".")[0]
    if hostname in required_locks and str(args.standing_lock) != required_locks[hostname]:
        raise ValueError("direct serval workers require the machine's standing lock as --standing-lock; "
                         "do not wrap the whole queue in an outer standing flock")
    if hostname == "serval02" and args.gpu_uuid != "GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3":
        raise ValueError("this study may use only the recorded serval02 card 1 UUID, never card 0")
    layout(args.run_dir)
    gpu = gpu_identity(args.gpu_uuid, args.context)
    identity = worker_identity(args.context, gpu)
    stop = {"requested": False}

    def stop_after_case(signum, frame):
        """Leave the current recorded runner alone and stop claiming subsequent cases."""
        stop["requested"] = True

    signal.signal(signal.SIGTERM, stop_after_case)
    signal.signal(signal.SIGINT, stop_after_case)
    record_path = args.run_dir / "queue" / "workers" / f"{identity['worker_id']}.json"
    lock_path = Path("/tmp") / f"sl5nw_atari_pixel_benchmark_{args.gpu_uuid}.lock"
    with lock_path.open("a+") as gpu_lock:
        try:
            fcntl.flock(gpu_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another pixel benchmark worker already holds this GPU's queue lock") from error
        try:
            while not stop["requested"]:
                atomic_json(record_path, {**identity, "state": "waiting", "time": utc_now()})
                outcome = perform_next_case(args.run_dir, args.context, identity, args.python,
                                             gpu_lock.fileno(), args.standing_lock)
                atomic_json(record_path, {**identity, **outcome, "time": utc_now()})
                if args.once or (args.run_dir / "STUDY_COMPLETE").exists():
                    break
                if outcome["state"] == "idle" and exit_when_idle(idle_exit=args.idle_exit):
                    break
                if not outcome["ran_case"]:
                    time.sleep(args.poll_seconds)
        finally:
            atomic_json(record_path, {**identity, "state": "stopped", "time": utc_now()})
            fcntl.flock(gpu_lock.fileno(), fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
