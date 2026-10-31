"""Run receipt-authorized retries with one metadata parent per Slurm node allocation.

Each generic worker uses one proved physical UUID. The original full Slurm GPU
allocation remains unchanged in every child; only CUDA visibility is narrowed.
This program creates no approvals, ordinary pending cases, or Slurm jobs.
"""

import argparse
from contextlib import contextmanager
import csv
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
import uuid

import infrastructure_retry as retry
import controller
import queue_worker as ordinary
import measure_gpu_scaling as scaling


def verify_worker_source(approval):
    """Require the approved actual helper checkout rather than an archive under another Git root."""
    root = Path(approval["helpers"]["source_root"])
    if Path(__file__).resolve() != root / retry.PREFIX / "infrastructure_retry_worker.py":
        raise ValueError("execute the retry worker from its exact committed approval helper root")
    for name, module in (("infrastructure_retry", retry), ("controller", controller), ("queue_worker", ordinary),
                         ("measure_gpu_scaling", scaling), ("measure_selected", scaling.selected)):
        if Path(module.__file__).resolve() != root / retry.PREFIX / (name + ".py"):
            raise ValueError("the retry parent imported another helper: " + name)


def claim_retry(run_dir, identity, assignment_path):
    """Claim one approved held case directly from failed into running, never through ordinary pending."""
    run, evidence = Path(run_dir), retry.LocalEvidence()
    with controller.queue_lock(run):
        ready = controller.ready_specification(run)
        if ready is None:
            return None
        for approval_path in sorted((run / "queue/infrastructure_retries/approvals").glob("*.json")):
            claim_path = run / "queue/infrastructure_retries/claims" / approval_path.name
            if claim_path.exists():
                continue
            approval, case = retry.validate_approval(approval_path)
            verify_worker_source(approval)
            if (case["source_root"], case["source_commit"]) != (ready["source_root"], ready["source_commit"]):
                raise ValueError("the retry approval differs from the admitted original numerical source")
            failed = run / "queue/failed" / approval_path.name
            lease_path = run / "queue/leases" / approval_path.name
            prior = controller.read_json(lease_path)
            if (controller.case_states(run, case["unit_id"]) != ["failed"] or failed.read_bytes() != evidence.file(approval["case"]["path"])
                    or retry.digest(lease_path.read_bytes()) != approval["previous_lease"]["sha256"]
                    or prior.get("attempt") != 3 or prior.get("state") != "failed"):
                raise ValueError("the held case changed after its one-shot retry approval")
            _, attempt = controller.attempt_paths(run, case["unit_id"], 4)
            if attempt.exists():
                raise ValueError("the approved fourth attempt already has a directory; do not consume it again")
            approval_record = retry.file_record(approval_path, evidence)
            lease = {**identity, "token": uuid.uuid4().hex, "unit_id": case["unit_id"], "attempt": 4,
                "state": "claimed", "claimed_at": controller.utc_now(), "runner_pid": None, "runner_start_time": None,
                "infrastructure_retry_approval": approval_record, "assignment": retry.file_record(assignment_path, evidence)}
            # The immutable one-shot claim is published first. Any partial transition holds for manual review.
            retry.write_once(claim_path, retry.json_bytes({"approval": approval_record, "lease": lease}))
            attempt.mkdir(parents=True, exist_ok=False)
            controller.atomic_json(attempt / "lease.json", lease)
            controller.atomic_json(lease_path, lease)
            failed.rename(run / "queue/running" / failed.name)
            return case, lease, approval_path, approval
    return None


@contextmanager
def interruption_cleanup():
    """Route termination through exact-owned numerical-child cleanup before releasing its GPU lock."""
    def interrupted(number, frame):
        """Raise within the current owner so its cleanup can finish before the process exits."""
        raise InterruptedError("infrastructure retry received signal " + str(number))
    previous = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}
    for number in previous:
        signal.signal(number, interrupted)
    try:
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def stop_owned(children):
    """Stop only still-live direct children whose birth counter and parent relationship match."""
    for child in children:
        process = child["process"]
        if process.poll() is not None:
            continue
        try:
            fields = Path(f"/proc/{process.pid}/stat").read_text().rsplit(")", 1)[1].split()
        except FileNotFoundError:
            continue
        if int(fields[1]) != os.getpid() or (child.get("process_start_time") is not None and fields[19] != child["process_start_time"]):
            raise RuntimeError("refuse cleanup of a process without this unit's exact parent/birth ownership")
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            if controller.process_matches(process.pid, fields[19]):
                process.kill()
                process.wait(timeout=10)


def run_claim(run_dir, claimed, assignment_path, gpu_lock_fd):
    """Record the new child PID before release, then validate its sidecar before completing the claim."""
    case, lease, approval_path, approval = claimed
    _, attempt = controller.attempt_paths(run_dir, case["unit_id"], 4)
    source = Path(approval["helpers"]["source_root"]) / retry.PREFIX / "run_infrastructure_retry.py"
    command = [approval["python"], str(source), "--approval", str(approval_path), "--assignment", str(assignment_path),
               "--lease-token", lease["token"]]
    retry.write_once(attempt / "command.txt", ("\n".join(command) + "\n").encode())
    children = []
    with (attempt / "runner.log").open("x") as log:
        try:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                cwd=case["source_root"], pass_fds=(gpu_lock_fd,))
            owned = {"process": process, "process_start_time": None}
            children.append(owned)
            owned["process_start_time"] = controller.process_start_time(process.pid)
            if owned["process_start_time"] is None:
                raise RuntimeError("the retry runner ended before its exact birth counter was recorded")
            lease.update(runner_pid=process.pid, runner_start_time=owned["process_start_time"], state="running")
            with controller.queue_lock(run_dir):
                lease_path = Path(run_dir) / "queue/leases" / (case["unit_id"] + ".json")
                if controller.read_json(lease_path)["token"] != lease["token"]:
                    raise ValueError("the retry lease changed before child release")
                controller.atomic_json(attempt / "lease.json", lease)
                controller.atomic_json(lease_path, lease)
            exit_code = process.wait()
        except BaseException:
            stop_owned(children)
            raise
    # A missing sidecar is never accepted merely because the original runner wrote a raw result.
    result_path = attempt / "result.json"
    if result_path.exists():
        try:
            retry.validate_execution(result_path, case, lease)
        except (ValueError, KeyError, FileNotFoundError) as error:
            controller.atomic_json(attempt / "failure.json", {"case_completed": False, "retryable": False,
                "failure_stage": "infrastructure_retry_proof", "error": str(error), "attempt": 4,
                "source_commit": retry.SOURCE_COMMIT, "time": controller.utc_now()})
    return finish_retry(run_dir, case, lease, exit_code)


def finish_retry(run_dir, case, lease, exit_code):
    """Preserve raw bytes while holding a failed fourth attempt with no additional retry allowance."""
    _, attempt = controller.attempt_paths(run_dir, case["unit_id"], 4)
    path = attempt / "result.json"
    valid = path.exists() and controller.scientific_case_complete(controller.read_json(path)) and not (attempt / "failure.json").exists()
    if valid:
        retry.validate_execution(path, case, lease)
        return ordinary.finish_claim(run_dir, case, lease, exit_code)
    # The ordinary finisher must not mistake an unverified raw result for completed work.
    with controller.queue_lock(run_dir):
        queue = Path(run_dir) / "queue"
        lease_path = queue / "leases" / (case["unit_id"] + ".json")
        current = controller.read_json(lease_path)
        if current["token"] != lease["token"]:
            raise ValueError("the failed retry no longer owns its lease")
        failure_path = attempt / "failure.json"
        if not failure_path.exists():
            controller.atomic_json(failure_path, {"case_completed": False, "retryable": False, "attempt": 4,
                "failure_stage": "infrastructure_retry_exit", "exit_code": exit_code, "time": controller.utc_now()})
        unit = attempt.parent.parent
        controller.atomic_json(unit / "failure.json", controller.read_json(failure_path))
        current.update(state="failed", finished_at=controller.utc_now(), runner_exit_code=exit_code)
        controller.atomic_json(lease_path, current)
        controller.atomic_json(attempt / "lease.json", current)
        (queue / "running" / (case["unit_id"] + ".json")).rename(queue / "failed" / (case["unit_id"] + ".json"))
        controller.atomic_json(unit / "progress.json", {"state": "failed", "attempt": 4,
            "unit_id": case["unit_id"], "time": controller.utc_now(), "retry_allowance_exhausted": True})
    return "failed"


def bind_worker_cpus(assignment, gpu_uuid, environment):
    """Limit this metadata worker and its future numerical child to the exact approved eight CPUs."""
    cpus = retry.validate_cpu_allocation(assignment, gpu_uuid)
    proof = assignment["cpu_allocation"]
    if (environment.get("SLURM_CPUS_PER_TASK") != proof["slurm_cpus_per_task"]
            or environment.get("SLURM_CPU_BIND_TYPE") != proof["slurm_cpu_bind_type"]):
        raise ValueError("the retry worker changed its original allocated CPU declaration")
    os.sched_setaffinity(0, set(cpus))
    if sorted(os.sched_getaffinity(0)) != cpus:
        raise ValueError("the retry metadata worker did not receive its assigned eight CPUs")
    environment.update(retry.THREAD_ENVIRONMENT)
    return cpus


def run_worker(run_dir, assignment_path, gpu_uuid):
    """Use one physical card for generic approved receipts until this separate queue is empty."""
    assignment = controller.read_json(assignment_path)
    retry.validate_assignment(assignment, gpu_uuid)
    proof = assignment["allocation"]
    if ({name: os.environ[name] for name in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS") if os.environ.get(name)} != proof["slurm_allocations"]
            or os.environ.get("SLURM_JOB_ID") != proof["slurm_job_id"]
            or not controller.process_matches(assignment["parent_pid"], assignment["parent_process_start_time"])):
        raise ValueError("the retry worker lost its parent or original full allocation")
    cpus = bind_worker_cpus(assignment, gpu_uuid, os.environ)
    identity = ordinary.worker_identity("jaguar03_rtx_a4500", ordinary.gpu_identity(gpu_uuid, "jaguar03_rtx_a4500"))
    identity.update(cpu_affinity=cpus, thread_environment={name: os.environ[name] for name in retry.THREAD_ENVIRONMENT})
    record = Path(run_dir) / "queue/workers" / (identity["worker_id"] + ".json")
    with (Path("/tmp") / f"sl5nw_atari_pixel_benchmark_{gpu_uuid}.lock").open("a+") as gpu_lock, interruption_cleanup():
        fcntl.flock(gpu_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while True:
                if not ordinary.gpu_readiness(gpu_uuid, "jaguar03_rtx_a4500")["ready"]:
                    raise ValueError("the assigned retry GPU has another numerical process")
                claimed = claim_retry(run_dir, identity, assignment_path)
                if claimed is None:
                    break
                case, lease, _, _ = claimed
                controller.atomic_json(record, {**identity, "state": "running", "unit_id": case["unit_id"], "attempt": 4,
                    "retry_scope": "explicit allocator-history infrastructure retry", "time": controller.utc_now()})
                try:
                    destination = run_claim(run_dir, claimed, assignment_path, gpu_lock.fileno())
                except BaseException:
                    finish_retry(run_dir, case, lease, -1)
                    raise
                controller.atomic_json(record, {**identity, "state": destination, "unit_id": case["unit_id"], "attempt": 4,
                    "time": controller.utc_now()})
        finally:
            controller.atomic_json(record, {**identity, "state": "stopped", "time": controller.utc_now()})


def run_parent(arguments):
    """Start generic metadata workers for the complete proved allocation; submit no jobs."""
    run, directory = arguments.run_dir.resolve(strict=True), arguments.output_dir.resolve()
    if directory == run or run not in directory.parents or "jax" in sys.modules:
        raise ValueError("use a new campaign child output directory and a nonnumerical retry parent")
    approvals = list((run / "queue/infrastructure_retries/approvals").glob("*.json"))
    if len(approvals) != 5:
        raise ValueError("the approved retry batch contains exactly five case-specific receipts")
    for path in approvals:
        approval, _ = retry.validate_approval(path)
        verify_worker_source(approval)
        if approval["python"] != str(arguments.python.absolute()):
            raise ValueError("all retry workers must use the exact approved Python")
    rows = list(csv.reader(subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True, timeout=15).splitlines()))
    roster = [row[0].strip() for row in rows if len(row) == 1]
    declared_gpus = os.environ.get("SLURM_GPUS_ON_NODE")
    if (not 1 <= arguments.gpu_count <= 8 or len(roster) != arguments.gpu_count
            or (declared_gpus is not None and (not declared_gpus.isdecimal() or int(declared_gpus) != arguments.gpu_count))):
        raise ValueError("the retry node parent requires exactly the requested allocated GPU count")
    proof = scaling.allocation_proof({"gpu_count": len(roster), "gpu_uuids": roster}, os.environ, socket.gethostname())
    cpus = retry.cpu_allocation(os.environ, roster, os.sched_getaffinity(0))
    assignment = {"schema_version": 1, "hostname": "jaguar03", "parent_pid": os.getpid(),
        "parent_process_start_time": controller.process_start_time(os.getpid()), "gpu_uuids": roster,
        "expected_gpu_count": arguments.gpu_count, "allocation": proof, "cpu_allocation": cpus,
        "approvals": [retry.file_record(path, retry.LocalEvidence()) for path in sorted(approvals)]}
    retry.validate_assignment(assignment, roster[0])
    directory.mkdir(parents=True, exist_ok=False)
    assignment_path = directory / "assignment.json"
    retry.write_once(assignment_path, retry.json_bytes(assignment))
    manifest = {"kind": "explicit_allocator_history_infrastructure_retries", "source_commit": retry.SOURCE_COMMIT,
        "helpers": approval["helpers"], "assignment": retry.file_record(assignment_path, retry.LocalEvidence()),
        "approvals": assignment["approvals"], "ordinary_case_bytes_changed": False, "maximum_attempts_changed": False}
    retry.write_once(directory / "manifest.yaml", retry.json_bytes(manifest))
    retry.write_once(directory / "config_resolved.yaml", retry.json_bytes({"runtime_environment": retry.RUNTIME,
        "allocator": retry.ALLOCATOR, "preparation": retry.PREPARATION, "approved_attempt": 4,
        "original_cases": [controller.read_json(controller.read_json(path)["case"]["path"]) for path in sorted(approvals)]}))
    command = [str(arguments.python), str(Path(__file__).resolve()), "--run-dir", str(run),
        "--output-dir", str(directory), "--python", str(arguments.python), "--gpu-count", str(arguments.gpu_count)]
    retry.write_once(directory / "command.txt", (shlex.join(command) + "\n").encode())
    retry.write_once(directory / "experiment_background.md", (
        "## Purpose\n\nRun one explicitly approved fourth attempt for each of five original A4500 comparison cases. "
        "The old GPU allocation failures remain infrastructure failures, with their original files preserved.\n\n"
        "## Key hyperparameters\n\nEvery original case retains two policies, 128 environments per policy, 128 rollout steps, "
        "four epochs, four minibatches, fifty normalization warmup rollouts and seeds one and two. Each of seven alternating "
        "timing pairs contains one complete update per arm. Float32 neural arithmetic, float64 statistics, highest "
        "precision, deterministic XLA operations, disabled preallocation and allocator fraction 0.85 are unchanged.\n\n"
        "## Code and config changes\n\nBefore learner/game imports, each fresh runner allocates, synchronizes and deletes "
        "one nine-GiB uint8 GPU array. It then calls the original D28 case runner with its original checks and timing. "
        "Separate receipts bind actual memory, process, runtime and raw result bytes. The case retry limit stays three; "
        "the receipt grants only attempt four. No fifth attempt or extra hypothesis is granted.\n\n"
        "## Git state\n\nThe numerical source and each operational helper hash are retained in manifest.yaml. "
        "All original cases, checks, failures and attempts remain separate immutable evidence.\n").encode())
    children, handles = [], []
    try:
        with interruption_cleanup():
            for gpu in roster:
                log = (directory / (gpu + ".log")).open("x")
                handles.append(log)
                command = [str(arguments.python), str(Path(__file__).resolve()), "--run-dir", str(run),
                    "--assignment", str(assignment_path), "--worker-gpu-uuid", gpu]
                environment = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
                child = {"process": process, "process_start_time": None}
                children.append(child)
                child["process_start_time"] = controller.process_start_time(process.pid)
                if child["process_start_time"] is None:
                    raise RuntimeError("a retry metadata worker ended before recording its birth counter")
                retry.write_once(directory / (gpu + ".process.json"), retry.json_bytes({"pid": process.pid,
                    "process_start_time": child["process_start_time"], "command": command, "gpu_uuid": gpu,
                    "assigned_cpu_affinity": cpus["gpu_cpu_affinities"][gpu]}))
            while True:
                codes = [(child["process"].pid, child["process"].poll()) for child in children]
                failed = [(pid, code) for pid, code in codes if code not in (None, 0)]
                if failed:
                    raise RuntimeError("receipt-aware metadata workers failed: " + str(failed))
                if all(code is not None for _, code in codes):
                    break
                time.sleep(1)
    except BaseException:
        stop_owned(children)
        raise
    finally:
        for log in handles:
            log.close()


def main():
    """Run one node parent or one assigned metadata worker without changing Slurm allocation fields."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gpu-count", type=int, default=5)
    parser.add_argument("--assignment", type=Path)
    parser.add_argument("--worker-gpu-uuid")
    arguments = parser.parse_args()
    if arguments.worker_gpu_uuid:
        if arguments.assignment is None:
            raise ValueError("an assigned worker requires its original node assignment")
        run_worker(arguments.run_dir, arguments.assignment, arguments.worker_gpu_uuid)
    else:
        if arguments.output_dir is None:
            raise ValueError("the node parent requires a fresh --output-dir")
        run_parent(arguments)


if __name__ == "__main__":
    main()
