"""Validate an explicit H100 NVL placement policy and run ordinary Slurm queue work.

This module creates no policy, prepares no site test, submits no job and changes
no source admission. A separately reviewed policy and a successful fresh site
validation in the current allocation are required before any queue claim.
Controller and final-evidence dispatch must be reviewed before activation.
"""

import argparse
import ast
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import uuid


PREFIX = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"
EXECUTION_KIND = "h100_nvl_slurm_original_case"
HELPERS = ("h100_nvl_slurm_queue.py", "h100_nvl_slurm_case.py", "h100_nvl_slurm_evidence.py", "h100_nvl_slurm_site_validation.py",
    "controller.py", "ideas.py", "admit_measurements.py", "train_optimized.py", "diagnose_gpu_execution.py",
    "post_comparison_validation.py", "infrastructure_retry.py", "pregather_async_retry.py")
TERMINAL_NAMES = ("attempt_result", "unit_result", "attempt_progress", "unit_progress")


def load(name):
    """Load an exact adjacent metadata file without replacing frozen numerical module names."""
    path = Path(__file__).resolve().with_name(name + ".py")
    specification = importlib.util.spec_from_file_location("h100_queue_" + name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def site_module(directory):
    """Verify the historical site's real committed helper bytes before loading its original verifier."""
    local = load("h100_nvl_slurm_site_validation")
    directory = Path(directory).resolve(strict=True)
    manifest = local.read(directory / "manifest.yaml")
    plan = local.bound(manifest["plan"])
    local.verify_source_identity(plan["helpers"])
    path = Path(plan["helpers"]["source_root"]) / PREFIX / "h100_nvl_slurm_site_validation.py"
    expected = local.sha256(Path(__file__).with_name("h100_nvl_slurm_site_validation.py"))
    if plan["helpers"]["files"].get(PREFIX + path.name) != expected or local.sha256(path) != expected:
        raise ValueError("the site's verifier must be the exact reviewed original site-validation implementation")
    specification = importlib.util.spec_from_file_location("h100_queue_original_site", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def validate_policy(path, *, admission=True):
    """Reject absent, expanded or stale permission; retain the exact passed-site UUID scope only."""
    local = load("h100_nvl_slurm_site_validation")
    path = Path(path).resolve(strict=True)
    policy = local.read(path)
    if (policy.get("schema_version") != 1 or policy.get("kind") != "h100_nvl_slurm_placement_policy"
            or policy.get("scope") != "exact_passed_site" or policy.get("decision") != "approved"
            or not isinstance(policy.get("review_rationale"), str) or not policy["review_rationale"].strip()
            or policy.get("runtime_environment") != local.RUNTIME
            or policy.get("source_commit") != local.SOURCE_COMMIT
            or policy.get("ordinary_attempt_limit") != 3 or policy.get("scientific_hypothesis_credit") != 0):
        raise ValueError("an explicit reviewed exact-site placement policy is required")
    reviewed = datetime.fromisoformat(policy["reviewed_at_utc"])
    if reviewed.utcoffset() is None:
        raise ValueError("the placement review timestamp must include its timezone")
    for name in ("case_time_budget_seconds", "shutdown_reserve_seconds"):
        if type(policy.get(name)) is not int or policy[name] < 60:
            raise ValueError("the placement policy must explicitly reserve positive case and shutdown time")
    site = site_module(policy["site_directory"])
    directory, plan = site.prepared(policy["site_directory"])
    if policy["site_result"] != site.receipt(directory / "result.json"):
        raise ValueError("the placement policy changed its actual site result bytes")
    if (directory / "failure.json").exists():
        raise ValueError("a failed site cannot permit queue workers")
    scope = site.verify_result(directory, plan, site.bound(policy["site_result"]))
    if admission:
        site.check_admission(plan["campaign_run_dir"], plan["admission"])
    expected_targets = [{"hostname": scope["hostname"], "gpu_uuid": scope["gpu_uuid"]}]
    if (policy.get("approved_targets") != expected_targets
            or policy.get("campaign_run_dir") != plan["campaign_run_dir"]
            or policy.get("admission") != plan["admission"]):
        raise ValueError("placement permission must cover only the exact passed node/UUID and current admission")
    local.verify_source_identity(policy["worker_helpers"])
    expected_helpers = {PREFIX + name for name in HELPERS}
    if (set(policy["worker_helpers"]["files"]) != expected_helpers
            or Path(__file__).resolve() != Path(policy["worker_helpers"]["source_root"]) / PREFIX / Path(__file__).name):
        raise ValueError("execute only the complete committed worker metadata closure named by the policy")
    return policy, site, plan


def allocation_validation(policy, directory, *, admission=True):
    """Require original full gates and a reference update from this exact current Slurm allocation."""
    site = site_module(directory)
    directory, plan = site.prepared(directory)
    historical = site_module(policy["site_directory"])
    _, historical_plan = historical.prepared(policy["site_directory"])
    compare_site_plans(historical_plan, plan)
    if (directory / "failure.json").exists():
        raise ValueError("the current allocation retained a failed site check")
    verified = site.verify_result(directory, plan, site.read(directory / "result.json"))
    if admission:
        site.check_admission(plan["campaign_run_dir"], policy["admission"])
    target = {"hostname": verified["hostname"], "gpu_uuid": verified["gpu_uuid"]}
    if (target not in policy["approved_targets"] or plan["campaign_run_dir"] != policy["campaign_run_dir"]
            or plan["admission"] != policy["admission"] or verified["job_id"] != os.environ.get("SLURM_JOB_ID")
            or verified["hostname"] != socket.gethostname().split(".")[0]):
        raise ValueError("fresh allocation checks do not cover this owned job and approved physical site")
    # Preserve original physical fields, for example JOB_GPUS='7', even after CUDA becomes GPU-<uuid>.
    assignment = site.read(directory / "site_assignment.json")
    original = assignment["allocation"]["slurm_allocations"]
    if any(os.environ.get(name) != value for name, value in original.items()):
        raise ValueError("the worker no longer has the full verified Slurm GPU allocation")
    site.require_owned_job(policy["campaign_run_dir"], os.environ, Path("/proc/self/cgroup").read_text())
    site.require_runtime(os.environ)
    return site, directory, plan, assignment


def compare_site_plans(reference, allocation):
    """Retain the exact original interpreter, full configuration, software, assets and numerical source at each site check."""
    names = ("campaign_run_dir", "numerical_source", "reference", "packages", "asset_record", "jaxtari_source_root",
        "python", "cleanrl_reference", "configuration", "runtime_environment", "admission")
    if any(reference[name] != allocation[name] for name in names):
        raise ValueError("the fresh allocation changed the complete reviewed reference configuration or software inputs")


def fresh_software(site, plan):
    """Rehash the actual packages, JAXtari Python, GPU images and assets without importing JAX."""
    training = site.load("train_optimized")
    installed = {"jaxtari_source": training.jaxtari_python_receipt(plan["jaxtari_source_root"]),
        "gpu_libraries": training.installed_gpu_library_receipts(), "packages": site.package_versions()}
    assets = training.verify_assets(plan["asset_record"], os.environ)
    if (installed != site.bound(plan["reference"]["shared_installed_backend"])
            or assets != site.bound(plan["reference"]["shared_asset_receipt"])):
        raise ValueError("the allocation's shared software or assets differ from the validated reference")
    return installed, assets


def current_allocation(policy, site, assignment):
    """Resolve the physical card inside the active step and retain every original job/step GPU field."""
    cgroup = Path("/proc/self/cgroup").read_text()
    step = os.environ.get("SLURM_STEP_ID", "")
    if not step.isdecimal() or not re.search(r"(?:^|/)step_" + re.escape(step) + r"(?:/|$)", cgroup, re.MULTILINE):
        raise ValueError("resolve the worker GPU inside its explicit owned srun step")
    owned = site.require_owned_job(policy["campaign_run_dir"], os.environ, cgroup)
    csv_text = site.query(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,driver_version", "--format=csv,noheader,nounits"])
    xml_text = site.query(["nvidia-smi", "-q", "-x"])
    proof = site.resolve_allocation(os.environ, socket.gethostname().split(".")[0], csv_text, xml_text)
    original = assignment["allocation"]
    fields = ("hostname", "gpu_uuid", "job_id", "physical_minor", "driver", "slurm_allocations")
    if (any(proof[name] != original[name] for name in fields)
            or {"hostname": proof["hostname"], "gpu_uuid": proof["gpu_uuid"]} not in policy["approved_targets"]
            or proof["driver"] != site.DRIVER or sorted(os.sched_getaffinity(0)) != assignment["cpu_affinity"]):
        raise ValueError("the active owned GPU/driver/CPU allocation differs from its fresh successful site checks")
    os.environ["CUDA_VISIBLE_DEVICES"] = proof["gpu_uuid"]
    site.require_runtime(os.environ)
    return {"allocation": proof, "owned_job": owned, "step_id": step, "cpu_affinity": sorted(os.sched_getaffinity(0))}


def step_command(policy_path, allocation_directory, output_directory):
    """Describe policy check, fresh full site checks and worker execution inside one explicitly bound srun step."""
    policy, _, _ = validate_policy(policy_path)
    site = site_module(allocation_directory)
    _, plan = site.prepared(allocation_directory)
    python = plan["python"]
    queue_path = str(Path(policy["worker_helpers"]["source_root"]) / PREFIX / "h100_nvl_slurm_queue.py")
    site_path = str(Path(plan["helpers"]["source_root"]) / PREFIX / "h100_nvl_slurm_site_validation.py")
    commands = [[python, queue_path, "validate-policy", "--policy", str(policy_path)],
        [python, site_path, "run", "--directory", str(allocation_directory)],
        [python, queue_path, "run", "--policy", str(policy_path), "--allocation-validation", str(allocation_directory),
         "--output-dir", str(output_directory)]]
    script = "set -eu\n" + "\n".join(shlex.join(command) for command in commands[:-1]) + "\nexec " + shlex.join(commands[-1])
    return ["srun", "--cpu-bind=threads", "bash", "--noprofile", "--norc", "-c", script]


@contextmanager
def stop_after_case():
    """Record ordinary shutdown requests without interrupting a running measured case."""
    requested = []
    previous = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}
    def request(number, frame):
        """Keep the current comparison intact and stop before the next claim."""
        requested.append(number)
    for number in previous:
        signal.signal(number, request)
    try:
        yield requested
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def require_dispatch(policy):
    """Keep this new path inactive until both committed consumers explicitly declare its strict validator."""
    root = Path(policy["worker_helpers"]["source_root"])
    for name in ("controller.py", "post_comparison_validation.py"):
        tree = ast.parse((root / PREFIX / name).read_bytes())
        declarations = [node.value.value for node in tree.body if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "H100_NVL_SLURM_QUEUE_MODULE" for target in node.targets)
            and isinstance(node.value, ast.Constant)]
        if declarations != ["h100_nvl_slurm_queue.py"]:
            raise ValueError("strict H100 Slurm completion dispatch is not yet published in " + name)


def remaining_time(job_id, record, now=None):
    """Read only the owned job's explicit UTC end time; reject unlimited or ambiguous allocation bounds."""
    fields = dict(item.split("=", 1) for item in shlex.split(record) if "=" in item)
    if fields.get("JobId") != job_id or fields.get("JobState") != "RUNNING":
        raise ValueError("remaining-time proof must describe this exact running owned UVA job")
    end = datetime.fromisoformat(fields["EndTime"])
    if end.utcoffset() is None:
        end = end.replace(tzinfo=timezone.utc)
    seconds = (end - (datetime.now(timezone.utc) if now is None else now)).total_seconds()
    if not math.isfinite(seconds):
        raise ValueError("Slurm did not provide a finite remaining allocation time")
    return seconds


def claim_next(policy_path, identity, allocation_record, seconds_left):
    """Claim one ordinary H100 case under the current source lock, preserving bytes and the original limit."""
    policy, site, plan = validate_policy(policy_path)
    require_dispatch(policy)
    controller = load("controller")
    run = Path(policy["campaign_run_dir"])
    if isinstance(seconds_left, bool) or not isinstance(seconds_left, (int, float)) or not math.isfinite(seconds_left):
        raise ValueError("the claim requires a finite measured remaining walltime")
    if seconds_left < policy["case_time_budget_seconds"] + policy["shutdown_reserve_seconds"]:
        return None
    with controller.queue_lock(run):
        # Recheck receipts after taking the source lock; a policy/admission change must stop a waiting worker.
        if site.receipt(policy_path) != identity["placement_policy"]:
            raise ValueError("the placement policy changed before claim")
        for item in policy["admission"]["files"]:
            site.bound(item)
        ready = controller.ready_specification(run)
        if ready is None or (ready["source_root"], ready["source_commit"]) != (plan["numerical_source"]["source_root"], site.SOURCE_COMMIT):
            raise ValueError("the current source admission no longer permits this worker")
        for path in sorted((run / "queue/pending").glob("*.json")):
            case = controller.read_json(path)
            if (case["hardware_class"] != "h100" or case["source_root"] != ready["source_root"]
                    or case["source_commit"] != site.SOURCE_COMMIT):
                continue
            if (case["maximum_attempts"] != 3 or case.get("reference_environment", {})
                    or case.get("reference_execution_baseline", {})):
                raise ValueError("this worker accepts only original ordinary cases with unchanged attempt limits")
            specification = run / "queue/specifications" / path.name
            if path.read_bytes() != specification.read_bytes() or controller.case_states(run, case["unit_id"]) != ["pending"]:
                raise ValueError("the immutable pending case changed or occupies multiple queue states")
            supplied = {name: value for name, value in case.items() if name != "specification_digest"}
            if controller.content_digest(supplied) != case["specification_digest"]:
                raise ValueError("the ordinary case specification digest changed")
            lease_path = run / "queue/leases" / path.name
            previous = controller.read_json(lease_path) if lease_path.exists() else None
            attempt = previous["attempt"] + 1 if previous else 1
            if attempt > case["maximum_attempts"]:
                raise ValueError("the ordinary case exhausted its original explicit retry limit")
            controller.verify_source(case["source_root"], case["source_commit"])
            _, attempt_dir = controller.attempt_paths(run, case["unit_id"], attempt)
            lease = {**identity, "execution_kind": EXECUTION_KIND, "allocation_validation": allocation_record,
                "token": uuid.uuid4().hex, "unit_id": case["unit_id"], "attempt": attempt, "state": "claimed",
                "claimed_at": controller.utc_now(), "runner_pid": None, "runner_start_time": None}
            # Retain original transaction order: an interrupted rename without a lease remains recoverable.
            path.rename(run / "queue/running" / path.name)
            controller.atomic_json(lease_path, lease)
            attempt_dir.mkdir(parents=True, exist_ok=False)
            controller.atomic_json(attempt_dir / "lease.json", lease)
            return case, lease
    return None


def requires_execution(case, lease, hardware):
    """Detect this Slurm path even if an optional lease marker was removed."""
    return (lease.get("execution_kind") == EXECUTION_KIND or
        (case.get("hardware_class") == "h100" and (bool(hardware.get("slurm_job_id"))
         or hardware.get("hostname", "").split(".")[0] in load("h100_nvl_slurm_site_validation").HOSTS)))


def terminal_paths(unit, attempt):
    """Name the four and only four canonical terminal files allowed to be withheld before proof."""
    return {"attempt_result": attempt / "result.json", "unit_result": unit / "result.json",
        "attempt_progress": attempt / "progress.json", "unit_progress": unit / "progress.json"}


def validate_execution(attempt, case, lease=None, *, require_completed=False, evidence=None):
    """Validate captured raw execution evidence without depending on current admission or reader packages."""
    return load("h100_nvl_slurm_evidence").validate_execution(
        attempt, case, lease, require_completed=require_completed, evidence=evidence)


def durable_directory(path):
    """Synchronize metadata directory entries before publishing a terminal queue rename."""
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_bytes(path, contents):
    """Replace a canonical metadata file with the unchanged original serialized bytes and sync its directory."""
    temporary = path.with_name(path.name + ".h100_slurm_" + uuid.uuid4().hex)
    with temporary.open("xb") as stream:
        stream.write(contents)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    durable_directory(path.parent)


def finish_claim(run, case, lease, exit_code):
    """Publish original terminal bytes only after a zero exit and verified same-process library evidence."""
    controller = load("controller")
    site = load("h100_nvl_slurm_site_validation")
    run = Path(run)
    unit, attempt = controller.attempt_paths(run, case["unit_id"], lease["attempt"])
    complete = False
    failure_reason = "the original child did not exit successfully"
    if exit_code == 0:
        try:
            execution_receipt = site.receipt(attempt / "slurm_execution.json")
            terminal_receipt = site.receipt(attempt / "slurm_child_terminal.json")
            record, result = validate_execution(attempt, case, lease)
            complete = True
        except (ValueError, KeyError, FileNotFoundError) as error:
            failure_reason = str(error)
    with controller.queue_lock(run):
        lease_path = run / "queue/leases" / (case["unit_id"] + ".json")
        current = controller.read_json(lease_path)
        if current["token"] != lease["token"] or current["state"] not in ("running", "completed"):
            raise ValueError("the current running lease no longer belongs to this exact child")
        if current["state"] == "completed" and not complete:
            raise ValueError("an already completed publication cannot be replaced with a different unsuccessful child outcome")
        if complete:
            if (site.receipt(attempt / "slurm_execution.json") != execution_receipt
                    or site.receipt(attempt / "slurm_child_terminal.json") != terminal_receipt):
                raise ValueError("the successful child proof changed before terminal publication")
            for item in site.bound(lease["placement_policy"])["admission"]["files"]:
                site.bound(item)
            # Read and verify every original serialized file under the lock before publishing any terminal bytes.
            contents = {name: Path(item["path"]).read_bytes() for name, item in record["terminal_files"].items()}
            if any(hashlib.sha256(contents[name]).hexdigest() != item["sha256"] for name, item in record["terminal_files"].items()):
                raise ValueError("a staged terminal file changed before publication")
            intent = {"kind": "h100_nvl_slurm_terminal_publication", "token": lease["token"], "attempt": lease["attempt"],
                "unit_id": case["unit_id"], "execution": execution_receipt, "child_terminal": terminal_receipt,
                "terminal_files": {name: item["sha256"] for name, item in record["terminal_files"].items()}}
            intent_path = attempt / "slurm_publication_intent.json"
            continuing = intent_path.exists()
            if continuing and site.read(intent_path) != intent:
                raise ValueError("an earlier terminal publication intent belongs to different proof or bytes")
            if not continuing:
                site.write_once(intent_path, intent)
            if current["state"] == "completed":
                expected_paths = terminal_paths(unit, attempt)
                if any(not path.exists() or path.read_bytes() != contents[name] for name, path in expected_paths.items()):
                    raise ValueError("completed canonical files differ from their unchanged publication intent")
                if controller.read_json(attempt / "lease.json") != current:
                    raise ValueError("the two preserved completed leases differ")
                states = controller.case_states(run, case["unit_id"])
                if states == ["running"]:
                    running = run / "queue/running" / (case["unit_id"] + ".json")
                    if running.read_bytes() != (run / "queue/specifications" / running.name).read_bytes():
                        raise ValueError("the interrupted publication changed its immutable running case")
                    running.rename(run / "queue/completed" / running.name)
                    durable_directory(running.parent)
                    durable_directory(run / "queue/completed")
                elif states != ["completed"]:
                    raise ValueError("terminal publication must occupy exactly one running or completed queue state")
                return "completed"
            # Original payload bytes are copied unchanged: staged attempt_result -> canonical result.json.
            for name, destination in terminal_paths(unit, attempt).items():
                if destination.name == "result.json" and destination.exists():
                    saved = site.read(destination)
                    if destination.parent == attempt or saved.get("attempt", lease["attempt"]) >= lease["attempt"]:
                        if not continuing or destination.read_bytes() != contents[name]:
                            raise ValueError("a canonical current-attempt result lacks its matching publication intent")
                publish_bytes(destination, contents[name])
            destination_state = "completed"
        else:
            failure_path = attempt / "failure.json"
            failure = (controller.read_json(failure_path) if failure_path.exists() else
                {"case_completed": False, "failure_stage": "h100_slurm_execution_proof", "retryable": True,
                 "error": failure_reason, "exit_code": exit_code, "attempt": lease["attempt"], "time": controller.utc_now()})
            if not failure_path.exists():
                controller.atomic_json(failure_path, failure)
            controller.atomic_json(unit / "failure.json", failure)
            controller.atomic_json(unit / "progress.json", {**failure, "state": "failed"})
            destination_state = "failed"
        current.update(state=destination_state, runner_exit_code=exit_code, finished_at=controller.utc_now())
        controller.atomic_json(attempt / "lease.json", current)
        controller.atomic_json(lease_path, current)
        durable_directory(attempt)
        durable_directory(lease_path.parent)
        running = run / "queue/running" / (case["unit_id"] + ".json")
        running.rename(run / "queue" / destination_state / running.name)
        durable_directory(running.parent)
        durable_directory(run / "queue" / destination_state)
    return destination_state


def run_child(run, case, lease, policy_path, allocation_directory, lock_fd):
    """Keep the per-card lock through exact-child cleanup and record the PID before original-runner release."""
    site = load("h100_nvl_slurm_site_validation")
    controller = load("controller")
    _, attempt = controller.attempt_paths(run, case["unit_id"], lease["attempt"])
    command = [sys.executable, str(Path(__file__).with_name("h100_nvl_slurm_case.py")), "--policy", str(policy_path),
        "--allocation-validation", str(allocation_directory), "--case", str(Path(run) / "queue/specifications" / (case["unit_id"] + ".json")),
        "--attempt", str(lease["attempt"]), "--lease-token", lease["token"]]
    site.write_once(attempt / "slurm_command.json", command)
    with (attempt / "runner.log").open("x") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=case["source_root"],
            start_new_session=True, pass_fds=(lock_fd,))
        birth = None
        try:
            birth = site.process_start_time(process.pid)
            if birth is None:
                raise RuntimeError("the numerical child ended before its birth counter was recorded")
            owner = {"pid": process.pid, "process_start_time": birth, "parent_pid": lease["worker_pid"],
                "parent_start_time": lease["worker_start_time"], "job_id": lease["slurm_job_id"]}
            site.write_once(attempt / "slurm_child_owner.json", owner)
            with controller.queue_lock(run):
                path = Path(run) / "queue/leases" / (case["unit_id"] + ".json")
                if controller.read_json(path)["token"] != lease["token"]:
                    raise ValueError("the lease changed before numerical child release")
                lease.update(runner_pid=process.pid, runner_start_time=birth, state="running")
                controller.atomic_json(attempt / "lease.json", lease)
                controller.atomic_json(path, lease)
            code = process.wait()
            site.write_once(attempt / "slurm_child_terminal.json", {"owner": site.receipt(attempt / "slurm_child_owner.json"),
                "exit_code": code, "ended_at_utc": controller.utc_now(), "log": site.receipt(attempt / "runner.log"),
                "absent_files": [str(attempt / "failure.json")] if not (attempt / "failure.json").exists() else []})
        except BaseException:
            site.stop_child(process, birth)
            raise
    return finish_claim(run, case, lease, code)


def run_worker(policy_path, allocation_directory, output_directory):
    """Run an explicitly permitted single-card worker; refuse every check before the first claim if absent."""
    policy, _, _ = validate_policy(policy_path)
    require_dispatch(policy)
    site, allocation_directory, plan, assignment = allocation_validation(policy, allocation_directory)
    output = Path(output_directory).resolve()
    if output.exists() or output.parent != Path(policy["campaign_run_dir"]) / "remote_setup":
        raise ValueError("the metadata worker needs a new explicit output directory under remote_setup")
    proof = assignment["allocation"]
    if str(Path(sys.executable).absolute()) != plan["python"]:
        raise ValueError("the worker must use the validated exact Python interpreter")
    os.sched_setaffinity(0, assignment["cpu_affinity"])
    if sorted(os.sched_getaffinity(0)) != assignment["cpu_affinity"]:
        raise ValueError("the worker must hold the exact verified eight CPUs")
    os.environ["CUDA_VISIBLE_DEVICES"] = proof["gpu_uuid"]
    initial_allocation = current_allocation(policy, site, assignment)
    identity = {"worker_id": uuid.uuid4().hex, "worker_pid": os.getpid(), "worker_start_time": site.process_start_time(os.getpid()),
        "hostname": proof["hostname"], "slurm_job_id": proof["job_id"], "hardware_class": "h100", "gpu_uuid": proof["gpu_uuid"],
        "gpu_name": site.MODEL, "cpu_affinity": assignment["cpu_affinity"], "placement_policy": site.receipt(policy_path)}
    output.mkdir()
    site.write_once(output / "manifest.yaml", {"worker": identity, "allocation_validation": site.receipt(allocation_directory / "result.json"),
        "source_commit": site.SOURCE_COMMIT, "helpers": policy["worker_helpers"], "runtime_environment": site.RUNTIME,
        "allocation_at_worker_start": initial_allocation})
    site.write_once(output / "config_resolved.yaml", policy)
    site.write_once(output / "command.txt", shlex.join([sys.executable, str(Path(__file__).resolve()), "run", "--policy", str(policy_path),
        "--allocation-validation", str(allocation_directory), "--output-dir", str(output)]) + "\n")
    site.write_once(output / "experiment_background.md", "# Purpose\nRun reviewed original H100 Slurm comparisons.\n\n# Key hyperparameters\nUse each immutable case without changes.\n\n# Code and config changes\nAdd allocation and successful-child evidence; preserve numerical source.\n\n# Git state\n" + json.dumps(policy["worker_helpers"]) + "\n")
    controller = load("controller")
    run = Path(policy["campaign_run_dir"])
    with (Path("/tmp") / ("sl5nw_atari_pixel_benchmark_" + proof["gpu_uuid"] + ".lock")).open("a+") as lock, stop_after_case() as stop:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            if stop:
                controller.atomic_json(output / "result.json", {"state": "stopped_after_case", "requested_signals": stop})
                return
            current_allocation(policy, site, assignment)
            site.require_selected_card(proof["gpu_uuid"])
            fresh_software(site, plan)
            validate_policy(policy_path)
            if stop:
                continue
            # Query only this owned UVA job; TZ=UTC makes Slurm's unzoned EndTime unambiguous.
            raw = subprocess.check_output(["scontrol", "show", "job", proof["job_id"], "-o"], text=True,
                timeout=20, env={**os.environ, "TZ": "UTC"})
            seconds = remaining_time(proof["job_id"], raw)
            claimed = claim_next(policy_path, identity, site.receipt(allocation_directory / "result.json"), seconds)
            if claimed is None:
                controller.atomic_json(output / "result.json", {"state": "idle_or_insufficient_walltime", "remaining_seconds": seconds})
                return
            case, lease = claimed
            controller.atomic_json(run / "queue/workers" / (identity["worker_id"] + ".json"),
                {**identity, "state": "running", "unit_id": case["unit_id"], "attempt": lease["attempt"], "time": controller.utc_now()})
            run_child(run, case, lease, policy_path, allocation_directory, lock.fileno())
            controller.atomic_json(run / "queue/workers" / (identity["worker_id"] + ".json"), {**identity, "state": "idle", "time": controller.utc_now()})


def main():
    """Validate an existing policy or execute an already admitted worker; never create permission or jobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validation = commands.add_parser("validate-policy")
    validation.add_argument("--policy", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--policy", type=Path, required=True)
    run.add_argument("--allocation-validation", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "validate-policy":
        validate_policy(arguments.policy)
        print(json.dumps({"valid": True, "scope": "exact_passed_site", "submitted": False, "claimed": False}))
    else:
        run_worker(arguments.policy, arguments.allocation_validation, arguments.output_dir)


if __name__ == "__main__":
    main()
