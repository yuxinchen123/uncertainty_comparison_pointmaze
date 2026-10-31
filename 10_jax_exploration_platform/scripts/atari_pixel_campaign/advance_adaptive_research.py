"""Prepare and explicitly activate a five-idea admission revision without numerical imports.

Preparation preserves original bytes and changes no live admission, registry or
case. Activation is a separate command; it never starts workers or chooses ideas.
Real preparation recomputes measured statistics and requires an owned Slurm
compute allocation. Only synthetic standard-library tests run on a login host.
"""

import argparse
import ast
import copy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import admit_measurements as admission
import controller
import diagnose_gpu_execution as diagnostic
import ideas
import post_comparison_validation as validation


SCRIPT_PREFIX = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"
ACCEPTANCE_CRITERIA = {"minimum_paired_rounds": 7,
    "median_speedup_strictly_greater_than": 1.0,
    "maximum_one_sided_sign_probability": 0.05,
    "require_correctness": True, "reject_contamination": True}
TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
                   "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}
LAUNCH_PROTOCOL = "queue-metadata-lock-through-source-check-and-submission-v1"


def json_bytes(value):
    """Serialize staged metadata deterministically without changing copied original bytes."""
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def sha256(contents):
    """Identify exact file bytes independently of JSON whitespace."""
    return hashlib.sha256(contents).hexdigest()


def write_once(path, contents):
    """Publish complete immutable bytes exclusively, accepting only an identical retry."""
    path = Path(path)
    if path.exists():
        if path.read_bytes() != contents:
            raise ValueError("immutable revision artifact changed: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".admission-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink()


def replace_bytes(path, contents):
    """Atomically replace a live file only after complete bytes reach disk."""
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=".admission-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def normalized_name(value):
    """Compare names and mechanism descriptions without case or separator differences."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("hypothesis names and descriptions must be nonempty text")
    return " ".join(re.sub(r"[_\W]+", " ", value.casefold()).split())


def copy_counts(value):
    """Require an explicit distinct matched-count list including one policy."""
    if (not isinstance(value, list) or not value or 1 not in value
            or any(type(count) is not int or count < 1 for count in value)
            or len(value) != len(set(value))):
        raise ValueError("matched_copy_counts must be distinct positive integers including one")
    return sorted(value)


def source_identity(root, commit):
    """Require a clean detached committed numerical source, including its tests and runners."""
    root = Path(root).resolve(strict=True)
    controller.verify_source(root, commit)
    branch = subprocess.run(["git", "-C", str(root), "symbolic-ref", "--quiet", "HEAD"],
                            capture_output=True, text=True, check=False)
    changed = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain",
                                       "--untracked-files=all"], text=True).strip()
    if branch.returncode != 1 or changed:
        raise ValueError("the new numerical source must be a clean detached committed checkout")
    return diagnostic.core_identity(root, commit)


def launcher_capability(evidence):
    """Require the committed launcher to serialize validation, submission and ledger publication."""
    path = Path(__file__).resolve().parent / "launch_queue.py"
    contents = evidence.file(path)
    tree = ast.parse(contents)
    markers = [ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == "SOURCE_ADMISSION_LOCK_PROTOCOL"
                       for target in node.targets)]
    main = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"]
    if markers != [LAUNCH_PROTOCOL] or len(main) != 1:
        raise ValueError("worker launcher lacks the reviewed source-admission lock capability")
    guarded = [node for node in ast.walk(main[0]) if isinstance(node, ast.With)
               and any(isinstance(item.context_expr, ast.Call)
                       and isinstance(item.context_expr.func, ast.Name)
                       and item.context_expr.func.id == "source_admission_lock" for item in node.items)]
    def calls(node):
        """Find source validation and submission calls without executing a launcher."""
        return [item for item in ast.walk(node) if isinstance(item, ast.Call)
                and isinstance(item.func, ast.Name)
                and item.func.id in {"admitted_source", "launch_direct", "launch_slurm"}]
    if len(guarded) != 1 or {item.func.id for item in calls(guarded[0])} != {
            "admitted_source", "launch_direct", "launch_slurm"} or calls(main[0]) != calls(guarded[0]):
        raise ValueError("worker launcher source validation and complete launch batch must share the admission lock")
    root = path.parents[3]
    relative = str(path.relative_to(root))
    original = subprocess.check_output(["git", "-C", str(root), "show", "HEAD:" + relative])
    if contents != original:
        raise ValueError("source-admission launcher capability must be committed before preparing a revision")


def committed_files(receipt, evidence, required=(), require_worktree=False):
    """Verify a receipt's exact Git blobs and staged files without importing their code."""
    root, commit, files = Path(receipt["source_root"]), receipt["source_commit"], receipt["files"]
    if not root.is_absolute() or not re.fullmatch(r"[0-9a-f]{40}", commit) or not set(required) <= set(files):
        raise ValueError("source/helper receipt lacks its committed required files")
    for relative, expected in files.items():
        if not relative.startswith(SCRIPT_PREFIX) or ".." in Path(relative).parts:
            raise ValueError("a helper receipt must name campaign source files")
        original = subprocess.check_output(["git", "-C", str(root), "show", f"{commit}:{relative}"])
        if sha256(original) != expected or (require_worktree and evidence.file(root / relative) != original):
            raise ValueError("a producer/helper differs from its committed receipt")


def checked_configuration(value, root):
    """Validate complete source fields and performance values while fixing the game and learner."""
    value = validation.configuration(value)
    defaults = diagnostic.default_configuration(root)
    if any(set(value[scope]) != set(defaults[scope]) for scope in ("learner", "environment")):
        raise ValueError("a baseline must explicitly contain every pinned learner/environment field")
    tree = ast.parse((Path(root) / "10_jax_exploration_platform/src/exploration_platform/atari_pixels/config.py").read_text())
    choices = [ast.literal_eval(node.value) for node in ast.walk(tree) if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == "choices" for target in node.targets)]
    if len(choices) != 1:
        raise ValueError("the source must declare explicit learner performance choices")
    for name, default in defaults["learner"].items():
        setting = value["learner"][name]
        if name.startswith("per_copy_"):
            if setting:
                raise ValueError("adaptive infrastructure admits no per-policy hyperparameter settings")
        elif type(default) is bool:
            if type(setting) is not bool:
                raise ValueError("learner Boolean settings require JSON booleans: " + name)
        elif type(default) is int:
            if type(setting) is not int or setting < 1:
                raise ValueError("learner counts must be positive integers: " + name)
        elif type(default) is float:
            if type(setting) not in (int, float) or not math.isfinite(setting):
                raise ValueError("learner numeric settings must be finite: " + name)
        elif name not in choices[0] or setting not in choices[0][name] or type(setting) is not str:
            raise ValueError("learner performance setting has no supported value: " + name)
    environment = {name: setting for name, setting in value["environment"].items()
                   if setting != defaults["environment"][name]}
    for name, default in defaults["environment"].items():
        setting = value["environment"][name]
        if type(default) in (bool, int, str) and type(setting) is not type(default):
            raise ValueError("environment setting has the wrong exact type: " + name)
        if type(default) is float and (type(setting) not in (int, float) or not math.isfinite(setting)):
            raise ValueError("environment numeric setting must be finite: " + name)
    controller.reference_environment_settings(environment, root)
    controller.reference_execution_settings(value["execution"], root)
    return value


def actual_runtime(value, expected):
    """Require observed full precision and disabled persistent compilation caches in the numerical process."""
    if (not isinstance(value, dict) or validation.runtime_environment({key: value.get(key) for key in validation.RUNTIME_KEYS}) != expected
            or value.get("JAX_ENABLE_COMPILATION_CACHE") != "false"
            or value.get("JAX_COMPILATION_CACHE_DIR") not in (None, "")
            or value.get("JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES") != ""):
        raise ValueError("actual numerical runtime lacks the required precision or disabled persistent-cache evidence")


def completed_cohorts(registry, evidence):
    """Validate every initial and previously chosen result at its original matched counts."""
    if len(registry["initial_ideas"]) != 30 or len(registry["adaptive_batches"]) != 4:
        raise ValueError("the study requires thirty initial ideas and four adaptive batches")
    if [(row.get("position"), row.get("name")) for row in registry["initial_ideas"]] != [
            (entry.position, entry.name) for entry in ideas.INITIAL_IDEAS]:
        raise ValueError("initial coverage must retain every original catalog identity and position")
    if [batch.get("batch") for batch in registry["adaptive_batches"]] != [1, 2, 3, 4]:
        raise ValueError("the four adaptive batches must retain their original order")
    cohorts = [(registry["initial_ideas"], copy_counts(registry.get("initial_matched_copy_counts")),
                registry["reference_commit"])]
    target = None
    for batch in registry["adaptive_batches"]:
        entries = batch["hypotheses"]
        if len(entries) != 5:
            raise ValueError("each adaptive batch must retain exactly five slots")
        if [entry.get("position") for entry in entries] != list(range(26 + batch["batch"] * 5, 31 + batch["batch"] * 5)):
            raise ValueError("adaptive slots must retain their original five positions")
        names = [entry.get("name") for entry in entries]
        if all(name is None for name in names):
            if target is None:
                target = batch["batch"]
        elif target is not None or any(name is None for name in names):
            raise ValueError("adaptive batches must be chosen completely and in order")
        else:
            cohorts.append((entries, copy_counts(batch.get("matched_copy_counts")), batch["reference_commit"]))
    if target is None:
        raise ValueError("all twenty adaptive slots are already registered")
    for entries, counts, source in cohorts:
        for entry in entries:
            for context in validation.CONTEXTS:
                for copies in counts:
                    records = [row for row in entry.get("results", {}).get(context, []) if row.get("copies") == copies]
                    if not records:
                        raise ValueError(f"unfinished prior measurement: {entry['name']} / {context} / {copies}")
                    for record in records:
                        validation.validate_cohort_case(record["result_path"], entry["name"], context, copies, source, evidence)
    return target


def queue_is_empty(run_dir):
    """Reject active, failed or partially retained claims before changing source admission."""
    queue = Path(run_dir) / "queue"
    if any(list((queue / state).glob("*.json")) for state in ("pending", "running", "failed")):
        raise ValueError("source admission cannot change with pending, running or failed cases")
    if any(controller.read_json(path).get("state") in {"claimed", "running"}
           for path in (queue / "leases").glob("*.json")):
        raise ValueError("source admission cannot change while a case lease is active")


def observed_time(value):
    """Require an explicit timezone-aware observation time in terminal-process receipts."""
    if not isinstance(value, str):
        raise ValueError("worker-stop evidence needs an observation time")
    moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if moment.tzinfo is None:
        raise ValueError("worker-stop observation times must include their timezone")
    return moment


def stopped_workers(run_dir, receipt_path, evidence):
    """Require exact dead-process or owned terminal-job proof for every old worker record."""
    receipts = evidence.read(receipt_path)
    if receipts.get("schema_version") != 1 or any(not isinstance(receipts.get(key), list)
            for key in ("workers", "jobs", "processes", "runners")):
        raise ValueError("worker-stop receipt schema_version must be one")
    # Before: a pending job has no worker file; after: its exact submitted row still requires terminal proof.
    run_dir = Path(run_dir)
    submissions = run_dir / "slurm/submissions.tsv"
    rows = [line for line in evidence.file(submissions).decode().splitlines()
            if len(line.split("\t")) == 4 and line.split("\t")[1] == "queue-worker"]
    jobs = {row["job_id"]: row for row in receipts["jobs"]}
    if len(jobs) != len(receipts["jobs"]) or set(jobs) != {line.split("\t")[0] for line in rows}:
        raise ValueError("terminal receipts must cover all submitted queue-worker jobs, including unstarted jobs")
    for line in rows:
        job = jobs[line.split("\t")[0]]
        if job.get("submission_row") != line or job.get("state") not in TERMINAL_STATES:
            raise ValueError("queue-worker job proof changed its exact owned launch row or is not terminal")
        observed_time(job.get("observed_at"))
    launches = evidence.read(run_dir / "processes.json")
    launches = [row for row in launches if row.get("kind") == "queue_worker"]
    direct = {(row["hostname"], row["pid"], row["start_time"]): row for row in receipts["processes"]}
    if len(direct) != len(receipts["processes"]) or set(direct) != {
            (row["hostname"], row["pid"], row["start_time"]) for row in launches}:
        raise ValueError("dead-process receipts must cover every recorded direct queue-worker launch")
    for launch in launches:
        proof = direct[launch["hostname"], launch["pid"], launch["start_time"]]
        if (proof.get("original_process_alive") is not False
                or proof.get("launch_record_digest") != controller.content_digest(launch)
                or observed_time(proof.get("observed_at")) < observed_time(launch["time"])):
            raise ValueError("direct supervisor lacks an exact later dead-original-process observation")
    workers = [evidence.read(path) for path in sorted((Path(run_dir) / "queue/workers").glob("*.json"))]
    indexed = {row["worker_id"]: row for row in receipts["workers"]}
    if len(indexed) != len(receipts["workers"]) or set(indexed) != {row["worker_id"] for row in workers}:
        raise ValueError("worker-stop evidence must cover each exact old worker once")
    for worker in workers:
        proof = indexed[worker["worker_id"]]
        keys = ("hostname", "worker_pid", "worker_start_time", "slurm_job_id")
        if any(proof.get(key) != worker.get(key) for key in keys):
            raise ValueError("worker-stop receipt does not match the original PID, birth and job")
        if observed_time(proof.get("observed_at")) < observed_time(worker["time"]):
            raise ValueError("worker-stop evidence predates the last worker metadata")
        if proof.get("kind") == "dead_original_process":
            if proof.get("original_process_alive") is not False:
                raise ValueError("original worker process was not observed dead")
        elif proof.get("kind") == "terminal_slurm_job":
            ledger = Path(proof["ledger_path"]).resolve(strict=True)
            if ledger.parent != Path(run_dir) / "slurm" or not re.fullmatch(r"submitted_jobids[^/]*\.txt", ledger.name):
                raise ValueError("worker terminal evidence must use this campaign's exact job ledger")
            if str(worker["slurm_job_id"]) not in evidence.file(ledger).decode().splitlines():
                raise ValueError("worker terminal evidence names a job outside the owned ledger")
            if proof.get("state") not in TERMINAL_STATES or not worker["slurm_job_id"]:
                raise ValueError("worker job is not terminal")
            if worker["slurm_job_id"] not in jobs or proof["state"] != jobs[worker["slurm_job_id"]]["state"]:
                raise ValueError("worker terminal proof differs from the exact submitted-job observation")
        else:
            raise ValueError("worker-stop evidence must prove a terminal job or dead original process")
    # A direct parent may exit while its child survives, so check original numerical child identities too.
    leases = [evidence.read(path) for path in sorted((run_dir / "queue/leases").glob("*.json"))]
    leases = [row for row in leases if row.get("runner_pid") is not None]
    children = {(row["unit_id"], row["attempt"]): row for row in receipts["runners"]}
    if len(children) != len(receipts["runners"]) or set(children) != {(row["unit_id"], row["attempt"]) for row in leases}:
        raise ValueError("stop evidence must cover every recorded original numerical child")
    for lease in leases:
        proof = children[lease["unit_id"], lease["attempt"]]
        if any(proof.get(key) != lease.get(key) for key in (
                "hostname", "runner_pid", "runner_start_time", "slurm_job_id")):
            raise ValueError("numerical child stop proof changed its original PID, birth or job")
        observed = observed_time(proof.get("observed_at"))
        for key in ("claimed_at", "finished_at"):
            if key in lease and observed < observed_time(lease[key]):
                raise ValueError("numerical child stop observation predates its recorded execution")
        terminal = (proof.get("kind") == "terminal_slurm_job" and lease.get("slurm_job_id") in jobs
                    and proof.get("state") == jobs[lease["slurm_job_id"]]["state"])
        dead = proof.get("kind") == "dead_original_process" and proof.get("original_process_alive") is False
        if not (terminal or dead):
            raise ValueError("an original numerical child has no terminal-job or dead-process observation")
    return workers


def producer_record(path, mode, run_dir, root, commit, runtime, context, core, evidence):
    """Bind a selected-measurement result to its prepared committed producer and exact source."""
    path = Path(path).resolve(strict=True)
    if path.name != "result.json" or path.parent.parent != Path(run_dir) / "data":
        raise ValueError("selected evidence must be a direct measurement unit in this campaign")
    result, plan = evidence.read(path), evidence.read(path.parent / "measurement_plan.json")
    prepared = evidence.read(path.parent / "measurement_prepared.json")
    if prepared != {"plan_sha256": sha256(evidence.file(path.parent / "measurement_plan.json"))}:
        raise ValueError("measurement preparation does not match its immutable plan")
    expected = {"schema_version": 1, "kind": "selected_configuration_measurement", "mode": mode,
                "source_root": str(root), "source_commit": commit, "hardware_class": context,
                "campaign_run_dir": str(run_dir), "unit_id": path.parent.name,
                "compilation_cache_policy": "disabled"}
    if any(plan.get(key) != value for key, value in expected.items()) or plan.get("core") != core:
        raise ValueError("measurement plan used another source, context, mode or cache protocol")
    if result.get("measurement_mode") != mode or result.get("helpers") != plan["helpers"]:
        raise ValueError("result does not identify its original measurement mode and helpers")
    validation.identity(result, context, commit, runtime)
    if validation.runtime_environment(plan["runtime_environment"]) != runtime:
        raise ValueError("measurement plan runtime differs from the admitted full-precision runtime")
    committed_files(plan["paired_source"], evidence,
                    [SCRIPT_PREFIX + name for name in ("paired_benchmark.py", "ideas.py")], require_worktree=True)
    if (plan["paired_source"]["source_root"], plan["paired_source"]["source_commit"]) != (str(root), commit):
        raise ValueError("measurement Executor source is not the new detached source")
    committed_files(plan["helpers"], evidence, [SCRIPT_PREFIX + name for name in (
        "measure_selected.py", "diagnose_gpu_execution.py", "post_comparison_validation.py", "train_optimized.py")])
    for relative, expected_hash in plan["helpers"]["files"].items():
        if sha256(evidence.file(path.parent / "code" / Path(relative).name)) != expected_hash:
            raise ValueError("the executed staged helper differs from its committed receipt")
    selected = checked_configuration(plan["configuration"], root)
    iterations, minimum = plan.get("iterations"), plan.get("minimum_updates_per_round")
    if (plan.get("rounds") != 7 or type(iterations) is not int or type(minimum) is not int
            or minimum < 10 or iterations < minimum
            or result.get("protocol") != {"rounds": 7, "requested_minimum_updates_per_round": minimum,
                                           "actual_updates_per_round": iterations}):
        raise ValueError("selected measurements must bind seven complete ten-or-more-update windows to their plan")
    if mode == "packing-point":
        if plan["reference_configuration"] is not None or result.get("configuration") != selected:
            raise ValueError("isolated capacity retained another variant or changed its configuration")
    elif (result.get("candidate") != selected
          or result.get("reference") != checked_configuration(plan["reference_configuration"], root)):
        raise ValueError("comparison configurations differ from the immutable producer plan")
    configurations = [selected] + ([plan["reference_configuration"]] if mode != "packing-point" else [])
    multiple = 1
    for configuration in configurations:
        execution = configuration["execution"]
        multiple = math.lcm(multiple, execution["iterations_per_call"])
        if execution["report_readout"] != "none":
            multiple = math.lcm(multiple, execution["report_every_iterations"])
    rounds = result.get("rounds", [])
    rows = [pair[arm] for pair in rounds for arm in ("reference", "candidate")] if mode == "confirmation" else rounds
    if (iterations % multiple or (mode != "correctness" and len(rounds) != 7)
            or any(row.get("iterations") != iterations for row in rows)):
        raise ValueError("actual measured rounds differ from the complete execution/report windows in the plan")
    process = validation.execution_identity(result)
    hardware = evidence.read(path.parent / "hardware.json")
    actual_runtime(hardware.get("runtime_environment"), runtime)
    if (hardware.get("hostname"), hardware.get("pid"), hardware.get("process_start_time")) != process:
        raise ValueError("measurement hardware differs from its actual process execution")
    if hardware.get("gpu_uuid") != plan.get("gpu_uuid") or result.get("copy_seeds") != plan.get("copy_seeds"):
        raise ValueError("measurement GPU or independent policy seeds differ from their plan")
    return selected, process


def profile_record(path, run_dir, root, commit, runtime, context, baseline, core, evidence):
    """Validate a versioned selected-baseline profile and actual positive CUDA summary rows."""
    path = Path(path).resolve(strict=True)
    if path.name != "result.json" or path.parent.parent != Path(run_dir) / "data":
        raise ValueError("profile must belong to a direct measurement unit in this campaign")
    result, plan = evidence.read(path), evidence.read(path.parent / "measurement_plan.json")
    expected = {"schema_version": 1, "kind": "selected_configuration_profile",
                "source_root": str(root), "source_commit": commit, "hardware_class": context,
                "campaign_run_dir": str(run_dir), "unit_id": path.parent.name,
                "compilation_cache_policy": "disabled"}
    if any(plan.get(key) != value or result.get(key) != value for key, value in expected.items()):
        raise ValueError("profile must identify the explicit selected-configuration schema and source")
    if evidence.read(path.parent / "measurement_prepared.json") != {
            "plan_sha256": sha256(evidence.file(path.parent / "measurement_plan.json"))}:
        raise ValueError("profile preparation differs from the recorded immutable plan")
    if plan.get("core") != core or result.get("core") != core or result.get("helpers") != plan.get("helpers"):
        raise ValueError("selected profile source/helper identity differs from its prepared source")
    if result.get("paired_source") != plan.get("paired_source"):
        raise ValueError("profile result differs from its recorded frozen Executor source")
    if (result.get("configuration") != plan.get("configuration")
            or result.get("copy_seeds") != plan.get("copy_seeds")
            or result.get("profile_updates") != plan.get("profile_updates")):
        raise ValueError("profile result changed its prepared configuration, policy seeds or update window")
    for record in (plan, result):
        validation.identity(record, context, commit, runtime)
        if validation.configuration_without_copies(checked_configuration(record["configuration"], root)) != baseline:
            raise ValueError("profile did not measure the shared accepted baseline")
    committed_files(plan["helpers"], evidence, [SCRIPT_PREFIX + name for name in (
        "profile_selected.py", "diagnose_gpu_execution.py", "post_comparison_validation.py", "train_optimized.py")])
    committed_files(plan["paired_source"], evidence,
                    [SCRIPT_PREFIX + name for name in ("paired_benchmark.py", "ideas.py")], require_worktree=True)
    if (plan["paired_source"]["source_root"], plan["paired_source"]["source_commit"]) != (str(root), commit):
        raise ValueError("selected profile Executor source differs from the admitted detached source")
    for relative, expected_hash in plan["helpers"]["files"].items():
        if sha256(evidence.file(path.parent / "code" / Path(relative).name)) != expected_hash:
            raise ValueError("a staged profile helper differs from its committed receipt")
    process = validation.execution_identity(result)
    hardware = result["hardware"]
    actual_runtime(result.get("actual_runtime_environment", hardware.get("runtime_environment")), runtime)
    gpu = plan["gpu_uuid"]
    if (result.get("completed") is not True or result.get("gpu_uuid") != gpu
            or hardware.get("gpu_uuid") != gpu or not gpu.startswith("GPU-")
            or (hardware.get("hostname"), hardware.get("pid"), hardware.get("process_start_time")) != process
            or admission.verified_hardware({**hardware, "runtime_environment": runtime}) != context):
        raise ValueError("profile lacks actual completed process/GPU evidence")
    execution = result["configuration"]["execution"]
    multiple = execution["iterations_per_call"]
    if execution["report_readout"] != "none":
        multiple = math.lcm(multiple, execution["report_every_iterations"])
    count, start, end = (result.get(key) for key in (
        "profile_updates", "profile_start_iteration", "profile_end_iteration"))
    if (any(type(value) is not int for value in (count, start, end)) or count < 1
            or start < 0 or end - start != count or count % multiple):
        raise ValueError("profile work does not contain complete execution groups and report windows")
    seeds = result["copy_seeds"]
    copies = result["configuration"]["learner"]["n_copies"]
    signatures = result.get("compiled_signatures_before")
    warmup = result.get("compilation_warmup", [])
    if (len(seeds) != copies or any(type(seed) is not int or not 0 <= seed < 2 ** 32 for seed in seeds)
            or len(seeds) != len(set(seeds)) or result.get("capture_completed") is not True
            or result.get("normalization_warmup_iterations") != 50
            or not isinstance(signatures, list) or not signatures
            or any(type(size) is not int or size < 0 for size in signatures) or sum(signatures) < 1
            or signatures != result.get("compiled_signatures_after")
            or not 2 <= len(warmup) <= 6
            or any(row.get("updates") != count for row in warmup)
            or any(row.get("compiled_signatures") != signatures for row in warmup[-2:])):
        raise ValueError("profile lacks independent policies, fifty warmup rollouts or stable compiled signatures")
    for row in warmup:
        validation.positive(row.get("seconds"), "profile compilation warmup seconds")
    trace = path.parent / "trace"
    if result.get("trace_analysis_record") != str(trace / "analysis.json"):
        raise ValueError("profile must retain the strict analysis beside its original raw capture")
    analysis = evidence.read(trace / "analysis.json")
    report = trace / "capture.nsys-rep"
    if (analysis.get("completed") is not True or analysis.get("exit_code") != 0
            or analysis.get("report") != str(report) or not evidence.file(report)
            or sha256(evidence.file(report)) != result.get("trace_report_sha256")):
        raise ValueError("profile has no successfully analyzed original raw trace")
    for name, column in (("cuda_gpu_kern_sum", "Instances"), ("cuda_api_sum", "Num Calls")):
        summary = analysis.get("summaries", {}).get(name, {})
        summary_path = Path(summary.get("path", ""))
        if summary_path.parent != trace:
            raise ValueError("profile summary must remain beside the original raw trace")
        evidence.file(summary_path)
        if diagnostic.summary_csv_evidence(summary_path, column) != {key: summary[key] for key in ("rows", "columns", "sha256")}:
            raise ValueError("profile summary hash or actual measurement rows changed")
    tool = result["profile_tool"]
    if (tool != plan.get("profile_tool") or not Path(tool["path"]).is_absolute()
            or not isinstance(tool.get("version"), str) or not tool["version"].strip()
            or not re.fullmatch(r"[0-9a-f]{64}", tool.get("sha256", ""))):
        raise ValueError("profile must identify the original profiling tool and its exact hash")
    return process


def register_five(registry, hypotheses, batch_number, commit, counts, profiling, root):
    """Register exactly five distinct caller-specified mechanisms and apply supported settings."""
    if not isinstance(hypotheses, list) or len(hypotheses) != 5:
        raise ValueError("an adaptive revision must register exactly five caller-specified ideas")
    previous = list(registry["initial_ideas"])
    previous += [row for batch in registry["adaptive_batches"] for row in batch["hypotheses"] if row.get("name")]
    names = {normalized_name(row["name"]) for row in previous}
    mechanisms = {normalized_name(row["mechanism"]) for row in previous}
    signatures = {ideas.canonical_signature(**ideas.trial_changes(entry, index))
                  for entry in ideas.INITIAL_IDEAS for index in (None, *range(len(entry.parameter_trials)))}
    signatures.update(ideas.canonical_signature(**{scope: row[scope] for scope in ("learner", "environment", "execution")})
                      for row in previous)
    required = {"name", "hypothesis", "mechanism", "learner", "environment", "execution",
                "correctness_checks", "measurement_settings", "acceptance_criteria"}
    updated = copy.deepcopy(registry)
    target = next(batch for batch in updated["adaptive_batches"] if batch["batch"] == batch_number)
    settings = None
    for slot, proposal in zip(target["hypotheses"], hypotheses):
        if set(proposal) - required - {"reference_execution"} or not required <= set(proposal):
            raise ValueError("adaptive hypothesis fields are missing or unsupported")
        name, mechanism = normalized_name(proposal["name"]), normalized_name(proposal["mechanism"])
        normalized_name(proposal["hypothesis"])
        if name in names or mechanism in mechanisms:
            raise ValueError("an adaptive name or mechanism repeats an earlier registered idea")
        if (not isinstance(proposal["correctness_checks"], list) or not proposal["correctness_checks"]
                or any(not isinstance(check, str) or not check.strip() for check in proposal["correctness_checks"])):
            raise ValueError("each adaptive idea requires concrete additional correctness checks")
        current = proposal["measurement_settings"]
        if (not isinstance(current, dict) or set(current) != {"paired_rounds", "minimum_updates_per_round"}
                or type(current["paired_rounds"]) is not int or current["paired_rounds"] != 7
                or type(current["minimum_updates_per_round"]) is not int or current["minimum_updates_per_round"] < 1):
            raise ValueError("only seven paired rounds and a positive complete-update minimum are supported")
        if settings is not None and current != settings:
            raise ValueError("all five ideas must use one supported common measurement setting")
        settings = dict(current)
        if controller.content_digest(proposal["acceptance_criteria"]) != controller.content_digest(ACCEPTANCE_CRITERIA):
            raise ValueError("custom acceptance criteria are unsupported by the fixed paired benchmark")
        changes = {scope: proposal[scope] for scope in ("learner", "environment", "execution")}
        if any(not isinstance(value, dict) for value in changes.values()) or not any(changes.values()):
            raise ValueError("each new idea requires concrete source-implemented performance changes")
        signature = ideas.canonical_signature(**changes)
        if signature in signatures:
            raise ValueError("an adaptive implementation repeats an initial trial or earlier adaptive signature")
        if set(changes["learner"]) & (set(controller.REFERENCE_ALGORITHM) | {"n_copies", "per_copy_learning_rates",
                                                                       "per_copy_intrinsic_coefficients", "per_copy_scalar_settings"}):
            raise ValueError("adaptive changes must preserve every fixed learner and per-copy setting")
        controller.reference_environment_settings(changes["environment"], root)
        controller.validate_execution(changes["execution"])
        controller.validate_execution(proposal.get("reference_execution", {}))
        slot.update(copy.deepcopy(proposal), state="specified", results={context: [] for context in validation.CONTEXTS})
        entry = ideas.registry_idea(proposal["name"], updated)
        if ideas.implementation_status(entry, ideas.source_capabilities(Path(root) / "10_jax_exploration_platform"))["state"] != "ready_for_correctness_check":
            raise ValueError("an adaptive performance switch is not implemented in the pinned source")
        names.add(name)
        mechanisms.add(mechanism)
        signatures.add(signature)
    target.update(reference_commit=commit, profiling_evidence=profiling, matched_copy_counts=counts)
    return updated, settings


def correctness_window_supported(reference, candidate):
    """Reject group pairs the pinned maximum-window correctness comparison cannot execute."""
    groups = [execution["iterations_per_call"] for execution in (reference, candidate)]
    required = groups + [execution["report_every_iterations"] for execution in (reference, candidate)
                        if execution["report_readout"] != "none"]
    if any(max(required) % group for group in groups):
        raise ValueError("the pinned correctness harness cannot compare these execution groups/report windows")


def validate_revision(run_dir, specification, evidence):
    """Validate all prerequisites and derive a complete new admission without live writes."""
    run_dir = Path(run_dir).resolve(strict=True)
    required = {"schema_version", "source_root", "source_commit", "matched_copy_counts", "runtime_environment",
                "worker_stop_record", "contexts", "hypotheses"}
    if set(specification) != required or specification["schema_version"] != 1:
        raise ValueError("revision specification schema_version must be one with only supported fields")
    launcher_capability(evidence)
    old = evidence.read(run_dir / controller.SENTINEL_NAME)
    registry = evidence.read(run_dir / "study_registry.json")
    if old.get("authorized") is not True:
        raise ValueError("adaptive revision requires an existing authorized initial admission")
    batch_number = completed_cohorts(registry, evidence)
    queue_is_empty(run_dir)
    stopped_workers(run_dir, specification["worker_stop_record"], evidence)
    root, commit = Path(specification["source_root"]).resolve(strict=True), specification["source_commit"]
    core = source_identity(root, commit)
    runtime = validation.runtime_environment(specification["runtime_environment"])
    counts = copy_counts(specification["matched_copy_counts"])
    if set(specification["contexts"]) != set(validation.CONTEXTS):
        raise ValueError("the revision must prove both specified GPU contexts")
    gates, baseline, processes = {}, None, set()
    for context in validation.CONTEXTS:
        inputs = specification["contexts"][context]
        if set(inputs) != {"learner_record", "renderer_record", "correctness_record", "confirmation_record",
                           "profile_record", "capacity_records"}:
            raise ValueError("each GPU context must provide exactly its complete gates, profile and capacity records")
        checks = {kind: admission.verified_check(inputs[kind + "_record"], commit, context, kind, run_dir, root)
                  for kind in ("learner", "renderer")}
        for receipt in checks.values():
            evidence.file(receipt["path"])
        result = validation.validate_combination({**inputs, "hardware_class": context,
            "source_commit": commit, "runtime_environment": runtime}, evidence)
        selected = checked_configuration(result["configuration"], root)
        normalized = validation.configuration_without_copies(selected)
        if baseline is not None and normalized != baseline:
            raise ValueError("both GPU contexts must prove the same complete accepted baseline")
        baseline = normalized
        for mode, key in (("correctness", "correctness_record"), ("confirmation", "confirmation_record")):
            _, process = producer_record(inputs[key], mode, run_dir, root, commit, runtime, context, core, evidence)
            if process in processes:
                raise ValueError("revision evidence reused a process instead of fresh isolated executions")
            processes.add(process)
        process = profile_record(inputs["profile_record"], run_dir, root, commit, runtime, context, baseline, core, evidence)
        if process in processes:
            raise ValueError("profiling must use its own recorded process execution")
        processes.add(process)
        capacity = []
        for path in inputs["capacity_records"]:
            config, process = producer_record(path, "packing-point", run_dir, root, commit, runtime, context, core, evidence)
            point = validation.packing_point(path, context, commit, runtime, evidence)
            if validation.configuration_without_copies(config) != baseline or process in processes:
                raise ValueError("capacity changed the baseline or reused another measurement process")
            processes.add(process)
            capacity.append(point)
        if sorted(point["copies"] for point in capacity) != counts:
            raise ValueError("fresh isolated capacity is required exactly once at every requested count, including one")
        gates[context] = {"passed": True, "source_commit": commit, "hardware_class": context,
                          "checks": checks, "combination": result, "capacity_checks": capacity}
    profiling = {context: specification["contexts"][context]["profile_record"] for context in validation.CONTEXTS}
    updated, settings = register_five(registry, specification["hypotheses"], batch_number, commit, counts, profiling, root)
    learner = {**baseline["learner"], "n_copies": 1}
    defaults = diagnostic.default_configuration(root)
    environment = {key: value for key, value in baseline["environment"].items() if value != defaults["environment"][key]}
    ready = {"authorized": True, "source_root": str(root), "source_commit": commit,
        "contexts": list(validation.CONTEXTS), "matched_copy_counts": counts,
        "seed_start": old.get("seed_start", 1), "rounds": settings["paired_rounds"],
        "iterations": settings["minimum_updates_per_round"], "reference_environment": environment,
        "reference_execution": baseline["execution"], "adaptive_batch": batch_number,
        "registered_hypotheses": [entry["name"] for entry in specification["hypotheses"]]}
    if "maximum_attempts" in old:
        ready["maximum_attempts"] = old["maximum_attempts"]
    for entry in specification["hypotheses"]:
        candidate = {"learner": {**learner, **entry["learner"]},
                     "environment": {**baseline["environment"], **entry["environment"]},
                     "execution": {**baseline["execution"], **entry["execution"]}}
        checked_configuration(candidate, root)
        if validation.configuration_without_copies(candidate) == baseline:
            raise ValueError("an adaptive candidate is identical to its accepted baseline")
        registered = ideas.registry_idea(entry["name"], updated)
        reference_execution, candidate_execution = controller.execution_arms(registered, baseline["execution"], root)
        correctness_window_supported(reference_execution, candidate_execution)
    return old, registry, updated, ready, gates, learner, batch_number


def require_validation_allocation(run_dir):
    """Keep real timing-statistic validation inside an owned Slurm compute allocation."""
    identifier = os.environ.get("SLURM_JOB_ID", "")
    host = socket.gethostname().split(".")[0].lower()
    ledgers = sorted((Path(run_dir) / "slurm").glob("submitted_jobids*.txt"))
    owned = {line for ledger in ledgers for line in ledger.read_text().splitlines()}
    cgroup = Path("/proc/self/cgroup").read_text()
    if (not identifier.isdecimal() or identifier not in owned or host.startswith(("portal", "login"))
            or not re.search(r"(?:^|/)job_" + re.escape(identifier) + r"(?:/|$)", cgroup, re.MULTILINE)):
        raise ValueError("prepare recomputes measured statistics; run it inside this campaign's owned Slurm compute allocation")


def prepare(run_dir, specification_path):
    """Stage an immutable fully validated revision while retaining all live campaign bytes."""
    run_dir = Path(run_dir).resolve(strict=True)
    evidence = validation.Evidence()
    specification = evidence.read(Path(specification_path).resolve(strict=True))
    with controller.queue_lock(run_dir):
        # An identical already-prepared input only needs byte checks; it never reinterprets a later live registry.
        existing = [path.parent for path in (run_dir / "admission_revisions").glob("*/prepared.json")
                    if controller.read_json(path.parent / "specification.json") == specification]
        if len(existing) > 1:
            raise ValueError("one caller specification has multiple prepared revisions")
        if existing:
            verify_prepared(existing[0])
            return existing[0]
        require_validation_allocation(run_dir)
        old, _, after, ready, gates, learner, number = validate_revision(run_dir, specification, evidence)
        old_paths = [Path(run_dir / controller.SENTINEL_NAME), Path(run_dir / "study_registry.json"),
                     Path(old["reference_configuration_path"]), *map(Path, old["gate_records"].values())]
        for path in old_paths:
            evidence.file(path)
        identity = {"specification": specification, "old_sentinel_sha256": sha256(evidence.file(old_paths[0])),
                    "registry_before_sha256": sha256(evidence.file(old_paths[1]))}
        revision = run_dir / "admission_revisions" / (
            f"adaptive-batch-{number:02d}_source-commit-{ready['source_commit']}_admission-sha256-{controller.content_digest(identity)}")
        ready.update(gate_records={context: str(revision / "staged/gates" / f"{context}.json") for context in gates},
                     reference_configuration_path=str(revision / "staged/reference_configuration.json"),
                     admission_revision=str(revision))
        artifacts = {"specification.json": json_bytes(specification),
            "history/SOURCE_AND_GATES_READY.json": evidence.file(old_paths[0]),
            "history/study_registry.json": evidence.file(old_paths[1]),
            "staged/SOURCE_AND_GATES_READY.json": json_bytes(ready),
            "staged/study_registry.json": json_bytes(after), "staged/reference_configuration.json": json_bytes(learner)}
        artifacts.update({f"staged/gates/{context}.json": json_bytes(record) for context, record in gates.items()})
        for receipt in evidence.records():
            artifacts["history/evidence/" + receipt["sha256"]] = evidence.file(receipt["path"])
        case_ready = {**ready, "reference_configuration": learner, "campaign_run_dir": str(run_dir)}
        for name in ready["registered_hypotheses"]:
            entry = ideas.registry_idea(name, after)
            for context in validation.CONTEXTS:
                for copies in ready["matched_copy_counts"]:
                    case = controller.immutable_case(entry, context, copies, case_ready)
                    if len((case["unit_id"] + ".json").encode()) >= 255:
                        raise ValueError("an adaptive case filename exceeds the filesystem component limit")
                    artifacts[f"staged/cases/{case['unit_id']}.json"] = json_bytes(case)
        manifest = {"schema_version": 1, "run_dir": str(run_dir), "revision": str(revision),
            "batch": number, "source_commit": ready["source_commit"], "evidence": evidence.records(),
            "artifacts": {name: sha256(contents) for name, contents in sorted(artifacts.items())}}
        for name, contents in artifacts.items():
            write_once(revision / name, contents)
        write_once(revision / "prepared.json", json_bytes(manifest))
    return revision


def verify_prepared(revision):
    """Check every staged/history byte before reading a prepared revision for activation."""
    revision = Path(revision).resolve(strict=True)
    manifest = controller.read_json(revision / "prepared.json")
    if manifest.get("schema_version") != 1 or manifest.get("revision") != str(revision):
        raise ValueError("prepared revision identity is invalid")
    for relative, expected in manifest["artifacts"].items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts or sha256((revision / relative).read_bytes()) != expected:
            raise ValueError("prepared revision artifact changed: " + relative)
    return manifest


def activated_registry_matches(actual, staged, batch_number):
    """Allow later result indexing without allowing the registered mechanisms to change."""
    actual, staged = copy.deepcopy(actual), copy.deepcopy(staged)
    for registry in (actual, staged):
        registry.pop("matched_copy_counts", None)
        for entry in registry["initial_ideas"]:
            entry.pop("implementation", None)
        for batch in registry["adaptive_batches"]:
            if batch["batch"] == batch_number:
                for entry in batch["hypotheses"]:
                    entry["results"] = {}
    return actual == staged


def activate(revision):
    """Recoverably publish one prepared registry and atomically switch admission last."""
    revision = Path(revision).resolve(strict=True)
    manifest = verify_prepared(revision)
    run_dir = Path(manifest["run_dir"])
    if revision.parent != run_dir / "admission_revisions":
        raise ValueError("revision directory must belong to the original campaign")
    with controller.queue_lock(run_dir):
        old = (revision / "history/SOURCE_AND_GATES_READY.json").read_bytes()
        before = (revision / "history/study_registry.json").read_bytes()
        new = (revision / "staged/SOURCE_AND_GATES_READY.json").read_bytes()
        after = (revision / "staged/study_registry.json").read_bytes()
        sentinel, registry = run_dir / controller.SENTINEL_NAME, run_dir / "study_registry.json"
        transaction_path = revision / "transaction.json"
        identity = {"prepared_sha256": sha256((revision / "prepared.json").read_bytes()),
            "registry_before_sha256": sha256(before), "registry_after_sha256": sha256(after),
            "sentinel_before_sha256": sha256(old), "sentinel_after_sha256": sha256(new)}
        transaction = controller.read_json(transaction_path) if transaction_path.exists() else None
        if transaction is not None and (transaction.get("state") not in {"publishing", "active"}
                or any(transaction.get(key) != value for key, value in identity.items())):
            raise ValueError("existing transaction metadata belongs to different publication endpoints")
        if sentinel.read_bytes() == new and activated_registry_matches(
                controller.read_json(registry), json.loads(after), manifest["batch"]):
            if transaction is None:
                raise ValueError("new admission has no preceding recorded activation transaction")
            if transaction["state"] != "active":
                controller.atomic_json(transaction_path, {**identity, "state": "active", "time": controller.utc_now()})
            return {"state": "active", "revision": str(revision), "already_active": True}
        if sentinel.read_bytes() != old or registry.read_bytes() not in (before, after):
            raise ValueError("live admission or registry differs from both transaction endpoints")
        if transaction is not None and transaction["state"] == "active":
            raise ValueError("an activated admission was subsequently reverted outside its transaction")
        queue_is_empty(run_dir)
        evidence = validation.Evidence()
        launcher_capability(evidence)
        specification = controller.read_json(revision / "specification.json")
        stopped_workers(run_dir, specification["worker_stop_record"], evidence)
        source_identity(specification["source_root"], specification["source_commit"])
        for receipt in manifest["evidence"]:
            # The exact two live transaction endpoints were checked above; recovery may already have the new registry.
            if Path(receipt["path"]) in {sentinel, registry}:
                continue
            if sha256(evidence.file(receipt["path"])) != receipt["sha256"]:
                raise ValueError("prepared admission evidence changed before activation")
        controller.atomic_json(transaction_path, {**identity, "state": "publishing", "time": controller.utc_now()})
        replace_bytes(registry, after)
        replace_bytes(sentinel, new)
        controller.atomic_json(transaction_path, {**identity, "state": "active", "time": controller.utc_now()})
    return {"state": "active", "revision": str(revision), "already_active": False}


def argument_parser():
    """Keep metadata preparation and explicit live activation as separate commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--run-dir", type=Path, required=True)
    preparation.add_argument("--specification", type=Path, required=True)
    activation = commands.add_parser("activate")
    activation.add_argument("--revision", type=Path, required=True)
    return parser


if __name__ == "__main__":
    arguments = argument_parser().parse_args()
    if arguments.command == "prepare":
        print(json.dumps({"prepared_revision": str(prepare(arguments.run_dir, arguments.specification))}, indent=2))
    else:
        print(json.dumps(activate(arguments.revision), indent=2))
