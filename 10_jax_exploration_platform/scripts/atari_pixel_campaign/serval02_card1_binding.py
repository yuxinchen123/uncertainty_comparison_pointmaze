"""Bind only the authorized direct card under its actual standing lock.

History is immutable and separate from the newly registered process. No Slurm
configuration trust applies here. This module never contacts a host, launches
work, acquires a lock, or grants worker/scientific admission.
"""

import ast
import ctypes
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import uuid


PROFILE = {"schema_version": 1, "name": "direct-serval02-card-1-current-process-and-standing-lock-version-1",
    "hostname": "serval02", "username": "sl5nw", "uid": 2618919, "hardware_class": "h100",
    "gpu_uuid": "GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3", "model": "NVIDIA H100 NVL",
    "driver": "590.48.01", "memory_mib": 95830, "gpu_count": 1,
    "standing_lock": "/tmp/sl5nw_serval02_card1.lock", "lock_device": 64512, "lock_inode": 23077017,
    "lock_mode": 420, "kernel_device_enforcement_verified": False}
CAPABILITY = "approved_direct_uuid_held_standing_lock_current_cuda_context_and_assigned_device_windows_v1"
SOURCE_COMMIT = "d28e6735e79788f618acb0ebfde8609db1f776af"
ORIGINAL_ADMISSION_SHA256 = "4a71ef2b245ae784933c55bb2b5159aa1bb7f8f626eb94b15390e2f9a304ee7c"
ORIGINAL_GATE_SHA256 = {"h100": "7d020f2a4436ed9f572d0009f4e6e4df8d935546ef8abc2afd1824ace1a59def",
    "jaguar03_rtx_a4500": "2e43c7caf3549df23a4be4cadd338f9413c599e81e25728e26d0b472afbd44f7"}
PREFIX = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"
HELPERS = ("serval02_card1_binding.py", "uva_slurm_configured_isolation.py", "device_memory_limit.py",
    "h100_nvl_slurm_site_validation.py", "uva_active_configuration_observation.py", "passive_installed_slurm_observation.py",
    "collect_worker_exit_receipts.py", "post_comparison_validation.py", "portable_evidence.py", "ideas.py", "controller.py",
    "diagnose_gpu_execution.py", "train_optimized.py", "infrastructure_retry.py", "pregather_async_retry.py",
    "h100_nvl_slurm_queue.py", "h100_nvl_slurm_case.py", "h100_nvl_slurm_evidence.py", "h100_nvl_slurm_qualification.py",
    "h100_nvl_allocation_observation.py", "validate_training_deployment.py", "short_learning.py")
BOUNDARY = {"worker_admitted": False, "admission_granted": False, "case_credit": 0, "idea_credit": 0,
    "site_qualification_credit": 0, "scientific_credit": 0, "packing_point_credit": 0, "throughput_credit": 0,
    "kernel_device_enforcement_verified": False, "uva_configuration_service_trust_used": False}


def adjacent(name):
    path = Path(__file__).with_name(name + ".py")
    specification = importlib.util.spec_from_file_location("direct_card_one_" + name, path)
    module = importlib.util.module_from_spec(specification); specification.loader.exec_module(module)
    return module


# Only generic byte/capture operations are reused. Never invoke the UVA policy validators.
common = adjacent("uva_slurm_configured_isolation")
require, digest, encoded, exact = common.require, common.digest, common.encoded, common.exact
receipt, read_receipt, publish = common.receipt, common.read_receipt, common.publish


def require_profile(value):
    require(exact(value, PROFILE), "the exact separate direct card-one profile is required")
    return value


def pending_identity():
    return {"schema_version": 1, "kind": "authorized_direct_uuid_pending_current_process_verification",
        "authorized_gpu_uuid": PROFILE["gpu_uuid"], "pci_address": None, "physical_minor": None, "current_context": None}


def unit_directory(plan):
    require(isinstance(plan.get("unit_id"), str) and Path(plan["unit_id"]).name == plan["unit_id"]
        and plan["unit_id"] not in {"", ".", ".."}, "one literal direct unit identity is required")
    return Path(plan["campaign_run_dir"]) / "data" / plan["unit_id"]


def validate_plan(plan, consumer=None):
    value = plan.get("direct_binding")
    require(isinstance(value, dict) and set(value) == {"schema_version", "profile", "capability", "consumer",
        "physical_identity", "frozen_inputs", "current_launch_inputs_path"}, "explicit direct profile plan required")
    require(type(value["schema_version"]) is int and value["schema_version"] == 1 and value["capability"] == CAPABILITY,
        "unsupported direct profile capability")
    require_profile(value["profile"])
    require(value["consumer"] in {"selected_measurement", "component_profile"} and (consumer is None or consumer == value["consumer"])
        and plan.get("gpu_uuid") == PROFILE["gpu_uuid"] and plan.get("hardware_class") == "h100"
        and plan.get("source_commit") == SOURCE_COMMIT and exact(value["physical_identity"], pending_identity()),
        "direct authorization does not establish a current physical binding")
    require((value["consumer"] == "selected_measurement" and plan.get("kind") == "selected_configuration_measurement"
        and plan.get("mode") in {"correctness", "confirmation"}
        or value["consumer"] == "component_profile" and plan.get("kind") == "selected_configuration_profile" and plan.get("mode") in {None, "profile"})
        and not any(key in plan for key in ("configured_isolation", "device_memory_limit_protocol", "final_protocol",
            "final_confirmation_protocol", "completed_study", "completed_study_manifest", "scaling_protocol")),
        "unsupported packing, capacity, final, scaling, portable or UVA consumer")
    unit = unit_directory(plan)
    require(value["frozen_inputs"]["path"] == str(unit / "direct_history/inputs.json")
        and value["current_launch_inputs_path"] == str(unit / "direct_current_launch.json"), "history and current launch must use distinct reserved unit paths")
    return value


def source_identity(plan):
    value = plan["helpers"]; root = Path(value["source_root"])
    require(root.is_absolute() and root.resolve(strict=True) == root and re.fullmatch(r"[0-9a-f]{40}", value["source_commit"]), "exact committed helper source required")
    for name in HELPERS:
        relative = PREFIX + name
        raw = subprocess.check_output(["git", "-C", str(root), "show", value["source_commit"] + ":" + relative])
        require(value["files"].get(relative) == digest(raw) == receipt(root / relative)["sha256"]
            == receipt(Path(__file__).with_name(name))["sha256"], "direct helper closure changed committed or staged bytes")
    return value


def aware(value):
    moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(moment.tzinfo is not None, "an explicit timezone-aware observation time is required")
    return moment


def shared_portable_module():
    """Preserve the strict reader's one class identity without accepting ambient code."""
    name, path = "portable_evidence", Path(__file__).with_name("portable_evidence.py").resolve(strict=True)
    if name in sys.modules:
        module = sys.modules[name]
        require(getattr(module, "__file__", None) is not None and Path(module.__file__).resolve(strict=True) == path,
            "the portable reader module must be this exact adjacent staged file")
        return module
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        if sys.modules.get(name) is module:
            del sys.modules[name]
        raise
    return module


def snapshot_exit_history(exit_plan, observations, stop, status):
    """Replay only saved original-worker records; no current ledger or scheduler query."""
    collector = adjacent("collect_worker_exit_receipts")
    require(type(exit_plan.get("schema_version")) is int and exit_plan["schema_version"] == 1
        and exit_plan.get("kind") == "worker_exit_observation_plan" and exit_plan.get("owner") == "sl5nw"
        and type(exit_plan.get("owner_uid")) is int and exit_plan["owner_uid"] == 2618919
        and exit_plan.get("collector_sha256") == receipt(Path(__file__).with_name("collect_worker_exit_receipts.py"))["sha256"],
        "worker-exit snapshot changed its original owner or collector")
    inputs = exit_plan["inputs"]; files = inputs["files"]
    fixed = {"processes.json", "slurm/submissions.tsv", "slurm/submitted_jobids_uva.txt"}
    require(fixed <= set(files) and all(name in fixed or re.fullmatch(r"queue/(?:workers|leases)/[^/]+\.json", name) for name in files), "unexpected historical input path")
    for value in files.values():
        require(set(value) == {"sha256", "text"} and digest(value["text"].encode()) == value["sha256"], "historical input snapshot changed bytes")
    launches = [row for row in json.loads(files["processes.json"]["text"]) if row.get("kind") == "queue_worker"]
    workers = [json.loads(files[name]["text"]) for name in sorted(files) if name.startswith("queue/workers/")]
    runners = [json.loads(files[name]["text"]) for name in sorted(files) if name.startswith("queue/leases/") and json.loads(files[name]["text"]).get("runner_pid") is not None]
    jobs = collector.required_jobs(files["slurm/submissions.tsv"]["text"], workers, runners, files["slurm/submitted_jobids_uva.txt"]["text"].splitlines())
    require(inputs["launches"] == launches and inputs["workers"] == workers and inputs["runners"] == runners and inputs["jobs"] == jobs,
        "worker-exit parsed records no longer equal their frozen original inputs")
    require(len(jobs) == 25, "the exact original twenty-five numerical jobs must all be covered")
    scheduler = observations["scheduler"]
    require(scheduler["command"] == ["sacct", "-X", "--jobs=" + ",".join(jobs), "--noheader", "--parsable2",
        "--format=JobIDRaw,User,State,ExitCode,NodeList,Start,End"]
        and scheduler["squeue_command"] == ["squeue", "--jobs=" + ",".join(jobs), "--noheader", "--format=%i|%u|%T"]
        and scheduler["squeue_stdout"] == "", "old attempts remain active, unreported or use another exact owned query")
    require(aware(scheduler["observed_at"]) >= aware(exit_plan["prepared_at"]), "terminal scheduler observation predates its frozen input plan")
    rebuilt_jobs = {}
    for line in scheduler["stdout"].splitlines():
        row = line.split("|")
        require(len(row) == 7 and row[0] in jobs and row[0] not in rebuilt_jobs and row[1] == "sl5nw", "foreign, duplicate or missing original accounting identity")
        identifier, owner, raw_state, code, node, start, end = row
        state = raw_state.split()[0].rstrip("+")
        require(state in collector.TERMINAL_STATES, "an old numerical job is not terminal")
        rebuilt_jobs[identifier] = {"job_id": identifier, "owner": owner, "state": state, "state_raw": raw_state,
            "exit_code": code, "node_list": node, "start": start, "end": end, "submission_row": jobs[identifier], "observed_at": scheduler["observed_at"]}
    require(set(rebuilt_jobs) == set(jobs), "missing exact original terminal jobs")
    expected = set(collector.direct_identities(inputs)); direct = {}
    for record in observations["direct_processes"]:
        raw = json.loads(record["stdout"])
        require(raw["hostname"] == record["hostname"] == "serval02" and raw["observed_at"] == record["observed_at"], "historical direct observation host or time changed")
        require(aware(raw["observed_at"]) >= aware(exit_plan["prepared_at"]), "direct exit observation predates its frozen input plan")
        for row in raw["observations"]:
            key = ("serval02", row["pid"], row["expected_birth"])
            require(key in expected and key not in direct and row["original_process_alive"] is False
                and row.get("observed_birth") != row["expected_birth"], "old direct process remains alive or lacks exact birth coverage")
            direct[key] = {**row, "hostname": "serval02", "observed_at": raw["observed_at"]}
    require(set(direct) == expected, "old supervisor or numerical-child coverage is incomplete")
    rebuilt, waiting = collector.make_receipt(exit_plan, rebuilt_jobs, direct)
    require(waiting == [] and exact(stop, rebuilt) and status.get("state") == "complete" and status.get("waiting") == [],
        "the original stop receipt does not replay exactly from the frozen raw observations")
    for launch in launches:
        require(aware(direct[launch["hostname"], launch["pid"], str(launch["start_time"])] ["observed_at"]) >= aware(launch["time"]), "stop observation predates the launch")
    for worker in workers:
        if not worker.get("slurm_job_id"):
            require(aware(direct[worker["hostname"], worker["worker_pid"], str(worker["worker_start_time"])]["observed_at"]) >= aware(worker["time"]), "worker exit observation predates its last metadata")
    for runner in runners:
        if not runner.get("slurm_job_id"):
            observed = aware(direct[runner["hostname"], runner["runner_pid"], str(runner["runner_start_time"])]["observed_at"])
            require(all(observed >= aware(runner[key]) for key in ("claimed_at", "finished_at") if key in runner), "numerical child exit observation predates its execution")
    require(type(status.get("schema_version")) is int and status["schema_version"] == 1
        and exact(status.get("counts"), {key: len(inputs[key]) for key in ("workers", "jobs", "launches", "runners")}), "worker-stop status coverage changed")
    return {"jobs": 25, "direct_processes": len(direct), "source": "immutable historical snapshots only"}


def validate_history(plan):
    """Rederive the full initial study and exit before any new direct contact."""
    options = validate_plan(plan); unit = unit_directory(plan); source_identity(plan)
    inputs = read_receipt(options["frozen_inputs"])
    names = {"admission", "registry", "evidence_manifest", "exit_plan", "exit_observations", "worker_stop", "exit_status"}
    require(type(inputs.get("schema_version")) is int and inputs["schema_version"] == 1
        and inputs.get("kind") == "serval02_card1_frozen_history" and set(inputs) == names | {"schema_version", "kind"}, "unsupported frozen historical input schema")
    for name in names:
        path = Path(inputs[name]["path"])
        require(path.is_relative_to(unit / "direct_history") and path != Path(plan["campaign_run_dir"]) / "processes.json", "historical proof cannot use a live ledger or external copy")
    documents = {name: read_receipt(inputs[name]) for name in names}
    require(inputs["admission"]["sha256"] == ORIGINAL_ADMISSION_SHA256
        and documents["exit_status"]["prepared_plan_sha256"] == inputs["exit_plan"]["sha256"], "frozen original admission or stop-plan byte identity changed")
    manifest = documents["evidence_manifest"]
    require(all(not row["relative_path"].startswith("/") for row in manifest["files"]), "frozen evidence cannot fall back to live paths")
    resolver = shared_portable_module().ExactPathResolver(inputs["evidence_manifest"]["path"], inputs["evidence_manifest"]["sha256"])
    validation = adjacent("post_comparison_validation"); evidence = validation.Evidence(resolver)
    admission, registry = documents["admission"], documents["registry"]
    require(admission.get("authorized") is True and admission.get("source_commit") == SOURCE_COMMIT
        and admission.get("source_root") == plan["source_root"] and set(admission["contexts"]) == {"h100", "jaguar03_rtx_a4500"}, "the immutable original source admission changed")
    require(set(admission["gate_records"]) == set(ORIGINAL_GATE_SHA256), "both exact original hardware gates are required")
    for context, path in admission["gate_records"].items():
        gate = evidence.read(path)
        require(digest(evidence.file(path)) == ORIGINAL_GATE_SHA256[context] and gate.get("passed") is True
            and gate.get("hardware_class") == context and gate.get("source_commit") == SOURCE_COMMIT, "an original hardware gate is missing or changed")
    initial = registry["initial_ideas"]
    expected = {entry.name: entry.position for entry in adjacent("ideas").IDEAS_BY_NAME.values() if entry.position <= 30}
    require(len(initial) == 30 and exact({entry["name"]: entry["position"] for entry in initial}, expected)
        and exact(registry.get("initial_matched_copy_counts"), [1, 2]), "the exact original thirty ideas and two copy counts are required")
    checked, case_processes = 0, []
    for entry in initial:
        for context in ("h100", "jaguar03_rtx_a4500"):
            for copies in (1, 2):
                rows = [row for row in entry["results"][context] if type(row["copies"]) is int and row["copies"] == copies]
                require(len(rows) == 1, "each original idea/context/copy count needs one completed result")
                result_path = Path(rows[0]["result_path"])
                case = evidence.read(result_path.parent / "case.json")
                require(result_path.is_absolute() and ".." not in result_path.parts
                    and str(result_path) == rows[0]["result_path"]
                    and result_path.is_relative_to(Path(plan["campaign_run_dir"]) / "data")
                    and case.get("campaign_run_dir") == plan["campaign_run_dir"],
                    "each original case and logical result path must belong to this exact campaign data tree")
                validation.validate_cohort_case(rows[0]["result_path"], entry["name"], context, copies, SOURCE_COMMIT, evidence)
                hardware = evidence.read(Path(rows[0]["result_path"]).parent / "hardware.json")
                case_processes.append((context, hardware))
                checked += 1
    require(checked == 120, "the original study has not completed all 120 comparisons")
    stop = snapshot_exit_history(documents["exit_plan"], documents["exit_observations"], documents["worker_stop"], documents["exit_status"])
    require(documents["exit_plan"]["run_dir"] == plan["campaign_run_dir"], "worker-stop history belongs to another campaign")
    original = Path(plan["campaign_run_dir"])
    covered_files = set(documents["exit_plan"]["inputs"]["files"])
    expected_files = set()
    for row in manifest["files"]:
        path = Path(row["original_path"])
        if path.is_relative_to(original):
            relative = str(path.relative_to(original))
            if relative in {"processes.json", "slurm/submissions.tsv", "slurm/submitted_jobids_uva.txt"} or re.fullmatch(r"queue/(?:workers|leases)/[^/]+\.json", relative):
                expected_files.add(relative)
    require(covered_files == expected_files, "worker-stop inputs must cover the complete frozen worker, lease and ownership-file universe")
    for name, record in documents["exit_plan"]["inputs"]["files"].items():
        require(evidence.file(Path(plan["campaign_run_dir"]) / name) == record["text"].encode(), "the original case history differs from the worker-exit snapshot")
    covered_jobs = set(documents["exit_plan"]["inputs"]["jobs"])
    covered_direct = set(adjacent("collect_worker_exit_receipts").direct_identities(documents["exit_plan"]["inputs"]))
    for context, hardware in case_processes:
        job = hardware.get("slurm_job_id")
        if job:
            require(isinstance(job, str) and re.fullmatch(r"[1-9][0-9]*", job) and job in covered_jobs,
                "a validated original case has no covered exact terminal Slurm job")
        else:
            require(context == "h100" and hardware.get("hostname") == "serval02"
                and ("serval02", hardware["pid"], hardware["process_start_time"]) in covered_direct,
                "a validated original direct child is absent from the exact birth-qualified exit proof")
            if "worker_pid" in hardware or "worker_start_time" in hardware:
                require(("serval02", hardware.get("worker_pid"), hardware.get("worker_start_time")) in covered_direct,
                    "a validated original direct child's parent is absent from the exit proof")
    return {"inputs": options["frozen_inputs"], "comparisons": checked, "stop": stop}


def current_process():
    """Read only current process metadata, keeping counters out of stable identity."""
    raw = Path("/proc/self/stat").read_text(); fields = raw[raw.rfind(")") + 2:].split()
    status = Path("/proc/self/status").read_text()
    uids = "".join(line + "\n" for line in status.splitlines() if line.startswith("Uid:"))
    value = {"hostname": socket.gethostname().split(".")[0], "pid": os.getpid(), "parent_pid": os.getppid(),
        "process_start_time": fields[19], "uid": os.getuid(), "effective_uid": os.geteuid(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "process_group": os.getpgrp()}
    return {"identity": value, "stat": raw, "stat_sha256": digest(raw.encode()), "uid_bytes": uids,
        "uid_sha256": digest(uids.encode())}


def current_admission_claim(plan):
    """Reference the frozen copy separately from the current canonical-file check."""
    inputs = read_receipt(plan["direct_binding"]["frozen_inputs"])
    frozen = inputs["admission"]
    require(frozen["sha256"] == ORIGINAL_ADMISSION_SHA256
        and Path(frozen["path"]).is_relative_to(unit_directory(plan) / "direct_history"), "current admission needs the exact frozen original copy")
    return {"schema_version": 1, "kind": "current_direct_admission_matches_frozen_original",
        "canonical_path": str(Path(plan["campaign_run_dir"]) / "SOURCE_AND_GATES_READY.json"),
        "frozen_admission": frozen, "source_root": plan["source_root"], "source_commit": SOURCE_COMMIT}


def original_source_scopes():
    """Read the exact controller source scopes without importing its numerical helper names."""
    # Before: executing controller registered staged ideas globally; after: only its two literal paths are read.
    path = Path(__file__).absolute().with_name("controller.py")
    require(path.resolve(strict=True) == path and not path.is_symlink() and path.is_file(),
        "the original source-scope declaration must be the exact adjacent controller file")
    tree = ast.parse(path.read_bytes())
    declarations = [node for node in ast.walk(tree) if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
        and any(isinstance(target, ast.Name) and target.id == "SOURCE_SCOPES"
            for target in (node.targets if isinstance(node, ast.Assign) else (node.target,)))]
    require(len(declarations) == 1 and declarations[0] in tree.body
        and isinstance(declarations[0], ast.Assign) and len(declarations[0].targets) == 1,
        "the controller must declare exactly one top-level literal source-scope tuple")
    scopes = ast.literal_eval(declarations[0].value)
    require(type(scopes) is tuple and scopes == (
        "10_jax_exploration_platform/src/exploration_platform/atari_pixels",
        "10_jax_exploration_platform/scripts/atari_pixel_campaign"),
        "the original admitted numerical and worker source scopes changed")
    return scopes


def check_live_admission(claim):
    """Check live authority at execution/window boundaries, never during historical replay."""
    require(type(claim.get("schema_version")) is int and claim["schema_version"] == 1
        and claim.get("kind") == "current_direct_admission_matches_frozen_original"
        and claim["frozen_admission"]["sha256"] == ORIGINAL_ADMISSION_SHA256
        and claim["source_commit"] == SOURCE_COMMIT, "unsupported live direct admission claim")
    require(receipt(claim["canonical_path"])["sha256"] == claim["frozen_admission"]["sha256"], "current canonical source admission changed")
    admission = read_receipt(claim["frozen_admission"])
    require(admission.get("authorized") is True and admission.get("source_root") == claim["source_root"]
        and admission.get("source_commit") == claim["source_commit"], "current source root does not equal the frozen original admission")
    root = Path(claim["source_root"])
    require(root.is_absolute() and root.is_dir() and root.resolve(strict=True) == root,
        "the current admitted numerical source must be its exact ordinary checkout")
    command = ["git", "-C", str(root)]
    require(subprocess.check_output(command + ["rev-parse", "--show-toplevel"], text=True).strip() == str(root)
        and subprocess.check_output(command + ["rev-parse", "HEAD"], text=True).strip() == SOURCE_COMMIT
        and subprocess.run(command + ["symbolic-ref", "--quiet", "HEAD"], capture_output=True).returncode == 1,
        "the current numerical source must remain detached at the exact original commit")
    scopes = original_source_scopes()
    require(not subprocess.check_output(command + ["status", "--porcelain", "--untracked-files=all", "--", *scopes], text=True).strip(),
        "the current admitted numerical or original worker source has changes")
    roster = subprocess.check_output(command + ["ls-tree", "-rz", SOURCE_COMMIT, "--", *scopes])
    entries = [row for row in roster.split(b"\0") if row]
    require(entries, "the current original source closure is empty")
    for row in entries:
        header, relative = row.split(b"\t", 1)
        mode, kind, blob = header.decode().split()
        require(mode in {"100644", "100755"} and kind == "blob", "the original source closure contains an unsupported file")
        path = root / relative.decode()
        committed = subprocess.check_output(command + ["cat-file", "blob", blob])
        require(receipt(path)["sha256"] == digest(committed), "current original source bytes differ from their frozen Git blob")
    return claim


def metadata_directory(plan, directory, role):
    """Allow only the actual unit or its explicitly separate profiling supervisor."""
    unit = unit_directory(plan)
    profile = plan["direct_binding"]["consumer"] == "component_profile"
    expected = unit / "supervisor" if profile and role == "owner" else unit
    path = Path(directory)
    current_child = role == "producer_guarded_child" and profile and plan.get("capture_process_protocol") == "direct-h100-owned-nsight-and-retained-descriptor-version-1"
    require((role in {"owner", "inherited"} or current_child) and (profile or role == "owner") and path.is_absolute() and path == expected,
        "direct metadata output must be this exact selected unit, profiling supervisor or inherited-child unit")
    require(path.resolve(strict=False) == path and path.parent.resolve(strict=True) == path.parent
        and (not path.exists() or path.is_dir() and not path.is_symlink()), "direct metadata output uses an alias or non-directory")
    return path


def binding_directory(plan, metadata, directory):
    """A profiling parent can hold custody but cannot replace the actual GPU child."""
    profile_role = "producer_guarded_child" if plan.get("capture_process_protocol") == "direct-h100-owned-nsight-and-retained-descriptor-version-1" else "inherited"
    require(metadata["role"] == (profile_role if plan["direct_binding"]["consumer"] == "component_profile" else "owner"),
        "only the actual inherited profiling child or selected owner may bind a device")
    path = metadata_directory(plan, directory, metadata["role"])
    require(path == unit_directory(plan), "the resolved direct device binding belongs only to the actual unit")
    return path


def validate_process(value):
    process = value["identity"]
    for key in ("pid", "parent_pid"):
        require(type(process[key]) is int and process[key] > 0, "current direct process requires positive numeric identities")
    fields = value["stat"][value["stat"].rfind(")") + 2:].split()
    require(process["hostname"] == "serval02" and type(process["uid"]) is int
        and process["uid"] == process["effective_uid"] == 2618919 and process["pid"] != process["parent_pid"]
        and digest(value["stat"].encode()) == value["stat_sha256"] and value["stat"].split(" ", 1)[0] == str(process["pid"])
        and fields[1] == str(process["parent_pid"]) and fields[19] == process["process_start_time"]
        and type(process["process_group"]) is int and process["process_group"] > 0 and fields[2] == str(process["process_group"])
        and re.fullmatch(r"[1-9][0-9]*", process["process_start_time"])
        and digest(value["uid_bytes"].encode()) == value["uid_sha256"]
        and re.fullmatch(r"Uid:\s*2618919\s+2618919\s+2618919\s+2618919\s*", value["uid_bytes"])
        and isinstance(process["cpu_affinity"], list) and process["cpu_affinity"]
        and len(set(process["cpu_affinity"])) == len(process["cpu_affinity"])
        and all(type(cpu) is int and cpu >= 0 for cpu in process["cpu_affinity"]), "current direct PID, birth, all four UIDs or host changed")
    return process


def lock_metadata(fd):
    """Observe an already held descriptor without acquiring or opening a second lock."""
    require(type(fd) is int and fd >= 0, "an actual held lock descriptor is required")
    information = os.fstat(fd); path = Path(PROFILE["standing_lock"]); named = path.lstat()
    with Path("/proc/self/fdinfo/" + str(fd)).open() as stream:
        raw = stream.read(8193)
    require(len(raw.encode()) <= 8192, "own lock fdinfo exceeds its bound")
    fields = lambda info: {"device": info.st_dev, "inode": info.st_ino, "uid": info.st_uid,
        "mode": stat.S_IMODE(info.st_mode), "regular_file": stat.S_ISREG(info.st_mode)}
    return {"path": str(path), "fd": fd, "fd_stat": fields(information), "path_stat": fields(named),
        "path_symlink": stat.S_ISLNK(named.st_mode), "fd_target": os.readlink("/proc/self/fd/" + str(fd)),
        "fdinfo": raw, "fdinfo_sha256": digest(raw.encode())}


def validate_lock(value, holder_pid):
    expected = {"device": PROFILE["lock_device"], "inode": PROFILE["lock_inode"], "uid": 2618919,
        "mode": PROFILE["lock_mode"], "regular_file": True}
    require(value["path"] == value["fd_target"] == PROFILE["standing_lock"] and type(value["fd"]) is int and value["fd"] >= 0
        and value["path_symlink"] is False and exact(value["fd_stat"], expected) and exact(value["path_stat"], expected)
        and digest(value["fdinfo"].encode()) == value["fdinfo_sha256"], "the approved existing lock inode, owner, path or descriptor changed")
    rows = re.findall(r"^lock:\s*[0-9]+:\s+FLOCK\s+ADVISORY\s+WRITE\s+([1-9][0-9]*)\s+([0-9a-fA-F]+):([0-9a-fA-F]+):([0-9]+)\s+0\s+EOF\s*$", value["fdinfo"], re.M)
    require(len(rows) == 1 and int(rows[0][0]) == holder_pid and int(rows[0][1], 16) == os.major(expected["device"])
        and int(rows[0][2], 16) == os.minor(expected["device"]) and int(rows[0][3]) == expected["inode"],
        "this descriptor does not prove the exact holder's exclusive standing FLOCK")
    return value


def launch_inputs(plan):
    """Current registration is a new immutable record; historical snapshots never use it."""
    unit = unit_directory(plan); value = read_receipt(receipt(plan["direct_binding"]["current_launch_inputs_path"]))
    require(type(value.get("schema_version")) is int and value["schema_version"] == 1
        and value.get("kind") == "serval02_card1_current_launch" and set(value) == {"schema_version", "kind", "plan", "process_row", "process_ledger"}
        and value["plan"] == receipt(unit / "measurement_plan.json") and read_receipt(value["plan"]) == plan, "current direct registration changed its immutable plan")
    ledger, row = value["process_ledger"], value["process_row"]
    require(set(ledger) == {"snapshot", "source_path"} and ledger["source_path"] == str(Path(plan["campaign_run_dir"]) / "processes.json")
        and ledger["snapshot"]["path"] == str(unit / "direct_current_process_ledger_snapshot.json"), "current process registration needs a separate reserved copied ledger")
    for candidate in (read_receipt(ledger["snapshot"]), json.loads(common.ordinary(ledger["source_path"]).read_bytes())):
        require(isinstance(candidate, list) and sum(item == row for item in candidate) == 1
            and sum(item.get("hostname") == row["hostname"] and item.get("pid") == row["pid"] and item.get("start_time") == row["start_time"] for item in candidate) == 1,
            "current direct process registration is missing, duplicate or substituted")
    require(row.get("hostname") == "serval02" and row.get("gpu_uuid") == PROFILE["gpu_uuid"]
        and row.get("kind") == plan["kind"] and row.get("unit") == plan["unit_id"]
        and row.get("source_commit") == SOURCE_COMMIT and row.get("source_root") == plan["source_root"]
        and row.get("helper_commit") == plan["helpers"]["source_commit"]
        and type(row.get("pid")) is int and row["pid"] > 0 and re.fullmatch(r"[1-9][0-9]*", row["start_time"]),
        "the actual current process row changed its target, source or identity")
    return value


def capture_owner_process(pid):
    """Read only a recorded own holder/profiler, checking UID before its process contents."""
    path = Path("/proc") / str(pid)
    require(path.stat().st_uid == 2618919, "a recorded direct holder/profiler is not this owner")
    raw, status = (path / "stat").read_text(), (path / "status").read_text()
    uids = "".join(line + "\n" for line in status.splitlines() if line.startswith("Uid:"))
    return {"pid": pid, "stat": raw, "stat_sha256": digest(raw.encode()), "uid_bytes": uids, "uid_sha256": digest(uids.encode())}


def validate_owner_process(value, pid, birth, parent_pid=None):
    fields = value["stat"][value["stat"].rfind(")") + 2:].split()
    require(value["pid"] == pid and value["stat"].split(" ", 1)[0] == str(pid)
        and fields[19] == birth and (parent_pid is None or fields[1] == str(parent_pid))
        and digest(value["stat"].encode()) == value["stat_sha256"]
        and digest(value["uid_bytes"].encode()) == value["uid_sha256"]
        and re.fullmatch(r"Uid:\s*2618919\s+2618919\s+2618919\s+2618919\s*", value["uid_bytes"]), "recorded holder/profiler PID, birth, parent or all four UIDs changed")


def collect_metadata(plan, directory, lock_fd, parent_lock_receipt=None):
    """Prove own/inherited custody before CUDA and preserve partial failures."""
    role = "owner" if parent_lock_receipt is None else "inherited"
    validate_plan(plan)
    directory = metadata_directory(plan, directory, role)
    value = {"schema_version": 1, "kind": "serval02_card1_process_metadata", "profile": PROFILE,
        "capability": CAPABILITY, "role": role, "output_directory": str(directory), **BOUNDARY}
    try:
        require(not any(name.split(".", 1)[0] in {"jax", "jaxlib", "numpy", "torch"} for name in sys.modules), "direct admission must precede numerical imports")
        validate_plan(plan); value["source"] = source_identity(plan)
        value["history"] = validate_history(plan)
        value["current_admission"] = current_admission_claim(plan)
        check_live_admission(value["current_admission"])
        value["before"] = current_process(); process = validate_process(value["before"])
        value["environment"] = {key: val for key, val in os.environ.items() if key == "CUDA_VISIBLE_DEVICES" or key.startswith("SLURM_")}
        require(value["environment"] == {"CUDA_VISIBLE_DEVICES": PROFILE["gpu_uuid"]}, "the direct UUID selector changed or Slurm allocation variables are present")
        value["lock_before"] = lock_metadata(lock_fd)
        value["current_launch"] = launch_inputs(plan)
        if parent_lock_receipt is None:
            validate_lock(value["lock_before"], process["pid"])
        else:
            require(parent_lock_receipt["path"] == str(unit_directory(plan) / "supervisor/direct_process_metadata.json"), "parent lock proof is not this exact supervisor")
            value["parent_lock_receipt"] = parent_lock_receipt; parent = read_receipt(parent_lock_receipt)
            value["parent_metadata"] = parent
            require(parent.get("role") == "owner", "inherited custody cannot use another inherited child")
            validate_metadata(plan, parent)
            holder = validate_process(parent["before"])
            validate_lock(value["lock_before"], holder["pid"])
            value["capture_release"] = read_receipt(receipt(unit_directory(plan) / "capture_release.json"))
            value["profiler_process"] = read_receipt(receipt(unit_directory(plan) / "profiler_process.json"))
            require(value["current_launch"]["process_row"]["pid"] == holder["pid"]
                and value["current_launch"]["process_row"]["start_time"] == holder["process_start_time"]
                and value["profiler_process"]["parent_pid"] == holder["pid"]
                and value["profiler_process"]["parent_start_time"] == holder["process_start_time"]
                and process["process_group"] == value["profiler_process"]["pid"], "recorded holder/profiler is not this current direct launch")
            value["holder_now"] = capture_owner_process(holder["pid"])
            value["profiler_now"] = capture_owner_process(value["profiler_process"]["pid"])
        value["after"] = current_process(); value["lock_after"] = lock_metadata(lock_fd)
        validate_metadata(plan, value)
    except BaseException as error:
        value.update(capture_error={"type": type(error).__name__, "error": str(error)})
        raise
    finally:
        publish(directory / "direct_process_metadata.json", value)
    return value


def live_custody(metadata, fd):
    """Recheck only this process and its proved descriptor; never reacquire the lock."""
    if metadata.get("role") == "producer_guarded_child":
        return adjacent("direct_profile_process").live_capture_custody(metadata, fd, sys.modules[__name__])
    current = current_process(); process = validate_process(current)
    require(process == metadata["after"]["identity"]
        and {key: val for key, val in os.environ.items() if key == "CUDA_VISIBLE_DEVICES" or key.startswith("SLURM_")}
            == metadata["environment"], "current direct process or original selector changed")
    holder = metadata["parent_metadata"]["before"]["identity"] if metadata["role"] in {"inherited", "producer_guarded_child"} else process
    value = lock_metadata(fd); validate_lock(value, holder["pid"])
    require(value == metadata["lock_after"], "the direct descriptor changed after admission")
    if metadata["role"] == "inherited":
        validate_owner_process(capture_owner_process(holder["pid"]), holder["pid"], holder["process_start_time"])
        owner = metadata["profiler_process"]
        validate_owner_process(capture_owner_process(owner["pid"]), owner["pid"], owner["process_start_time"], holder["pid"])
    return {"process": process, "lock": value}


def native_identity(device, observation):
    """Use the already mapped direct driver and current context; create or reset nothing."""
    require(device.platform == "gpu" and device.device_kind == PROFILE["model"]
        and type(device.local_hardware_id) is int and device.local_hardware_id == 0, "one actual direct H100 NVL JAX device is required")
    observation.update(pid=os.getpid(), jax_local_hardware_id=0, jax_device_kind=device.device_kind,
        jax_identity_basis="sole local JAX GPU, sole CUDA-visible ordinal, and that device's current CUDA context")
    with Path("/proc/self/maps").open() as stream:
        raw = stream.read(1024 * 1024 + 1)
    require(len(raw.encode()) <= 1024 * 1024, "own mapped-image metadata exceeds its bound")
    selected = "".join(line + "\n" for line in raw.splitlines() if line.split() and Path(line.split()[-1]).name.startswith("libcuda.so"))
    observation.update(mapped_driver_bytes=selected, mapped_driver_sha256=digest(selected.encode()))
    paths = {line.split()[-1] for line in selected.splitlines()}
    require(len(paths) == 1, "one already mapped direct CUDA driver is required")
    path, = paths
    require(path.endswith("." + PROFILE["driver"]), "actual direct driver differs from its separately reviewed profile")
    observation.update(driver_library=path, driver_library_sha256=receipt(path)["sha256"])
    driver = ctypes.CDLL(path, mode=os.RTLD_LOCAL | os.RTLD_NOLOAD)
    signatures = {"cuDeviceGetCount": [ctypes.POINTER(ctypes.c_int)], "cuDeviceGet": [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
        "cuCtxGetCurrent": [ctypes.POINTER(ctypes.c_void_p)], "cuCtxGetDevice": [ctypes.POINTER(ctypes.c_int)],
        "cuDeviceGetUuid_v2": [ctypes.c_void_p, ctypes.c_int], "cuDeviceGetPCIBusId": [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]}
    for name, arguments in signatures.items():
        function = getattr(driver, name); function.argtypes, function.restype = arguments, ctypes.c_int
    count, handle, current, context = ctypes.c_int(), ctypes.c_int(), ctypes.c_int(), ctypes.c_void_p()
    codes = {"cuDeviceGetCount": driver.cuDeviceGetCount(ctypes.byref(count)), "cuDeviceGet": driver.cuDeviceGet(ctypes.byref(handle), 0),
        "cuCtxGetCurrent": driver.cuCtxGetCurrent(ctypes.byref(context)), "cuCtxGetDevice": driver.cuCtxGetDevice(ctypes.byref(current))}
    observation.update(api_return_codes=codes, cuda_device_count=count.value, cuda_device_handle=handle.value,
        current_context=context.value, current_context_device=current.value)
    require(all(type(code) is int and code == 0 for code in codes.values()) and count.value == 1
        and context.value and current.value == handle.value, "the sole visible CUDA device needs its existing current context")
    raw_uuid, bus = (ctypes.c_ubyte * 16)(), ctypes.create_string_buffer(32)
    codes["cuDeviceGetUuid_v2"] = driver.cuDeviceGetUuid_v2(ctypes.byref(raw_uuid), handle.value)
    codes["cuDeviceGetPCIBusId"] = driver.cuDeviceGetPCIBusId(bus, len(bus), handle.value)
    require(all(code == 0 for code in codes.values()), "native direct CUDA UUID/PCI failed")
    observation.update(gpu_uuid="GPU-" + str(uuid.UUID(bytes=bytes(raw_uuid))), pci_address=common.pci(bus.value.decode()))
    require(observation["gpu_uuid"] == PROFILE["gpu_uuid"], "the current CUDA context is not the authorized card-one UUID")
    return observation


def bind_process(plan, metadata, jax_device, directory):
    """Resolve one attempt and preserve its partial failures, without changing the plan."""
    validate_plan(plan)
    directory = binding_directory(plan, metadata, directory)
    require(not any((directory / name).exists() for name in ("direct_device_binding.json", "direct_device_binding_attempt.json")), "preserve the previous direct binding attempt")
    attempt = {"schema_version": 1, "kind": "serval02_card1_device_binding_attempt", "profile": PROFILE,
        "capability": CAPABILITY, "native": {}, **BOUNDARY}
    try:
        validate_metadata(plan, metadata)
        check_live_admission(metadata["current_admission"])
        attempt["custody_before"] = live_custody(metadata, metadata["lock_after"]["fd"])
        require("jax" in sys.modules and sys.modules["jax"].local_devices(backend="gpu") == [jax_device], "the actual process must expose exactly the one direct JAX GPU")
        native = native_identity(jax_device, attempt["native"])
        gpu, bus = native["gpu_uuid"], native["pci_address"]
        command = ["nvidia-smi", "-i", gpu, "--query-gpu=uuid,name,pci.bus_id,memory.total,driver_version", "--format=csv,noheader,nounits"]
        attempt["hardware"] = common.bounded_query(command)
        common.validate_query(attempt["hardware"], command)
        path = Path("/proc/driver/nvidia/gpus") / bus / "information"
        attempt["proc"] = {"path": str(path), "read_status": "started"}
        with path.open() as stream:
            raw = stream.read(32769)
        attempt["proc"] = {"path": str(path), "raw": raw, "sha256": digest(raw.encode())}
        require(len(raw.encode()) <= 32768, "assigned direct proc metadata exceeds its bound")
        minors = re.findall(r"^Device Minor:\s*([0-9]+)\s*$", raw, re.M)
        require(len(minors) == 1, "one physical proc minor is required; never use compute-capability minor")
        path = Path("/dev/nvidia" + minors[0]); info = path.lstat()
        attempt["device"] = {"path": str(path), "character_device": stat.S_ISCHR(info.st_mode), "symlink": stat.S_ISLNK(info.st_mode),
            "major": os.major(info.st_rdev), "minor": os.minor(info.st_rdev)}
        attempt["driver_capture_expected_path"] = str(directory / "captured_driver" / Path(native["driver_library"]).name)
        attempt["driver_capture"] = adjacent("h100_nvl_slurm_site_validation").capture_driver_image(native, directory)
        attempt["custody_after"] = live_custody(metadata, metadata["lock_after"]["fd"])
        check_live_admission(metadata["current_admission"])
        value = {"schema_version": 1, "kind": "serval02_card1_current_process_resolved_device_binding", "profile": PROFILE,
            "capability": CAPABILITY, "plan": receipt(unit_directory(plan) / "measurement_plan.json"),
            "output_directory": str(directory),
            "metadata_sha256": digest(encoded(metadata)), "process": metadata["after"]["identity"],
            "original_environment": metadata["environment"], "gpu_uuid": gpu, "pci_address": bus,
            "physical_minor": int(minors[0]), "physical_total_bytes": PROFILE["memory_mib"] * 1024 * 1024,
            **{key: attempt[key] for key in ("native", "hardware", "proc", "device", "driver_capture", "custody_before", "custody_after")},
            "startup_memory_observed": False, "other_gpu_processes_observed": False, **BOUNDARY}
        validate_binding(plan, metadata, value, directory)
        publish(directory / "direct_device_binding.json", value)
        attempt.update(passed=True, binding=receipt(directory / "direct_device_binding.json"))
        return value
    except BaseException as error:
        attempt.update(passed=False, error_type=type(error).__name__, error=str(error)); raise
    finally:
        publish(directory / "direct_device_binding_attempt.json", attempt)


def validate_binding(plan, metadata, value, directory=None):
    """Validate only direct proof; no old/default/UVA/final reader admission follows."""
    import csv
    validate_metadata(plan, metadata)
    directory = binding_directory(plan, metadata, unit_directory(plan) if directory is None else directory)
    require(type(value.get("schema_version")) is int and value["schema_version"] == 1
        and value.get("kind") == "serval02_card1_current_process_resolved_device_binding"
        and value.get("capability") == CAPABILITY, "an exact separate direct binding is required")
    require_profile(value["profile"])
    require(all(exact(value.get(key), expected) for key, expected in BOUNDARY.items())
        and value.get("startup_memory_observed") is False and value.get("other_gpu_processes_observed") is False
        and value.get("output_directory") == str(directory)
        and value["plan"] == receipt(unit_directory(plan) / "measurement_plan.json") and read_receipt(value["plan"]) == plan
        and value["metadata_sha256"] == digest(encoded(metadata)) and value["process"] == metadata["after"]["identity"]
        and value["original_environment"] == metadata["environment"], "direct binding changed source, process, selector or claim scope")
    process = value["process"]; native = value["native"]
    codes = {name: 0 for name in ("cuDeviceGetCount", "cuDeviceGet", "cuCtxGetCurrent", "cuCtxGetDevice", "cuDeviceGetUuid_v2", "cuDeviceGetPCIBusId")}
    require(type(native["pid"]) is int and native["pid"] == process["pid"] and type(native["cuda_device_count"]) is int and native["cuda_device_count"] == 1
        and type(native["jax_local_hardware_id"]) is int and native["jax_local_hardware_id"] == 0
        and type(native["cuda_device_handle"]) is int and native["cuda_device_handle"] >= 0
        and type(native["current_context"]) is int and native["current_context"] > 0
        and type(native["current_context_device"]) is int and native["current_context_device"] == native["cuda_device_handle"]
        and native["jax_device_kind"] == PROFILE["model"] and exact(native["api_return_codes"], codes)
        and native["jax_identity_basis"] == "sole local JAX GPU, sole CUDA-visible ordinal, and that device's current CUDA context"
        and native["gpu_uuid"] == value["gpu_uuid"] == PROFILE["gpu_uuid"] and common.pci(native["pci_address"]) == value["pci_address"], "same-process direct context/UUID/PCI/JAX relation changed")
    require(native["driver_library"].endswith("." + PROFILE["driver"])
        and digest(native["mapped_driver_bytes"].encode()) == native["mapped_driver_sha256"]
        and {line.split()[-1] for line in native["mapped_driver_bytes"].splitlines()} == {native["driver_library"]}, "actually mapped direct driver identity changed")
    hardware = value["hardware"]
    common.validate_query(hardware, ["nvidia-smi", "-i", PROFILE["gpu_uuid"], "--query-gpu=uuid,name,pci.bus_id,memory.total,driver_version", "--format=csv,noheader,nounits"])
    rows = list(csv.reader(hardware["stdout"].splitlines())); require(len(rows) == 1 and len(rows[0]) == 5, "one exact assigned direct hardware row required")
    gpu, model, bus, memory, driver = [item.strip() for item in rows[0]]
    require(gpu == PROFILE["gpu_uuid"] and model == PROFILE["model"] and common.pci(bus) == value["pci_address"]
        and memory == str(PROFILE["memory_mib"]) and driver == PROFILE["driver"]
        and type(value["physical_total_bytes"]) is int and value["physical_total_bytes"] == int(memory) * 1024 * 1024,
        "fresh physical UUID/model/driver/total changed or received an arbitrary override")
    proc = value["proc"]
    require(proc["path"] == "/proc/driver/nvidia/gpus/" + value["pci_address"] + "/information"
        and digest(proc["raw"].encode()) == proc["sha256"], "current direct proc path or bytes changed")
    for key, expected in (("GPU UUID", gpu), ("Model", model), ("Bus Location", value["pci_address"]), ("Device Minor", str(value["physical_minor"]))):
        require(re.findall(r"^" + re.escape(key) + r":\s*(.*?)\s*$", proc["raw"], re.M) == [expected], "direct native identity does not join proc/minor metadata")
    require(type(value["physical_minor"]) is int and value["physical_minor"] >= 0
        and exact(value["device"], {"path": "/dev/nvidia" + str(value["physical_minor"]), "character_device": True,
            "symlink": False, "major": 195, "minor": value["physical_minor"]}), "physical direct character device changed")
    holder = metadata["parent_metadata"]["before"]["identity"] if metadata["role"] in {"inherited", "producer_guarded_child"} else process
    for custody in (value["custody_before"], value["custody_after"]):
        require(custody["process"] == process and custody["lock"] == metadata["lock_after"], "direct window custody differs from admission")
        validate_lock(custody["lock"], holder["pid"])
    capture = value["driver_capture"]; target = directory / "captured_driver" / Path(native["driver_library"]).name
    require(type(capture.get("schema_version")) is int and capture["schema_version"] == 1 and capture.get("kind") == "captured_mapped_cuda_driver_image"
        and type(capture.get("pid")) is int and capture["pid"] == process["pid"]
        and capture["original"] == {"path": native["driver_library"], "sha256": native["driver_library_sha256"]}
        and capture["captured"] == {"path": str(target), "sha256": native["driver_library_sha256"]}
        and type(capture["bytes"]) is int and capture["bytes"] > 0 and receipt(target) == capture["captured"]
        and target.stat().st_size == capture["bytes"], "immutable direct mapped driver copy changed")
    return value


def validate_metadata(plan, value):
    if value.get("role") == "producer_guarded_child":
        return adjacent("direct_profile_process").validate_capture_metadata(plan, value, sys.modules[__name__])
    validate_plan(plan)
    require(type(value.get("schema_version")) is int and value["schema_version"] == 1 and value.get("kind") == "serval02_card1_process_metadata"
        and value.get("capability") == CAPABILITY and "capture_error" not in value, "partial or unsupported direct metadata")
    require_profile(value["profile"])
    metadata_directory(plan, value["output_directory"], value["role"])
    require(all(exact(value.get(key), expected) for key, expected in BOUNDARY.items()), "direct metadata cannot grant admission")
    before, after = validate_process(value["before"]), validate_process(value["after"])
    require(before == after and value["environment"] == {"CUDA_VISIBLE_DEVICES": PROFILE["gpu_uuid"]}
        and value["source"] == source_identity(plan) and value["history"] == validate_history(plan)
        and value["current_launch"] == launch_inputs(plan)
        and value["current_admission"] == current_admission_claim(plan), "current direct process, source, frozen history or launch changed")
    row = value["current_launch"]["process_row"]
    holder = before
    if value["role"] == "inherited":
        require(plan["direct_binding"]["consumer"] == "component_profile", "only an explicit profile child may inherit custody")
        parent = value["parent_metadata"]
        require(value["parent_lock_receipt"]["path"] == str(unit_directory(plan) / "supervisor/direct_process_metadata.json")
            and read_receipt(value["parent_lock_receipt"]) == parent and parent["role"] == "owner", "parent custody receipt changed")
        validate_metadata(plan, parent); holder = parent["before"]["identity"]
        owner, release = value["profiler_process"], value["capture_release"]
        require(release == {**owner, "capture_identity": {"hostname": "serval02", "pid": before["pid"], "process_start_time": before["process_start_time"]}}
            and owner["parent_pid"] == holder["pid"] and owner["parent_start_time"] == holder["process_start_time"]
            and before["process_group"] == owner["pid"] and value["lock_before"]["fd"] == parent["lock_before"]["fd"], "actual parent/Nsight/child custody chain changed")
        validate_owner_process(value["holder_now"], holder["pid"], holder["process_start_time"])
        validate_owner_process(value["profiler_now"], owner["pid"], owner["process_start_time"], holder["pid"])
    else:
        require(value["role"] == "owner" and not any(key in value for key in ("parent_metadata", "parent_lock_receipt", "holder_now")), "unsupported direct custody role")
    require(row["pid"] == holder["pid"] and row["start_time"] == holder["process_start_time"], "current launch row does not name this actual lock owner")
    validate_lock(value["lock_before"], holder["pid"]); validate_lock(value["lock_after"], holder["pid"])
    require(value["lock_before"] == value["lock_after"], "direct lock custody changed during metadata capture")
    return value
