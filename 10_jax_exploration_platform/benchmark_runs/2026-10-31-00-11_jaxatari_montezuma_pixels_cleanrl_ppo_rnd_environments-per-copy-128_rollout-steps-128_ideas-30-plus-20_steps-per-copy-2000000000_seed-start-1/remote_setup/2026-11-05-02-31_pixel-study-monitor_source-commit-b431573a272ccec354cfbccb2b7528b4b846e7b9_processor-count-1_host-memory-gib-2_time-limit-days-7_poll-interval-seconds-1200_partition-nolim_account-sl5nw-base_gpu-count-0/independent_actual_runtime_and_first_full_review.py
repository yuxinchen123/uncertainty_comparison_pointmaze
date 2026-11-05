"""Independently verify the replacement monitor's actual runtime and first full report."""
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import tarfile
import time
from zoneinfo import ZoneInfo

P = Path(__file__).resolve().parent
R = P.parent.parent
REPO = R.parents[2]
JOB = "6821437"
OLD = "6783003"
SOURCE = "b431573a272ccec354cfbccb2b7528b4b846e7b9"
UID = 2618919
NUMERICAL = {"jax", "jaxlib", "numpy", "torch", "flax", "optax", "cupy", "scipy", "jaxtari", "exploration_platform"}
START = time.monotonic()
CHECKS = []
PINNED = {}
ALLOWED_COMMANDS = []


def require(condition, statement):
    """Record every actual required assertion and stop at a failed one."""
    assert condition, statement
    CHECKS.append(statement)


def guard(event, args):
    """Refuse numerical, network, device and unapproved subprocess operations."""
    # Only exact declared local source and owned scheduler reads may spawn.
    if event == "import" and args[0].split(".")[0] in NUMERICAL:
        raise RuntimeError("numerical imports are forbidden in this runtime review")
    if event in {"socket.connect", "socket.bind", "ctypes.dlopen"}:
        raise RuntimeError("network and library-load operations are forbidden")
    if event == "open" and isinstance(args[0], (str, bytes)) and os.fsdecode(args[0]).startswith("/dev/nvidia"):
        raise RuntimeError("GPU device access is forbidden")
    if event == "subprocess.Popen" and args[1] not in ALLOWED_COMMANDS:
        raise RuntimeError("subprocess is outside the exact local/owned read allowlist")


def data(path):
    """Read each ordinary small file once and retain its exact CPU-local identity."""
    # Cache bytes only for this validation; later identity changes still refuse.
    path = Path(path)
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and not path.is_symlink(), "regular metadata or source file: " + str(path))
    identity = (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    if path not in PINNED:
        raw = path.read_bytes()
        require(len(raw) == info.st_size, "complete small file read: " + str(path))
        PINNED[path] = (identity, raw)
    require(PINNED[path][0] == identity, "unchanged validation file: " + str(path))
    return PINNED[path][1]


def document(path):
    """Read one exact small JSON document."""
    return json.loads(data(path))


def checksum(raw):
    """Compute the SHA256 of an already read small byte sequence."""
    return hashlib.sha256(raw).hexdigest()


def reference(path):
    """Name the exact saved bytes of a small local document."""
    return {"path": str(path), "sha256": checksum(data(path))}


def field(raw, name):
    """Require one unambiguous native scheduler field."""
    values = re.findall(r"(?:^|\s)" + re.escape(name) + r"=([^\s]*)", raw)
    require(len(values) == 1, "one native scheduler field: " + name)
    return values[0]


def tres(raw):
    """Keep every explicit resource key and reject duplicates."""
    pairs = [part.split("=", 1) for part in raw.split(",")]
    require(all(len(pair) == 2 for pair in pairs) and len({pair[0] for pair in pairs}) == len(pairs), "unambiguous scheduler resource keys")
    return dict(pairs)


def process(pid, cgroup_expected):
    """Read only a process in this exact owned batch after checking all four UIDs."""
    root = Path("/proc") / str(pid)
    status = (root / "status").read_text()
    values = re.findall(r"^Uid:\s+(.+)$", status, re.M)
    require(len(values) == 1 and [int(value) for value in values[0].split()] == [UID] * 4, "all four UIDs of actual owned batch process")
    cgroup = (root / "cgroup").read_text()
    require(cgroup == cgroup_expected, "exact actual batch cgroup before further process reads")
    stat_raw = (root / "stat").read_text()
    return {"pid": pid, "birth": int(stat_raw.rsplit(")", 1)[1].split()[19]), "uids": [UID] * 4,
            "cgroup": cgroup, "argv": [item.decode() for item in (root / "cmdline").read_bytes().split(b"\0")[:-1]],
            "fd1": os.readlink(root / "fd/1"), "fd2": os.readlink(root / "fd/2")}


def save(name, raw):
    """Durably retain a new review artifact without replacing any old result."""
    path = P / name
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return {"path": str(path), "sha256": checksum(raw), "bytes": len(raw)}


# Install the metadata-only audit after ordinary standard-library imports.
sys.addaudithook(guard)
require(os.environ.get("SLURM_JOB_ID") == JOB and os.environ.get("SLURM_STEP_ID", "").isdigit(), "actual numeric step inside replacement monitor allocation")
require(os.getresuid() == (UID, UID, UID), "actual reviewer UID values")
review_cgroup = Path("/proc/self/cgroup").read_text()
require("/job_" + JOB + "/step_" + os.environ["SLURM_STEP_ID"] + "/" in review_cgroup, "actual reviewer kernel cgroup joins owned new step")
for ledger in ("submitted_jobids.txt", "submitted_jobids_uva.txt"):
    require((R / "slurm" / ledger).read_text().splitlines().count(JOB) == 1, "exact new ID occurs once in owned " + ledger)
rows = [line.split("\t") for line in (R / "slurm/submissions.tsv").read_text().splitlines() if line.split("\t")[0] == JOB]
require(rows == [[JOB, "monitor", SOURCE, str(P / "monitor.sbatch")]], "exact actual source/script submission row")

# Join the actual release to the prepared plan and submission rather than a guessed job.
plan = document(P / "monitor_replacement_plan.json")
submission = document(P / "monitor_submission_receipt.json")
release = document(P / "monitor_release_receipt.json")
binding = document(P / "monitor_release_binding.json")
require(submission["job_id"] == JOB and submission["returncode"] == 0 and submission["submitted_held"] is True, "actual once-held new submission")
require(submission["plan"] == reference(P / "monitor_replacement_plan.json"), "actual submission joins frozen plan")
require(release["job_id"] == JOB and release["returncode"] == 0 and release["binding"] == reference(P / "monitor_release_binding.json") and binding["command"] == ["scontrol", "release", JOB], "actual reviewed single release")
allocation = document(P / "runtime_allocation.json")
require(allocation["job_id"] == JOB and allocation["returncode"] == 0 and allocation["stderr"] == "" and allocation["uid"] == UID, "actual successful source-gated startup allocation record")
host = socket.gethostname().split(".")[0]
require(allocation["hostname"].split(".")[0] == host and not any(word in host for word in ("portal", "login")), "actual reviewer and monitor allocated compute host")
query = ["scontrol", "show", "job", JOB, "-o"]
ALLOWED_COMMANDS.append(query)
current = subprocess.run(query, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=20, env={**os.environ, "TZ": "UTC"})
require(current.stderr == b"" and len(current.stdout.splitlines()) == 1, "fresh exact owned scheduler job read")
raw = current.stdout.decode()
for key, value in {"JobId": JOB, "JobState": "RUNNING", "UserId": "sl5nw(2618919)", "Account": "sl5nw_base", "Partition": "nolim", "QOS": "normal", "NodeList": host, "BatchHost": host, "Requeue": "0", "Restarts": "0", "NumTasks": "1", "CPUs/Task": "1", "NumNodes": "1", "MinMemoryNode": "2G", "TimeLimit": "7-00:00:00", "Command": str(P / "monitor.sbatch")}.items():
    require(field(raw, key) == value, "actual current allocation " + key)
requested, allocated = tres(field(raw, "ReqTRES")), tres(field(raw, "AllocTRES"))
require(requested["cpu"] == "1" and allocated["cpu"] in {"1", "2"} and allocated["cpu"] == field(raw, "NumCPUs"), "exact one-CPU request and actual allocated logical count")
for resources in (requested, allocated):
    require(resources["mem"] == "2G" and resources["node"] == "1", "exact two-GiB one-node resources")
    require(all(value == "0" for key, value in resources.items() if key.startswith("gres/gpu")), "zero requested and allocated GPUs")
node = allocation["node_observation"]
require(node["returncode"] == 0 and node["stderr"] == "" and node["command"] == ["scontrol", "show", "node", host, "-o"], "actual startup topology belongs to allocated host")
require(field(node["stdout"], "NodeName") == host, "one exact allocated topology node")
if allocated["cpu"] == "2":
    require(field(node["stdout"], "ThreadsPerCore") == "2", "two logical CPUs allowed only with actual two-thread core")

# Check the exact 287-file Git source and small archive without reading scientific payloads.
manifest = document(P / "source_snapshot.json")
require(manifest["source_commit"] == SOURCE and len(manifest["files"]) == 287, "exact source commit and 287 source files")
source = Path(manifest["source_root"])
require(source.resolve(strict=True) == source, "ordinary source root")
tree_command = ["git", "-C", str(REPO), "ls-tree", "-r", "-z", SOURCE, "--"] + manifest["source_prefixes"]
ALLOWED_COMMANDS.append(tree_command)
tree_raw = subprocess.check_output(tree_command, timeout=30)
tree = {}
for row in tree_raw.split(b"\0"):
    if row:
        header, name = row.split(b"\t", 1)
        mode, kind, blob = header.decode().split()
        require(kind == "blob" and name.decode() not in tree, "one ordinary committed source blob")
        tree[name.decode()] = (mode, blob)
require(set(tree) == set(manifest["files"]), "complete declared source Git prefix coverage")
paths = list(source.rglob("*"))
require(not any(path.is_symlink() for path in paths), "no source snapshot symlinks")
require({str(path.relative_to(source)) for path in paths if path.is_file()} == set(manifest["files"]), "no missing or additional staged source files")
source_checks = {}
for name, expected in manifest["files"].items():
    path = source / name
    contents = data(path)
    blob = hashlib.sha1(b"blob " + str(len(contents)).encode() + b"\0" + contents).hexdigest()
    require(checksum(contents) == expected["sha256"] and len(contents) == expected["bytes"] and blob == expected["git_blob"], "exact staged source bytes/Git blob: " + name)
    require(tree[name] == (expected["git_mode"], blob), "actual committed mode/blob: " + name)
    require(stat.S_IMODE(path.stat().st_mode) & 0o111 == (0o111 if expected["git_mode"] == "100755" else 0), "actual source executable mode: " + name)
    source_checks[name] = {"sha256": checksum(contents), "git_blob": blob, "git_mode": expected["git_mode"], "bytes": len(contents)}
archive = data(P / "source_git_archive.tar")
require(checksum(archive) == manifest["archive"]["sha256"], "exact prepared source archive bytes")
archive_checks = {}
with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as handle:
    for member in handle.getmembers():
        if member.isdir():
            continue
        require(member.isfile() and member.name.startswith(SOURCE + "/"), "ordinary exact source archive member")
        name = member.name[len(SOURCE) + 1:]
        require(name in manifest["files"] and name not in archive_checks, "one declared archive file")
        contents = handle.extractfile(member).read()
        expected = manifest["files"][name]
        require(checksum(contents) == expected["sha256"] and len(contents) == expected["bytes"] and member.mode & 0o111 == (0o111 if expected["git_mode"] == "100755" else 0), "exact archive bytes/mode: " + name)
        archive_checks[name] = checksum(contents)
require(set(archive_checks) == set(manifest["files"]), "complete archive coverage")
freeze = document(P / "static_source_freeze.json")
require(len(freeze["files"]) == 322 and freeze["source_commit"] == SOURCE, "all 322 reviewed static artifacts")
for name, expected in freeze["files"].items():
    path = Path(name)
    contents = data(path)
    require(path.is_relative_to(R) and checksum(contents) == expected["sha256"] and len(contents) == expected["bytes"] and oct(stat.S_IMODE(path.stat().st_mode)) == expected["mode"], "exact static artifact bytes/mode: " + name)

# Identify only actual owned batch members and bind the live monitor to the source and log.
startup_cgroup = allocation["cgroup"]
components = [line.split(":", 2)[2] for line in startup_cgroup.splitlines() if line.startswith("0::")]
require(len(components) == 1 and "/job_" + JOB + "/step_batch/" in components[0], "actual startup batch cgroup path")
batch_path = Path("/sys/fs/cgroup") / components[0].lstrip("/")
members_raw = (batch_path / "cgroup.procs").read_bytes()
members = [int(value) for value in members_raw.split()]
require(members and len(members) == len(set(members)), "actual nonempty unique owned batch member list")
expected_argv = ["/usr/bin/python3", "-B", str(source / "10_jax_exploration_platform/scripts/atari_pixel_campaign/monitor.py"), "--run-dir", str(R), "--loop", "--interval", "1200"]
matching = []
for pid in members:
    actual = process(pid, startup_cgroup)
    if actual["argv"] == expected_argv:
        matching.append(actual)
require(len(matching) == 1, "one live exact owned monitor process")
monitor = matching[0]
log_path = R / "slurm/jobs" / ("monitor_" + JOB + ".log")
require(monitor["fd1"] == monitor["fd2"] == str(log_path), "actual monitor stdout/stderr descriptors join exact owned log")

# Join the real first full log report to status and all five display sections.
log_raw = data(log_path)
header, body = log_raw.decode().split("\n", 1)
require(json.loads(header) == {"source_commit": SOURCE, "files_verified": 287, "full_tick_executed": False, "numerical_imports": False}, "actual source guard completed before full loop")
report_raw = data(R / "20_mins_monitoring/latest_report.md")
status_raw = data(R / "20_mins_monitoring/latest_status.json")
report = report_raw.decode()
status = json.loads(status_raw)
table = report.split("```text\n", 1)[1].rsplit("```", 1)[0]
require(body == table + "\n", "exact first full new-monitor log joins current report without mixed monitor writes")
sections = [line for line in table.splitlines() if line in ["TRAINING", "WAITING", "HELD", "FINISHED", "PLANNED"]]
require(sections == ["TRAINING", "WAITING", "HELD", "FINISHED", "PLANNED"], "all five complete report sections")
report_time = datetime.datetime.strptime(status["time"], "%Y-%m-%d %H:%M:%S PT").replace(tzinfo=ZoneInfo("America/Los_Angeles")).astimezone(datetime.timezone.utc)
released_time = datetime.datetime.fromisoformat(release["at_utc"])
startup_time = datetime.datetime.fromisoformat(allocation["observed_at_utc"])
require(report_time >= startup_time > released_time and datetime.datetime.now(datetime.timezone.utc) - report_time < datetime.timedelta(minutes=10), "actual first full report chronology after release and source/allocation guard")
require(status["time"] in report and status["campaign"]["queue_counts"] == {"pending": 0, "running": 0, "completed": 120, "failed": 0}, "actual complete initial comparison counts")
require(status["campaign"].get("authorization_error") is None and status["campaign"].get("final_verification_error") is None, "no actual controller admission errors")
jobs = {row[0]: row for row in status["jobs"]}
require(jobs[JOB][1] == "RUNNING" and jobs[JOB][2] == host, "actual complete report includes this running replacement")
require(jobs[OLD][1] == "RUNNING", "old monitor remains active before independent adoption")
first_copy = [unit for unit in status["units"] if unit["run"].startswith("2026-11-05-02-24_common-timing_")]
require(len(first_copy) == 1, "one actual prepared first-copy timing display row")
row = first_copy[0]
require(row["state"] == "planned" and row["slurm_job_id"] == "" and row["direct_metadata_display_only"] is True and row["scientific_credit"] == 0 and row["admission_authority"] is False, "bound unlaunched H100 timing remains PLANNED with zero scientific authority")
require(status["final_training"]["state"] == "disabled", "final training remains disabled")
current_actions = json.loads((R / "next_actions.json").read_bytes())
proposed_index = document(P / "proposed_post_initial_reporting_evidence.json")
require(current_actions["post_initial_reporting_evidence"] == proposed_index and len(proposed_index["common_correctness"]) == 4 and len(proposed_index["common_confirmation"]) == 2, "only six genuine reviewed reporting entries and no fabricated H100 timing results")
require(process(monitor["pid"], startup_cgroup) == monitor, "same actual batch monitor UID/birth/cgroup/argv/log descriptors after review")
for path, (identity, contents) in PINNED.items():
    info = path.lstat()
    require(identity == (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_size, info.st_mtime_ns, info.st_ctime_ns), "all read small file identities remain unchanged: " + str(path))
require(not NUMERICAL.intersection(sys.modules), "no numerical modules imported")

# Save independent immutable raw copies and truthful zero-credit scope before handover.
snapshots = {name: save(name, raw) for name, raw in {"independent_first_report.md": report_raw, "independent_first_status.json": status_raw, "independent_first_monitor_log.raw": log_raw, "independent_runtime_allocation.json": data(P / "runtime_allocation.json"), "independent_batch_cgroup_procs.raw": members_raw, "independent_source_git_tree.raw": tree_raw, "independent_current_owned_job.raw": current.stdout}.items()}
review = {"schema_version": 1, "kind": "independent_actual_cpu_monitor_runtime_and_first_full_report", "passed": True, "job_id": JOB, "source_commit": SOURCE, "actual_reviewer_job_id": JOB, "actual_reviewer_step_id": os.environ["SLURM_STEP_ID"], "reviewer_hostname": socket.gethostname(), "reviewer_pid": os.getpid(), "reviewer_cgroup": review_cgroup, "reviewer_uids": list(os.getresuid()), "monitor_process": monitor, "source_file_count": 287, "static_artifact_count": 322, "archive_sha256": checksum(archive), "source_manifest": reference(P / "source_snapshot.json"), "static_freeze": reference(P / "static_source_freeze.json"), "source_files": source_checks, "archive_files": archive_checks, "requested_cpus": 1, "allocated_logical_cpus": int(allocated["cpu"]), "threads_per_core": int(field(node["stdout"], "ThreadsPerCore")), "host_memory_gib": 2, "gpu_count": 0, "requeue": 0, "restarts": 0, "interval_seconds": 1200, "report_sections": sections, "campaign_counts": status["campaign"]["queue_counts"], "actual_first_copy_display": row, "snapshots": snapshots, "checks": CHECKS, "elapsed_seconds": time.monotonic() - START, "finished_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "full_tick_executed_by_reviewer": False, "historical_large_payloads_rehashed": False, "own_process_stdlib_imports_completed_before_audit": True, "ctypes_library_loads_requested": 0, "numerical_imports": [], "gpu_operations": 0, "remote_actions": 0, "actual_scheduler_job_actions": 0, "scientific_credit": 0, "worker_admission": False, "old_monitor_stopped_or_adopted_by_reviewer": False}
receipt = save("independent_actual_runtime_and_first_full_review.json", (json.dumps(review, indent=2, sort_keys=True) + "\n").encode())
print(json.dumps({"passed": True, "review": receipt, "checks": len(CHECKS), "elapsed_seconds": time.monotonic() - START, "actual_step_id": os.environ["SLURM_STEP_ID"]}))
