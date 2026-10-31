"""Launch recorded queue workers only after the parent admits an immutable study source.

This parent-side utility imports no array or training library. It can be copied
outside the detached study checkout without changing the admitted numerical code.
"""

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import getpass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET


PIXEL_PYTHON = "/p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python"
SERVAL02_CARD1 = "GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3"
CONTEXTS = ("h100", "jaguar03_rtx_a4500")


def atomic_text(path, value):
    """Replace a metadata file only after its complete contents reach disk."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temporary.open("x") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def append_process(run_dir, record):
    """Append to the same locked, atomic process ledger used by the direct-unit launcher."""
    ledger = Path(run_dir) / "processes.json"
    with ledger.with_suffix(".lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        records = json.loads(ledger.read_text()) if ledger.exists() else []
        records.append(record)
        atomic_text(ledger, json.dumps(records, indent=2) + "\n")


def admitted_source(source_root, run_dir):
    """Require a detached checkout and the controller's complete source-and-gates admission."""
    source_root, run_dir = Path(source_root), Path(run_dir)
    if not source_root.is_absolute() or not run_dir.is_absolute():
        raise ValueError("--source-root and --run-dir must be absolute paths")
    source_root, run_dir = source_root.resolve(strict=True), run_dir.resolve(strict=True)
    branch = subprocess.run(["git", "-C", str(source_root), "symbolic-ref", "--quiet", "HEAD"],
                            capture_output=True, text=True)
    if branch.returncode == 0:
        raise ValueError("the admitted source must be a detached Git checkout")
    if branch.returncode != 1:
        raise RuntimeError("could not verify detached source: " + branch.stderr.strip())

    # Reuse the frozen controller's validation instead of weakening its gate checks here.
    scripts = source_root / "10_jax_exploration_platform/scripts/atari_pixel_campaign"
    sys.path.insert(0, str(scripts))
    specification = importlib.util.spec_from_file_location("pixel_queue_admission", scripts / "controller.py")
    controller = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(controller)
    ready = controller.ready_specification(run_dir)
    if ready is None:
        raise ValueError("the parent has not admitted this source and both hardware checks")
    if Path(ready["source_root"]).resolve() != source_root:
        raise ValueError("--source-root differs from the source admitted in SOURCE_AND_GATES_READY.json")
    return source_root, run_dir, ready["source_commit"]


def allocated_gpu_uuid(environment=None):
    """Match the single visible A4500 to Slurm's physical allocation without an NVML index lookup."""
    environment = os.environ if environment is None else environment
    visible = environment.get("CUDA_VISIBLE_DEVICES", "")
    allocations = [environment[name] for name in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS")
                   if environment.get(name)]
    if not visible or "," in visible or not allocations or any("," in value for value in allocations):
        raise ValueError("one CUDA-visible GPU and one recorded Slurm GPU allocation are required")
    if any(not (value.isdecimal() or value.startswith("GPU-")) for value in allocations):
        raise ValueError("Slurm must identify the single allocated GPU by physical index or UUID")
    # Before: physical GPU five may appear at visible NVML index zero; its minor remains five.
    # This driver's CSV query omits minor_number, so read that field from its XML report.
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name",
        "--format=csv,noheader"], capture_output=True, text=True, check=True, timeout=15)
    rows = list(csv.reader(result.stdout.splitlines()))
    if len(rows) != 1 or len(rows[0]) != 3:
        raise ValueError("nvidia-smi did not resolve exactly the allocated GPU")
    index, gpu_uuid, name = (part.strip() for part in rows[0])
    if not gpu_uuid.startswith("GPU-") or "A4500" not in name:
        raise ValueError("the jaguar03 worker must hold one physical RTX A4500")
    result = subprocess.run(["nvidia-smi", "-q", "-x"], capture_output=True,
                            text=True, check=True, timeout=15)
    devices = ET.fromstring(result.stdout).findall("gpu")
    if (len(devices) != 1 or devices[0].findtext("uuid") != gpu_uuid
            or devices[0].findtext("product_name") != name):
        raise ValueError("nvidia-smi's visible row and physical device report disagree")
    minor = devices[0].findtext("minor_number", "")
    if not index.isdecimal() or not minor.isdecimal():
        raise ValueError("the visible GPU must record a numeric index and physical device minor number")
    # Cross-check physical identity before allowing CUDA's remapped ordinal zero.
    for allocated in allocations:
        if allocated.isdecimal() and int(allocated) != int(minor):
            raise ValueError("Slurm's physical GPU allocation conflicts with the visible device minor number")
        if allocated.startswith("GPU-") and gpu_uuid != allocated:
            raise ValueError("nvidia-smi returned a different GPU UUID from the allocated one")
    if visible not in {"0", index, minor, gpu_uuid}:
        raise ValueError("CUDA visibility conflicts with the physical GPU recorded by Slurm")
    return gpu_uuid


def current_reservation():
    """Read jaguar03's current reservation and use it only when it names this account."""
    text = subprocess.check_output(["scontrol", "show", "node", "jaguar03"], text=True)
    fields = dict(token.split("=", 1) for token in text.split() if "=" in token)
    name = fields.get("ReservationName", "")
    if not name or name == "(null)":
        return None
    text = subprocess.check_output(["scontrol", "show", "reservation", name, "-o"], text=True)
    fields = dict(token.split("=", 1) for token in text.split() if "=" in token)
    users = fields.get("Users", "").split(",")
    if getpass.getuser() not in users and "ALL" not in users:
        raise ValueError("jaguar03's current reservation does not name this submitting account")
    return name


def worker_command(source_root, run_dir, context, python, gpu, *, standing_lock=None):
    """Build one immutable metadata worker with no standing GPU lock held around its queue."""
    worker = Path(source_root) / "10_jax_exploration_platform/scripts/atari_pixel_campaign/queue_worker.py"
    command = [str(python), str(worker), "--run-dir", str(run_dir), "--context", context,
               "--gpu-uuid", gpu, "--python", str(python), "--idle-exit"]
    if standing_lock is not None:
        command += ["--standing-lock", standing_lock]
    return command


def environment_lines(source_root):
    """Set exact source, precision and assets before any numerical child can start."""
    platform = Path(source_root) / "10_jax_exploration_platform"
    environment = platform / "scripts/atari_pixel_campaign/worker_environment.sh"
    return [f"for attempt in {{1..90}}; do test -r {shlex.quote(str(environment))} && break; sleep 2; done",
            f"test -r {shlex.quote(str(environment))}", f"source {shlex.quote(str(environment))}",
            f"export PYTHONPATH={shlex.quote(str(platform / 'src'))}",
            f"cd {shlex.quote(str(source_root))}"]


def launch_direct(source_root, run_dir, commit, python, directory, identifier):
    """Record a serval02 supervisor before releasing it to claim any card-1 case."""
    script, log = directory / (identifier + ".sh"), directory / (identifier + ".log")
    release = directory / (identifier + ".released")
    command = worker_command(source_root, run_dir, "h100", python, SERVAL02_CARD1,
                              standing_lock="/tmp/sl5nw_serval02_card1.lock")
    lines = ["#!/bin/bash", "set -euo pipefail",
             f"for attempt in {{1..180}}; do test -r {shlex.quote(str(release))} && break; sleep 1; done",
             f"test -r {shlex.quote(str(release))}", *environment_lines(source_root),
             f"export CUDA_VISIBLE_DEVICES={SERVAL02_CARD1}", "exec " + shlex.join(command)]
    atomic_text(script, "\n".join(lines) + "\n")
    remote = ("nohup bash " + shlex.quote(str(script)) + " >" + shlex.quote(str(log))
              + " 2>&1 </dev/null & worker_pid=$!; "
              + "/usr/bin/python3 - \"$worker_pid\" <<'PY'\n"
              + "import json,pathlib,socket,sys\n"
              + "pid=int(sys.argv[1]); text=pathlib.Path(f'/proc/{pid}/stat').read_text()\n"
              + "print(json.dumps({'pid':pid,'start_time':text.rsplit(')',1)[1].split()[19],"
              + "'hostname':socket.gethostname()}),flush=True)\nPY")
    result = subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
                             "serval02", remote], capture_output=True, text=True, check=True)
    identity = json.loads(result.stdout.strip().splitlines()[-1])
    if type(identity.get("pid")) is not int or identity["pid"] < 1 or not identity.get("start_time"):
        raise ValueError("the remote launcher did not return a valid PID and birth counter")
    record = {**identity, "host": "serval02", "unit": identifier, "kind": "queue_worker",
              "context": "h100", "gpu_uuid": SERVAL02_CARD1, "source_root": str(source_root),
              "source_commit": commit, "script": str(script), "log": str(log),
              "time": datetime.now(timezone.utc).isoformat()}
    append_process(run_dir, record)
    atomic_text(directory / (identifier + ".pid"), str(identity["pid"]) + "\n")
    # This durable release follows the durable PID record; there is no unrecorded GPU child.
    atomic_text(release, json.dumps(record) + "\n")
    return record


def append_submission(run_dir, identifier, commit, script):
    """Record the exact sbatch ID before any later query or worker submission."""
    slurm = Path(run_dir) / "slurm"
    with (slurm / "submissions.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        with (slurm / "submitted_jobids_uva.txt").open("a") as ledger:
            ledger.write(identifier + "\n")
            ledger.flush()
            os.fsync(ledger.fileno())
        with (slurm / "submissions.tsv").open("a") as ledger:
            ledger.write("\t".join((identifier, "queue-worker", commit, str(script))) + "\n")
            ledger.flush()
            os.fsync(ledger.fileno())


def launch_slurm(source_root, run_dir, commit, python, directory, identifier, metadata_wrapper):
    """Submit one eight-CPU A4500 worker whose allocated physical UUID is resolved on its node."""
    script, log = directory / (identifier + ".sh"), directory / (identifier + "_%j.log")
    resolver = ["/usr/bin/python3", str(metadata_wrapper), "--context", "jaguar03_rtx_a4500",
                "--source-root", str(source_root), "--run-dir", str(run_dir), "--resolve-slurm-gpu"]
    command = worker_command(source_root, run_dir, "jaguar03_rtx_a4500", python, "GPU_UUID_ARGUMENT")
    # Quote every fixed argument, leaving only the verified UUID as a shell variable.
    command_text = " ".join('"$gpu_uuid"' if part == "GPU_UUID_ARGUMENT" else shlex.quote(part)
                             for part in command)
    lines = ["#!/bin/bash", "set -euo pipefail", *environment_lines(source_root),
             "gpu_uuid=$(" + shlex.join(resolver) + ")", 'export CUDA_VISIBLE_DEVICES="$gpu_uuid"',
             "exec " + command_text]
    atomic_text(script, "\n".join(lines) + "\n")
    resources = ["-p", "gpu", "--nodelist=jaguar03", "--gres=gpu:nvidia_rtx_a4500:1",
                 "--cpus-per-task=8", "--mem=64G", "--time=12:00:00", "--nodes=1", "--ntasks=1"]
    reservation = current_reservation()
    if reservation is not None:
        resources += ["--reservation=" + reservation, "--qos=csresnolim"]
    arguments = ["sbatch", "--parsable", "--job-name=atari_pixel_queue", "--output=" + str(log),
                 *resources, str(script)]
    result = subprocess.run(arguments, capture_output=True, text=True, check=True)
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdecimal():
        raise RuntimeError("sbatch returned an invalid exact job identifier: " + result.stdout)
    append_submission(run_dir, job_id, commit, script)
    return {"kind": "queue_worker", "context": "jaguar03_rtx_a4500", "job_id": job_id,
            "source_root": str(source_root), "source_commit": commit, "script": str(script),
            "log": str(log).replace("%j", job_id), "command": arguments}


def main():
    """Validate parent admission, then launch only the explicitly requested worker count."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", choices=CONTEXTS, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--python", type=Path, default=Path(PIXEL_PYTHON))
    parser.add_argument("--resolve-slurm-gpu", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    source_root, run_dir, commit = admitted_source(args.source_root, args.run_dir)
    if args.resolve_slurm_gpu:
        if args.context != "jaguar03_rtx_a4500":
            raise ValueError("Slurm GPU resolution is only used for the jaguar03 workers")
        print(allocated_gpu_uuid(), flush=True)
        return
    if args.workers < 1 or (args.context == "h100" and args.workers != 1):
        raise ValueError("request a positive worker count; serval02 card 1 has exactly one supervisor")
    if not args.python.is_absolute() or not os.access(args.python, os.X_OK):
        raise ValueError("--python must be the absolute executable path to the shared pixel environment")
    directory = run_dir / "slurm" / "queue_workers"
    directory.mkdir(parents=True, exist_ok=True)

    # Keep this parent-only metadata utility immutable even if the integration branch changes.
    contents = Path(__file__).read_bytes()
    digest = hashlib.sha256(contents).hexdigest()
    metadata_wrapper = directory / ("launch_queue_" + digest + ".py")
    if metadata_wrapper.exists():
        if metadata_wrapper.read_bytes() != contents:
            raise ValueError("an existing metadata wrapper has different contents")
    else:
        atomic_text(metadata_wrapper, contents.decode("utf-8"))
    for _ in range(args.workers):
        identifier = "queue_worker_" + args.context + "_source_" + commit[:12] + "_" + uuid.uuid4().hex[:12]
        record = (launch_direct(source_root, run_dir, commit, args.python, directory, identifier)
                  if args.context == "h100" else launch_slurm(source_root, run_dir, commit,
                      args.python, directory, identifier, metadata_wrapper))
        record["metadata_wrapper_sha256"] = digest
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
