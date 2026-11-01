"""Prepare one committed ACES metadata packet and mirror its exact original bytes; submit nothing."""

import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
from zoneinfo import ZoneInfo


DIRECTORY = Path(__file__).resolve().parent
RUN = DIRECTORY.parents[1]
REMOTE_RUN = "/scratch/user/u.sl359075/rnd_atari_pixels/runs/" + RUN.name
HELPER_ROOT = "/scratch/user/u.sl359075/rnd_atari_pixels/git/checkouts/study_helpers_11ceab5c640b"
COMMIT = "11ceab5c640b484d18ca40e05785bbf55c6de86f"
PYTHON = "/scratch/user/u.sl359075/envs/rnd_atari_pixels_jax/bin/python"
HELPER = HELPER_ROOT + "/10_jax_exploration_platform/scripts/atari_pixel_campaign/aces_allocation_observation.py"
SSH = ["ssh", "-T", "-o", "ControlMaster=no", "-o", "ProxyCommand=false", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "aces"]
COMMANDS = []


def write_once(path, contents):
    """Retain identical bytes and reject any conflicting existing mirror artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != contents:
            raise ValueError("existing mirror differs; preserve it: " + str(path))
        return
    with path.open("xb") as stream:
        stream.write(contents)
        stream.flush()
        os.fsync(stream.fileno())


def save(name, value):
    """Publish complete local transport metadata beside the source-staging receipts."""
    write_once(DIRECTORY / name, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def remote(program):
    """Run standard-library metadata only through the already checked existing connection."""
    command = SSH + [shlex.join([PYTHON, "-c", program])]
    COMMANDS.append(command)
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError(json.dumps({"command": command, "returncode": result.returncode,
                                       "stdout": result.stdout, "stderr": result.stderr}))
    return json.loads(result.stdout)


def main():
    """Inspect old resource files, prepare once, then preserve exact remote packet and initial data bytes."""
    unit = (datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d-%H-%M")
        + "_ACES_allocation-observation_gpu-count-1_processor-count-8_host-memory-gib-32_time-limit-minutes-5_partition-gpu_account-158590754315_helper-commit-" + COMMIT)
    remote_packet, remote_data = REMOTE_RUN + "/remote_setup/" + unit, REMOTE_RUN + "/data/" + unit
    local_packet, local_data = RUN / "remote_setup" / unit, RUN / "data" / unit
    save("observation_paths.json", {"unit": unit, "local_campaign_run_dir": str(RUN), "remote_campaign_run_dir": REMOTE_RUN,
        "local_packet_directory": str(local_packet), "remote_packet_directory": remote_packet,
        "local_data_directory": str(local_data), "remote_data_directory": remote_data,
        "helper_root": HELPER_ROOT, "helper_commit": COMMIT, "jobs_submitted": 0})

    # Only absent old basis files may be staged; existing remote files are inspected and never overwritten.
    basis = "aces_reference_copies_1_source_d28e6735e797"
    inputs = {"slurm/submitted_jobids_aces.txt": RUN / "slurm/submitted_jobids_aces.txt",
        "slurm/submissions_aces.tsv": RUN / "slurm/submissions_aces.tsv",
        "slurm/aces_reference_d28e673.sbatch": RUN / "remote_setup/aces_reference_probe/aces_reference_d28e673.sbatch",
        "data/" + basis + "/hardware.json": RUN / "data" / basis / "hardware.json",
        "data/" + basis + "/progress.json": RUN / "data" / basis / "progress.json"}
    inspection = remote("""from pathlib import Path
import hashlib,json
run=Path(RUN)
records=[]
for name in NAMES:
    path=run/name
    if path.is_symlink():raise RuntimeError("resource basis path is a symlink: "+str(path))
    records.append({"relative_path":name,"exists":path.is_file(),"sha256":hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None})
print(json.dumps({"files":records,"files_written":0}))
""".replace("RUN", repr(REMOTE_RUN)).replace("NAMES", repr(list(inputs))))
    save("resource_basis_before.json", inspection)
    missing = [row["relative_path"] for row in inspection["files"] if not row["exists"]]
    if missing:
        payload = [{"relative_path": name, "sha256": hashlib.sha256(inputs[name].read_bytes()).hexdigest(),
                    "base64": base64.b64encode(inputs[name].read_bytes()).decode()} for name in missing]
        staged = remote("""from pathlib import Path
import base64,hashlib,json,os
run=Path(RUN)
records=[]
for row in PAYLOAD:
    path=run/row["relative_path"]
    contents=base64.b64decode(row["base64"],validate=True)
    if hashlib.sha256(contents).hexdigest()!=row["sha256"]:raise RuntimeError("supplied resource bytes differ")
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as stream:stream.write(contents);stream.flush();os.fsync(stream.fileno())
    records.append({"path":str(path),"sha256":row["sha256"]})
print(json.dumps({"staged_only_missing_files":records,"existing_files_overwritten":False}))
""".replace("RUN", repr(REMOTE_RUN)).replace("PAYLOAD", repr(payload)))
        save("resource_basis_missing_files_staged.json", staged)
    checked = remote("""from pathlib import Path
import importlib.util,json
spec=importlib.util.spec_from_file_location("actual_committed_aces_observer",HELPER)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
source=module.helper_source(ROOT)
if source["source_commit"]!=COMMIT:raise RuntimeError("helper commit differs before preparation")
basis,snapshot=module.resource_basis(Path(RUN),"2271806")
print(json.dumps({"helpers":source,"resource_basis":basis,"existing_aces_ledger_bytes":len(snapshot),"numerical_imports":False}))
""".replace("HELPER", repr(HELPER)).replace("ROOT", repr(HELPER_ROOT)).replace("COMMIT", repr(COMMIT)).replace("RUN", repr(REMOTE_RUN)))
    save("resource_basis_verified.json", checked)

    command = [PYTHON, HELPER, "prepare", "--directory", remote_packet, "--campaign", REMOTE_RUN,
        "--helper-root", HELPER_ROOT, "--python", PYTHON, "--basis-job-id", "2271806", "--partition", "gpu"]
    prepared = remote("""import json,subprocess
command=COMMAND
p=subprocess.run(command,capture_output=True,text=True)
if p.returncode:raise RuntimeError(json.dumps({"command":command,"returncode":p.returncode,"stdout":p.stdout,"stderr":p.stderr}))
print(json.dumps({"command":command,"returncode":p.returncode,"stdout":p.stdout,"stderr":p.stderr,"prepared":json.loads(p.stdout)}))
""".replace("COMMAND", repr(command)))
    save("prepare_command_receipt.json", prepared)
    if prepared["prepared"].get("submitted") is not False:
        raise RuntimeError("prepare unexpectedly reported a submission")

    # Copy explicit prepared files only; original remote paths and all JSON bytes remain unchanged.
    packet_names = ["allocation_observation_plan.json", "config_resolved.yaml", "command.txt", "job.sbatch",
        "experiment_background.md", "manifest.yaml", "resource_basis/aces_submitted_jobids.txt"]
    data_names = ["experiment_background.md", "manifest.yaml", "config_resolved.yaml", "command.txt", "progress.json"]
    document = remote("""from pathlib import Path
import base64,hashlib,json
records=[]
for role,root,names in GROUPS:
    for name in names:
        path=Path(root)/name
        if path.is_symlink():raise RuntimeError("prepared artifact is a symlink")
        contents=path.read_bytes()
        records.append({"role":role,"relative_path":name,"remote_path":str(path),"sha256":hashlib.sha256(contents).hexdigest(),"bytes":len(contents),"base64":base64.b64encode(contents).decode()})
print(json.dumps({"files":records}))
""".replace("GROUPS", repr([("packet", remote_packet, packet_names), ("data", remote_data, data_names)])))
    mirrored = []
    for row in document["files"]:
        contents = base64.b64decode(row["base64"], validate=True)
        if hashlib.sha256(contents).hexdigest() != row["sha256"] or len(contents) != row["bytes"]:
            raise ValueError("prepared remote artifact changed during mirror transport")
        path = (local_packet if row["role"] == "packet" else local_data) / row["relative_path"]
        write_once(path, contents)
        mirrored.append({key: value for key, value in {**row, "local_path": str(path)}.items() if key != "base64"})
    save("observation_mirror_receipt.json", {"schema_version": 1, "kind": "aces_allocation_observation_exact_mirror",
        "cluster": "aces", "owner": "u.sl359075", "helper_commit": COMMIT, "files": mirrored,
        "local_packet_directory": str(local_packet), "remote_packet_directory": remote_packet,
        "local_data_directory": str(local_data), "remote_data_directory": remote_data,
        "jobs_submitted": 0, "original_remote_paths_rewritten": False,
        "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    save("preparation_transport_commands.json", {"commands": COMMANDS, "existing_master_only": True,
        "authentication_attempted": False, "numerical_imports": False, "jobs_submitted": 0})
    print(json.dumps({"unit": unit, "local_packet": str(local_packet), "remote_packet": remote_packet,
        "mirrored_files": len(mirrored), "held_submission_command": prepared["prepared"]["submission_command"],
        "submitted": False}), flush=True)


if __name__ == "__main__":
    main()
