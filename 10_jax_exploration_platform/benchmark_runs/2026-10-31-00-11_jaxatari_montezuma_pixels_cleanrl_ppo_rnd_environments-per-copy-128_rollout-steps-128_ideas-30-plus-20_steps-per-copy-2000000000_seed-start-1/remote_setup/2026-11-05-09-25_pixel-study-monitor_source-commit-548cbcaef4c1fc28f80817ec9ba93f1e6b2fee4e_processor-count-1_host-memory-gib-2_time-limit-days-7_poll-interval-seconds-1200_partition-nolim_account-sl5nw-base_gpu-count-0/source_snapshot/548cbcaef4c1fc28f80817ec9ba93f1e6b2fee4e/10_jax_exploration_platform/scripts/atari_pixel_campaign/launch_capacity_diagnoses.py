"""Record and submit three explicit fresh-process A4500 capacity diagnoses.

The numerical implementation remains in its original detached Git checkout.
Inputs come from two failed immutable cases and one completed full reference.
Preparation of the selected-measurement unit happens only on its allocated GPU
host after the physical UUID is proved. These are diagnoses, not study cases.
"""

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys


SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import launch_queue
import measure_selected as measurement

NUMERICAL_SOURCE = "d28e6735e79788f618acb0ebfde8609db1f776af"
JAXTARI_SOURCE = "04ab9aec318c72162e2e53503a594a489d7e162b"
DIAGNOSES = {"batch_rollout_intrinsic_rewards": {"hoist_rollout_bonus": True},
             "unroll_minibatch_update_loop": {"update_unroll": 2}}
PIXEL_PYTHON = "/p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python"


def immutable_json(path, value):
    """Publish complete immutable metadata without changing an earlier diagnosis artifact."""
    measurement.write_once(Path(path), measurement.json_bytes(value))


def derive_configurations(reference_path, case_paths):
    """Copy an actually completed full reference and apply only the two recorded failed-case changes."""
    record, reference_receipt = measurement.read_document(reference_path)
    if (record.get("case_completed") is not True or record.get("source_commit") != NUMERICAL_SOURCE
            or record.get("hardware_class") != "jaguar03_rtx_a4500" or record.get("copies") != 2):
        raise ValueError("the capacity reference must come from a completed two-copy D28 A4500 case")
    reference = measurement.validation.configuration(record["reference"])
    if reference["learner"]["n_copies"] != 2:
        raise ValueError("the recorded reference must retain two independent policies")
    variants = [{"name": "original_reference", "configuration": reference, "inputs": [reference_receipt]}]
    found = set()
    for path in case_paths:
        case, case_receipt = measurement.read_document(path)
        name = case.get("hypothesis")
        if (name not in DIAGNOSES or name in found or case.get("source_commit") != NUMERICAL_SOURCE
                or case.get("hardware_class") != "jaguar03_rtx_a4500" or case.get("seed_start") != 1
                or case.get("configuration") != reference["learner"]
                or case.get("candidate") != {"learner": DIAGNOSES[name], "environment": {}, "execution": {}}
                or case.get("reference_environment") is not None or case.get("reference_execution") is not None):
            raise ValueError("the failed-case identity or exact implementation change differs from the authorized diagnoses")
        candidate = copy.deepcopy(reference)
        candidate["learner"].update(case["candidate"]["learner"])
        variants.append({"name": name, "configuration": candidate, "inputs": [reference_receipt, case_receipt]})
        found.add(name)
    if found != set(DIAGNOSES):
        raise ValueError("supply the two original failed cases, without replacing or omitting either")
    return variants


def asset_documents(tree_path, assets_root, destination):
    """Convert the existing asset checksum list into the exact training-helper manifest schema."""
    tree_path, assets_root, destination = Path(tree_path).resolve(strict=True), Path(assets_root).resolve(strict=True), Path(destination)
    if destination.resolve().is_relative_to(assets_root):
        raise ValueError("asset metadata must be stored outside the immutable asset tree")
    files = {}
    for line in tree_path.read_text().splitlines():
        digest, relative = line.split("  ", 1)
        name = Path(relative)
        if (len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest)
                or name.is_absolute() or ".." in name.parts or relative in files):
            raise ValueError("the original asset checksum tree contains an invalid or duplicate path/hash")
        files[relative] = digest
    if not files:
        raise ValueError("the original asset checksum tree is empty")
    manifest = destination / "assets_manifest.json"
    immutable_json(manifest, {"schema_version": 1, "source_commit": JAXTARI_SOURCE, "files": files})
    record = {"root": str(assets_root), "manifest": str(manifest.resolve()), "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
    record_path = destination / "asset_record.json"
    immutable_json(record_path, record)
    return record_path


def frozen_helper(repository, destination):
    """Create or verify a real detached Git helper checkout at the pushed current commit."""
    repository, destination = Path(repository).resolve(strict=True), Path(destination).absolute()
    commit = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    remote = subprocess.check_output(["git", "-C", str(repository), "ls-remote", "origin", "refs/heads/merge-atari-cleanrl"], text=True).split()[0]
    if remote != commit:
        raise ValueError("push the exact helper commit before freezing or submitting its diagnoses")
    if not destination.exists():
        subprocess.run(["git", "-C", str(repository), "worktree", "add", "--detach", str(destination), commit], check=True)
    actual = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
    branch = subprocess.run(["git", "-C", str(destination), "symbolic-ref", "--quiet", "HEAD"], capture_output=True, text=True)
    if actual != commit or branch.returncode != 1:
        raise ValueError("the helper checkout is not detached at the exact pushed commit")
    return destination, commit


def record_submission(run_directory, identifier, helper_commit, script):
    """Flush the exact returned sbatch ID before any later filesystem work or query."""
    slurm = Path(run_directory) / "slurm"
    with (slurm / "submissions.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        for name in ("submitted_jobids_uva.txt", "capacity_diagnosis_jobids.txt"):
            with (slurm / name).open("a") as stream:
                stream.write(identifier + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        with (slurm / "submissions.tsv").open("a") as stream:
            stream.write("\t".join((identifier, "isolated-capacity-diagnosis", helper_commit, str(script))) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def submit(arguments):
    """Record all original input identities and submit exactly three one-GPU isolated processes."""
    run, core = arguments.run_dir.resolve(strict=True), arguments.source_root.resolve(strict=True)
    measurement.diagnostic.core_identity(core, NUMERICAL_SOURCE)
    variants = derive_configurations(arguments.reference_record, arguments.failed_case)
    helper, commit = frozen_helper(arguments.repo, arguments.helper_root)
    setup = run / "remote_setup" / ("isolated_capacity_diagnoses_helpers-" + commit[:12])
    setup.mkdir(parents=True, exist_ok=True)
    assets = asset_documents(arguments.assets_tree, arguments.assets_root, setup)
    runtime = setup / "runtime_environment.json"
    immutable_json(runtime, {"JAX_ENABLE_X64": "true", "JAX_DEFAULT_MATMUL_PRECISION": "highest",
                             "XLA_FLAGS": "--xla_gpu_exclude_nondeterministic_ops=true"})
    scripts = helper / "10_jax_exploration_platform/scripts/atari_pixel_campaign"
    worker = core / "10_jax_exploration_platform/scripts/atari_pixel_campaign/worker_environment.sh"
    reservation = launch_queue.current_reservation()
    units = []
    for variant in variants:
        name = variant["name"]
        unit = ("isolated_capacity_" + name + "_copies-2_environments-128_rollout-steps-128_epochs-4_minibatches-4"
                "_seed-start-1_updates-per-round-10_helpers-" + commit[:12])
        configuration = setup / (name + ".configuration.json")
        immutable_json(configuration, variant["configuration"])
        plan = {"kind": "isolated_capacity_diagnosis", "eligible_for_formal_campaign_admission": False,
            "hypothesis_count_contribution": 0, "unit_id": unit, "name": name,
            "source_root": str(core), "source_commit": NUMERICAL_SOURCE,
            "helper_root": str(helper), "helper_commit": commit,
            "launcher_sha256": measurement.training.file_sha256(scripts / "launch_capacity_diagnoses.py"),
            "worker_environment_sha256": measurement.training.file_sha256(worker),
            "configuration": str(configuration), "configuration_sha256": measurement.training.file_sha256(configuration),
            "runtime_environment": str(runtime), "runtime_environment_sha256": measurement.training.file_sha256(runtime),
            "asset_record": str(assets), "asset_record_sha256": measurement.training.file_sha256(assets),
            "jaxtari_source_root": str(arguments.jaxtari_source_root.resolve(strict=True)),
            "campaign_run_dir": str(run), "inputs": variant["inputs"], "seed_start": 1,
            "rounds": 7, "iterations": 10, "normalization_warmup_iterations": 50,
            "resources": {"gpu_count": 1, "gpu_type": "nvidia_rtx_a4500", "cpus": 8, "memory": "64G", "wall_time": "12:00:00"}}
        plan_path = setup / (name + ".diagnosis_plan.json")
        immutable_json(plan_path, plan)
        script = run / "slurm/jobs" / (unit + ".sh")
        command = ["/usr/bin/python3", str(scripts / "launch_capacity_diagnoses.py"), "--run-probe", str(plan_path),
                   "--submitted-script", str(script)]
        contents = "\n".join(["#!/bin/bash", "set -euo pipefail", "source " + shlex.quote(str(worker)),
            "exec " + shlex.join(command)]) + "\n"
        measurement.write_once(script, contents.encode())
        units.append((plan_path, script, unit))
    identifiers = []
    for plan_path, script, unit in units:
        submitted = setup / (unit + ".submitted.json")
        if submitted.exists():
            raise ValueError("an earlier submitted diagnosis must retain its original ID and outputs")
        command = ["sbatch", "--parsable", "-p", "gpu", "--nodelist=jaguar03", "--nodes=1", "--ntasks=1",
                   "--gres=gpu:nvidia_rtx_a4500:1", "--cpus-per-task=8", "--mem=64G", "--time=12:00:00",
                   "--job-name=atari_capacity_diagnosis", "--output=" + str(run / "slurm/jobs" / (unit + "_%j.log"))]
        if reservation is not None:
            command.extend(["--reservation=" + reservation, "--qos=csresnolim"])
        command.append(str(script))
        result = subprocess.run(command, capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL)
        identifier = result.stdout.strip().split(";", 1)[0]
        if not identifier.isdecimal():
            raise RuntimeError("sbatch did not return an exact numeric job ID: " + result.stdout)
        record_submission(run, identifier, commit, script)
        immutable_json(submitted, {"job_id": identifier, "command": command, "plan": str(plan_path), "helper_commit": commit})
        identifiers.append(identifier)
        print(json.dumps({"submitted_job_id": identifier, "unit_id": unit, "source_commit": NUMERICAL_SOURCE,
                          "helper_commit": commit}), flush=True)
    return identifiers


def run_probe(plan_path, submitted_script):
    """Resolve this allocated GPU, prepare its immutable unit here, and execute its recorded command."""
    plan, receipt = measurement.read_document(plan_path)
    helper = Path(plan["helper_root"])
    scripts = helper / "10_jax_exploration_platform/scripts/atari_pixel_campaign"
    if (Path(__file__).resolve() != scripts / "launch_capacity_diagnoses.py"
            or measurement.training.file_sha256(Path(__file__)) != plan["launcher_sha256"]):
        raise ValueError("the diagnosis launcher differs from its exact frozen helper source")
    measurement.training.source_receipt(helper, plan["helper_commit"], [str((scripts / name).relative_to(helper))
        for name in ("launch_capacity_diagnoses.py", "launch_queue.py", *measurement.HELPERS)])
    if socket.gethostname().split(".")[0] != "jaguar03" or not os.environ.get("SLURM_JOB_ID", "").isdecimal():
        raise ValueError("capacity preparation and execution require the actual jaguar03 Slurm allocation")
    if measurement.training.file_sha256(Path(plan["configuration"])) != plan["configuration_sha256"]:
        raise ValueError("the recorded full capacity configuration changed")
    for name in ("runtime_environment", "asset_record"):
        if measurement.training.file_sha256(Path(plan[name])) != plan[name + "_sha256"]:
            raise ValueError("a recorded capacity runtime/asset input changed: " + name)
    worker = Path(plan["source_root"]) / "10_jax_exploration_platform/scripts/atari_pixel_campaign/worker_environment.sh"
    if measurement.training.file_sha256(worker) != plan["worker_environment_sha256"]:
        raise ValueError("the original numerical worker environment changed")
    gpu = launch_queue.allocated_gpu_uuid(os.environ)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    run = Path(plan["campaign_run_dir"])
    job = run / "slurm/jobs" / (os.environ["SLURM_JOB_ID"] + "_jaguar03")
    job.mkdir(parents=True, exist_ok=True)
    measurement.write_once(job / "job.sbatch", Path(submitted_script).read_bytes())
    immutable_json(job / "assignment.json", {**plan, "diagnosis_plan_receipt": receipt, "gpu_uuid": gpu,
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        "slurm_step_gpus": os.environ.get("SLURM_STEP_GPUS")})
    hardware = measurement.preflight({"gpu_uuid": gpu, "hardware_class": "jaguar03_rtx_a4500"})
    immutable_json(job / "hardware.json", hardware)
    directory = run / "data" / plan["unit_id"]
    immutable_json(directory / "diagnosis_intent.json", {**plan, "diagnosis_plan_receipt": receipt, "gpu_uuid": gpu})
    prepare = [PIXEL_PYTHON, str(scripts / "measure_selected.py"), "--prepare-only", "--mode", "packing-point",
        "--source-root", plan["source_root"], "--source-commit", plan["source_commit"],
        "--configuration", plan["configuration"], "--runtime-environment", plan["runtime_environment"],
        "--asset-record", plan["asset_record"], "--jaxtari-source-root", plan["jaxtari_source_root"],
        "--run-dir", str(run), "--unit-id", plan["unit_id"], "--gpu-uuid", gpu,
        "--hardware-class", "jaguar03_rtx_a4500", "--seed-start", "1", "--minimum-updates-per-round", "10"]
    subprocess.run(prepare, check=True)
    command = shlex.split((directory / "command.txt").read_text())
    if command != [PIXEL_PYTHON, str(directory / "code/measure_selected.py"), "--prepared-unit", str(directory)]:
        raise ValueError("the prepared diagnosis command differs from its recorded staged producer")
    os.execv(command[0], command)


def parse_arguments(argv=None):
    """Expose only the three requested diagnoses and their private allocation-host entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-probe", type=Path)
    parser.add_argument("--submitted-script", type=Path)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--helper-root", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--reference-record", type=Path)
    parser.add_argument("--failed-case", type=Path, action="append", default=[])
    parser.add_argument("--assets-tree", type=Path)
    parser.add_argument("--assets-root", type=Path)
    parser.add_argument("--jaxtari-source-root", type=Path)
    args = parser.parse_args(argv)
    if args.run_probe is not None:
        if args.submitted_script is None:
            parser.error("the allocation-host entrypoint requires the exact submitted script")
        return args
    if len(args.failed_case) != 2 or any(getattr(args, name) is None for name in ("repo", "helper_root", "source_root",
            "run_dir", "reference_record", "assets_tree", "assets_root", "jaxtari_source_root")):
        parser.error("submission requires exact frozen sources, the completed reference, two failed cases, assets and run directory")
    return args


if __name__ == "__main__":
    arguments = parse_arguments()
    if arguments.run_probe is not None:
        run_probe(arguments.run_probe, arguments.submitted_script)
    else:
        submit(arguments)
