"""Prepare and verify a separate ACES full-site check without admitting workers.

Original numerical source admission and the UVA helper defaults are unchanged.
Fresh root-owned cached configuration must establish the allocated physical
device before an identity child initializes JAX with Slurm's original selector.
"""

import argparse
import csv
from datetime import datetime, timezone
from importlib import metadata
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import pwd
import re
import shlex
import socket
import stat
import subprocess
import sys
import time
import traceback


PREFIX = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"
SOURCE_COMMIT = "d28e6735e79788f618acb0ebfde8609db1f776af"
KINDS = ("identity", "learner", "renderer", "probe")
JOB_SHAPE = {"account": "158590754315", "partition": "gpu", "qos": "normal", "nodes": 1,
    "ntasks": 1, "gres": "gpu:h100:1", "cpus_per_task": 8, "memory": "32G", "time": "03:00:00"}
SUBMISSION_CONTROL = {"submit_held": True, "helper_submits_or_releases": False,
    "release_exact_saved_id_only": True, "ledger": "slurm/submitted_jobids_aces.txt",
    "parent_procedure": ["Submit the recorded held command.",
        "Write and fsync the exact returned ACES job ID, submission row and submission receipt.",
        "Verify persistent monitoring before releasing only that saved ID; save the successful release receipt."]}


def load(name):
    """Load one adjacent standard-library helper with no numerical imports."""
    path = Path(__file__).resolve().with_name(name + ".py")
    specification = importlib.util.spec_from_file_location("aces_full_site_" + name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def common():
    """Reuse only policy-independent functions from the captured-driver UVA helper."""
    return load("h100_nvl_slurm_site_validation")


def qualifier():
    """Keep the explicit ACES device policy separate from the unchanged UVA default."""
    return load("aces_slurm_qualification")


def ordinary_path(path, *, directory=False):
    """Reject symlinks in the complete lexical path before reading known ACES artifacts."""
    path = Path(path).absolute()
    if path != path.resolve(strict=True):
        raise ValueError("ACES artifact paths and all parent directories must be exact ordinary paths")
    mode = path.lstat().st_mode
    if not (stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)):
        raise ValueError("an ACES artifact has a different ordinary file or directory type")
    return path


def receipt(path):
    """Bind exact lexical ordinary-file bytes without resolving a substituted symlink."""
    return common().receipt(ordinary_path(path))


def read_exact(path):
    """Read one ordinary exact-path artifact after rejecting all symlink parents."""
    return common().read(ordinary_path(path))


def bound(record):
    """Read only the exact saved ordinary-file path and SHA, never a symlink substitute."""
    if not isinstance(record, dict) or set(record) != {"path", "sha256"} or receipt(record["path"]) != record:
        raise ValueError("an exact ACES artifact path or its recorded bytes changed")
    return read_exact(record["path"])


def require_boundary(value):
    """Reject numeric substitutes for permissions or scientific credit."""
    for name, expected in qualifier().BOUNDARY.items():
        if type(value.get(name)) is not type(expected) or value[name] != expected:
            raise ValueError("an ACES site check cannot grant credit or admission: " + name)


def helpers():
    """Pin the complete existing helper closure and both new ACES entry points."""
    old = common()
    return tuple(dict.fromkeys((*old.HELPERS, *old.QUALIFICATION_HELPERS,
        "aces_allocation_observation.py", "aces_slurm_qualification.py", "aces_slurm_site_validation.py",
        "aces_source_admission.py", "portable_evidence.py", "h100_nvl_slurm_evidence.py")))


def admission_inputs(run, binding=None):
    """Use explicit exact copied evidence only when ACES preparation requests it."""
    if binding is not None:
        return load("aces_source_admission").snapshot(binding)
    old = common()
    admission = old.snapshot_admission(run)
    numerical = old.source_identity(admission["source_root"], SOURCE_COMMIT, old.numerical_paths(admission["source_root"]))
    return admission, numerical, old.original_configuration(admission["source_root"], admission)


def check_admission(plan):
    """Keep ordinary admission checks unchanged and separately recheck explicit portable inputs."""
    if "portable_admission" in plan:
        return load("aces_source_admission").check(plan)
    return common().check_admission(plan["campaign_run_dir"], plan["admission"])


def submission_command(run, directory):
    """Record a held ACES command without reserving, submitting or releasing anything."""
    return ["sbatch", "--parsable", "--hold", *["--" + name + "=" + str(JOB_SHAPE[key]) for name, key in
        (("account", "account"), ("partition", "partition"), ("qos", "qos"), ("nodes", "nodes"), ("ntasks", "ntasks"),
         ("gres", "gres"), ("cpus-per-task", "cpus_per_task"), ("mem", "memory"), ("time", "time"))],
        "--job-name=ACES_H100_full_site_qualification", "--output=" + str(Path(run) / "slurm/jobs/aces_full_site_qualification_%j.log"),
        str(Path(directory) / "job.sbatch")]


def software_observations(plan):
    """Hash actual target packages, pinned JAXtari code and assets without importing numerical libraries."""
    old, training = common(), load("train_optimized")
    assets = training.verify_assets(plan["asset_record"], dict(os.environ))
    installed = {"packages": old.package_versions(),
        "jaxtari_source": training.jaxtari_python_receipt(plan["jaxtari_source_root"]),
        "gpu_libraries": training.installed_gpu_library_receipts()}
    installed["packages"].update({name: metadata.version(name) for name in ("pytest", "torch")})
    return {"installed_backend": installed, "asset_receipt": assets}


def checked_software(plan):
    """Require actual native package, source, library and asset observations to match preparation."""
    software = software_observations(plan)
    if software != bound(plan["software_at_prepare"]):
        raise ValueError("the actual target packages, JAXtari source, GPU images or assets changed")
    return software


def prepare(arguments):
    """Write a new unsubmitted ACES plan and standard records using the selected actual interpreter."""
    old, policy = common(), qualifier()
    run, helper = arguments.run_dir.resolve(strict=True), arguments.helper_root.resolve(strict=True)
    directory = arguments.output_dir.absolute()
    if (directory.parent != run / "remote_setup" or directory.exists() or directory.is_symlink()
            or pwd.getpwuid(os.getuid()).pw_name != policy.PROFILE["username"] or os.getuid() != policy.PROFILE["uid"]):
        raise ValueError("prepare a fresh ACES site packet under this campaign as its actual ACES owner")
    if str(Path(sys.executable).absolute()) != str(arguments.python.absolute()):
        raise ValueError("prepare with the actual selected ACES interpreter; do not infer another environment")
    commit = old.query(["git", "-C", str(helper), "rev-parse", "HEAD"]).strip()
    identity = old.source_identity(helper, commit, [PREFIX + name for name in helpers()])
    if Path(__file__).resolve() != helper / PREFIX / "aces_slurm_site_validation.py":
        raise ValueError("prepare through the exact detached committed ACES entry point")
    # The optional mapping relocates original bytes; native software is still observed below.
    portable_arguments = [getattr(arguments, name, None) for name in
        ("admission_run_dir", "evidence_manifest", "evidence_manifest_sha256")]
    if any(value is not None for value in portable_arguments) and not all(value is not None for value in portable_arguments):
        raise ValueError("portable ACES preparation needs the original run, exact manifest and SHA256 together")
    binding = None if portable_arguments[0] is None else {"schema_version": 1,
        "kind": "aces_original_source_portable_admission", "campaign_run_dir": str(portable_arguments[0]),
        "evidence_manifest": {"path": str(portable_arguments[1]), "sha256": portable_arguments[2]}}
    if binding is not None and Path(binding["campaign_run_dir"]).name != run.name:
        raise ValueError("the portable original and native ACES campaign names differ")
    admission, numerical, configuration = admission_inputs(run, binding)
    reference = old.cleanrl_identity(arguments.cleanrl_root)
    old.require_gate_controls(os.environ, reference)
    unit = run / "data" / directory.name
    if unit.exists() or unit.is_symlink():
        raise ValueError("the ACES qualification data shard must also be new")
    plan = {"schema_version": 1, "kind": "aces_full_site_qualification", "cluster": "aces",
        "campaign_run_dir": str(run), "data_unit": {"name": directory.name, "path": str(unit)},
        "source_commit": SOURCE_COMMIT, "helpers": identity, "numerical_source": numerical, "admission": admission,
        "cleanrl_reference": reference, "configuration": configuration,
        "runtime_environment": old.RUNTIME, "gate_environment_policy": old.GATE_ENVIRONMENT_POLICY,
        "qualification_profile": policy.PROFILE, "python": str(arguments.python.absolute()),
        "asset_record": read_exact(arguments.asset_record), "jaxtari_source_root": str(arguments.jaxtari_source_root.resolve(strict=True)),
        "owner": {"username": policy.PROFILE["username"], "uid": policy.PROFILE["uid"]},
        "job_shape": JOB_SHAPE, "submission_control": SUBMISSION_CONTROL,
        "reference_updates": 1, "child_order": list(KINDS), **policy.BOUNDARY,
        "created_at_utc": datetime.now(timezone.utc).isoformat()}
    if binding is not None:
        plan["portable_admission"] = binding
    software = software_observations(plan)
    directory.mkdir()
    unit.mkdir()
    plan["admission_archive"] = (old.archive_admission(directory, admission) if binding is None else
        load("aces_source_admission").archive(directory, admission, binding))
    old.write_once(directory / "software_at_prepare.json", software)
    plan["software_at_prepare"] = receipt(directory / "software_at_prepare.json")
    old.write_once(directory / "site_validation_plan.json", plan)
    executable = helper / PREFIX / "aces_slurm_site_validation.py"
    command = [plan["python"], "-I", str(executable), "run", "--directory", str(directory)]
    batch = ["#!/bin/bash", "set -euo pipefail", "export TZ=UTC"]
    batch.extend("export " + key + "=" + shlex.quote(value) for key, value in old.RUNTIME.items())
    batch.append("export CLEANRL_REFERENCE_FILE=" + shlex.quote(reference["file"]["path"]))
    batch.append("exec " + shlex.join(["srun", "--unbuffered", "--ntasks=1", "--cpus-per-task=8", "--cpu-bind=threads", *command]))
    old.write_once(directory / "job.sbatch", "\n".join(batch) + "\n")
    old.write_once(directory / "command.txt", shlex.join(submission_command(run, directory)) + "\n")
    old.write_once(directory / "config_resolved.yaml", plan)
    old.write_once(directory / "experiment_background.md", "## Purpose\n\nVerify the original pixel Montezuma learner on one owned ACES H100 PCIe allocation. This operational site check grants no scientific, comparison or idea credit and admits no workers.\n\n## Key hyperparameters\n\nOne policy, 128 environments, 128 rollout steps, four epochs, four minibatches, fifty observation warmup rollouts and one complete update. Networks and optimizer state remain float32; running statistics remain float64. Highest arithmetic precision and deterministic operations remain required.\n\n## Code and config changes\n\nThe separate ACES profile requires driver 580.178.04 and recorded root-owned cached GRES and task/cgroup configuration. Ambiguity is saved and rejected before JAX. Slurm's CUDA selector is preserved until a fresh identity child proves the actual initialized CUDA UUID and PCI address. Full original learner and renderer checks then run in distinct fresh children; a final probe repeats identity, captures its actual mapped driver bytes, checks all dtypes and finite state, and performs one update. No full NVIDIA XML, process queries or unassigned device probes are permitted. Configured device restrictions are recorded; actual kernel enforcement is not independently attested.\n\n## Git state\n\nNumerical source: " + SOURCE_COMMIT + ". Helpers: " + commit + ". Preparation never submits or releases a job or changes existing source admission. A future selected optimized source requires its own separate qualification.\n")
    manifest = {"schema_version": 1, "kind": "aces_full_site_qualification_preparation", "cluster": "aces",
        "artifacts": {name: receipt(directory / name) for name in ("site_validation_plan.json", "job.sbatch",
            "command.txt", "config_resolved.yaml", "experiment_background.md", "software_at_prepare.json")},
        "submission_command": submission_command(run, directory), "submission_control": SUBMISSION_CONTROL, **policy.BOUNDARY}
    old.write_once(directory / "manifest.yaml", manifest)
    for name in ("experiment_background.md", "manifest.yaml", "config_resolved.yaml", "command.txt"):
        old.write_once(unit / name, (directory / name).read_bytes().decode())
    return {"directory": str(directory), "data_unit": plan["data_unit"], "submitted": False, **policy.BOUNDARY}


def prepared(directory):
    """Verify immutable preparation and source bytes without reading historical node-local driver files."""
    old, policy = common(), qualifier()
    directory = ordinary_path(directory, directory=True)
    manifest = read_exact(directory / "manifest.yaml")
    if (type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1
            or manifest.get("kind") != "aces_full_site_qualification_preparation" or manifest.get("cluster") != "aces"):
        raise ValueError("the ACES preparation manifest has a different schema, kind or cluster")
    plan = bound(manifest["artifacts"]["site_validation_plan.json"])
    require_boundary(manifest)
    require_boundary(plan)
    policy.require_profile(plan["qualification_profile"])
    if (type(plan.get("schema_version")) is not int or plan["schema_version"] != 1
            or plan.get("kind") != "aces_full_site_qualification" or plan.get("cluster") != "aces"
            or plan.get("source_commit") != SOURCE_COMMIT or plan.get("runtime_environment") != old.RUNTIME
            or plan.get("gate_environment_policy") != old.GATE_ENVIRONMENT_POLICY or plan.get("job_shape") != JOB_SHAPE
            or plan.get("submission_control") != SUBMISSION_CONTROL or plan.get("child_order") != list(KINDS)
            or plan.get("owner") != {"username": policy.PROFILE["username"], "uid": policy.PROFILE["uid"]}
            or type(plan.get("reference_updates")) is not int or plan["reference_updates"] != 1
            or manifest.get("submission_command") != submission_command(plan["campaign_run_dir"], directory)):
        raise ValueError("the ACES preparation changed its fixed source, configuration or scope")
    expected = {"site_validation_plan.json", "job.sbatch", "command.txt", "config_resolved.yaml", "experiment_background.md", "software_at_prepare.json"}
    if set(manifest["artifacts"]) != expected:
        raise ValueError("the complete standard ACES preparation is required")
    for name, record in manifest["artifacts"].items():
        if record != receipt(directory / name):
            raise ValueError("a prepared ACES input changed its exact path or bytes")
    if plan.get("software_at_prepare") != manifest["artifacts"]["software_at_prepare.json"]:
        raise ValueError("the target software baseline must be its exact immutable prepared artifact")
    if (Path(__file__).resolve() != Path(plan["helpers"]["source_root"]) / PREFIX / "aces_slurm_site_validation.py"
            or set(plan["helpers"]["files"]) != {PREFIX + name for name in helpers()}):
        raise ValueError("execute the complete committed ACES helper closure")
    old.verify_source_identity(plan["helpers"])
    old.verify_source_identity(plan["numerical_source"])
    for entry in plan["admission_archive"]["files"]:
        ordinary_path(entry["captured"]["path"])
    old.validate_admission_archive(directory, plan["admission"], plan["admission_archive"])
    if (plan["numerical_source"]["source_commit"] != SOURCE_COMMIT
            or sorted(plan["numerical_source"]["files"]) != old.numerical_paths(plan["numerical_source"]["source_root"])
            or old.cleanrl_identity(plan["cleanrl_reference"]["source_root"]) != plan["cleanrl_reference"]
            or ("portable_admission" not in plan and
                old.original_configuration(plan["numerical_source"]["source_root"], plan["admission"]) != plan["configuration"])):
        raise ValueError("the complete original numerical source, configuration or CleanRL reference changed")
    if "portable_admission" in plan:
        check_admission(plan)
    unit = Path(plan["data_unit"]["path"])
    ordinary_path(unit, directory=True)
    if (plan["data_unit"]["name"] != directory.name or unit != Path(plan["campaign_run_dir"]) / "data" / directory.name
            or unit.is_symlink() or unit.resolve(strict=True).parent != Path(plan["campaign_run_dir"]) / "data"):
        raise ValueError("the ACES site data must be its exact ordinary direct child")
    return directory, unit, plan


def owned_job(plan, directory, environment, *, wait_seconds=30):
    """Bind exact ACES owner, durable held submission and release before collecting allocation metadata."""
    old, policy = common(), qualifier()
    run = Path(plan["campaign_run_dir"])
    job, step = environment.get("SLURM_JOB_ID", ""), environment.get("SLURM_STEP_ID", "")
    cgroup = Path("/proc/self/cgroup").read_text()
    ledger = run / "slurm/submitted_jobids_aces.txt"
    if (not re.fullmatch(r"[1-9][0-9]*", job) or not re.fullmatch(r"[0-9]+", step)
            or ledger.read_text().splitlines().count(job) != 1
            or pwd.getpwuid(os.getuid()).pw_name != policy.PROFILE["username"] or os.getuid() != policy.PROFILE["uid"]
            or not all(re.search(r"(?:^|/)" + re.escape(part) + r"(?:/|$)", cgroup, re.M)
                for part in ("uid_" + str(os.getuid()), "job_" + job, "step_" + step))):
        raise ValueError("the full site check requires its exact owned ACES job/step/user/cgroup")
    submission = read_exact(directory / "submission_receipt.json")
    row = [job, directory.name, plan["helpers"]["source_commit"], str(run)]
    if (type(submission.get("schema_version")) is not int or submission["schema_version"] != 1
            or submission.get("kind") != "owned_aces_full_site_qualification_submission"
            or submission.get("cluster") != "aces" or submission.get("job_id") != job
            or submission.get("held_command") != submission_command(run, directory)
            or submission.get("submission_row") != row or submission.get("ledger_path") != str(ledger)
            or submission.get("plan_sha256") != old.sha256(directory / "site_validation_plan.json")
            or submission.get("manifest_sha256") != old.sha256(directory / "manifest.yaml")
            or submission.get("job_script_sha256") != old.sha256(directory / "job.sbatch")
            or load("aces_allocation_observation").submission_rows(run / "slurm/submissions_aces.tsv", job) != "\t".join(row)):
        raise ValueError("the ACES site submission must bind its exact durable ledger row and held preparation")
    deadline = time.monotonic() + wait_seconds
    while not (directory / "release_receipt.json").exists() and time.monotonic() < deadline:
        time.sleep(.2)
    release = read_exact(directory / "release_receipt.json")
    if (type(release.get("schema_version")) is not int or release["schema_version"] != 1
            or release.get("kind") != "owned_aces_full_site_qualification_release" or release.get("cluster") != "aces"
            or release.get("job_id") != job or release.get("command") != ["scontrol", "release", job]
            or type(release.get("returncode")) is not int or release["returncode"] != 0
            or release.get("submission_receipt_sha256") != old.sha256(directory / "submission_receipt.json")):
        raise ValueError("the ACES site release must bind this exact saved submission")
    return {"ledger": receipt(ledger), "submission": receipt(directory / "submission_receipt.json"),
        "release": receipt(directory / "release_receipt.json"), "job_id": job, "step_id": step, "cgroup": cgroup}


def allocation_observation(plan, directory):
    """Save original scheduler, hardware and root-owned configuration before any numerical import."""
    old, policy, observer = common(), qualifier(), load("aces_allocation_observation")
    ownership = owned_job(plan, directory, os.environ)
    host = socket.gethostname().split(".")[0]
    if not re.fullmatch(r"ac[0-9]+", host):
        raise ValueError("ACES qualification cannot execute on a login node")
    queries = {kind: observer.run_query(["scontrol", "show", kind, "-d", identity]) for kind, identity in
        (("job", ownership["job_id"]), ("step", ownership["job_id"] + "." + ownership["step_id"]), ("node", host))}
    generic = observer.generic_collector()
    environment = {key: os.environ.get(key) for key in observer.allocation_fields(generic)}
    isolation = observer.configured_observations(generic, os.environ, ownership["cgroup"])
    owned = {**ownership, "hostname": host, "uid": os.getuid(), "pid": os.getpid(),
        "process_start_time": old.process_start_time(os.getpid()), "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "scheduler": queries, "allocation_environment": environment}
    gpu = observer.run_query(["nvidia-smi", "--query-gpu=" + ",".join(observer.GPU_FIELDS), "--format=csv,noheader,nounits"])
    gpu["fields"] = list(observer.GPU_FIELDS)
    return {"schema_version": 1, "kind": "aces_full_site_allocation_observation", "qualification_profile": policy.PROFILE,
        "owned_allocation": owned, "hardware_queries": {"gpu_csv": gpu},
        "host_gpu_inventory": generic.host_inventory(), "device_isolation_observations": isolation,
        "cached_configuration": policy.configuration_observations(environment, isolation["running_slurm_configuration"]),
        "upstream_devices_list_caveat": {"sources": list(policy.UPSTREAM_DEVICE_SOURCES),
            "scope": "upstream behavior only; no actual patched-kernel attestation"}, **policy.BOUNDARY}


def child_environment(plan, assignment, kind):
    """Preserve Slurm's CUDA selector until the identity child proves its physical UUID and PCI."""
    old = common()
    environment = {**os.environ, **old.RUNTIME, "XDG_DATA_HOME": plan["asset_record"]["root"],
        "CLEANRL_REFERENCE_FILE": plan["cleanrl_reference"]["file"]["path"],
        "PYTHONPATH": str(Path(plan["numerical_source"]["source_root"]) / "10_jax_exploration_platform/src")}
    if kind != "identity":
        environment["CUDA_VISIBLE_DEVICES"] = assignment["gpu_uuid"]
    return environment


def run_child(plan, directory, unit, kind, assignment):
    """Record a fresh child and birth counter before releasing it to execute a single required stage."""
    old = common()
    command = [plan["python"], "-I", str(Path(plan["helpers"]["source_root"]) / PREFIX / "aces_slurm_site_validation.py"),
        "child", "--directory", str(directory), "--kind", kind]
    process, birth = None, None
    with (unit / (kind + ".log")).open("x") as log:
        try:
            process = subprocess.Popen(command, cwd=plan["numerical_source"]["source_root"],
                env=child_environment(plan, assignment, kind), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            birth = old.process_start_time(process.pid)
            if birth is None:
                raise RuntimeError("the ACES child exited before its birth was recorded")
            owner = {"pid": process.pid, "process_start_time": birth, "parent_pid": os.getpid(),
                "parent_start_time": old.process_start_time(os.getpid()), "job_id": assignment["job_id"],
                "step_id": assignment["step_id"], "kind": kind}
            old.write_once(unit / (kind + "_owner.json"), owner)
            old.write_once(unit / (kind + ".released"), "released\n")
            code = process.wait()
            old.write_once(unit / (kind + "_terminal.json"), {**owner, "exit_code": code,
                "command": command, "log": receipt(unit / (kind + ".log"))})
            if code != 0:
                raise RuntimeError("the ACES " + kind + " stage failed with exit code " + str(code))
        except BaseException:
            if process is not None:
                old.stop_child(process, birth)
            raise


def initialize_identity(plan, unit, process):
    """Initialize only the assigned JAX device and capture its already mapped driver bytes."""
    old, policy = common(), qualifier()
    if any(name == prefix or name.startswith(prefix + ".") for name in sys.modules
            for prefix in ("jax", "numpy", "jaxtari", "exploration_platform")):
        raise ValueError("the identity stage needs a fresh process without numerical imports")
    sys.path.insert(0, str(Path(plan["numerical_source"]["source_root"]) / "10_jax_exploration_platform/src"))
    import jax
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    if jax.config.jax_enable_x64 is not True or jax.config.jax_default_matmul_precision != "highest":
        raise ValueError("the actual initialized JAX precision controls differ from the original configuration")
    devices = jax.devices()
    if len(devices) != 1:
        raise ValueError("the actual initialized JAX backend must expose one assigned whole GPU")
    proof = policy.validate_cuda_identity(policy.cuda_device_identity(devices[0]), process["allocation"], process["pid"])
    capture_dir = unit / (process["kind"] + "_driver")
    capture_dir.mkdir()
    capture = old.capture_driver_image(proof, capture_dir)
    return jax, devices[0], {"identity": proof, "driver_capture": capture}


def validate_loaded(loaded, installed, identity):
    """Verify actual mapped CUDA/cuDNN package images and the separately captured mapped driver."""
    old = common()
    if (type(loaded.get("pid")) is not int or loaded["pid"] != identity["pid"]
            or loaded.get("platform") != "gpu" or not loaded.get("platform_version")
            or any(type(loaded.get(name)) is not int or loaded[name] <= 0 for name in ("cudaRuntimeGetVersion", "cudnnGetVersion"))):
        raise ValueError("the ACES reference needs actual GPU/CUDA/cuDNN API observations")
    mapped = {path for path in loaded.get("loaded_shared_libraries", []) if Path(path).name.startswith(("libcudnn", "libcudart"))}
    verified = loaded.get("verified_package_libraries", {})
    if (set(verified) != mapped or not any(Path(path).name.startswith("libcudnn.so") for path in mapped)
            or not any(Path(path).name.startswith("libcudart.so") for path in mapped)
            or any(verified[path] != installed["gpu_libraries"].get(path) for path in mapped)
            or any(loaded.get(name) not in verified for name in ("cudnn_library", "cuda_runtime_library"))
            or identity["driver_library"] not in loaded.get("loaded_shared_libraries", [])):
        raise ValueError("the actual mapped CUDA/cuDNN/driver images differ from recorded target bytes")
    for path in mapped:
        if old.sha256(path) != verified[path]["sha256"]:
            raise ValueError("an installed mapped GPU image changed its recorded bytes")


def reference_probe(plan, unit, process, jax, device, identity):
    """Run the complete original reference update and verify all network/statistics dtypes and finite state."""
    old, training = common(), load("train_optimized")
    from exploration_platform.atari_pixels.config import PixelPPOConfig
    from exploration_platform.atari_pixels.environment import AtariPixelEnvironment
    from exploration_platform.atari_pixels.training import PixelPPORND
    values = dict(plan["configuration"]["learner"])
    for name in ("per_copy_learning_rates", "per_copy_intrinsic_coefficients", "per_copy_scalar_settings"):
        values[name] = tuple(values[name])
    trainer = PixelPPORND(PixelPPOConfig(**values), AtariPixelEnvironment(**plan["configuration"]["environment"]))
    state = trainer.warmup(trainer.initialize((1,)))
    state, _ = trainer.train_iteration(state)
    jax.block_until_ready(state)
    import jax.numpy as jnp
    leaves = [leaf for leaf in jax.tree.leaves(state) if getattr(leaf, "dtype", None) is not None and leaf.dtype.kind in "fc"]
    finite = {"passed": bool(jax.device_get(jnp.all(jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in leaves])))) if leaves else False,
        "floating_leaves_checked": len(leaves), "scope": "complete updated PixelTrainState"}
    groups = {"parameters": state.parameters, "target": state.target, "optimizer_first": state.optimizer.first,
        "optimizer_second": state.optimizer.second, "observation_statistics": state.observation_statistics,
        "reward_statistics": state.reward_statistics}
    dtypes = {name: sorted({str(leaf.dtype) for leaf in jax.tree.leaves(tree)}) for name, tree in groups.items()}
    if dtypes != old.EXPECTED_DTYPES or int(state.iteration) != 1 or not finite["passed"]:
        raise ValueError("the ACES update did not preserve finite original float32 networks and float64 statistics")
    paths = {name: str(Path(module.__file__).resolve()) for name, module in list(sys.modules.items())
        if name == "exploration_platform" or name.startswith("exploration_platform.atari_pixels")}
    old.validate_numerical_paths(paths, plan["numerical_source"]["source_root"])
    loaded = training.actual_gpu_library_receipt(device.client, process["software"]["installed_backend"]["gpu_libraries"])
    distribution = metadata.distribution("jaxtari")
    if Path(sys.modules["jaxtari"].__file__).resolve() != Path(distribution.locate_file("jaxtari/__init__.py")).resolve():
        raise ValueError("the ACES probe imported a shadowed JAXtari package")
    validate_loaded(loaded, process["software"]["installed_backend"], identity["identity"])
    result = {**process, **identity, "kind": "original_aces_reference_site_probe", "completed": True,
        "iteration": int(state.iteration), "dtypes": dtypes, "finite_state": finite,
        "loaded_libraries": loaded, "numerical_module_paths": paths,
        "formal_measurement": False, "capacity_evidence": False, **qualifier().BOUNDARY}
    old.write_once(unit / "probe_result.json", result)


def child(directory, kind):
    """Recheck this exact released child and actual allocation before executing its required original stage."""
    old, policy = common(), qualifier()
    directory, unit, plan = prepared(directory)
    old.require_runtime(os.environ)
    old.require_gate_controls(os.environ, plan["cleanrl_reference"])
    parent = read_exact(unit / "parent_owner.json")
    started = time.monotonic()
    while not (unit / (kind + ".released")).exists():
        if time.monotonic() - started > 60 or old.process_start_time(parent["pid"]) != parent["process_start_time"]:
            raise RuntimeError("the ACES parent did not durably release this child")
        time.sleep(.1)
    owner = read_exact(unit / (kind + "_owner.json"))
    if (owner["pid"] != os.getpid() or owner["process_start_time"] != old.process_start_time(os.getpid())
            or owner["parent_pid"] != parent["pid"] or owner["parent_start_time"] != parent["process_start_time"]
            or owner["kind"] != kind or str(Path(sys.executable).absolute()) != plan["python"]):
        raise ValueError("the ACES stage has a different released process or interpreter")
    observation = allocation_observation(plan, directory)
    old.write_once(unit / (kind + "_allocation_observation.json"), observation)
    allocation = policy.validate_allocation(observation)
    parent_allocation = read_exact(unit / "site_assignment.json")
    if any(allocation[name] != parent_allocation[name] for name in ("job_id", "step_id", "hostname", "gpu_uuid", "pci_address", "cpu_affinity")):
        raise ValueError("the ACES child changed its owned physical allocation")
    selector = parent_allocation["cuda_selector_original"] if kind == "identity" else allocation["gpu_uuid"]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != selector:
        raise ValueError("the ACES child changed the required original or verified CUDA selector")
    software = checked_software(plan)
    process = {**owner, "source_commit": SOURCE_COMMIT, "configuration": plan["configuration"],
        "runtime_environment": old.require_runtime(os.environ), "cuda_selector": selector, "software": software,
        "allocation": allocation, "allocation_observation": receipt(unit / (kind + "_allocation_observation.json")),
        "cleanrl_reference": plan["cleanrl_reference"], "gate_environment_policy": old.GATE_ENVIRONMENT_POLICY,
        "original_command": old.original_command(plan, kind) if kind in ("learner", "renderer") else None}
    old.write_once(unit / (kind + "_process.json"), process)
    if kind in ("learner", "renderer"):
        command = old.original_command(plan, kind)
        os.execv(command[0], command)
    jax, device, identity = initialize_identity(plan, unit, process)
    if kind == "identity":
        old.write_once(unit / "identity_result.json", {**process, **identity, "completed": True, **policy.BOUNDARY})
    else:
        reference_probe(plan, unit, process, jax, device, identity)


def verify_identity_result(value, process, unit, kind):
    """Verify saved same-process CUDA identity and captured shared driver bytes portably."""
    policy, old = qualifier(), common()
    require_boundary(value)
    if value.get("completed") is not True or any(value.get(key) != expected for key, expected in process.items() if key != "kind"):
        raise ValueError("the saved identity belongs to another actual child")
    if value.get("kind") != ("original_aces_reference_site_probe" if kind == "probe" else process["kind"]):
        raise ValueError("the saved identity or reference probe changed its exact stage kind")
    identity = policy.validate_cuda_identity(value["identity"], process["allocation"], process["pid"])
    capture = value["driver_capture"]
    if type(capture.get("schema_version")) is not int or type(capture.get("pid")) is not int:
        raise ValueError("the saved driver capture must use exact integer schema and process identity")
    folder = ordinary_path(unit / (kind + "_driver"), directory=True)
    ordinary_path(folder / "captured_driver", directory=True)
    ordinary_path(folder / "captured_driver" / Path(identity["driver_library"]).name)
    old.validate_driver_capture(value["driver_capture"], identity, unit / (kind + "_driver"), process["pid"])
    return identity


def verify_result(directory, unit, plan, result):
    """Rebuild successful required stages without touching any historical node-local CUDA driver."""
    old, policy = common(), qualifier()
    ordinary_path(directory, directory=True)
    ordinary_path(unit, directory=True)
    require_boundary(result)
    policy.require_profile(result.get("qualification_profile"))
    if (type(result.get("schema_version")) is not int or result["schema_version"] != 1
            or result.get("kind") != "aces_full_site_evidence" or result.get("passed") is not True
            or result.get("plan") != receipt(directory / "site_validation_plan.json")
            or result.get("qualification_profile") != policy.PROFILE or set(result.get("children", {})) != set(KINDS)):
        raise ValueError("the ACES result changed its complete plan, profile or required stages")
    expected_records = {"parent_observation": "parent_allocation_observation.json", "assignment": "site_assignment.json",
        "parent_owner": "parent_owner.json", "identity": "identity_result.json", "probe": "probe_result.json"}
    if any(result[name] != receipt(unit / filename) for name, filename in expected_records.items()):
        raise ValueError("the ACES result substituted a parent, assignment or numerical result path")
    parent_observation = bound(result["parent_observation"])
    parent = bound(result["parent_owner"])
    parent_owned = parent_observation["owned_allocation"]
    if parent != {"pid": parent_owned["pid"], "process_start_time": parent_owned["process_start_time"]}:
        raise ValueError("the recorded parent does not identify the actual owned allocation observer")
    assignment = policy.validate_allocation(parent_observation)
    if assignment != bound(result["assignment"]):
        raise ValueError("the saved ACES physical assignment changed")
    identities = set()
    processes = {}
    for kind in KINDS:
        item = result["children"][kind]
        owner, process, terminal = (bound(item[name]) for name in ("owner", "process", "terminal"))
        if any(item[name] != receipt(unit / (kind + suffix)) for name, suffix in
                (("owner", "_owner.json"), ("process", "_process.json"), ("terminal", "_terminal.json"))):
            raise ValueError("a required ACES child substituted another artifact")
        identity = (owner.get("pid"), owner.get("process_start_time"))
        if (type(identity[0]) is not int or identity[0] <= 0 or not str(identity[1]).isdigit() or identity in identities
                or type(terminal.get("exit_code")) is not int or terminal["exit_code"] != 0
                or any(process.get(key) != value or terminal.get(key) != value for key, value in owner.items())
                or owner["kind"] != kind or owner["job_id"] != assignment["job_id"] or owner["step_id"] != assignment["step_id"]
                or owner.get("parent_pid") != parent["pid"] or owner.get("parent_start_time") != parent["process_start_time"]
                or process.get("source_commit") != SOURCE_COMMIT or process.get("configuration") != plan["configuration"]
                or process.get("runtime_environment") != old.RUNTIME
                or process.get("software") != bound(plan["software_at_prepare"])
                or terminal.get("log") != receipt(unit / (kind + ".log"))):
            raise ValueError("ACES stages need distinct successful children with exact source, runtime and software")
        expected_command = [plan["python"], "-I", str(Path(plan["helpers"]["source_root"]) / PREFIX / "aces_slurm_site_validation.py"),
            "child", "--directory", str(directory), "--kind", kind]
        observation = bound(process["allocation_observation"])
        if (process["allocation_observation"] != receipt(unit / (kind + "_allocation_observation.json"))
                or observation["owned_allocation"]["pid"] != owner["pid"]
                or observation["owned_allocation"]["process_start_time"] != owner["process_start_time"]
                or terminal.get("command") != expected_command):
            raise ValueError("the ACES child changed its exact command or actual observer process")
        expected_selector = assignment["cuda_selector_original"] if kind == "identity" else assignment["gpu_uuid"]
        if (process.get("cuda_selector") != expected_selector
                or observation["owned_allocation"]["allocation_environment"].get("CUDA_VISIBLE_DEVICES") != expected_selector):
            raise ValueError("the actual saved ACES stage changed its original or verified CUDA selector")
        fresh = policy.validate_allocation(observation)
        if fresh != process["allocation"] or any(fresh[key] != assignment[key] for key in ("job_id", "step_id", "hostname", "gpu_uuid", "pci_address", "cpu_affinity")):
            raise ValueError("an ACES child lacks its exact fresh allocation proof")
        expected = old.original_command(plan, kind) if kind in ("learner", "renderer") else None
        if (process.get("original_command") != expected or process.get("cleanrl_reference") != plan["cleanrl_reference"]
                or process.get("gate_environment_policy") != old.GATE_ENVIRONMENT_POLICY):
            raise ValueError("the original complete learner or renderer gate changed")
        identities.add(identity)
        processes[kind] = process
    first = bound(result["identity"])
    verify_identity_result(first, processes["identity"], unit, "identity")
    probe = bound(result["probe"])
    identity = verify_identity_result(probe, processes["probe"], unit, "probe")
    if any(first["identity"][key] != identity[key] for key in ("driver_library", "driver_library_sha256")):
        raise ValueError("the identity and reference probe mapped different actual CUDA driver images")
    if (probe.get("kind") != "original_aces_reference_site_probe" or type(probe.get("iteration")) is not int or probe["iteration"] != 1
            or probe.get("dtypes") != old.EXPECTED_DTYPES or probe.get("formal_measurement") is not False
            or probe.get("capacity_evidence") is not False or probe.get("finite_state", {}).get("passed") is not True
            or probe["finite_state"].get("scope") != "complete updated PixelTrainState"
            or type(probe["finite_state"].get("floating_leaves_checked")) is not int or probe["finite_state"]["floating_leaves_checked"] < 1):
        raise ValueError("the ACES probe lacks its complete finite float32/float64 original update")
    validate_loaded(probe["loaded_libraries"], processes["probe"]["software"]["installed_backend"], identity)
    old.validate_numerical_paths(probe["numerical_module_paths"], plan["numerical_source"]["source_root"])
    return {"passed": True, "scope": "one exact original-source ACES site qualification only",
        "hostname": assignment["hostname"], "job_id": assignment["job_id"], "gpu_uuid": assignment["gpu_uuid"],
        "kernel_device_enforcement_verified": False, **policy.BOUNDARY}


def run(directory):
    """Stop on ambiguous preflight evidence or execute the required stages without publishing admission."""
    old, policy = common(), qualifier()
    directory, unit, plan = prepared(directory)
    if any((unit / name).exists() for name in ("result.json", "failure.json", "parent_allocation_observation.json")):
        raise ValueError("a previous ACES site launch is immutable; prepare a new packet")
    try:
        check_admission(plan)
        old.require_runtime(os.environ)
        old.require_gate_controls(os.environ, plan["cleanrl_reference"])
        observation = allocation_observation(plan, directory)
        old.write_once(unit / "parent_allocation_observation.json", observation)
        assignment = policy.validate_allocation(observation)
        old.write_once(unit / "site_assignment.json", assignment)
        old.write_once(unit / "parent_owner.json", {"pid": os.getpid(), "process_start_time": old.process_start_time(os.getpid())})
        with (Path("/tmp") / ("sl359075_aces_site_" + assignment["gpu_uuid"] + ".lock")).open("a+") as lock, old.interrupted():
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            for kind in KINDS:
                check_admission(plan)
                if kind != "identity":
                    verify_identity_result(read_exact(unit / "identity_result.json"), read_exact(unit / "identity_process.json"), unit, "identity")
                run_child(plan, directory, unit, kind, assignment)
            check_admission(plan)
            result = {"schema_version": 1, "kind": "aces_full_site_evidence", "passed": True,
                "plan": receipt(directory / "site_validation_plan.json"), "qualification_profile": policy.PROFILE,
                "parent_observation": receipt(unit / "parent_allocation_observation.json"), "assignment": receipt(unit / "site_assignment.json"),
                "parent_owner": receipt(unit / "parent_owner.json"),
                "children": {kind: {name: receipt(unit / (kind + suffix)) for name, suffix in
                    (("owner", "_owner.json"), ("process", "_process.json"), ("terminal", "_terminal.json"))} for kind in KINDS},
                "identity": receipt(unit / "identity_result.json"), "probe": receipt(unit / "probe_result.json"),
                "completed_at_utc": datetime.now(timezone.utc).isoformat(), **policy.BOUNDARY}
            verify_result(directory, unit, plan, result)
            old.write_once(unit / "result.json", result)
            return result
    except BaseException as error:
        old.write_once(unit / "failure.json", {"kind": "aces_full_site_failure", "passed": False,
            "error": str(error), "exception_type": type(error).__name__, "traceback": traceback.format_exc(), **policy.BOUNDARY})
        raise


def main():
    """Prepare, execute or portably verify one exact ACES packet; never submit or release jobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    for flag in ("run-dir", "helper-root", "output-dir", "python", "cleanrl-root", "asset-record", "jaxtari-source-root"):
        preparation.add_argument("--" + flag, type=Path, required=True)
    preparation.add_argument("--admission-run-dir", type=Path)
    preparation.add_argument("--evidence-manifest", type=Path)
    preparation.add_argument("--evidence-manifest-sha256")
    for name in ("run", "verify", "child"):
        command = commands.add_parser(name)
        command.add_argument("--directory", type=Path, required=True)
        if name == "child":
            command.add_argument("--kind", choices=KINDS, required=True)
    arguments = parser.parse_args()
    if arguments.command == "prepare":
        value = prepare(arguments)
    elif arguments.command == "run":
        value = run(arguments.directory)
    elif arguments.command == "child":
        value = child(arguments.directory, arguments.kind)
    else:
        directory, unit, plan = prepared(arguments.directory)
        if (unit / "failure.json").exists():
            raise ValueError("the saved ACES site check has a preserved failure and cannot qualify")
        value = verify_result(directory, unit, plan, read_exact(unit / "result.json"))
    print(json.dumps(value, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
