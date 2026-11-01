"""Run target-validated training through the existing complete-state training loop.

Admission and physical placement precede numerical imports. Logical source and
asset identities remain unchanged; operational paths are used only to locate
verified bytes. The first complete execution call supplies the lazy mapped-GPU
library proof. This entrypoint submits no jobs and implements no PPO updates.
"""

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys

import deployed_training_admission as admission
import measure_selected as measurement
import portable_evidence
import train_optimized as training
import validate_training_deployment as deployment


SCRIPTS = Path(__file__).resolve().parent
HELPERS = ("train_deployed.py", "deployed_training_admission.py", "validate_training_deployment.py", "portable_evidence.py",
           "train_optimized.py", "measure_selected.py", "post_comparison_validation.py", "infrastructure_retry.py", "pregather_async_retry.py",
           "h100_nvl_slurm_queue.py", "h100_nvl_slurm_case.py", "h100_nvl_slurm_evidence.py", "h100_nvl_slurm_site_validation.py",
           "h100_nvl_slurm_qualification.py", "h100_nvl_allocation_observation.py",
           "controller.py", "ideas.py", "queue_worker.py", "diagnose_gpu_execution.py", 'device_memory_limit.py')


def helper_receipt():
    """Pin the exact detached runner/helper Git closure before any device context is created."""
    root = SCRIPTS.parents[2]
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    deployment.git_root(str(root), commit, admission.validation.Evidence())
    for name, module in (("deployed_training_admission.py", admission), ("measure_selected.py", measurement),
                         ("portable_evidence.py", portable_evidence), ("train_optimized.py", training),
                         ("validate_training_deployment.py", deployment),
                         ("post_comparison_validation.py", admission.validation),
                         ("diagnose_gpu_execution.py", deployment.diagnostic)):
        if Path(module.__file__).resolve() != SCRIPTS / name:
            raise ValueError("the deployed runner imported a helper from another checkout: " + name)
    return training.source_receipt(root, commit, [str((SCRIPTS / name).relative_to(root)) for name in HELPERS])


def compute_placement(environment, hostname, cgroup=None):
    """Require the actual Slurm job cgroup or the existing authorized serval02 card before metadata GPU queries."""
    placement = training.training_placement(environment, hostname)
    if placement["kind"] == "slurm":
        cgroup = Path("/proc/self/cgroup").read_text() if cgroup is None else cgroup
        if not re.search(r"(?:^|/)job_" + re.escape(placement["slurm_job_id"]) + r"(?:/|$)", cgroup, re.MULTILINE):
            raise ValueError("the deployed runner is outside its recorded Slurm compute allocation")
    return placement


def runtime_environment(required, environment):
    """Apply the original game/preallocation defaults before recording the actual precision/cache environment."""
    for name, expected in (("XLA_PYTHON_CLIENT_PREALLOCATE", "false"), ("JAXTARI_SKIP_SPRITE_UPDATE", "1")):
        if environment.get(name, expected) != expected:
            raise ValueError("deployed training inherited another game/preallocation setting: " + name)
        environment[name] = expected
    return training.enforce_runtime(required, environment)


def deployment_receipt(specification):
    """Recheck the admitted receipt bytes before using them for actual placement and backend gates."""
    path = Path(specification["operational_paths"]["deployment_verification"])
    contents = path.read_bytes()
    receipt = portable_evidence.document(contents)
    if (hashlib.sha256(contents).hexdigest() != specification["deployment_verification_sha256"]
            or admission.validation.canonical_digest(receipt) != specification["validation"]["deployment_digest"]):
        raise ValueError("the deployment receipt changed after raw-evidence admission")
    return receipt


def immutable_specification(specification):
    """Keep operational transport fields out of the immutable training-unit specification digest."""
    return {key: value for key, value in specification.items()
            if key not in {"validation", "operational_paths", "evidence_resolution"}}


class DeploymentMetadata:
    """Gate actual metadata while leaving the existing trainer in charge of every state/update/checkpoint."""

    def __init__(self, specification, receipt, server, hardware, assets, directory, attempt_directory):
        """Retain checked metadata and per-attempt output paths, never a learner state or array."""
        if server != receipt["target"]["server"]:
            raise ValueError("the assigned training server differs from the admitted deployment")
        deployment.hardware_identity(hardware, receipt["target"])
        observed = deployment.actual_runtime(hardware["runtime_environment"], receipt["runtime_environment"])
        if observed != receipt["actual_runtime_environment"]:
            raise ValueError("actual training runtime differs from its target confirmation/packing")
        if {key: value for key, value in assets.items() if key != "root"} != receipt["assets"]:
            raise ValueError("actual game assets differ from the deployment's verified manifest/tree")
        self.specification, self.receipt, self.server = specification, receipt, server
        self.hardware, self.assets = hardware, assets
        self.directory, self.attempt_directory = directory, attempt_directory
        self.installed = None

    def before_numerical_imports(self, installed):
        """Require the actual installed package/Python/library metadata before JAX or model initialization."""
        expected = self.receipt["backend"]
        public = {key: installed["jaxtari_source"][key] for key in ("source_commit", "version", "tree_digest", "python_files")}
        observed = {"packages": installed["packages"], "jaxtari_source": public, "gpu_libraries": installed["gpu_libraries"]}
        wanted = {key: expected[key] for key in observed}
        if admission.validation.canonical_digest(observed) != admission.validation.canonical_digest(wanted):
            raise ValueError("installed training packages/Python/GPU-library bytes differ from the target backend")
        self.installed = copy.deepcopy(installed)
        training.atomic_json(self.attempt_directory / "installed_backend.json", installed)
        training.atomic_json(self.attempt_directory / "placement_hardware.json", self.hardware)
        training.atomic_json(self.attempt_directory / "asset_receipt.json", self.assets)

    def identity_metadata(self, runtime, backend, assets):
        """Keep logical paths and source bytes in checkpoint identity while recording physical paths per attempt."""
        if self.installed is None:
            raise ValueError("installed backend admission must precede checkpoint identity")
        observed_runtime = deployment.actual_runtime(runtime, self.receipt["runtime_environment"])
        if (observed_runtime != self.receipt["actual_runtime_environment"] or assets != self.assets
                or backend.get("driver_version") != self.receipt["target"]["driver_version"]
                or backend.get("packages") != self.installed["packages"]
                or backend.get("jaxtari_source") != self.installed["jaxtari_source"]
                or backend.get("installed_gpu_libraries") != self.installed["gpu_libraries"]):
            raise ValueError("training identity metadata changed after its installed backend/asset/runtime gate")
        identity_backend, identity_assets = copy.deepcopy(backend), copy.deepcopy(assets)
        # Before: operational public-source/root paths. After: original immutable identities with all verified bytes retained.
        identity_backend["jaxtari_source"] = {**self.receipt["backend"]["jaxtari_source"],
                                              "source_root": self.specification["jaxtari_source_root"]}
        identity_assets["root"] = self.specification["asset_record"]["root"]
        return {"runtime_environment": observed_runtime,
            "backend": identity_backend, "assets": identity_assets,
            "deployment": {"deployment_digest": self.specification["validation"]["deployment_digest"],
                "specification_digest": self.specification["validation"]["specification_digest"], "target": self.receipt["target"],
                "backend_digest": admission.validation.canonical_digest(self.receipt["backend"]), "assets": self.receipt["assets"]}}

    def after_first_update(self, loaded_libraries):
        """Verify lazy libraries after the first call, before running progress or later training checkpoints.

        The shared trainer may already have saved the initial warmup state. That
        initial save cannot establish completed training or replace this gate.
        """
        if self.installed is None:
            raise ValueError("installed backend admission must precede mapped-library verification")
        match = deployment.match_actual_backend(self.receipt, self.server, self.hardware, self.installed, loaded_libraries)
        training.atomic_json(self.attempt_directory / "deployment_backend_match.json", match)
        training.atomic_json(self.directory / "deployment_backend_match.json", match)


def run(arguments):
    """Admit one explicit unit, lock its actual GPU, and delegate all numerical work to the shared trainer."""
    if any(name in sys.modules for name in ("jax", "numpy", "torch")):
        raise RuntimeError("deployed training must begin in a fresh process before numerical imports")
    placement = compute_placement(os.environ, socket.gethostname())
    resolver = portable_evidence.ExactPathResolver(arguments.evidence_manifest, arguments.evidence_manifest_sha256) if arguments.evidence_manifest else None
    specification = admission.validate_deployed_training_admission(arguments.final_spec, resolver)
    if arguments.unit_id != specification["unit_id"]:
        raise ValueError("the requested unit differs from the reserved deployed-training seed block")
    receipt = deployment_receipt(specification)
    if arguments.server != receipt["target"]["server"]:
        raise ValueError("the requested server differs from the target deployment receipt")
    helpers = helper_receipt()
    paths = specification["operational_paths"]
    source_root = Path(paths["source_root"])
    training.source_receipt(source_root, specification["source_commit"])
    runtime = runtime_environment(specification["runtime_environment"], os.environ)
    for name in ("experiment_background.md", "manifest.yaml", "config_resolved.yaml", "command.txt"):
        if not (arguments.run_dir / name).is_file():
            raise ValueError("the deployed campaign is missing its launch-time record: " + name)
    directory = arguments.run_dir / "data" / arguments.unit_id
    gpu = placement["gpu_uuid"] if placement["kind"] == "serval02" else measurement.allocated_gpu_uuid("h100", os.environ)
    plan = {"gpu_uuid": gpu, "hardware_class": "h100"}
    with training.unit_lock(directory):
        training.prepare_unit(directory, immutable_specification(specification))
        attempt, attempt_directory = training.new_attempt(directory)
        progress = {"state": "waiting_for_placement", "attempt": attempt, "pid": os.getpid(), "time": training.utc_now(),
                    "placement": placement, "gpu_uuid": gpu, "server": arguments.server}
        training.atomic_json(directory / "progress.json", progress)
        training.atomic_json(attempt_directory / "progress.json", progress)
        try:
            # The shared placement guard retains the standing card lock without overwriting attempt-aware progress.
            with training.placement_guard(placement):
                hardware = {**measurement.preflight(plan), "runtime_environment": runtime}
                deployment.hardware_identity(hardware, receipt["target"])
                record = {"root": paths["asset_root"], "manifest": paths["asset_manifest"], "sha256": specification["asset_record"]["sha256"]}
                assets = training.verify_assets(record, os.environ)
                training.atomic_json(attempt_directory / "operational_paths.json", paths)
                hooks = DeploymentMetadata(specification, receipt, arguments.server, hardware, assets, directory, attempt_directory)
                training._train(arguments, specification, source_root, helpers, runtime, assets, directory, attempt, attempt_directory,
                                operational_paths=paths, metadata_hooks=hooks)
        except Exception as error:
            # The shared trainer journals its own failures; this covers placement/assets/metadata failures before entry.
            if not (attempt_directory / "failure.json").exists():
                failure = {"error": repr(error), "time": training.utc_now(), "pid": os.getpid(), "attempt": attempt}
                training.atomic_json(attempt_directory / "failure.json", failure)
                training.atomic_json(directory / "failure.json", failure)
                failed = {**progress, **failure, "state": "failed"}
                training.atomic_json(attempt_directory / "progress.json", failed)
                training.atomic_json(directory / "progress.json", failed)
            raise


def parse_arguments(argv=None):
    """Expose identity/placement/stop inputs only, with no learner overrides or job submission."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-spec", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--server", required=True)
    parser.add_argument("--evidence-manifest", type=Path)
    parser.add_argument("--evidence-manifest-sha256")
    parser.add_argument("--checkpoint-seconds", type=int, default=1200)
    parser.add_argument("--maximum-seconds", type=float, default=0)
    arguments = parser.parse_args(argv)
    portable_evidence.original_path(str(arguments.final_spec))
    if (Path(arguments.unit_id).name != arguments.unit_id or arguments.unit_id in {"", ".", ".."}
            or not arguments.server or not 0 < arguments.checkpoint_seconds <= 1200
            or not math.isfinite(arguments.maximum_seconds) or arguments.maximum_seconds < 0
            or arguments.evidence_manifest_sha256 is not None and arguments.evidence_manifest is None):
        parser.error("use an explicit unit/server, checkpoint interval in (0,1200], finite stop time and a complete manifest input")
    arguments.run_dir = arguments.run_dir.resolve(strict=True)
    return arguments


if __name__ == "__main__":
    run(parse_arguments())
