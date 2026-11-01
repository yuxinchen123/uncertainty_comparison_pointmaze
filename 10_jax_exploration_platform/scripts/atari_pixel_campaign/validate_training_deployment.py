"""Validate final training on an explicit server and actual hardware/software group.

This preserves the full original fifty-idea verification and requires separate
target correctness, confirmation and packing evidence. It creates no jobs and
does not change training admission or the numerical runner. Real validation
recomputes measured statistics and belongs on allocated compute.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import subprocess

import diagnose_gpu_execution as diagnostic
import post_comparison_validation as validation
import train_optimized as training


PREFIX = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"
JAXTARI_COMMIT = training.JAXTARI_COMMIT
REQUIRED_HELPERS = {PREFIX + name for name in ("measure_selected.py", "diagnose_gpu_execution.py",
                                               "post_comparison_validation.py", "infrastructure_retry.py", "pregather_async_retry.py",
                                               "h100_nvl_slurm_queue.py", "h100_nvl_slurm_case.py", "h100_nvl_slurm_evidence.py", "h100_nvl_slurm_site_validation.py",
                                               "h100_nvl_slurm_qualification.py", "h100_nvl_allocation_observation.py",
                                               "train_optimized.py", "short_learning.py", "device_memory_limit.py", "validate_training_deployment.py", "portable_evidence.py")}


def sha256(value):
    """Require a complete lowercase SHA256 value before trusting an artifact receipt."""
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("a deployment artifact needs its complete SHA256")
    return value


class DeploymentEvidence(validation.Evidence):
    """Retain streamed binary receipts beside ordinary raw JSON/source evidence."""

    def __init__(self, resolver=None):
        """Start separate installed-backend caches without changing global verification receipts."""
        super().__init__(resolver)
        self.verified_backends = {}
        self.images = {}

    def image(self, path, expected):
        """Hash an actual installed image without loading the shared library or its full contents."""
        expected = sha256(expected)
        key = self.path_identity(path)
        if not Path(path).is_absolute():
            raise ValueError("GPU library images require absolute original paths")
        physical = self.physical_file(path)
        actual = training.file_sha256(physical)
        if actual != expected or key in self.images and self.images[key] != expected:
            raise ValueError("a deployment GPU library differs from its recorded bytes: " + key)
        if key in self.receipts and self.receipts[key]["sha256"] != expected:
            raise ValueError("a deployment evidence image changed during validation: " + key)
        self.images[key] = expected
        self.receipts[key] = {"path": key, "sha256": expected}

    def recheck_images(self):
        """Require every large image to retain its verified bytes at receipt publication."""
        for path, expected in list(self.images.items()):
            self.image(path, expected)


def git_root(root, commit, evidence):
    """Require a real detached Git top at the exact recorded source or helper commit."""
    if not Path(root).is_absolute() or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("deployment sources need an absolute Git root and full commit")
    physical = evidence.git_root(root, commit)
    top = subprocess.check_output(["git", "-C", str(physical), "rev-parse", "--show-toplevel"], text=True).strip()
    actual = subprocess.check_output(["git", "-C", str(physical), "rev-parse", "HEAD"], text=True).strip()
    branch = subprocess.run(["git", "-C", str(physical), "symbolic-ref", "--quiet", "HEAD"], capture_output=True, text=True)
    changed = subprocess.check_output(["git", "-C", str(physical), "status", "--porcelain", "--untracked-files=all"], text=True).strip()
    if Path(top).resolve(strict=True) != physical.resolve(strict=True) or actual != commit or branch.returncode != 1 or changed:
        raise ValueError("deployment evidence requires an unchanged detached Git source")
    return physical


def committed_files(receipt, required, evidence):
    """Verify every committed helper/Executor byte named by a target measurement receipt."""
    root, commit, files = receipt["source_root"], receipt["source_commit"], receipt["files"]
    physical = git_root(root, commit, evidence)
    verified = {}
    for relative, expected in files.items():
        if not relative.startswith(PREFIX) or ".." in Path(relative).parts:
            raise ValueError("deployment helper receipts require explicit campaign source paths")
        original = subprocess.check_output(["git", "-C", str(physical), "show", commit + ":" + relative])
        if hashlib.sha256(original).hexdigest() != sha256(expected):
            raise ValueError("a target helper differs from its committed source bytes")
        verified[relative] = original
    if not validation.helper_requirements(required, verified) <= set(files):
        raise ValueError("target measurement omitted a required source helper")


def actual_runtime(value, expected):
    """Require observed deterministic full precision and disabled persistent compilation caches."""
    if (not isinstance(value, dict) or validation.runtime_environment({key: value.get(key) for key in validation.RUNTIME_KEYS}) != expected
            or value.get("JAX_ENABLE_COMPILATION_CACHE") != "false"
            or value.get("JAX_COMPILATION_CACHE_DIR") not in (None, "")
            or value.get("JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES") != ""):
        raise ValueError("target evidence lacks its actual full-precision uncached runtime")
    # Before: GPU selectors name this process's card; after: hardware records retain them outside the reusable runtime group.
    return json.loads(json.dumps({key: item for key, item in value.items()
                                 if key not in {"CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"}}, allow_nan=False))


def hardware_identity(hardware, target):
    """Match an actual selected GPU to an explicit server node group, model, memory and driver."""
    required = {"server", "hostnames", "gpu_model", "total_memory_bytes", "driver_version"}
    if (set(target) != required or not isinstance(target["server"], str) or not target["server"].strip()
            or not isinstance(target["hostnames"], list) or not target["hostnames"]
            or any(not isinstance(host, str) or not host or host != host.split(".")[0] for host in target["hostnames"])
            or len(set(target["hostnames"])) != len(target["hostnames"])
            or type(target["total_memory_bytes"]) is not int or target["total_memory_bytes"] <= 0
            or not isinstance(target["gpu_model"], str) or not target["gpu_model"]
            or not isinstance(target["driver_version"], str) or not target["driver_version"]):
        raise ValueError("deployment needs an explicit server, concrete node group and complete GPU identity")
    rows = list(csv.reader(hardware.get("gpu_query", "").splitlines()))
    if len(rows) != 1 or len(rows[0]) != 4:
        raise ValueError("target hardware must record exactly one selected GPU query")
    gpu, model, memory, driver = (part.strip() for part in rows[0])
    size = re.fullmatch(r"(\d+) MiB", memory)
    if (not size or not gpu.startswith("GPU-") or hardware.get("gpu_uuid") != gpu
            or hardware.get("hostname") not in target["hostnames"] or model != target["gpu_model"]
            or int(size.group(1)) * 2 ** 20 != target["total_memory_bytes"] or driver != target["driver_version"]):
        raise ValueError("actual target node/model/memory/driver differs from its deployment group")
    if (type(hardware.get("pid")) is not int or hardware["pid"] < 1
            or not isinstance(hardware.get("process_start_time"), str) or not hardware["process_start_time"].isdigit()):
        raise ValueError("target hardware requires its actual process PID and birth counter")
    return {"server": target["server"], "gpu_model": model, "total_memory_bytes": int(size.group(1)) * 2 ** 20,
            "driver_version": driver}


def jaxtari_identity(record, packages, evidence):
    """Check the complete installed Python-file receipt against the pinned public JAXtari Git tree."""
    files = record.get("python_files", {})
    if (record.get("source_commit") != JAXTARI_COMMIT or record.get("version") != packages["jaxtari"]
            or not files or validation.canonical_digest(files) != record.get("tree_digest")):
        raise ValueError("target JAXtari source/version/tree digest is incomplete or changed")
    root = git_root(record["source_root"], JAXTARI_COMMIT, evidence)
    entries = subprocess.check_output(["git", "-C", str(root), "ls-tree", "-rz", JAXTARI_COMMIT, "--", "src/jaxtari"])
    expected = {}
    for entry in entries.split(b"\0"):
        if entry:
            header, encoded = entry.split(b"\t", 1)
            name = encoded.decode()
            if name.endswith(".py"):
                expected[name.removeprefix("src/")] = header.decode().split()[2]
    if set(files) != set(expected):
        raise ValueError("target JAXtari receipt changed its complete Python file set")
    for name, blob in expected.items():
        contents = subprocess.check_output(["git", "-C", str(root), "show", JAXTARI_COMMIT + ":src/" + name])
        if files[name] != {"git_blob": blob, "sha256": hashlib.sha256(contents).hexdigest()}:
            raise ValueError("target JAXtari Python differs from the pinned public Git bytes")
    return {key: record[key] for key in ("source_commit", "version", "tree_digest", "python_files")}


def backend_identity(installed, loaded, evidence):
    """Bind full package versions and every installed/mapped cuDNN/CUDA-runtime image to actual bytes."""
    if not isinstance(installed, dict) or set(installed) != {"packages", "jaxtari_source", "gpu_libraries"}:
        raise ValueError("target result needs its full installed backend record")
    packages, libraries = installed["packages"], installed["gpu_libraries"]
    if (set(packages) != set(training.PACKAGE_NAMES) or any(not isinstance(value, str) or not value for value in packages.values())
            or not isinstance(libraries, dict) or not libraries):
        raise ValueError("target backend needs every required installed package and GPU library")
    cache_key = validation.canonical_digest(installed)
    if cache_key not in evidence.verified_backends:
        python_source = jaxtari_identity(installed["jaxtari_source"], packages, evidence)
        for path, receipt in libraries.items():
            name = Path(path).name
            package = "nvidia-cudnn-cu12" if name.startswith("libcudnn") else "nvidia-cuda-runtime-cu12" if name.startswith("libcudart") else None
            if (package is None or set(receipt) != {"package", "version", "sha256"}
                    or receipt["package"] != package or receipt["version"] != packages[package]):
                raise ValueError("target GPU image lacks its matching installed package/version")
            evidence.image(path, receipt["sha256"])
        if not any(Path(path).name.startswith("libcudnn.so") for path in libraries) or not any(
                Path(path).name.startswith("libcudart.so") for path in libraries):
            raise ValueError("target backend omits its cuDNN or CUDA runtime image")
        evidence.verified_backends[cache_key] = python_source
    verified = loaded.get("verified_package_libraries", {})
    mapped = {path for path in loaded.get("loaded_shared_libraries", []) if Path(path).name.startswith(("libcudnn", "libcudart"))}
    if (loaded.get("platform") != "gpu" or not isinstance(loaded.get("platform_version"), str) or not loaded["platform_version"]
            or type(loaded.get("cudnnGetVersion")) is not int or loaded["cudnnGetVersion"] <= 0
            or type(loaded.get("cudaRuntimeGetVersion")) is not int or loaded["cudaRuntimeGetVersion"] <= 0
            or not mapped or set(verified) != mapped or any(path not in libraries or value != libraries[path] for path, value in verified.items())
            or any(loaded.get(name) not in verified for name in ("cudnn_library", "cuda_runtime_library"))):
        raise ValueError("target mapped GPU libraries differ from the verified installed images")
    return {"packages": packages, "jaxtari_source": evidence.verified_backends[cache_key], "gpu_libraries": libraries,
            "mapped_library_api": {name: loaded[name] for name in (
                "platform", "platform_version", "cudnnGetVersion", "cudaRuntimeGetVersion")}}


def asset_identity(directory, plan, evidence):
    """Validate the producer's manifest/asset receipt without reading the individual game assets again."""
    record = plan.get("asset_record")
    if not isinstance(record, dict) or set(record) != {"root", "manifest", "sha256"}:
        raise ValueError("measurement needs its complete original game-asset record")
    if not Path(record["root"]).is_absolute() or not Path(record["manifest"]).is_absolute():
        raise ValueError("game-asset root and manifest need absolute original paths")
    root = evidence.path_identity(record["root"])
    manifest_path = evidence.path_identity(record["manifest"])
    if Path(manifest_path).is_relative_to(root):
        raise ValueError("the recorded game-asset manifest must remain outside its asset root")
    contents = evidence.file(record["manifest"])
    if hashlib.sha256(contents).hexdigest() != sha256(record["sha256"]):
        raise ValueError("the game-asset manifest differs from its prepared SHA256")
    manifest = json.loads(contents)
    if (not isinstance(manifest, dict) or set(manifest) != {"schema_version", "source_commit", "files"}
            or type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1
            or manifest["source_commit"] != JAXTARI_COMMIT or not isinstance(manifest["files"], dict) or not manifest["files"]):
        raise ValueError("the game-asset manifest must identify the pinned public source and complete file map")
    # Before: filename/hash rows in the original manifest; after: metadata-only tree identity, with no asset-file reads.
    for name, digest in manifest["files"].items():
        if (not isinstance(name, str) or not name or "\0" in name or PurePosixPath(name).is_absolute()
                or ".." in PurePosixPath(name).parts or str(PurePosixPath(name)) != name or name == "."):
            raise ValueError("game-asset manifest names must be canonical relative file paths")
        sha256(digest)
    expected = {"root": root, "manifest_sha256": record["sha256"], "source_commit": JAXTARI_COMMIT,
                "file_count": len(manifest["files"]), "tree_digest": validation.canonical_digest(manifest["files"])}
    receipt_path = directory / "asset_receipt.json"
    receipt = evidence.read(receipt_path)
    if type(receipt.get("file_count")) is not int or receipt != expected:
        raise ValueError("the producer game-asset receipt disagrees with its actual manifest/tree/source/file count")
    return {"identity": {key: value for key, value in expected.items() if key != "root"},
            "root": root, "manifest": manifest_path, "asset_receipt": evidence.path_identity(receipt_path),
            "measurement_plan": evidence.path_identity(directory / "measurement_plan.json")}


def global_asset_proof(final, evidence):
    """Bind the global selected H100 confirmation to its immutable producer plan and actual asset proof."""
    path = final["specification"]["contexts"]["h100"]["confirmation_record"]
    directory = Path(path).parent
    result = evidence.read(path)
    plan = evidence.read(directory / "measurement_plan.json")
    if 'device_memory_limit_protocol' in plan:
        raise ValueError('an instrumented attempt cannot supply ordinary target measurements; remeasure it normally')
    if (evidence.read(directory / "measurement_prepared.json") != {
            "plan_sha256": hashlib.sha256(evidence.file(directory / "measurement_plan.json")).hexdigest()}
            or type(plan.get("schema_version")) is not int or plan["schema_version"] != 1
            or plan.get("kind") != "selected_configuration_measurement" or plan.get("mode") != "confirmation"
            or result.get("measurement_mode") != "confirmation" or plan.get("hardware_class") != "h100"
            or plan.get("source_commit") != final["source_commit"]
            or plan.get("configuration") != final["selected"]["h100"]["configuration"]
            or plan.get("configuration") != result["candidate"] or plan.get("reference_configuration") != result["reference"]
            or validation.runtime_environment(plan["runtime_environment"]) != final["runtime_environment"]
            or result.get("helpers") != plan.get("helpers")):
        raise ValueError("global selected confirmation lacks its original prepared producer/asset proof")
    committed_files(plan["helpers"], REQUIRED_HELPERS, evidence)
    for relative, expected in plan["helpers"]["files"].items():
        if hashlib.sha256(evidence.file(directory / "code" / Path(relative).name)).hexdigest() != expected:
            raise ValueError("global selected confirmation changed its executed producer helper")
    return asset_identity(directory, plan, evidence)


def target_record(path, mode, specification, evidence):
    """Require a result's prepared source/helpers/configuration and raw-bound actual hardware/backend."""
    directory = Path(path).parent
    if Path(path).name != "result.json" or directory.parent != Path(specification["campaign_run_dir"]) / "data":
        raise ValueError("target measurements must be direct prepared units in their recorded target campaign")
    result = evidence.read(path)
    plan = evidence.read(directory / "measurement_plan.json")
    if "configured_isolation" in plan or "configured_isolation" in result:
        raise ValueError("configured-isolation target/final/portable records are unsupported")
    if 'device_memory_limit_protocol' in plan:
        raise ValueError('an instrumented attempt cannot supply ordinary target measurements; remeasure it normally')
    if evidence.read(directory / "measurement_prepared.json") != {
            "plan_sha256": hashlib.sha256(evidence.file(directory / "measurement_plan.json")).hexdigest()}:
        raise ValueError("target result differs from its immutable preparation")
    expected = {"schema_version": 1, "kind": "selected_configuration_measurement", "mode": mode,
        "source_root": specification["source_root"], "source_commit": specification["source_commit"],
        "hardware_class": specification["hardware_class"], "campaign_run_dir": specification["campaign_run_dir"],
        "unit_id": directory.name, "compilation_cache_policy": "disabled"}
    if any(plan.get(key) != value for key, value in expected.items()) or result.get("measurement_mode") != mode:
        raise ValueError("target measurement changed its source/context/mode or disabled-cache protocol")
    validation.identity(result, specification["hardware_class"], specification["source_commit"], specification["runtime_environment"])
    if validation.runtime_environment(plan["runtime_environment"]) != specification["runtime_environment"]:
        raise ValueError("target plan changed its full-precision runtime")
    source = git_root(specification["source_root"], specification["source_commit"], evidence)
    core = diagnostic.core_identity(source, specification["source_commit"])
    if plan.get("core") != {**core, "source_root": specification["source_root"]}:
        raise ValueError("target measurement changed its exact numerical core")
    paired = plan["paired_source"]
    if (paired["source_root"], paired["source_commit"]) != (specification["source_root"], specification["source_commit"]):
        raise ValueError("target Executor source differs from the selected numerical commit")
    committed_files(paired, {PREFIX + name for name in ("paired_benchmark.py", "ideas.py")}, evidence)
    committed_files(plan["helpers"], REQUIRED_HELPERS, evidence)
    if result.get("helpers") != plan["helpers"]:
        raise ValueError("target result changed its committed producer helpers")
    for relative, expected_hash in plan["helpers"]["files"].items():
        if hashlib.sha256(evidence.file(directory / "code" / Path(relative).name)).hexdigest() != expected_hash:
            raise ValueError("a target staged helper differs from its committed receipt")
    configuration = validation.configuration(plan["configuration"])
    defaults = diagnostic.default_configuration(source)
    configurations = [configuration]
    if mode == "packing-point":
        if plan["reference_configuration"] is not None or result.get("configuration") != configuration:
            raise ValueError("target packing retained another variant or changed its configuration")
    else:
        reference = validation.configuration(plan["reference_configuration"])
        configurations.append(reference)
        if result.get("candidate") != configuration or result.get("reference") != reference:
            raise ValueError("target result differs from its prepared full configuration")
    if any(set(config[scope]) != set(defaults[scope]) for config in configurations for scope in ("learner", "environment")):
        raise ValueError("target configurations must resolve every pinned learner/environment field")
    multiple = 1
    for config in configurations:
        execution = config["execution"]
        multiple = math.lcm(multiple, execution["iterations_per_call"])
        if execution["report_readout"] != "none":
            multiple = math.lcm(multiple, execution["report_every_iterations"])
    final_confirmation = mode == "confirmation" and "final_confirmation_protocol" in plan
    rounds_count, minimum = (11, 20) if final_confirmation else (7, 10)
    updates = ((minimum + multiple - 1) // multiple) * multiple
    if final_confirmation:
        validation.final_protocol(plan["final_confirmation_protocol"])
    if (plan.get("rounds") != rounds_count or plan.get("minimum_updates_per_round") != minimum or plan.get("iterations") != updates
            or result.get("protocol") != {"rounds": rounds_count, "requested_minimum_updates_per_round": minimum, "actual_updates_per_round": updates}):
        raise ValueError("target measurements changed their complete final-confirmation or seven-by-ten packing protocol")
    rounds = result.get("rounds", [])
    rows = [pair[arm] for pair in rounds for arm in ("reference", "candidate")] if mode == "confirmation" else rounds
    if mode not in {"correctness", "learning-check"} and (len(rounds) != rounds_count or any(row.get("iterations") != updates for row in rows)):
        raise ValueError("target rounds changed their prepared complete-update window")
    if mode == "correctness" and rounds:
        raise ValueError("target result differs from its prepared full configuration")
    seeds = plan["copy_seeds"]
    if (result.get("copy_seeds") != seeds or len(seeds) != configuration["learner"]["n_copies"]
            or len(set(seeds)) != len(seeds) or any(type(seed) is not int or not 0 <= seed < 2 ** 32 for seed in seeds)):
        raise ValueError("target result changed its explicit independent policy seeds")
    hardware, installed = evidence.read(directory / "hardware.json"), evidence.read(directory / "installed_backend.json")
    if result.get("hardware") != hardware or result.get("installed_backend") != installed:
        raise ValueError("target result does not bind its exact hardware and installed-backend sidecars")
    process = validation.execution_identity(result)
    if (hardware.get("hostname"), hardware.get("pid"), hardware.get("process_start_time")) != process or hardware.get("gpu_uuid") != plan["gpu_uuid"]:
        raise ValueError("target result used another process or physical GPU")
    if any(row.get("memory", {}).get("gpu_uuid") != plan["gpu_uuid"] for row in rows):
        raise ValueError("target measured rounds used another physical GPU")
    runtime = actual_runtime(hardware.get("runtime_environment"), specification["runtime_environment"])
    hardware_identity(hardware, specification["target"])
    if result["loaded_gpu_libraries"].get("pid") != process[1]:
        raise ValueError("target mapped-library receipt belongs to another process")
    backend = backend_identity(installed, result["loaded_gpu_libraries"], evidence)
    assets = asset_identity(directory, plan, evidence)
    return result, configuration, process, runtime, backend, assets


def seed_admission(path, unit_id, seeds, configuration, context, evidence):
    """Bind one immutable reserved seed block while permitting later disjoint manifest additions."""
    manifest = json.loads(evidence.file(path, record=False))
    if (type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1 or manifest.get("kind") != "final_policy_seed_blocks"
            or not isinstance(manifest.get("units"), dict)):
        raise ValueError("deployment requires the shared global final-policy seed manifest")
    all_seeds = []
    for key, row in manifest["units"].items():
        current = row.get("copy_seeds", [])
        if (set(row) != {"unit_id", "hardware_class", "copy_seeds", "configuration_digest"}
                or row.get("unit_id") != key or row.get("hardware_class") not in validation.CONTEXTS or not current
                or any(type(seed) is not int or not 0 <= seed < 2 ** 32 for seed in current)
                or current != list(range(current[0], current[0] + len(current)))):
            raise ValueError("a global final seed block has an invalid identity")
        sha256(row["configuration_digest"])
        all_seeds.extend(current)
    expected = {"unit_id": unit_id, "hardware_class": context, "copy_seeds": seeds,
                "configuration_digest": validation.canonical_digest(configuration)}
    if (len(seeds) != configuration["learner"]["n_copies"] or len(all_seeds) != len(set(all_seeds))
            or manifest["units"].get(unit_id) != expected):
        raise ValueError("deployment seeds/configuration differ from their globally disjoint reserved block")
    return {"path": evidence.path_identity(path), "unit": expected, "unit_digest": validation.canonical_digest(expected),
            "scope": "selected immutable seed block; later disjoint additions are permitted"}


def global_processes(final, evidence):
    """Collect already used H100 process identities so a new target cannot reuse NVL executions."""
    context = final["specification"]["contexts"]["h100"]
    processes = set()
    for name in ("correctness_record", "confirmation_record", "learning_record"):
        if name not in context:
            continue
        processes.add(validation.execution_identity(evidence.read(context[name])))
    for point in final["selected"]["h100"]["packing"]["points"] + final["selected"]["h100"]["packing"]["confirmation"]:
        processes.add(tuple(point["process"]))
    capacity = final['selected']['h100']['packing'].get('capacity_exclusion')
    if capacity is not None:
        processes.add(tuple(capacity['process']))
    registry = evidence.read(final["specification"]["study_registry"])
    entries = list(registry["initial_ideas"]) + [entry for batch in registry["adaptive_batches"] for entry in batch["hypotheses"]]
    for entry in entries:
        for row in entry["results"]["h100"]:
            record = evidence.read(row["result_path"])
            if record.get("execution_identity"):
                processes.add(validation.execution_identity(record))
            hardware = evidence.read(Path(row["result_path"]).parent / "hardware.json")
            if hardware.get("hostname"):
                processes.add((hardware["hostname"], hardware["pid"], hardware["process_start_time"]))
    return processes


def validate_deployment(specification, resolver=None, specification_path=None, *, historical_read_only=False):
    """Rebuild global evidence and derive one target-bound training configuration and seed admission."""
    evidence = DeploymentEvidence(resolver)
    required = {"schema_version", "final_verification", "source_root", "source_commit", "hardware_class", "runtime_environment",
        "campaign_run_dir", "target", "correctness_record", "confirmation_record", "packing_records", "packing_confirmation_records",
        "selected_copies", "unit_id", "copy_seeds", "seed_manifest"}
    if not historical_read_only:
        required |= {"learning_record", "learning_analysis_record"}
    extras = ({'packing_stop_rule', 'packing_memory_limit_record'}
              if specification.get('packing_stop_rule') == 'instrumented_device_memory_limit' else {'packing_stop_rule'})
    if (set(specification) not in (required, required | extras)
            or "packing_stop_rule" in specification and specification["packing_stop_rule"] not in {
                "measured_memory_headroom", 'instrumented_device_memory_limit'}
            or (specification.get('packing_stop_rule') == 'instrumented_device_memory_limit'
                and 'packing_memory_limit_record' not in specification)
            or type(specification["schema_version"]) is not int or specification["schema_version"] != 1
            or specification["hardware_class"] != "h100"):
        raise ValueError("this deployment specification requires the exact version-one H100 target fields")
    if specification_path is not None and evidence.read(specification_path) != specification:
        raise ValueError("the deployment specification changed")
    original_specification = specification
    runtime = validation.runtime_environment(specification["runtime_environment"])
    specification = {**specification, "runtime_environment": runtime}
    final = evidence.read(specification["final_verification"])
    if not historical_read_only:
        validation.require_current_final(final)
        specification["completed_registry_sha256"] = hashlib.sha256(evidence.file(final["specification"]["study_registry"])).hexdigest()
    rebuilt = validation.validate_final_verification(final["specification"], validation.Evidence(resolver), final.get("specification_path"))
    if rebuilt != final or final.get("kind") != "final_verification":
        raise ValueError("the complete original final verification changed or no longer passes")
    if specification["source_commit"] != final["source_commit"] or runtime != final["runtime_environment"]:
        raise ValueError("target deployment must retain the globally verified numerical source and runtime")
    global_assets = global_asset_proof(final, evidence)
    used = global_processes(final, evidence)
    combination = validation.validate_combination(specification, evidence, final=not historical_read_only)
    if not historical_read_only:
        final_confirmation = validation.validate_final_confirmation(specification, evidence)
        short_learning = validation.validate_short_learning(specification, final_confirmation, evidence)
    packing = validation.validate_packing(specification, evidence)
    selected = final["selected"]["h100"]["configuration"]
    if (not packing["complete"] or packing["configuration"] != combination["configuration"]
            or validation.configuration_without_copies(combination["configuration"]) != validation.configuration_without_copies(selected)):
        raise ValueError("target packing/confirmation must select the same globally accepted implementation")
    original_reference = evidence.read(final["specification"]["contexts"]["h100"]["confirmation_record"])["reference"]
    paths = [(specification["correctness_record"], "correctness"), (specification["confirmation_record"], "confirmation")]
    if not historical_read_only:
        paths.append((specification["learning_record"], "learning-check"))
    paths += [(path, "packing-point") for path in specification["packing_records"] + specification["packing_confirmation_records"]]
    identities, backend, actual, observations = set(), None, None, []
    for path, mode in paths:
        result, configuration, process, observed_runtime, observed_backend, observed_assets = target_record(path, mode, specification, evidence)
        if process in used or process in identities:
            raise ValueError("target deployment reused an original/global or another target process execution")
        identities.add(process)
        if mode != "packing-point" and validation.configuration_without_copies(result["reference"]) != validation.configuration_without_copies(original_reference):
            raise ValueError("target confirmation changed the globally recorded reference controls")
        if backend is not None and (observed_backend != backend or observed_runtime != actual):
            raise ValueError("target measurements crossed different software/runtime groups")
        if observed_assets["identity"] != global_assets["identity"]:
            raise ValueError("target game assets differ from the global selected confirmation's manifest/tree/source/file count")
        backend, actual = observed_backend, observed_runtime
        observations.append({"path": evidence.path_identity(path), "mode": mode, "process": list(process),
                             "gpu_uuid": result["hardware"]["gpu_uuid"], "asset_proof": observed_assets})
    if specification.get('packing_stop_rule') == 'instrumented_device_memory_limit':
        # Rebuild the separate failed-target proof independently; never admit it through target_record's completed windows.
        import device_memory_limit as capacity
        excluded = capacity.validate_exclusion(specification['packing_memory_limit_record'], specification, evidence)
        process = tuple(excluded['process'])
        hardware_identity(excluded['hardware'], specification['target'])
        if (process in used or process in identities or excluded['backend'] != backend or excluded['runtime'] != actual
                or excluded['assets']['identity'] != global_assets['identity']):
            raise ValueError('target capacity exclusion reused a process or crossed its qualified backend/runtime/assets')
        observations.append({'path': evidence.path_identity(specification['packing_memory_limit_record']),
            'mode': 'device-memory-capacity-exclusion', 'process': list(process), 'gpu_uuid': excluded['gpu_uuid'],
            'asset_proof': excluded['assets'], 'packing_point_credit': 0})
    seeds = seed_admission(specification["seed_manifest"], specification["unit_id"], specification["copy_seeds"],
                           packing["configuration"], "h100", evidence)
    evidence.recheck_images()
    receipt = {"schema_version": 1, "kind": "training_deployment_verification", "source_root": specification["source_root"],
        "source_commit": specification["source_commit"], "hardware_class": "h100", "target": specification["target"],
        "configuration": packing["configuration"], "copy_seeds": specification["copy_seeds"], "unit_id": specification["unit_id"],
        "runtime_environment": runtime, "actual_runtime_environment": actual, "backend": backend,
        "assets": global_assets["identity"], "global_asset_proof": global_assets,
        "global_final_verification": {"path": evidence.path_identity(specification["final_verification"]),
            "sha256": hashlib.sha256(evidence.file(specification["final_verification"])).hexdigest(),
            "specification_digest": final["specification_digest"]},
        "combination": combination, "packing": packing,
        "seed_admission": seeds, "executions": observations,
        "specification": original_specification, "specification_digest": validation.canonical_digest(original_specification),
        "specification_path": evidence.path_identity(specification_path) if specification_path is not None else None,
        "evidence": evidence.records()}
    if not historical_read_only:
        receipt["short_learning"] = short_learning
    return receipt


def match_actual_backend(receipt, server, hardware, installed_backend, loaded_gpu_libraries):
    """Match current compute metadata and verified bytes without requiring the prior physical GPU UUID."""
    if receipt.get("kind") != "training_deployment_verification" or server != receipt["target"]["server"]:
        raise ValueError("actual training server differs from its validated deployment")
    hardware_identity(hardware, receipt["target"])
    if loaded_gpu_libraries.get("pid") != hardware.get("pid"):
        raise ValueError("actual mapped-library receipt belongs to another process")
    actual = actual_runtime(hardware.get("runtime_environment"), receipt["runtime_environment"])
    evidence = DeploymentEvidence()
    backend = backend_identity(installed_backend, loaded_gpu_libraries, evidence)
    if actual != receipt["actual_runtime_environment"] or backend != receipt["backend"]:
        raise ValueError("actual training backend/runtime differs from the target confirmation and packing")
    return {"kind": "training_deployment_backend_match", "deployment_digest": validation.canonical_digest(receipt),
            "server": server, "hostname": hardware["hostname"], "gpu_uuid": hardware["gpu_uuid"],
            "backend_digest": validation.canonical_digest(backend), "verified_images": evidence.records()}


def main():
    """Validate an explicit compute-side deployment specification and print its raw-evidence receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specification", type=Path, required=True)
    parser.add_argument("--evidence-manifest", type=Path)
    parser.add_argument("--evidence-manifest-sha256")
    args = parser.parse_args()
    if args.evidence_manifest_sha256 and args.evidence_manifest is None:
        parser.error("an expected manifest hash requires --evidence-manifest")
    resolver = None
    if args.evidence_manifest is not None:
        from portable_evidence import ExactPathResolver
        resolver = ExactPathResolver(args.evidence_manifest, args.evidence_manifest_sha256)
    evidence = validation.Evidence(resolver)
    result = validate_deployment(evidence.read(args.specification), resolver, args.specification)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
