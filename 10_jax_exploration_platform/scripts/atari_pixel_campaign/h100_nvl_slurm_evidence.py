"""Validate captured H100 execution bytes without consulting the reader's runtime.

Operational allocation checks stay in the queue and site helpers. This module
reads historical artifacts only through post_comparison_validation.Evidence.
Original admission paths are identities; their immutable captured copies are
the inputs, so later admission revisions can reuse the original paths safely.
"""

import ast
import csv
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess


PREFIX = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"
EXECUTION_KIND = "h100_nvl_slurm_original_case"
TERMINAL_NAMES = ("attempt_result", "unit_result", "attempt_progress", "unit_progress")
JAXTARI_COMMIT = "04ab9aec318c72162e2e53503a594a489d7e162b"


def load(name):
    """Load the exact adjacent standard-library helper without replacing numerical module names."""
    path = Path(__file__).resolve().with_name(name + ".py")
    spec = importlib.util.spec_from_file_location("h100_history_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(condition, message):
    """Reject an incomplete raw proof with a specific explanation."""
    if not condition:
        raise ValueError(message)


def digest(contents):
    """Identify unchanged raw bytes rather than a reserialized document."""
    return hashlib.sha256(contents).hexdigest()


def file_receipt(path, evidence):
    """Read one exact original identity through the supplied evidence resolver."""
    contents = evidence.file(path)
    return {"path": evidence.path_identity(path), "sha256": digest(contents)}


def bound(record, evidence, document=True):
    """Require an explicit absolute path and the exact originally recorded bytes."""
    require(isinstance(record, dict) and set(record) == {"path", "sha256"}
        and Path(record["path"]).is_absolute() and re.fullmatch(r"[0-9a-f]{64}", record["sha256"]),
        "historical evidence needs an exact absolute path and SHA256")
    contents = evidence.file(record["path"])
    require(digest(contents) == record["sha256"], "historical evidence changed: " + record["path"])
    return json.loads(contents) if document else contents


def git_file(root, commit, relative, evidence):
    """Verify a historical file against its exact Git blob through a mapped real Git root."""
    require(Path(root).is_absolute() and re.fullmatch(r"[0-9a-f]{40}", commit)
        and not Path(relative).is_absolute() and ".." not in Path(relative).parts,
        "historical source needs an absolute root, full commit and relative file")
    physical = evidence.git_root(root, commit)
    command = ["git", "-C", str(physical)]
    top = subprocess.check_output([*command, "rev-parse", "--show-toplevel"], text=True).strip()
    head = subprocess.check_output([*command, "rev-parse", "HEAD"], text=True).strip()
    require(Path(top).resolve() == Path(physical).resolve() and head == commit,
        "historical source is not its exact real Git root and commit")
    require(not subprocess.check_output([*command, "status", "--porcelain", "--untracked-files=all", "--", relative], text=True).strip(),
        "historical source has changed used files")
    original = subprocess.check_output([*command, "show", commit + ":" + relative])
    contents = evidence.file(Path(root) / relative)
    require(contents == original, "historical source differs from its committed blob: " + relative)
    return contents


def git_record(record, evidence):
    """Rebuild a complete detached helper/core receipt, including every named source byte."""
    files = record.get("files", {})
    require(bool(files), "historical source receipt has no files")
    physical = evidence.git_root(record["source_root"], record["source_commit"])
    branch = subprocess.run(["git", "-C", str(physical), "symbolic-ref", "--quiet", "HEAD"], capture_output=True)
    require(branch.returncode == 1, "historical helper/core source must be detached")
    for relative, expected in files.items():
        contents = git_file(record["source_root"], record["source_commit"], relative, evidence)
        if isinstance(expected, dict):
            blob = hashlib.sha1(b"blob " + str(len(contents)).encode() + b"\0" + contents).hexdigest()
            require(expected == {"git_blob": blob, "sha256": digest(contents)}, "historical core blob receipt changed")
        else:
            require(digest(contents) == expected, "historical helper SHA256 changed: " + relative)
    return record


def numerical_paths(root, commit, evidence):
    """Enumerate the same complete committed numerical/gate closure used by site preparation."""
    physical = evidence.git_root(root, commit)
    scopes = ["10_jax_exploration_platform/src/exploration_platform/__init__.py",
        "10_jax_exploration_platform/src/exploration_platform/atari_pixels", PREFIX,
        "10_jax_exploration_platform/tests/atari_pixels", "10_jax_exploration_platform/tests/test_atari_pixel_environment.py"]
    names = subprocess.check_output(["git", "-C", str(physical), "ls-tree", "-r", "--name-only", commit, "--", *scopes], text=True).splitlines()
    return sorted(name for name in names if name.endswith((".py", "worker_environment.sh")))


def configuration(source, evidence):
    """Read all original learner and environment defaults as literals, without importing their modules."""
    root, commit = source["source_root"], source["source_commit"]
    relative = "10_jax_exploration_platform/src/exploration_platform/atari_pixels/"
    tree = ast.parse(git_file(root, commit, relative + "config.py", evidence))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PixelPPOConfig")
    learner = {node.target.id: ast.literal_eval(node.value) for node in cls.body if isinstance(node, ast.AnnAssign) and node.value is not None}
    tree = ast.parse(git_file(root, commit, relative + "environment.py", evidence))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AtariPixelEnvironment")
    constructor = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    environment = {argument.arg: ast.literal_eval(value) for argument, value in zip(constructor.args.args[-len(constructor.args.defaults):], constructor.args.defaults)}
    learner["n_copies"] = 1
    # Before: literal tuple defaults; after: the same JSON list values used in the immutable plan.
    return json.loads(json.dumps({"learner": learner, "environment": environment}))


def check_record(receipt, context, kind, plan, evidence):
    """Validate the original complete gate command and its captured hardware/precision evidence."""
    site = load("h100_nvl_slurm_site_validation")
    record = bound(receipt, evidence)
    source = plan["numerical_source"]
    roots = [Path(source["source_root"]), Path(plan["campaign_run_dir"]) / "data/code_snapshots" / source["source_commit"]]
    commands = {root: ([plan["python"], "-m", "pytest", "-q", str(root / "10_jax_exploration_platform/tests/atari_pixels"),
        str(root / "10_jax_exploration_platform/tests/test_atari_pixel_environment.py")] if kind == "learner" else
        [plan["python"], str(root / PREFIX / "check_rendering.py"), "--candidate", "all"]) for root in roots}
    matches = [root for root, command in commands.items() if record.get("command") == command]
    require(record.get("passed") is True and record.get("exit_code") == 0 and record.get("source_commit") == site.SOURCE_COMMIT
        and len(matches) == 1, "historical admission did not pass the exact complete original gate")
    # Verify each executed snapshot byte against the original committed source.
    selected = matches[0]
    for relative in source["files"]:
        original = git_file(source["source_root"], source["source_commit"], relative, evidence)
        require(evidence.file(selected / relative) == original, "historical gate snapshot source differs from its committed bytes")
    hardware = record["hardware"]
    runtime = hardware["runtime_environment"]
    require(runtime.get("JAX_ENABLE_X64") == "true" and runtime.get("JAX_DEFAULT_MATMUL_PRECISION") == "highest"
        and shlex.split(runtime.get("XLA_FLAGS", "")) == [site.RUNTIME["XLA_FLAGS"]], "historical gate changed its actual precision/runtime")
    require("H100" in hardware["gpu_query"] if context == "h100" else
        hardware.get("hostname") == "jaguar03" and "A4500" in hardware["gpu_query"], "historical gate used another hardware context")
    return {**receipt, "command": record["command"], "hardware": hardware}


def admission_archive(directory, plan, evidence):
    """Read only the four captured admission copies while retaining original paths as identities."""
    site = load("h100_nvl_slurm_site_validation")
    admission, archive = plan["admission"], plan["admission_archive"]
    originals = admission["files"]
    require(archive.get("schema_version") == 1 and archive.get("kind") == "original_source_admission_bytes"
        and len(originals) == 4 and len({row["path"] for row in originals}) == 4 and len(archive.get("files", [])) == 4,
        "historical site needs exactly four ordered admission copies")
    copies = {}
    for index, (original, row) in enumerate(zip(originals, archive["files"])):
        expected = Path(directory) / "admission_files" / ("record-" + str(index + 1) + "_" + Path(original["path"]).name)
        require(set(row) == {"original", "captured"} and row["original"] == original
            and row["captured"] == {"path": str(expected), "sha256": original["sha256"]}, "historical admission copy changed its identity/order/path")
        copies[original["path"]] = bound(row["captured"], evidence)
    run = Path(plan["campaign_run_dir"])
    ready = copies[str(run / "SOURCE_AND_GATES_READY.json")]
    require(ready.get("authorized") is True and ready.get("source_commit") == site.SOURCE_COMMIT
        and ready.get("source_root") == plan["numerical_source"]["source_root"]
        and set(ready.get("contexts", [])) == {"h100", "jaguar03_rtx_a4500"}, "captured source admission did not authorize this original source")
    expected_paths = [str(run / "SOURCE_AND_GATES_READY.json"), ready["reference_configuration_path"],
        ready["gate_records"]["h100"], ready["gate_records"]["jaguar03_rtx_a4500"]]
    require([row["path"] for row in originals] == expected_paths and admission.get("source_root") == ready["source_root"]
        and admission.get("source_commit") == ready["source_commit"], "captured admission roles or source identity changed")
    expected = configuration(plan["numerical_source"], evidence)
    require(plan["configuration"] == expected and {**copies[ready["reference_configuration_path"]], "n_copies": 1} == expected["learner"],
        "historical full configuration differs from frozen defaults or captured admission")
    require(admission["reference_configuration"] == originals[1], "historical reference configuration receipt changed")
    for context in ("h100", "jaguar03_rtx_a4500"):
        gate = copies[ready["gate_records"][context]]
        require(gate.get("passed") is True and gate.get("source_commit") == site.SOURCE_COMMIT
            and gate.get("hardware_class") == context, "captured hardware source gate did not pass")
        checked = {kind: check_record(gate["checks"][kind], context, kind, plan, evidence) for kind in ("learner", "renderer")}
        if context == "h100":
            require(checked == admission["original_h100_checks"], "historical H100 gate receipts changed")
    return admission


def image(path, expected, evidence):
    """Stream an explicitly mapped historical image and register its unchanged logical SHA256."""
    require(Path(path).is_absolute() and re.fullmatch(r"[0-9a-f]{64}", expected), "historical images need absolute paths and SHA256")
    physical = evidence.physical_file(path)
    hasher = hashlib.sha256()
    with physical.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    key = evidence.path_identity(path)
    row = {"path": key, "sha256": expected}
    require(hasher.hexdigest() == expected and (key not in evidence.receipts or evidence.receipts[key] == row),
        "historical library/import bytes changed: " + str(path))
    evidence.receipts[key] = row


def backend(installed, loaded, reference, evidence):
    """Verify captured packages, pinned JAXtari source and actual mapped library bytes without reader imports."""
    site = load("h100_nvl_slurm_site_validation")
    post = load("post_comparison_validation")
    require(set(installed) == {"packages", "jaxtari_source", "gpu_libraries"}
        and set(installed["packages"]) == set(site.PACKAGE_NAMES)
        and all(isinstance(value, str) and value for value in installed["packages"].values()), "historical backend lacks its full package map")
    python = installed["jaxtari_source"]
    require(python.get("source_commit") == JAXTARI_COMMIT and python.get("version") == installed["packages"]["jaxtari"]
        and python.get("tree_digest") == post.canonical_digest(python.get("python_files")) and python.get("python_files"),
        "historical JAXtari version/tree identity changed")
    root = evidence.git_root(python["source_root"], JAXTARI_COMMIT)
    entries = subprocess.check_output(["git", "-C", str(root), "ls-tree", "-rz", JAXTARI_COMMIT, "--", "src/jaxtari"])
    files = {}
    for entry in entries.split(b"\0"):
        if entry:
            header, name = entry.split(b"\t", 1)
            if name.endswith(b".py"):
                relative = name.decode()
                contents = git_file(python["source_root"], JAXTARI_COMMIT, relative, evidence)
                files[relative.removeprefix("src/")] = {"git_blob": header.decode().split()[2], "sha256": digest(contents)}
    require(files == python["python_files"], "historical JAXtari Python differs from its pinned Git tree")
    libraries = installed["gpu_libraries"]
    require(libraries and any(Path(path).name.startswith("libcudnn.so") for path in libraries)
        and any(Path(path).name.startswith("libcudart.so") for path in libraries), "historical backend omitted CUDA/cuDNN images")
    for path, row in libraries.items():
        package = "nvidia-cudnn-cu12" if Path(path).name.startswith("libcudnn") else "nvidia-cuda-runtime-cu12" if Path(path).name.startswith("libcudart") else None
        require(package and set(row) == {"package", "version", "sha256"} and row["package"] == package
            and row["version"] == installed["packages"][package], "historical GPU image package/version changed")
        image(path, row["sha256"], evidence)
    require(all(loaded.get(name) == value for name, value in reference.items()), "historical mapped API versions changed")
    mapped = {path for path in loaded["loaded_shared_libraries"] if Path(path).name.startswith(("libcudnn", "libcudart"))}
    require(any(Path(path).name.startswith("libcudnn.so") for path in mapped)
        and any(Path(path).name.startswith("libcudart.so") for path in mapped)
        and mapped <= set(libraries) and loaded.get("verified_package_libraries") == {path: libraries[path] for path in mapped}
        and Path(loaded.get("cudnn_library", "")).name.startswith("libcudnn.so")
        and Path(loaded.get("cuda_runtime_library", "")).name.startswith("libcudart.so")
        and all(loaded.get(name) in mapped for name in ("cudnn_library", "cuda_runtime_library")), "historical mapped library set/bytes changed")
    drivers = [path for path in loaded["loaded_shared_libraries"] if Path(path).name.startswith("libcuda.so")]
    require(drivers and all(path.endswith("." + site.DRIVER) for path in drivers), "historical mapped driver version changed")


def assets(plan, actual, evidence):
    """Check the exhaustive captured asset manifest and receipt without walking individual game assets."""
    post = load("post_comparison_validation")
    record = plan["asset_record"]
    require(set(record) == {"root", "manifest", "sha256"} and Path(record["root"]).is_absolute()
        and Path(record["manifest"]).is_absolute() and not Path(record["manifest"]).is_relative_to(record["root"]), "historical asset manifest is not outside its original root")
    manifest = bound({"path": record["manifest"], "sha256": record["sha256"]}, evidence)
    require(set(manifest) == {"schema_version", "source_commit", "files"} and manifest["schema_version"] == 1
        and manifest["source_commit"] == JAXTARI_COMMIT and manifest["files"], "historical asset manifest source/file set changed")
    for name, value in manifest["files"].items():
        require(isinstance(name, str) and name and str(PurePosixPath(name)) == name and name != "."
            and not PurePosixPath(name).is_absolute() and ".." not in PurePosixPath(name).parts and "\0" not in name
            and re.fullmatch(r"[0-9a-f]{64}", value), "historical asset file entry changed")
    expected = {"root": record["root"], "manifest_sha256": record["sha256"], "source_commit": JAXTARI_COMMIT,
        "file_count": len(manifest["files"]), "tree_digest": post.canonical_digest(manifest["files"])}
    require(type(actual.get("file_count")) is int and actual == expected, "historical asset receipt differs from captured manifest")


def software_reference(reference, evidence):
    """Rebuild the actual earlier formal packet through its existing strict historical validator."""
    post, site = load("post_comparison_validation"), load("h100_nvl_slurm_site_validation")
    software = reference["software"]
    raw = bound(software["result"], evidence)
    path = Path(software["result"]["path"])
    case = evidence.read(path.with_name("case.json"))
    require(raw.get("attempt") == 4 and raw.get("case_completed") is True and raw.get("correctness_passed") is True
        and raw.get("measurement_valid") is True and software["installed_backend"] == reference["shared_installed_backend"]
        and software["asset_receipt"] == reference["shared_asset_receipt"]
        and software["installed_backend"]["path"] == str(path.with_name("installed_backend.json"))
        and software["asset_receipt"]["path"] == str(path.with_name("asset_receipt.json"))
        and case["source_commit"] == site.SOURCE_COMMIT,
        "historical shared software did not come from its exact completed packet")
    proof = post.validate_cohort_case(path, case["hypothesis"], case["hardware_class"], case["configuration"]["n_copies"], site.SOURCE_COMMIT, evidence)
    installed = bound(software["installed_backend"], evidence)
    actual_assets = bound(software["asset_receipt"], evidence)
    require(software["packages"] == installed["packages"] and software["runtime_provenance"] == proof,
        "historical shared software provenance/package map changed")
    return installed, actual_assets


def reference_identity(plan, evidence):
    """Derive canonical H100 hardware/API fields from the preserved original raw observations."""
    site = load("h100_nvl_slurm_site_validation")
    reference = plan["reference"]
    directory = Path(reference["hardware"]["path"]).parent
    for key, name in (("hardware", "hardware.json"), ("loaded_libraries", "loaded_gpu_libraries.json"),
            ("diagnostic_plan", "diagnostic_plan.json"), ("dtype_record", "dtypes.json")):
        require(reference[key]["path"] == str(directory / name), "historical reference companion path changed")
    hardware = bound(reference["hardware"], evidence)
    fields = [value.strip() for value in next(csv.reader([hardware["gpu_query"]]))]
    require(fields == [site.REFERENCE_GPU, site.MODEL, str(site.MEMORY_MIB) + " MiB", site.DRIVER]
        and hardware.get("hostname") == "serval02", "historical canonical H100 model/memory/driver changed")
    runtime = hardware["runtime_environment"]
    require(all(runtime.get(name) == site.RUNTIME[name] for name in ("JAX_ENABLE_X64", "JAX_DEFAULT_MATMUL_PRECISION",
        "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION", "OMP_NUM_THREADS"))
        and shlex.split(runtime.get("XLA_FLAGS", "")) == [site.RUNTIME["XLA_FLAGS"]], "historical reference precision/runtime changed")
    diagnostic = bound(reference["diagnostic_plan"], evidence)
    require(diagnostic["core"]["source_root"] == plan["numerical_source"]["source_root"]
        and diagnostic["core"]["source_commit"] == site.SOURCE_COMMIT, "historical reference numerical source changed")
    git_record(diagnostic["core"], evidence)
    require(bound(reference["dtype_record"], evidence) == site.EXPECTED_DTYPES, "historical reference dtypes changed")
    loaded = bound(reference["loaded_libraries"], evidence)
    api = {name: loaded[name] for name in ("platform", "platform_version", "cudnnGetVersion", "cudaRuntimeGetVersion")}
    require(api["platform"] == "gpu" and isinstance(api["platform_version"], str) and api["platform_version"]
        and type(api["cudnnGetVersion"]) is int and api["cudnnGetVersion"] > 0
        and type(api["cudaRuntimeGetVersion"]) is int and api["cudaRuntimeGetVersion"] > 0
        and reference["backend"] == api and reference["model"] == site.MODEL
        and reference["memory_mib"] == site.MEMORY_MIB and reference["driver"] == site.DRIVER, "historical canonical API fields changed")
    installed, actual_assets = software_reference(reference, evidence)
    require(plan["packages"] == installed["packages"] and plan["fresh_packages_at_prepare"] == installed["packages"], "historical site package map changed")
    assets(plan, actual_assets, evidence)
    return installed, actual_assets


def assignment_identity(assignment, plan):
    """Rebuild captured physical/CPU identities without querying a live job, host or ledger."""
    site, retry = load("h100_nvl_slurm_site_validation"), load("infrastructure_retry")
    proof = assignment["allocation"]
    environment = {**proof["slurm_allocations"], "SLURM_JOB_ID": proof["job_id"], "CUDA_VISIBLE_DEVICES": proof["cuda_visible_devices_original"]}
    require(site.resolve_allocation(environment, proof["hostname"], proof["gpu_csv"], proof["gpu_xml"]) == proof
        and proof["driver"] == site.DRIVER and assignment["runtime_environment"] == site.RUNTIME, "historical allocation/runtime changed")
    cpu = assignment["cpu_allocation"]
    computed = retry.cpu_allocation({"SLURM_CPUS_PER_TASK": cpu["slurm_cpus_per_task"], "SLURM_CPU_BIND_TYPE": cpu["slurm_cpu_bind_type"],
        "SLURM_CPU_BIND_LIST": cpu["slurm_cpu_bind_list"], "SLURM_CPU_BIND": cpu["slurm_cpu_bind"]}, [proof["gpu_uuid"]], cpu["parent_available_cpus"])
    require(computed == cpu and assignment["cpu_affinity"] == cpu["gpu_cpu_affinities"][proof["gpu_uuid"]]
        and len(assignment["cpu_affinity"]) == 8, "historical eight-CPU assignment changed")
    owned = assignment["owned_job"]
    require(owned["ledger_path"] == str(Path(plan["campaign_run_dir"]) / "slurm/submitted_jobids_uva.txt")
        and owned["job_id"] == proof["job_id"] and proof["job_id"] in owned["ledger_prefix"]
        and re.search(r"(?:^|/)job_" + re.escape(proof["job_id"]) + r"(?:/|$)", owned["cgroup"], re.MULTILINE), "historical allocation lacks its captured owned UVA job/cgroup proof")
    modes = assignment["gpu_modes"]
    require(modes.get("gpu_uuid") == proof["gpu_uuid"] and modes.get("mig_mode") == "Disabled"
        and modes.get("compute_processes") == [] and modes.get("confidential_compute") == "OFF"
        and type(modes.get("memory_used_mib")) is int and 0 <= modes["memory_used_mib"] < 1024, "historical allocation did not prove an idle whole H100")
    return proof


def numerical_module_paths(paths, source):
    """Require exact recorded original numerical module paths without importing their packages."""
    site = load("h100_nvl_slurm_site_validation")
    site.validate_numerical_paths(paths, source)


def site_evidence(directory, worker_helpers, evidence):
    """Rebuild a complete historical site packet using captured admission and raw successful children."""
    site = load("h100_nvl_slurm_site_validation")
    directory = Path(directory)
    failure = str(directory / "failure.json")
    require(failure not in evidence.resolver.files if evidence.resolver is not None else not evidence.physical_file(failure).exists(),
        "historical site retains an explicitly present failure")
    manifest = evidence.read(directory / "manifest.yaml")
    require(manifest["plan"]["path"] == str(directory / "site_validation_plan.json"), "historical site plan path changed")
    plan = bound(manifest["plan"], evidence)
    require(plan.get("schema_version") == 2 and plan.get("kind") == "h100_nvl_slurm_site_validation"
        and plan.get("source_commit") == site.SOURCE_COMMIT and plan.get("runtime_environment") == site.RUNTIME
        and plan.get("job_shape") == site.JOB_SHAPE and plan.get("reference_updates") == 1
        and plan.get("submission_control") == site.SUBMISSION_CONTROL and plan.get("gate_environment_policy") == site.GATE_ENVIRONMENT_POLICY
        and plan.get("workers_permitted") is False and plan.get("campaign_claims_permitted") is False
        and plan.get("scientific_hypothesis_credit") == 0, "historical site changed its original source/runtime/no-claims protocol")
    require(manifest["job_script"]["path"] == str(directory / "job.sbatch") and manifest.get("helpers") == plan["helpers"], "historical site manifest/helper binding changed")
    bound(manifest["job_script"], evidence, document=False)
    git_record(plan["helpers"], evidence)
    require(set(plan["helpers"]["files"]) == {PREFIX + name for name in site.HELPERS}
        and plan["helpers"]["files"][PREFIX + "h100_nvl_slurm_site_validation.py"] == worker_helpers["files"][PREFIX + "h100_nvl_slurm_site_validation.py"],
        "historical site verifier/helper closure differs from reviewed worker code")
    git_record(plan["numerical_source"], evidence)
    require(plan["numerical_source"]["source_commit"] == site.SOURCE_COMMIT
        and sorted(plan["numerical_source"]["files"]) == numerical_paths(plan["numerical_source"]["source_root"], site.SOURCE_COMMIT, evidence), "historical site omitted original numerical/gate source")
    admission_archive(directory, plan, evidence)
    cleanrl = plan["cleanrl_reference"]
    original = git_file(cleanrl["source_root"], site.CLEANRL_COMMIT, "cleanrl/ppo_rnd_envpool.py", evidence)
    require(cleanrl["source_commit"] == site.CLEANRL_COMMIT and cleanrl["file"]["path"] == str(Path(cleanrl["source_root"]) / "cleanrl/ppo_rnd_envpool.py")
        and bound(cleanrl["file"], evidence, False) == original
        and cleanrl["git_blob"] == hashlib.sha1(b"blob " + str(len(original)).encode() + b"\0" + original).hexdigest(), "historical original CleanRL file changed")
    installed, actual_assets = reference_identity(plan, evidence)
    result = evidence.read(directory / "result.json")
    require(result.get("schema_version") == 1 and result.get("kind") == "h100_nvl_slurm_site_evidence" and result.get("passed") is True
        and result.get("workers_permitted") is False and result.get("campaign_claims_permitted") is False and result.get("scientific_hypothesis_credit") == 0
        and result.get("plan") == file_receipt(directory / "site_validation_plan.json", evidence)
        and result["assignment"]["path"] == str(directory / "site_assignment.json") and result["probe"]["path"] == str(directory / "probe_result.json"), "historical site did not produce its exact passed no-claims result")
    assignment = bound(result["assignment"], evidence)
    proof = assignment_identity(assignment, plan)
    identities = set()
    for kind in ("learner", "renderer", "probe"):
        rows = result["children"][kind]
        documents = {}
        for name, suffix in (("owner", "_owner.json"), ("process", "_process.json"), ("terminal", "_terminal.json")):
            require(rows[name]["path"] == str(directory / (kind + suffix)), "historical site child record path changed")
            documents[name] = bound(rows[name], evidence)
        owner, process, terminal = (documents[name] for name in ("owner", "process", "terminal"))
        identity = owner.get("pid"), owner.get("process_start_time")
        require(type(identity[0]) is int and identity[0] > 0 and isinstance(identity[1], str) and identity[1].isdigit() and identity not in identities
            and terminal.get("exit_code") == 0 and terminal["log"]["path"] == str(directory / (kind + ".log"))
            and all(process.get(key) == value and terminal.get(key) == value for key, value in owner.items())
            and owner.get("parent_pid") == assignment["parent_pid"] and owner.get("parent_start_time") == assignment["parent_start_time"]
            and owner.get("kind") == kind and owner.get("job_id") == proof["job_id"] and process.get("gpu_uuid") == proof["gpu_uuid"]
            and process.get("runtime_environment") == site.RUNTIME and process.get("cpu_affinity") == assignment["cpu_affinity"]
            and process.get("source_commit") == site.SOURCE_COMMIT and process.get("configuration") == plan["configuration"]
            and process.get("cleanrl_reference") == cleanrl and process.get("gate_environment_policy") == site.GATE_ENVIRONMENT_POLICY
            and process.get("installed_backend") == installed and process.get("asset_receipt") == actual_assets, "historical site child identity/runtime/source changed")
        identities.add(identity)
        bound(terminal["log"], evidence, False)
        expected = site.original_command(plan, kind) if kind != "probe" else None
        require(process.get("original_command") == expected, "historical site child did not execute its full original gate")
    probe = bound(result["probe"], evidence)
    process = bound(result["children"]["probe"]["process"], evidence)
    finite = probe.get("finite_state", {})
    require(probe.get("completed") is True and probe.get("kind") == "original_reference_site_probe" and probe.get("iteration") == 1
        and probe.get("dtypes") == site.EXPECTED_DTYPES and finite.get("passed") is True and type(finite.get("floating_leaves_checked")) is int
        and finite["floating_leaves_checked"] > 0 and finite.get("scope") == "complete updated PixelTrainState"
        and all(probe.get(name) == process.get(name) for name in ("pid", "process_start_time", "gpu_uuid", "source_commit"))
        and probe.get("formal_measurement") is False and probe.get("capacity_evidence") is False and probe.get("scientific_hypothesis_credit") == 0
        and probe["loaded_libraries"].get("pid") == probe["pid"], "historical site probe lacks its real dtype/finite/update/process proof")
    backend(installed, probe["loaded_libraries"], plan["reference"]["backend"], evidence)
    numerical_module_paths(probe["numerical_module_paths"], plan["numerical_source"]["source_root"])
    return plan, assignment, proof


def policy_evidence(receipt, evidence):
    """Validate reviewed exact-site permission without consulting current admission or current packages."""
    site, queue = load("h100_nvl_slurm_site_validation"), load("h100_nvl_slurm_queue")
    policy = bound(receipt, evidence)
    require(policy.get("schema_version") == 1 and policy.get("kind") == "h100_nvl_slurm_placement_policy"
        and policy.get("scope") == "exact_passed_site" and policy.get("decision") == "approved"
        and isinstance(policy.get("review_rationale"), str) and policy["review_rationale"].strip()
        and policy.get("runtime_environment") == site.RUNTIME and policy.get("source_commit") == site.SOURCE_COMMIT
        and policy.get("ordinary_attempt_limit") == 3 and policy.get("scientific_hypothesis_credit") == 0,
        "historical case lacks its explicit reviewed placement policy")
    from datetime import datetime
    require(datetime.fromisoformat(policy["reviewed_at_utc"]).utcoffset() is not None, "historical review timestamp lacks its timezone")
    require(all(type(policy.get(name)) is int and policy[name] >= 60 for name in ("case_time_budget_seconds", "shutdown_reserve_seconds")), "historical reviewed time bounds changed")
    git_record(policy["worker_helpers"], evidence)
    require(set(policy["worker_helpers"]["files"]) == {PREFIX + name for name in queue.HELPERS}, "historical worker helper closure changed")
    plan, assignment, proof = site_evidence(policy["site_directory"], policy["worker_helpers"], evidence)
    require(policy["site_result"] == file_receipt(Path(policy["site_directory"]) / "result.json", evidence)
        and policy["approved_targets"] == [{"hostname": proof["hostname"], "gpu_uuid": proof["gpu_uuid"]}]
        and policy["campaign_run_dir"] == plan["campaign_run_dir"] and policy["admission"] == plan["admission"], "historical policy expanded or changed the exact approved site")
    return policy, plan


def jaxtari_import(record, installed, module_path, evidence):
    """Verify the captured compute distribution/import receipt without inspecting the reader's packages."""
    python = installed["jaxtari_source"]
    require(set(record) == {"schema_version", "kind", "distribution_name", "distribution_version", "package_root", "module_file", "source_commit", "source_tree_digest"}
        and record["schema_version"] == 1 and record["kind"] == "verified_installed_jaxtari_import"
        and record["distribution_name"] == "jaxtari" and record["distribution_version"] == installed["packages"]["jaxtari"]
        and record["source_commit"] == JAXTARI_COMMIT and record["source_tree_digest"] == python["tree_digest"]
        and Path(record["package_root"]).is_absolute() and record["module_file"]["path"] == str(Path(record["package_root"]) / "__init__.py")
        and record["module_file"]["path"] == module_path and record["module_file"]["sha256"] == python["python_files"]["jaxtari/__init__.py"]["sha256"],
        "historical JAXtari distribution/import receipt changed")
    image(module_path, record["module_file"]["sha256"], evidence)


def module_paths(case, paths, imported, installed, evidence):
    """Validate recorded source/import paths using only original Git and captured installed bytes."""
    source = Path(case["source_root"])
    campaign = {name: str(source / PREFIX / (name + ".py")) for name in ("run_case", "controller", "ideas", "paired_benchmark")}
    require(set(campaign) <= set(paths) and "jaxtari" in paths and "exploration_platform" in paths, "historical case omitted original module identities")
    for name, expected in campaign.items():
        require(paths[name] == expected, "historical case imported another original campaign module")
    numerical = {name: path for name, path in paths.items() if name == "exploration_platform" or name.startswith("exploration_platform.atari_pixels")}
    require(set(paths) == set(campaign) | set(numerical) | {"jaxtari"}, "historical case listed unexpected modules")
    numerical_module_paths(numerical, source)
    jaxtari_import(imported, installed, paths["jaxtari"], evidence)


def failure_absence(attempt, terminal, evidence):
    """Require the parent's one explicit post-exit absence observation and reject contradictory mapped failures."""
    path = str(Path(attempt) / "failure.json")
    require(terminal.get("absent_files") == [path], "historical zero-exit receipt lacks its exact current-attempt failure absence")
    if evidence.resolver is not None:
        require(path not in evidence.resolver.files, "an explicitly mapped failure contradicts the captured absence")
    else:
        require(not evidence.physical_file(path).exists(), "the current attempt retains a failure record")


def scientific_result(result, case, evidence):
    """Require original configurations, full correctness reports and every complete original paired timing window."""
    post = load("post_comparison_validation")
    require(result.get("case_completed") is True, "historical raw scientific result is incomplete")
    environment, execution, _ = post.case_source_defaults(case, evidence)
    reference = {"learner": case["configuration"], "environment": {**environment, **case.get("reference_environment", {})},
        "execution": {**execution, **(case.get("reference_execution") or {})}}
    candidate = {scope: {**reference[scope], **case["candidate"][scope]} for scope in reference}
    require(post.configuration(result["reference"]) == post.configuration(reference)
        and post.configuration(result["candidate"]) == post.configuration(candidate)
        and post.canonical_digest(reference) != post.canonical_digest(candidate) and result.get("implementation_distinct") is True,
        "historical result changed its original reference/candidate configurations")
    if result.get("failure_stage") == "correctness":
        require(result.get("decision") == "rejected" and result.get("correctness_attempted") is True
            and result.get("correctness_passed") is False and isinstance(result.get("error"), str) and result["error"],
            "historical scientific rejection lacks its actual correctness failure")
        return
    post.correctness(result)
    require(result.get("measurement_valid") is True and result.get("contaminated") is False
        and type(case.get("rounds")) is int and case["rounds"] >= 7 and len(result.get("rounds", [])) == case["rounds"]
        and type(case.get("iterations")) is int and case["iterations"] >= 1, "historical result omitted complete valid paired windows")
    for index, pair in enumerate(result["rounds"]):
        require(pair.get("order") == (["reference", "candidate"] if index % 2 == 0 else ["candidate", "reference"]), "historical paired order changed")
        for arm in ("reference", "candidate"):
            require(pair.get(arm, {}).get("iterations") == case["iterations"], "historical paired complete-update count changed")
            post.measured_round(pair[arm], case["configuration"]["n_copies"], minimum_updates=case["iterations"])


def validate_execution(attempt, case, lease=None, *, require_completed=False, evidence=None):
    """Rebuild a strict staged/completed execution proof through exact historical evidence paths."""
    post, site = load("post_comparison_validation"), load("h100_nvl_slurm_site_validation")
    evidence = post.Evidence() if evidence is None else evidence
    attempt = Path(attempt)
    saved = evidence.read(attempt / "lease.json")
    completed = require_completed or lease is None or evidence.resolver is not None or saved.get("state") == "completed"
    lease = saved if lease is None else lease
    stable = ("token", "unit_id", "attempt", "runner_pid", "runner_start_time", "worker_pid", "worker_start_time",
        "slurm_job_id", "gpu_uuid", "cpu_affinity", "execution_kind", "placement_policy", "allocation_validation")
    require(lease.get("execution_kind") == EXECUTION_KIND and type(lease.get("attempt")) is int and 1 <= lease["attempt"] <= 3
        and saved.get("state") in {"running", "completed"} and all(saved.get(name) == lease.get(name) for name in stable)
        and (not completed or saved.get("state") == "completed" and saved.get("runner_exit_code") == 0), "historical execution lacks its matching original ordinary lease")
    run = Path(case["campaign_run_dir"])
    unit = run / "data" / case["unit_id"]
    require(attempt == unit / "attempts" / ("attempt-" + str(lease["attempt"]).zfill(4))
        and evidence.read(attempt / "case.json") == case and case["specification_digest"] == post.canonical_digest({key: value for key, value in case.items() if key != "specification_digest"})
        and case.get("source_commit") == site.SOURCE_COMMIT and case.get("hardware_class") == "h100" and case.get("maximum_attempts") == 3,
        "historical execution changed its exact immutable case/attempt/source")
    policy, reference = policy_evidence(lease["placement_policy"], evidence)
    record = evidence.read(attempt / "slurm_execution.json")
    terminal = evidence.read(attempt / "slurm_child_terminal.json")
    owner = evidence.read(attempt / "slurm_child_owner.json")
    failure_absence(attempt, terminal, evidence)
    require(record["released_lease"]["path"] == str(attempt / "slurm_released_lease.json"), "historical released-lease snapshot path changed")
    released = bound(record["released_lease"], evidence)
    require(terminal.get("exit_code") == 0 and terminal["owner"] == file_receipt(attempt / "slurm_child_owner.json", evidence)
        and terminal["log"] == file_receipt(attempt / "runner.log", evidence) and released.get("state") == "running"
        and all(released.get(name) == lease.get(name) for name in stable)
        and record.get("kind") == EXECUTION_KIND and record.get("schema_version") == 1
        and record.get("placement_policy") == lease["placement_policy"] and record.get("allocation_validation") == lease["allocation_validation"]
        and record.get("lease_token") == lease["token"] and record.get("attempt") == lease["attempt"]
        and record.get("specification_digest") == case["specification_digest"] and record.get("source_commit") == site.SOURCE_COMMIT
        and record.get("pid") == lease["runner_pid"] and record.get("process_start_time") == lease["runner_start_time"]
        and owner.get("pid") == record["pid"] and owner.get("process_start_time") == record["process_start_time"]
        and owner.get("parent_pid") == lease["worker_pid"] and owner.get("parent_start_time") == lease["worker_start_time"]
        and owner.get("job_id") == lease["slurm_job_id"] and record.get("gpu_uuid") == lease["gpu_uuid"] and record.get("job_id") == lease["slurm_job_id"]
        and record.get("runtime_environment") == site.RUNTIME and record.get("cpu_affinity") == lease["cpu_affinity"], "historical case lacks its exact successful child/runtime/library identities")
    require(record["hardware"]["path"] == str(attempt / "hardware.json"), "historical case hardware path changed")
    hardware = bound(record["hardware"], evidence)
    fields = [value.strip() for value in next(csv.reader([hardware["gpu_query"]]))]
    require(fields == [record["gpu_uuid"], site.MODEL, str(site.MEMORY_MIB) + " MiB", site.DRIVER]
        and hardware.get("pid") == record["pid"] and hardware.get("process_start_time") == record["process_start_time"]
        and hardware.get("gpu_uuid") == record["gpu_uuid"] and hardware.get("slurm_job_id") == record["job_id"]
        and hardware.get("source_root") == case["source_root"] and hardware.get("source_commit") == case["source_commit"], "historical case hardware/source does not match its actual child")
    require(record["allocation_validation"]["path"].endswith("/result.json"), "historical allocation result path changed")
    bound(record["allocation_validation"], evidence)
    allocation_plan, assignment, scope = site_evidence(Path(record["allocation_validation"]["path"]).parent, policy["worker_helpers"], evidence)
    require(all(reference[name] == allocation_plan[name] for name in ("campaign_run_dir", "numerical_source", "reference", "packages", "asset_record", "jaxtari_source_root", "python", "cleanrl_reference", "configuration", "runtime_environment", "admission"))
        and {"hostname": scope["hostname"], "gpu_uuid": scope["gpu_uuid"]} in policy["approved_targets"]
        and scope["job_id"] == record["job_id"] and scope["gpu_uuid"] == record["gpu_uuid"]
        and hardware.get("hostname", "").split(".")[0] == scope["hostname"] and record["cpu_affinity"] == assignment["cpu_affinity"]
        and case["source_root"] == allocation_plan["numerical_source"]["source_root"], "historical case used another approved allocation/source")
    expected_backend = bound(reference["reference"]["shared_installed_backend"], evidence)
    expected_assets = bound(reference["reference"]["shared_asset_receipt"], evidence)
    require(record["installed_backend"] == expected_backend and record["asset_receipt"] == expected_assets
        and record["loaded_libraries"].get("pid") == record["pid"], "historical case changed its captured installed backend/assets")
    backend(record["installed_backend"], record["loaded_libraries"], reference["reference"]["backend"], evidence)
    module_paths(case, record["module_paths"], record["jaxtari_import"], record["installed_backend"], evidence)
    files = record["terminal_files"]
    require(set(files) == set(TERMINAL_NAMES), "historical case omitted one of four original terminal writes")
    contents = {}
    for name in TERMINAL_NAMES:
        require(files[name]["path"] == str(attempt / "slurm_terminal_staging" / (name + ".json")), "historical staged terminal path changed")
        contents[name] = bound(files[name], evidence, False)
    result = json.loads(contents["attempt_result"])
    require(contents["attempt_result"] == contents["unit_result"] and all(result.get(name) == value for name, value in
        {"specification_digest": case["specification_digest"], "attempt": lease["attempt"], "unit_id": case["unit_id"], "source_commit": site.SOURCE_COMMIT,
         "hardware_class": "h100", "idea": case["hypothesis"], "copies": case["configuration"]["n_copies"]}.items()), "historical raw result changed its immutable case identity")
    scientific_result(result, case, evidence)
    require(result.get("failure_stage") == "correctness" or all(
        pair[arm]["memory"]["gpu_uuid"] == record["gpu_uuid"]
        for pair in result["rounds"] for arm in ("reference", "candidate")),
        "historical timed rows identify another physical GPU")
    for name in ("attempt_progress", "unit_progress"):
        progress = json.loads(contents[name])
        require(progress.get("state") == "completed" and progress.get("case_completed") is True and progress.get("attempt") == lease["attempt"]
            and progress.get("unit_id") == case["unit_id"] and progress.get("pid") == record["pid"] and progress.get("decision") == result["decision"], "historical completed progress changed")
    if completed:
        require(evidence.read(run / "queue/leases" / (case["unit_id"] + ".json")) == saved, "historical completed lease copies differ")
        intent = evidence.read(attempt / "slurm_publication_intent.json")
        require(intent == {"kind": "h100_nvl_slurm_terminal_publication", "token": lease["token"], "attempt": lease["attempt"], "unit_id": case["unit_id"],
            "execution": file_receipt(attempt / "slurm_execution.json", evidence), "child_terminal": file_receipt(attempt / "slurm_child_terminal.json", evidence),
            "terminal_files": {name: files[name]["sha256"] for name in TERMINAL_NAMES}}, "historical completed publication intent changed")
        paths = {"attempt_result": attempt / "result.json", "unit_result": unit / "result.json", "attempt_progress": attempt / "progress.json", "unit_progress": unit / "progress.json"}
        require(all(evidence.file(path) == contents[name] for name, path in paths.items()), "historical canonical publication differs from staged original bytes")
    return record, result
