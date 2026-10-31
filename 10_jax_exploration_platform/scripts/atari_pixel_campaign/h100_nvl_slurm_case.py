"""Execute one unchanged original case and withhold terminal publication until same-process proof exists."""

import argparse
from contextlib import contextmanager
from importlib import metadata
import importlib
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback


PREFIX = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"
FORBIDDEN = ("controller", "ideas", "run_case", "paired_benchmark", "exploration_platform", "jaxtari", "jax", "jaxlib", "numpy", "torch")
PROGRESS_STATES = {"validating_source_and_configuration", "initializing", "correctness_passed", "running", "failed", "completed"}


def load_queue():
    """Use the adjacent committed metadata helper without importing a current numerical module name."""
    path = Path(__file__).with_name("h100_nvl_slurm_queue.py")
    specification = importlib.util.spec_from_file_location("h100_case_queue", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def require_clean_imports():
    """Refuse a process that already imported a current campaign module or any numerical package."""
    if any(name == item or name.startswith(item + ".") for name in sys.modules for item in FORBIDDEN):
        raise ValueError("the original numerical case requires a fresh process without shadowing imports")


def verify_module_paths(case, paths):
    """Verify original campaign/model modules and the actually imported installed JAXtari package."""
    source = Path(case["source_root"])
    campaign = {name: str(source / PREFIX / (name + ".py")) for name in ("run_case", "controller", "ideas", "paired_benchmark")}
    required = {*campaign, "exploration_platform", "exploration_platform.atari_pixels.config",
        "exploration_platform.atari_pixels.environment", "exploration_platform.atari_pixels.training",
        "exploration_platform.atari_pixels.networks", "exploration_platform.atari_pixels.state", "jaxtari"}
    if not required <= set(paths):
        raise ValueError("the original case omitted its actual campaign/model/JAXtari module paths")
    for name, path in paths.items():
        if name in campaign:
            expected = campaign[name]
        elif name == "jaxtari":
            expected = str(Path(metadata.distribution("jaxtari").locate_file("jaxtari/__init__.py")).resolve())
        elif name == "exploration_platform" or name.startswith("exploration_platform.atari_pixels"):
            relative = Path(*name.split("."))
            expected = source / "10_jax_exploration_platform/src" / relative
            expected = str(expected / "__init__.py" if name in ("exploration_platform", "exploration_platform.atari_pixels") else expected.with_suffix(".py"))
        else:
            raise ValueError("the case proof contains an unexpected numerical module name")
        if path != expected:
            raise ValueError("the case imported a different original module: " + name)
    return paths


def load_original(case):
    """Load unchanged D28 run_case/controller/ideas only after all metadata checks preserve a clean namespace."""
    require_clean_imports()
    source = Path(case["source_root"])
    sys.path.insert(0, str(source / PREFIX))
    sys.path.insert(0, str(source / "10_jax_exploration_platform/src"))
    runner = importlib.import_module("run_case")
    for name in ("run_case", "controller", "ideas"):
        if Path(sys.modules[name].__file__).resolve() != source / PREFIX / (name + ".py"):
            raise ValueError("the numerical child imported a different original " + name)
    return runner


def jaxtari_import_receipt(installed):
    """Capture the actual compute distribution/import identity for later package-independent validation."""
    distribution = metadata.distribution("jaxtari")
    root = Path(distribution.locate_file("jaxtari")).resolve(strict=True)
    actual = Path(sys.modules["jaxtari"].__file__).resolve(strict=True)
    contents = actual.read_bytes()
    python = installed["jaxtari_source"]
    expected = python["python_files"]["jaxtari/__init__.py"]["sha256"]
    if (actual != root / "__init__.py" or distribution.version != installed["packages"]["jaxtari"]
            or hashlib.sha256(contents).hexdigest() != expected):
        raise ValueError("the actual JAXtari import differs from its verified installed distribution/Git bytes")
    return {"schema_version": 1, "kind": "verified_installed_jaxtari_import", "distribution_name": "jaxtari",
        "distribution_version": distribution.version, "package_root": str(root),
        "module_file": {"path": str(actual), "sha256": expected}, "source_commit": python["source_commit"],
        "source_tree_digest": python["tree_digest"]}


@contextmanager
def terminal_writer(runner, unit, attempt, queue):
    """Redirect only the four approved original terminal writes; publish all intermediate and failure records normally."""
    original = runner.atomic_json
    destinations = queue.terminal_paths(unit, attempt)
    paths = {path: name for name, path in destinations.items()}
    staged = attempt / "slurm_terminal_staging"
    staged.mkdir(exist_ok=False)
    retained = {}

    def write(path, value):
        """Keep exact original serialization while refusing any unknown terminal path, state or duplicate write."""
        path = Path(path).absolute()
        is_progress = path.name == "progress.json"
        if is_progress and value.get("state") not in PROGRESS_STATES:
            raise ValueError("the original case used an unknown progress state")
        terminal = path.name == "result.json" or value.get("state") == "completed" or value.get("case_completed") is True
        if terminal:
            name = paths.get(path)
            if name is None or value.get("case_completed") is not True:
                raise ValueError("the original case attempted an unknown terminal write")
            if is_progress and value.get("state") != "completed":
                raise ValueError("a terminal progress write requires the original completed state")
            if name in retained:
                raise ValueError("the original case wrote the same terminal file twice")
            destination = staged / (name + ".json")
            original(destination, value)
            retained[name] = queue.load("h100_nvl_slurm_site_validation").receipt(destination)
            return
        original(path, value)

    runner.atomic_json = write
    try:
        yield retained
    finally:
        runner.atomic_json = original


def execute(policy_path, allocation_directory, case_path, attempt_number, token):
    """Validate the fresh allocation, call the exact original benchmark, then record actual mapped libraries in that process."""
    require_clean_imports()
    queue = load_queue()
    policy, _, _ = queue.validate_policy(policy_path, admission=False)
    site, directory, plan, assignment = queue.allocation_validation(policy, allocation_directory, admission=False)
    site.require_gate_controls(os.environ, plan["cleanrl_reference"])
    if str(Path(sys.executable).absolute()) != plan["python"]:
        raise ValueError("the numerical child changed its validated interpreter")
    case = site.read(case_path)
    run = Path(policy["campaign_run_dir"])
    if (Path(case_path).resolve() != run / "queue/specifications" / (case["unit_id"] + ".json")
            or case["campaign_run_dir"] != str(run) or case["hardware_class"] != "h100"
            or case["source_commit"] != site.SOURCE_COMMIT or case["maximum_attempts"] != 3
            or not 1 <= attempt_number <= 3 or case["source_root"] != plan["numerical_source"]["source_root"]):
        raise ValueError("the Slurm child must execute an unchanged original ordinary H100 case")
    installed, assets = queue.fresh_software(site, plan)
    queue.current_allocation(policy, site, assignment)
    require_clean_imports()
    runner = load_original(case)
    unit, attempt = runner.attempt_paths(run, case["unit_id"], attempt_number)
    lease = runner.wait_for_recorded_release(run, case, attempt_number, token)
    owner = site.read(attempt / "slurm_child_owner.json")
    if (lease.get("execution_kind") != queue.EXECUTION_KIND or lease["placement_policy"] != site.receipt(policy_path)
            or lease["allocation_validation"] != site.receipt(directory / "result.json")
            or owner["pid"] != os.getpid() or owner["process_start_time"] != site.process_start_time(os.getpid())
            or owner["parent_pid"] != lease["worker_pid"] or owner["parent_start_time"] != lease["worker_start_time"]
            or not runner.process_matches(owner["parent_pid"], owner["parent_start_time"])
            or lease["slurm_job_id"] != assignment["allocation"]["job_id"]
            or lease["gpu_uuid"] != assignment["allocation"]["gpu_uuid"]
            or sorted(os.sched_getaffinity(0)) != lease["cpu_affinity"]
            or lease["cpu_affinity"] != assignment["cpu_affinity"]
            or os.environ.get("CUDA_VISIBLE_DEVICES") != lease["gpu_uuid"]):
        raise ValueError("the numerical child did not inherit its exact recorded parent/allocation/eight CPUs")
    site.require_selected_card(lease["gpu_uuid"])
    site.write_once(attempt / "slurm_released_lease.json", lease)
    try:
        with terminal_writer(runner, unit, attempt, queue) as terminal_files:
            result = runner.run_case(run, case, attempt_number, token)
        if set(terminal_files) != set(queue.TERMINAL_NAMES):
            raise ValueError("the original runner did not produce all four staged terminal records")
        # Library inspection follows original training in the same process and adds no timed work.
        import jax
        devices = jax.devices()
        if len(devices) != 1 or devices[0].platform != "gpu" or devices[0].device_kind != site.MODEL:
            raise ValueError("the original case did not execute on exactly the approved H100 NVL")
        training = site.load("train_optimized")
        loaded = training.actual_gpu_library_receipt(devices[0].client, installed["gpu_libraries"])
        site.validate_loaded(loaded, plan, installed)
        paths = {name: str(Path(module.__file__).resolve()) for name, module in list(sys.modules.items())
            if name in ("run_case", "controller", "ideas", "paired_benchmark", "exploration_platform", "jaxtari")
            or name.startswith("exploration_platform.atari_pixels")}
        verify_module_paths(case, paths)
        record = {"schema_version": 1, "kind": queue.EXECUTION_KIND, "placement_policy": site.receipt(policy_path),
            "allocation_validation": site.receipt(directory / "result.json"), "lease_token": token,
            "released_lease": site.receipt(attempt / "slurm_released_lease.json"), "attempt": attempt_number,
            "hardware": site.receipt(attempt / "hardware.json"),
            "specification_digest": case["specification_digest"], "source_commit": site.SOURCE_COMMIT,
            "pid": os.getpid(), "process_start_time": site.process_start_time(os.getpid()), "job_id": lease["slurm_job_id"],
            "gpu_uuid": lease["gpu_uuid"], "runtime_environment": site.require_runtime(os.environ),
            "cpu_affinity": sorted(os.sched_getaffinity(0)), "installed_backend": installed, "asset_receipt": assets,
            "module_paths": paths, "jaxtari_import": jaxtari_import_receipt(installed),
            "loaded_libraries": loaded, "terminal_files": terminal_files}
        site.write_once(attempt / "slurm_execution.json", record)
        return result
    except BaseException as error:
        # Preserve the original failure convention while withholding every staged completed result.
        failure = {"case_completed": False, "failure_stage": "runner_exception", "retryable": True,
            "exception_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(),
            "attempt": attempt_number, "pid": os.getpid(), "time": runner.utc_now(), "source_commit": case["source_commit"]}
        runner.atomic_json(attempt / "failure.json", failure)
        runner.atomic_json(unit / "failure.json", failure)
        runner.atomic_json(unit / "progress.json", {**failure, "state": "failed"})
        raise


def main():
    """Execute only an already claimed exact original case; never claim, submit or create placement permission."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("policy", "allocation-validation", "case"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--lease-token", required=True)
    arguments = parser.parse_args()
    result = execute(arguments.policy, arguments.allocation_validation, arguments.case, arguments.attempt, arguments.lease_token)
    print(json.dumps({"case_completed": True, "unit_id": result["unit_id"], "decision": result["decision"]}), flush=True)


if __name__ == "__main__":
    main()
