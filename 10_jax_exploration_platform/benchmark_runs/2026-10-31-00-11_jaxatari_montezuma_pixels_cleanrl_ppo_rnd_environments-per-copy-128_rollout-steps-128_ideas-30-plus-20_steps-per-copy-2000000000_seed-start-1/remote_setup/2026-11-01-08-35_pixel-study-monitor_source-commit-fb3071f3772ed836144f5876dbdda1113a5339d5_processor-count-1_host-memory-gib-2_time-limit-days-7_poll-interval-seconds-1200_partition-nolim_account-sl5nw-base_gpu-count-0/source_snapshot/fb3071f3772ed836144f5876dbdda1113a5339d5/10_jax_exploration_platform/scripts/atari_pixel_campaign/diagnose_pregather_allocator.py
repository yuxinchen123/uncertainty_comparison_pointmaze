"""Diagnose the original two-copy pregather memory failure with CUDA's async allocator.

The numerical source, all original correctness checks, seeds and seven paired
rounds remain unchanged. Only allocator selection and separate memory callbacks
differ. This program never claims a case, submits a job or grants idea credit.
"""

import argparse
import ast
from contextlib import contextmanager
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback


SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import measure_selected as measurement

training, diagnostic, validation = measurement.training, measurement.diagnostic, measurement.validation
HELPERS = ("diagnose_pregather_allocator.py", *measurement.HELPERS)
SOURCE_COMMIT = "d28e6735e79788f618acb0ebfde8609db1f776af"
CASE_SHA256 = "d661816f161c9a0b756d9cb3d376c1553e662292204dfc881095aacf4613037e"
ALLOCATOR = {"XLA_PYTHON_CLIENT_ALLOCATOR": "cuda_async", "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.85",
             "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
THREADS = {"OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def allocator_source_receipt(path):
    """Verify installed allocator support by parsing source without importing jaxlib or NumPy."""
    path = Path(path).resolve(strict=True)
    contents = path.read_bytes()
    tree = ast.parse(contents)
    function = next((node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == "generate_pjrt_gpu_plugin_options"), None)
    if function is None:
        raise ValueError("installed jaxlib has no explicit GPU allocator parser")
    allowed = next((ast.literal_eval(node.value) for node in ast.walk(function) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "allowed_allocators" for target in node.targets)), ())
    reads = {node.args[0].value for node in ast.walk(function) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "getenv" and node.args
             and isinstance(node.args[0], ast.Constant)}
    if "cuda_async" not in allowed or not {"XLA_PYTHON_CLIENT_ALLOCATOR", "XLA_PYTHON_CLIENT_MEM_FRACTION",
                                           "XLA_PYTHON_CLIENT_PREALLOCATE"} <= reads:
        raise ValueError("installed jaxlib does not explicitly support the requested async allocator and original controls")
    return {"path": str(path), "sha256": hashlib.sha256(contents).hexdigest(),
            "function": function.name, "allowed_allocators": list(allowed)}


def original_inputs(case_path, reference_path):
    """Use the exact held fourth-attempt pregather case and a completed original full reference."""
    case, case_receipt = measurement.read_document(case_path)
    reference, reference_receipt = measurement.read_document(reference_path)
    failure_path = Path(case_path).parent / "failure.json"
    failure, failure_receipt = measurement.read_document(failure_path)
    if (case_receipt["sha256"] != CASE_SHA256 or case.get("source_commit") != SOURCE_COMMIT
            or case.get("hypothesis_position") != 23 or case.get("hypothesis") != "gather_epoch_minibatches_in_advance"
            or case.get("candidate") != {"learner": {"pre_gather_minibatches": True}, "environment": {}, "execution": {}}
            or (case.get("rounds"), case.get("iterations"), case.get("seed_start")) != (7, 1, 1)
            or case.get("reference_environment") is not None or case.get("reference_execution") is not None
            or Path(case_path).parent.name != "attempt-0004" or failure.get("attempt") != 4
            or failure.get("source_commit") != SOURCE_COMMIT or failure.get("case_completed") is not False
            or "jit_update" not in failure.get("error", "") or "Out of memory" not in failure.get("error", "")):
        raise ValueError("use the exact original pregather case and its held fourth-attempt update allocation failure")
    configuration = validation.configuration(reference["reference"])
    if (reference.get("source_commit") != SOURCE_COMMIT or reference.get("case_completed") is not True
            or reference.get("hardware_class") != "jaguar03_rtx_a4500" or reference.get("copies") != 2
            or configuration["learner"] != case["configuration"] or configuration["learner"]["n_copies"] != 2):
        raise ValueError("the completed original reference differs from the held two-copy case")
    return case, configuration, {"case": case_receipt, "original_failure": failure_receipt, "reference": reference_receipt}


def helper_receipt():
    """Keep the operational diagnostic's committed source distinct from the frozen numerical source."""
    root = SCRIPTS.parents[2]
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    return training.source_receipt(root, commit, [str((SCRIPTS / name).relative_to(root)) for name in HELPERS])


def prepare(arguments):
    """Write reviewable immutable inputs and launch records using standard-library metadata only."""
    case, configuration, inputs = original_inputs(arguments.case, arguments.reference_record)
    root, run = arguments.source_root.resolve(strict=True), arguments.run_dir.resolve(strict=True)
    if (str(root) != case["source_root"] or str(run) != case["campaign_run_dir"]
            or not arguments.gpu_uuid.startswith("GPU-") or Path(arguments.unit_id).name != arguments.unit_id
            or arguments.unit_id in {"", ".", ".."}):
        raise ValueError("record the original source/campaign, one physical UUID and a descriptive diagnostic unit")
    measurement.validate_complete_fields(root, configuration)
    runtime, inputs["runtime_environment"] = measurement.read_document(arguments.runtime_environment)
    runtime = validation.runtime_environment(runtime)
    assets, inputs["asset_record"] = measurement.read_document(arguments.asset_record)
    if set(assets) != {"root", "manifest", "sha256"}:
        raise ValueError("use the explicit unchanged original game-asset manifest record")
    paths = ["10_jax_exploration_platform/scripts/atari_pixel_campaign/" + name
             for name in ("controller.py", "ideas.py", "paired_benchmark.py")]
    helpers = helper_receipt()
    plan = {"schema_version": 1, "kind": "pregather_async_allocator_diagnosis", "hypothesis_count_contribution": 0,
        "eligible_for_formal_campaign_admission": False, "source_root": str(root), "source_commit": SOURCE_COMMIT,
        "core": diagnostic.core_identity(root, SOURCE_COMMIT), "paired_source": training.source_receipt(root, SOURCE_COMMIT, paths),
        "helpers": helpers, "configuration": configuration, "case": case, "inputs": inputs,
        "original_case_bytes_sha256": inputs["case"]["sha256"], "allocator": ALLOCATOR, "thread_environment": THREADS,
        "runtime_environment": runtime, "compilation_cache_policy": "disabled", "memory_preparation": None,
        "asset_record": assets, "jaxtari_source_root": str(arguments.jaxtari_source_root.resolve(strict=True)),
        "allocator_parser": allocator_source_receipt(arguments.allocator_parser_source),
        "hardware_class": "jaguar03_rtx_a4500", "gpu_uuid": arguments.gpu_uuid, "copy_seeds": [1, 2],
        "rounds": 7, "iterations": 1, "latency_calls": 3, "campaign_run_dir": str(run), "unit_id": arguments.unit_id,
        "python": str(arguments.python.absolute())}
    directory = run / "data" / arguments.unit_id
    command = [str(arguments.python), str(directory / "code/diagnose_pregather_allocator.py"), "--prepared-unit", str(directory)]
    artifacts = {directory / "measurement_plan.json": measurement.json_bytes(plan),
        directory / "manifest.yaml": measurement.json_bytes(plan), directory / "config_resolved.yaml": measurement.json_bytes(plan),
        directory / "inputs/original_case.json": Path(arguments.case).read_bytes(),
        directory / "command.txt": (shlex.join(command) + "\n").encode(),
        directory / "experiment_background.md": (
            "## Purpose\n\nDiagnose the held pregather update allocation failure in a separate fresh process. "
            "This diagnostic contributes zero ideas and cannot complete a campaign case.\n\n"
            "## Key hyperparameters\n\nThe original case retains two policies, 128 environments and 128 rollout steps per policy, "
            "four epochs, four minibatches, fifty normalization warmup rollouts, seeds one and two, and seven alternating "
            "pairs with one complete update per arm. Neural arithmetic is float32/highest, statistics are float64 and "
            "deterministic XLA operations remain enabled.\n\n"
            "## Code and config changes\n\nSelect the installed jaxlib cuda_async allocator while retaining allocator fraction "
            "0.85 and disabled preallocation. Do not prepare a large GPU array. Original fixed-input and whole-iteration "
            "checks run unchanged, with separate memory callbacks around those calls. Missing allocator statistics are "
            "recorded as unavailable. The original paired result and memory journal remain separate.\n\n"
            f"## Git state\n\nNumerical source: `{SOURCE_COMMIT}`. Helper source: `{helpers['source_commit']}`. "
            "Exact input, source, helper and installed allocator-parser hashes are in measurement_plan.json.\n").encode()}
    artifacts.update({directory / "code" / name: (SCRIPTS / name).read_bytes() for name in HELPERS})
    for path, contents in artifacts.items():
        if path.exists() and path.read_bytes() != contents:
            raise ValueError("an earlier pregather diagnostic artifact is immutable: " + str(path))
    for path, contents in artifacts.items():
        measurement.write_once(path, contents)
    measurement.write_once(directory / "measurement_prepared.json", measurement.json_bytes({
        "plan_sha256": hashlib.sha256(measurement.json_bytes(plan)).hexdigest()}))
    return directory


@contextmanager
def observe_checks(paired, publish):
    """Call each original correctness function unchanged with metadata observations around its boundary."""
    originals = {name: getattr(paired, name) for name in ("check_fixed_inputs", "correctness_checks")}
    def observed(name, original):
        """Retain arguments, return object and original exception without retaining numerical intermediates."""
        def call(*arguments, **keywords):
            """Record actual memory before and after the unchanged original function call."""
            publish(name + "_entered")
            try:
                result = original(*arguments, **keywords)
            except BaseException as error:
                try:
                    publish(name + "_failed")
                except Exception as observation_error:
                    raise error from observation_error
                raise
            publish(name + "_completed")
            return result
        return call
    for name, original in originals.items():
        setattr(paired, name, observed(name, original))
    try:
        yield
    finally:
        for name, original in originals.items():
            setattr(paired, name, original)


def memory_snapshot(device, observation):
    """Record actual device/counter observations or explicit absence without inventing BFC statistics."""
    record = {"time_seconds": time.time()}
    try:
        record["device_memory"] = observation.sample()
    except Exception as error:
        record["device_memory_unavailable"] = repr(error)
    try:
        values = device.memory_stats()
        record["allocator"] = None if values is None else {name: int(value) for name, value in values.items()}
    except Exception as error:
        record["allocator"] = None
        record["allocator_statistics_unavailable"] = repr(error)
    return record


def load_original_modules(plan):
    """Load the exact frozen controller, ideas and paired functions before numerical imports in a fresh child."""
    forbidden = ("controller", "ideas", "paired_benchmark", "exploration_platform", "jax", "numpy", "jaxtari")
    if any(name == prefix or name.startswith(prefix + ".") for name in sys.modules for prefix in forbidden):
        raise ValueError("the diagnostic needs a fresh process before current or numerical modules are imported")
    root = Path(plan["source_root"]) / "10_jax_exploration_platform"
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "scripts/atari_pixel_campaign"))
    modules = {name: importlib.import_module(name) for name in ("controller", "ideas", "paired_benchmark")}
    for name, module in modules.items():
        if Path(module.__file__).resolve() != root / "scripts/atari_pixel_campaign" / (name + ".py"):
            raise ValueError("the diagnostic imported another version of " + name)
    return modules["paired_benchmark"]


def compute(directory, plan, hardware, libraries):
    """Run the original full paired benchmark using one explicitly selected allocator and separate observations."""
    paired = load_original_modules(plan)
    import jax
    import jaxlib.xla_client as xla_client
    import jaxtari
    from exploration_platform.atari_pixels.config import PixelPPOConfig
    parsed = xla_client.generate_pjrt_gpu_plugin_options()
    if (Path(xla_client.__file__).resolve() != Path(plan["allocator_parser"]["path"])
            or parsed.get("allocator") != "cuda_async" or parsed.get("memory_fraction") != .85 or parsed.get("preallocate") is not False):
        raise ValueError("actual installed jaxlib parsed another allocator configuration")
    diagnostic.atomic_json(directory / "actual_allocator.json", {"source": plan["allocator_parser"], "parsed_options": parsed,
        "environment": {name: os.environ[name] for name in ALLOCATOR}, "memory_preparation": None})
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu" or "A4500" not in devices[0].device_kind:
        raise ValueError("the diagnostic must hold exactly its assigned RTX A4500")
    measurement.verify_imported_sources(Path(plan["source_root"]) / "10_jax_exploration_platform", paired)
    observation = measurement.MemoryObservation(devices[0], plan["gpu_uuid"])
    def publish(phase):
        """Flush real memory observations separately from the original numerical reports."""
        record = {"phase": phase, **memory_snapshot(devices[0], observation), "pid": os.getpid()}
        with (directory / "memory_snapshots.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    def progress(value):
        """Preserve original progress and observe only initialization/correctness stages outside timing."""
        diagnostic.record_progress(directory, value["state"], **{key: item for key, item in value.items() if key != "state"})
        if value["state"] in {"initializing", "correctness_passed"}:
            publish(value["state"] + ("_" + value["variant"] if "variant" in value else ""))
    started = time.time()
    publish("before_original_benchmark")
    try:
        with observe_checks(paired, publish):
            result = paired.run_paired_benchmark(PixelPPOConfig(**plan["configuration"]["learner"]),
                plan["case"]["candidate"]["learner"], plan["case"]["candidate"]["environment"], plan["case"]["candidate"]["execution"],
                rounds=7, iterations=1, seed_start=1, latency_calls=3, progress=progress,
                reference_environment=plan["configuration"]["environment"], reference_execution=plan["configuration"]["execution"])
    except BaseException as error:
        try:
            publish("original_benchmark_failed")
        except Exception as observation_error:
            raise error from observation_error
        raise
    publish("original_benchmark_completed")
    diagnostic.atomic_json(directory / "original_paired_result.json", result)
    diagnostic.atomic_json(directory / "result.json", {"completed": True, "kind": plan["kind"], "hypothesis_count_contribution": 0,
        "eligible_for_formal_campaign_admission": False, "source_commit": SOURCE_COMMIT, "helpers": plan["helpers"],
        "configuration": plan["configuration"], "runtime_environment": plan["runtime_environment"], "allocator": plan["allocator"],
        "hardware": hardware, "original_result": {"path": str(directory / "original_paired_result.json"),
            "sha256": training.file_sha256(directory / "original_paired_result.json")},
        "loaded_gpu_libraries": training.actual_gpu_library_receipt(devices[0].client, libraries),
        "started_at_seconds": started, "finished_at_seconds": time.time()})
    diagnostic.record_progress(directory, "completed", completed=True, rounds=7)


def run_prepared(directory):
    """Validate the staged command, exact physical allocation and explicit runtime before any JAX import."""
    directory = Path(directory).resolve(strict=True)
    plan = json.loads((directory / "measurement_plan.json").read_text())
    with (directory / "diagnosis.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any((directory / name).exists() for name in ("hardware.json", "result.json", "failure.json", "memory_snapshots.jsonl")):
            raise ValueError("preserve the earlier diagnostic and prepare a new output unit")
        if (Path(__file__).resolve() != directory / "code/diagnose_pregather_allocator.py"
                or str(Path(sys.executable).absolute()) != plan["python"]
                or directory != Path(plan["campaign_run_dir"]) / "data" / plan["unit_id"]
                or json.loads((directory / "measurement_prepared.json").read_text()) != {
                    "plan_sha256": training.file_sha256(directory / "measurement_plan.json")}):
            raise ValueError("execute the original immutable staged diagnostic command")
        for name in HELPERS:
            if training.file_sha256(directory / "code" / name) != plan["helpers"]["files"]["10_jax_exploration_platform/scripts/atari_pixel_campaign/" + name]:
                raise ValueError("a staged diagnostic helper changed")
        measurement.verify_prepared(directory, plan)
        if allocator_source_receipt(Path(plan["allocator_parser"]["path"])) != plan["allocator_parser"]:
            raise ValueError("installed allocator-parser source changed")
        if any(os.environ.get(name) for name in ("TF_GPU_ALLOCATOR", "XLA_CLIENT_MEM_FRACTION")):
            raise ValueError("the diagnostic cannot inherit another allocator control or memory-fraction alias")
        actual = os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR")
        if actual not in (None, "cuda_async"):
            raise ValueError("a conflicting allocator was explicitly inherited")
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "cuda_async"
        runtime = training.enforce_runtime(plan["runtime_environment"], os.environ)
        if (any(os.environ.get(name) != value for name, value in {**ALLOCATOR, **THREADS, "JAXTARI_SKIP_SPRITE_UPDATE": "1"}.items())
                or len(os.sched_getaffinity(0)) != 8):
            raise ValueError("preserve the original allocator fraction, game runtime and eight allocated CPU threads")
        try:
            with measurement.card_admission(directory, plan):
                hardware = measurement.preflight(plan)
                assets = training.verify_assets(plan["asset_record"], os.environ)
                public = training.jaxtari_python_receipt(plan["jaxtari_source_root"])
                libraries = training.installed_gpu_library_receipts()
                diagnostic.atomic_json(directory / "hardware.json", {**hardware, "runtime_environment": runtime,
                    "cpu_affinity": sorted(os.sched_getaffinity(0)), "thread_environment": {key: os.environ[key] for key in THREADS}})
                diagnostic.atomic_json(directory / "asset_receipt.json", assets)
                diagnostic.atomic_json(directory / "installed_backend.json", {"jaxtari_source": public, "gpu_libraries": libraries})
                compute(directory, plan, hardware, libraries)
        except BaseException as error:
            diagnostic.atomic_json(directory / "failure.json", {"completed": False, "kind": plan["kind"],
                "hypothesis_count_contribution": 0, "eligible_for_formal_campaign_admission": False,
                "source_commit": SOURCE_COMMIT, "helpers": plan["helpers"], "runtime_environment": runtime,
                "allocator": plan["allocator"], "error": str(error), "exception_type": type(error).__name__,
                "traceback": traceback.format_exc(), "pid": os.getpid(), "time": training.utc_now()})
            diagnostic.record_progress(directory, "failed", error=repr(error))
            raise


def parse_arguments(argv=None):
    """Accept explicit immutable inputs or one prepared execution without learner overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-unit", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    names = ("source-root", "run-dir", "case", "reference-record", "runtime-environment", "asset-record",
             "jaxtari-source-root", "allocator-parser-source")
    for name in names:
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--unit-id")
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--python", type=Path, default=Path("/p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python"))
    arguments = parser.parse_args(argv)
    fields = [name.replace("-", "_") for name in (*names, "unit-id", "gpu-uuid")]
    if arguments.prepared_unit:
        if arguments.prepare_only or any(getattr(arguments, name) is not None for name in fields):
            parser.error("prepared execution cannot override its original diagnostic inputs")
    elif any(getattr(arguments, name) is None for name in fields):
        parser.error("preparation requires every original input, installed parser, output and physical GPU")
    return arguments


if __name__ == "__main__":
    arguments = parse_arguments()
    if arguments.prepared_unit:
        run_prepared(arguments.prepared_unit)
    else:
        output = prepare(arguments)
        print(json.dumps({"prepared": True, "unit_directory": str(output)}, indent=2), flush=True)
        if not arguments.prepare_only:
            os.execv(str(arguments.python), [str(arguments.python), str(output / "code/diagnose_pregather_allocator.py"),
                                           "--prepared-unit", str(output)])
