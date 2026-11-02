"""Test one explicit allocator-region preparation before original D28 initialization.

A fresh process allocates, synchronizes and deletes exactly one nine-GiB uint8
device array, records actual memory at all three boundaries, then calls the
original paired benchmark. The byte array never enters learner/game state.
Original numerical files, settings, tolerances and timed work remain pinned.
"""

import argparse
import fcntl
import gc
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
import diagnose_correctness_cache as original_inputs
import measure_selected as measurement

training, diagnostic, validation = measurement.training, measurement.diagnostic, measurement.validation
HELPERS = ("diagnose_allocator_region.py", "diagnose_correctness_cache.py", *measurement.HELPERS)
SOURCE_COMMIT = original_inputs.SOURCE_COMMIT
REGION_SHAPE = (9, 1024, 1024, 1024)
REGION_BYTES = 9 * 2 ** 30


def helper_receipt():
    """Pin this distinct diagnostic and every used input/measurement helper to committed bytes."""
    root = SCRIPTS.parents[2]
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    return training.source_receipt(root, commit, [str((SCRIPTS / name).relative_to(root)) for name in HELPERS])


def prepare(arguments):
    """Stage the original failed case and a separate explicit memory-history intervention."""
    if not arguments.gpu_uuid.startswith("GPU-") or Path(arguments.unit_id).name != arguments.unit_id or arguments.unit_id in {"", ".", ".."}:
        raise ValueError("record an exact physical UUID and plain descriptive diagnostic unit identifier")
    root, run = arguments.source_root.resolve(strict=True), arguments.run_dir.resolve(strict=True)
    case, reference, candidate, inputs = original_inputs.input_configuration(arguments.case, arguments.reference_record)
    if Path(case["source_root"]).resolve() != root:
        raise ValueError("use the original immutable case's actual numerical checkout")
    measurement.validate_complete_fields(root, reference)
    runtime, inputs["runtime_environment"] = measurement.read_document(arguments.runtime_environment)
    runtime = validation.runtime_environment(runtime)
    assets, inputs["asset_record"] = measurement.read_document(arguments.asset_record)
    if set(assets) != {"root", "manifest", "sha256"}:
        raise ValueError("use an immutable asset root/manifest/hash record")
    core = diagnostic.core_identity(root, SOURCE_COMMIT)
    paired = training.source_receipt(root, SOURCE_COMMIT, [
        "10_jax_exploration_platform/scripts/atari_pixel_campaign/" + name for name in ("paired_benchmark.py", "ideas.py")])
    directory = run / "data" / arguments.unit_id
    helpers = helper_receipt()
    plan = {"schema_version": 1, "kind": "allocator_region_history_diagnosis",
        "eligible_for_formal_campaign_admission": False, "hypothesis_count_contribution": 0,
        "source_root": str(root), "source_commit": SOURCE_COMMIT, "core": core, "paired_source": paired,
        "helpers": helpers, "configuration": reference, "candidate_configuration": candidate,
        "case": case, "original_case_bytes_sha256": inputs["case"]["sha256"], "inputs": inputs,
        "runtime_environment": runtime, "asset_record": assets, "compilation_cache_policy": "disabled",
        "jaxtari_source_root": str(arguments.jaxtari_source_root.resolve()) if arguments.jaxtari_source_root else None,
        "hardware_class": "jaguar03_rtx_a4500", "gpu_uuid": arguments.gpu_uuid, "copy_seeds": [1, 2],
        "rounds": case["rounds"], "iterations": case["iterations"], "latency_calls": 3,
        "intervention": {"phase": "before learner/game imports and initialization", "allocation_count": 1,
            "dtype": "uint8", "shape": list(REGION_SHAPE), "bytes": REGION_BYTES,
            "operations": ["allocate", "synchronize", "record allocated memory", "delete array", "gc.collect", "record released memory"]},
        "allocator": {"XLA_PYTHON_CLIENT_PREALLOCATE": "false", "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.85"},
        "campaign_run_dir": str(run), "unit_id": arguments.unit_id, "python": str(arguments.python.absolute())}
    artifacts = {directory / "measurement_plan.json": measurement.json_bytes(plan),
        directory / "manifest.yaml": measurement.json_bytes(plan), directory / "config_resolved.yaml": measurement.json_bytes({
            "reference": reference, "candidate": candidate, "runtime_environment": runtime, "allocator": plan["allocator"],
            "memory_preparation": plan["intervention"]}), directory / "inputs/original_case.json": arguments.case.read_bytes(),
        directory / "command.txt": (shlex.join([str(arguments.python), str(directory / "code/diagnose_allocator_region.py"),
                                               "--prepared-unit", str(directory)]) + "\n").encode(),
        directory / "experiment_background.md": (
            "## Purpose\n\nTest allocator-region history as a cause of the recorded paired-process A4500 memory failure. "
            "This separate diagnostic contributes zero ideas and supplies no scientific acceptance record.\n\n"
            "## Key hyperparameters\n\nThe original two-policy case retains 128 environments per policy, 128 rollout steps, "
            "four epochs, four minibatches, fifty normalization warmup rollouts and seeds one and two. "
            f"Its seven original paired rounds contain {case['iterations']} complete updates per variant. "
            "Neural/optimizer arithmetic is float32/highest; statistics are float64; deterministic XLA operations, "
            "allocator fraction 0.85 and disabled preallocation remain explicit.\n\n"
            "## Code and config changes\n\nBefore importing or initializing the learner/game, this helper allocates, "
            "synchronizes and deletes one nine-GiB uint8 device array. That array never enters model or game state. "
            "Actual before/allocated/released device, process and allocator memory and exact array bytes are recorded. "
            "The original paired benchmark then runs its original checks and timing protocol. No checkpoint is saved.\n\n"
            f"## Git state\n\nNumerical source: `{SOURCE_COMMIT}`. Helper source: `{helpers['source_commit']}`. "
            "Exact original case bytes and source/helper hashes are recorded in measurement_plan.json.\n").encode()}
    artifacts.update({directory / "code" / name: (SCRIPTS / name).read_bytes() for name in HELPERS})
    for path, contents in artifacts.items():
        if path.exists() and path.read_bytes() != contents:
            raise ValueError("an earlier allocator-region diagnostic artifact is immutable: " + str(path))
    for path, contents in artifacts.items():
        measurement.write_once(path, contents)
    measurement.write_once(directory / "measurement_prepared.json", measurement.json_bytes({
        "plan_sha256": hashlib.sha256(measurement.json_bytes(plan)).hexdigest()}))
    return directory


def prepare_region(array_runtime, array_library, snapshot, publish):
    """Allocate and release one exact-sized array before any original learner initialization."""
    before = snapshot()
    region = array_library.zeros(REGION_SHAPE, dtype=array_library.uint8)
    array_runtime.block_until_ready(region)
    nbytes = int(region.nbytes)
    if nbytes != REGION_BYTES or str(region.dtype) != "uint8":
        region.delete()
        raise ValueError("the diagnostic array differs from its exact nine-GiB uint8 specification")
    allocated = snapshot()
    region.delete()
    del region
    collected = gc.collect()
    released = snapshot()
    record = {"allocation_count": 1, "array_nbytes": nbytes, "array_dtype": "uint8", "array_shape": list(REGION_SHAPE),
              "before": before, "allocated": allocated, "released": released,
              "unreachable_python_objects_collected": collected, "allocator_pool_release_assumed": False}
    publish(record)
    return record


def run_prepared(directory):
    """Verify a fresh immutable diagnostic, exact card and original runtime before opening JAX."""
    directory = Path(directory).resolve(strict=True)
    plan = json.loads((directory / "measurement_plan.json").read_text())
    with (directory / "diagnosis.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any((directory / name).exists() for name in ("result.json", "failure.json", "hardware.json", "region_preparation.json")):
            raise ValueError("retain the earlier allocator-region diagnosis and prepare a new unit")
        if (json.loads((directory / "measurement_prepared.json").read_text()) != {
                "plan_sha256": training.file_sha256(directory / "measurement_plan.json")}
                or Path(__file__).resolve() != directory / "code/diagnose_allocator_region.py"
                or str(Path(sys.executable).absolute()) != plan["python"]
                or directory != Path(plan["campaign_run_dir"]) / "data" / plan["unit_id"]):
            raise ValueError("execute the unchanged staged allocator-region command and recorded Python/output")
        for name in HELPERS:
            relative = "10_jax_exploration_platform/scripts/atari_pixel_campaign/" + name
            if training.file_sha256(directory / "code" / name) != plan["helpers"]["files"][relative]:
                raise ValueError("a staged allocator-region helper changed: " + name)
        for name, module in (("measure_selected.py", measurement), ("diagnose_correctness_cache.py", original_inputs)):
            if Path(module.__file__).resolve() != directory / "code" / name:
                raise ValueError("the allocator-region diagnostic imported a mutable helper")
        measurement.verify_prepared(directory, plan)
        if training.file_sha256(directory / "inputs/original_case.json") != plan["original_case_bytes_sha256"]:
            raise ValueError("the original immutable case bytes changed")
        runtime = training.enforce_runtime(plan["runtime_environment"], os.environ)
        if any(os.environ.get(name) != value for name, value in plan["allocator"].items()):
            raise ValueError("the allocator-region diagnostic requires the original fraction 0.85 and disabled preallocation")
        if os.environ.get("JAXTARI_SKIP_SPRITE_UPDATE") != "1" or "jax" in sys.modules:
            raise ValueError("preserve the original game setting and admit this process before importing JAX")
        try:
            with measurement.card_admission(directory, plan):
                hardware = measurement.preflight(plan)
                assets = training.verify_assets(plan["asset_record"], os.environ)
                jaxtari = training.jaxtari_python_receipt(plan.get("jaxtari_source_root"))
                libraries = training.installed_gpu_library_receipts()
                diagnostic.atomic_json(directory / "hardware.json", {**hardware, "runtime_environment": runtime})
                diagnostic.atomic_json(directory / "asset_receipt.json", assets)
                diagnostic.atomic_json(directory / "installed_backend.json", {"jaxtari_source": jaxtari, "gpu_libraries": libraries})
                compute(directory, plan, hardware, libraries)
        except BaseException as error:
            diagnostic.atomic_json(directory / "failure.json", {"kind": plan["kind"], "error": repr(error),
                "exception_type": type(error).__name__, "traceback": traceback.format_exc(), "source_commit": SOURCE_COMMIT,
                "helper_commit": plan["helpers"]["source_commit"], "region_preparation_recorded": (directory / "region_preparation.json").exists()})
            diagnostic.record_progress(directory, "failed", error=repr(error))
            raise


def compute(directory, plan, hardware, libraries):
    """Prepare one physical region, then import and call the original paired implementation."""
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu" or "A4500" not in devices[0].device_kind:
        raise ValueError("the allocator-region diagnostic requires its single assigned RTX A4500")
    observation = measurement.MemoryObservation(devices[0], plan["gpu_uuid"])
    def snapshot():
        """Read real memory without assuming the allocator returns cached regions to the driver."""
        return {"time_seconds": time.time(), "device_memory": observation.sample(),
                "allocator": {name: int(value) for name, value in (devices[0].memory_stats() or {}).items()}}
    def publish(record):
        """Flush all three memory states before importing or initializing model/game code."""
        diagnostic.atomic_json(directory / "region_preparation.json", {**record, "source_commit": SOURCE_COMMIT,
            "helpers": plan["helpers"], "runtime_environment": plan["runtime_environment"], "phase": "before learner/game imports"})
        diagnostic.record_progress(directory, "allocator_region_prepared", actual_array_nbytes=record["array_nbytes"])
    diagnostic.record_progress(directory, "preparing_allocator_region")
    region = prepare_region(jax, jnp, snapshot, publish)
    root = Path(plan["source_root"]) / "10_jax_exploration_platform"
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "scripts/atari_pixel_campaign"))
    import jaxtari
    from exploration_platform.atari_pixels.config import PixelPPOConfig
    paired = importlib.import_module("paired_benchmark")
    measurement.verify_imported_sources(root, paired)
    def progress(value):
        """Keep the unchanged paired function's actual correctness and timing stages."""
        diagnostic.record_progress(directory, value["state"], **{name: item for name, item in value.items() if name != "state"})
    started = time.time()
    result = paired.run_paired_benchmark(PixelPPOConfig(**plan["configuration"]["learner"]),
        plan["case"]["candidate"]["learner"], plan["case"]["candidate"]["environment"], plan["case"]["candidate"]["execution"],
        rounds=plan["rounds"], iterations=plan["iterations"], seed_start=1, progress=progress,
        reference_environment=plan["configuration"]["environment"], reference_execution=plan["configuration"]["execution"],
        latency_calls=plan["latency_calls"])
    diagnostic.atomic_json(directory / "result.json", {"completed": True, "kind": plan["kind"],
        "eligible_for_formal_campaign_admission": False, "hypothesis_count_contribution": 0,
        "source_commit": SOURCE_COMMIT, "helpers": plan["helpers"], "runtime_environment": plan["runtime_environment"],
        "configuration": plan["configuration"], "candidate_configuration": plan["candidate_configuration"],
        "hardware": hardware, "original_case_bytes_sha256": plan["original_case_bytes_sha256"],
        "region_preparation": region, "paired_measurement": result,
        "loaded_gpu_libraries": training.actual_gpu_library_receipt(devices[0].client, libraries),
        "started_at_seconds": started, "finished_at_seconds": time.time()})
    diagnostic.record_progress(directory, "completed", completed=True, rounds=7, actual_array_nbytes=REGION_BYTES)


def parse_arguments(argv=None):
    """Use explicit original inputs or an immutable prepared command without learner overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-unit", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    for name in ("source-root", "run-dir", "case", "reference-record", "runtime-environment", "asset-record", "jaxtari-source-root"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--unit-id")
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--python", type=Path, default=Path("/p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python"))
    arguments = parser.parse_args(argv)
    if arguments.prepared_unit is not None:
        if arguments.prepare_only or any(getattr(arguments, name) is not None for name in (
                "source_root", "run_dir", "case", "reference_record", "runtime_environment", "asset_record",
                "jaxtari_source_root", "unit_id", "gpu_uuid")):
            parser.error("prepared execution cannot override its original case, allocation or runtime")
    elif any(getattr(arguments, name) is None for name in ("source_root", "run_dir", "case", "reference_record",
            "runtime_environment", "asset_record", "unit_id", "gpu_uuid")):
        parser.error("preparation needs original case/reference/source/runtime/assets/output and physical GPU")
    return arguments


if __name__ == "__main__":
    arguments = parse_arguments()
    if arguments.prepared_unit is not None:
        run_prepared(arguments.prepared_unit)
    else:
        directory = prepare(arguments)
        print(json.dumps({"prepared": True, "unit_directory": str(directory)}, indent=2), flush=True)
        if not arguments.prepare_only:
            os.execv(str(arguments.python), [str(arguments.python), str(directory / "code/diagnose_allocator_region.py"),
                                           "--prepared-unit", str(directory)])
