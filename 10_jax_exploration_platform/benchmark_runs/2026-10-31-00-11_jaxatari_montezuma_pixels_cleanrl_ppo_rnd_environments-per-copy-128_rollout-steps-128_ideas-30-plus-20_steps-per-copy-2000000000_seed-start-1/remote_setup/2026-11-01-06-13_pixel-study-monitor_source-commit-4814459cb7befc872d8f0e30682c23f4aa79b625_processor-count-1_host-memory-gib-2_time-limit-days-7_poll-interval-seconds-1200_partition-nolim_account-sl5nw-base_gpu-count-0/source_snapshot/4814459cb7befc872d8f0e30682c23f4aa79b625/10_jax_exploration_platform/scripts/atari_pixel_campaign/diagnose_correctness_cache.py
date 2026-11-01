"""Diagnose temporary correctness-executable retention without changing D28 files.

The only intervention wraps the frozen check_fixed_inputs function: after its
original host comparisons succeed, record memory, clear JAX executable caches,
collect unreachable Python objects, and record memory again. The unchanged
paired benchmark then performs its original whole-iteration checks and timings.
This is a diagnostic, not an accepted optimization or a study result.
"""

import argparse
from contextlib import contextmanager
import copy
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
import measure_selected as measurement

training, diagnostic, validation = measurement.training, measurement.diagnostic, measurement.validation
HELPERS = ("diagnose_correctness_cache.py", *measurement.HELPERS)
SOURCE_COMMIT = "d28e6735e79788f618acb0ebfde8609db1f776af"
SUPPORTED_CASES = {"batch_rollout_intrinsic_rewards": {"hoist_rollout_bonus": True},
                   "unroll_minibatch_update_loop": {"update_unroll": 2}}


def input_configuration(case_path, reference_path):
    """Read the actual original case and full completed reference without inventing defaults."""
    case, case_receipt = measurement.read_document(case_path)
    reference, reference_receipt = measurement.read_document(reference_path)
    name = case.get("hypothesis")
    if (name not in SUPPORTED_CASES or case.get("source_commit") != SOURCE_COMMIT
            or reference.get("source_commit") != SOURCE_COMMIT or reference.get("case_completed") is not True
            or reference.get("copies") != 2 or reference.get("hardware_class") != "jaguar03_rtx_a4500"
            or case.get("hardware_class") != "jaguar03_rtx_a4500" or case.get("seed_start") != 1
            or case.get("reference_environment") is not None or case.get("reference_execution") is not None
            or case.get("candidate") != {"learner": SUPPORTED_CASES[name], "environment": {}, "execution": {}}):
        raise ValueError("use an original D28 two-copy A4500 case and its completed full reference")
    configuration = validation.configuration(reference["reference"])
    if (case.get("configuration") != configuration["learner"] or configuration["learner"]["n_copies"] != 2
            or case.get("rounds") != 7 or type(case.get("iterations")) is not int or case["iterations"] < 1):
        raise ValueError("the original case/reference configuration or measurement protocol differs")
    candidate = copy.deepcopy(configuration)
    candidate["learner"].update(case["candidate"]["learner"])
    return case, configuration, candidate, {"case": case_receipt, "completed_reference": reference_receipt}


def helper_receipt():
    """Keep committed diagnostic-helper bytes separate from the untouched numerical source."""
    root = SCRIPTS.parents[2]
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    return training.source_receipt(root, commit, [str((SCRIPTS / name).relative_to(root)) for name in HELPERS])


def prepare(arguments):
    """Stage an explicit original-protocol diagnostic without importing arrays or accessing a GPU."""
    if not arguments.gpu_uuid.startswith("GPU-") or Path(arguments.unit_id).name != arguments.unit_id or arguments.unit_id in {"", ".", ".."}:
        raise ValueError("record an exact physical UUID and plain descriptive diagnostic unit identifier")
    root, run = arguments.source_root.resolve(strict=True), arguments.run_dir.resolve(strict=True)
    case, reference, candidate, inputs = input_configuration(arguments.case, arguments.reference_record)
    if Path(case["source_root"]).resolve() != root:
        raise ValueError("the source root must be the original immutable case's actual D28 checkout")
    measurement.validate_complete_fields(root, reference)
    runtime, inputs["runtime_environment"] = measurement.read_document(arguments.runtime_environment)
    runtime = validation.runtime_environment(runtime)
    assets, inputs["asset_record"] = measurement.read_document(arguments.asset_record)
    if set(assets) != {"root", "manifest", "sha256"}:
        raise ValueError("use the immutable asset root/manifest/hash schema")
    core = diagnostic.core_identity(root, SOURCE_COMMIT)
    paired = training.source_receipt(root, SOURCE_COMMIT, [
        "10_jax_exploration_platform/scripts/atari_pixel_campaign/" + name for name in ("paired_benchmark.py", "ideas.py")])
    directory = run / "data" / arguments.unit_id
    plan = {"schema_version": 1, "kind": "temporary_correctness_cache_diagnosis",
        "eligible_for_formal_campaign_admission": False, "hypothesis_count_contribution": 0,
        "source_root": str(root), "source_commit": SOURCE_COMMIT, "core": core, "paired_source": paired,
        "helpers": helper_receipt(), "configuration": reference, "candidate_configuration": candidate,
        "case": case, "original_case_bytes_sha256": inputs["case"]["sha256"], "inputs": inputs,
        "runtime_environment": runtime, "asset_record": assets, "compilation_cache_policy": "disabled",
        "jaxtari_source_root": str(arguments.jaxtari_source_root.resolve()) if arguments.jaxtari_source_root else None,
        "hardware_class": "jaguar03_rtx_a4500", "gpu_uuid": arguments.gpu_uuid, "copy_seeds": [1, 2],
        "rounds": case["rounds"], "iterations": case["iterations"], "latency_calls": 3,
        "intervention": "jax.clear_caches then gc.collect after original fixed-input comparisons return",
        "allocator": {"XLA_PYTHON_CLIENT_PREALLOCATE": "false", "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.85"},
        "campaign_run_dir": str(run), "unit_id": arguments.unit_id, "python": str(arguments.python.absolute())}
    artifacts = {directory / "measurement_plan.json": measurement.json_bytes(plan),
        directory / "manifest.yaml": measurement.json_bytes(plan), directory / "config_resolved.yaml": measurement.json_bytes({
            "reference": reference, "candidate": candidate, "runtime_environment": runtime, "allocator": plan["allocator"]}),
        directory / "inputs/original_case.json": arguments.case.read_bytes(),
        directory / "command.txt": (shlex.join([str(arguments.python), str(directory / "code/diagnose_correctness_cache.py"),
                                               "--prepared-unit", str(directory)]) + "\n").encode(),
        directory / "experiment_background.md": (
            "## Purpose\n\nTest whether temporary fixed-input correctness executables contribute to the recorded two-copy A4500 "
            "memory failure. This is a diagnostic, not a fix or a scientific campaign case.\n\n"
            "## Key hyperparameters\n\nThe original case retains two independent policies, 128 environments per policy, "
            "128 rollout steps, four epochs, four minibatches, fifty normalization warmup rollouts, and seeds one and two. "
            f"Its original seven paired timing rounds contain {case['iterations']} complete updates per variant. "
            "Float32/highest neural arithmetic, float64 statistics, deterministic XLA operations, original allocator "
            "fraction 0.85 and disabled preallocation remain explicit.\n\n"
            "## Code and config changes\n\nOnly a separate diagnostic wrapper calls jax.clear_caches and gc.collect after "
            "the unchanged fixed-input host comparisons finish and before whole-iteration correctness checks begin. "
            "Real device/process/allocator memory is recorded immediately before and after; GPU-pool release is not assumed. "
            "No timed training, tolerance, original case bytes, numerical source or registry entry is changed.\n\n"
            f"## Git state\n\nNumerical source: `{SOURCE_COMMIT}`. Exact helper/source/file identities are recorded "
            "in measurement_plan.json.\n").encode()}
    artifacts.update({directory / "code" / name: (SCRIPTS / name).read_bytes() for name in HELPERS})
    for path, contents in artifacts.items():
        if path.exists() and path.read_bytes() != contents:
            raise ValueError("an earlier diagnostic artifact is immutable: " + str(path))
    for path, contents in artifacts.items():
        measurement.write_once(path, contents)
    measurement.write_once(directory / "measurement_prepared.json", measurement.json_bytes({
        "plan_sha256": hashlib.sha256(measurement.json_bytes(plan)).hexdigest()}))
    return directory


@contextmanager
def release_after_fixed_inputs(paired, array_runtime, snapshot, publish):
    """Intervene only after every original fixed-input host comparison has succeeded."""
    original, calls = paired.check_fixed_inputs, []
    def checked_then_release(reference, candidate, host_state):
        """Preserve the complete original check result and expose one explicit cache-release boundary."""
        reports = original(reference, candidate, host_state)
        before = snapshot(reference, candidate)
        array_runtime.clear_caches()
        collected = gc.collect()
        after = snapshot(reference, candidate)
        record = {"fixed_input_comparisons_completed": True, "original_reports": reports,
                  "before": before, "after": after, "unreachable_python_objects_collected": collected,
                  "cache_release_api": "jax.clear_caches", "allocator_pool_release_assumed": False}
        calls.append(record)
        publish(record)
        return reports
    paired.check_fixed_inputs = checked_then_release
    try:
        yield calls
    finally:
        paired.check_fixed_inputs = original


def run_prepared(directory):
    """Admit exact staged metadata and a single allocated A4500 before the numerical diagnostic."""
    directory = Path(directory).resolve(strict=True)
    plan = json.loads((directory / "measurement_plan.json").read_text())
    with (directory / "diagnosis.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any((directory / name).exists() for name in ("result.json", "failure.json", "hardware.json", "cache_release.json")):
            raise ValueError("retain the prior diagnosis and prepare a new immutable unit")
        if (json.loads((directory / "measurement_prepared.json").read_text()) != {
                "plan_sha256": training.file_sha256(directory / "measurement_plan.json")}
                or Path(__file__).resolve() != directory / "code/diagnose_correctness_cache.py"
                or str(Path(sys.executable).absolute()) != plan["python"]
                or directory != Path(plan["campaign_run_dir"]) / "data" / plan["unit_id"]):
            raise ValueError("execute the unchanged staged diagnostic with its recorded plan, Python and output location")
        for name in HELPERS:
            relative = "10_jax_exploration_platform/scripts/atari_pixel_campaign/" + name
            if training.file_sha256(directory / "code" / name) != plan["helpers"]["files"][relative]:
                raise ValueError("a staged diagnostic helper changed: " + name)
        if Path(measurement.__file__).resolve() != directory / "code/measure_selected.py":
            raise ValueError("the diagnostic must import its own staged measurement helper")
        measurement.verify_prepared(directory, plan)
        if training.file_sha256(directory / "inputs/original_case.json") != plan["original_case_bytes_sha256"]:
            raise ValueError("the diagnostic changed the original immutable case bytes")
        runtime = training.enforce_runtime(plan["runtime_environment"], os.environ)
        for name, expected in plan["allocator"].items():
            if os.environ.get(name) != expected:
                raise ValueError("the diagnostic must retain the original worker allocator setting: " + name)
        if os.environ.get("JAXTARI_SKIP_SPRITE_UPDATE") != "1" or "jax" in sys.modules:
            raise ValueError("preserve the original game setting and admit the process before importing JAX")
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
                "exception_type": type(error).__name__, "traceback": traceback.format_exc(),
                "source_commit": SOURCE_COMMIT, "helper_commit": plan["helpers"]["source_commit"],
                "cache_release_recorded": (directory / "cache_release.json").exists()})
            diagnostic.record_progress(directory, "failed", error=repr(error))
            raise


def compute(directory, plan, hardware, libraries):
    """Run the original paired function with one explicitly recorded non-timed cache intervention."""
    root = Path(plan["source_root"]) / "10_jax_exploration_platform"
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "scripts/atari_pixel_campaign"))
    import jax
    import jaxtari
    from exploration_platform.atari_pixels.config import PixelPPOConfig
    paired = importlib.import_module("paired_benchmark")
    measurement.verify_imported_sources(root, paired)
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu" or "A4500" not in devices[0].device_kind:
        raise ValueError("the correctness-cache diagnostic requires exactly the assigned RTX A4500")
    observation = measurement.MemoryObservation(devices[0], plan["gpu_uuid"])
    def snapshot(reference, candidate):
        """Read real backend/device/process memory without claiming reserved-pool memory is returned."""
        return {"time_seconds": time.time(), "device_memory": observation.sample(),
                "allocator": {name: int(value) for name, value in (devices[0].memory_stats() or {}).items()},
                "whole_iteration_compiled_signatures": {"reference": list(reference.cache_sizes()), "candidate": list(candidate.cache_sizes())}}
    def publish(record):
        """Flush the phase boundary before any original whole-iteration executable can fail."""
        diagnostic.atomic_json(directory / "cache_release.json", {**record, "source_commit": SOURCE_COMMIT,
            "helpers": plan["helpers"], "runtime_environment": plan["runtime_environment"],
            "loaded_gpu_libraries": training.actual_gpu_library_receipt(devices[0].client, libraries)})
        diagnostic.record_progress(directory, "temporary_correctness_caches_released")
    def progress(value):
        """Preserve the frozen function's actual stages and measured round records."""
        diagnostic.record_progress(directory, value["state"], **{name: item for name, item in value.items() if name != "state"})
    started = time.time()
    with release_after_fixed_inputs(paired, jax, snapshot, publish) as releases:
        original = paired.run_paired_benchmark(PixelPPOConfig(**plan["configuration"]["learner"]),
            plan["case"]["candidate"]["learner"], plan["case"]["candidate"]["environment"],
            plan["case"]["candidate"]["execution"], rounds=plan["rounds"], iterations=plan["iterations"],
            seed_start=1, progress=progress, reference_environment=plan["configuration"]["environment"],
            reference_execution=plan["configuration"]["execution"], latency_calls=plan["latency_calls"])
    if len(releases) != 1:
        raise ValueError("the diagnostic did not observe exactly one original fixed-input correctness boundary")
    result = {"completed": True, "kind": plan["kind"], "eligible_for_formal_campaign_admission": False,
        "hypothesis_count_contribution": 0, "source_commit": SOURCE_COMMIT, "helpers": plan["helpers"],
        "runtime_environment": plan["runtime_environment"], "configuration": plan["configuration"],
        "candidate_configuration": plan["candidate_configuration"], "hardware": hardware,
        "original_case_bytes_sha256": plan["original_case_bytes_sha256"], "cache_release_calls": len(releases),
        "paired_measurement": original, "started_at_seconds": started, "finished_at_seconds": time.time()}
    diagnostic.atomic_json(directory / "result.json", result)
    diagnostic.record_progress(directory, "completed", completed=True, rounds=7, cache_release_calls=1)


def parse_arguments(argv=None):
    """Use explicit original case metadata or its immutable staged command, never hidden overrides."""
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
            parser.error("prepared execution cannot override its original case or runtime")
        return arguments
    if any(getattr(arguments, name) is None for name in ("source_root", "run_dir", "case", "reference_record",
            "runtime_environment", "asset_record", "unit_id", "gpu_uuid")):
        parser.error("preparation requires original case/reference/source/runtime/assets/output and physical GPU")
    return arguments


if __name__ == "__main__":
    arguments = parse_arguments()
    if arguments.prepared_unit is not None:
        run_prepared(arguments.prepared_unit)
    else:
        directory = prepare(arguments)
        print(json.dumps({"prepared": True, "unit_directory": str(directory)}, indent=2), flush=True)
        if not arguments.prepare_only:
            os.execv(str(arguments.python), [str(arguments.python), str(directory / "code/diagnose_correctness_cache.py"),
                                           "--prepared-unit", str(directory)])
