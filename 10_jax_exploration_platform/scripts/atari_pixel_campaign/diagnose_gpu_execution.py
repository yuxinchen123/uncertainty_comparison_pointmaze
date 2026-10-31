"""Measure one frozen full-size reference policy and trace one later update separately.

Only run this helper on the recorded GPU host after the launcher has acquired the
standing card lock. Importing this file performs standard-library metadata work only.
"""

import argparse
import ast
import ctypes
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import sysconfig
import time
import traceback


GPU_UUID = "GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3"
CORE_SCOPES = ("10_jax_exploration_platform/src/exploration_platform/atari_pixels",
               "10_jax_exploration_platform/scripts/atari_pixel_campaign/worker_environment.sh")


def atomic_json(path, value):
    """Replace a progress record only after its complete contents reach disk."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def core_identity(source_root, commit):
    """Verify the unchanged frozen core against Git and retain a SHA-256 hash for every used file."""
    root = Path(source_root).resolve(strict=True)
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if actual != commit or len(commit) != 40:
        raise ValueError("the frozen learner checkout does not match the requested full commit")
    branch = subprocess.run(["git", "-C", str(root), "symbolic-ref", "--quiet", "HEAD"],
                            capture_output=True, text=True)
    if branch.returncode != 1:
        raise ValueError("the diagnostic core must be a detached frozen Git checkout")
    changes = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain",
        "--untracked-files=all", "--", *CORE_SCOPES], text=True).strip()
    if changes:
        raise ValueError("the frozen learner or worker environment has local changes: " + changes)
    entries = subprocess.check_output(["git", "-C", str(root), "ls-tree", "-rz", commit, "--", *CORE_SCOPES])
    files = {}
    for entry in entries.split(b"\0"):
        if not entry:
            continue
        header, encoded_path = entry.split(b"\t", 1)
        mode, kind, expected = header.decode().split()
        path = encoded_path.decode()
        if kind != "blob" or mode not in {"100644", "100755"}:
            raise ValueError("diagnostic core source must contain ordinary tracked files")
        contents = (root / path).read_bytes()
        actual_blob = hashlib.sha1(b"blob " + str(len(contents)).encode() + b"\0" + contents).hexdigest()
        if actual_blob != expected:
            raise ValueError("a frozen source file differs from its recorded commit: " + path)
        files[path] = {"git_blob": expected, "sha256": hashlib.sha256(contents).hexdigest()}
    if not files:
        raise ValueError("the requested commit contains no pixel learner source")
    return {"source_root": str(root), "source_commit": commit, "files": files}


def default_configuration(source_root):
    """Read literal frozen defaults without importing JAX or creating a numerical context."""
    directory = Path(source_root) / "10_jax_exploration_platform/src/exploration_platform/atari_pixels"
    config_tree = ast.parse((directory / "config.py").read_text())
    config_class = next(node for node in config_tree.body if isinstance(node, ast.ClassDef) and node.name == "PixelPPOConfig")
    learner = {node.target.id: ast.literal_eval(node.value) for node in config_class.body
               if isinstance(node, ast.AnnAssign) and node.value is not None}
    environment_tree = ast.parse((directory / "environment.py").read_text())
    environment_class = next(node for node in environment_tree.body
                             if isinstance(node, ast.ClassDef) and node.name == "AtariPixelEnvironment")
    constructor = next(node for node in environment_class.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    environment = {argument.arg: ast.literal_eval(value) for argument, value in
                   zip(constructor.args.args[-len(constructor.args.defaults):], constructor.args.defaults)}
    return {"learner": learner, "environment": environment,
            "execution": {"compile_updates": 2, "timed_rounds": 2, "updates_per_round": 2,
                          "traced_updates": 1, "seed": 1, "precision": "float32",
                          "statistics_precision": "float64", "matmul_precision": "highest",
                          "deterministic_operations": True, "formal_benchmark": False,
                          "capacity_evidence": False}}


def record_progress(directory, phase, **fields):
    """Flush one phase record so the existing monitor can distinguish progress from waiting."""
    record = {"record": "diagnostic", "state": phase, "pid": os.getpid(),
              "time": datetime.now(timezone.utc).isoformat(), **fields}
    with (directory / "metrics.jsonl").open("a") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
        stream.flush()
    atomic_json(directory / "progress.json", record)


def hardware_record():
    """Record only the selected H100 and the exact runtime settings used by this process."""
    command = ["nvidia-smi", "-i", GPU_UUID,
               "--query-gpu=uuid,name,memory.total,driver_version", "--format=csv,noheader"]
    query = subprocess.check_output(command, text=True).strip()
    if not query.startswith(GPU_UUID + ",") or "H100" not in query or "\n" in query:
        raise ValueError("the diagnostic did not resolve exactly the permitted H100")
    runtime = {name: os.environ.get(name, "") for name in (
        "JAX_ENABLE_X64", "JAX_DEFAULT_MATMUL_PRECISION", "XLA_FLAGS", "CUDA_VISIBLE_DEVICES",
        "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION", "OMP_NUM_THREADS")}
    flags = [flag for flag in shlex.split(runtime["XLA_FLAGS"])
             if flag.startswith("--xla_gpu_exclude_nondeterministic_ops=")]
    if (runtime["CUDA_VISIBLE_DEVICES"] != GPU_UUID or runtime["JAX_ENABLE_X64"] != "true"
            or runtime["JAX_DEFAULT_MATMUL_PRECISION"] != "highest"
            or flags != ["--xla_gpu_exclude_nondeterministic_ops=true"]):
        raise ValueError("the diagnostic must retain the exact GPU, precision and deterministic settings")
    return {"hostname": socket.gethostname(), "gpu_query": query,
            "runtime_environment": runtime, "pid": os.getpid()}


def cuda_profiler():
    """Load the environment's CUDA runtime for the explicit one-update capture range."""
    path = Path(sysconfig.get_paths()["purelib"]) / "nvidia/cuda_runtime/lib/libcudart.so.12"
    library = ctypes.CDLL(str(path))
    for name in ("cudaProfilerStart", "cudaProfilerStop"):
        function = getattr(library, name)
        function.argtypes = []
        function.restype = ctypes.c_int
    return library, path


def loaded_gpu_libraries(backend):
    """Record the backend and query versions from the CUDA libraries actually loaded after compilation."""
    paths = sorted({line.rsplit(None, 1)[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                    if "/" in line and any(name in line for name in ("libcudnn", "libcudart", "libcuda.so"))})
    cudnn_paths = [path for path in paths if re.fullmatch(r"libcudnn\.so(?:\.\d+)*", Path(path).name)]
    runtime_paths = [path for path in paths if re.fullmatch(r"libcudart\.so(?:\.\d+)*", Path(path).name)]
    if len(cudnn_paths) != 1 or len(runtime_paths) != 1:
        raise ValueError("the compiled process must identify one loaded cuDNN library and CUDA runtime")
    # RTLD_NOLOAD queries the existing objects and cannot substitute another package version.
    cudnn = ctypes.CDLL(cudnn_paths[0], mode=os.RTLD_NOLOAD | os.RTLD_LOCAL)
    cudnn.cudnnGetVersion.argtypes = []
    cudnn.cudnnGetVersion.restype = ctypes.c_size_t
    runtime = ctypes.CDLL(runtime_paths[0], mode=os.RTLD_NOLOAD | os.RTLD_LOCAL)
    runtime.cudaRuntimeGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    runtime.cudaRuntimeGetVersion.restype = ctypes.c_int
    cuda_version = ctypes.c_int()
    cuda_status = runtime.cudaRuntimeGetVersion(ctypes.byref(cuda_version))
    if cuda_status != 0:
        raise RuntimeError(f"cudaRuntimeGetVersion failed with status {cuda_status}")
    return {"platform": backend.platform, "platform_version": backend.platform_version,
            "loaded_shared_libraries": paths, "cudnn_library": cudnn_paths[0],
            "cudnnGetVersion": cudnn.cudnnGetVersion(), "cuda_runtime_library": runtime_paths[0],
            "cudaRuntimeGetVersion": cuda_version.value}


def run(args):
    """Run two compile updates, four timed updates and one separately traced reference update."""
    directory = args.unit_dir.resolve(strict=True)
    plan = json.loads((directory / "diagnostic_plan.json").read_text())
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != plan["helper"]["sha256"]:
        raise ValueError("the staged diagnostic helper changed after preparation")
    if core_identity(Path(plan["core"]["source_root"]), plan["core"]["source_commit"]) != plan["core"]:
        raise ValueError("the frozen core changed after preparation")
    config_document = default_configuration(plan["core"]["source_root"])
    if json.loads(json.dumps(config_document)) != plan["configuration"]:
        raise ValueError("the frozen default configuration changed after preparation")
    atomic_json(directory / "hardware.json", hardware_record())

    # All numerical imports happen here, after the parent has recorded this process
    # and the host-side launcher has waited for predecessors and acquired the card.
    platform = Path(plan["core"]["source_root"]) / "10_jax_exploration_platform"
    sys.path.insert(0, str(platform / "src"))
    import jax
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    from exploration_platform.atari_pixels.config import PixelPPOConfig
    from exploration_platform.atari_pixels.environment import AtariPixelEnvironment
    from exploration_platform.atari_pixels.training import PixelPPORND
    if any("H100" not in device.device_kind for device in jax.devices()) or len(jax.devices()) != 1:
        raise ValueError("JAX must expose exactly the selected H100")
    config = PixelPPOConfig(**config_document["learner"])
    if json.loads(json.dumps(asdict(config))) != plan["configuration"]["learner"]:
        raise ValueError("the constructed learner differs from its recorded defaults")
    environment = AtariPixelEnvironment(**config_document["environment"])
    trainer = PixelPPORND(config, environment)
    atomic_json(directory / "framework.json", {"jax": jax.__version__,
        "devices": [str(device) for device in jax.devices()], "configuration": asdict(config)})
    record_progress(directory, "initializing")
    started = time.monotonic()
    state = trainer.initialize((1,))
    jax.block_until_ready(state)
    record_progress(directory, "normalization_warmup", initialize_seconds=time.monotonic() - started)
    state = trainer.warmup(state)
    jax.block_until_ready(state)
    initialization_seconds = time.monotonic() - started

    # Check leaf dtypes without copying model arrays back to the host.
    groups = {"parameters": state.parameters, "target": state.target,
              "optimizer_first": state.optimizer.first, "optimizer_second": state.optimizer.second,
              "observation_statistics": state.observation_statistics,
              "reward_statistics": state.reward_statistics}
    dtypes = {name: sorted({str(leaf.dtype) for leaf in jax.tree.leaves(values)}) for name, values in groups.items()}
    if any(values != (["float64"] if "statistics" in name else ["float32"]) for name, values in dtypes.items()):
        raise ValueError("the learner contains an unexpected numerical dtype: " + str(dtypes))
    atomic_json(directory / "dtypes.json", dtypes)
    record_progress(directory, "compile_updates", initialization_seconds=initialization_seconds)
    compile_started = time.monotonic()
    for update in range(2):
        state, metrics = trainer.train_iteration(state)
        jax.block_until_ready(state)
        record_progress(directory, "compile_updates", updates_completed=update + 1)
    compilation_seconds = time.monotonic() - compile_started
    atomic_json(directory / "loaded_gpu_libraries.json", loaded_gpu_libraries(jax.devices()[0].client))

    # The capture remains disabled for these four updates. Nsight may be injected,
    # so these diagnostic timings are never admitted as formal benchmark results.
    rounds = []
    for index in range(2):
        record_progress(directory, "timed_updates", round=index, updates_completed=2 + index * 2)
        measured_started = time.monotonic()
        for _ in range(2):
            state, metrics = trainer.train_iteration(state)
        jax.block_until_ready(state)
        elapsed = time.monotonic() - measured_started
        measurement = {"round": index, "seconds": elapsed, "updates": 2,
                       "seconds_per_update": elapsed / 2, "transitions": config.batch_size * 2,
                       "transitions_per_second": config.batch_size * 2 / elapsed,
                       "trace_capture_active": False}
        rounds.append(measurement)
        record_progress(directory, "timed_round_completed", **measurement)
    atomic_json(directory / "timing_result.json", {"diagnostic_only": True, "completed": True,
        "initialization_seconds": initialization_seconds, "compilation_seconds": compilation_seconds,
        "copies": 1, "rounds": rounds, "core": plan["core"], "helper": plan["helper"]})

    # Trace a seventh update after all timing measurements are durably saved.
    trace = {"requested": args.trace_enabled, "completed": False}
    if args.trace_enabled:
        profiler, runtime_path = cuda_profiler()
        record_progress(directory, "traced_update", updates_completed=6)
        code = profiler.cudaProfilerStart()
        if code != 0:
            raise RuntimeError(f"cudaProfilerStart failed with CUDA status {code}")
        try:
            state, metrics = trainer.train_iteration(state)
            jax.block_until_ready(state)
        finally:
            stop_code = profiler.cudaProfilerStop()
        if stop_code != 0:
            raise RuntimeError(f"cudaProfilerStop failed with CUDA status {stop_code}")
        trace.update(completed=True, updates=1, cuda_runtime=str(runtime_path))
    else:
        trace["reason"] = "Nsight Systems executable was not available on the GPU host"
    result = {"completed": True, "diagnostic_only": True, "formal_benchmark": False,
              "capacity_evidence": False, "copies": 1, "rounds": rounds,
              "initialization_seconds": initialization_seconds, "compilation_seconds": compilation_seconds,
              "training_updates": 7 if args.trace_enabled else 6, "trace": trace,
              "source_commit": plan["core"]["source_commit"], "helper": plan["helper"]}
    atomic_json(directory / "result.json", result)
    record_progress(directory, "completed", completed=True, training_updates=result["training_updates"])
    print(json.dumps(result), flush=True)


def main():
    """Expose only the fixed diagnostic protocol and report failures in the same unit directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit-dir", type=Path, required=True)
    parser.add_argument("--trace-enabled", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except BaseException as error:
        atomic_json(args.unit_dir / "failure.json", {"error": repr(error), "traceback": traceback.format_exc(),
                    "time": datetime.now(timezone.utc).isoformat(), "diagnostic_only": True})
        raise


if __name__ == "__main__":
    main()
