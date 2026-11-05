"""Hash the complete runtime dependency union on owned CPU without a native or numerical operation."""
import ast
import ctypes
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    """Require the exact monitor step and retain every staged helper plus protected numerical source hash."""
    if os.environ.get("SLURM_JOB_ID") != "6783003" or not os.environ.get("SLURM_STEP_ID", "").isdigit():
        raise ValueError("use only the owned CPU monitor's numeric step")
    root = Path(__file__).resolve().parents[2]
    scripts = root / "10_jax_exploration_platform/scripts/atari_pixel_campaign"
    receipt = Path(sys.argv[1])
    if receipt.exists():
        raise ValueError("retain the previous source proof")
    def audit(event, args):
        """Allow fixed local Git source inspection and refuse tools, numerical libraries or networking."""
        if event == "subprocess.Popen" and (args[0] != "git" or args[1][0] != "git"):
            raise RuntimeError("only local Git source bytes may be read")
        if event in {"ctypes.dlopen", "ctypes.dlsym", "os.system", "socket.connect", "socket.bind", "socket.getaddrinfo"}:
            raise RuntimeError("no library/device/network operations in source proof")
        if event == "import" and str(args[0]).split(".")[0] in {"jax", "jaxlib", "numpy", "torch", "jaxtari", "cupy", "scipy", "pynvml", "optax", "flax"}:
            raise RuntimeError("no numerical imports in source proof")
    sys.addaudithook(audit)
    sys.path.insert(0, str(scripts))
    import group_runtime
    names = group_runtime.helper_names()
    files = {str(scripts / name): hashlib.sha256((scripts / name).read_bytes()).hexdigest() for name in names}
    numerical = subprocess.check_output(["git", "-C", str(root), "ls-tree", "-r", "--name-only", "HEAD", "--",
        "10_jax_exploration_platform/src/exploration_platform/atari_pixels"], text=True).splitlines()
    numerical += ["10_jax_exploration_platform/scripts/atari_pixel_campaign/" + name for name in (
        "paired_benchmark.py", "train_optimized.py", "diagnose_gpu_execution.py")]
    protected = {}
    for name in numerical:
        if name.endswith(".py"):
            raw = (root / name).read_bytes()
            if raw != subprocess.check_output(["git", "-C", str(root), "show", "HEAD:" + name]):
                raise ValueError("a protected numerical source changed")
            protected[name] = hashlib.sha256(raw).hexdigest()
    value = {"kind": "group_runtime_closed_source_and_numerical_preservation", "passed": True,
        "job_id": os.environ["SLURM_JOB_ID"], "step_id": os.environ["SLURM_STEP_ID"], "helper_count": len(names),
        "helper_files": files, "protected_numerical_files": protected, "protected_numerical_file_count": len(protected),
        "stdlib_bootstrap_before_audit": "ctypes own-process Python API initialized before audit; every later library operation refused",
        "numerical_imports": [], "actual_tool_gpu_job_or_library_actions_after_audit": 0, "scientific_credit": 0}
    with receipt.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({"passed": True, "helper_count": len(names), "protected_numerical_file_count": len(protected),
        "step": value["step_id"], "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
