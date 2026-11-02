"""Verify actual shared software/assets and staged files; perform no numerical or device work."""
from datetime import datetime, timezone
from importlib import metadata
import importlib.util
import json
import fcntl
import stat
import subprocess
import os
from pathlib import Path
import sys
import traceback

P = Path(__file__).resolve(strict=True).parent
specification = importlib.util.spec_from_file_location("direct_metadata_preparation_guard", P / "prepare_common_correctness.py")
guard = importlib.util.module_from_spec(specification); specification.loader.exec_module(guard)
guard.require(os.environ.get("SLURM_JOB_ID") == "6783003" and os.environ.get("SLURM_STEP_ID", "").isdecimal(),
    "only the handed-over owned monitor CPU may verify metadata")
# This is the owned campaign CPU-memory read lock, never the GPU standing lock.
request = guard.read(P / "preparation_request.json")
campaign = Path(request["campaign_run_dir"])
for name in ("submitted_jobids.txt", "submitted_jobids_uva.txt"):
    guard.require((campaign / "slurm" / name).read_text().splitlines().count("6783003") == 1, "owned CPU ID missing or duplicated")
command = ["scontrol", "show", "job", "-o", "6783003"]
observed = subprocess.run(command, capture_output=True, text=True)
guard.write(P / "software_verification_running_guard.json", {"command": command, "returncode": observed.returncode,
    "stdout": observed.stdout, "stderr": observed.stderr, "job_id": "6783003", "step_id": os.environ["SLURM_STEP_ID"]})
guard.require(os.getuid() == os.geteuid() == 2618919 and observed.returncode == 0 and observed.stderr == ""
    and "JobId=6783003 " in observed.stdout and "UserId=sl5nw(2618919) " in observed.stdout
    and "JobState=RUNNING " in observed.stdout, "fresh exact-owned CPU monitor is not running")
resource_lock = campaign / "queue/cpu_monitor_large_artifact_read.lock"
guard.require(resource_lock.parent.resolve(strict=True) == resource_lock.parent, "ordinary CPU read-lock parent required")
resource_fd = os.open(resource_lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
resource_stat = os.fstat(resource_fd)
guard.require(stat.S_ISREG(resource_stat.st_mode) and resource_stat.st_uid == 2618919, "owned ordinary CPU read lock required")
fcntl.flock(resource_fd, fcntl.LOCK_EX)
guard.install_guards()
try:
    result = guard.read(P / "metadata_preparation_result.json")
    plan_ref = result["plan"]; guard.require(guard.receipt(plan_ref["path"]) == plan_ref, "prepared plan bytes changed")
    plan = guard.read(plan_ref["path"]); unit = Path(plan_ref["path"]).parent
    scripts = Path(plan["helpers"]["source_root"]) / "10_jax_exploration_platform/scripts/atari_pixel_campaign"
    sys.path.insert(0, str(scripts))
    import direct_measurement as direct
    import measure_selected as selected
    import train_optimized as training
    direct.verify_saved_sources(unit, plan)
    qualified = guard.read(request["qualification_plan"]["path"])
    direct.verify_qualification_source_receipts(qualified)
    backend_package = direct.backend_support(plan).validate_prepared(plan)
    guard.require(guard.receipt(unit / "direct_backend_image_expected.json") == plan["direct_backend_image"]["expected"],
        "new exact prepared backend sidecar changed")
    assets = training.verify_assets(plan["asset_record"], {})
    reference = guard.read(Path(request["qualification_plan"]["path"]).parent / "installed_backend.json")
    actual = selected.installed_backend_receipt(training.jaxtari_python_receipt(plan["jaxtari_source_root"]),
        training.installed_gpu_library_receipts())
    guard.require(actual == reference, "actual current installed software differs from the exact completed shared-software reference")
    guard.require(assets == guard.read(Path(request["qualification_plan"]["path"]).parent / "asset_receipt.json") and assets["file_count"] == 4531,
        "the exact original 4531-file asset tree differs")
    torch_version = metadata.version("torch")
    guard.require(torch_version == "2.6.0+cpu", "the original CPU Torch reference distribution changed")
    staged = {path: guard.receipt(unit / "code" / Path(path).name)["sha256"] for path in plan["helpers"]["files"]}
    guard.require(staged == plan["helpers"]["files"] and len(staged) == 26, "the exact conditional staged helper closure changed")
    guard.require(not any((unit / name).exists() for name in ("direct_current_launch.json", "direct_device_binding.json",
        "direct_process_metadata.json", "direct_terminal_process.json", "result.json", "direct_qualification_result.json")),
        "metadata preparation must retain unresolved actual execution")
    raw = Path("/proc/self/stat").read_text()
    value = {"schema_version": 1, "kind": "actual_direct_common_correctness_shared_software_asset_and_staged_metadata_verification",
        "passed": True, "at_utc": datetime.now(timezone.utc).isoformat(), "job_id": "6783003",
        "step_id": os.environ["SLURM_STEP_ID"], "pid": os.getpid(), "birth": raw.rsplit(")", 1)[1].split()[19],
        "cgroup": Path("/proc/self/cgroup").read_text(), "python": sys.executable, "python_version": sys.version,
        "plan": plan_ref, "helper_files": staged, "packages": actual["packages"], "torch_version": torch_version,
        "backend_expected": plan["direct_backend_image"]["expected"], "actual_prepared_backend_package": backend_package,
        "jaxtari_source": actual["jaxtari_source"], "installed_library_images": actual["gpu_libraries"], "assets": assets,
        "numerical_tests_run": False, "target_gpu_runtime_verified": False, "server_contact": False, "standing_lock_probed": False,
        "device_opened": False, "worker_admitted": False, "qualification_credit": 0, "scientific_credit": 0}
    path = P / "shared_software_assets_staged_verification.json"; guard.write(path, value)
    print(json.dumps({"passed": True, "record": guard.receipt(path), "helper_files": len(staged),
        "asset_files": assets["file_count"], "library_images": len(actual["gpu_libraries"])}), flush=True)
except BaseException as error:
    guard.write(P / "shared_software_assets_staged_failure.json", {"error_type": type(error).__name__,
        "error": str(error), "traceback": traceback.format_exc(), "job_id": "6783003",
        "step_id": os.environ["SLURM_STEP_ID"], "server_contact": False, "device_opened": False})
    raise
finally:
    os.close(resource_fd)

