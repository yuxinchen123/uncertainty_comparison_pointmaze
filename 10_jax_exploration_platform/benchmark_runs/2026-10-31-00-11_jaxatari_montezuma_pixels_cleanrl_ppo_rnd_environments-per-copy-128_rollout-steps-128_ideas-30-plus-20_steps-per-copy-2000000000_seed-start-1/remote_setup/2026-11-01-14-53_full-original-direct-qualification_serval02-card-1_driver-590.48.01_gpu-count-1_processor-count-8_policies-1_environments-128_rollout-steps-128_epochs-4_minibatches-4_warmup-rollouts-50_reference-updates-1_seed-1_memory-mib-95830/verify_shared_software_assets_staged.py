"""Verify actual shared software/assets and staged files; perform no numerical or device work."""
from datetime import datetime, timezone
from importlib import metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

P = Path(__file__).resolve(strict=True).parent
specification = importlib.util.spec_from_file_location("direct_metadata_preparation_guard", P / "prepare_direct_qualification_metadata.py")
guard = importlib.util.module_from_spec(specification); specification.loader.exec_module(guard)
guard.require(os.environ.get("SLURM_JOB_ID") == "6782816" and os.environ.get("SLURM_STEP_ID", "").isdecimal(),
    "only the handed-over owned monitor CPU may verify metadata")
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
    direct.verify_qualification_source_receipts(plan)
    assets = training.verify_assets(plan["asset_record"], {})
    reference = guard.read(plan["qualification_software"]["installed_backend"]["path"])
    actual = selected.installed_backend_receipt(training.jaxtari_python_receipt(plan["jaxtari_source_root"]),
        training.installed_gpu_library_receipts())
    guard.require(actual == reference, "actual current installed software differs from the exact completed shared-software reference")
    guard.require(assets == guard.read(plan["qualification_software"]["asset_receipt"]["path"]) and assets["file_count"] == 4531,
        "the exact original 4531-file asset tree differs")
    torch_version = metadata.version("torch")
    guard.require(torch_version == "2.6.0+cpu", "the original CPU Torch reference distribution changed")
    staged = {path: guard.receipt(unit / "code" / Path(path).name)["sha256"] for path in plan["helpers"]["files"]}
    guard.require(staged == plan["helpers"]["files"] and len(staged) == 25, "the exact conditional staged helper closure changed")
    guard.require(not any((unit / name).exists() for name in ("direct_current_launch.json", "direct_device_binding.json",
        "direct_process_metadata.json", "direct_terminal_process.json", "result.json", "direct_qualification_result.json")),
        "metadata preparation must retain unresolved actual execution")
    raw = Path("/proc/self/stat").read_text()
    value = {"schema_version": 1, "kind": "actual_direct_qualification_shared_software_asset_and_staged_metadata_verification",
        "passed": True, "at_utc": datetime.now(timezone.utc).isoformat(), "job_id": "6782816",
        "step_id": os.environ["SLURM_STEP_ID"], "pid": os.getpid(), "birth": raw.rsplit(")", 1)[1].split()[19],
        "cgroup": Path("/proc/self/cgroup").read_text(), "python": sys.executable, "python_version": sys.version,
        "plan": plan_ref, "helper_files": staged, "packages": actual["packages"], "torch_version": torch_version,
        "jaxtari_source": actual["jaxtari_source"], "installed_library_images": actual["gpu_libraries"], "assets": assets,
        "numerical_tests_run": False, "target_gpu_runtime_verified": False, "server_contact": False, "standing_lock_probed": False,
        "device_opened": False, "worker_admitted": False, "qualification_credit": 0, "scientific_credit": 0}
    path = P / "shared_software_assets_staged_verification.json"; guard.write(path, value)
    print(json.dumps({"passed": True, "record": guard.receipt(path), "helper_files": len(staged),
        "asset_files": assets["file_count"], "library_images": len(actual["gpu_libraries"])}), flush=True)
except BaseException as error:
    guard.write(P / "shared_software_assets_staged_failure.json", {"error_type": type(error).__name__,
        "error": str(error), "traceback": traceback.format_exc(), "job_id": "6782816",
        "step_id": os.environ["SLURM_STEP_ID"], "server_contact": False, "device_opened": False})
    raise

