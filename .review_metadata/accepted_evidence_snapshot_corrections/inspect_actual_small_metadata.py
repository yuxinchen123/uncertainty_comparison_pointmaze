"""Inspect only genuine accepted small JSON schemas; never open their payload or device references."""
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
STARTED = time.monotonic()
assert os.environ.get("SLURM_JOB_ID") == "6783003"
STEP = os.environ["SLURM_STEP_ID"]
assert STEP.isdecimal() and Path("/proc/self/cgroup").read_text().splitlines() == ["0::/system.slice/slurmstepd.scope/job_6783003/step_" + STEP + "/user/task_0"]
assert os.getresuid() == (2618919, 2618919, 2618919)
READ = {}


def read(value):
    path = Path(value["path"])
    assert path.suffix == ".json" and path.stat().st_size <= 16 * 1024 * 1024
    if str(path) not in READ:
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == value["sha256"]
        READ[str(path)] = json.loads(raw)
    return READ[str(path)]


def references(value, field=""):
    result = []
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str) and Path(value["path"]).suffix != ".json":
            result.append({"field": field, "path": value["path"]})
        for key, item in value.items():
            result.extend(references(item, field + "/" + key))
    elif isinstance(value, list):
        for item in value:
            result.extend(references(item, field))
    return result


summary = json.loads((ROOT / ".review_metadata/accepted_evidence_snapshot_independent_review/independent_primary_review_summary.json").read_bytes())
rows = []
for item in summary["real_small_schema_inspection"]:
    review, result, plan = (read(item[key]) for key in ("review", "result", "plan"))
    raw = read(review["raw_audit"]) if "raw_audit" in review else None
    asset_input = read(plan["inputs"]["asset_record"])
    asset_ref = raw["asset_receipt"] if raw is not None else plan["qualification_software"]["asset_receipt"]
    asset = read(asset_ref)
    rows.append({"role": item["role"], "asset_input_reference": plan["inputs"]["asset_record"],
        "asset_input": asset_input, "plan_asset": plan["asset_record"], "asset_receipt_reference": asset_ref,
        "asset_receipt": asset, "asset_receipt_keys": sorted(asset),
        "non_json_references": references(review, "review") + references(result, "result") + references(plan, "plan") + references(raw, "raw")})
output = {"kind": "actual_small_accepted_asset_and_reference_schema_inspection", "job_id": "6783003", "step_id": STEP,
    "elapsed_seconds": time.monotonic() - STARTED, "rows": rows, "small_json_files_read": len(READ),
    "payload_bytes_opened": 0, "gpu_operations": 0, "scientific_credit": 0}
path = Path(__file__).with_name("actual_small_schema_" + STEP + ".json")
with path.open("x") as stream:
    json.dump(output, stream, indent=2, sort_keys=True); stream.write("\n")
print(json.dumps({"receipt": str(path), "step_id": STEP, "small_json_files_read": len(READ), "elapsed_seconds": output["elapsed_seconds"]}))
