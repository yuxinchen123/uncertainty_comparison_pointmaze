"""Run one affected size contract and the six actual small controller-stream joins only."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
STEP = os.environ["SLURM_STEP_ID"]
STARTED = time.monotonic()
FORBIDDEN = {"jax", "jaxlib", "numpy", "torch", "flax", "optax", "scipy", "cupy", "jaxtari", "exploration_platform"}
assert os.environ.get("SLURM_JOB_ID") == "6783003" and STEP.isdecimal() and os.getresuid() == (2618919, 2618919, 2618919)
assert Path("/proc/self/cgroup").read_text().splitlines() == ["0::/system.slice/slurmstepd.scope/job_6783003/step_" + STEP + "/user/task_0"]


def audit(event, values):
    if event == "import" and values[0].split(".")[0] in FORBIDDEN:
        raise RuntimeError("no numerical import")
    if event in {"socket.connect", "socket.bind", "socket.getaddrinfo", "ctypes.dlopen", "ctypes.dlsym", "os.system", "subprocess.Popen"}:
        raise RuntimeError("no network, library or tool operation")
    if event == "open" and isinstance(values[0], (str, bytes)) and os.fsdecode(values[0]).startswith(("/dev/nvidia", "/dev/dri", "/tmp/sl5nw_serval02_card1.lock")):
        raise RuntimeError("no GPU device or standing-lock access")


sys.addaudithook(audit)
test_path = ROOT / "10_jax_exploration_platform/scripts/atari_pixel_campaign/tests/test_accepted_evidence_controller_size.py"
specification = importlib.util.spec_from_file_location("affected_controller_size_tests", test_path)
module = importlib.util.module_from_spec(specification); specification.loader.exec_module(module)
snapshot = module.snapshot
capture = io.StringIO(); result = unittest.TextTestRunner(stream=capture, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
classification = json.loads((Path(__file__).with_name("actual_reference_classification_432.json")).read_bytes())
authorities = {row["references"][0]["document"]["path"]: row["references"][0]["document"] for row in classification["all_unresolved"]
    if Path(row["original_path"]).name in {"direct_controller_stdout.raw", "direct_controller_stderr.raw"}}
assert len(authorities) == 3
actual = []
with snapshot.ValidationFiles(150) as files:
    inventory = snapshot.AcceptedInventory(files)
    for authority in authorities.values():
        outputs = files.json(authority); unit = Path(authority["path"]).parent
        inventory.accepted_units[unit] = {"exact_hashed_actual_output_authority": authority}
        for stream in ("stdout", "stderr"):
            value = {key: outputs[stream][key] for key in ("path", "sha256")}
            inventory.add(value, metadata=False, shared_metadata_role="accepted_controller_" + stream,
                shared_declaration={"authority": authority, "reference_field": [stream]})
            entry = files.file(value["path"], value["sha256"], keep_bytes=True, expected_size_bytes=outputs[stream]["size_bytes"])
            actual.append({"reference": value, "authority": authority, "field": stream,
                "declared_size_bytes": outputs[stream]["size_bytes"], "actual_pinned_bytes": entry["bytes"],
                "actual_pinned_stat_size": entry["identity"]["st_size"]})
    files.finish(files.paths)
    metadata_bytes = files.bytes_hashed
source = test_path.parents[1] / "accepted_evidence_snapshot.py"
receipt = {"schema_version": 1, "kind": "affected_controller_size_contract_and_actual_six_streams", "job_id": "6783003", "step_id": STEP,
    "affected_methods": result.testsRun, "passed": result.wasSuccessful(), "failures": len(result.failures), "errors": len(result.errors),
    "elapsed_seconds": time.monotonic() - STARTED, "actual_controller_stream_joins": actual, "actual_small_metadata_bytes_hashed": metadata_bytes,
    "candidate_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "test_source_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
    "old_five_scope_controls_repeated": False, "full_142_resolution_repeated": False, "large_payload_files_opened": 0,
    "preaudit_operations": "Standard-library imports and exact own CPU-step/process checks only.", "numerical_imports": sorted(FORBIDDEN.intersection(sys.modules)),
    "gpu_operations": 0, "network_operations": 0, "scientific_credit": 0}
for suffix, raw in ((".json", (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()), (".stdout.raw", capture.getvalue().encode())):
    with Path(__file__).with_name("controller_size_checks_" + STEP + suffix).open("xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
print(json.dumps({key: value for key, value in receipt.items() if key != "actual_controller_stream_joins"}, sort_keys=True))
if not result.wasSuccessful():
    print(capture.getvalue(), file=sys.stderr)
raise SystemExit(0 if result.wasSuccessful() else 1)
