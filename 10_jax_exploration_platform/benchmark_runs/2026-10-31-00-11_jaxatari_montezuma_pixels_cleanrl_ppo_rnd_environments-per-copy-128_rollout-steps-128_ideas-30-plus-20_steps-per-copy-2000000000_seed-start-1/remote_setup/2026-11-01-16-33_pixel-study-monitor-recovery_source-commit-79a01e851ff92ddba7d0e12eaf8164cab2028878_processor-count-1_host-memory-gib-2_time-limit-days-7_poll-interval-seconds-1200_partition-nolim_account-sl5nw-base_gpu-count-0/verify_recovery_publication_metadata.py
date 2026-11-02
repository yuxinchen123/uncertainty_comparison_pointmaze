"""Check actual recovery approval and mocked published-object reads; no live command."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

P = Path(__file__).absolute().parent
spec = importlib.util.spec_from_file_location("recovery_submission_publication_metadata", P / "submit_monitor_held.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
source_before = hashlib.sha256((P / "submit_monitor_held.py").read_bytes()).hexdigest()
inputs, plan = module.validate_inputs()
repository = Path(inputs["repository"])
seen = []
def object_bytes(command):
    assert command[:4] == ["git", "-C", str(repository), "show"]
    relative = command[4].split(":", 1)[1]
    seen.append(relative)
    return (repository / relative).read_bytes()
with patch.object(module.subprocess, "run") as ancestor, patch.object(module.subprocess, "check_output", side_effect=object_bytes):
    module.validate_publication(inputs, "a" * 40)
    ancestor.assert_called_once_with(["git", "-C", str(repository), "merge-base", "--is-ancestor",
        "a" * 40, "origin/merge-atari-cleanrl"], check=True)
assert len(seen) == len(inputs["required_files"]) + 5
with patch.object(module.subprocess, "run"), patch.object(module.subprocess, "check_output", return_value=b"private incorrect object"):
    try:
        module.validate_publication(inputs, "a" * 40)
    except ValueError:
        pass
    else:
        raise AssertionError("changed publication bytes were accepted")
assert source_before == hashlib.sha256((P / "submit_monitor_held.py").read_bytes()).hexdigest()
assert not any(name.split(".")[0] in {"jax", "jaxlib", "numpy", "torch", "flax", "optax"} for name in sys.modules)
value = {"schema_version": 1, "kind": "recovery_actual_root_approval_and_mock_publication_metadata_verification",
    "passed": True, "metadata_host": "portal05", "metadata_allocation": None, "source_sha256": source_before,
    "actual_root_approval_sha256": hashlib.sha256((P / "root_recovery_approval.json").read_bytes()).hexdigest(),
    "actual_static_inputs_checked": len(inputs["required_files"]), "mocked_publication_objects_checked": len(seen),
    "changed_object_rejected": True, "real_git_publication_claimed": False, "live_commands": 0,
    "actual_full_campaign_report": False, "prelaunch_full_candidate_available": False, "new_jobs": 0,
    "scientific_credit": 0}
raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
target = P / "held_submission_root_approval_and_publication_metadata.json"
with target.open("xb") as stream:
    stream.write(raw); stream.flush(); os.fsync(stream.fileno())
descriptor = os.open(P, os.O_DIRECTORY); os.fsync(descriptor); os.close(descriptor)
print(json.dumps({"receipt": str(target), "sha256": hashlib.sha256(raw).hexdigest(),
                  "passed": True, "objects": len(seen)}))

