"""Approve only the corrected exact CPU-context guard and reuse unchanged claim checks."""
import ast
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import stat
import sys
import time
import unittest
from unittest.mock import patch

P = Path(__file__).resolve().parent
R = P.parent.parent
START = time.monotonic()


def sha(path):
    """Bind one ordinary small source or review record to exact bytes."""
    path = Path(path)
    assert path.is_file() and not path.is_symlink()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path):
    """Name one exact immutable metadata result."""
    return {"path": str(path), "sha256": sha(path)}


def load(name, path):
    """Load frozen metadata source without executing its main entry."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Verify only the four new frozen paths and the preserved completed claim review.
assert os.environ.get("SLURM_JOB_ID") == "6821437" and os.environ.get("SLURM_STEP_ID", "").isdigit()
assert os.getresuid() == (2618919, 2618919, 2618919)
actual_step = os.environ["SLURM_STEP_ID"]
cgroup = Path("/proc/self/cgroup").read_text()
assert "/job_6821437/step_" + actual_step + "/" in cgroup and len(actual_step) > 1
freeze_path = P / "corrected_source_freeze.json"
assert sha(freeze_path) == "4f00b2e1a7196e56063b5e13934d1d52c0ef531d43838f1879da65fbe0450aec"
freeze = json.loads(freeze_path.read_text())
w = Path(freeze["source_root"])
for relative, expected in freeze["files"].items():
    path = w / relative
    raw = path.read_bytes()
    assert sha(path) == expected["sha256"] and oct(stat.S_IMODE(path.stat().st_mode)) == expected["mode"]
    assert hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() == expected["git_blob"]
prior_path = P / "independent_claim_source_and_probe_context_block.json"
assert sha(prior_path) == "d357d65ca1989d933004118aa76e7665dd369e3a6f63e2a24040a4820e8dbe2e"
prior = json.loads(prior_path.read_text())
assert prior["claim_source_passed"] is True and prior["focused_methods_passed"] == 6 and prior["actual_new_claim_equals_genuine_legacy_claim"] is True
assert prior["legacy_branch_and_return_ast_unchanged"] is True
code = w / "10_jax_exploration_platform/scripts/atari_pixel_campaign"
assert sha(code / "serval02_card1_binding.py") == "af72d3de3e84f7b8be08a73f90f5becf143b1d32c90083c1773b10343751361b"
old_probe_path = P / "preserved_probe_context_prefix_rejection/source/10_jax_exploration_platform/scripts/atari_pixel_campaign/probe_direct_current_admission.py"
old_functions = {node.name: node for node in ast.parse(old_probe_path.read_text()).body if isinstance(node, ast.FunctionDef)}
new_functions = {node.name: node for node in ast.parse((code / "probe_direct_current_admission.py").read_text()).body if isinstance(node, ast.FunctionDef)}
assert set(new_functions) - set(old_functions) == {"owned_step_path"}
assert {name for name in old_functions if ast.dump(old_functions[name], include_attributes=False) != ast.dump(new_functions[name], include_attributes=False)} == {"probe"}
old_calls = [node for node in ast.walk(old_functions["probe"]) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "require"]
new_calls = [node for node in ast.walk(new_functions["probe"]) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "require"]
assert len(old_calls) == len(new_calls)
assert sum(ast.dump(a, include_attributes=False) != ast.dump(b, include_attributes=False) for a, b in zip(old_calls, new_calls)) == 1

# Run only the three affected context methods, not the six already completed claim cases.
tests = load("independent_probe_exact_context_methods", code / "tests/test_direct_admission_probe_context.py")
stream = io.StringIO()
result = unittest.TextTestRunner(stream=stream).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
assert result.wasSuccessful() and result.testsRun == 3
probe = load("independent_corrected_probe_exact_context", code / "probe_direct_current_admission.py")
assert probe.owned_step_path(cgroup, "6821437", actual_step) is True
false_step = actual_step[:-1]
fixture = P / "helper_receipt_only_fixture_plan.json"
os.environ["SLURM_STEP_ID"] = false_step
refused = None
try:
    with patch.object(probe, "git", side_effect=AssertionError("invalid context must stop before Git or claim work")):
        try:
            probe.probe(fixture, sha(fixture), fixture=True, canonical_source=False)
        except ValueError as error:
            refused = str(error)
finally:
    os.environ["SLURM_STEP_ID"] = actual_step
assert refused == "the current CPU process must belong to the recorded owned step"
for relative, expected in freeze["files"].items():
    assert sha(w / relative) == expected["sha256"]
assert not {"jax", "jaxlib", "numpy", "torch", "flax", "optax", "cupy", "scipy", "jaxtari", "exploration_platform"}.intersection(sys.modules)

# Combine the unchanged claim pass with this narrow corrected-context pass without relabelling the old rejection.
review = {"schema_version": 1, "kind": "independent_shared_current_admission_claim_and_corrected_probe_source_review", "passed": True, "approved": True,
          "independent_of_author": True, "source_commit": freeze["head"], "source_tree": freeze["tree"], "source_freeze": ref(freeze_path),
          "central_claim_checks_reused": ref(prior_path), "central_claim_methods_repeated": False, "genuine_claim_or_large_payloads_replayed_after_context_fix": False,
          "corrected_context_methods": 3, "context_output": stream.getvalue(), "actual_real_prefix_context_rejected": True,
          "actual_step_id": actual_step, "declared_false_prefix_step": false_step, "exact_rejection": refused,
          "actual_owned_cpu_job_id": "6821437", "actual_cpu_process": {"pid": os.getpid(), "hostname": socket.gethostname(), "uids": list(os.getresuid()),
             "birth": Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19], "cgroup": cgroup},
          "new_claim_source_only_function_changed": "current_admission_claim", "legacy_claim_and_live_authority_functions_preserved": True,
          "accepted_snapshot_producer_unchanged": True, "other28measurementhelpers_unchanged": True,
          "source_fixture_is_actual_new_unit_proof": False, "future_actual_prepared_unit_probe_required": True,
          "required_before_target_launch": ["Publish the exact reviewed source and records.", "Prepare a new unattempted unit from the reviewed clean helper commit after exact failed-controller exit.", "Run --prepared-unit on that real plan and helper namespace, before binding/SSH; use its actual hash.", "Verify current canonical authority and current source, own host/UID/UUID/CUDA/standing descriptor through unchanged target gates.", "Never retry or rewrite the failed02-24unit."],
          "old_probe_rejection_preserved": True, "current_canonical_source_or_gpu_host_checked_by_reviewer": False,
          "numerical_gpu_target_remote_or_job_actions": False, "scientific_credit": 0, "worker_admission": False, "elapsed_seconds": time.monotonic() - START}
out = P / "independent_corrected_claim_and_probe_source_review.json"
with out.open("x") as handle:
    json.dump(review, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
print(json.dumps({"passed": True, "review": ref(out), "actual_cpu_step_id": actual_step, "context_methods": 3, "elapsed_seconds": review["elapsed_seconds"]}))
