"""Review the claim adapter and reproduce only an incorrect CPU-step prefix check."""
import ast
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import time
import unittest

P = Path(__file__).resolve().parent
R = P.parent.parent
START = time.monotonic()


def load(name, path):
    """Load one exact metadata-only source without its main entry."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    """Hash one small actual ordinary input."""
    path = Path(path)
    assert path.is_file() and not path.is_symlink()
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Bind the held source and exact real CPU context before making a declared negative fixture.
assert os.environ["SLURM_JOB_ID"] == "6821437" and os.environ["SLURM_STEP_ID"].isdigit()
assert os.getresuid() == (2618919, 2618919, 2618919)
actual_step = os.environ["SLURM_STEP_ID"]
actual_cgroup = Path("/proc/self/cgroup").read_text()
assert "/job_6821437/step_" + actual_step + "/" in actual_cgroup and len(actual_step) > 1
freeze_path = P / "source_freeze.json"
assert sha(freeze_path) == "4af4489c32e798e91fc134b4b7b70aedc880b3f461ad3f46294c6d17d1e1ba2f"
freeze = json.loads(freeze_path.read_text())
w = Path(freeze["source_root"])
for relative, expected in freeze["files"].items():
    path = w / relative
    assert sha(path) == expected["sha256"] and oct(stat.S_IMODE(path.stat().st_mode)) == expected["mode"]
code = w / "10_jax_exploration_platform/scripts/atari_pixel_campaign"
old_plan_path = R / "data/2026-11-05-02-24_common-timing_cache-room-backgrounds-true_render-final-two-only-true_h100-nvl_serval02-card-1_copies-1_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_paired-rounds-7_updates-per-arm-10_seed-start-1/measurement_plan.json"
old_plan = json.loads(old_plan_path.read_text())
old_binding = old_plan_path.parent / "code/serval02_card1_binding.py"
old = ast.parse(old_binding.read_text())
new = ast.parse((code / "serval02_card1_binding.py").read_text())
old_functions = {node.name: node for node in old.body if isinstance(node, ast.FunctionDef)}
new_functions = {node.name: node for node in new.body if isinstance(node, ast.FunctionDef)}
assert set(old_functions) == set(new_functions)
assert {name for name in old_functions if ast.dump(old_functions[name], include_attributes=False) != ast.dump(new_functions[name], include_attributes=False)} == {"current_admission_claim"}
legacy = old_functions["current_admission_claim"]
changed = new_functions["current_admission_claim"]
branch = next(node for node in changed.body if isinstance(node, ast.If))
assert [ast.dump(node, include_attributes=False) for node in branch.orelse] == [ast.dump(node, include_attributes=False) for node in legacy.body[2:-1]]
assert ast.dump(changed.body[-1], include_attributes=False) == ast.dump(legacy.body[-1], include_attributes=False)
unchanged_helpers = []
for relative, expected in old_plan["helpers"]["files"].items():
    if relative.endswith("/serval02_card1_binding.py"):
        continue
    assert sha(w / relative) == expected
    unchanged_helpers.append(relative)
assert len(unchanged_helpers) == 28

# Run the six focused contract methods independently; no original raw case or learner is rerun.
tests = load("independent_claim_contract_cases", code / "tests/test_current_admission_snapshot.py")
stream = io.StringIO()
result = unittest.TextTestRunner(stream=stream).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
assert result.wasSuccessful() and result.testsRun == 6

# Invoke the actual probe with one deliberately false prefix step while the kernel context stays exact.
probe = load("independent_probe_step_prefix_negative", code / "probe_direct_current_admission.py")
fixture_path = P / "helper_receipt_only_fixture_plan.json"
fixture_hash = sha(fixture_path)
forged_step = actual_step[:-1]
os.environ["SLURM_STEP_ID"] = forged_step
try:
    evidence = probe.probe(fixture_path, fixture_hash, fixture=True, canonical_source=False)
finally:
    os.environ["SLURM_STEP_ID"] = actual_step
assert evidence["cpu_step"]["step"] == forged_step and evidence["cpu_step"]["cgroup"] == actual_cgroup
assert "/step_" + forged_step + "/" not in actual_cgroup

# Independently compare the actual new claim with the genuine unchanged legacy qualification claim.
core = load("independent_claim_legacy_comparison", code / "serval02_card1_binding.py")
qualified = R / "data/2026-11-01-20-53_original-qualification_h100-nvl_serval02-card-1_driver-590.48.01_gpu-count-1_processor-count-8_policies-1_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_reference-updates-1_seed-1_memory-mib-95830/measurement_plan.json"
legacy_claim = core.current_admission_claim(json.loads(qualified.read_text()))
assert evidence["claim"] == legacy_claim and evidence["claim"]["frozen_admission"]["sha256"] == "4a71ef2b245ae784933c55bb2b5159aa1bb7f8f626eb94b15390e2f9a304ee7c"
for relative, expected in freeze["files"].items():
    assert sha(w / relative) == expected["sha256"]
audit = {"schema_version": 1, "kind": "independent_claim_source_pass_and_probe_exact_context_block", "claim_source_passed": True,
    "probe_source_approved": False, "full_source_approved": False, "actual_owned_cpu_job": "6821437", "actual_step_id": actual_step,
    "actual_kernel_cgroup": actual_cgroup, "declared_false_prefix_step": forged_step, "probe_wrongly_accepted_prefix_context": True,
    "focused_methods_passed": 6, "focused_output": stream.getvalue(), "binding_functions_changed": ["current_admission_claim"],
    "legacy_branch_and_return_ast_unchanged": True, "other28helpers_unchanged": unchanged_helpers,
    "actual_new_claim_equals_genuine_legacy_claim": True, "claim": evidence["claim"], "invalid_context_probe_output": evidence,
    "source_freeze": {"path": str(freeze_path), "sha256": sha(freeze_path)}, "required_probe_fix": "Match exact /job_<id>/ and /step_<id>/ cgroup boundaries; prefix IDs must refuse before source/claim work.",
    "actual_new_prepared_unit_checked": False, "helper_receipt_fixture_only": True, "canonical_live_source_checked_by_reviewer": False,
    "gpu_target_or_numerical_or_remote_action": False, "scientific_credit": 0, "elapsed_seconds": time.monotonic() - START}
out = P / "independent_claim_source_and_probe_context_block.json"
with out.open("x") as handle:
    json.dump(audit, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
print(json.dumps({"claim_source_passed": True, "probe_source_approved": False, "prefix_gap_reproduced": True, "receipt": str(out), "sha256": sha(out), "actual_step_id": actual_step, "elapsed_seconds": audit["elapsed_seconds"]}))
