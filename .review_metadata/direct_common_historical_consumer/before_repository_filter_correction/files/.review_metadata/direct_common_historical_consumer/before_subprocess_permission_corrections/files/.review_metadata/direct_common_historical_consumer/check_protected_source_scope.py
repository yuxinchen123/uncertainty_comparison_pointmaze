"""Compare real old/current metadata syntax and all thirteen frozen numerical sources."""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "10_jax_exploration_platform/scripts/atari_pixel_campaign"
sys.path.insert(0, str(SCRIPTS))
import direct_common_confirmation as common


def digest(path):
    """Hash only a declared small source file for this source-scope fixture."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    """Reject any numerical or original execution change while declaring the exact metadata addition."""
    if os.environ.get("SLURM_JOB_ID") != "6783003" or not os.environ.get("SLURM_STEP_ID", "").isdigit():
        raise RuntimeError("source-scope checks need this exact owned numeric CPU step")
    snapshot_path = Path(__file__).with_name("protected_numerical_source_snapshot.json")
    expected = json.loads(snapshot_path.read_bytes())
    if len(expected) != 13:
        raise RuntimeError("all thirteen declared numerical files are required")
    old_fixture = json.loads(Path(__file__).with_name("actual_old_twenty_six_source_fixture_inputs.json").read_bytes())
    old_code = Path(old_fixture["staged_code"])
    frozen_root = Path("/p/rlprojects/RND_worktrees/atari-cleanrl-source-d28e673")
    for relative, wanted in expected.items():
        if digest(ROOT / relative) != wanted or digest(frozen_root / relative) != wanted:
            raise RuntimeError("a protected numerical file changed: " + relative)
    scopes = {"measure_selected.py": ("_compute", "run_prepared", "save_measurement_result", "producer_support", "card_admission"),
        "direct_measurement.py": ("run_qualification", "support", "capture_backend", "capture_backend_completion",
            "post_binding_checks", "process_sample", "validate_window")}
    unchanged = {}
    for filename, names in scopes.items():
        def syntax(path):
            """Extract exact syntax without importing old or current producer execution code."""
            return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(path.read_bytes()).body
                if isinstance(node, ast.FunctionDef)}
        old, current = syntax(old_code / filename), syntax(SCRIPTS / filename)
        for name in names:
            if old.get(name) != current.get(name) or name not in old:
                raise RuntimeError("an original execution function changed: " + filename + ":" + name)
            unchanged[filename + ":" + name] = hashlib.sha256(old[name].encode()).hexdigest()
    extension = common.qualification_reader_extension(old_code / "direct_measurement.py", SCRIPTS / "direct_measurement.py")
    import direct_historical_consumer as historical
    print(json.dumps({"schema_version": 1, "kind": "exact_original_numerical_and_execution_source_comparison",
        "passed": True, "actual_step_id": os.environ["SLURM_STEP_ID"], "actual_process": historical.process(os.getpid()),
        "protected_numerical_files": expected, "unchanged_execution_function_syntax_sha256": unchanged,
        "qualification_reader_metadata_extension": extension, "entire_reader_ast_unchanged": False,
        "numerical_imports": [], "gpu_operations": 0, "scientific_credit": 0,
        "saved_scientific_evidence_recomputed": False, "qualification_established": False,
        "admission_granted": False}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
