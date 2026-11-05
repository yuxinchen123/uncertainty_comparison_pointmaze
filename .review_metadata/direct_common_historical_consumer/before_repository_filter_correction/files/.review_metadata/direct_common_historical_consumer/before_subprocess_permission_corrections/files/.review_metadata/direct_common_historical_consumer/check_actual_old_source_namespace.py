"""Import the real old twenty-six metadata sources without reading scientific result payloads."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "10_jax_exploration_platform/scripts/atari_pixel_campaign"
sys.path.insert(0, str(SCRIPTS))
import direct_historical_consumer as historical
import direct_common_confirmation as guard


def main():
    """Use exact staged/source/Git old bytes and retain their original module paths and globals."""
    if os.environ.get("SLURM_JOB_ID") != "6783003" or not os.environ.get("SLURM_STEP_ID", "").isdigit():
        raise RuntimeError("the real old source fixture needs the exact owned numeric CPU step")
    path = Path(__file__).with_name("actual_old_twenty_six_source_fixture_inputs.json")
    raw = path.read_bytes()
    fixture = historical.document(raw)
    helpers = fixture["helpers"]
    historical.require(len(helpers["files"]) == 26 and helpers["source_commit"] == historical.OLD_COMMIT,
        "only the actual original twenty-six-source fixture is allowed")
    staged = Path(fixture["staged_code"])
    old_root = Path(helpers["source_root"])
    current_plan = {"helpers": {"files": {**helpers["files"],
        historical.PREFIX + historical.NAME: "unexecuted current fixture",
        historical.PREFIX + historical.GUARD: "unexecuted current fixture"}}}
    sources = historical.old_closure(current_plan, staged.parent, {"helpers": helpers})
    exact_git = [["git", "-C", str(old_root), "show", historical.OLD_COMMIT + ":" + relative]
        for relative in helpers["files"]]
    bootstrap = guard.stdlib_ctypes_bootstrap()
    guard.validate_bootstrap_record(bootstrap)

    def audit(event, values):
        """Permit only the old read-only source checks, never another Python or device operation."""
        if event == "subprocess.Popen" and values[1] not in exact_git:
            raise RuntimeError("only the exact old source Git commands may run")
        historical.readonly_audit(guard, {str(old_root)}, event, values)

    sys.addaudithook(audit)
    loader = historical.install_old_modules(sources)
    for name, source in sources.items():
        module = sys.modules[name]
        historical.require(module.__file__ == source["path"], "old namespace used another staged path")
        historical.require(module.__dict__.get("__file__") == source["path"], "old globals changed their original file")
    support = sys.modules["direct_measurement"]
    for name in ("serval02_card1_binding", "configured_measurement", "post_comparison_validation"):
        historical.require(support.adjacent(name) is sys.modules[name], "old adjacent import replaced an exact old global")
    historical.require(len(loader.loaded) == 26 and not historical.NUMERICAL.intersection(sys.modules),
        "real old import fixture has incomplete old modules or a numerical import")
    historical.old_closure(current_plan, staged.parent, {"helpers": helpers})
    historical.require(path.read_bytes() == raw, "source fixture inputs changed")
    print(json.dumps({"schema_version": 1, "kind": fixture["kind"], "passed": True,
        "actual_step_id": os.environ["SLURM_STEP_ID"], "actual_process": historical.process(os.getpid()),
        "fixture_sha256": historical.digest(raw), "old_staged_files_actually_loaded": loader.loaded,
        "stdlib_ctypes_bootstrap": bootstrap, "later_ctypes_load_or_symbol_events": 0,
        "native_standard_library_extension_activity": "not counted as GPU or ctypes operations",
        "numerical_imports": [], "gpu_operations": 0, "scientific_credit": 0,
        "saved_scientific_evidence_recomputed": False, "qualification_established": False,
        "admission_granted": False}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
