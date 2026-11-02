"""Independently check frozen direct-controller metadata with private CPU children only."""
import ast
from datetime import datetime, timezone
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import socket
import stat
import sys
import time
import unittest
from unittest import mock

PACKET = Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-17-17_original-qualification_h100-nvl_serval02-card-1_driver-590.48.01_gpu-count-1_processor-count-8_policies-1_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_reference-updates-1_seed-1_memory-mib-95830")
REVIEW = Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-17-54-39_fresh-original-h100-qualification-independent-review_gpu-count-0")
RUN = Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1")
OLD_PACKET = next(path for path in (RUN / "remote_setup").iterdir()
    if path.name.startswith("2026-11-01-14-53_full-original-direct-qualification_"))
EXPECTED = {
    "packet_metadata_freeze.json": "f7b544a5f5fcf8fb2ad6b1c974d8eb553042c92291a74580921a223047c08d87",
    "controller_source_freeze.json": "fcd15589d215e4730d42bca66c84ab5d389a202bede040ee06a79ee9eb07ee3d",
    "complete_publication_roster.json": "515a65bf7e037fccf6a3bcf624c2107e4b1137d7e2fe3acd42644b4c5b8f3f3c",
    "direct_qualification_controller.py": "8596beef9d5be09e88fff0d0c0f2f85717f2695b4442bce751c2a7b2858c25e5",
    "test_direct_qualification_controller.py": "76384eda5778578e12e613115892575aacc39cea3888e0254c21b4647a03b406",
}
BLOCKED = {"jax", "jaxlib", "numpy", "torch", "cupy", "flax", "optax", "scipy", "jaxtari", "exploration_platform"}

def sha(path):
    """Hash only exact reviewed local bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    """Save one new durable independent receipt without replacement."""
    with Path(path).open("xb") as stream:
        stream.write((json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())

def definitions(path):
    """Return complete named function and class ASTs without importing the source."""
    tree = ast.parse(Path(path).read_text())
    return {node.name: ast.dump(node, include_attributes=False) for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}

def main():
    """Run frozen private-process tests and independent static/publication checks."""
    # Check exact owned CPU placement before loading the frozen test module.
    assert os.environ.get("SLURM_JOB_ID") == "6783003"
    step = os.environ["SLURM_STEP_ID"]
    assert step.isdigit() and os.getuid() == 2618919
    assert "/job_6783003/step_" + step + "/" in Path("/proc/self/cgroup").read_text()
    for name in ("submitted_jobids_uva.txt", "submitted_jobids.txt"):
        assert "6783003" in (RUN / "slurm" / name).read_text().split()
    assert any(line.startswith("6783003\tmonitor\t79a01e851ff92ddba7d0e12eaf8164cab2028878\t")
        for line in (RUN / "slurm/submissions.tsv").read_text().splitlines())
    print("ACTUAL_OWNED_STEP=6783003." + step, flush=True)
    assert not BLOCKED & {name.split(".", 1)[0] for name in sys.modules}

    # Verify all three final rosters and preserve exact before/after bytes and modes.
    for name, expected in EXPECTED.items():
        assert sha(PACKET / name) == expected, name
    base = json.loads((PACKET / "packet_metadata_freeze.json").read_bytes())
    supplemental = json.loads((PACKET / "controller_source_freeze.json").read_bytes())
    roster = json.loads((PACKET / "complete_publication_roster.json").read_bytes())
    assert len(base["files"]) == 66 and len(supplemental["files"]) == 42
    assert len(roster["files"]) == 83
    protected = {str(PACKET / name): expected for name, expected in EXPECTED.items()}
    protected.update(base["files"])
    protected.update(supplemental["files"])
    for item in roster["files"]:
        path = Path(item["path"])
        assert sha(path) == item["sha256"] and path.stat().st_size == item["bytes"]
        assert format(stat.S_IMODE(path.stat().st_mode), "04o") == item["mode"]
        protected[str(path)] = item["sha256"]
    before = {name: sha(name) for name in protected}
    assert before == protected
    modes_before = {name: stat.S_IMODE(Path(name).stat().st_mode) for name in protected}

    # Refuse all numerical imports, host contact, dynamic GPU loading, and target lock access.
    class NoNumericalImports(importlib.abc.MetaPathFinder):
        """Reject numerical imports before module execution."""
        def find_spec(self, fullname, path=None, target=None):
            """Fail the test if a blocked module is requested."""
            if fullname.split(".", 1)[0] in BLOCKED:
                raise AssertionError("numerical import forbidden: " + fullname)
    sys.meta_path.insert(0, NoNumericalImports())
    actions = []
    def audit(event, arguments):
        """Permit only real private CPU fixture children, with no target access."""
        if event.startswith("socket.") and event != "socket.gethostname":
            raise AssertionError("network access forbidden")
        if event in {"ctypes.dlopen", "ctypes.dlsym"}:
            raise AssertionError("dynamic-library access forbidden")
        if event == "open":
            path = str(arguments[0])
            if path.startswith(("/dev/nvidia", "/dev/dri")) or path == "/tmp/sl5nw_serval02_card1.lock":
                raise AssertionError("target device or standing lock access forbidden")
        if event == "subprocess.Popen":
            executable, command, cwd, environment = arguments
            command = list(command)
            assert len(command) == 4 and command[0] == sys.executable
            assert command[2] == "--prepared-unit"
            assert str(command[1]).startswith("/tmp/direct-controller-private-cpu-")
            assert str(command[1]).endswith("/campaign/data/private-cpu-fixture/code/measure_selected.py")
            assert command[3] == str(Path(command[1]).parent.parent) == str(cwd)
            actions.append({"command": command, "cwd": str(cwd)})
    sys.addaudithook(audit)

    # Re-run the exact frozen twenty tests; every process is a disposable local CPU fixture.
    specification = importlib.util.spec_from_file_location("independent_direct_controller_tests",
        PACKET / "test_direct_qualification_controller.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    started = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.testsRun == 20 and result.wasSuccessful()

    # Independently check every old/new function body, not only the selected launch function.
    old_path = OLD_PACKET / "direct_qualification_controller.py"
    assert sha(old_path) == "f8c6a95d587d1ccf1b643622af63e034babc08742904106029d886a90fc70d70"
    old_defs = definitions(old_path)
    new_defs = definitions(PACKET / "direct_qualification_controller.py")
    assert len(new_defs) == 26 and old_defs == new_defs
    controller = module.controller
    assert controller.HELPER_COMMIT == "6066d9ea35fbc9c045855144550971c2ac5895f9"
    assert controller.PLAN_SHA256 == "515e4089bec5fa65931b5f08fc018d716209243fd8311cc0f4740f954769472e"
    assert controller.PACKET_FREEZE_SHA256 == EXPECTED["packet_metadata_freeze.json"]
    assert controller.PREPARATION_SHA256 == "e94b3ef28a4a748cd732bf79bd9981109dde0ce8c92dfcc4f0a6a8395665c5da"
    assert controller.SHARED_SOFTWARE_SHA256 == "bfd50821f4f861822e379f30ef0b1823bda11a990879ca255ca787cd11236cd1"

    # The actual absent review must fail before querying publication or spawning any target.
    assert not (PACKET / "controller_root_independent_review.json").exists()
    with mock.patch.object(controller.subprocess, "check_output",
            side_effect=AssertionError("missing-review case may not query publication")):
        try:
            controller.check_publication("a" * 40)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("actual packet accepted absent root approval")
    assert not any((controller.UNIT / name).exists() for name in controller.ATTEMPT_NAMES)

    # Retain complete fixture observations and verify all protected bytes after the tests.
    after = {name: sha(name) for name in protected}
    assert after == before
    assert modes_before == {name: stat.S_IMODE(Path(name).stat().st_mode) for name in protected}
    assert not BLOCKED & {name.split(".", 1)[0] for name in sys.modules}
    record = {
        "schema_version": 1, "kind": "fresh_original_h100_controller_independent_metadata_review",
        "at_utc": datetime.now(timezone.utc).isoformat(), "job_id": "6783003", "step_id": step,
        "pid": os.getpid(), "hostname": socket.gethostname(),
        "cgroup": Path("/proc/self/cgroup").read_text(), "affinity": sorted(os.sched_getaffinity(0)),
        "passed": True, "tests_run": result.testsRun, "additional_independent_checks": 3,
        "additional_checks": ["complete_66_and_42_freezes_and_83_publication_roster_modes",
            "all_26_function_class_asts_equal_with_exact_new_pins",
            "actual_absent_root_review_refused_before_git_query_and_unit_unattempted"],
        "elapsed_seconds": time.monotonic() - started, "protected_before": before, "protected_after": after,
        "protected_modes_before": modes_before, "protected_modes_after": modes_before,
        "source_expected": EXPECTED, "private_cpu_child_actions": actions,
        "private_cpu_fixture_results": module.ControllerTests.fixture_results,
        "all_26_function_class_asts_equal": True, "target_host_contacts": 0,
        "actual_target_process_started": False, "gpu_or_standing_lock_operations": 0,
        "numerical_modules_loaded": [], "worker_admitted": False, "admission_granted": False,
        "scientific_credit": 0, "case_credit": 0, "idea_credit": 0,
        "actual_gpu_qualification_still_required": True}
    output = REVIEW / ("controller_independent_6783003_" + step + ".json")
    write(output, record)
    print(json.dumps({"passed": True, "tests": 20, "additional_checks": 3,
        "receipt": str(output), "sha256": sha(output)}, sort_keys=True))

if __name__ == "__main__":
    main()

