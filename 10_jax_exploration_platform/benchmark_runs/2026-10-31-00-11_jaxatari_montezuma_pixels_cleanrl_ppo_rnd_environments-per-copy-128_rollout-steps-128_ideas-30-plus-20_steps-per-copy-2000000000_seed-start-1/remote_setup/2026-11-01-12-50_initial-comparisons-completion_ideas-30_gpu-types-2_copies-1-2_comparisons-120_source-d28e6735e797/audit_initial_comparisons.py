"""Recheck every original comparison on owned CPU without starting experiments."""
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

RUN = Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1")
OUTPUT = Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-12-50_initial-comparisons-completion_ideas-30_gpu-types-2_copies-1-2_comparisons-120_source-d28e6735e797")
SOURCE = "d28e6735e79788f618acb0ebfde8609db1f776af"
HELPER_ROOT = Path("/p/rlprojects/RND_worktrees/2026-11-01-12-36_full-original-qualification-source_commit-43eade68238a8d010ad1d1ea681e348022a00060")
HELPER_COMMIT = "43eade68238a8d010ad1d1ea681e348022a00060"
PREFIX = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"
def require(condition, message):
    if not condition:
        raise ValueError(message)
def digest(raw):
    return hashlib.sha256(raw).hexdigest()
def publish(name, value):
    with (OUTPUT / name).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(OUTPUT, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
def main():
    require(os.environ.get("SLURM_JOB_ID") == "6781231" and os.environ.get("SLURM_STEP_ID", "").isdigit(),
            "use this campaign's actual owned numeric CPU step")
    require(subprocess.check_output(["git", "-C", str(HELPER_ROOT), "rev-parse", "HEAD"], text=True).strip() == HELPER_COMMIT,
            "the published helper commit changed")
    require(not subprocess.check_output(["git", "-C", str(HELPER_ROOT), "status", "--porcelain", "--untracked-files=no"], text=True).strip(),
            "the detached helper has changed tracked bytes")
    helper_files = {}
    for name in ("post_comparison_validation.py", "ideas.py"):
        relative = PREFIX + name
        raw = (HELPER_ROOT / relative).read_bytes()
        require(raw == subprocess.check_output(["git", "-C", str(HELPER_ROOT), "show", HELPER_COMMIT + ":" + relative]),
                "helper bytes differ from the published Git blob")
        helper_files[relative] = digest(raw)
    require(digest((RUN / "SOURCE_AND_GATES_READY.json").read_bytes()) ==
            "4a71ef2b245ae784933c55bb2b5159aa1bb7f8f626eb94b15390e2f9a304ee7c", "original admission bytes changed")
    sys.path.insert(0, str(HELPER_ROOT / PREFIX))
    import post_comparison_validation as validation
    import ideas
    registry_path = RUN / "study_registry.json"
    registry_raw = registry_path.read_bytes()
    registry = json.loads(registry_raw)
    require(registry["reference_commit"] == SOURCE and len(registry["initial_ideas"]) == 30
            and registry["initial_matched_copy_counts"] == [1, 2], "original cohort identity changed")
    evidence = validation.Evidence()
    rows, decisions, contexts, common = [], collections.Counter(), collections.Counter(), []
    for entry in registry["initial_ideas"]:
        entry_rows = []
        for context in validation.CONTEXTS:
            for copies in (1, 2):
                summaries = [row for row in entry["results"][context] if row["copies"] == copies]
                require(len(summaries) == 1, "an original hardware/copy case is missing or duplicated")
                summary = summaries[0]
                path = summary["result_path"]
                validation.validate_cohort_case(path, entry["name"], context, copies, SOURCE, evidence)
                result = evidence.read(path)
                if result.get("failure_stage") == "correctness":
                    decision = "rejected"
                else:
                    actual = ideas.summarize_pairs(result["rounds"], correctness_passed=result["correctness_passed"],
                                                   contaminated=result["contaminated"])
                    for key, value in actual.items():
                        require(result.get(key) == value, "original paired decision no longer reconstructs: " + key)
                    decision = actual["decision"]
                require(summary["decision"] == decision, "registry decision differs from actual original evidence")
                item = {"name": entry["name"], "hardware_class": context, "copies": copies,
                        "result_path": path, "result_sha256": digest(Path(path).read_bytes()),
                        "decision": decision, "correctness_passed": result.get("correctness_passed"),
                        "median_speedup": result.get("median_speedup")}
                rows.append(item)
                entry_rows.append(item)
                decisions[decision] += 1
                contexts[context] += 1
        if all(row["decision"] == "accepted" for row in entry_rows):
            common.append(entry["name"])
    require(len(rows) == 120 and dict(contexts) == {"h100": 60, "jaguar03_rtx_a4500": 60},
            "the original cohort is incomplete")
    require(registry_path.read_bytes() == registry_raw, "the original registry changed during validation")
    result = {"schema_version": 1, "kind": "initial_thirty_raw_completion_audit", "passed": True,
              "at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "owned_cpu_job": "6781231",
              "step_id": os.environ["SLURM_STEP_ID"], "pid": os.getpid(), "source_commit": SOURCE,
              "helper_commit": HELPER_COMMIT, "helper_files": helper_files,
              "script_sha256": digest(Path(__file__).read_bytes()), "registry_sha256": digest(registry_raw),
              "comparisons": len(rows), "hardware_counts": dict(contexts), "decision_counts": dict(decisions),
              "individually_accepted_on_both_gpus_and_both_copy_counts": common,
              "combined_configuration_correctness_credit": 0, "adaptive_ideas_selected": 0,
              "new_case_credit": 0, "new_idea_credit": 0, "rows": rows, "raw_evidence": evidence.records()}
    publish("initial_comparisons_audit.json", result)
    print(json.dumps({key: result[key] for key in ("passed", "step_id", "comparisons", "hardware_counts", "decision_counts",
        "individually_accepted_on_both_gpus_and_both_copy_counts")}), flush=True)
if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        publish("initial_comparisons_audit_failure.json", {"schema_version": 1, "kind": "initial_thirty_raw_completion_audit_failure",
            "at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "job_id": os.environ.get("SLURM_JOB_ID"),
            "step_id": os.environ.get("SLURM_STEP_ID"), "error": repr(error), "traceback": traceback.format_exc()})
        raise
