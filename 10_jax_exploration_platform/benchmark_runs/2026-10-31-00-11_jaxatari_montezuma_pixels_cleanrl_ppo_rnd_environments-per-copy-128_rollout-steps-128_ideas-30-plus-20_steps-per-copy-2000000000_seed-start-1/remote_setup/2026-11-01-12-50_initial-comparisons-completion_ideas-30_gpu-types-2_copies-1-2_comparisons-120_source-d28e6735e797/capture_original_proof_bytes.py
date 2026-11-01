"""Preserve physically copied original proof bytes before later legitimate appends."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

RUN = Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1")
ROOT = Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-12-50_initial-comparisons-completion_ideas-30_gpu-types-2_copies-1-2_comparisons-120_source-d28e6735e797")
EXIT_PLAN = Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-12-53_original-worker-exit-proof_comparisons-120_owned-uva-numerical-jobs-25_serval02-supervisor-3567981/exit_observation_plan.json")
EXIT_OBSERVATION = Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-12-53_original-worker-exit-proof_comparisons-120_owned-uva-numerical-jobs-25_serval02-supervisor-3567981/observation_2026-11-01-20-54")
def digest(raw):
    return hashlib.sha256(raw).hexdigest()
def save(path, raw):
    with path.open("xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    path.chmod(0o444)
def main():
    if os.environ.get("SLURM_JOB_ID") != "6781231":
        raise ValueError("preserve evidence only on the exact owned CPU allocation")
    audit = json.loads((ROOT / "initial_comparisons_audit.json").read_text())
    if audit["comparisons"] != 120 or audit["passed"] is not True:
        raise ValueError("actual complete initial audit is required")
    raw_registry = (RUN / "study_registry.json").read_bytes()
    if digest(raw_registry) != audit["registry_sha256"]:
        raise ValueError("the complete original registry changed before physical preservation")
    target = ROOT / "frozen_original_evidence"
    target.mkdir()
    content = target / "content"; content.mkdir()
    seen, mappings, total = set(), [], 0
    def capture(logical, expected, raw=None):
        nonlocal total
        destination = content / (expected + ".raw")
        if expected not in seen:
            if raw is None:
                value = hashlib.sha256()
                with Path(logical).open("rb") as source, destination.open("xb") as output:
                    while True:
                        block = source.read(1048576)
                        if not block: break
                        value.update(block); output.write(block); total += len(block)
                    output.flush(); os.fsync(output.fileno())
                if value.hexdigest() != expected:
                    raise ValueError("original evidence changed before physical capture: " + logical)
                destination.chmod(0o444)
            else:
                if digest(raw) != expected: raise ValueError("frozen original byte identity differs")
                save(destination, raw); total += len(raw)
            seen.add(expected)
        mappings.append({"logical_path": logical, "captured_path": str(destination), "sha256": expected,
                         "bytes": destination.stat().st_size})
    for row in audit["raw_evidence"]:
        capture(row["path"], row["sha256"])
    capture(str(RUN / "study_registry.json"), digest(raw_registry), raw_registry)
    exit_raw = EXIT_PLAN.read_bytes(); exit_plan = json.loads(exit_raw)
    for relative, row in exit_plan["inputs"]["files"].items():
        capture(str(RUN / relative), row["sha256"], row["text"].encode())
    for source in [ROOT / "audit_initial_comparisons.py", ROOT / "initial_comparisons_audit.json", EXIT_PLAN,
                   EXIT_OBSERVATION / "worker_stop.json", EXIT_OBSERVATION / "observations.json", EXIT_OBSERVATION / "status.json",
                   RUN / "SOURCE_AND_GATES_READY.json"]:
        raw = source.read_bytes(); capture(str(source), digest(raw), raw)
    result = {"schema_version": 1, "kind": "physical_original_comparison_evidence_capture", "passed": True,
              "at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "owned_cpu_job": "6781231",
              "step_id": os.environ["SLURM_STEP_ID"], "audit_sha256": digest((ROOT / "initial_comparisons_audit.json").read_bytes()),
              "original_registry_sha256": audit["registry_sha256"], "exit_plan_sha256": digest(exit_raw),
              "logical_mappings": mappings, "unique_payloads": len(seen), "physical_payload_bytes": total,
              "new_scientific_credit": 0, "source_plan_modified": False,
              "note": "Owner records use exact bytes captured in the immutable actual exit plan; later live appends cannot change these copies."}
    save(target / "capture_manifest.json", (json.dumps(result, indent=2) + "\n").encode())
    for directory in (content, target):
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)
    print(json.dumps({key: result[key] for key in ("passed", "step_id", "unique_payloads", "physical_payload_bytes")}), flush=True)
if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        path = ROOT / "physical_capture_failure.json"
        with path.open("x") as stream:
            json.dump({"error": repr(error), "step_id": os.environ.get("SLURM_STEP_ID"), "traceback": traceback.format_exc()}, stream, indent=2)
            stream.flush(); os.fsync(stream.fileno())
        raise
