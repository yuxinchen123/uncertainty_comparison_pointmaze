"""Wait only for an absent final launch envelope, then validate it once before any GPU work."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import sys
import time
import traceback

MAX_WAIT_SECONDS = 180
EXPECTED_MODE = "correctness"
EXPECTED_QUALIFICATION_INPUTS = {'result': {'path': '/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/data/2026-11-01-13-50_original-qualification_explicit-pytest-root_jaguar03_gpu-count-1_processor-count-8_host-memory-gib-64_time-limit-hours-3_copies-1_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_reference-updates-1_seed-1/target_qualification_result.json', 'sha256': '0c4f94c2d58fdd7277a261481d7a52c4bb5b19e42612a8eae67dff6890c62cd1'}, 'review': {'path': '/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-14-30-02_full-original-qualification-stage-and-terminal-review_job-6782778_gpu-count-0_processor-count-1/terminal_full_review/independent_actual_review.json', 'sha256': '004fcae1c5f24dec23da5add7452486c2ed5c1fcec89bf853617791a36fee74a'}}
EXPECTED_PLAN_INPUTS = {'configuration': {'path': '/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-15-09_common_cache-room-backgrounds-true_render-final-two-only-true_correctness_jaguar03_copies-2_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_seed-start-1/candidate.json', 'sha256': '6a8a2c0726d37279fb8a4539175ea841168730a443ab4d192da6fcea5ac66141'}, 'reference_configuration': {'path': '/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-15-09_common_cache-room-backgrounds-true_render-final-two-only-true_correctness_jaguar03_copies-2_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_seed-start-1/reference.json', 'sha256': 'b9d95a0817157ba1a4210cfb9f58f26d780604210e3b3779729e441767b19d86'}, 'runtime_environment': {'path': '/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-15-09_common_cache-room-backgrounds-true_render-final-two-only-true_correctness_jaguar03_copies-2_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_seed-start-1/runtime.json', 'sha256': 'ab2f12e8d463f2de1d6da487cf3dbdceda1012672a0311e25532add5e7ceb03c'}, 'asset_record': {'path': '/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-15-09_common_cache-room-backgrounds-true_render-final-two-only-true_correctness_jaguar03_copies-2_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_seed-start-1/assets.json', 'sha256': '342ee71b36df5f804fc20588fccb2f6b414a6ff06b6fe97470b379ad6abf5427'}, 'configured_isolation_profile': {'path': '/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-15-09_common_cache-room-backgrounds-true_render-final-two-only-true_correctness_jaguar03_copies-2_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_seed-start-1/profile.json', 'sha256': '97ab4dd8ca59c974603ae6a4a8899007471e485fe34952d229ff2d3d84a2ab50'}}
PREFIX = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"
NUMERICAL_ROOTS = {"jax", "jaxlib", "numpy", "scipy", "torch", "flax", "optax", "jaxtari", "pynvml"}

def install_metadata_guard():
    """Check actual module state and forbid numerical, real library, network and child-process work."""
    require(not any(name.split(".")[0] in NUMERICAL_ROOTS for name in sys.modules), "numerical module already loaded before startup gate")
    observations = {"numerical_import_attempts": [], "forbidden_operation_attempts": []}
    def guard(event, args):
        if event == "import" and args[0].split(".")[0] in NUMERICAL_ROOTS:
            observations["numerical_import_attempts"].append(args[0])
            raise RuntimeError("numerical import forbidden in startup gate")
        if event in {"subprocess.Popen", "socket.connect", "os.system"} or (event == "ctypes.dlopen" and args[0] is not None):
            observations["forbidden_operation_attempts"].append(event)
            raise RuntimeError("process, library, device or network work forbidden in startup gate")
    sys.addaudithook(guard)
    return observations

def require(condition, message):
    """Refuse changed identity or incomplete actual release evidence."""
    if not condition:
        raise ValueError(message)

def read_json(path):
    """Read one exact ordinary JSON file and reject duplicate object keys."""
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file() and not path.is_symlink(), "launch input must be an exact ordinary file")
    def unique(pairs):
        """Do not choose one of two conflicting JSON values."""
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate launch input key")
            result[key] = value
        return result
    raw = path.read_bytes()
    return json.loads(raw, object_pairs_hook=unique), hashlib.sha256(raw).hexdigest()

def publish(path, value):
    """Keep startup success/failure evidence immutable and durable."""
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())

def validate_once(unit, expected_plan, expected_helper, current_job, validator=None):
    """Join final inputs to their exact pre-release envelope and staged committed source once."""
    plan, digest = read_json(unit / "measurement_plan.json")
    ready, unused = read_json(unit / "measurement_prepared.json")
    require(digest == expected_plan and ready == {"plan_sha256": expected_plan}
        and plan["helpers"]["source_commit"] == expected_helper and plan["gpu_uuid"] is None
        and plan.get("kind") == "selected_configuration_measurement" and plan.get("mode") == EXPECTED_MODE
        and "target_qualification" not in plan and "final_confirmation_protocol" not in plan and "completed_study" not in plan
        and plan.get("target_qualification_inputs") == EXPECTED_QUALIFICATION_INPUTS
        and plan.get("inputs") == EXPECTED_PLAN_INPUTS, "frozen common-candidate correctness identity changed")
    for name, field in (("configuration", "configuration"), ("reference_configuration", "reference_configuration"), ("runtime_environment", "runtime_environment"), ("asset_record", "asset_record"), ("configured_isolation_profile", None)):
        record = EXPECTED_PLAN_INPUTS[name]
        value, actual_sha = read_json(record["path"])
        require(actual_sha == record["sha256"] and value == (plan[field] if field is not None else plan["configured_isolation"]["profile"]), "a full planned configuration or runtime input changed")
    for name, record in EXPECTED_QUALIFICATION_INPUTS.items():
        unused, actual_sha = read_json(record["path"])
        require(actual_sha == record["sha256"], "the actual full-original qualification or independent review changed")
    pre, pre_sha = read_json(unit / "configured_isolation_pre_release_inputs.json")
    final, final_sha = read_json(unit / "owned_allocation_inputs.json")
    require(type(pre.get("schema_version")) is int and pre["schema_version"] == 1
        and type(final.get("schema_version")) is int and final["schema_version"] == 1
        and pre.get("kind") == "uva_configured_isolation_pre_release_inputs"
        and final.get("kind") == "uva_configured_isolation_owned_inputs"
        and pre.get("job_id") == final.get("job_id") == current_job
        and final.get("plan") == {"path": str(unit / "measurement_plan.json"), "sha256": expected_plan}, "actual owned launch identity differs")
    require(set(pre) == {"schema_version", "kind", "job_id", "plan", "ledger", "submissions", "submission", "submission_row"}
        and set(final) == set(pre) | {"release"}, "unexpected pre-release or final envelope fields")
    expected_final = {**pre, "kind": "uva_configured_isolation_owned_inputs", "release": final["release"]}
    require(final == expected_final, "post-release inputs changed a pre-release binding")
    files = plan["helpers"]["files"]
    require(files and PREFIX + "uva_slurm_configured_isolation.py" in files, "configured identity helper is absent")
    for relative, expected_sha in files.items():
        require(relative.startswith(PREFIX) and Path(relative).name == relative.removeprefix(PREFIX), "staged source escaped its exact helper directory")
        path = unit / "code" / Path(relative).name
        require(path.resolve(strict=True) == path and path.is_file() and not path.is_symlink()
            and hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha, "a staged helper changed")
    if validator is None:
        sys.path.insert(0, str(unit / "code"))
        path = unit / "code/uva_slurm_configured_isolation.py"
        specification = importlib.util.spec_from_file_location("qualification_owned_inputs_gate", path)
        core = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(core)
        require(Path(core.__file__).resolve() == path, "startup imported another qualifier")
        observed = core.owned_inputs(plan)
    else:
        observed = validator(plan)
    require(observed == final, "strict owned launch input reconstruction differs")
    return {"plan_sha256": expected_plan, "helper_commit": expected_helper, "job_id": current_job,
        "pre_release_inputs_sha256": pre_sha, "final_inputs_sha256": final_sha,
        "actual_release_receipt": final["release"], "gpu_operations": 0, "numerical_imports": 0}

def wait_for_inputs(unit, expected_plan, expected_helper, current_job, clock=time.monotonic, pause=time.sleep, validator=None):
    """Wait only on final-file absence; any present partial or invalid binding fails immediately."""
    require(re.fullmatch(r"[0-9a-f]{64}", expected_plan) and re.fullmatch(r"[0-9a-f]{40}", expected_helper)
        and re.fullmatch(r"[1-9][0-9]*", current_job), "actual job and frozen source hashes are required")
    started = clock()
    polls = 0
    while not os.path.lexists(unit / "owned_allocation_inputs.json"):
        elapsed = clock() - started
        if elapsed >= MAX_WAIT_SECONDS:
            raise TimeoutError("actual post-release inputs were absent for the full180-second metadata barrier")
        pause(min(1, MAX_WAIT_SECONDS - elapsed)); polls += 1
    result = validate_once(unit, expected_plan, expected_helper, current_job, validator)
    require(clock() - started <= MAX_WAIT_SECONDS, "metadata startup exceeded its180-second budget")
    return {"schema_version": 1, "kind": "common_candidate_correctness_owned_input_startup_gate",
        "waited_seconds": clock() - started, "absence_polls": polls, "maximum_wait_seconds": MAX_WAIT_SECONDS,
        "wait_policy": "absent final file only; present invalid evidence is never retried", **result}

def main():
    """Enforce the real180-second budget and preserve any failed startup before numerical srun."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--helper-commit", required=True)
    arguments = parser.parse_args()
    unit = arguments.unit.absolute()
    job = os.environ.get("SLURM_JOB_ID", "")
    started = time.time()
    observations = install_metadata_guard()
    def timeout(signum, frame):
        """Interrupt a delayed metadata read without starting any GPU operation."""
        raise TimeoutError("metadata startup barrier exceeded180 seconds")
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, MAX_WAIT_SECONDS)
    try:
        result = wait_for_inputs(unit, arguments.plan_sha256, arguments.helper_commit, job)
        require(not any(name.split(".")[0] in NUMERICAL_ROOTS for name in sys.modules), "numerical module appeared during startup gate")
        result.update(started_at_seconds=started, finished_at_seconds=time.time(),
            interpreter_flags={"isolated": sys.flags.isolated, "no_site": sys.flags.no_site, "dont_write_bytecode": sys.dont_write_bytecode},
            imported_module_names=sorted(sys.modules), observed_actions=observations)
        require(sys.flags.isolated == 1 and sys.flags.no_site == 1 and sys.dont_write_bytecode is True,
            "startup gate requires an isolated no-site no-bytecode stdlib interpreter")
        publish(unit / "owned_inputs_wait_completed.json", result)
    except BaseException as error:
        publish(unit / "owned_inputs_wait_failure.json", {"schema_version": 1,
            "kind": "common_candidate_correctness_owned_input_startup_failure", "job_id": job,
            "plan_sha256": arguments.plan_sha256, "helper_commit": arguments.helper_commit,
            "maximum_wait_seconds": MAX_WAIT_SECONDS, "started_at_seconds": started,
            "finished_at_seconds": time.time(), "error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc(), "gpu_operations": 0, "numerical_imports": 0, "observed_actions": observations})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

if __name__ == "__main__":
    main()
