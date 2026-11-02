"""Check raw direct qualification first, write its real review, then call the unchanged consumer."""
import argparse
import ast
import copy
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.abc
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import re
import subprocess
import sys
import traceback

ROOT = Path(__file__).absolute().parent
UNIT = '/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/data/2026-11-01-17-17_original-qualification_h100-nvl_serval02-card-1_driver-590.48.01_gpu-count-1_processor-count-8_policies-1_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_reference-updates-1_seed-1_memory-mib-95830'
UNIT = Path(UNIT)
SOURCE_SHA = 'beead126531ee254a7999adae2e2e0e7292f18154a2e697e8a22ce1e93a33294'
RAW_SHA = '96c8f15609dba406300f57b8dadaee47789a56acf72c1e4fa96b8ef35156a32d'
PACKET = Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-17-17_original-qualification_h100-nvl_serval02-card-1_driver-590.48.01_gpu-count-1_processor-count-8_policies-1_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_reference-updates-1_seed-1_memory-mib-95830')
PLAN_SHA = '515e4089bec5fa65931b5f08fc018d716209243fd8311cc0f4740f954769472e'
HELPER_COMMIT = '6066d9ea35fbc9c045855144550971c2ac5895f9'
CONTROLLER_SHA = '8596beef9d5be09e88fff0d0c0f2f85717f2695b4442bce751c2a7b2858c25e5'
HISTORY_PREPARATION_SHA = '3da2c4491a0c9c123b7de529e0e25c916e7654bb0fd535fcb0169edde49860e0'
CONTROLLER_PUBLICATION = '691e5aab7ccc03d25a31e9a3bf1239361b97ee49'
LAUNCH_COMMAND_PUBLICATION = 'a13047dd11dcfe0d9b23528eab18df2f8388a899'
AUDIT_JOB_ID = '6783003'
NUMERICAL = {"jax", "jaxlib", "numpy", "torch", "flax", "optax", "jaxtari", "cupy", "scipy"}


def require(value, message):
    """Refuse an incomplete or different actual evidence chain."""
    if not value:
        raise ValueError(message)


def digest(path):
    """Hash one exact ordinary artifact without following a different path."""
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path and stat.S_ISREG(path.lstat().st_mode),
        "an exact ordinary evidence artifact is required")
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def ref(path):
    """Bind a saved artifact to its actual current bytes."""
    return {"path": str(path), "sha256": digest(path)}


def doc(path):
    """Read exact JSON while refusing duplicate keys."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate raw-review metadata key")
            result[key] = value
        return result
    digest(path)
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique)


def put(path, value):
    """Write new durable review evidence without replacing an earlier attempt."""
    path = Path(path)
    require(path.parent.resolve(strict=True) == path.parent, "ordinary review output parent required")
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return ref(path)


def verify_source_contract():
    """Prove every raw consumer statement remains, including the three final direct gates."""
    require(digest(UNIT / "measurement_plan.json") == PLAN_SHA
        and doc(UNIT / "measurement_plan.json")["helpers"]["source_commit"] == HELPER_COMMIT
        and digest(PACKET / "direct_qualification_controller.py") == CONTROLLER_SHA
        and digest(PACKET / "history_preparation_inputs.json") == HISTORY_PREPARATION_SHA,
        "the new qualification plan, helper, controller or frozen history changed")
    contract = doc(ROOT / "raw_review_source_contract.json")
    require(digest(contract["staged_source"]["path"]) == contract["staged_source"]["sha256"] == SOURCE_SHA
        and digest(contract["new_raw_reader"]["path"]) == contract["new_raw_reader"]["sha256"] == RAW_SHA,
        "the exact staged source or raw review source changed")
    original = next(node for node in ast.parse(Path(contract["staged_source"]["path"]).read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "verify_qualification_inputs")
    raw = next(node for node in ast.parse(Path(contract["new_raw_reader"]["path"]).read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "verify_unreviewed_actual_raw_qualification")
    removed = contract["removed_statement_indices"]
    require(len(removed) == 2 and removed == [4, 47]
        and [ast.unparse(original.body[index]) for index in removed] == contract["removed_statements"]
        and contract["removed_statements"][0] == "review = identity.read_receipt(inputs['review'])"
        and "independent actual qualification review is required" in contract["removed_statements"][1],
        "only the existing review load and review approval predicate may be removed")
    expected = [ast.dump(node, include_attributes=False) for index, node in enumerate(original.body) if index not in removed]
    observed = [ast.dump(node, include_attributes=False) for node in raw.body]
    require(expected == observed and len(original.body) == 52 and len(raw.body) == 50
        and ast.dump(original.args, include_attributes=False) == ast.dump(raw.args, include_attributes=False)
        and hashlib.sha256("\n".join(observed).encode()).hexdigest() == contract["retained_body_ast_sha256"],
        "a raw qualification statement was added, omitted or changed")
    final_calls = [node.value.func.id for node in raw.body[-4:-1]
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)]
    require(final_calls == ["validate_startup_checks", "validate_backend_capture", "terminal_process"],
        "the startup, captured backend or terminal-process gate is missing")
    pinned_history_git_roots(doc(UNIT / "measurement_plan.json"))
    return contract



def canonical_original(value):
    """Require the same canonical original POSIX identity used by the staged resolver."""
    require(type(value) is str and value.startswith("/") and not value.startswith("//")
        and "\0" not in value and ".." not in PurePosixPath(value).parts
        and str(PurePosixPath(value)) == value, "invalid original historical Git identity")
    return value


def pinned_history_git_roots(plan):
    """Derive only the eight ordinary staged Git paths from the exact pinned manifest."""
    inputs_ref = plan["direct_binding"]["frozen_inputs"]
    require(type(inputs_ref) is dict and set(inputs_ref) == {"path", "sha256"}
        and inputs_ref["path"] == str(UNIT / "direct_history/inputs.json")
        and digest(inputs_ref["path"]) == inputs_ref["sha256"], "frozen historical inputs changed")
    inputs = doc(inputs_ref["path"])
    mapped = inputs["evidence_manifest"]
    require(type(mapped) is dict and set(mapped) == {"path", "sha256"}
        and mapped["path"] == str(UNIT / "direct_history/evidence_manifest.json")
        and digest(mapped["path"]) == mapped["sha256"], "pinned historical manifest changed")
    require(digest(PACKET / "history_preparation_inputs.json") == HISTORY_PREPARATION_SHA,
        "the exact eight original historical source declarations changed")
    declared = doc(PACKET / "history_preparation_inputs.json")["git_sources"]
    require(type(declared) is list and len(declared) == 8, "the source declaration must contain exactly eight roots")
    expected = {}
    for row in declared:
        require(type(row) is dict and set(row) == {"original_root", "commit", "transport_repository"},
            "unsupported original historical source declaration")
        original = canonical_original(row["original_root"])
        canonical_original(row["transport_repository"])
        require(original not in expected and type(row["commit"]) is str
            and re.fullmatch(r"[0-9a-f]{40}", row["commit"]), "duplicate or invalid historical source identity")
        expected[original] = row["commit"]
    manifest = doc(mapped["path"])
    require(type(manifest) is dict and set(manifest) == {"schema_version", "kind", "files", "git_roots"}
        and type(manifest["schema_version"]) is int and manifest["schema_version"] == 1
        and manifest["kind"] == "portable_raw_evidence" and type(manifest["files"]) is list
        and type(manifest["git_roots"]) is list and len(manifest["git_roots"]) == 8,
        "the exact version-one eight-root staged evidence manifest is required")
    staging = Path(mapped["path"]).parent
    require(staging.resolve(strict=True) == staging and staging.is_dir(), "ordinary historical staging root required")
    observed, targets = {}, set()
    for row in manifest["git_roots"]:
        require(type(row) is dict and set(row) == {"original_root", "relative_path", "commit"},
            "unsupported staged historical Git declaration")
        original = canonical_original(row["original_root"])
        relative = row["relative_path"]
        require(type(relative) is str and relative and "\0" not in relative
            and not PurePosixPath(relative).is_absolute() and relative != "."
            and ".." not in PurePosixPath(relative).parts and str(PurePosixPath(relative)) == relative,
            "historical Git target must be one canonical relative path")
        target = staging / relative
        require(target.resolve(strict=True) == target and target.is_relative_to(staging)
            and stat.S_ISDIR(target.lstat().st_mode), "historical Git target must be an ordinary staged directory")
        require(original not in observed and str(target) not in targets
            and original in expected and type(row["commit"]) is str and row["commit"] == expected[original],
            "duplicate, unlisted or changed historical source mapping")
        observed[original] = row["commit"]
        targets.add(str(target))
    require(observed == expected and len(targets) == 8, "the complete exact staged historical Git roster is required")
    return targets


def allowed_git_command(command, roots):
    """Permit only the existing read-only operations on the exact admitted root strings."""
    return (isinstance(command, (list, tuple)) and len(command) > 3
        and list(command[:2]) == ["git", "-C"] and type(command[2]) is str and command[2] in roots
        and command[3] in {"rev-parse", "status", "show", "ls-tree", "diff", "diff-files",
            "diff-index", "cat-file", "merge-base", "symbolic-ref"})

def extra_actual_joins(direct, plan, result, metadata, binding):
    """Join actual source labels, work, controller outcome and finite ordered execution times."""
    identity = direct.core()
    process = metadata["after"]["identity"]
    execution = result["execution_identity"]
    require(all(identity.exact(execution.get(key), process[key]) for key in ("hostname", "pid", "process_start_time")),
        "actual execution identity changed")
    for key in ("started_at_seconds", "finished_at_seconds"):
        require(type(execution.get(key)) in {int, float} and math.isfinite(execution[key]), "actual execution time is not finite")
    require(execution["started_at_seconds"] > 0 and execution["finished_at_seconds"] >= execution["started_at_seconds"],
        "actual execution times are not ordered")
    require(identity.exact(result["copy_seeds"], plan["copy_seeds"])
        and result["measurement_mode"] == plan["mode"]
        and identity.exact(result["protocol"], {"rounds": plan["rounds"],
            "requested_minimum_updates_per_round": plan["minimum_updates_per_round"], "actual_updates_per_round": plan["iterations"]}),
        "actual independent seeds or saved preparation protocol changed")
    for key, expected in direct.BOUNDARY.items():
        require(identity.exact(result.get(key), expected), "actual result changed a zero-credit or unverified-trust field")
    for key, planned in (("candidate", "configuration"), ("configuration", "configuration"), ("reference", "reference_configuration")):
        if key in result:
            require(identity.exact(result[key], plan[planned]), "an actual result configuration differs from its full original plan")
    spawn = doc(UNIT / "direct_controller_spawn.json")
    child = spawn["observed_child"]
    require(type(spawn["schema_version"]) is int and spawn["schema_version"] == 1
        and spawn["kind"] == "actual_direct_qualification_popen_return"
        and all(child[key] == process[key] for key in ("hostname", "pid", "process_start_time", "parent_pid"))
        and spawn["popen_pid"] == process["pid"] and spawn["controller"]["pid"] == process["parent_pid"],
        "actual terminal child differs from its original controller spawn")
    launch = doc(UNIT / "direct_current_launch.json")
    require(launch["plan"] == ref(UNIT / "measurement_plan.json")
        and launch["process_row"]["pid"] == process["pid"]
        and launch["process_row"]["start_time"] == process["process_start_time"]
        and launch["process_row"]["helper_commit"] == HELPER_COMMIT,
        "the actual launch row does not name this qualified helper and child")
    wait = doc(UNIT / "direct_process_wait_return.json")
    parent_before = wait["waiter_before"]["stat"].rsplit(")", 1)[1].split()
    parent_after = wait["waiter_after"]["stat"].rsplit(")", 1)[1].split()
    require(spawn["controller"]["process_start_time"] == parent_before[19] == parent_after[19]
        and type(spawn["cpu_affinity_is_exclusive"]) is bool and not spawn["cpu_affinity_is_exclusive"]
        and len(child["affinity"]) == 8 and child["affinity"] == spawn["controller"]["affinity"],
        "the actual controller birth or eight nonexclusive CPU selectors changed")
    launch_command = doc(PACKET / "actual_launch_command_preparation.json")
    require(launch_command["command"][-3:] == ["--launch", "--publication-commit", CONTROLLER_PUBLICATION]
        and launch_command["host"] == "serval02" and launch_command["automatic_retry_permitted"] is False
        and spawn["controller"]["command"] == launch_command["command"][8:],
        "the actual observed controller argv differs from the reviewed launch")
    transport = doc(PACKET / "root_remote_launch_return.json")
    intent = doc(PACKET / "root_remote_launch_intent.json")
    require(transport["returncode"] == 0 and transport["reported_controller_pid"] == process["parent_pid"]
        and intent["published_commit"] == LAUNCH_COMMAND_PUBLICATION
        and intent["command_preparation_sha256"] == digest(PACKET / "actual_launch_command_preparation.json"),
        "the real transport return or published exact command changed")
    outcome = doc(UNIT / "direct_controller_outcome.json")
    require(type(outcome["schema_version"]) is int and outcome["schema_version"] == 1
        and outcome["kind"] == "direct_qualification_controller_outcome" and outcome["passed"] is True
        and outcome["actual_child_started"] is True and outcome["actual_wait_saved"] is True
        and outcome["failure"] is None and outcome["qualification_independently_verified"] is False,
        "the original controller has not finished its successful own-child wait")
    outputs = doc(UNIT / "direct_controller_output_hashes.json")
    require(outputs["kind"] == "actual_direct_child_output_hashes_after_wait"
        and outputs["wait"] == ref(UNIT / "direct_process_wait_return.json"), "actual controller output wait binding changed")
    for key, filename in (("stdout", "direct_controller_stdout.raw"), ("stderr", "direct_controller_stderr.raw")):
        path = UNIT / filename
        require(outputs[key] == {**ref(path), "size_bytes": path.stat().st_size}, "actual controller output bytes changed after wait")
    return {"execution_identity": execution, "copy_seeds": result["copy_seeds"], "controller_spawn": ref(UNIT / "direct_controller_spawn.json"),
        "controller_outcome": ref(UNIT / "direct_controller_outcome.json"), "controller_outputs": ref(UNIT / "direct_controller_output_hashes.json"),
        "native_cuda_ordinal": binding["native"].get("device_ordinal"), "actual_gpu_uuid": binding["gpu_uuid"],
        "kernel_device_enforcement_verified": False, "cpu_affinity_exclusive": False,
        "test_fixture_transition_total_known": False, "controlled_warmup_transitions": 819200,
        "controlled_reference_transitions": 16384, "controlled_reference_updates": 1,
        "controlled_optimizer_minibatch_updates": 16}


def main(argv=None):
    """Default validates source only; actual terminal review requires a separately approved owned CPU step."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-terminal-review", action="store_true")
    parser.add_argument("--audit-job-id")
    arguments = parser.parse_args(argv)
    verify_source_contract()
    if not arguments.execute_terminal_review:
        print(json.dumps({"source_contract_passed": True, "raw_statements_retained": 50,
            "scheduler_queries": 0, "actual_result_reads": 0, "independent_review_written": False,
            "formal_consumer_called": False})); return
    require(arguments.audit_job_id == AUDIT_JOB_ID and arguments.audit_job_id.isdecimal()
        and os.environ.get("SLURM_JOB_ID") == arguments.audit_job_id
        and os.environ.get("SLURM_STEP_ID", "").isdecimal() and os.getuid() == os.geteuid() == 2618919
        and os.getresuid() == (2618919, 2618919, 2618919),
        "actual review requires the separately verified current owned CPU monitor step")
    result_path = UNIT / "direct_qualification_result.json"
    terminal_path = UNIT / "direct_terminal_process.json"
    # An unfinished producer creates no terminal audit attempt or failure record.
    require(result_path.is_file() and terminal_path.is_file() and (UNIT / "direct_controller_outcome.json").is_file(),
        "actual qualification and successful own-controller terminal evidence do not yet exist; no audit attempted")
    plan = doc(UNIT / "measurement_plan.json")
    terminal = doc(terminal_path)
    wait = doc(terminal["return_record"]["path"])
    require(type(wait.get("returncode")) is int and wait["returncode"] == 0
        and wait.get("exited_normally") is True and doc(UNIT / "direct_controller_outcome.json").get("passed") is True
        and not any((UNIT / name).exists() for name in ("failure.json", "direct_controller_failure.json", "direct_controller_timeout_observation.json")),
        "actual direct qualification is not successfully terminal; no success review attempted")
    output = ROOT / "terminal_full_review"
    output.mkdir()
    blocked = []
    class NoNumerical(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in NUMERICAL:
                blocked.append(name); raise RuntimeError("numerical import forbidden")
    sys.meta_path.insert(0, NoNumerical())
    ctypes.CDLL = ctypes.PyDLL = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("shared-library load forbidden"))
    sys.addaudithook(lambda event, args: (_ for _ in ()).throw(RuntimeError("network forbidden"))
        if event in {"socket.connect", "socket.bind", "socket.getaddrinfo"} else None)
    history_sources = doc(PACKET / "history_preparation_inputs.json")["git_sources"]
    allowed_roots = {plan["source_root"], plan["helpers"]["source_root"], plan["paired_source"]["source_root"],
        "/p/rlprojects/RND/08_cleanrl_ppo_rnd/cleanrl", "/p/rlprojects/RND/reference_repos/jaxtari_pixel_study"}
    allowed_roots.update(row["original_root"] for row in history_sources)
    allowed_roots.update(row["transport_repository"] for row in history_sources)
    allowed_roots.update(pinned_history_git_roots(plan))
    real_popen = subprocess.Popen
    class ReadOnlyGit(real_popen):
        def __init__(self, command, *args, **kwargs):
            require(allowed_git_command(command, allowed_roots) and not kwargs.get("shell", False),
                "only exact read-only local source queries are allowed")
            super().__init__(command, *args, **kwargs)
    subprocess.Popen = ReadOnlyGit
    try:
        for relative, expected in plan["helpers"]["files"].items():
            require(digest(UNIT / "code" / Path(relative).name) == expected, "the staged actual helper closure changed")
        sys.path.insert(0, str(UNIT / "code"))
        import direct_measurement as direct
        require(Path(direct.__file__).absolute() == UNIT / "code/direct_measurement.py", "the raw reader imported an unrelated helper")
        specification = importlib.util.spec_from_file_location("raw_actual_direct_qualification", ROOT / "raw_direct_qualification_checks.py")
        raw = importlib.util.module_from_spec(specification); specification.loader.exec_module(raw)
        result_ref = ref(result_path)
        original = doc(result_path)
        consumer = copy.deepcopy(plan)
        consumer["direct_qualification_inputs"] = {"result": result_ref}
        # All fifty raw statements run before there is any review argument or approval file.
        validated = raw.verify_unreviewed_actual_raw_qualification(consumer)
        require(validated == original and not blocked, "the actual raw validation changed its result")
        metadata = doc(UNIT / "direct_process_metadata.json")
        binding = doc(UNIT / "direct_device_binding.json")
        extra = extra_actual_joins(direct, plan, original, metadata, binding)
        extra_ref = put(output / "extra_actual_joins.json", extra)
        process = metadata["after"]["identity"]
        review = {"schema_version": 1, "kind": "independent_full_original_direct_qualification_review",
            "approved": True, "at_utc": datetime.now(timezone.utc).isoformat(), "result": result_ref,
            "independent_of_producer": True, "terminal_process": ref(terminal_path), "actual_returncode": 0,
            **{key: process[key] for key in ("hostname", "pid", "process_start_time")},
            "audit_job_id": arguments.audit_job_id, "audit_step_id": os.environ["SLURM_STEP_ID"],
            "source_contract": ref(ROOT / "raw_review_source_contract.json"),
            "raw_verifier": ref(ROOT / "raw_direct_qualification_checks.py"), "extra_actual_joins": extra_ref,
            "raw_validation_passed_before_review_existed": True, "placeholder_or_partial_approval_used": False,
            "scientific_credit": 0, "case_credit": 0, "idea_credit": 0, "final_training_credit": 0,
            "site_qualification_credit": 0, "admission_granted": False, "worker_admitted": False}
        review_ref = put(output / "independent_actual_review.json", review)
        consumer["direct_qualification_inputs"]["review"] = review_ref
        final = direct.verify_qualification_inputs(consumer)
        require(final == original and ref(result_path) == result_ref, "the unchanged full consumer or actual result differed")
        proof = {"schema_version": 1, "kind": "actual_direct_full_original_raw_review_then_unmodified_consumer_verification",
            "passed": True, "result": result_ref, "independent_review": review_ref,
            "source_contract": ref(ROOT / "raw_review_source_contract.json"), "extra_actual_joins": extra_ref,
            "audit_job_id": arguments.audit_job_id, "audit_step_id": os.environ["SLURM_STEP_ID"],
            "raw_checks_before_review": True, "unchanged_full_consumer_passed": True,
            "numerical_imports": 0, "gpu_actions": 0, "scheduler_queries": 0, "network_actions": 0,
            "scientific_credit": 0, "case_credit": 0, "idea_credit": 0, "admission_granted": False,
            "worker_admitted": False}
        print(json.dumps({"verification": put(output / "verification.json", proof), "review": review_ref}))
    except BaseException:
        put(output / "failure.json", {"traceback": traceback.format_exc(), "blocked_numerical_imports": blocked,
            "scientific_credit": 0, "admission_granted": False}); raise


if __name__ == "__main__":
    main()
