"""Maintain a persistent pixel-benchmark queue using only standard-library metadata work.

Only the parent session writes SOURCE_AND_GATES_READY.json. This controller never
authorizes source code, creates GPU jobs, chooses the additional twenty ideas, or
starts long training. Existing case specifications and attempt records are kept.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ideas import (IDEAS_BY_NAME, REQUIRED_HARDWARE, implementation_status,
                   investigated_hypotheses, new_registry, record_result,
                   refresh_implementations, registry_idea, source_capabilities)


SENTINEL_NAME = "SOURCE_AND_GATES_READY.json"
SOURCE_SCOPES = ("10_jax_exploration_platform/src/exploration_platform/atari_pixels",
                 "10_jax_exploration_platform/scripts/atari_pixel_campaign")
QUEUE_STATES = ("pending", "running", "completed", "failed")
REFERENCE_ALGORITHM = {
    "n_envs": 128, "num_steps": 128, "action_count": 18,
    "update_epochs": 4, "num_minibatches": 4, "total_timesteps": 2_000_000_000,
    "learning_rate": 1e-4, "adam_epsilon": 1e-5, "anneal_learning_rate": True,
    "gamma_ext": 0.999, "gamma_int": 0.99, "gae_lambda": 0.95,
    "clip_coefficient": 0.1, "entropy_coefficient": 0.001, "value_coefficient": 0.5,
    "max_gradient_norm": 0.5, "intrinsic_coefficient": 1.0, "extrinsic_coefficient": 2.0,
    "predictor_update_proportion": 0.25, "observation_warmup_iterations": 50,
    "upstream_reward_filter_axis": True, "warmup_keeps_initial_observation": True,
}


def utc_now():
    """Record timezone-aware UTC metadata independently of the host clock display settings."""
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    """Read a complete JSON record, leaving malformed data as a visible error."""
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    """Replace metadata only after its complete contents have been written and flushed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}-{uuid.uuid4().hex}")
    with temporary.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@contextmanager
def queue_lock(run_dir):
    """Serialize queue metadata transitions; timed GPU work never holds this shared lock."""
    directory = Path(run_dir) / "queue"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "metadata.lock").open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def layout(run_dir):
    """Create queue bookkeeping directories without authorizing or launching any experiment."""
    run_dir = Path(run_dir)
    for name in (*QUEUE_STATES, "specifications", "leases", "workers", "indexed_results"):
        (run_dir / "queue" / name).mkdir(parents=True, exist_ok=True)


def content_digest(value):
    """Identify a complete immutable specification rather than its convenient filename alone."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def verify_source(source_root, source_commit):
    """Refuse changed or uncommitted study code instead of silently running another version."""
    root = Path(source_root)
    if not root.is_absolute() or not root.is_dir() or len(source_commit) != 40:
        raise ValueError("source_root must be an existing absolute Git checkout and source_commit a full hash")
    actual = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    if actual != source_commit:
        raise ValueError(f"immutable source changed: expected {source_commit}, found {actual}")
    changes = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all",
                              "--", *SOURCE_SCOPES], capture_output=True, text=True, check=True).stdout.strip()
    if changes:
        raise ValueError("study source has uncommitted changes: " + changes)
    return {"source_root": str(root), "source_commit": actual}


def ready_specification(run_dir):
    """Read the parent's source/gate authorization and verify both hardware contexts."""
    path = Path(run_dir) / SENTINEL_NAME
    if not path.exists():
        return None
    ready = read_json(path)
    if ready.get("authorized") is not True:
        return None
    if set(ready.get("contexts", [])) != set(REQUIRED_HARDWARE):
        raise ValueError("the ready sentinel must name exactly h100 and jaguar03_rtx_a4500")
    verify_source(ready["source_root"], ready["source_commit"])
    counts = ready.get("matched_copy_counts", [])
    if not counts or any(type(value) is not int or value < 1 for value in counts) or len(set(counts)) != len(counts):
        raise ValueError("matched_copy_counts must contain distinct positive policy-copy counts")
    if 1 not in counts:
        raise ValueError("the initial comparison must include one policy copy")
    for context in REQUIRED_HARDWARE:
        gate = read_json(Path(ready["gate_records"][context]))
        if gate.get("passed") is not True or gate.get("source_commit") != ready["source_commit"]:
            return None
    configuration = read_json(Path(ready["reference_configuration_path"]))
    if any(configuration.get(name) != value for name, value in REFERENCE_ALGORITHM.items()):
        raise ValueError("the ready sentinel's reference configuration differs from the agreed CleanRL settings")
    if (configuration.get("per_copy_learning_rates") or configuration.get("per_copy_intrinsic_coefficients")
            or configuration.get("per_copy_scalar_settings")):
        raise ValueError("this study uses one common CleanRL setting; per-copy interfaces are checked in the gates")
    ready = {**ready, "reference_configuration": configuration,
             "campaign_run_dir": str(Path(run_dir).resolve())}
    return ready


def immutable_case(entry, context, copies, ready):
    """Record source, algorithm, execution changes and random identities before a worker claims them."""
    seed_start = ready.get("seed_start", 1)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", entry.name):
        raise ValueError("hypothesis names must be plain path components made of words, digits and separators")
    identifier = (f"{entry.name}__hardware-{context}__copies-{copies}__seed-start-{seed_start}"
                  f"__source-commit-{ready['source_commit']}")
    case = {"schema_version": 1, "unit_id": identifier, "hypothesis": entry.name,
            "campaign_run_dir": ready["campaign_run_dir"],
            "hypothesis_position": entry.position, "hardware_class": context,
            "source_root": ready["source_root"], "source_commit": ready["source_commit"],
            "configuration": {**ready["reference_configuration"], "n_copies": copies},
            "candidate": {"learner": entry.learner, "environment": entry.environment,
                          "execution": entry.execution},
            "reference_execution": entry.reference_execution,
            "rounds": ready.get("rounds", 7), "iterations": ready.get("iterations", 10),
            "seed_start": seed_start, "additional_correctness_checks": list(entry.additional_checks),
            "maximum_attempts": ready.get("maximum_attempts", 3)}
    if (any(type(case[name]) is not int for name in ("rounds", "iterations", "maximum_attempts", "seed_start"))
            or case["rounds"] < 7 or case["iterations"] < 1 or case["maximum_attempts"] < 1
            or case["seed_start"] < 0):
        raise ValueError("benchmark rounds, updates and retry limits are invalid")
    case["specification_digest"] = content_digest(case)
    return case


def case_states(run_dir, identifier):
    """Find every queue state for one case, so duplicate-state corruption is visible."""
    queue = Path(run_dir) / "queue"
    return [name for name in QUEUE_STATES if (queue / name / f"{identifier}.json").exists()]


def enqueue_case(run_dir, case):
    """Create one immutable case once; an existing identifier must have identical contents."""
    queue = Path(run_dir) / "queue"
    path = queue / "specifications" / f"{case['unit_id']}.json"
    if path.exists():
        if read_json(path) != case:
            raise ValueError("an immutable case identifier was reused for different contents")
        states = case_states(run_dir, case["unit_id"])
        if not states and not (queue / "leases" / path.name).exists():
            atomic_json(queue / "pending" / path.name, case)
            return True
        if len(states) != 1:
            raise ValueError("an existing case must occupy exactly one queue state")
        return False
    if case_states(run_dir, case["unit_id"]):
        raise ValueError("queue state exists without its immutable case specification")
    atomic_json(path, case)
    atomic_json(queue / "pending" / path.name, case)
    return True


def attempt_paths(run_dir, identifier, attempt):
    """Keep every attempt's detailed files while exposing current unit progress to monitoring."""
    if Path(identifier).name != identifier or identifier in (".", ".."):
        raise ValueError("a unit identifier must be one plain filename component")
    unit = Path(run_dir) / "data" / identifier
    return unit, unit / "attempts" / f"attempt-{attempt:04d}"


def retry_failed(run_dir, identifier):
    """Retry an explicitly retryable finished failure without altering old attempt files."""
    queue = Path(run_dir) / "queue"
    failed = queue / "failed" / f"{identifier}.json"
    if not failed.exists():
        return False
    case = read_json(failed)
    unit = Path(run_dir) / "data" / identifier
    failure = read_json(unit / "failure.json")
    lease = read_json(queue / "leases" / failed.name)
    if not failure.get("retryable") or lease["attempt"] >= case["maximum_attempts"]:
        return False
    if case_states(run_dir, identifier) != ["failed"]:
        raise ValueError("a failed case occupies more than one queue state")
    failed.rename(queue / "pending" / failed.name)
    return True


def process_start_time(pid):
    """Read Linux's process birth counter, distinguishing a reused PID from its original owner."""
    path = Path(f"/proc/{pid}/stat")
    try:
        text = path.read_text()
    except FileNotFoundError:
        return None
    # The command name may contain spaces; fields after the final ')' start at stat field 3.
    return text.rsplit(")", 1)[1].split()[19]


def process_matches(pid, start_time):
    """Determine liveness without sending a signal or terminating any process."""
    return bool(pid and start_time and process_start_time(pid) == str(start_time))


def scientific_case_complete(result):
    """Distinguish a completed scientific rejection from contaminated or interrupted timing."""
    return bool(result and result.get("case_completed") and (
        result.get("measurement_valid") or result.get("failure_stage") == "correctness"
        and result.get("correctness_attempted") and result.get("decision") == "rejected"))


def recover_local_claim(run_dir, identifier, expected_token):
    """Recover a verified dead local attempt; never infer death from elapsed lease age."""
    queue = Path(run_dir) / "queue"
    path = queue / "running" / f"{identifier}.json"
    if not path.exists():
        raise ValueError("the requested case is not running")
    lease_path = queue / "leases" / path.name
    lease = read_json(lease_path)
    if lease["token"] != expected_token or lease["hostname"] != socket.gethostname():
        raise ValueError("recovery must run on the recorded host with its exact lease token")
    if process_matches(lease["worker_pid"], lease["worker_start_time"]) or process_matches(
            lease.get("runner_pid"), lease.get("runner_start_time")):
        raise ValueError("the recorded worker or runner is still alive; leave its claim alone")
    case = read_json(path)
    unit, attempt = attempt_paths(run_dir, identifier, lease["attempt"])
    result_path = attempt / "result.json"
    result = read_json(result_path) if result_path.exists() else None
    if scientific_case_complete(result):
        atomic_json(unit / "result.json", result)
        path.rename(queue / "completed" / path.name)
        lease.update(state="completed", recovered_at=utc_now())
        atomic_json(unit / "progress.json", {"state": "completed", "attempt": lease["attempt"],
                                            "decision": result["decision"], "time": utc_now()})
    else:
        failure_path = attempt / "failure.json"
        failure = (read_json(failure_path) if failure_path.exists() else
                   {"failure_stage": "unrelated_gpu_use" if result and result.get("contaminated")
                    else "worker_interruption", "retryable": True,
                    "case_completed": False, "time": utc_now(), "attempt": lease["attempt"]})
        if not failure_path.exists():
            atomic_json(failure_path, failure)
        atomic_json(unit / "failure.json", failure)
        path.rename(queue / "failed" / path.name)
        lease.update(state="failed", recovered_at=utc_now())
        atomic_json(unit / "progress.json", {"state": "failed", "attempt": lease["attempt"],
                                            "failure_stage": failure["failure_stage"], "time": utc_now()})
    atomic_json(lease_path, lease)
    return case


def index_completed(run_dir, registry):
    """Index every durable completed attempt once, including numerical rejections."""
    queue = Path(run_dir) / "queue"
    indexed = 0
    for path in sorted((queue / "completed").glob("*.json")):
        case = read_json(path)
        marker = queue / "indexed_results" / path.name
        if marker.exists():
            continue
        lease = read_json(queue / "leases" / path.name)
        _, attempt = attempt_paths(run_dir, case["unit_id"], lease["attempt"])
        result_path = attempt / "result.json"
        result = read_json(result_path)
        if not result.get("case_completed") or result.get("specification_digest") != case["specification_digest"]:
            raise ValueError("a completed queue case lacks its matching complete result")
        expected = {"idea": case["hypothesis"], "hardware_class": case["hardware_class"],
                    "copies": case["configuration"]["n_copies"], "source_commit": case["source_commit"]}
        if any(result.get(name) != value for name, value in expected.items()):
            raise ValueError("completed result identity differs from its immutable case")
        if not scientific_case_complete(result):
            raise ValueError("an invalid timing attempt was incorrectly moved to completed")
        # Registry persistence happens before this marker. On restart, recognize its result
        # path rather than appending it a second time after an interrupted controller tick.
        entries = list(registry["initial_ideas"])
        entries.extend(row for batch in registry["adaptive_batches"] for row in batch["hypotheses"])
        present = any(record.get("result_path") == str(result_path)
                      for entry in entries for records in entry.get("results", {}).values()
                      for record in records)
        if not present:
            record_result(registry, result, result_path)
            atomic_json(Path(run_dir) / "study_registry.json", registry)
        atomic_json(marker, {"result_path": str(result_path), "indexed_at": utc_now()})
        indexed += 1
    return indexed


def investigated_at_counts(registry, names, copy_counts):
    """Require each requested matched-copy measurement, not merely one result per GPU type."""
    entries = list(registry["initial_ideas"])
    entries.extend(entry for batch in registry["adaptive_batches"] for entry in batch["hypotheses"])
    for name in names:
        entry = next(row for row in entries if row["name"] == name)
        for context in REQUIRED_HARDWARE:
            for copies in copy_counts:
                records = [row for row in entry["results"].get(context, []) if row.get("copies") == copies]
                if not any(row.get("implementation_distinct") and (
                        row.get("measurement_valid") and row.get("correctness_passed")
                        or row.get("correctness_attempted") and row.get("failure_stage") == "correctness"
                        and row.get("decision") == "rejected") for row in records):
                    return False
    return True


def recover_unstarted_claims(run_dir):
    """Repair only claims that crashed before any lease or child process could be created."""
    queue = Path(run_dir) / "queue"
    repaired = 0
    for path in (queue / "running").glob("*.json"):
        if (queue / "leases" / path.name).exists():
            continue
        if (Path(run_dir) / "data" / path.stem).exists():
            raise ValueError("a running case has data but no lease; explicit investigation is required")
        path.rename(queue / "pending" / path.name)
        repaired += 1
    return repaired


def final_requirements_met(registry, flags):
    """Expose readiness only after fifty investigated ideas and separate final verification records."""
    if len(investigated_hypotheses(registry)) != 50:
        return False
    required = ("final_combination_correctness_passed", "final_confirmation_passed",
                "h100_packing_complete", "a4500_packing_complete", "eight_gpu_scaling_complete")
    return all(flags.get(name) is True for name in required)


def tick(run_dir):
    """Advance metadata, retry eligible infrastructure failures, and report the next unmet stage."""
    run_dir = Path(run_dir).resolve()
    layout(run_dir)
    with queue_lock(run_dir):
        repaired = recover_unstarted_claims(run_dir)
        authorization_error = None
        try:
            ready = ready_specification(run_dir)
        except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as error:
            ready = None
            authorization_error = str(error)
        registry_path = run_dir / "study_registry.json"
        registry = read_json(registry_path) if registry_path.exists() else None
        if registry is None:
            if ready is None:
                state = {"phase": "waiting_for_parent_source_and_gates", "queued": 0,
                         "authorization_error": authorization_error, "time": utc_now()}
                atomic_json(run_dir / "campaign_state.json", state)
                return state
            registry = new_registry(ready["source_commit"], Path(ready["source_root"]) / "10_jax_exploration_platform")
        index_completed(run_dir, registry)
        retried = sum(retry_failed(run_dir, path.stem)
                      for path in list((run_dir / "queue" / "failed").glob("*.json")))
        completed = set(investigated_hypotheses(registry))
        initial_counts = registry.get("initial_matched_copy_counts", (
            ready["matched_copy_counts"] if ready is not None else registry.get("matched_copy_counts", [1])))
        initial_complete = investigated_at_counts(registry, IDEAS_BY_NAME, initial_counts)
        queued = 0
        unavailable = []
        if ready is not None:
            platform_root = Path(ready["source_root"]) / "10_jax_exploration_platform"
            capabilities = source_capabilities(platform_root)
            refresh_implementations(registry, platform_root)
            registry["matched_copy_counts"] = ready["matched_copy_counts"]
            if not initial_complete:
                registry.setdefault("initial_matched_copy_counts", list(initial_counts))
                counts_required = registry["initial_matched_copy_counts"]
                if registry["reference_commit"] != ready["source_commit"]:
                    if any(entry["results"][context] for entry in registry["initial_ideas"]
                           for context in REQUIRED_HARDWARE):
                        raise ValueError("changing an investigated initial reference needs an explicit new study revision")
                    registry["reference_commit"] = ready["source_commit"]
                selected = [registry_idea(name, registry) for name in IDEAS_BY_NAME]
            else:
                selected = []
                for batch in registry["adaptive_batches"]:
                    names = [entry["name"] for entry in batch["hypotheses"]]
                    if any(name is None for name in names):
                        break
                    batch_counts = batch.get("matched_copy_counts", ready["matched_copy_counts"])
                    if not investigated_at_counts(registry, names, batch_counts):
                        if batch["reference_commit"] == ready["source_commit"]:
                            selected = [registry_idea(name, registry) for name in names]
                            batch.setdefault("matched_copy_counts", list(batch_counts))
                            counts_required = batch["matched_copy_counts"]
                        break
            for entry in selected:
                status = implementation_status(entry, capabilities)
                if status["state"] != "ready_for_correctness_check":
                    unavailable.append(entry.name)
                    continue
                for context in REQUIRED_HARDWARE:
                    for copies in sorted(counts_required):
                        queued += enqueue_case(run_dir, immutable_case(entry, context, copies, ready))
            registry["reference_freeze_required_before_measurement"] = False
        counts = {name: len(list((run_dir / "queue" / name).glob("*.json"))) for name in QUEUE_STATES}
        flags_path = run_dir / "final_verification.json"
        flags = read_json(flags_path) if flags_path.exists() else {}
        if (ready is not None and flags.get("source_commit") == ready["source_commit"]
                and final_requirements_met(registry, flags)
                and not any(counts[name] for name in ("pending", "running", "failed"))):
            phase = "ready_for_final"
        elif initial_complete:
            active_adaptive = any(read_json(path)["hypothesis_position"] >= 31
                                  for name in ("pending", "running", "failed")
                                  for path in (run_dir / "queue" / name).glob("*.json"))
            phase = ("adaptive_benchmarks" if active_adaptive else "adaptive_research_pending"
                     if len(completed) < 50 else "final_verification_pending")
        elif ready is None:
            phase = "waiting_for_parent_source_and_gates"
        elif unavailable:
            phase = "waiting_for_implementations"
        else:
            phase = "initial_benchmarks"
        remaining = [name for name in IDEAS_BY_NAME
                     if not investigated_at_counts(registry, [name], initial_counts)]
        remaining_work = [{"unit_id": case["unit_id"], "hypothesis": case["hypothesis"],
                           "hardware_class": case["hardware_class"], "copies": case["configuration"]["n_copies"],
                           "state": name}
                          for name in ("pending", "running", "failed")
                          for case in (read_json(path) for path in sorted((run_dir / "queue" / name).glob("*.json")))]
        state = {"phase": phase, "queued": queued, "retried": retried, "repaired_unstarted": repaired,
                 "queue_counts": counts,
                 "investigated_hypotheses": len(completed), "initial_remaining": remaining,
                 "remaining_work": remaining_work,
                 "pending_implementations": unavailable, "time": utc_now(),
                 "authorization_error": authorization_error,
                 "next_action": ("parent chooses five hypotheses from profiles" if phase == "adaptive_research_pending"
                                 else "verify all final requirements before launching training" if phase == "ready_for_final"
                                 else phase)}
        atomic_json(registry_path, registry)
        atomic_json(run_dir / "campaign_state.json", state)
        return state


def main():
    """Perform a metadata-only tick or recover one specifically identified dead local claim."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--recover-unit")
    parser.add_argument("--expected-token")
    parser.add_argument("--list-remaining", action="store_true")
    args = parser.parse_args()
    if args.recover_unit:
        if not args.expected_token:
            raise ValueError("exact lease token is required for recovery")
        with queue_lock(args.run_dir):
            recover_local_claim(args.run_dir, args.recover_unit, args.expected_token)
    state = tick(args.run_dir)
    if not args.list_remaining:
        state.pop("remaining_work", None)
    print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    main()
