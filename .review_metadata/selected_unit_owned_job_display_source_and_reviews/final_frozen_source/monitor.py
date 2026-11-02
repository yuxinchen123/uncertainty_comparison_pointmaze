"""Inspect this study's own jobs and bounded per-unit progress records."""

import argparse
import csv
import hashlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import textwrap
import time
from zoneinfo import ZoneInfo


def read_json(path):
    """Read a completed atomic status record, treating absence as not yet started."""
    return json.loads(path.read_text()) if path.exists() else None


def own_jobs(directory):
    """Query only exact job identifiers found in this study's UVA ledger."""
    ledger = directory / "slurm" / "submitted_jobids_uva.txt"
    identifiers = ledger.read_text().split() if ledger.exists() else []
    if not identifiers:
        return []
    result = subprocess.run(
        ["squeue", "-h", "-j", ",".join(identifiers), "-o", "%i|%T|%N|%b|%R"],
        capture_output=True, text=True, check=True)
    return [line.split("|", 4) for line in result.stdout.splitlines() if line.strip()]


def measurement_progress(value, result):
    """Show measured rounds and their result instead of a missing update count."""
    # Before: {state: completed, decision: accepted}; after: 7/7 pairs; +11.1%.
    if result and result.get("case_completed"):
        count = result.get("paired_rounds", 0)
        gain = result.get("median_speedup")
        outcome = result["decision"].replace("_", " ")
        if gain is not None:
            outcome += f"; {100 * (gain - 1):+.1f}%"
        return f"{count}/{count} pairs; {outcome}" if count else outcome
    if "total_rounds" in value and "round" in value:
        return f"{value['variant']} {value['round']}/{value['total_rounds']} rounds"
    return None


def table_row(name, pool, gpu, state, progress):
    """Wrap complete words into a table with no line longer than 100 characters."""
    # Long mechanism names continue on a second line instead of losing their meaning.
    cells = [str(item) for item in (name, pool, gpu, state, progress)]
    widths = (32, 12, 12, 10, 24)
    wrapped = [textwrap.wrap(cell, width=width) or [""]
               for cell, width in zip(cells, widths)]
    return "\n".join("  ".join(
        f"{parts[index] if index < len(parts) else '':{width}}"
        for parts, width in zip(wrapped, widths)).rstrip()
        for index in range(max(map(len, wrapped))))


def gpu_label(name):
    """Keep the actual GPU model readable within the narrow status column."""
    # For example, NVIDIA RTX A4500 becomes 1 RTX A4500; the card count stays explicit.
    if name.startswith(("1× ", "8× ")):
        return name
    return "1 " + name.removeprefix("NVIDIA ") if name != "unknown" else name




def final_training_progress(value):
    """Show final policy counts together with the exact continuation or phase reason."""
    progress = ", ".join(str(count) + " " + name for name, count in value.get("counts", {}).items())
    return (progress + "; " if progress else "") + value["reason"]


def recorded_scaling_group(directory):
    """Display a scaling allocation only when its recorded roster matches its immutable plan."""
    plan_path = directory / "scaling_plan.json"
    assignment = read_json(directory / "assignment.json") or read_json(directory / "allocation.json")
    if not assignment or not plan_path.exists():
        return None
    plan = read_json(plan_path)
    proof = assignment.get("allocation", {})
    cards = proof.get("cards", [])
    roster = plan.get("gpu_uuids", [])
    if (assignment.get("plan_sha256") != hashlib.sha256(plan_path.read_bytes()).hexdigest()
            or assignment.get("hostname") != "jaguar03" or proof.get("hostname") != "jaguar03"
            or len(roster) not in {1, 8} or len(roster) != len(set(roster))
            or assignment.get("gpu_uuids") != roster or len(cards) != len(roster)
            or {card.get("gpu_uuid") for card in cards} != set(roster)
            or len({card.get("physical_minor") for card in cards}) != len(roster)
            or any(card.get("name") != "NVIDIA RTX A4500" for card in cards)):
        return None
    return {"pool": "jaguar03", "gpu": [str(len(roster)) + "× RTX A4500"],
            "slurm_job_id": proof.get("slurm_job_id", "")}


def process_hosts(records):
    """Join historical host and current hostname fields without accepting conflicting identities."""
    # Before: [{unit: "qualification", hostname: "serval02"}]; after: {qualification: serval02}.
    if records is None:
        return {}
    if not isinstance(records, list):
        raise ValueError("process records must be a list")
    hosts = {}
    hostname_pattern = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*"
    for record in records:
        # Each exact unit has one declared host identity; both supported field names must agree.
        if not isinstance(record, dict):
            raise ValueError("each process record must be an object")
        unit = record.get("unit")
        if (not isinstance(unit, str) or not unit or unit in {".", ".."}
                or "/" in unit or "\x00" in unit or len(unit.encode("utf-8")) > 255):
            raise ValueError("process unit must be an ordinary direct-child name")
        values = [record[key] for key in ("host", "hostname") if key in record]
        if (not values or any(not isinstance(value, str) or len(value) > 253
                or re.fullmatch(hostname_pattern, value) is None for value in values)):
            raise ValueError("process host or hostname must be a valid explicit hostname: " + unit)
        if len(set(values)) != 1:
            raise ValueError("process host and hostname conflict: " + unit)
        if unit in hosts:
            raise ValueError("duplicate process unit identity: " + unit)
        hosts[unit] = values[0]
    return hosts


def selected_unit_job(directory, unit, jobs):
    """Display configured selected units only after joining their exact owned submission records."""
    plan_path = unit / "measurement_plan.json"
    if not plan_path.exists() and not plan_path.is_symlink():
        return None
    # Read only metadata: this display join never imports a producer or qualifies its result.
    def contents(path):
        """Read an ordinary artifact without following a path alias."""
        if path.resolve(strict=True) != path or not path.is_file() or path.is_symlink():
            raise ValueError("selected display metadata requires an exact ordinary path")
        return path.read_bytes()

    def document(path, reference=None):
        """Bind one exact JSON object to its optional raw-byte receipt."""
        raw = contents(path)
        if reference is not None and reference != {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}:
            raise ValueError("selected display metadata byte receipt changed")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("selected display metadata must be a JSON object")
        return value

    plan = document(plan_path)
    if (not isinstance(plan, dict) or plan.get("kind") not in
            {"selected_configuration_measurement", "selected_configuration_profile"}
            or "configured_isolation" not in plan or "direct_binding" in plan):
        return None
    packet = directory / "remote_setup" / unit.name
    if (type(plan.get("schema_version")) is not int or plan["schema_version"] != 1
            or plan.get("campaign_run_dir") != str(directory) or plan.get("unit_id") != unit.name
            or plan.get("hardware_class") != "jaguar03_rtx_a4500" or not isinstance(plan["configured_isolation"], dict)
            or plan["configured_isolation"].get("consumer") != {"selected_configuration_measurement": "selected_measurement",
                "selected_configuration_profile": "component_profile"}[plan["kind"]]
            or plan["configured_isolation"].get("owned_inputs_path") != str(unit / "owned_allocation_inputs.json")):
        raise ValueError("selected plan does not identify this exact campaign and unit")
    submission_path, pre_path = packet / "submission_receipt.json", unit / "configured_isolation_pre_release_inputs.json"
    present = lambda path: path.exists() or path.is_symlink()
    if not present(submission_path):
        if present(pre_path) or present(unit / "owned_allocation_inputs.json"):
            raise ValueError("selected ownership inputs have no submission receipt")
        return {"job_id": "", "state": "planned", "progress": "not submitted; prepared unit"}
    if not present(pre_path):
        return {"job_id": "", "state": "planned", "progress": "submission recorded; ownership metadata incomplete"}
    submission, pre = document(submission_path), document(pre_path)
    identifier = submission.get("job_id")
    helper = plan["helpers"]["source_commit"]
    row = [identifier, plan["kind"], helper, str(packet / "job.sbatch")]
    plan_ref = {"path": str(plan_path), "sha256": hashlib.sha256(contents(plan_path)).hexdigest()}
    if (not isinstance(identifier, str) or re.fullmatch(r"[1-9][0-9]*", identifier) is None
            or type(submission.get("schema_version")) is not int or submission["schema_version"] != 1
            or submission.get("kind") != "owned_uva_configured_isolation_submission"
            or submission.get("cluster") != "uva" or submission.get("source_commit") != helper
            or submission.get("packet") != str(packet) or submission.get("data_unit") != str(unit)
            or type(submission.get("returncode")) is not int or submission["returncode"] != 0
            or submission.get("plan_sha256") != plan_ref["sha256"]
            or submission.get("script_sha256") != hashlib.sha256(contents(packet / "job.sbatch")).hexdigest()
            or submission.get("submission_row") != row
            or submission.get("ledgers_and_submission_row_fsynced_before_this_receipt") is not True):
        raise ValueError("selected submission does not bind this exact plan, script and helper source")
    command = submission.get("held_command")
    if (not isinstance(command, list) or not command or command[0] != "sbatch"
            or "--hold" not in command or "--no-requeue" not in command or command[-1] != row[-1]):
        raise ValueError("selected submission is not the exact held job script")
    if (set(pre) != {"schema_version", "kind", "job_id", "plan", "ledger", "submissions", "submission", "submission_row"}
            or type(pre["schema_version"]) is not int or pre["schema_version"] != 1
            or pre["kind"] != "uva_configured_isolation_pre_release_inputs" or pre["job_id"] != identifier
            or pre["plan"] != plan_ref or pre["submission_row"] != row):
        raise ValueError("selected pre-release ownership envelope changed")
    document(submission_path, pre["submission"])
    # Before: a receipt names 123; after: private and live UVA records each contain the same exact row once.
    for key, filename, private_name, member in (
            ("ledger", "submitted_jobids_uva.txt", "configured_isolation_owned_job_ids.txt", identifier),
            ("submissions", "submissions.tsv", "configured_isolation_owned_submissions.tsv", "\t".join(row))):
        record, live, private = pre[key], directory / "slurm" / filename, unit / private_name
        if (set(record) != {"snapshot", "source_path"} or record["source_path"] != str(live)
                or record["snapshot"] != {"path": str(private), "sha256": hashlib.sha256(contents(private)).hexdigest()}):
            raise ValueError("selected private ownership snapshot changed")
        for path in (private, live):
            if path.resolve(strict=True) != path or not path.is_file() or path.is_symlink():
                raise ValueError("selected ownership ledger requires an exact ordinary path")
            lines = path.read_text().splitlines()
            if (lines.count(member) != 1 or key == "ledger" and len(lines) != len(set(lines))
                    or key == "submissions" and [line for line in lines if line.split("\t", 1)[0] == identifier] != [member]):
                raise ValueError("selected job is missing, duplicated or conflicting in its owned UVA records")
    # A final envelope is optional before release; when present it must preserve every pre-release field.
    final_path = unit / "owned_allocation_inputs.json"
    if present(final_path):
        final = document(final_path)
        if (set(final) != set(pre) | {"release"} or type(final.get("schema_version")) is not int
                or final.get("kind") != "uva_configured_isolation_owned_inputs"
                or {key: value for key, value in final.items() if key not in {"kind", "release"}}
                != {key: value for key, value in pre.items() if key != "kind"}):
            raise ValueError("selected final ownership inputs differ from their pre-release envelope")
        returned = document(packet / "release_receipt.json", final["release"])
        if (type(returned.get("schema_version")) is not int or returned["schema_version"] != 1
                or returned.get("kind") != "owned_uva_configured_isolation_release" or returned.get("cluster") != "uva"
                or returned.get("job_id") != identifier or returned.get("plan_sha256") != plan_ref["sha256"]
                or returned.get("data_unit") != str(unit) or type(returned.get("returncode")) is not int
                or returned["returncode"] != 0 or returned.get("command") != ["scontrol", "release", identifier]
                or returned.get("submission_receipt_sha256") != pre["submission"]["sha256"]):
            raise ValueError("selected final inputs have no matching successful release receipt")
    observed = [job for job in jobs if job[0] == identifier]
    if len(observed) > 1 or observed and (len(observed[0]) != 5 or not all(isinstance(value, str) for value in observed[0])):
        raise ValueError("selected job has conflicting scheduler rows")
    state, progress = "waiting", "owned submission; no current scheduler observation"
    if observed:
        _, scheduler_state, _, _, reason = observed[0]
        if scheduler_state == "RUNNING":
            state, progress = "running", "allocation running; producer progress pending"
        elif scheduler_state == "PENDING":
            state = "held" if reason.strip("()") == "JobHeldUser" else "waiting"
            progress = "held; awaiting release" if state == "held" else "pending; " + reason.strip("()")
        else:
            progress = "owned scheduler state " + scheduler_state
    return {"job_id": identifier, "state": state, "progress": progress,
            "node": observed[0][2] if observed else "",
            "requested_gpu": "RTX A4500 requested" if "--gres=gpu:nvidia_rtx_a4500:1" in command else None}


def unit_rows(directory, excluded_unit_names=(), jobs=()):
    """Read one small status file per unit without retaining training histories."""
    rows = []
    hosts = process_hosts(read_json(directory / "processes.json"))
    actions = read_json(directory / "next_actions.json") or {}
    superseded = set(actions.get("packing_device_memory_failure_receipt", {}).get("preserved_unsubmitted_preparations", []))
    # A 252-byte auxiliary directory is valid, but adding ".json" would exceed
    # the filesystem's filename limit. Match only filenames that actually exist.
    specifications = {path.stem: path for path in (directory / "queue/specifications").glob("*.json")}
    final_statuses = {path.stem: path for path in (directory / "final_training_queue/status").glob("*.json")}
    for path in sorted((directory / "data").glob("*/progress.json")):
        if path.parent.name in excluded_unit_names:
            continue
        value = read_json(path)
        hardware = read_json(path.parent / "hardware.json") or {}
        specification_path = specifications.get(path.parent.name)
        specification = read_json(specification_path) if specification_path is not None else None
        result = read_json(path.parent / "result.json") if value["state"] == "completed" else None
        age = time.time() - path.stat().st_mtime
        phase = value["state"]
        state = ("waiting" if phase.startswith("waiting") else phase
                 if phase in {"completed", "checkpointed", "failed", "initializing", "normalization_warmup"}
                 else "running")
        # A previous attempt's failure stays on disk; only the current attempt sets its state.
        attempt = value.get("attempt")
        failure_root = (path.parent / "attempts" / f"attempt-{attempt:04d}"
                        if attempt is not None else path.parent)
        failure = (read_json(failure_root / "failure.json")
                   or read_json(failure_root / "supervisor_failure.json"))
        if failure:
            state = "failed"
        update = value.get("update", value.get("round", 0))
        target = value.get("target_updates", "")
        progress = f"{update}/{target} updates" if target else str(update)
        if value.get("record") == "benchmark":
            arguments = read_json(path.parent / "arguments.json") or {}
            progress = f"{value['round'] + 1}/{arguments.get('rounds', '?')} measured rounds"
        elif state == "completed" and value.get("rounds"):
            noun = "round" if value["rounds"] == 1 else "rounds"
            progress = f"{value['rounds']} {noun} complete"
        elif state == "completed" and value.get("passed"):
            progress = "checks passed"
        if phase not in ("running", "completed", "checkpointed", "failed"):
            progress = phase.replace("_", " ")
            if value.get("variant"):
                progress += "; " + value["variant"]
        measured = measurement_progress(value, result)
        if measured is not None:
            progress = measured
        elif specification and phase not in {"running", "completed", "failed", "checkpointed"} and state != "waiting":
            progress = f"0/{specification['rounds']} rounds; {progress}"
        if age > 2400 and state not in ("completed", "checkpointed", "failed", "waiting"):
            progress += f"; no record for {int(age / 60)} min"
        if failure:
            progress = (failure["error"] if "error" in failure
                        else failure["failure_stage"].replace("_", " "))
            if "exit_code" in failure:
                progress += f"; exit {failure['exit_code']}"
        visible = hardware.get("cuda_visible_devices", "")
        # Benchmark records put the UUID first; training records put it last.
        queries = list(csv.reader(hardware.get("gpu_query", "").splitlines(), skipinitialspace=True))
        selected = next((row for row in queries if len(row) >= 2 and visible in {row[0], row[-1]}),
                        queries[0] if len(queries) == 1 else [])
        name = path.parent.name
        group = recorded_scaling_group(path.parent) if not hardware else None
        final_status_path = final_statuses.get(name)
        final_status = read_json(final_status_path) if final_status_path is not None else None
        if final_status and final_status.get("unit_id") == name and final_status.get("state") == "held":
            state, progress = "held", final_status.get("reason", "final unit requires diagnosis")
        if specification:
            copies = specification["configuration"]["n_copies"]
            noun = "copy" if copies == 1 else "copies"
            name = specification["hypothesis"].replace("_", " ") + f" ({copies} {noun})"
        # Before: a prepared diagnostic was called running. After: only submitted work can appear active.
        if phase == "planned" and not failure and state != "held":
            packet = directory / "remote_setup" / path.parent.name
            known_jobs = [value.get("job_id"), value.get("slurm_job_id"), hardware.get("slurm_job_id"),
                          group.get("slurm_job_id") if group else None]
            submitted = any(known_jobs) or (packet / "submission_receipt.json").exists()
            executed = any((path.parent / name).exists() for name in ("result.json", "allocation_admission.json", "child_exit_records.json"))
            state = "planned"
            progress = "submission recorded; execution not observed" if submitted else "prepared; execution evidence exists" if executed else "not submitted; 0 completed fixtures"
            if (value.get("kind") == "device_memory_capture_component_regression"
                    and not submitted and not executed and str(packet) in superseded):
                state, progress = "superseded", "superseded before submission; 0 completed fixtures"
        selected_job = None
        try:
            selected_job = selected_unit_job(directory, path.parent, jobs)
        except (OSError, ValueError, TypeError, KeyError) as error:
            # Keep an ownership metadata error visible without calling it a failed numerical experiment.
            selected_job = {"job_id": "", "state": "unknown", "progress": "selected ownership metadata error: " + str(error)}
        if selected_job is not None:
            if hardware.get("slurm_job_id") not in (None, "", selected_job["job_id"]):
                selected_job = {"job_id": "", "state": "unknown", "progress": "selected hardware names a different job"}
            if phase not in {"completed", "checkpointed", "failed"} and not failure:
                state = selected_job["state"]
                if state != "running" or phase in {"planned", "prepared", "prepared_not_submitted"}:
                    progress = selected_job["progress"]
            elif selected_job["state"] == "unknown":
                progress += "; " + selected_job["progress"]
            if selected_job["state"] == "unknown":
                hardware, selected, group = {}, [], None
        row = {"run": name, "pool": group["pool"] if group else hardware.get("hostname", hosts.get(path.parent.name,
                         selected_job.get("node") or "UVA gpu" if selected_job is not None else "unknown")),
                     "gpu": group["gpu"] if group else selected[1:2] or
                         ([selected_job["requested_gpu"]] if selected_job is not None and selected_job.get("requested_gpu") else []),
                     "slurm_job_id": selected_job["job_id"] if selected_job is not None else group["slurm_job_id"] if group else hardware.get("slurm_job_id", ""),
                     "state": state, "progress": progress, "age_seconds": age}
        if selected_job is not None:
            row["selected_owned_job"] = bool(selected_job["job_id"])
        rows.append(row)
    return rows


def owned_metadata_packet(directory, packet, receipt, identifiers, plan_filename):
    """Bind direct-child operational metadata to an exact owned UVA submission and its immutable bytes."""
    # A named packet becomes an owned operational job only after its byte receipts agree.
    identifier = receipt.get("job_id")
    if (receipt.get("cluster") != "uva" or not isinstance(identifier, str)
            or not re.fullmatch(r"[1-9][0-9]*", identifier) or identifier not in identifiers):
        raise ValueError("site submission job is not in this campaign's UVA ledger")
    if packet.is_symlink() or packet.resolve().parent != directory / "remote_setup":
        raise ValueError("site packet must be a direct child of this campaign's remote_setup")
    if (packet / "submission_receipt.json").is_symlink():
        raise ValueError("site submission receipt must stay inside its own packet")
    for filename, key in ((plan_filename, "plan_sha256"),
                          ("manifest.yaml", "manifest_sha256"), ("job.sbatch", "job_script_sha256")):
        path = packet / filename
        if (path.is_symlink() or path.resolve().parent != packet
                or hashlib.sha256(path.read_bytes()).hexdigest() != receipt.get(key)):
            raise ValueError("site submission byte receipt differs: " + filename)
    return identifier, read_json(packet / plan_filename), read_json(packet / "manifest.yaml")


def site_packet(directory, packet, receipt, identifiers):
    """Bind one direct-child site submission to this campaign's exact owned UVA job."""
    identifier, plan, manifest = owned_metadata_packet(directory, packet, receipt, identifiers, "site_validation_plan.json")
    if (plan.get("kind") != "h100_nvl_slurm_site_validation" or type(plan.get("schema_version")) is not int
            or plan["schema_version"] not in {1, 2, 3}
            or plan.get("campaign_run_dir") != str(directory)
            or plan.get("job_shape", {}).get("gres") != "gpu:nvidia_h100_nvl:1"):
        raise ValueError("site plan does not identify this campaign and requested H100 NVL")
    if plan["schema_version"] == 3:
        # This display check identifies the explicit profile; the pinned site verifier owns full qualification.
        expected = {"schema_version": 1, "name": "slurm-configured-device-isolation-driver-580.159.04",
            "expected_driver": "580.159.04", "numeric_gpu_mapping_reconstructed": False,
            "device_bpf_rules_independently_inspected": False, "workers_permitted": False,
            "campaign_claims_permitted": False, "scientific_hypothesis_credit": 0}
        profile = plan.get("qualification_profile")
        if (not isinstance(profile, dict) or plan.get("reference", {}).get("driver") != "590.48.01"
                or any(profile.get(key) != value or type(profile.get(key)) is not type(value) for key, value in expected.items())):
            raise ValueError("schema 3 site display requires the explicit approved qualification profile and unchanged reference driver")
    elif plan.get("qualification_profile") is not None:
        raise ValueError("a site qualification profile requires the explicit schema 3 marker")
    for value in (receipt, plan):
        if (value.get("workers_permitted") is not False or value.get("campaign_claims_permitted") is not False
                or type(value.get("scientific_hypothesis_credit")) is not int
                or value["scientific_hypothesis_credit"] != 0):
            raise ValueError("site records cannot permit workers, claims or scientific credit")
    for key, filename, digest in (("plan", "site_validation_plan.json", "plan_sha256"),
                                  ("job_script", "job.sbatch", "job_script_sha256")):
        if manifest.get(key) != {"path": str(packet / filename), "sha256": receipt[digest]}:
            raise ValueError("site manifest refers to a different " + key)
    if (manifest.get("submission_command") != receipt.get("held_command")
            or not receipt.get("held_command") or receipt["held_command"][-1] != str(packet / "job.sbatch")):
        raise ValueError("site submission command differs from its prepared script")
    return identifier, plan


def allocation_observation_packet(directory, packet, receipt, identifiers):
    """Verify an observer's immutable packet and separate data shard without qualifying its allocation."""
    identifier, plan, manifest = owned_metadata_packet(directory, packet, receipt, identifiers, "allocation_observation_plan.json")
    unit = directory / "data" / packet.name
    boundary = {"mapping_accepted": False, "compatibility_verified": False, "workers_permitted": False,
                "campaign_claims_permitted": False, "scientific_hypothesis_credit": 0}
    if (plan.get("kind") != "h100_nvl_allocation_observation" or type(plan.get("schema_version")) is not int
            or plan["schema_version"] != 1 or plan.get("campaign_run_dir") != str(directory)
            or "allocation observation" not in packet.name.replace("-", " ").replace("_", " ").lower()
            or plan.get("data_unit") != {"path": str(unit), "name": packet.name} or receipt.get("data_unit") != plan["data_unit"]
            or unit.is_symlink() or unit.resolve().parent != directory / "data"):
        raise ValueError("allocation observation must bind its exact campaign and separate data shard")
    shape = plan.get("job_shape", {})
    if (shape.get("account") != "sl5nw_base" or shape.get("partition") != "gpu"
            or shape.get("gres") != "gpu:nvidia_h100_nvl:1"):
        raise ValueError("allocation observation must identify its UVA gpu/sl5nw pool and one H100 NVL")
    for value in (receipt, plan, manifest):
        if any(value.get(key) != expected or type(value.get(key)) is not type(expected) for key, expected in boundary.items()):
            raise ValueError("allocation observations cannot accept mappings, qualify hardware or grant admission or credit")
    required = {"allocation_observation_plan.json", "config_resolved.yaml", "job.sbatch", "command.txt", "experiment_background.md"}
    if (manifest.get("kind") != "prepared_h100_nvl_allocation_observation" or type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != 1 or set(manifest.get("artifacts", {})) != required
            or manifest.get("submission_command") != receipt.get("held_command")):
        raise ValueError("allocation observation manifest or submission command differs")
    command = receipt.get("held_command", [])
    if not command or command[-1] != str(packet / "job.sbatch"):
        raise ValueError("allocation observation submission points at another script")
    for argument in ("--account=sl5nw_base", "--partition=gpu", "--gres=gpu:nvidia_h100_nvl:1"):
        prefix = argument.split("=", 1)[0] + "="
        if [value for value in command if value.startswith(prefix)] != [argument]:
            raise ValueError("allocation observation submission differs from its requested pool or GPU")
    for filename, proof in manifest["artifacts"].items():
        path = packet / filename
        if path.is_symlink() or proof != {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}:
            raise ValueError("allocation observation artifact differs: " + filename)
    for filename in ("manifest.yaml", "config_resolved.yaml", "command.txt", "experiment_background.md"):
        if (unit / filename).is_symlink() or (unit / filename).read_bytes() != (packet / filename).read_bytes():
            raise ValueError("allocation observation data-shard preparation differs: " + filename)
    return identifier, plan, unit


def allocation_observation_hardware(unit, identifier):
    """Display only the observer's recorded host and single visible GPU, without resolving a physical mapping."""
    hardware = read_json(unit / "hardware.json")
    if not hardware:
        return {"observed_host": None, "observed_gpu": None, "gpu_uuid": None}
    if str(hardware.get("slurm_job_id")) != identifier:
        raise ValueError("allocation observation hardware belongs to another job")
    record = {"observed_host": hardware.get("hostname"), "observed_gpu": None, "gpu_uuid": None}
    cards = list(csv.reader(hardware.get("gpu_query", "").splitlines(), skipinitialspace=True))
    if cards:
        if len(cards) != 1 or len(cards[0]) != 5:
            raise ValueError("allocation observation hardware does not contain one visible GPU row")
        index, model, _, driver, gpu = [value.strip() for value in cards[0]]
        if hardware.get("cuda_visible_devices") not in {index, gpu}:
            raise ValueError("allocation observation visible selector differs from its recorded query")
        record.update(observed_gpu=model, gpu=[model], gpu_uuid=gpu, driver=driver)
    return record


def allocation_observation_result(unit, packet, receipt, identifier):
    """Recognize completed raw collection while explicitly leaving qualification and admission unverified."""
    result = read_json(unit / "result.json")
    boundary = {"mapping_accepted": False, "compatibility_verified": False, "workers_permitted": False,
                "campaign_claims_permitted": False, "scientific_hypothesis_credit": 0}
    if (type(result.get("schema_version")) is not int or result["schema_version"] != 1 or result.get("kind") != "h100_nvl_allocation_observation_result"
            or result.get("passed") is not True or result.get("observations_completed") is not True
            or any(result.get(key) != expected or type(result.get(key)) is not type(expected) for key, expected in boundary.items())):
        raise ValueError("raw allocation collection cannot claim qualification, admission or research credit")
    path = unit / "hardware_observations.json"
    if path.is_symlink() or result.get("observations") != {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}:
        raise ValueError("raw allocation collection differs from its observation bytes")
    observations = read_json(path)
    if (observations.get("job_id") != identifier or observations.get("plan") != {
            "path": str(packet / "allocation_observation_plan.json"), "sha256": receipt["plan_sha256"]}
            or any(observations.get(key) != expected or type(observations.get(key)) is not type(expected) for key, expected in boundary.items())):
        raise ValueError("raw allocation observations identify another job or qualification boundary")
    hardware = read_json(unit / "hardware.json")
    if (not hardware or str(hardware.get("slurm_job_id")) != identifier or not hardware.get("gpu_query")
            or not hardware.get("hostname") or hardware["hostname"] != observations.get("hostname")):
        raise ValueError("completed allocation collection lacks matching recorded host and GPU metadata")
    return result


def site_hardware(packet, identifier):
    """Display only allocation identity actually recorded for this exact owned job."""
    # Missing allocation evidence stays unknown, even when a planned node was named elsewhere.
    observed = {"pool": "UVA", "gpu": ["H100 NVL requested"], "requested_gpu": "NVIDIA H100 NVL",
                "observed_gpu": None, "gpu_uuid": None, "driver": None, "allocation_records": []}
    for filename in ("allocation_observation.json", "allocation_details.json", "site_assignment.json"):
        path = packet / filename
        if path.exists():
            observed["allocation_records"].append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    observation = read_json(packet / "allocation_observation.json")
    if observation:
        if observation.get("owned_job", {}).get("job_id") != identifier:
            raise ValueError("site allocation observation belongs to another job")
        observed["pool"] = observation.get("hostname") or "UVA"
        cards = list(csv.reader(observation.get("gpu_csv", "").splitlines()))
        if len(cards) == 1 and len(cards[0]) == 5:
            _, gpu, model, _, driver = [value.strip() for value in cards[0]]
            observed.update(observed_gpu=model, gpu_uuid=gpu, driver=driver, gpu=[model])
    assignment = read_json(packet / "site_assignment.json")
    if assignment:
        proof = assignment.get("allocation", {})
        if proof.get("job_id") != identifier or assignment.get("owned_job", {}).get("job_id") != identifier:
            raise ValueError("site assignment belongs to another job")
        observed.update(pool=proof["hostname"], observed_gpu=proof["model"], gpu=[proof["model"]],
                        gpu_uuid=proof["gpu_uuid"], driver=proof["driver"])
    return observed


def site_accounting(identifier, identity, cache):
    """Cache exact owned-job terminal accounting without inferring a successful site check."""
    # For example, 123|COMPLETED|0:0|serval07 still means ended without a raw site result.
    terminal = {"BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED", "NODE_FAIL",
                "OUT_OF_MEMORY", "PREEMPTED", "REVOKED", "TIMEOUT"}
    saved = cache.get(identifier)
    if saved and saved.get("identity") == identity and saved.get("state") in terminal:
        return saved
    command = ["sacct", "-n", "-P", "-j", identifier, "--format=JobIDRaw,State%64,ExitCode,NodeList"]
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=20,
                            env={**os.environ, "TZ": "UTC"})
    records = [line.split("|") for line in result.stdout.splitlines() if line.split("|", 1)[0] == identifier]
    if len(records) != 1 or len(records[0]) != 4:
        raise ValueError("exact owned site job has no unambiguous accounting row")
    _, state, exit_code, node = records[0]
    record = {"identity": identity, "job_id": identifier, "command": command,
              "observed_at_utc": datetime.now(timezone.utc).isoformat(), "stdout": result.stdout,
              "state": state.split()[0].removesuffix("+"), "exit_code": exit_code, "node": node}
    record["terminal"] = record["state"] in terminal
    if record["state"] in terminal:
        cache[identifier] = record
    return record


def site_validation_rows(directory, jobs):
    """Keep owned site checks, allocation observations and operational anomalies apart from scientific cases."""
    reports = directory / "20_mins_monitoring"
    reports.mkdir(parents=True, exist_ok=True)
    status_path = reports / "site_validation_status.json"
    now = datetime.now(timezone.utc).isoformat()
    anomalies, previous = {}, {}
    try:
        previous = read_json(status_path) or {}
        anomalies = previous.get("anomalies", {})
        cache = previous.get("terminal_accounting", {})
        if not isinstance(anomalies, dict) or not isinstance(cache, dict):
            raise ValueError("site monitoring anomalies and accounting must be mappings")
    except (OSError, ValueError, TypeError, AttributeError) as error:
        anomalies, cache = {}, {}
        anomalies["monitoring record"] = {"reason": "invalid site monitoring record: " + str(error), "observed_at_utc": now}
    ledger = directory / "slurm/submitted_jobids_uva.txt"
    identifiers = set(ledger.read_text().splitlines()) if ledger.exists() else set()
    active = {job[0]: job for job in jobs}
    rows, owned, records = [], set(), []
    for path in sorted((directory / "remote_setup").glob("*/submission_receipt.json")):
        packet, identifier = path.parent, ""
        row = {"kind": "site_validation", "run": "H100 NVL site validation", "pool": "UVA", "gpu": ["H100 NVL requested"],
               "state": "failed", "slurm_job_id": "", "progress": "site submission record requires diagnosis",
               "workers_permitted": False, "campaign_claims_permitted": False, "scientific_hypothesis_credit": 0}
        record = {"packet": str(packet), "observed_at_utc": now}
        try:
            receipt = read_json(path)
            observation = receipt.get("kind") == "owned_h100_nvl_allocation_observation_submission"
            if receipt.get("kind") not in {"owned_h100_nvl_site_validation_submission", "owned_h100_nvl_allocation_observation_submission"}:
                continue
            if observation:
                row.update(kind="allocation_observation", run="H100 NVL allocation observation", pool="UVA gpu/sl5nw",
                           metadata_collected=False, qualification_verified=False, compatibility_verified=False, mapping_accepted=False)
            if receipt.get("cluster") == "uva" and receipt.get("job_id") in identifiers:
                identifier = receipt["job_id"]
                owned.add(identifier)
                row["slurm_job_id"] = identifier
                if (observation and "allocation observation" in packet.name.replace("-", " ").replace("_", " ").lower()
                        and receipt.get("data_unit") == {"path": str(directory / "data" / packet.name), "name": packet.name}):
                    row["data_unit_name"] = packet.name
            if observation:
                identifier, plan, unit = allocation_observation_packet(directory, packet, receipt, identifiers)
                row.update(data_unit_name=unit.name, **allocation_observation_hardware(unit, identifier))
            else:
                identifier, plan = site_packet(directory, packet, receipt, identifiers)
                unit = packet
                if plan["schema_version"] == 3:
                    row.update(qualification_profile_name=plan["qualification_profile"]["name"],
                               expected_driver=plan["qualification_profile"]["expected_driver"], reference_driver=plan["reference"]["driver"])
            owned.add(identifier)
            row.update(slurm_job_id=identifier)
            if not observation:
                row.update(**site_hardware(packet, identifier))
            identity = {"submission_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "plan_sha256": receipt["plan_sha256"], "job_script_sha256": receipt["job_script_sha256"]}
            record.update(job_id=identifier, identity=identity, submission_receipt={"path": str(path), "sha256": identity["submission_sha256"]})
            if (unit / "failure.json").exists():
                failure = read_json(unit / "failure.json")
                error = failure.get("error") if isinstance(failure, dict) else None
                row["progress"] = ("collection failed: " if observation else "site failed: ") + (str(error) if error else "failure file exists without a valid error record")
            elif observation and (unit / "result.json").exists():
                allocation_observation_result(unit, packet, receipt, identifier)
                row.update(state="completed", display_state="COLLECTED", metadata_collected=True,
                           progress="metadata collected; qualification required; zero case and idea credit")
            elif (unit / "result.json").exists():
                result = read_json(packet / "result.json")
                if (result.get("schema_version") != 1 or result.get("kind") != "h100_nvl_slurm_site_evidence" or result.get("passed") is not True
                        or result.get("workers_permitted") is not False or result.get("campaign_claims_permitted") is not False
                        or type(result.get("scientific_hypothesis_credit")) is not int
                        or result["scientific_hypothesis_credit"] != 0
                        or result.get("plan") != {"path": str(packet / "site_validation_plan.json"), "sha256": receipt["plan_sha256"]}):
                    raise ValueError("raw site result does not retain the no admission and zero credit boundary")
                if plan["schema_version"] == 3 and result.get("qualification_profile") != plan["qualification_profile"]:
                    raise ValueError("raw schema 3 site result differs from its explicit qualification profile")
                row.update(state="completed", progress="raw checks complete; independent review required; no worker admission; zero case and idea credit")
            elif identifier in active:
                _, state, node, _, reason = active[identifier]
                if state.startswith("RESV_DEL") or "Held" in reason:
                    row.update(state="held", progress=reason)
                elif state == "PENDING":
                    row.update(state="waiting", progress=reason)
                elif state in {"RUNNING", "COMPLETING", "CONFIGURING"}:
                    row.update(state="running", progress="collecting metadata; qualification required" if observation else "site checks " + state.lower())
                    if node and node not in {"(null)", "None"}:
                        if observation:
                            if row.get("observed_host") and row["observed_host"] != node:
                                raise ValueError("recorded observation host differs from its exact owned scheduler job")
                            row["observed_host"] = node
                        elif row["pool"] != "UVA" and row["pool"] != node:
                            raise ValueError("recorded site host differs from the exact owned scheduler job")
                        else:
                            row["pool"] = node
                else:
                    row.update(state="held", progress="site scheduler state requires inspection: " + state)
            else:
                accounting = site_accounting(identifier, identity, cache)
                record["accounting"] = accounting
                row.update(state="failed", progress=("allocation observation" if observation else "site") + " job ended without result or failure: " + accounting["state"] + "; exit " + accounting["exit_code"])
                if accounting["node"] and accounting["node"] not in {"None", "Unknown", "(null)"}:
                    row["observed_host" if observation else "pool"] = accounting["node"]
                if not accounting["terminal"]:
                    row.update(state="held", progress="site absent from active query; accounting state " + accounting["state"])
            for filename in ("failure.json", "result.json"):
                outcome = unit / filename
                if outcome.exists():
                    record[filename] = {"path": str(outcome), "sha256": hashlib.sha256(outcome.read_bytes()).hexdigest()}
        except (OSError, ValueError, TypeError, KeyError, AttributeError, subprocess.SubprocessError) as error:
            label = "allocation observation" if row["kind"] == "allocation_observation" else "site"
            row.update(state="failed", progress=label + " monitoring record invalid: " + str(error))
        if row["kind"] == "allocation_observation":
            host = row.get("observed_host")
            row["progress"] = (host + "; " if host else "") + row["progress"]
        if row["state"] in {"failed", "held"}:
            anomalies[str(packet)] = {"job_id": identifier, "reason": row["progress"], "observed_at_utc": now}
        rows.append(row)
        records.append({**record, "row": row})
    # The monitor owns only this small operational file, not any site or admission artifact.
    status = {"schema_version": 1, "kind": "site_validation_monitoring", "observed_at_utc": now,
              "sites": [record for record in records if record["row"]["kind"] == "site_validation"],
              "allocation_observations": [record for record in records if record["row"]["kind"] == "allocation_observation"],
              "terminal_accounting": cache, "anomalies": anomalies}
    with tempfile.NamedTemporaryFile(mode="w", dir=reports, prefix="site_validation_status.", delete=False) as stream:
        json.dump(status, stream, indent=2)
        stream.write("\n")
    os.replace(stream.name, status_path)
    return rows, owned, status


def aces_observation_packet(directory, packet, receipt, identifiers):
    """Bind an ACES observation mirror to its exact cluster ledger and immutable remote packet."""
    # A remote job number remains an ACES identity even when UVA issued the same number.
    identifier = receipt.get("job_id")
    if (receipt.get("cluster") != "aces" or type(receipt.get("schema_version")) is not int or receipt["schema_version"] != 1
            or not isinstance(identifier, str) or not re.fullmatch(r"[1-9][0-9]*", identifier)
            or identifier not in identifiers or packet.is_symlink() or packet.resolve().parent != directory / "remote_setup"):
        raise ValueError("ACES observation requires its exact ACES ledger and direct-child packet")
    remote_run = Path("/scratch/user/u.sl359075/rnd_atari_pixels/runs") / directory.name
    remote_packet = remote_run / "remote_setup" / packet.name
    files = {}
    for name, key in (("allocation_observation_plan.json", "plan_sha256"), ("manifest.yaml", "manifest_sha256"),
                      ("job.sbatch", "job_script_sha256")):
        path = packet / name
        if path.is_symlink() or path.resolve().parent != packet or hashlib.sha256(path.read_bytes()).hexdigest() != receipt.get(key):
            raise ValueError("ACES observation byte receipt differs: " + name)
        files[name] = path
    plan, manifest = read_json(files["allocation_observation_plan.json"]), read_json(files["manifest.yaml"])
    unit = directory / "data" / packet.name
    if unit.is_symlink() or unit.resolve().parent != directory / "data":
        raise ValueError("ACES observation data shard must remain inside this campaign's data directory")
    if (type(plan.get("schema_version")) is not int or plan["schema_version"] != 1 or plan.get("kind") != "aces_allocation_observation"
            or plan.get("cluster") != "aces" or plan.get("campaign_run_dir") != str(remote_run)
            or plan.get("data_unit") != {"path": str(remote_run / "data" / packet.name), "name": packet.name}
            or receipt.get("data_unit") != plan["data_unit"]):
        raise ValueError("ACES observation plan must identify its original remote campaign and exact data shard")
    if (manifest.get("kind") != "aces_allocation_observation_preparation" or manifest.get("cluster") != "aces"
            or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1
            or plan.get("owner", {}).get("username") != "u.sl359075" or type(plan["owner"].get("uid")) is not int
            or plan["owner"]["uid"] < 1):
        raise ValueError("ACES preparation needs its exact metadata schema and actual account owner")
    shape = plan.get("job_shape", {})
    expected_shape = {"account": "158590754315", "qos": "normal", "nodes": 1, "ntasks": 1,
                      "gres": "gpu:h100:1", "cpus_per_task": 8, "memory": "32G", "time": "00:05:00"}
    if (shape.get("partition") not in {"gpu", "gpu_debug"} or any(type(shape.get(key)) is not type(value)
            or shape.get(key) != value for key, value in expected_shape.items())):
        raise ValueError("ACES observation plan changed its exact requested pool or resource shape")
    boundary = {"mapping_accepted": False, "compatibility_verified": False, "workers_permitted": False,
                "campaign_claims_permitted": False, "scientific_hypothesis_credit": 0}
    for record in (plan, manifest, receipt):
        if any(type(record.get(key)) is not type(value) or record.get(key) != value for key, value in boundary.items()):
            raise ValueError("ACES metadata cannot accept a mapping, compatibility, admission or research credit")
    command = receipt.get("held_command")
    if (not isinstance(command, list) or command != manifest.get("submission_command") or command[:1] != ["sbatch"]
            or "--hold" not in command or command[-1] != str(remote_packet / "job.sbatch")
            or "--account=158590754315" not in command or "--qos=normal" not in command
            or "--gres=gpu:h100:1" not in command or "--cpus-per-task=8" not in command
            or "--mem=32G" not in command or "--time=00:05:00" not in command
            or not any("--partition=" + name in command for name in ("gpu", "gpu_debug"))):
        raise ValueError("ACES observation must retain its held one-H100, eight-CPU, five-minute command")
    required = {"allocation_observation_plan.json", "config_resolved.yaml", "job.sbatch", "command.txt", "experiment_background.md",
                "resource_basis/aces_submitted_jobids.txt"}
    for name, original in manifest.get("artifacts", {}).items():
        path = packet / name
        if (name not in required or path.is_symlink() or not path.resolve().is_relative_to(packet)
                or original.get("path") != str(remote_packet / name)
                or hashlib.sha256(path.read_bytes()).hexdigest() != original.get("sha256")):
            raise ValueError("ACES prepared artifact changed: " + name)
    if set(manifest.get("artifacts", {})) != required:
        raise ValueError("ACES observation must retain all launch artifacts and its resource ledger snapshot")
    release_path = packet / "release_receipt.json"
    release = read_json(release_path)
    if release is not None and (release_path.is_symlink() or type(release.get("schema_version")) is not int
            or release["schema_version"] != 1 or release.get("kind") != "owned_aces_allocation_observation_release"
            or release.get("cluster") != "aces" or release.get("job_id") != identifier
            or release.get("command") != ["scontrol", "release", identifier] or type(release.get("returncode")) is not int
            or release["returncode"] != 0 or release.get("submission_receipt_sha256") != hashlib.sha256((packet / "submission_receipt.json").read_bytes()).hexdigest()):
        raise ValueError("ACES release receipt must bind this exact owned submission")
    rows = (directory / "slurm/submissions_aces.tsv").read_text().splitlines()
    matching = [line for line in rows if line.split("\t", 1)[0] == identifier]
    expected_row = [identifier, packet.name, plan["helpers"]["source_commit"], str(remote_run)]
    if len(matching) != 1 or receipt.get("submission_row") != expected_row or "\t".join(expected_row) != matching[0]:
        raise ValueError("ACES observation needs its one exact recorded submission row")
    if receipt.get("ledger_path") != str(remote_run / "slurm/submitted_jobids_aces.txt"):
        raise ValueError("ACES submission must name its literal original ACES ledger")
    return identifier, plan, unit, remote_run, boundary


def aces_observation_snapshot(directory, packet, receipt, remote_run, identifier):
    """Read one fresh root-mirrored exact-ID observation without opening a remote connection."""
    # Before: original remote argv/stdout and copied bytes; after: one bounded local status.
    path = packet / "remote_monitor_snapshot.json"
    snapshot = read_json(path)
    expected = {"schema_version": 1, "kind": "aces_allocation_monitor_snapshot", "cluster": "aces",
                "user": "u.sl359075", "job_id": identifier, "local_campaign_run_dir": str(directory),
                "origin_host": "portal05", "query_host": "alogin1.cluster",
                "remote_campaign_run_dir": str(remote_run), "plan_sha256": receipt["plan_sha256"],
                "job_script_sha256": receipt["job_script_sha256"],
                "submission_receipt_sha256": hashlib.sha256((packet / "submission_receipt.json").read_bytes()).hexdigest(),
                "release_receipt_sha256": hashlib.sha256((packet / "release_receipt.json").read_bytes()).hexdigest()
                    if (packet / "release_receipt.json").exists() else None}
    if (not snapshot or path.is_symlink() or not set(expected) <= set(snapshot)
            or any(type(snapshot.get(key)) is not type(value) or snapshot.get(key) != value
                                               for key, value in expected.items())):
        raise ValueError("ACES status snapshot is absent or identifies another cluster, owner or submission")
    observed = datetime.fromisoformat(snapshot["observed_at_utc"].replace("Z", "+00:00"))
    if observed.tzinfo is None or observed.utcoffset().total_seconds() != 0:
        raise ValueError("ACES observation timestamp must explicitly use UTC")
    age = (datetime.now(timezone.utc) - observed).total_seconds()
    if age < 0 or age > 2400:
        displayed = observed.astimezone(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S PT")
        raise ValueError("ACES remote status requires a fresh observation; last snapshot " + displayed)
    queue, accounting = snapshot["squeue"], snapshot["sacct"]
    queue_command = ["squeue", "-h", "-j", identifier, "-o", "%i|%u|%T|%N|%R"]
    accounting_command = ["sacct", "-X", "-n", "-P", "-j", identifier, "--format=JobIDRaw,User%64,State%64,ExitCode,NodeList"]
    queue_absent = (queue.get("returncode") == 1 and not queue.get("stdout")
                    and queue.get("stderr", "").strip() == "slurm_load_jobs error: Invalid job id specified")
    if (queue.get("command") != queue_command or type(queue.get("returncode")) is not int
            or queue["returncode"] != 0 and not queue_absent or accounting.get("command") != accounting_command
            or type(accounting.get("returncode")) is not int or accounting["returncode"] != 0):
        raise ValueError("ACES snapshot must retain successful exact-ID queue and accounting queries")
    parsed = []
    for record, count in ((queue, 5), (accounting, 5)):
        rows = [line.strip().split("|") for line in record.get("stdout", "").splitlines() if line.strip()]
        if len(rows) > 1 or any(len(row) != count or row[:2] != [identifier, "u.sl359075"] for row in rows):
            raise ValueError("ACES scheduler snapshot contains duplicate, foreign or mismatched rows")
        parsed.append(rows[0] if rows else None)
    if expected["release_receipt_sha256"] is None and (not parsed[0] or parsed[0][2] != "PENDING"
            or parsed[0][4] not in {"JobHeldUser", "(JobHeldUser)"}):
        raise ValueError("an unreleased ACES observation requires an actual current JobHeldUser row")
    # Each transported file retains its original remote path and exact unchanged bytes.
    names = {"progress.json", "hardware.json", "owned_allocation.json", "hardware_observations.json",
             "nvidia_hardware_queries.json", "result.json", "failure.json"}
    for relative, record in snapshot.get("mirrored_files", {}).items():
        destination = directory / relative
        if (Path(relative).parts[:2] != ("data", packet.name) or len(Path(relative).parts) != 3
                or destination.name not in names or destination.is_symlink()
                or destination.resolve().parent != (directory / "data" / packet.name).absolute()
                or record.get("source_path") != str(remote_run / relative)
                or hashlib.sha256(destination.read_bytes()).hexdigest() != record.get("sha256")):
            raise ValueError("ACES transported observation bytes changed or escaped their exact data shard")
    return snapshot, parsed[0], parsed[1]


def aces_allocation_observation_rows(directory):
    """Display only owned ACES metadata jobs; never query ACES identifiers on UVA."""
    ledger = directory / "slurm/submitted_jobids_aces.txt"
    identifiers = set(ledger.read_text().splitlines()) if ledger.exists() else set()
    rows, records = [], []
    for path in sorted((directory / "remote_setup").glob("*/submission_receipt.json")):
        packet = path.parent
        namespace = packet.name.replace("-", " ").replace("_", " ").lower()
        recognized = (packet.parent == directory / "remote_setup" and not packet.is_symlink()
                      and packet.resolve().parent == directory / "remote_setup"
                      and "aces" in namespace.split() and "allocation observation" in namespace)
        if not recognized:
            continue
        receipt_error = None
        try:
            receipt = read_json(path)
        except (OSError, ValueError) as error:
            receipt, receipt_error = {}, "ACES submission receipt cannot be read: " + str(error)
        if not isinstance(receipt, dict) or receipt.get("kind") != "owned_aces_allocation_observation_submission":
            receipt, receipt_error = {}, receipt_error or "ACES submission receipt has an invalid object shape or kind"
        row = {"kind": "allocation_observation", "cluster": "aces", "run": "ACES allocation observation",
               "pool": "ACES gpu/u.sl359075", "gpu": ["H100"], "state": "unknown", "slurm_job_id": receipt.get("job_id", ""),
               "progress": "current remote status requires observation", "metadata_collected": False,
               "qualification_verified": False, "mapping_accepted": False, "compatibility_verified": False,
               "workers_permitted": False, "campaign_claims_permitted": False, "scientific_hypothesis_credit": 0}
        # An invalid foreign receipt must never route its named metadata shard into UVA job suppression.
        row["data_unit_name"] = packet.name
        record = {"packet": str(packet)}
        try:
            if receipt_error:
                raise ValueError(receipt_error)
            identifier, plan, unit, remote_run, boundary = aces_observation_packet(directory, packet, receipt, identifiers)
            row["data_unit_name"] = unit.name
            row["pool"] = "ACES " + plan["job_shape"]["partition"] + "/u.sl359075"
            snapshot, queued, accounting = aces_observation_snapshot(directory, packet, receipt, remote_run, identifier)
            record["snapshot_sha256"] = hashlib.sha256((packet / "remote_monitor_snapshot.json").read_bytes()).hexdigest()
            row["observed_at_utc"] = snapshot["observed_at_utc"]
            mirrored = snapshot.get("mirrored_files", {})
            result_key, failure_key = str(Path("data") / unit.name / "result.json"), str(Path("data") / unit.name / "failure.json")
            if queued:
                _, _, state, node, reason = queued
                if state in {"RUNNING", "COMPLETING", "CONFIGURING"}:
                    row["observed_host"] = node
                row.update(state="held" if reason in {"JobHeldUser", "(JobHeldUser)"} else "waiting" if state == "PENDING" else "running"
                           if state in {"RUNNING", "COMPLETING", "CONFIGURING"} else "unknown", progress=reason or state)
            elif accounting:
                _, _, state, exit_code, node = accounting
                state = state.split()[0].removesuffix("+")
                row["observed_host"] = node
                terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}
                if state not in terminal:
                    raise ValueError("ACES job is absent from queue but accounting is not terminal: " + state)
                if result_key in mirrored and state == "COMPLETED" and exit_code == "0:0":
                    result = read_json(unit / "result.json")
                    observations_key = str(Path("data") / unit.name / "hardware_observations.json")
                    if (result.get("kind") != "aces_allocation_observation_result" or type(result.get("schema_version")) is not int
                            or result["schema_version"] != 1
                            or result.get("cluster") != "aces" or result.get("job_id") != identifier
                            or result.get("data_unit") != plan["data_unit"]
                            or result.get("plan") != {"path": str(remote_run / "remote_setup" / packet.name / "allocation_observation_plan.json"),
                                                       "sha256": receipt["plan_sha256"]}
                            or result.get("passed") is not True or result.get("observations_completed") is not True
                            or any(type(result.get(key)) is not type(value) or result.get(key) != value for key, value in boundary.items())
                            or observations_key not in mirrored
                            or result.get("observations") != {"path": str(remote_run / observations_key), "sha256": mirrored[observations_key]["sha256"]}):
                        raise ValueError("ACES collection result differs from its raw observation bytes or zero-credit boundary")
                    observations = read_json(unit / "hardware_observations.json")
                    owned_key = str(Path("data") / unit.name / "owned_allocation.json")
                    if (type(observations.get("schema_version")) is not int or observations["schema_version"] != 1
                            or observations.get("kind") != "aces_allocation_hardware_observations" or observations.get("cluster") != "aces"
                            or observations.get("job_id") != identifier or observations.get("plan") != result["plan"]
                            or observations.get("step_id") != result.get("step_id") or owned_key not in mirrored
                            or observations.get("owned_allocation") != {"path": str(remote_run / owned_key), "sha256": mirrored[owned_key]["sha256"]}):
                        raise ValueError("ACES observations must bind their exact job, step, plan and owned allocation")
                    owned = read_json(unit / "owned_allocation.json")
                    if (type(owned.get("schema_version")) is not int or owned["schema_version"] != 1
                            or owned.get("kind") != "aces_owned_allocation" or owned.get("cluster") != "aces"
                            or owned.get("job_id") != identifier or owned.get("step_id") != result.get("step_id")
                            or type(owned.get("uid")) is not int or owned.get("uid") != plan["owner"]["uid"] or owned.get("hostname") != node
                            or any(type(owned.get(key)) is not type(value) or owned.get(key) != value for key, value in boundary.items())
                            or any(type(observations.get(key)) is not type(value) or observations.get(key) != value for key, value in boundary.items())):
                        raise ValueError("ACES raw collection belongs to another owner, node, job or admission boundary")
                    row.update(state="completed", display_state="COLLECTED", metadata_collected=True,
                               progress="metadata collected; qualification required; zero case and idea credit")
                elif failure_key in mirrored:
                    failure = read_json(unit / "failure.json")
                    row.update(state="failed", progress="collection failed: " + str(failure.get("error", "inspect raw failure")))
                else:
                    row.update(state="failed", progress="job ended " + state + "; exit " + exit_code + "; no completed collection")
            else:
                raise ValueError("ACES exact job has no current queue or accounting row")
            displayed = datetime.fromisoformat(snapshot["observed_at_utc"].replace("Z", "+00:00")).astimezone(
                ZoneInfo("America/Los_Angeles")).strftime("%m-%d %H:%M PT")
            row["progress"] += "; observed " + displayed
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            row.update(state="unknown", progress="remote status unknown: " + str(error))
        rows.append(row)
        records.append({**record, "row": row})
    return rows, records


def tick(directory):
    """Write a combined report, preserving distinctions between active and planned work."""
    directory = directory.resolve()
    reports = directory / "20_mins_monitoring"
    reports.mkdir(parents=True, exist_ok=True)
    controller_error, controller_state = None, None
    if (directory / "SOURCE_AND_GATES_READY.json").exists():
        # Index finished cases and retry eligible failures; parent admission stays separate.
        try:
            from controller import tick as advance_queue
            controller_state = advance_queue(directory)
            (reports / "controller_error.json").unlink(missing_ok=True)
        except Exception as error:
            controller_error = str(error)
            (reports / "controller_error.json").write_text(json.dumps({
                "error": controller_error, "time": datetime.now(timezone.utc).isoformat()}) + "\n")
    from final_training_monitor import tick as advance_final_training
    final_training = advance_final_training(directory, controller_state=controller_state, controller_error=controller_error)
    jobs = own_jobs(directory)
    sites, site_jobs, site_status = site_validation_rows(directory, jobs)
    aces_rows, aces_status = aces_allocation_observation_rows(directory)
    from aces_site_monitor import rows as aces_full_site_rows
    aces_sites, aces_site_status = aces_full_site_rows(directory)
    aces_rows = aces_rows + aces_sites
    excluded_units = {row["data_unit_name"] for row in sites + aces_rows if row.get("data_unit_name")}
    units = unit_rows(directory, excluded_unit_names=excluded_units, jobs=jobs)
    selected_jobs = {unit["slurm_job_id"] for unit in units if unit.get("selected_owned_job")
                     and unit["state"] in {"running", "initializing", "normalization_warmup", "waiting", "held"}}
    display_units = units + sites + aces_rows
    now = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S PT")
    rows = [table_row("run or job", "pool", "gpu", "state", "progress")]
    rows.append("TRAINING")
    submissions = directory / "slurm" / "submissions.tsv"
    names = {}
    if submissions.exists():
        for line in submissions.read_text().splitlines():
            identifier, kind, _, _ = line.split("\t", 3)
            names[identifier] = kind
    observed_jobs = set()
    for path in (directory / "data").glob("*/hardware.json"):
        if path.parent.name in excluded_units:
            continue
        hardware = read_json(path)
        observed_jobs.add(hardware.get("slurm_job_id", ""))
    observed_jobs.update(unit.get("slurm_job_id", "") for unit in units)
    observed_jobs.update(site_jobs)
    for identifier, state, node, gpu, reason in jobs:
        if state == "RUNNING" and identifier not in observed_jobs:
            kind = names.get(identifier, "study job")
            log = directory / "slurm" / "jobs" / f"{kind}_{identifier}.log"
            progress = "monitor active" if kind == "monitor" else "checking"
            if log.exists() and kind != "monitor":
                with log.open("rb") as stream:
                    stream.seek(max(0, log.stat().st_size - 65536))
                    lines = stream.read().decode(errors="replace").splitlines()
                passed = sum(line.startswith("PASS") for line in lines)
                progress = f"{passed} checks passed" if passed else "initializing checks"
            rows.append(table_row(kind, "nolim" if kind == "monitor" else node,
                                  "none" if kind == "monitor" else gpu, state, progress))
    for unit in display_units:
        if unit["state"] in {"running", "initializing", "normalization_warmup"}:
            gpu = str(unit["gpu"][0]).strip() if unit["gpu"] else "unknown"
            rows.append(table_row(unit["run"], unit["pool"], gpu_label(gpu), "RUNNING", unit["progress"]))
    rows.append("WAITING")
    if final_training["state"] in {"waiting", "submitted"}:
        rows.append(table_row("final training continuation", "UVA", "RTX A4500", final_training["state"].upper(),
                              final_training_progress(final_training)))
    for unit in display_units:
        if unit["state"] == "waiting":
            gpu = gpu_label(unit["gpu"][0]) if unit.get("selected_owned_job") and unit["gpu"] or unit.get("kind") in {"site_validation", "allocation_observation", "aces_full_site"} else "none"
            rows.append(table_row(unit["run"], unit["pool"], gpu, "WAITING", unit["progress"]))
    for identifier, state, node, gpu, reason in jobs:
        if state == "PENDING" and identifier not in site_jobs and identifier not in selected_jobs:
            if "Held" not in reason:
                rows.append(table_row("scheduled study job", "UVA", gpu, state, reason))
    rows.append("HELD")
    for unit in display_units:
        if unit["state"] == "held":
            gpu = str(unit["gpu"][0]).strip() if unit["gpu"] else "unknown"
            rows.append(table_row(unit["run"], unit["pool"], gpu_label(gpu), "HELD", unit["progress"]))
    if final_training["state"] == "held":
        rows.append(table_row("final training continuation", "UVA", "RTX A4500", "HELD", final_training["reason"]))
    for identifier, state, node, gpu, reason in jobs:
        if identifier not in site_jobs and identifier not in selected_jobs and (state.startswith("RESV_DEL") or "Held" in reason):
            rows.append(table_row("held study job", "UVA", gpu, "HELD", reason))
    rows.append("FINISHED")
    if final_training["state"] == "finished":
        rows.append(table_row("final training continuation", "UVA", "RTX A4500", "FINISHED",
                              final_training_progress(final_training)))
    for unit in display_units:
        if unit["state"] in {"completed", "checkpointed", "failed", "superseded"}:
            gpu = str(unit["gpu"][0]).strip() if unit["gpu"] else "unknown"
            rows.append(table_row(unit["run"], unit["pool"], gpu_label(gpu),
                                  unit.get("display_state", "FINISHED" if unit["state"] in {"completed", "superseded"} else unit["state"].upper()),
                                  unit["progress"]))
    if controller_error:
        rows.append(table_row("queue controller", "UVA", "none", "FAILED", controller_error))
    rows.append("PLANNED")
    for unit in display_units:
        if unit["state"] in {"planned", "unknown"}:
            gpu = gpu_label(unit["gpu"][0]) if unit["gpu"] else "none"
            rows.append(table_row(unit["run"], unit["pool"], gpu, unit["state"].upper(), unit["progress"]))
    state_path = directory / "campaign_state.json"
    campaign = read_json(state_path) or {"phase": "implementation"}
    counts = campaign.get("queue_counts", {})
    rows.append(table_row("remaining comparisons", "study", "H100 / A4500", "PLANNED",
                          f"{counts.get('pending', 0)} awaiting current runs"))
    rows.append(table_row("next study phase", "study", "H100 / A4500", "PLANNED",
                          campaign.get("next_action", campaign["phase"]).replace("_", " ")))
    for index, unit in enumerate((row for row in aces_rows if row["state"] == "unknown"), 1):
        rows.extend(textwrap.wrap(str(index) + ". " + unit["run"] + ": UNKNOWN; " + unit["progress"], width=100))
    # Mark the end of the complete study only after the controller verifies every stage.
    text = "\n".join(rows) + "\n"
    print(text, flush=True)
    report = "# Pixel Montezuma study status\n\n" + now + "\n\n```text\n" + text + "```\n"
    (reports / "latest_report.md").write_text(report)
    (reports / "latest_status.json").write_text(json.dumps({
        "time": now, "jobs": jobs, "units": units, "campaign": campaign, "final_training": final_training,
        "site_validation": site_status, "allocation_observations": site_status["allocation_observations"],
        "aces_allocation_observations": aces_status, "aces_full_site_checks": aces_site_status}, indent=2) + "\n")
    return (directory / "STUDY_COMPLETE").exists()


def main():
    """Run one monitor tick or the persistent twenty-minute Slurm loop."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval", type=int, default=1200)
    args = parser.parse_args()
    while True:
        if tick(args.run_dir) or not args.loop:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
