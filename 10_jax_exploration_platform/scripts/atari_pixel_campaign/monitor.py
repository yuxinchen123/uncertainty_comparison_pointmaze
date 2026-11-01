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


def unit_rows(directory, excluded_unit_names=()):
    """Read one small status file per unit without retaining training histories."""
    rows = []
    processes = read_json(directory / "processes.json") or []
    hosts = {record["unit"]: record["host"] for record in processes}
    for path in sorted((directory / "data").glob("*/progress.json")):
        if path.parent.name in excluded_unit_names:
            continue
        value = read_json(path)
        hardware = read_json(path.parent / "hardware.json") or {}
        specification = read_json(directory / "queue" / "specifications" / (path.parent.name + ".json"))
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
        final_status = read_json(directory / "final_training_queue/status" / (name + ".json"))
        if final_status and final_status.get("unit_id") == name and final_status.get("state") == "held":
            state, progress = "held", final_status.get("reason", "final unit requires diagnosis")
        if specification:
            copies = specification["configuration"]["n_copies"]
            noun = "copy" if copies == 1 else "copies"
            name = specification["hypothesis"].replace("_", " ") + f" ({copies} {noun})"
        rows.append({"run": name, "pool": group["pool"] if group else hardware.get("hostname", hosts.get(path.parent.name, "unknown")),
                     "gpu": group["gpu"] if group else selected[1:2],
                     "slurm_job_id": group["slurm_job_id"] if group else hardware.get("slurm_job_id", ""),
                     "state": state, "progress": progress, "age_seconds": age})
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
    excluded_units = {row["data_unit_name"] for row in sites if row.get("data_unit_name")}
    units = unit_rows(directory, excluded_unit_names=excluded_units)
    display_units = units + sites
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
            gpu = gpu_label(unit["gpu"][0]) if unit.get("kind") in {"site_validation", "allocation_observation"} else "none"
            rows.append(table_row(unit["run"], unit["pool"], gpu, "WAITING", unit["progress"]))
    for identifier, state, node, gpu, reason in jobs:
        if state == "PENDING" and identifier not in site_jobs:
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
        if identifier not in site_jobs and (state.startswith("RESV_DEL") or "Held" in reason):
            rows.append(table_row("held study job", "UVA", gpu, "HELD", reason))
    rows.append("FINISHED")
    if final_training["state"] == "finished":
        rows.append(table_row("final training continuation", "UVA", "RTX A4500", "FINISHED",
                              final_training_progress(final_training)))
    for unit in display_units:
        if unit["state"] in {"completed", "checkpointed", "failed"}:
            gpu = str(unit["gpu"][0]).strip() if unit["gpu"] else "unknown"
            rows.append(table_row(unit["run"], unit["pool"], gpu_label(gpu),
                                  unit.get("display_state", "FINISHED" if unit["state"] == "completed" else unit["state"].upper()),
                                  unit["progress"]))
    if controller_error:
        rows.append(table_row("queue controller", "UVA", "none", "FAILED", controller_error))
    rows.append("PLANNED")
    state_path = directory / "campaign_state.json"
    campaign = read_json(state_path) or {"phase": "implementation"}
    counts = campaign.get("queue_counts", {})
    rows.append(table_row("remaining comparisons", "study", "H100 / A4500", "PLANNED",
                          f"{counts.get('pending', 0)} awaiting current runs"))
    rows.append(table_row("next study phase", "study", "H100 / A4500", "PLANNED",
                          campaign.get("next_action", campaign["phase"]).replace("_", " ")))
    # Mark the end of the complete study only after the controller verifies every stage.
    text = "\n".join(rows) + "\n"
    print(text, flush=True)
    report = "# Pixel Montezuma study status\n\n" + now + "\n\n```text\n" + text + "```\n"
    (reports / "latest_report.md").write_text(report)
    (reports / "latest_status.json").write_text(json.dumps({
        "time": now, "jobs": jobs, "units": units, "campaign": campaign, "final_training": final_training,
        "site_validation": site_status, "allocation_observations": site_status["allocation_observations"]}, indent=2) + "\n")
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
