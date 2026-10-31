"""Inspect this study's own jobs and bounded per-unit progress records."""

import argparse
import hashlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
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


def unit_rows(directory):
    """Read one small status file per unit without retaining training histories."""
    rows = []
    processes = read_json(directory / "processes.json") or []
    hosts = {record["unit"]: record["host"] for record in processes}
    for path in sorted((directory / "data").glob("*/progress.json")):
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
        queries = hardware.get("gpu_query", "").splitlines()
        selected = next((line for line in queries if line.startswith(visible + ",")),
                        queries[0] if len(queries) == 1 else "")
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
                     "gpu": group["gpu"] if group else selected.split(",")[1:2],
                     "slurm_job_id": group["slurm_job_id"] if group else hardware.get("slurm_job_id", ""),
                     "state": state, "progress": progress, "age_seconds": age})
    return rows


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
    jobs, units = own_jobs(directory), unit_rows(directory)
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
    for unit in units:
        if unit["state"] in {"running", "initializing", "normalization_warmup"}:
            gpu = str(unit["gpu"][0]).strip() if unit["gpu"] else "unknown"
            rows.append(table_row(unit["run"], unit["pool"], gpu_label(gpu), "RUNNING", unit["progress"]))
    rows.append("WAITING")
    if final_training["state"] in {"waiting", "submitted"}:
        rows.append(table_row("final training continuation", "UVA", "RTX A4500", final_training["state"].upper(),
                              final_training_progress(final_training)))
    for unit in units:
        if unit["state"] == "waiting":
            rows.append(table_row(unit["run"], unit["pool"], "none", "WAITING", unit["progress"]))
    for identifier, state, node, gpu, reason in jobs:
        if state == "PENDING":
            if "Held" not in reason:
                rows.append(table_row("scheduled study job", "UVA", gpu, state, reason))
    rows.append("HELD")
    for unit in units:
        if unit["state"] == "held":
            gpu = str(unit["gpu"][0]).strip() if unit["gpu"] else "unknown"
            rows.append(table_row(unit["run"], unit["pool"], gpu_label(gpu), "HELD", unit["progress"]))
    if final_training["state"] == "held":
        rows.append(table_row("final training continuation", "UVA", "RTX A4500", "HELD", final_training["reason"]))
    for identifier, state, node, gpu, reason in jobs:
        if state.startswith("RESV_DEL") or "Held" in reason:
            rows.append(table_row("held study job", "UVA", gpu, "HELD", reason))
    rows.append("FINISHED")
    if final_training["state"] == "finished":
        rows.append(table_row("final training continuation", "UVA", "RTX A4500", "FINISHED",
                              final_training_progress(final_training)))
    for unit in units:
        if unit["state"] in {"completed", "checkpointed", "failed"}:
            gpu = str(unit["gpu"][0]).strip() if unit["gpu"] else "unknown"
            rows.append(table_row(unit["run"], unit["pool"], gpu_label(gpu),
                                  "FINISHED" if unit["state"] == "completed" else unit["state"].upper(),
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
        "time": now, "jobs": jobs, "units": units, "campaign": campaign, "final_training": final_training}, indent=2) + "\n")
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
