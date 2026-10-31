"""Inspect this study's own jobs and bounded per-unit progress records."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
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


def unit_rows(directory):
    """Read one small status file per unit without retaining training histories."""
    rows = []
    processes = read_json(directory / "processes.json") or []
    hosts = {record["unit"]: record["host"] for record in processes}
    for path in sorted((directory / "data").glob("*/progress.json")):
        value = read_json(path)
        hardware = read_json(path.parent / "hardware.json") or {}
        age = time.time() - path.stat().st_mtime
        phase = value["state"]
        state = ("waiting" if phase.startswith("waiting") else phase
                 if phase in {"completed", "checkpointed", "failed", "initializing", "normalization_warmup"}
                 else "running")
        failure = (read_json(path.parent / "failure.json")
                   or read_json(path.parent / "supervisor_failure.json"))
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
        if age > 2400 and state not in ("completed", "checkpointed", "failed", "waiting"):
            progress += f"; no record for {int(age / 60)} min"
        if failure:
            progress = failure["error"][:100]
        visible = hardware.get("cuda_visible_devices", "")
        queries = hardware.get("gpu_query", "").splitlines()
        selected = next((line for line in queries if line.startswith(visible + ",")),
                        queries[0] if len(queries) == 1 else "")
        rows.append({"run": path.parent.name, "pool": hardware.get("hostname", hosts.get(path.parent.name, "unknown")),
                     "gpu": selected.split(",")[1:2],
                     "state": state, "progress": progress, "age_seconds": age})
    return rows


def tick(directory):
    """Write a combined report, preserving distinctions between active and planned work."""
    directory = directory.resolve()
    reports = directory / "20_mins_monitoring"
    reports.mkdir(parents=True, exist_ok=True)
    controller_error = None
    if (directory / "SOURCE_AND_GATES_READY.json").exists():
        # Index finished cases and retry eligible failures; parent admission stays separate.
        try:
            from controller import tick as advance_queue
            advance_queue(directory)
            (reports / "controller_error.json").unlink(missing_ok=True)
        except Exception as error:
            controller_error = str(error)
            (reports / "controller_error.json").write_text(json.dumps({
                "error": controller_error, "time": datetime.now(timezone.utc).isoformat()}) + "\n")
    jobs, units = own_jobs(directory), unit_rows(directory)
    now = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S PT")
    rows = ["run or job                  pool          gpu             state       progress"]
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
            rows.append(f'{kind[:26]:26}  {node[:12]:12}  {gpu[:14]:14}  '
                        f'{state:10}  {progress}')
    for unit in units:
        if unit["state"] in {"running", "initializing", "normalization_warmup"}:
            gpu = str(unit["gpu"][0]).strip() if unit["gpu"] else "unknown"
            rows.append(f'{unit["run"][:26]:26}  {unit["pool"][:12]:12}  '
                        f'{gpu[:14]:14}  {unit["state"][:10]:10}  {unit["progress"][:30]}')
    rows.append("WAITING")
    for unit in units:
        if unit["state"] == "waiting":
            rows.append(f'{unit["run"][:26]:26}  {unit["pool"][:12]:12}  {"none":14}  '
                        f'{"WAITING":10}  {unit["progress"][:30]}')
    for identifier, state, node, gpu, reason in jobs:
        if state == "PENDING":
            rows.append(f'{"scheduled study job":26}  {"UVA":12}  {gpu[:14]:14}  '
                        f'{state:10}  {reason[:25]}')
    rows.append("HELD")
    for identifier, state, node, gpu, reason in jobs:
        if state.startswith("RESV_DEL") or "Held" in reason:
            rows.append(f'{"held study job":26}  {"UVA":12}  {gpu[:14]:14}  '
                        f'{"HELD":10}  {reason[:25]}')
    rows.append("FINISHED")
    for unit in units:
        if unit["state"] in {"completed", "checkpointed", "failed"}:
            gpu = str(unit["gpu"][0]).strip() if unit["gpu"] else "unknown"
            rows.append(f'{unit["run"][:26]:26}  {unit["pool"][:12]:12}  '
                        f'{gpu[:14]:14}  {unit["state"][:10]:10}  {unit["progress"][:30]}')
    if controller_error:
        rows.append(f'{"queue controller":26}  {"UVA":12}  {"none":14}  '
                    f'{"FAILED":10}  {controller_error[:30]}')
    rows.append("PLANNED")
    state_path = directory / "campaign_state.json"
    campaign = read_json(state_path) or {"phase": "implementation"}
    rows.append(f'{"next study phase":26}  {"study":12}  {"H100 / A4500":14}  '
                f'{"PLANNED":10}  {campaign.get("next_action", campaign["phase"])[:30]}')
    # Mark the end of the complete study only after the controller verifies every stage.
    text = "\n".join(rows) + "\n"
    print(text, flush=True)
    report = "# Pixel Montezuma study status\n\n" + now + "\n\n```text\n" + text + "```\n"
    (reports / "latest_report.md").write_text(report)
    (reports / "latest_status.json").write_text(json.dumps({
        "time": now, "jobs": jobs, "units": units, "campaign": campaign}, indent=2) + "\n")
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
