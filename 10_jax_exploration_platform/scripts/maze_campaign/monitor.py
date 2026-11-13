"""One monitoring pass over a maze campaign run folder, or a loop of passes for a persistent Slurm job.

    python monitor.py --run-dir RUN                       one pass
    python monitor.py --run-dir RUN --loop [--top-up-phase PHASE] [--complete-when-drained]
                      [--report-command "<command that prints Markdown; {run_dir} is filled in>"]

Each pass
  1. returns to pending the units whose Slurm job no longer runs (ids are read only from this run
     folder's own ledgers);
  2. with --top-up-phase, submits worker jobs for pending units that have no worker (submit.py);
  3. counts the units of every queue folder and reads each unit's progress from its shard
     (windows written against windows expected);
  4. writes RUN/status.json, RUN/status.txt (the table for the terminal) and one Markdown report
     in RUN/20_mins_monitoring/outputs/.
A loop passes every 20 minutes, and every hour once three passes over more than an hour found
nothing changed and nothing to warn about. It ends by itself when RUN/SWEEP_COMPLETE exists. It
never cancels a job. Standard library only.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess
import textwrap
import time
from zoneinfo import ZoneInfo

from grid_campaign import common, load, own
import unit_queue

PASS_SECONDS = 1200
STABLE_PASS_SECONDS = 3600
STALE_SECONDS = 2400    # a running unit whose shard did not grow for this long is named in the warnings
WIDTH = 100             # the status table's longest line


def unit_rows(run_dir, previous, moment):
    """One row per unit of every queue folder: where it is, its record, and its progress from its shard."""
    # after: [{"name": "...", "state": "running", "windows_written": 37, "windows_expected": 200,
    #          "bytes": 912345, "unchanged_since": 1794285410.0, "stale": False, "unit": {its queue record}}, ...]
    rows = []
    for state in unit_queue.counts(run_dir):
        for unit in unit_queue.read_units(run_dir, state):
            if state == "done":
                progress = {"windows_written": unit["windows_expected"], "bytes": None}
            else:
                progress = unit_queue.shard_progress(Path(run_dir) / "data" / f"{unit['name']}.jsonl")
            # a shard of the same size as at the last pass has not grown since `unchanged_since`
            # (a new claim starts the count again: a continued unit first loads its checkpoint)
            before = previous.get(unit["name"], {})
            same = before.get("bytes") == progress["bytes"] and before.get("job_id") == unit.get("job_id")
            unchanged_since = before.get("unchanged_since", moment) if same else moment
            rows.append({"name": unit["name"], "state": state, "windows_written": progress["windows_written"],
                         "windows_expected": unit["windows_expected"], "bytes": progress["bytes"],
                         "unchanged_since": unchanged_since, "job_id": unit.get("job_id"),
                         "stale": state == "running" and moment - unchanged_since >= STALE_SECONDS, "unit": unit})
    return rows


def sweep_rows(rows):
    """Count the units of each sweep (phase, environment, reward setting, method) by queue folder."""
    # after: {("development-first-pass", "pointmaze-large", "arrival_plus_one_and_step_cost", "cleanrl-ppo-rnd"):
    #         {"pending": 1, "running": 2, "done": 1, "failed": 0, "elsewhere": 0, "windows_written": 512,
    #          "windows_expected": 800}}
    sweeps = {}
    for row in rows:
        unit = row["unit"]
        sweep = sweeps.setdefault(
            (unit["phase"], unit["environment"], unit["reward_setting"], unit["method"]),
            {"pending": 0, "running": 0, "done": 0, "failed": 0, "elsewhere": 0, "windows_written": 0,
             "windows_expected": 0})
        # units a remote server holds, or that are held back, are in other queue folders
        sweep[row["state"] if row["state"] in unit_queue.STATES else "elsewhere"] += 1
        sweep["windows_written"] += row["windows_written"] or 0
        sweep["windows_expected"] += row["windows_expected"]
    return sweeps


def node_rows(rows):
    """Count, per node, the jobs that ran units there and how their attempts ended, for every submitter."""
    # before: a unit with history [{"hostname": "jaguar03", "job_id": "71", "account": "sl5nw",
    #         "outcome": "complete", "wall_seconds": 5400.0}]
    # after:  {"jaguar03": {"jobs": {"71"}, "running": 0, "done": 1, "failed": 0, "seconds": [5400.0],
    #                       "submitters": {"sl5nw"}}}
    nodes = {}
    for row in rows:
        unit = row["unit"]
        attempts = list(unit.get("history", []))
        # a unit claimed now is an open attempt; a unit a remote server finished carries that server's result
        if row["state"] == "running":
            attempts.append(dict(unit, outcome="running"))
        if "server" in unit.get("result", {}):
            attempts.append({"hostname": f"{unit['result']['server']} {unit.get('hostname')}", "outcome": "complete",
                             "job_id": unit.get("job_id"), "account": unit["result"]["server"],
                             "wall_seconds": unit["result"].get("wall_seconds")})
        for attempt in attempts:
            node = nodes.setdefault(attempt.get("hostname") or "(unknown)", {
                "jobs": set(), "running": 0, "done": 0, "failed": 0, "seconds": [], "submitters": set()})
            node["jobs"].add(str(attempt.get("job_id")))
            node["submitters"].add(str(attempt.get("account")))
            if attempt["outcome"] == "running":
                node["running"] += 1
            elif attempt["outcome"] == "complete":
                node["done"] += 1
                node["seconds"] += [attempt["wall_seconds"]] if attempt.get("wall_seconds") else []
            elif attempt["outcome"] in ("failed", "out_of_card_memory", "worker_error"):
                node["failed"] += 1
    return nodes


def warnings_of(run_dir, rows, live, submissions):
    """List what a reader must look at: no progress, failed units, units without a worker, problem reports."""
    warnings = [f"no progress for {STALE_SECONDS // 60} minutes while its job runs: {row['name']}"
                for row in rows if row["stale"]]
    warnings += [f"failed: {row['name']} ({row['unit'].get('failure_reason')}); log {row['unit'].get('failure_log')}"
                 for row in rows if row["state"] == "failed"]
    # pending units against worker cards that hold no unit, and against the largest UVA card
    pending = [row["unit"] for row in rows if row["state"] == "pending"]
    running = sum(row["state"] == "running" for row in rows)
    cards = sum(record["cards"] for record in submissions if record["job_id"] in live and record["kind"] == "worker")
    spare = max(0, cards - running)
    if len(pending) > spare:
        warnings.append(f"{len(pending) - spare} pending units have no worker job; submit workers "
                        "(submit.py --phase <phase>)")
    # a job pinned to a node was planned to start at once; one that still waits a pass later is re-placed
    for record in submissions:
        job = live.get(record["job_id"])
        waited = time.time() - datetime.fromisoformat(record["submitted_at"]).timestamp() if job else 0
        if job and record.get("node") and job[3] == "PENDING" and waited > PASS_SECONDS:
            warnings.append(f"job {record['job_id']} is pinned to {record['node']} and still waits after "
                            f"{int(waited // 60)} minutes ({job[5]}): cancel it by its id (cancel_ledger.py "
                            f"--ledger <its ledger> --job-id {record['job_id']}) and submit workers again")
    largest = max(record["gpu_mem_mb"] for record in common.read_json(load("submit").RESOURCE_FILE)["classes"])
    oversized = [unit["name"] for unit in pending if unit["minimum_device_memory_mb"] > largest]
    warnings += [f"needs more card memory than any UVA card has: {name}" for name in oversized]
    # a unit whose attempts ran under two code copies mixes them in one result
    for row in rows:
        commits = {attempt["source_commit"] for attempt in row["unit"].get("history", []) if "source_commit" in attempt}
        if len(commits) > 1:
            warnings.append(f"attempts under {len(commits)} code copies: {row['name']}")
    warnings += [f"collaborator problem report: {path.name}"
                 for path in sorted((Path(run_dir) / "for_collaborator/problems/open").glob("*")) if path.is_file()]
    return warnings


def table_lines(table):
    """Format rows of (run or job, pool, gpu, state, progress) as text of at most WIDTH characters per line."""
    # a group heading is a row with one cell; a long first cell goes on its own line above the other four
    lines = [f"{'run or job':<28}{'pool':<22}{'gpu':<18}{'state':<12}progress"]
    for row in table:
        if len(row) == 1:
            lines.append(row[0])
            continue
        name, rest = row[0], f"{row[1]:<22}{row[2]:<18}{row[3]:<12}"
        if len(name) < 28:
            first = f"{name:<28}{rest}"
        else:
            lines += textwrap.wrap(name, WIDTH)
            first = "    " + rest
        wrapped = textwrap.wrap(row[4], WIDTH - len(first)) or [""]
        lines.append((first + wrapped[0]).rstrip())
        lines += [(" " * len(first) + text).rstrip() for text in wrapped[1:]]
    return lines


def status_table(rows, sweeps, live, submissions):
    """The session status table: what trains, what waits, what finished, what is planned."""
    records = {record["job_id"]: record for record in submissions}
    names = {record["gres_type"]: record["display_name"]
             for record in common.read_json(load("submit").RESOURCE_FILE)["classes"]}

    def pool(job_id):
        """The partition and account of a job, as the report names a pool."""
        record = records.get(str(job_id))
        return f"UVA {record['partition']}, {record['account']}" if record else f"job {job_id}"

    table = [("TRAINING",)]
    # one line per unit that trains now, with the card its job holds
    for row in rows:
        if row["state"] == "running":
            unit = row["unit"]
            written = "an unknown number" if row["windows_written"] is None else row["windows_written"]
            table.append((row["name"], pool(unit.get("job_id")), "1 " + str(unit.get("card", "")).replace("NVIDIA ", ""),
                          "RUNNING", f"window {written} of {row['windows_expected']}, {unit['policies']} policies"
                          + (f"; NO PROGRESS for {STALE_SECONDS // 60} minutes" if row["stale"] else "")))
    for job_id, job in sorted(live.items()):
        if records.get(job_id, {}).get("kind") == "monitor" and job[3] == "RUNNING":
            table.append(("monitor of this run folder", pool(job_id), "none (CPU job)", "RUNNING",
                          "one pass per cadence"))
    # worker jobs that wait, one line per pool, card type and reason; they hold no unit
    table.append(("WAITING",))
    waiting = {}
    for job_id, job in sorted(live.items()):
        record = records.get(job_id)
        if record and record["kind"] == "worker" and job[3] != "RUNNING":
            key = (pool(job_id), names.get(record["gpu_type"], record["gpu_type"]), job[3], job[5])
            waiting[key] = waiting.get(key, 0) + record["cards"]
    table += [("worker jobs without a unit", where, f"asks {card}", f"{count} {state}", f"reason: {reason}")
              for (where, card, state, reason), count in waiting.items()]
    # per sweep: the units that are done, the units that failed, and what still has to start
    table.append(("FINISHED",))
    for (phase, environment, reward, method), sweep in sweeps.items():
        label = f"{phase}, {environment}, reward {reward.replace('_', ' ')}, {method}"
        total = sum(sweep[state] for state in ("pending", "running", "done", "failed", "elsewhere"))
        if sweep["done"]:
            table.append((label, "all pools", "-", "FINISHED", f"{sweep['done']} of {total} units done; windows "
                          f"{sweep['windows_written']} of {sweep['windows_expected']}"))
        if sweep["failed"]:
            table.append((label, "all pools", "-", "FAILED", f"{sweep['failed']} of {total} units failed; "
                          "they are named in the notes below"))
    table.append(("PLANNED",))
    for (phase, environment, reward, method), sweep in sweeps.items():
        if sweep["pending"] or sweep["elsewhere"]:
            label = f"{phase}, {environment}, reward {reward.replace('_', ' ')}, {method}"
            table.append((label, "any pool with room", "a card that fits", "PLANNED",
                          f"{sweep['pending']} units pending (start when a worker is free); "
                          f"{sweep['elsewhere']} held by a remote server or held back"))
    return table


def markdown_report(run_dir, sweeps, nodes, warnings, cadence, extra):
    """The one Markdown report of a pass: units by sweep, units by node, warnings, and the interim results."""
    lines = [f"# Monitoring report: {Path(run_dir).name}", "",
             f"Written {common.pacific()}. Times are Pacific, converted from machines that run on Eastern. "
             f"The monitor passes every {cadence // 60} minutes.", "",
             "## Units by sweep", "",
             "Every submitter's units are counted: the queue is shared by both accounts and the remote servers.", "",
             "| sweep | pending | running | done | failed | held by a remote server<br>or held back "
             "| windows written<br>of expected |",
             "|---|---|---|---|---|---|---|"]
    for (phase, environment, reward, method), sweep in sweeps.items():
        lines.append(f"| {phase}<br>{environment}<br>reward {reward.replace('_', ' ')}<br>{method} | "
                     f"{sweep['pending']} | {sweep['running']} | {sweep['done']} | {sweep['failed']} | "
                     f"{sweep['elsewhere']} | {sweep['windows_written']:,} of {sweep['windows_expected']:,} |")
    lines += ["", "## Units by node", "",
              "| node | jobs | units running now | units done | attempts failed "
              "| mean minutes<br>per done unit | submitters |",
              "|---|---|---|---|---|---|---|"]
    for name, node in sorted(nodes.items()):
        minutes = f"{sum(node['seconds']) / len(node['seconds']) / 60:.1f}" if node["seconds"] else "—"
        lines.append(f"| {name} | {len(node['jobs'])} | {node['running']} | {node['done']} | {node['failed']} | "
                     f"{minutes} | {', '.join(sorted(node['submitters']))} |")
    lines += ["", "## Warnings", ""] + ([f"{index}. {text}" for index, text in enumerate(warnings, 1)] or ["None."])
    if extra:
        lines += ["", "## Interim results", "", extra.rstrip()]
    return "\n".join(lines) + "\n"


def one_pass(run_dir, top_up_phase=None, complete_when_drained=False, report_command=None, gpu_types=None):
    """Do one monitoring pass; returns (whether the sweep is complete, seconds until the next pass)."""
    run_dir = Path(run_dir)
    if not (run_dir / "queue").exists():
        raise SystemExit(f"{run_dir} has no queue yet: build units first (build_units.py)")
    stopped = (run_dir / "STOP").exists()
    # 1. units of jobs that ended go back to pending; liveness is read by exact id from the ledgers
    live = common.own_live_jobs(run_dir)
    recovered = unit_queue.recover_claims(run_dir, live)
    # 2. pending units without a worker get worker jobs, when this monitor was asked to send them
    submitted = []
    if top_up_phase and not stopped and not (run_dir / "SWEEP_COMPLETE").exists():
        submitted = own("submit").top_up(run_dir, top_up_phase, gpu_types)
        live = common.own_live_jobs(run_dir)
    # 3. counts and per-unit progress
    state_path = run_dir / "monitor_state.json"
    previous = common.read_json(state_path) if state_path.exists() else {"units": {}, "passes": []}
    moment = time.time()
    rows = unit_rows(run_dir, previous["units"], moment)
    counts = unit_queue.counts(run_dir)
    submissions = [common.read_json(path) for path in (run_dir / "slurm/submissions").glob("*.json")]
    warnings = warnings_of(run_dir, rows, live, submissions)
    sweeps, nodes = sweep_rows(rows), node_rows(rows)
    # the sweep is complete when someone wrote the sentinel, or, if asked, when every unit is done or failed
    drained = bool(rows) and all(row["state"] in ("done", "failed") for row in rows)
    if complete_when_drained and drained and not (run_dir / "SWEEP_COMPLETE").exists():
        (run_dir / "SWEEP_COMPLETE").write_text(common.now() + "\n")
    complete = (run_dir / "SWEEP_COMPLETE").exists()

    # the cadence: hourly once three passes over more than an hour saw the same queue and the same
    # jobs with nothing to warn about, and 20 minutes again at any change
    signature = json.dumps([counts, sorted((job_id, job[3]) for job_id, job in live.items())], sort_keys=True)
    quiet = not (warnings or recovered or submitted)
    passes = (previous["passes"] + [{"at": moment, "signature": signature, "quiet": quiet}])[-3:]
    stable = (len(passes) == 3 and len({entry["signature"] for entry in passes}) == 1
              and all(entry["quiet"] for entry in passes) and passes[-1]["at"] - passes[0]["at"] >= STABLE_PASS_SECONDS)
    cadence = STABLE_PASS_SECONDS if stable else PASS_SECONDS
    common.write_json(state_path, {"passes": passes, "units": {
        row["name"]: {"bytes": row["bytes"], "unchanged_since": row["unchanged_since"], "job_id": row["job_id"]}
        for row in rows}})

    # 4. the status file, the table for the terminal, and the Markdown report
    extra = None
    if report_command:
        result = subprocess.run(shlex.split(report_command.format(run_dir=run_dir)), capture_output=True, text=True)
        # a report command that fails must not end the monitoring; its error text goes into the report
        extra = (result.stdout if result.returncode == 0
                 else f"The report command failed:\n\n```\n{result.stderr[-2000:]}\n```")
    common.write_json(run_dir / "status.json", {
        "at": common.now(), "counts": counts, "complete": complete, "stopped": stopped, "cadence_seconds": cadence,
        "recovered_units": recovered, "submitted_jobs": submitted, "warnings": warnings,
        "jobs": list(live.values()),
        "units": [{key: row[key] for key in ("name", "state", "windows_written", "windows_expected", "stale")}
                  for row in rows]})
    lines = table_lines(status_table(rows, sweeps, live, submissions))
    notes = [f"{index}. {text}" for index, text in enumerate(warnings, 1)]
    notes.append(f"Updated {common.pacific()}; the monitor passes every {cadence // 60} minutes"
                 + ("; RUN/STOP exists, no unit is claimed" if stopped else "")
                 + ("; the sweep is complete" if complete else ""))
    # the notes under the table keep to the table's width as well
    lines += [line for note in notes for line in textwrap.wrap(note, WIDTH, subsequent_indent="   ")]
    (run_dir / "status.txt").write_text("\n".join(lines) + "\n")
    stamp = datetime.now().astimezone(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d-%H-%M")
    report = run_dir / "20_mins_monitoring/outputs" / f"{stamp}_monitoring.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(markdown_report(run_dir, sweeps, nodes, warnings, cadence, extra))
    print("\n".join(lines), flush=True)
    return complete, cadence


def main(argv=None):
    """Run one pass, or passes until the sweep is complete."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--loop", action="store_true", help="pass again and again until RUN/SWEEP_COMPLETE exists")
    parser.add_argument("--top-up-phase", help="also submit worker jobs for pending units, recorded under this phase")
    parser.add_argument("--complete-when-drained", action="store_true",
                        help="write RUN/SWEEP_COMPLETE once every unit is done or failed")
    parser.add_argument("--report-command", help="a command whose Markdown output is added to each report")
    parser.add_argument("--cpus-per-card", type=int, help="processors each worker job of a top-up asks for per card")
    parser.add_argument("--memory-gib-per-card", type=int,
                        help="host memory in GiB each worker job of a top-up asks for per card")
    parser.add_argument("--worker-hours", type=float, help="time limit in hours of each worker job of a top-up")
    parser.add_argument("--accounts", help="comma-separated accounts a top-up submits under; default: both")
    parser.add_argument("--gpu-types", help="comma-separated card types (gres names) a top-up may ask for; "
                                            "default: every type whose card can hold a pending unit")
    args = parser.parse_args(argv)
    # the request of the workers this monitor submits: the same three settings as submit.py's own options
    own("submit").set_request_per_card(args.cpus_per_card, args.memory_gib_per_card)
    own("submit").set_worker_hours(args.worker_hours)
    own("submit").set_identities(args.accounts.split(",") if args.accounts else None)
    while True:
        started = time.monotonic()
        complete, cadence = one_pass(args.run_dir.resolve(), args.top_up_phase, args.complete_when_drained,
                                     args.report_command, args.gpu_types.split(",") if args.gpu_types else None)
        if complete or not args.loop:
            break
        time.sleep(max(0, cadence - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
