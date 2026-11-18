"""One monitoring pass over a pixel sweep run folder, or a loop of passes for a persistent Slurm job.

    python monitor.py --run-dir RUN                       one pass
    python monitor.py --run-dir RUN --loop [--top-up-phase PHASE] [--complete-when-drained]
                      [--only-cards-free-now] [--leave-cards N] [--gpu-types a,b] [--accounts ...] [--worker-hours H]
                      [--cancel-waiting-after S]

Each pass
  1. returns to pending the units whose Slurm job no longer runs (ids are read only from this run
     folder's own ledgers);
  2. with --top-up-phase, submits worker jobs for pending units that have no worker (submit.py);
  3. counts the units of every queue folder and reads each unit's progress from its run folder's
     progress.json (updates done of the unit's target);
  4. writes RUN/status.json, RUN/status.txt (the table for the terminal) and one Markdown report in
     RUN/20_mins_monitoring/outputs/;
  5. rebuilds RUN/early_signals.md (report.py, in its own process: a report that cannot be built
     is named in the warnings and never ends the monitoring).
A loop passes every 20 minutes, and every hour once three passes over more than an hour found
nothing changed and nothing to warn about. It ends by itself when RUN/SWEEP_COMPLETE exists. It
never stops a run, and cancels a job only through a top-up with --cancel-waiting-after (its own
jobs of that pass that did not start, by id). Standard library only.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import textwrap
import time
from zoneinfo import ZoneInfo

from campaign_imports import HERE, common, grid, maze, own, unit_queue
import pixel_queue

maze_monitor = maze("monitor")
PASS_SECONDS = 1200
STABLE_PASS_SECONDS = 3600
STALE_SECONDS = 2400    # a running unit whose update count did not grow for this long is named in the warnings
WIDTH = maze_monitor.WIDTH
NOT_UNITS = ("packed",)   # queue folders that hold records of singles a pack replaced, not units


def sweep_words(phase, method):
    """Name a phase and a method in words, as the tables show them."""
    # before: ("grid", "fast-last-layer") ; after: "grid, fast last layer"
    return f"{phase}, {own('report').method_words(method)}"


def unit_rows(run_dir, previous, moment):
    """One row per unit of every queue folder: where it is, its record, and its progress from its run folder."""
    # after: [{"name": "...", "state": "running", "update": 4120, "target_updates": 12207, "copies": 1,
    #          "seconds_per_update": 1.41, "unchanged_since": 1795000000.0, "stale": False, "unit": {its record}}, ...]
    rows = []
    for state in unit_queue.counts(run_dir):
        if state in NOT_UNITS:
            continue
        for unit in unit_queue.read_units(run_dir, state):
            progress = pixel_queue.read_progress(pixel_queue.records_folder(run_dir, state, unit["name"]))
            # an update count equal to the last pass's has not grown since `unchanged_since` (a new claim
            # starts the count again: a continued unit first loads its checkpoint and compiles)
            before = previous.get(unit["name"], {})
            same = before.get("update") == progress["update"] and before.get("job_id") == unit.get("job_id")
            unchanged_since = before.get("unchanged_since", moment) if same else moment
            rows.append({"name": unit["name"], "state": state, "update": progress["update"],
                         "target_updates": unit["target_updates"], "copies": unit["copies"],
                         "seconds_per_update": progress["seconds_per_update"], "unchanged_since": unchanged_since,
                         "job_id": unit.get("job_id"),
                         "stale": state == "running" and moment - unchanged_since >= STALE_SECONDS, "unit": unit})
    return rows


def sweep_rows(rows):
    """Count the units and runs of each phase and method by queue folder, with the updates done of all."""
    # after: {("grid", "fast-last-layer"): {"pending": 500, "running": 60, "done": 16, "failed": 0, "elsewhere": 0,
    #          "runs": 576, "updates_done": 912000, "updates_target": 7031232}}
    sweeps = {}
    for row in rows:
        unit = row["unit"]
        sweep = sweeps.setdefault((unit["phase"], unit["method"]), {
            "pending": 0, "running": 0, "done": 0, "failed": 0, "elsewhere": 0, "runs": 0, "updates_done": 0,
            "updates_target": 0})
        # units a remote server holds are in other queue folders
        sweep[row["state"] if row["state"] in unit_queue.STATES else "elsewhere"] += 1
        sweep["runs"] += row["copies"]
        done = row["target_updates"] if row["state"] == "done" else min(row["update"], row["target_updates"])
        sweep["updates_done"] += done * row["copies"]
        sweep["updates_target"] += row["target_updates"] * row["copies"]
    return sweeps


def warnings_of(run_dir, rows, live, submissions, profiles, waiting_cards):
    """List what a reader must look at: no progress, failed units, units without a worker, problem reports."""
    warnings = [f"no progress for {STALE_SECONDS // 60} minutes while its job runs: {row['name']}"
                for row in rows if row["stale"]]
    warnings += [f"failed: {row['name']} ({row['unit'].get('failure_reason')}); log {row['unit'].get('failure_log')}"
                 for row in rows if row["state"] == "failed"]
    # pending units of each profile against the waiting worker cards of that profile
    pending = [row["unit"] for row in rows if row["state"] == "pending"]
    submit = own("submit")
    for name, profile in profiles.items():
        without = len(submit.units_of_profile(pending, name, profile)) - waiting_cards.get(name, 0)
        if without > 0:
            # single units run on UVA's cards; four-copy units also go to the remote servers' H100 cards
            advice = ("submit workers (submit.py --phase <phase>)" if profile["copies"] == 1 else
                      "hand them to a ready remote server (remote_fleet.py hand-over-tick) or submit workers on UVA")
            warnings.append(f"{without} pending units of the profile {name!r} have no worker job; {advice}")
    # a job pinned to a node was planned to start at once; one that still waits a pass later is re-placed
    for record in submissions:
        job = live.get(record["job_id"])
        waited = time.time() - datetime.fromisoformat(record["submitted_at"]).timestamp() if job else 0
        if job and record.get("node") and job[3] == "PENDING" and waited > PASS_SECONDS:
            warnings.append(f"job {record['job_id']} is pinned to {record['node']} and still waits after "
                            f"{int(waited // 60)} minutes ({job[5]}): cancel it by its id (cancel_ledger.py "
                            f"--ledger <its ledger> --job-id {record['job_id']}) and submit workers again")
    # a unit whose attempts ran under two code copies mixes them in one result
    for row in rows:
        commits = {attempt["source_commit"] for attempt in row["unit"].get("history", []) if "source_commit" in attempt}
        if len(commits) > 1:
            warnings.append(f"attempts under {len(commits)} code copies: {row['name']}")
    warnings += [f"collaborator problem report: {path.name}"
                 for path in sorted((Path(run_dir) / "for_collaborator/problems/open").glob("*")) if path.is_file()]
    return warnings


def progress_words(row):
    """Say how far a unit is: updates of its target, its share, its copies and its speed."""
    # after: "update 4,120 of 12,207 (34%), 1 copy, 1.41 s per update"
    share = 100 * row["update"] / row["target_updates"]
    speed = f", {row['seconds_per_update']:.2f} s per update" if row["seconds_per_update"] else ""
    return (f"update {row['update']:,} of {row['target_updates']:,} ({share:.0f}%), {row['copies']} "
            f"{'copy' if row['copies'] == 1 else 'copies'}{speed}")


def status_table(rows, sweeps, live, submissions):
    """The session status table: what trains, what waits, what finished, what is planned."""
    records = {record["job_id"]: record for record in submissions}
    names = {record["gres_type"]: record["display_name"]
             for record in common.read_json(grid("submit").RESOURCE_FILE)["classes"]}

    def pool(job_id):
        """The partition and account of a job, as the report names a pool."""
        record = records.get(str(job_id))
        return f"UVA {record['partition']}, {record['account']}" if record else f"job {job_id}"

    table = [("TRAINING",)]
    # one line per unit that trains now, with the card its job holds; a unit on a remote server names the server
    for row in rows:
        unit = row["unit"]
        if row["state"] == "running":
            table.append((row["name"], pool(unit.get("job_id")), "1 " + str(unit.get("card", "")).replace("NVIDIA ", ""),
                          "RUNNING", progress_words(row)
                          + (f"; NO PROGRESS for {STALE_SECONDS // 60} minutes" if row["stale"] else "")))
        elif row["state"].startswith("remote_"):
            table.append((row["name"], row["state"][len("remote_"):], "1 card there", "ON SERVER",
                          progress_words(row) + " (as last copied from the server)"))
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
    # per phase and method: the units that are done, the units that failed, and what still has to start
    table.append(("FINISHED",))
    for (phase, method), sweep in sweeps.items():
        total = sum(sweep[state] for state in ("pending", "running", "done", "failed", "elsewhere"))
        if sweep["done"]:
            table.append((sweep_words(phase, method), "all pools", "-", "FINISHED", f"{sweep['done']} of {total} units done; "
                          f"updates {sweep['updates_done']:,} of {sweep['updates_target']:,} over its {sweep['runs']} runs"))
        if sweep["failed"]:
            table.append((sweep_words(phase, method), "all pools", "-", "FAILED", f"{sweep['failed']} of {total} units "
                          "failed; they are named in the notes below"))
    table.append(("PLANNED",))
    for (phase, method), sweep in sweeps.items():
        if sweep["pending"]:
            table.append((sweep_words(phase, method), "any pool with room", "a card that fits", "PLANNED",
                          f"{sweep['pending']} units pending (start when a worker of their profile is free)"))
    return table


def markdown_report(run_dir, sweeps, nodes, warnings, cadence):
    """The one Markdown report of a pass: units by phase and method, units by node, and the warnings."""
    lines = [f"# Monitoring report: {Path(run_dir).name}", "",
             f"Written {common.pacific()}. Times are Pacific, converted from machines that run on Eastern. "
             f"The monitor passes every {cadence // 60} minutes. Early results: `early_signals.md` in the run folder.", "",
             "## Units by phase and method", "",
             "Every submitter's units are counted: the queue is shared by both accounts and the remote servers.", "",
             "| phase<br>method | pending | running | done | failed | held by a<br>remote server | runs | updates done<br>of the target |",
             "|---|---|---|---|---|---|---|---|"]
    for (phase, method), sweep in sweeps.items():
        lines.append(f"| {phase}<br>{own('report').method_words(method)} | {sweep['pending']} | {sweep['running']} | {sweep['done']} | "
                     f"{sweep['failed']} | {sweep['elsewhere']} | {sweep['runs']} | "
                     f"{sweep['updates_done']:,} of<br>{sweep['updates_target']:,} |")
    lines += ["", "## Units by node", "",
              "| node | jobs | units running now | units done | attempts failed "
              "| mean minutes<br>per done unit | submitters |",
              "|---|---|---|---|---|---|---|"]
    for name, node in sorted(nodes.items()):
        minutes = f"{sum(node['seconds']) / len(node['seconds']) / 60:.1f}" if node["seconds"] else "—"
        lines.append(f"| {name} | {len(node['jobs'])} | {node['running']} | {node['done']} | {node['failed']} | "
                     f"{minutes} | {', '.join(sorted(node['submitters']))} |")
    lines += ["", "## Warnings", ""] + ([f"{index}. {text}" for index, text in enumerate(warnings, 1)] or ["None."])
    return "\n".join(lines) + "\n"


def rebuild_early_signals(run_dir):
    """Rebuild RUN/early_signals.md in a process of its own; returns None, or why it could not be built."""
    result = subprocess.run([sys.executable, str(HERE / "report.py"), "--run-dir", str(run_dir)],
                            capture_output=True, text=True)
    return None if result.returncode == 0 else result.stderr.strip()[-1500:]


def one_pass(run_dir, top_up_phase=None, complete_when_drained=False, plan=None, report=True):
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
        submitted = own("submit").top_up(run_dir, top_up_phase, **(plan or {}))
        live = common.own_live_jobs(run_dir)
    # 3. counts and per-unit progress
    state_path = run_dir / "monitor_state.json"
    previous = common.read_json(state_path) if state_path.exists() else {"units": {}, "passes": []}
    moment = time.time()
    rows = unit_rows(run_dir, previous["units"], moment)
    counts = unit_queue.counts(run_dir)
    submissions = [common.read_json(path) for path in (run_dir / "slurm/submissions").glob("*.json")]
    profiles = pixel_queue.read_profiles()
    waiting_cards = own("submit").waiting_cards_by_profile(
        run_dir, profiles, common.read_json(grid("submit").RESOURCE_FILE)["classes"], live)
    warnings = warnings_of(run_dir, rows, live, submissions, profiles, waiting_cards)
    # 5. the early-signals report, from the record files as they are now
    if report:
        problem = rebuild_early_signals(run_dir)
        if problem:
            warnings.append(f"early_signals.md could not be rebuilt at this pass: {problem}")
    sweeps, nodes = sweep_rows(rows), maze_monitor.node_rows(rows)
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
        row["name"]: {"update": row["update"], "unchanged_since": row["unchanged_since"], "job_id": row["job_id"]}
        for row in rows}})

    # 4. the status file, the table for the terminal, and the Markdown report
    common.write_json(run_dir / "status.json", {
        "at": common.now(), "counts": counts, "complete": complete, "stopped": stopped, "cadence_seconds": cadence,
        "recovered_units": recovered, "submitted_jobs": submitted, "warnings": warnings,
        "jobs": list(live.values()),
        "units": [{key: row[key] for key in ("name", "state", "update", "target_updates", "copies", "stale")}
                  for row in rows]})
    lines = maze_monitor.table_lines(status_table(rows, sweeps, live, submissions))
    notes = [f"{index}. {text}" for index, text in enumerate(warnings, 1)]
    notes.append(f"Updated {common.pacific()}; the monitor passes every {cadence // 60} minutes"
                 + ("; RUN/STOP exists, no unit is claimed" if stopped else "")
                 + ("; the sweep is complete" if complete else ""))
    # the notes under the table keep to the table's width as well
    lines += [line for note in notes for line in textwrap.wrap(note, WIDTH, subsequent_indent="   ")]
    (run_dir / "status.txt").write_text("\n".join(lines) + "\n")
    stamp = datetime.now().astimezone(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d-%H-%M")
    path = run_dir / "20_mins_monitoring/outputs" / f"{stamp}_monitoring.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_report(run_dir, sweeps, nodes, warnings, cadence))
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
    parser.add_argument("--no-report", action="store_true", help="do not rebuild early_signals.md at this pass")
    submit = own("submit")
    submit.add_plan_options(parser)
    args = parser.parse_args(argv)
    # the request of the workers this monitor submits: the same options as submit.py's own
    plan = submit.plan_arguments(args)
    while True:
        started = time.monotonic()
        complete, cadence = one_pass(args.run_dir.resolve(), args.top_up_phase, args.complete_when_drained, plan,
                                     not args.no_report)
        if complete or not args.loop:
            break
        time.sleep(max(0, cadence - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
