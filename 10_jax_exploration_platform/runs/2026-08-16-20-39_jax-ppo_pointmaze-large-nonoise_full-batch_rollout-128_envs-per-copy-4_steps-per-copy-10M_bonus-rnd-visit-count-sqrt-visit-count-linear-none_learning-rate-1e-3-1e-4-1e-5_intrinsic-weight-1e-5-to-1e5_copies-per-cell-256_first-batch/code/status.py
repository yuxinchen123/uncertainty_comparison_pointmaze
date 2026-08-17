"""The 20-minute tick's two tables: where every job of this run stands, and what it has measured.

Reads the run's OWN id file (`slurm/submitted_jobids.txt`) and asks the scheduler about those ids
and no others — never `squeue -u`, because the uid is shared with other sessions.

Run:
  PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python code/status.py
"""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

RUN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RUN_DIR / "code"))
PACIFIC = ZoneInfo("America/Los_Angeles")

from aggregate import cell_table, read_units, scored_windows  # noqa: E402


def own_job_ids() -> list:
    """The ids this run submitted, in submission order, deduplicated."""
    id_file = RUN_DIR / "slurm" / "submitted_jobids.txt"
    if not id_file.exists():
        return []
    seen, ids = set(), []
    for line in id_file.read_text().split():
        if line.strip() and line.strip() not in seen:
            seen.add(line.strip())
            ids.append(line.strip())
    return ids


def scheduler_state(job_ids: list) -> dict:
    """State, node and elapsed time of each id, from sacct, which also knows finished jobs."""
    if not job_ids:
        return {}
    output = subprocess.run(
        ["sacct", "-j", ",".join(job_ids), "-X", "-n", "-P",
         "--format=JobID,State,NodeList,Elapsed,ExitCode"],
        capture_output=True, text=True).stdout
    states = {}
    for line in output.splitlines():
        parts = line.split("|")
        if len(parts) >= 5:
            states[parts[0]] = {"state": parts[1], "node": parts[2], "elapsed": parts[3],
                                "exit": parts[4]}
    return states


def queue_state() -> dict:
    """Where each unit's marker currently sits; the folder is the state."""
    state = {}
    for folder in ("pending", "running", "done", "failed"):
        for path in (RUN_DIR / "queue" / folder).glob("*.json"):
            state[path.stem] = folder
    return state


def job_rows() -> list:
    """One row per submission: its id, the unit it owns, the card it got, and where it stands."""
    states = scheduler_state(own_job_ids())
    rows = []
    for job_dir in sorted((RUN_DIR / "slurm" / "jobs").glob("*")):
        assignment_file = job_dir / "assignment.json"
        if not assignment_file.exists():
            continue
        assignment = json.loads(assignment_file.read_text())
        hardware = json.loads((job_dir / "hardware.json").read_text())
        job_id = assignment["slurm_job_id"]
        scheduler = states.get(job_id, {"state": "not a scheduler job", "node": "", "elapsed": "",
                                        "exit": ""})
        rows.append({"job": job_id, "folder": job_dir.name, "mode": assignment["mode"],
                     "unit": assignment["unit_id"], "host": hardware["host"],
                     "card": hardware["nvidia_smi"].split(",")[0],
                     "state": scheduler["state"], "elapsed": scheduler["elapsed"]})
    return rows


def progress_rows() -> list:
    """One row per unit: how far its shard has got and how long the rest should take."""
    rows = []
    queue = queue_state()
    for unit_id, unit in sorted(read_units().items()):
        if unit["start"] is None:
            continue
        start = unit["start"]
        windows = unit["windows"]
        last = windows[-1] if windows else None
        target = start["iterations"]
        done_iterations = last["last_iteration"] if last else 0
        seconds = last["seconds_since_first_iteration"] if last else 0.0
        rate = (done_iterations / seconds) if seconds else 0.0
        rows.append({
            "unit": unit_id,
            "queue": queue.get(unit_id, "unknown"),
            "copies": start["copies"],
            "windows": len(windows),
            "iterations": f"{done_iterations}/{target}",
            "percent": 100.0 * done_iterations / target,
            "phase_blocked_windows": len(scored_windows(unit)),
            "minutes_elapsed": seconds / 60.0,
            "minutes_left": ((target - done_iterations) / rate / 60.0) if rate else None,
            "complete": unit["complete"] is not None,
        })
    return rows


def main() -> None:
    """Print the running-status table, the per-unit progress table, and the interim best cells."""
    now = datetime.now().astimezone(PACIFIC).strftime("%Y-%m-%d %H:%M PT")
    print(f"# {RUN_DIR.name}\n# status at {now}\n")

    print("## submissions")
    print(f"{'job':>10} {'mode':>6} {'host':>10} {'card':>22} {'state':>12} {'elapsed':>10}  unit")
    for row in job_rows():
        print(f"{row['job']:>10} {row['mode']:>6} {row['host']:>10} {row['card'][:22]:>22} "
              f"{row['state'][:12]:>12} {row['elapsed']:>10}  {row['unit'][:40]}")

    print("\n## units")
    print(f"{'queue':>8} {'copies':>7} {'windows':>8} {'iterations':>14} {'%':>6} "
          f"{'min elapsed':>12} {'min left':>9}  unit")
    for row in progress_rows():
        left = f"{row['minutes_left']:.1f}" if row["minutes_left"] is not None else "-"
        print(f"{row['queue']:>8} {row['copies']:>7} {row['windows']:>8} "
              f"{row['iterations']:>14} {row['percent']:>6.1f} {row['minutes_elapsed']:>12.1f} "
              f"{left:>9}  {row['unit'][:40]}")

    rows = cell_table(completed_only=False)
    if rows:
        print("\n## interim best cell per arm (completed and running units, phase-blocked windows)")
        print(f"{'bonus':>30} {'rate':>8} {'weight':>9} {'reward':>10} {'coverage %':>11} "
              f"{'success':>8}")
        best = {}
        for row in rows:
            if row["whole_run_reward"] is None:
                continue
            if row["bonus"] not in best or row["whole_run_reward"] > best[row["bonus"]]["whole_run_reward"]:
                best[row["bonus"]] = row
        for bonus, row in sorted(best.items()):
            print(f"{bonus:>30} {row['learning_rate']:>8.0e} {row['intrinsic_weight']:>9.0e} "
                  f"{row['whole_run_reward']:>10.4f} {row['coverage_percent']:>11.2f} "
                  f"{row['success_rate']:>8.3f}")


if __name__ == "__main__":
    main()
