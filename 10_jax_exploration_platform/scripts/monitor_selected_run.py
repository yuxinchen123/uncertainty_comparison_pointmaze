"""Keep a selected training run monitored on a compute node and finalize its report."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from aggregate_selected_run import aggregate


def tick(run):
    """Read only this run's job ledgers, queue markers and streamed progress records."""
    summary = aggregate(run)
    ids = sorted({line.strip() for path in (run / "slurm").glob("submitted_jobids*.txt")
                  for line in path.read_text().splitlines() if line.strip().isdigit()})
    states = subprocess.run(["squeue", "-h", "-j", ",".join(ids),
                             "-o", "%i|%P|%T|%N|%R"], capture_output=True, text=True,
                            check=True).stdout if ids else ""
    counts = {name: len(list((run / "queue" / name).glob("*.json")))
              for name in ("pending", "running", "done", "failed")}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d-%H-%M")
    report = [f"# Selected-run status, {stamp} PT", "", "Cadence: 20 minutes.", "",
              f"Settings complete: {summary['units_complete']}/11. Queue: {counts}.", "",
              "```", "run or job           pool          gpu               state       progress"]
    # The hardware records carry the actual device; do not infer a card from a job request.
    for line in states.splitlines():
        job_id, pool, state, node, reason = line.split("|", 4)
        hardware = list((run / "slurm/jobs").glob(job_id + "_*/hardware.json"))
        gpu = json.loads(hardware[0].read_text()).get("gpu", "not yet recorded") if hardware else "none yet"
        report.append(f"{node[:20]:20} {pool[:13]:13} {gpu[:17]:17} {state[:11]:11} {reason[:25]}")
    report.append("```")
    report += ["", "| Setting | Iterations | Complete |", "|---|---:|---|"]
    for unit in summary["units"]:
        report.append(f"| {unit['unit_id']} | {unit['last_iteration']}/{unit['iterations']} | {unit['complete']} |")
    path = run / "20_mins_monitoring" / (stamp + "_report.md")
    path.write_text("\n".join(report) + "\n")
    print("\n".join(report[:8]), flush=True)
    return summary, counts


def main():
    """Monitor until all settings finish; publish the results, build the PDF, then stop."""
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--interval", type=int, default=1200)
    args = parser.parse_args()
    run = args.run_dir
    platform = Path(__file__).resolve().parent.parent
    tex = platform / "development_document/RND_JAX_exploration_development_document.tex"
    while True:
        summary, counts = tick(run)
        if counts["failed"]:
            (run / "logs/needs_attention.txt").write_text("Failed unit markers require diagnosis; no automatic application-error retry.\n")
        if summary["units_complete"] == 11:
            subprocess.run([sys.executable, str(platform / "scripts/write_corrected_report.py"),
                            "--run-dir", str(run), "--tex", str(tex)], check=True)
            subprocess.run(["/u/sl5nw/.TinyTeX/bin/x86_64-linux/latexmk", "-pdf",
                            "-interaction=nonstopmode", "-halt-on-error", tex.name],
                           cwd=tex.parent, check=True)
            (run / "SWEEP_COMPLETE").write_text(datetime.now().isoformat() + "\n")
            print("All 11 settings completed; report generated, PDF built, monitor stopped.", flush=True)
            return
        # This wait is inside a persistent compute job, not a blocking assistant tool call.
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
