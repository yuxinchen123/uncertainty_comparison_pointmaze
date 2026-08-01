#!/usr/bin/env python
"""ONE combined Markdown monitoring report for the train run 8.1.2 monitoring loop (the report of the
shared sweep-monitoring skill). Prints to stdout AND, with --snapshot, writes ONE file
20_mins_monitoring/outputs/<YYYY-MM-DD-HH-MM>_monitoring.md.

Sections:
  1. Title line (timestamp + sweep id).
  2. "## Running status" — the 10 (env_setup, arm) cells + TOTAL row of status_table, as Markdown.
  3. "## Per-node run counts (all submitters)" — one row per NODE any submitter's worker jobs ran on
     (job ids from slurm/submitted_jobids_<sid>.txt AND every
     for_collaborator/submitted_jobids_<sid>_<user>.txt, node and submitter from ONE `sacct -a -X`
     call, counts parsed from each job's worker log in logs/ or for_collaborator/logs/).
  4. "## Per-env interim metrics" — one table per env setup: each arm's current best config, ranked
     by whole-run reward, best value per metric column **bold** and second best <u>underlined</u>
     (the analysis-convention rule), with the env's frozen-bar line underneath.
  5. "## Stage-1 decision summary" — pruned / survivor counts from the decisions log.

The counting / scoring / record helpers are IMPORTED from the two CLI modules (status_table,
env_metrics_tables) and from slurm/{build_queue,stage1_controller} — the config and score source of
truth — so this report can never drift from the ad-hoc CLI tables or the controller.

Memory: records are reduced to four scalars at read time (env_metrics_tables.compact_record), never
held whole. Task-R (10M) records carry roughly ten times the episodes of a task-S (1M) record, so
this is what keeps the monitor job's memory bounded by the record COUNT, not the episode count.

RUN_DIR_OVERRIDE (env var, testing only): when set, queue/, data/, slurm/ and outputs/ are read and
written under that directory instead of the real run folder (same resolution as status_table).
"""
import argparse
import glob
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REAL_RUN_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)                                    # status_table / env_metrics_tables
sys.path.insert(0, os.path.join(REAL_RUN_DIR, "slurm"))     # build_queue / stage1_controller

import status_table        # noqa: E402  (build_rows / COLS / run_dir — the running-status counting)
import env_metrics_tables  # noqa: E402  (record loading, mean_se, cell_str, MCOLS, bar_line)
import build_queue         # noqa: E402  (ENV_SETUPS_RUN12 — config source of truth)

# Legend for the interim-metrics section, rendered as Markdown bullets.
LEGEND_BULLETS = [
    "- **whole-run reward** = mean per-episode extrinsic return over ALL training episodes of a run "
    "— the stage-1 racing score, so this column ranks configs exactly as the controller does.",
    "- Cells are `mean ± standard error` over the config's completed seeds; **completed seeds** is "
    "that count.",
    "- **verdict so far** = the config's stage-1 verdict (undecided / pruned / survivor); the "
    "task-R baseline arm is exempt from every decision, so its verdict is `N/A`.",
    "- Per metric column the best value is **bold** and the second best <u>underlined</u> (higher "
    "reward is better); the completed-seeds column is bookkeeping and is never marked.",
    "- The baseline row's runs are 10M-step task-R runs while every arm row is a 1M-step task-S run, "
    "and the frozen bar is the run-8.1 winner's 1M value — the baseline row is context, not a "
    "like-for-like comparison inside the table. Its `lr` is the adam rate of the run-8.1 "
    "original-small stack; every arm row's `lr` is a plain-SGD rate.",
]


def run_dir():
    """The run folder this report reads and writes (RUN_DIR_OVERRIDE redirects it for tests)."""
    return status_table.run_dir()


# ----------------------------------------------------------------------------------------------------
# Section 2 — running status (reuse status_table's counting; render as a Markdown table)
# ----------------------------------------------------------------------------------------------------
def md_status_table(sweep_id):
    """The 10 cell rows + TOTAL row from status_table.build_rows, as a Markdown pipe table. The TOTAL
    row's non-empty cells are bolded to set it off (Markdown has no mid-table rule)."""
    rows, total = status_table.build_rows(sweep_id)
    cols = status_table.COLS                                # (key, header, align)
    lines = ["| " + " | ".join(h for _, h, _ in cols) + " |"]
    lines.append("| " + " | ".join(":---" if a == "l" else "---:" for _, _, a in cols) + " |")
    # one data row per cell, then the bolded TOTAL row
    for r in rows:
        lines.append("| " + " | ".join(str(r[k]) for k, _, _ in cols) + " |")
    tcells = []
    for k, _, _ in cols:
        s = str(total[k])
        tcells.append(f"**{s}**" if s else s)              # never bold an empty cell (arm column)
    lines.append("| " + " | ".join(tcells) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------------
# Section 3 — per-node run counts (every submitter's worker jobs)
# ----------------------------------------------------------------------------------------------------
CLAIM_RE = re.compile(r"\] claimed \S+")            # worker.py: "] claimed <marker> :: ..."
DONE_RE = re.compile(r"rc=0 in (\d+)s -> done")     # worker.py finish (rc 0): captures N seconds
FAIL_RE = re.compile(r"-> failed")                  # worker.py finish (rc != 0)

NODE_COLS = [("node", "node", "l"), ("jobs", "jobs", "r"), ("running", "running_now", "r"),
             ("done", "done", "r"), ("failed", "failed", "r"), ("avg", "avg_finish_min", "r"),
             ("subs", "submitters", "l")]


def submitter_job_ids(sweep_id):
    """Ordered [(job_id, submitter_username)] over ALL submitters of this sweep, deduped on job id
    (owner first). Sources:
      - the owner's slurm/submitted_jobids_<sid>.txt            -> submitter "sl5nw"
      - every for_collaborator/submitted_jobids_<sid>_<user>.txt -> submitter "<user>" (from the name)
    The submitter here is a fallback; the authoritative one is sacct's User field (node_and_user_of_jobs).
    """
    pairs, seen = [], set()
    # owner file first, then every collaborator file (username parsed from the filename suffix)
    sources = [(os.path.join(run_dir(), "slurm", f"submitted_jobids_{sweep_id}.txt"), "sl5nw")]
    prefix = f"submitted_jobids_{sweep_id}_"
    for p in sorted(glob.glob(os.path.join(run_dir(), "for_collaborator", f"{prefix}*.txt"))):
        # basename "submitted_jobids_<sid>_yuxinchen.txt" -> user "yuxinchen"
        user = os.path.basename(p)[len(prefix):-len(".txt")]
        sources.append((p, user))
    for path, user in sources:
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            for line in fh:
                s = line.strip()
                if s.isdigit() and s not in seen:          # dedup, keep first-seen order
                    seen.add(s)
                    pairs.append((s, user))
    return pairs


def node_and_user_of_jobs(ids):
    """{job_id: (node_or_None, sacct_user_or_None)} via ONE sacct -a -X call over EVERY submitter's ids
    (-a lets a collaborator's job id resolve, not only the caller's).

    sacct -a -X -n -o JobID,NodeList%40,User%30 prints one allocation row per job; the User column is
    always the LAST token (so it survives a two-token "None assigned" NodeList):
      before: "6519874          bigcat03            yuxinchen "
      after : {"6519874": ("bigcat03", "yuxinchen")}   (NodeList "None assigned"/blank -> node None)
    """
    if not ids:
        return {}
    out = {}
    res = subprocess.run(
        ["sacct", "-a", "-j", ",".join(ids), "-X", "-n", "-o", "JobID,NodeList%40,User%30"],
        capture_output=True, text=True)
    for line in res.stdout.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        jid = parts[0]
        if len(parts) >= 3:                                # [jid, node, ..., user]
            node = None if parts[1] in ("None", "", "(null)") else parts[1]
            user = parts[-1]
        else:                                              # [jid, user] (blank NodeList)
            node, user = None, parts[1]
        out[jid] = (node, user)
    return out


def find_worker_log(job_id):
    """The worker log for a job id from EITHER the owner's logs/ OR the collaborators'
    for_collaborator/logs/ (both name their logs <...>_<jobid>.log). The leading "_" anchor keeps a
    shorter id from matching inside a longer id's filename; the monitor loop's own log is not a worker
    log, so it is skipped."""
    for d in (os.path.join(run_dir(), "logs"),
              os.path.join(run_dir(), "for_collaborator", "logs")):
        for m in sorted(glob.glob(os.path.join(d, f"*_{job_id}.log"))):
            if os.path.basename(m).startswith("monitor_loop_"):
                continue                                   # the monitor loop is not a worker job
            return m
    return None


def parse_worker_log(path):
    """(claims, done, failed, [finish_seconds]) from one worker log. Each worker line is exactly one of
    a claim / a done finish / a failed finish, so the counts are disjoint."""
    claims = done = failed = 0
    fin = []
    with open(path, errors="replace") as fh:
        for line in fh:
            # a claim line; a rc=0 finish (captures its seconds); or a rc!=0 finish
            if CLAIM_RE.search(line):
                claims += 1
            m = DONE_RE.search(line)
            if m:
                done += 1
                fin.append(int(m.group(1)))
            elif FAIL_RE.search(line):
                failed += 1
    return claims, done, failed, fin


def node_rows(sweep_id):
    """(rows, total_row) of the per-node table over ALL submitters' jobs. A job with no node mapping OR
    no worker log goes to the "(unknown)" bucket; its log (if any) is still parsed. Per node:
    running_now = claims - done - failed (floor 0); avg_finish_min = mean(finish_seconds)/60 over the
    node's done lines ("—" when none); submitters = the distinct usernames of the node's jobs."""
    pairs = submitter_job_ids(sweep_id)
    ids = [jid for jid, _ in pairs]
    file_user = {jid: u for jid, u in pairs}               # fallback submitter when sacct lacks the row
    nu = node_and_user_of_jobs(ids)
    # aggregate per bucket: {node: {jobs, claims, done, failed, fin[], subs{}}}
    agg = {}
    for jid in ids:
        node, sacct_user = nu.get(jid, (None, None))
        submitter = sacct_user or file_user.get(jid) or "?"   # sacct is authoritative, filename backs it
        log = find_worker_log(jid)
        bucket = node if (node and log) else "(unknown)"
        d = agg.setdefault(bucket, {"jobs": 0, "claims": 0, "done": 0, "failed": 0,
                                    "fin": [], "subs": set()})
        d["jobs"] += 1
        d["subs"].add(submitter)
        if log:
            c, dn, f, fin = parse_worker_log(log)
            d["claims"] += c
            d["done"] += dn
            d["failed"] += f
            d["fin"] += fin
    # rows sorted by node name, "(unknown)" always last; TOTAL sums every column
    names = sorted(k for k in agg if k != "(unknown)")
    if "(unknown)" in agg:
        names.append("(unknown)")
    rows = []
    tot = {"jobs": 0, "running": 0, "done": 0, "failed": 0, "fin": [], "subs": set()}
    for name in names:
        d = agg[name]
        running = max(0, d["claims"] - d["done"] - d["failed"])
        avg = f"{sum(d['fin']) / len(d['fin']) / 60:.1f}" if d["fin"] else "—"
        rows.append({"node": name, "jobs": d["jobs"], "running": running, "done": d["done"],
                     "failed": d["failed"], "avg": avg, "subs": ", ".join(sorted(d["subs"]))})
        tot["jobs"] += d["jobs"]
        tot["running"] += running
        tot["done"] += d["done"]
        tot["failed"] += d["failed"]
        tot["fin"] += d["fin"]
        tot["subs"] |= d["subs"]
    tot_avg = f"{sum(tot['fin']) / len(tot['fin']) / 60:.1f}" if tot["fin"] else "—"
    total_row = {"node": "TOTAL", "jobs": tot["jobs"], "running": tot["running"], "done": tot["done"],
                 "failed": tot["failed"], "avg": tot_avg, "subs": ", ".join(sorted(tot["subs"]))}
    return rows, total_row


def md_node_table(rows, total_row):
    """The per-node rows + a bolded TOTAL row, as a Markdown pipe table."""
    lines = ["| " + " | ".join(h for _, h, _ in NODE_COLS) + " |"]
    lines.append("| " + " | ".join(":---" if a == "l" else "---:" for _, _, a in NODE_COLS) + " |")
    if not rows:
        lines.append("| _(no worker jobs recorded yet)_ |  |  |  |  |  |  |")
        return "\n".join(lines)
    for r in rows:
        lines.append("| " + " | ".join(str(r[k]) for k, _, _ in NODE_COLS) + " |")
    lines.append("| " + " | ".join(f"**{total_row[k]}**" for k, _, _ in NODE_COLS) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------------
# Section 4 — per-env interim metrics (best bold / second underlined per metric column)
# ----------------------------------------------------------------------------------------------------
# (raw-value key, higher_is_better, display precision) — the marked metric columns, in table order.
MARK_SPEC = [("reward", True, 2)]


def mark_rows(rows):
    """In-place wrap the best cell (**bold**) and second-best cell (<u>underline</u>) of each marked
    metric column (the analysis-convention rule). Rows are ranked on the display-rounded mean so ties
    match what the reader sees; a row whose raw value is None is skipped for that column. The
    completed-seeds column, the verdict column and the label are never marked.
      before (reward col, rounded): rows -> [-679.99, -680.37, -681.01]
      after : -679.99 -> **...**, -680.37 -> <u>...</u>, -681.01 -> unchanged
    """
    for key, higher, prec in MARK_SPEC:
        present = [(i, round(r["_raw"][key], prec)) for i, r in enumerate(rows)
                   if r["_raw"][key] is not None]
        if not present:
            continue
        vals = sorted({v for _, v in present}, reverse=higher)   # best-first distinct rounded values
        best = vals[0]
        second = vals[1] if len(vals) > 1 else None
        for i, v in present:
            if v == best:
                rows[i][key] = f"**{rows[i][key]}**"
            elif second is not None and v == second:
                rows[i][key] = f"<u>{rows[i][key]}</u>"


def md_env_table(env_setup, rows, awaiting, bar_text):
    """One env's ranked+marked rows as a Markdown pipe table under a ### heading, with the frozen-bar
    line underneath and an italic note listing arms still awaiting a first completed record."""
    lines = [f"### {env_setup}", ""]
    if not rows:
        lines.append("_(no completed records yet)_")
        lines += ["", f"_{bar_text}_"]
        return "\n".join(lines)
    cols = env_metrics_tables.MCOLS
    lines.append("| " + " | ".join(h for _, h, _ in cols) + " |")
    lines.append("| " + " | ".join(":---" if a == "l" else "---:" for _, _, a in cols) + " |")
    for r in rows:
        lines.append("| " + " | ".join(str(r[k]) for k, _, _ in cols) + " |")
    lines += ["", f"_{bar_text}_"]
    if awaiting:
        lines += ["", f"_(awaiting first completed record: {', '.join(awaiting)})_"]
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------------
# Section 5 — stage-1 decision summary
# ----------------------------------------------------------------------------------------------------
def decision_counts(sweep_id):
    """(n_pruned, n_survivor) over the decided configs of slurm/stage1_decisions_<sid>.jsonl (both 0
    when the controller has not decided anything yet)."""
    verdicts = status_table.decision_verdicts(sweep_id)
    pruned = sum(1 for v in verdicts.values() if v == "pruned")
    survivors = sum(1 for v in verdicts.values() if v == "survivor")
    return pruned, survivors


# ----------------------------------------------------------------------------------------------------
# Report assembly
# ----------------------------------------------------------------------------------------------------
def build_report(sweep_id):
    """The full Markdown report string (all five sections)."""
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    out = [f"# Run 8.1.2 monitoring report — sweep_id `{sweep_id}` — {stamp}", ""]
    # 2. running status (task-S arms draw from pending_1m, the baseline arm from pending_10m)
    out += ["## Running status", "",
            "_Counts every submitter's runs by construction — the queue markers and the per-run "
            "records are shared across the owner and all collaborators. `pending` sums BOTH pools "
            "(`pending_1m` for the task-S arms, `pending_10m` for the task-R baseline); markers "
            "archived under `pruned/` are in no state column; `completed` counts the cell's "
            "completed per-run records. The baseline arm is exempt from stage-1 decisions (N/A)._",
            "",
            md_status_table(sweep_id), ""]
    # 3. per-node run counts (all submitters)
    nrows, ntot = node_rows(sweep_id)
    out += ["## Per-node run counts (all submitters)", "",
            "_All submitters included — job ids from this run's "
            f"`slurm/submitted_jobids_{sweep_id}.txt` AND every "
            f"`for_collaborator/submitted_jobids_{sweep_id}_*.txt`; worker logs parsed from both "
            "`logs/` and `for_collaborator/logs/`; node and submitter come from one `sacct` call._", "",
            md_node_table(nrows, ntot), ""]
    # 4. per-env interim metrics
    out += ["## Per-env interim metrics", "",
            "_Aggregates every submitter's completed records by construction (the per-run records "
            "are shared across the owner and all collaborators). One row per arm: that arm's "
            "current best config._", ""]
    out += LEGEND_BULLETS + [""]
    by_key = env_metrics_tables.load_completed(sweep_id)
    verdicts = status_table.decision_verdicts(sweep_id)
    for env_setup in build_queue.ENV_SETUPS_RUN12:
        rows, awaiting = env_metrics_tables.env_rows(env_setup, by_key, verdicts)
        mark_rows(rows)
        out += [md_env_table(env_setup, rows, awaiting,
                             env_metrics_tables.bar_line(env_setup, by_key)), ""]
    # 5. stage-1 decision summary
    pruned, survivors = decision_counts(sweep_id)
    n_configs = len(build_queue.CONFIGS)
    out += ["## Stage-1 decision summary", "",
            f"Task-S configs decided so far — pruned: **{pruned}**; survivors: **{survivors}**; "
            f"still racing: **{n_configs - pruned - survivors}** of {n_configs} "
            f"(log: `slurm/stage1_decisions_{sweep_id}.jsonl`).", ""]
    return "\n".join(out)


def write_snapshot(text):
    """Write the report to 20_mins_monitoring/outputs/<YYYY-MM-DD-HH-MM>_monitoring.md; returns the path."""
    out_dir = os.path.join(run_dir(), "20_mins_monitoring", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{time.strftime('%Y-%m-%d-%H-%M')}_monitoring.md")
    with open(path, "w") as fh:
        fh.write(text + "\n")
    return path


def main():
    """Print the combined Markdown report; with --snapshot also save one timestamped .md file."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_id", required=True)
    ap.add_argument("--snapshot", action="store_true")
    args = ap.parse_args()
    text = build_report(args.sweep_id)
    print(text)
    if args.snapshot:
        print(f"\n[snapshot] {write_snapshot(text)}")


if __name__ == "__main__":
    main()
