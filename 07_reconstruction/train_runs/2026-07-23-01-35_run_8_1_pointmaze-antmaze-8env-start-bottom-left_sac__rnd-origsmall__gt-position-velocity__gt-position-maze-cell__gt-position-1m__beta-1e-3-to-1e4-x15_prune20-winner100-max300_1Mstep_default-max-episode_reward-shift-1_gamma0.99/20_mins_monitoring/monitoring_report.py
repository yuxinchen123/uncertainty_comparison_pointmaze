#!/usr/bin/env python
"""ONE combined Markdown monitoring report for the point maze + ant maze train run 1 monitoring loop.

Replaces the two separate text snapshots (status_table.py + env_metrics_tables.py --snapshot) with a
single Markdown document, per the shared sweep-monitoring skill. Prints to stdout AND, with
--snapshot, writes ONE file 20_mins_monitoring/outputs/<YYYY-MM-DD-HH-MM>_monitoring.md.

Sections:
  1. Title line (timestamp + sweep id).
  2. "## Running status" — the same 28-cell + TOTAL table as status_table.py, as a Markdown table.
  3. "## Per-node run counts (owner fleet)" — one row per NODE used by the owner's worker jobs
     (job ids from slurm/submitted_jobids_<sid>.txt, node via one sacct call, counts parsed from each
     job's worker log). Collaborator jobs are excluded (their logs live outside this run folder).
  4. "## Per-env interim metrics" — the 8 per-env tables (same revised columns as env_metrics_tables),
     ranked by whole-run reward, with the best value per metric column in **bold** and the second best
     <u>underlined</u> (the user's analysis-convention rule).
  5. "## Prune / decision summary" — counts of pruned / winner_only decisions.

The counting/scoring/record helpers are IMPORTED from the two CLI modules (status_table,
env_metrics_tables) and from slurm/{build_queue,prune_controller} — the config/score source of truth —
so this report can never drift from the ad-hoc CLI tables.

RUN_DIR_OVERRIDE (env var, testing only): when set, queue/, data/, slurm/ and outputs/ are read/written
under that directory instead of the real run folder (matches status_table / env_metrics_tables).
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REAL_RUN_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)                                    # status_table / env_metrics_tables
sys.path.insert(0, os.path.join(REAL_RUN_DIR, "slurm"))     # build_queue / prune_controller

import status_table        # noqa: E402  (build_rows / COLS — the running-status counting)
import env_metrics_tables  # noqa: E402  (record helpers, mean_se, cell_str, MCOLS — interim metrics)
import build_queue         # noqa: E402  (ENV_SETUPS_RUN1 — config source of truth)
import prune_controller    # noqa: E402  (score_of_record — the racing/ranking score)

DATA_RUN_DIR = os.environ.get("RUN_DIR_OVERRIDE", REAL_RUN_DIR)

# Legend for the interim-metrics section (revised 2026-07-24 defs, copied from env_metrics_tables), plus
# a marking note. Rendered as Markdown bullets.
LEGEND_BULLETS = [
    "- **reward** = whole-run mean per-episode extrinsic return (the RANKING column and the prune "
    "score).",
    "- **reward_last100ep** = mean per-episode return over the LAST 100 episodes of each run.",
    "- **success_rate** = fraction of the LAST 100 episodes reaching the goal.",
    "- **steps_to_goal** = mean steps on SUCCESSFUL episodes, whole run (zero-success seeds excluded; "
    "a sub-count shows as `(n=..)` when it differs from N).",
    "- **coverage** = cumulative visit-count coverage % at the END of the run (after its final "
    "episodes; a last-100-only coverage is not in the record format).",
    "- Cells are `mean ± standard-error`. Per metric column the best value is **bold** and the second "
    "best <u>underlined</u> — higher is better for every column EXCEPT `steps_to_goal` (lower is "
    "better); `N` (completed seeds) is bookkeeping and is never marked.",
]


# ----------------------------------------------------------------------------------------------------
# Section 2 — running status (reuse status_table's counting; render as a Markdown table)
# ----------------------------------------------------------------------------------------------------
def md_status_table(sweep_id):
    """The 28 cell rows + TOTAL row from status_table.build_rows, as a Markdown pipe table. The TOTAL
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
        tcells.append(f"**{s}**" if s else s)              # never bold an empty cell (algorithm col)
    lines.append("| " + " | ".join(tcells) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------------
# Section 3 — per-node run counts (owner worker jobs only)
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
    sources = [(os.path.join(DATA_RUN_DIR, "slurm", f"submitted_jobids_{sweep_id}.txt"), "sl5nw")]
    prefix = f"submitted_jobids_{sweep_id}_"
    for p in sorted(glob.glob(os.path.join(DATA_RUN_DIR, "for_collaborator", f"{prefix}*.txt"))):
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
    for d in (os.path.join(DATA_RUN_DIR, "logs"),
              os.path.join(DATA_RUN_DIR, "for_collaborator", "logs")):
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
# (fmt/raw key, higher_is_better, display precision) — the marked metric columns, in table order.
MARK_SPEC = [("reward", True, 2), ("reward100", True, 2), ("success", True, 3),
             ("steps", False, 1), ("mcov", True, 2), ("cov1m", True, 2)]


def build_metric_row(algorithm, config_key, recs):
    """One env-table row: the formatted "mean ± se" cell strings PLUS the raw per-metric means (kept
    under "_raw" so marking can rank columns), for the algorithm's current best config. Mirrors
    env_metrics_tables.build_row but exposes the raw means."""
    mse = env_metrics_tables.mean_se
    cs = env_metrics_tables.cell_str
    # reward: every record has a computable score by construction of load_completed
    reward_vals = [env_metrics_tables.score_of(r) for r in recs]
    r_mean, r_se, n = mse(reward_vals)
    # reward over the LAST 100 episodes of each record
    r100_vals = [v for r in recs if (v := env_metrics_tables.record_reward_last100(r)) is not None]
    r100_mean, r100_se, _ = mse(r100_vals)
    # success rate over the LAST 100 episodes of each record
    succ_vals = [v for r in recs if (v := env_metrics_tables.record_success_rate(r)) is not None]
    s_mean, s_se, _ = mse(succ_vals)
    # steps-to-goal over records with at least one success (others excluded)
    steps_vals = [v for r in recs if (v := env_metrics_tables.record_steps_to_goal(r)) is not None]
    st_mean, st_se, st_n = mse(steps_vals)
    steps_txt = cs(st_mean, st_se, 1)
    if st_n and st_n != n:                                  # surface the sub-count when it differs
        steps_txt += f" (n={st_n})"
    # coverage from the last eval row of records that logged it
    mcov_vals = [v for r in recs
                 if (v := env_metrics_tables.record_last_eval(r, "visit_counts/coverage_pct")) is not None]
    mc_mean, mc_se, _ = mse(mcov_vals)
    cov1m_vals = [v for r in recs
                  if (v := env_metrics_tables.record_last_eval(r, "visit_counts_1m/coverage_pct")) is not None]
    c1_mean, c1_se, _ = mse(cov1m_vals)
    return {
        "label": env_metrics_tables.row_label(algorithm, config_key),
        "reward": cs(r_mean, r_se, 2),
        "reward100": cs(r100_mean, r100_se, 2),
        "success": cs(s_mean, s_se, 3),
        "steps": steps_txt,
        "mcov": cs(mc_mean, mc_se, 2),
        "cov1m": cs(c1_mean, c1_se, 2),
        "N": str(n),
        "_reward_mean": r_mean,                            # ranking key (whole-run reward)
        "_raw": {"reward": r_mean, "reward100": r100_mean, "success": s_mean,
                 "steps": st_mean, "mcov": mc_mean, "cov1m": c1_mean},
    }


def metric_rows(env_setup, by_key):
    """(rows, awaiting) for one env: one row per algorithm's current best config, ranked by whole-run
    reward (best first); `awaiting` lists the env's algorithms with no completed record yet."""
    rows, awaiting = [], []
    for algorithm in env_metrics_tables.env_algorithms(env_setup):
        keys = env_metrics_tables.algo_config_keys(env_setup, algorithm)
        best_k, recs = env_metrics_tables.best_config(by_key, keys)
        if best_k is None:
            awaiting.append(algorithm)
            continue
        rows.append(build_metric_row(algorithm, best_k, recs))
    rows.sort(key=lambda r: r["_reward_mean"], reverse=True)
    return rows, awaiting


def mark_rows(rows):
    """In-place wrap the best cell (**bold**) and second-best cell (<u>underline</u>) of each marked
    metric column (analysis-convention rule). Rows are ranked on the display-rounded mean so ties match
    what the reader sees; a row whose raw value is None (e.g. steps with no success) is skipped for that
    column. The N column and the label are never marked.
      before (reward col, rounded): rows -> [-79.99, -80.37, -81.01]
      after : -79.99 -> **...**, -80.37 -> <u>...</u>, -81.01 -> unchanged
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


def md_env_table(env_setup, rows, awaiting):
    """One env's ranked+marked rows as a Markdown pipe table under a ### heading; a trailing italic note
    lists algorithms still awaiting a first completed record."""
    lines = [f"### {env_setup}"]
    if not rows:
        lines.append("")
        lines.append("_(no completed records yet)_")
        return "\n".join(lines)
    cols = env_metrics_tables.MCOLS
    lines.append("")
    lines.append("| " + " | ".join(h for _, h, _ in cols) + " |")
    lines.append("| " + " | ".join(":---" if a == "l" else "---:" for _, _, a in cols) + " |")
    for r in rows:
        lines.append("| " + " | ".join(str(r[k]) for k, _, _ in cols) + " |")
    if awaiting:
        lines.append("")
        lines.append(f"_(awaiting first completed record: {', '.join(awaiting)})_")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------------
# Section 5 — prune / decision summary
# ----------------------------------------------------------------------------------------------------
def decision_counts(sweep_id):
    """(n_pruned, n_winner_only) from slurm/prune_decisions_<sid>.jsonl (both 0 when the file is absent)."""
    path = os.path.join(DATA_RUN_DIR, "slurm", f"prune_decisions_{sweep_id}.jsonl")
    pruned = winner = 0
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("verdict") == "pruned":
                    pruned += 1
                elif d.get("verdict") == "winner_only":
                    winner += 1
    return pruned, winner


# ----------------------------------------------------------------------------------------------------
# Report assembly
# ----------------------------------------------------------------------------------------------------
def build_report(sweep_id):
    """The full Markdown report string (all five sections)."""
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    out = [f"# Run 8.1 monitoring report — sweep_id `{sweep_id}` — {stamp}", ""]
    # 2. running status
    out += ["## Running status", "",
            "_Counts every submitter's runs by construction — the queue markers and the per-run "
            "records are shared across the owner and all collaborators._", "",
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
            "_Aggregates every submitter's completed records by construction (the per-run records are "
            "shared across the owner and all collaborators)._", ""]
    out += LEGEND_BULLETS + [""]
    by_key = env_metrics_tables.load_completed(sweep_id)
    for env_setup in build_queue.ENV_SETUPS_RUN1:
        rows, awaiting = metric_rows(env_setup, by_key)
        mark_rows(rows)
        out += [md_env_table(env_setup, rows, awaiting), ""]
    # 5. prune / decision summary
    pruned, winner = decision_counts(sweep_id)
    out += ["## Prune / decision summary", "",
            f"Prune decisions so far: **{pruned}** pruned, **{winner}** winner_only "
            f"(from `slurm/prune_decisions_{sweep_id}.jsonl`).", ""]
    return "\n".join(out)


def main():
    """Print the combined Markdown report; with --snapshot also save one timestamped .md file."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_id", required=True)
    ap.add_argument("--snapshot", action="store_true")
    args = ap.parse_args()
    text = build_report(args.sweep_id)
    print(text)
    if args.snapshot:
        out_dir = os.path.join(DATA_RUN_DIR, "20_mins_monitoring", "outputs")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{time.strftime('%Y-%m-%d-%H-%M')}_monitoring.md")
        with open(path, "w") as fh:
            fh.write(text + "\n")
        print(f"\n[snapshot] {path}")


if __name__ == "__main__":
    main()
