#!/usr/bin/env python
"""Optuna controller for Train run 3.2.1: applies the absolute-bar stopping rule to the file queue.

Role (never submits Slurm jobs, never adds queue entries — it only REMOVES pending work and records
decisions). Every cycle (10 min):

1. Freeze the bar on first start: recompute run 3.1.1's best `rnd_next_state` mean final
   training-episode reward from that run's JSONs (36.95 at the 2026-07-02 aggregation) and write it
   to optuna/frozen_bar.json; later starts reuse the frozen value.
2. Read every finished per-run JSON under data/<sweep_id>/local/ (completed=true only; partial
   checkpoints are infrastructure-killed attempts and never enter a mean). Score = the last
   train_history row's train/mean_extrinsic_reward (the 1M-step windowed training reward).
3. Per configuration with n >= 10 finished seeds: U = mean + 2.576*s/sqrt(n) (99% upper confidence
   limit). If U < bar, move ALL of that configuration's remaining queue/<sweep_id>/pending/ entries
   to queue/<sweep_id>/pruned/ (atomic renames; a worker claiming one at the same instant is
   harmless — one extra seed runs) and tell the optuna trial COMPLETE with the mean.
4. A configuration that reached the 50-seed cap, or has no pending/running entries left, is told
   COMPLETE with its final mean (state FAIL if it somehow produced no data at all).
5. Append every decision to optuna/decisions.jsonl and rewrite optuna/progress.txt.

Restart-safe by construction: all state is rebuilt each cycle from the filesystem and the optuna
journal (optuna/optuna_journal.log; three studies 3_2_1_adam / 3_2_1_adagrad / 3_2_1_sgd1t, one
trial per configuration, created in build_queue.CONFIGS order via enqueue_trial + ask). The sampler
decides nothing — every trial's parameters are enqueued; optuna is the crash-safe decision ledger.

Usage:  python optuna_controller.py --sweep_id <id> [--once] [--poll_seconds 600]
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import optuna
from optuna.distributions import CategoricalDistribution
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import TrialState

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import build_queue  # noqa: E402  (CONFIGS order + config_key are the shared source of truth)

optuna.logging.set_verbosity(optuna.logging.WARNING)

Z99 = 2.576                    # two-sided 99% normal value (user decision 2026-07-04: 99%, not 95%)
N_FLOOR = 10                   # minimum finished seeds before any stop decision
SEED_CAP = len(build_queue.SEEDS)   # 50
BAR_REFERENCE = 36.95          # run-3.1.1 best rnd_next_state mean at the 2026-07-02 aggregation
RUN311_LOCAL = os.path.join(
    os.path.dirname(RUN_DIR),
    "2026-06-26-05-15_run_3_1_1_pytorch_4algo_betagrid_ridge1-1e-2_input-sa-nexts_1Mstep_100seed_trainrewardonly",
    "data", "2026-06-26-05-45_4algo-betaridge-input", "local")
METHODS = ("adam", "adagrad", "sgd1t")
OPTUNA_DIR = os.path.join(RUN_DIR, "optuna")
JOURNAL = os.path.join(OPTUNA_DIR, "optuna_journal.log")
TRIAL_MAP = os.path.join(OPTUNA_DIR, "trial_map.json")
FROZEN_BAR = os.path.join(OPTUNA_DIR, "frozen_bar.json")
DECISIONS = os.path.join(OPTUNA_DIR, "decisions.jsonl")
PROGRESS = os.path.join(OPTUNA_DIR, "progress.txt")

# id -> config_key lookup: run ids are seed-outermost, so id % 184 indexes build_queue.CONFIGS
KEY_OF_INDEX = [build_queue.config_key(c["params"], c["beta"]) for c in build_queue.CONFIGS]
INDEX_OF_KEY = {k: i for i, k in enumerate(KEY_OF_INDEX)}
N_CONFIGS = len(KEY_OF_INDEX)
WIDTH = len(str(build_queue.RUN_TOTAL))


def log(msg):
    """Print a timestamped controller line, flushed (the slurm .log is the live decision trace)."""
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')} controller] {msg}", flush=True)


def append_decision(entry):
    """Append one decision dict as a JSON line to optuna/decisions.jsonl (the durable manifest)."""
    entry = dict(entry)
    entry["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(DECISIONS, "a") as fh:
        fh.write(json.dumps(entry) + "\n")


def final_train_reward(record):
    """Extract a record's score: the last train_history row's train/mean_extrinsic_reward, or None."""
    rows = record.get("train_history") or []
    if not rows:
        return None
    return rows[-1].get("train/mean_extrinsic_reward")


def freeze_bar():
    """Load the frozen bar, or on first start recompute it from the run-3.1.1 JSONs and freeze it.

    Recompute = run 3.1.1's rnd_next_state arm, grouped by beta, each run scored by its final
    training-episode reward; the bar is the best beta's mean (36.95/SE 3.92/n=48 at the 2026-07-02
    aggregation; the sweep kept filling afterwards, so the frozen value uses every finished seed)."""
    if os.path.exists(FROZEN_BAR):
        with open(FROZEN_BAR) as fh:
            return json.load(fh)["bar"]
    # group run-3.1.1 rnd_next_state finals by beta (missing 'completed' = old write-once = complete)
    by_beta = {}
    for path in glob.glob(os.path.join(RUN311_LOCAL, "*.json")):
        with open(path) as fh:
            rec = json.load(fh)
        if rec.get("algorithm") != "rnd_next_state" or not rec.get("completed", True):
            continue
        score = final_train_reward(rec)
        if score is not None:
            by_beta.setdefault(float(rec["beta"]), []).append(float(score))
    if not by_beta:
        raise RuntimeError(f"no run-3.1.1 rnd_next_state records found under {RUN311_LOCAL}")
    # bar = the best beta's mean; keep n and SE for the record
    beta_star, rewards = max(by_beta.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))
    n = len(rewards)
    mean = sum(rewards) / n
    sd = math.sqrt(sum((r - mean) ** 2 for r in rewards) / (n - 1)) if n > 1 else float("nan")
    frozen = {"bar": mean, "beta_star": beta_star, "n": n, "se": sd / math.sqrt(n),
              "reference_2026_07_02": BAR_REFERENCE,
              "source": RUN311_LOCAL, "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    os.makedirs(OPTUNA_DIR, exist_ok=True)
    with open(FROZEN_BAR, "w") as fh:
        json.dump(frozen, fh, indent=1)
    append_decision({"action": "freeze_bar", **frozen})
    # the run's background doc records the frozen value next to the 36.95 reference
    with open(os.path.join(RUN_DIR, "experiment_background.md"), "a") as fh:
        fh.write(f"\n**Frozen bar (controller, {frozen['frozen_at']}):** {mean:.4f} "
                 f"(beta*={beta_star:g}, n={n}, SE={frozen['se']:.3f}; reference 36.95 was the "
                 f"2026-07-02 aggregation at n=48).\n")
    log(f"froze bar = {mean:.4f} (beta*={beta_star:g}, n={n}); reference {BAR_REFERENCE}")
    return mean


def open_storage():
    """A fresh handle to the shared journal-file storage (NFS-safe lock)."""
    os.makedirs(OPTUNA_DIR, exist_ok=True)
    return JournalStorage(JournalFileBackend(JOURNAL, lock_obj=JournalFileOpenLock(JOURNAL)))


def method_configs(method):
    """The (config_index, key) list of one method's configurations, in CONFIGS order."""
    return [(i, k) for i, k in enumerate(KEY_OF_INDEX) if k.startswith(method + "|")]


def bootstrap_studies():
    """Create/load the three per-method studies and the config_key -> (study, trial_number) map.

    First start: one trial per configuration via enqueue_trial + ask (ask consumes the enqueue queue
    in order, so trial numbers follow CONFIGS order); config_key stored as a trial user attribute.
    Later starts: reuse trial_map.json; a map/journal count mismatch fails loud."""
    storage = open_storage()
    studies = {m: optuna.create_study(study_name=f"3_2_1_{m}", direction="maximize",
                                      sampler=optuna.samplers.RandomSampler(seed=0),
                                      storage=storage, load_if_exists=True)
               for m in METHODS}
    if os.path.exists(TRIAL_MAP):
        with open(TRIAL_MAP) as fh:
            tmap = json.load(fh)
        for m in METHODS:
            expected = len(method_configs(m))
            got = len(studies[m].trials)
            if got != expected:
                raise RuntimeError(f"study 3_2_1_{m} has {got} trials, expected {expected} "
                                   f"(trial_map.json exists — journal/map mismatch)")
        return studies, tmap
    # first start: every study must be empty, then one enqueued+asked trial per configuration
    for m in METHODS:
        if studies[m].trials:
            raise RuntimeError(f"study 3_2_1_{m} already has trials but {TRIAL_MAP} is missing")
    tmap = {}
    for m in METHODS:
        for _, key in method_configs(m):
            opt, readout, eta0, t0, beta = key.split("|")
            params = {"readout": readout, "beta": float(beta)}
            if m == "sgd1t":
                params.update({"eta0": float(eta0), "t0": float(t0)})
            studies[m].enqueue_trial(params)
            trial = studies[m].ask()
            # fix the parameters (returns the enqueued values, in enqueue order)
            trial.suggest_categorical("readout", ["mse", "l2"])
            trial.suggest_float("beta", 1e-3, 1e4, log=True)
            if m == "sgd1t":
                trial.suggest_float("eta0", 1e-3, 1e-1, log=True)
                trial.suggest_categorical("t0", [1000.0, 10000.0])
            trial.set_user_attr("config_key", key)
            tmap[key] = {"study": m, "trial_number": trial.number}
    with open(TRIAL_MAP + ".tmp", "w") as fh:
        json.dump(tmap, fh, indent=0)
    os.replace(TRIAL_MAP + ".tmp", TRIAL_MAP)
    log(f"bootstrapped studies: {[f'{m}:{len(studies[m].trials)}' for m in METHODS]}")
    return studies, tmap


def record_key(rec):
    """Canonical config key of one result JSON (matches build_queue.config_key exactly).
    before: rec has rnd_optimizer='sgd1t', rnd_bonus_readout='l2', rnd_sgd_eta0=0.003,
    rnd_sgd_t0=1000.0, beta=0.01  ->  after: 'sgd1t|l2|0.003|1000|0.01'."""
    opt = rec["rnd_optimizer"]
    eta0 = "%g" % float(rec["rnd_sgd_eta0"]) if opt == "sgd1t" else "-"
    t0 = "%g" % float(rec["rnd_sgd_t0"]) if opt == "sgd1t" else "-"
    return f"{opt}|{rec['rnd_bonus_readout']}|{eta0}|{t0}|%g" % float(rec["beta"])


def scan_results(sweep_id):
    """Read every finished result JSON; return ({key: [reward, ...]}, n_partial, n_unscorable)."""
    rewards_by_key, n_partial, n_unscorable = {}, 0, 0
    for path in glob.glob(os.path.join(RUN_DIR, "data", sweep_id, "local", "*.json")):
        with open(path) as fh:
            rec = json.load(fh)
        # completed=false checkpoints are killed/in-flight attempts: never in a mean, counted apart
        if not rec.get("completed", True):
            n_partial += 1
            continue
        score = final_train_reward(rec)
        if score is None:
            n_unscorable += 1
            continue
        rewards_by_key.setdefault(record_key(rec), []).append(float(score))
    return rewards_by_key, n_partial, n_unscorable


def scan_queue(sweep_id):
    """Count queue entries per config key and state. Returns {state: {key: count}} for the five
    marker folders (an entry's key comes from its leading run id: id % N_CONFIGS indexes CONFIGS)."""
    counts = {}
    for state in ("pending", "running", "done", "failed", "pruned"):
        per_key = {}
        folder = os.path.join(RUN_DIR, "queue", sweep_id, state)
        for name in os.listdir(folder):
            if not name.endswith(".json"):
                continue
            run_id = int(name.split("_", 1)[0])
            key = KEY_OF_INDEX[run_id % N_CONFIGS]
            per_key[key] = per_key.get(key, 0) + 1
        counts[state] = per_key
    return counts


def compute_stats(rewards):
    """(n, mean, sd, U): sample stats and the 99% upper confidence limit U = mean + 2.576*s/sqrt(n)."""
    n = len(rewards)
    mean = sum(rewards) / n
    if n < 2:
        return n, mean, float("inf"), float("inf")
    sd = math.sqrt(sum((r - mean) ** 2 for r in rewards) / (n - 1))
    return n, mean, sd, mean + Z99 * sd / math.sqrt(n)


def decide(rewards_by_key, queue_counts, bar, already_decided):
    """The stopping rule, pure (unit-tested). Returns [(key, action, stats_dict), ...] where action is
    'stop' (bar rule fired: n >= N_FLOOR and U < bar), 'complete' (reached the seed cap, or no
    pending+running work remains), or 'exhausted' (no work left AND no finished data at all)."""
    decisions = []
    pending, running = queue_counts.get("pending", {}), queue_counts.get("running", {})
    for key in KEY_OF_INDEX:
        if key in already_decided:
            continue
        rewards = rewards_by_key.get(key, [])
        left = pending.get(key, 0) + running.get(key, 0)
        if rewards:
            n, mean, sd, upper = compute_stats(rewards)
            stats = {"n": n, "mean": mean, "sd": sd, "U": upper, "bar": bar}
            # the bar rule: only from the seed floor on, and only when the whole 99% interval
            # has moved below the bar (stopping is the pessimistic-certain direction)
            if n >= N_FLOOR and upper < bar:
                decisions.append((key, "stop", stats))
            elif n >= SEED_CAP or left == 0:
                decisions.append((key, "complete", stats))
        else:
            # no finished data for this key: 'exhausted' only when entries existed and were all
            # consumed (done/failed/pruned) with nothing to show (e.g. every seed failed)
            consumed = sum(queue_counts.get(state, {}).get(key, 0)
                           for state in ("done", "failed", "pruned"))
            if left == 0 and consumed > 0:
                decisions.append((key, "exhausted", {"n": 0}))
    return decisions


def prune_pending(sweep_id, key):
    """Move ALL of one configuration's remaining pending entries to pruned/. Returns moved count.
    Atomic renames; a worker that claims a file in the same instant simply runs one extra seed."""
    idx = INDEX_OF_KEY[key]
    pending = os.path.join(RUN_DIR, "queue", sweep_id, "pending")
    pruned = os.path.join(RUN_DIR, "queue", sweep_id, "pruned")
    moved = 0
    for seed in build_queue.SEEDS:
        run_id = seed * N_CONFIGS + idx
        for path in glob.glob(os.path.join(pending, f"{run_id:0{WIDTH}d}_of_{build_queue.RUN_TOTAL}_*.json")):
            try:
                os.rename(path, os.path.join(pruned, os.path.basename(path)))
                moved += 1
            except OSError:
                pass  # a worker claimed it first; that seed just runs
    return moved


def decided_keys(studies, tmap):
    """Config keys whose optuna trial is already finished (COMPLETE or FAIL) — decisions are final."""
    done = set()
    for m in METHODS:
        finished = {t.number for t in studies[m].get_trials(
            deepcopy=False, states=(TrialState.COMPLETE, TrialState.FAIL))}
        for key, info in tmap.items():
            if info["study"] == m and info["trial_number"] in finished:
                done.add(key)
    return done


def write_progress(sweep_id, rewards_by_key, queue_counts, already_decided, bar, n_partial):
    """Rewrite optuna/progress.txt: per-method config states + queue totals (the 10-min snapshot)."""
    lines = [f"run 3.2.1 progress @ {time.strftime('%Y-%m-%dT%H:%M:%S')}  bar={bar:.4f}  "
             f"partial-checkpoints={n_partial}"]
    for m in METHODS:
        keys = [k for _, k in method_configs(m)]
        finished_runs = sum(len(rewards_by_key.get(k, [])) for k in keys)
        racing = [k for k in keys if k not in already_decided]
        lines.append(f"  {m:8s}: configs {len(keys):3d}  decided {len(keys) - len(racing):3d}  "
                     f"racing {len(racing):3d}  finished-runs {finished_runs}")
    totals = {s: sum(queue_counts.get(s, {}).values()) for s in
              ("pending", "running", "done", "failed", "pruned")}
    lines.append(f"  queue: {totals}")
    with open(PROGRESS + ".tmp", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    os.replace(PROGRESS + ".tmp", PROGRESS)
    return lines


def cycle(sweep_id, studies, tmap, bar):
    """One controller pass: scan, decide, prune, tell, record. Returns the number of decisions."""
    # queue first, results second: a run finishing between the two scans then shows up in the
    # results (its JSON is written before its marker moves), never the other way round — so a
    # 'left == 0' completion decision can never miss that run's reward
    queue_counts = scan_queue(sweep_id)
    rewards_by_key, n_partial, n_unscorable = scan_results(sweep_id)
    already = decided_keys(studies, tmap)
    decisions = decide(rewards_by_key, queue_counts, bar, already)
    for key, action, stats in decisions:
        info = tmap[key]
        study = studies[info["study"]]
        moved = prune_pending(sweep_id, key) if action == "stop" else 0
        # tell exactly once per configuration: value = mean over its finished seeds
        if action == "exhausted":
            study.tell(info["trial_number"], state=TrialState.FAIL)
        else:
            study.tell(info["trial_number"], stats["mean"])
        append_decision({"config_key": key, "action": action, "moved_to_pruned": moved,
                         "stopped_early": action == "stop", **stats})
        log(f"{action:9s} {key}  n={stats.get('n')}  mean={stats.get('mean', float('nan')):.3f}  "
            f"U={stats.get('U', float('nan')):.3f}  bar={bar:.3f}  pruned {moved} entries")
    lines = write_progress(sweep_id, rewards_by_key, queue_counts,
                           already | {k for k, _, _ in decisions}, bar, n_partial)
    if n_unscorable:
        log(f"WARNING: {n_unscorable} completed records had an empty train_history (skipped)")
    log(lines[0] + " | " + " | ".join(line.strip() for line in lines[1:]))
    return len(decisions)


def main():
    """Freeze the bar, bootstrap/load the studies, then run the 10-minute decision loop."""
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_id", required=True)
    p.add_argument("--once", action="store_true", help="run a single cycle and exit (testing)")
    p.add_argument("--poll_seconds", type=int, default=600)
    args = p.parse_args()
    bar = freeze_bar()
    studies, tmap = bootstrap_studies()
    while True:
        n = cycle(args.sweep_id, studies, tmap, bar)
        if args.once:
            log(f"--once: exiting after {n} decisions")
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
