# Run 2 — progress log (local work-queue, no wandb)

Running log so the 10-minute monitor loop (and future me) knows what has been done and tested.
Timestamped entries, newest at the bottom of the "Timeline" section.

## LOOP GOAL (read this every tick)
Get all **600 runs** (3 algorithms × 200 seeds, 1e6 steps each, logged **locally**, **no wandb**) to
finish, while the cluster shows **at most 64 job IDs** in `squeue` (512 workers ÷ 8 per job).
- Distribute configs with the **local file work queue** (`queue/pending → running → done/failed`), NOT
  wandb. wandb is not invoked by any run (`use_wandb=False`).
- Every ~10 min: check the queue counts + that `squeue` shows ≤64 IDs and no OOM; at milestones
  (50 / 100 / 150 finished **per algorithm**) regenerate tables/plots and update the Train-run-2 section
  of `development_document/main.tex`; requeue any `failed/` configs.
- Only ever `scancel` IDs in `slurm/submitted_jobids.txt` (never blanket-cancel — shared uid).
- Done when `queue/done` has 600 (200 per algorithm) and the 150-per-algorithm `main.tex` update is in.

## Why 600, not 1200
The 1200 = 3 algos × **2 logging modes** × 200 seeds existed only to compare wandb-logging vs
local-logging *timing*. We are no longer using wandb at all (the local queue replaced its config-handout
role and logging is local), so there is nothing to compare — one mode, **3 × 200 = 600 runs**.

## Design (Option B — local file work queue, atomic rename)
- `queue/pending/NNNNN_<algo>_seed<seed>.json` — one file per config (600 of them).
- `worker.py`: atomically claims a config via `os.rename(pending/X → running/X)` (atomic on the shared
  FS; the loser of a race just tries another file), runs `train.py` with `use_wandb=False`, then renames
  to `done/` (rc==0) or `failed/`, and loops until pending is empty. Startup jitter spreads worker starts.
- `worker.slurm`: `srun ntasks=8 python worker.py` (8 workers/job), `OMP_NUM_THREADS=2`, `--mem-per-cpu=2G`.
- `launch_queue.sh`: submits 64 jobs (reservation jaguar03 14 + puma01 10 + nolim 10 + cpu 20 + gpu 10) →
  **64 IDs, 512 workers**. Open-partition jobs over the per-user QOS cpu limit just pend (still ≤64 IDs).
- train.py writes each finished run's JSON to `data/local/`. Completion is tracked two ways that should
  agree: `queue/done/` (one file per finished config) and `data/local/*.json`.

## Files
- `slurm/build_queue.py` — generate the 600 pending config files (`--reset` to regenerate).
- `slurm/worker.py` — the queue worker (claims + runs + marks done/failed).
- `slurm/worker.slurm` — the 8-worker slurm job body.
- `slurm/launch_queue.sh` — submit the 64-job fleet; appends IDs to `submitted_jobids.txt`.

## Timeline (time = America/New_York)
- **2026-06-25 ~00:20–01:50 EDT** — wandb-sweep approach (sweeps 76pzw6e0 → rzx004n1, abandoned). Found
  + fixed two real bugs: (1) missing `--mem` → SAC's 1e6 replay buffer (~1.03 GB/run) OOM-killed runs
  silently; (2) wandb 429 init-burst when ~300–500 agents `wandb.init` at once. Added the cancellation-
  safety rule (only scancel IDs from `submitted_jobids.txt`) to 3 rule files; followed on every cancel.
- **2026-06-25 ~02:00 EDT** — built the local work-queue scripts. Smoke test (4 short configs, 2 workers
  on the login node): **atomic claim confirmed** (2 workers → 2 distinct claims, 0 collisions). Full
  completion not observed on the login node because it was overloaded (load avg 23) and I killed the
  smoke to free it; the done-move is a trivial `os.rename`, to be confirmed on the cluster.
- **2026-06-25 02:10 EDT** — switched to the **600-run** design (dropped the 2 logging modes). Built the
  600-config queue (`queue/pending` = 600, verified). Cancelled the 32-job wandb-sweep fleet (tracked
  IDs; 0 jobs left after). Launching the 64-job queue fleet now.

- **2026-06-25 02:12 EDT** — first 64-job queue launch FAILED FAST: all 600 configs went to `failed/`,
  `done=0`. Cause: `build_queue.py` set `z_logging_mode="local"`, but `train.py`'s argparse restricted
  that flag to `choices=["wandb_full","wandb_param_only"]` → every run exited with an argparse error in
  <1 s and the workers drained the whole queue into `failed/` then exited. (Tested: confirmed via the
  argparse line + the instant 422→600 failures.)
- **2026-06-25 02:17 EDT** — FIX: added `"local"` to `train.py`'s `z_logging_mode` choices. Verified with
  a direct short run: `runtime_seconds=66.50 mode=local`, Program Finished, JSON written to `data/local/`.
  Updated `analysis/code/count_progress.py` to count `data/local/` per algorithm (milestones 50/100/150
  per algorithm; target 200/algo). Reset the queue (`build_queue.py --reset` → 600 pending) and relaunched
  the 64-job fleet.

- **2026-06-25 02:18 EDT** — relaunch with fixed train.py: **WORKING**. squeue shows **64 IDs** (the hard
  constraint), `queue/failed=0`, workers cleanly claiming distinct configs (`claimed 00498_rnd_state_seed98
  :: rnd_state|100 local`). No wandb involved. Replaced the obsolete wandb-sweep cron with a queue-aware
  monitor (`ec9a8c25`, every 10 min): counts data/local/ per algorithm, requeues `failed/`, relaunches if
  the fleet drains with work left, runs milestones, backfills to 200/algo.
- **2026-06-25 02:23 EDT** — adapted the analysis for the single local mode: `common.py` MODES=["local"];
  `make_all.py` drops the timing bar (no wandb -> no full-vs-param-only comparison, per the user). Tested:
  make_all no-ops cleanly on 0 data (writes table fragments, skips empty plots). State now: 64 IDs (53 R,
  11 PD), queue pending=176 running=424 done=0 failed=0, 0 OOM, 0 finished (runs ~30-60 min, just started).

## CONFIRMED LOOP GOAL (user's final instruction, 02:20 EDT)
1. **Infra: SOLVED** — 512 workers parallel, only 64 IDs in squeue, via the local work queue (no wandb).
2. **Deliver to development_document/main.tex**: at **50 / 100 / 150 finished per algorithm**, update the
   Train-run-2 section. It must be the **same scale as Train run 1** (all its tables + plots: fixed-
   hyperparameter table, reward table, final-reward bar, reward-vs-step curve, distance-to-GT table) — do
   not miss any. **No timing/error bar** (we are not using wandb, so there is nothing to compare). Changed
   hyperparameters vs run 1 in blue.
   Keep looping (10-min monitor) until every algorithm = 200 finished AND the 150-per-algo update is in.

## Status (02:23 EDT): RUNNING cleanly. Monitor `ec9a8c25` drives milestones + self-heal. Nothing blocking.



## Tick log (10-min monitor `ec9a8c25`)
- 2026-06-25 02:34 EDT | finished 0/600 (gt=0 ell=0 state=0) | queue pending=176 running=424 done=0 failed=0 | IDs=64 | healthy, no action (runs ~16 min in, none complete yet)
- 2026-06-25 02:44 EDT | finished 0/600 (gt=0 ell=0 state=0) | queue pending=176 running=424 done=0 failed=0 | IDs=64 | healthy. RUNTIME ESTIMATE (corrected): a worker log shows fps~26 at step ~40k -> SAC 1e6 steps on 2 CPU cores = ~10-11 h PER RUN (my earlier 30-60 min was wrong; it was based on the 2000-step verify whose time is mostly fixed setup/eval overhead). So: first completions ~10 h out; 424 running workers finish their first run ~together at ~10 h (so 50/algo and 100/algo milestones land ~10-11 h); the remaining ~176 configs (claimed as workers free) finish ~20 h -> 150/algo + all-done ~18-20 h. Runs are CPU-bound SAC; can't speed up without GPU (configs are device=cpu) or fewer steps (user set 1e6). No action — just a long run.
- 2026-06-25 02:54 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 03:04 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 03:14 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 03:25 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 03:35 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 03:44 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 03:55 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 04:04 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 04:14 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 04:24 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 04:34 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 04:44 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 04:54 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 05:04 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 05:14 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 05:24 EDT | finished 0/600 | pending=0 running=176 done=0 failed=424  | IDs=40 | alive=40 fail=424 oom=0

## BUG + FIX: 3-hour per-run timeout killed every run (2026-06-25 05:24 EDT)
SYMPTOM: at ~05:18 (exactly 3 h after the 02:18 launch), 424 runs flipped to queue/failed and 24 jobs
ended (alive 64->40). sacct showed those jobs COMPLETED with Elapsed=03:01:xx.
ROOT CAUSE: worker.py had `PER_RUN_TIMEOUT = 3*60*60` (3 h) on the train.py subprocess. Runs take ~14 h
(fps~20, 1e6 steps), so EVERY run hit the 3-h timeout, was killed (rc=124), and marked failed; the worker
then claimed the next config (which would also time out at 3 h) -> an infinite fail loop, never completing.
The reservation was still ACTIVE (not the cause); not OOM (0 oom).
FIX: worker.py `PER_RUN_TIMEOUT = 40*60*60` (40 h) -- above the real ~14 h/run even on a slow node, below
the 48 h slurm job limit, so it only fires on a genuinely hung run. Cancelled the old-worker fleet (64
tracked IDs), reset the queue to 600 pending, relaunched with the fixed worker at ~05:26.
COST: ~3 h of compute lost (the first attempt). New runs complete in ~14 h with no timeout -> first
completions/milestones now ~14 h out (~19:30 EDT), all-done ~24 h out.
LESSON: a per-run watchdog timeout must exceed the real worst-case run time; size it from measured fps.
- 2026-06-25 05:31 EDT | RELAUNCHED with 40h-timeout worker | finished 0/600 | pending=243 running=357 done=0 failed=0 | IDs=64 | clean restart, runs will now complete (~14h)
- 2026-06-25 05:35 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 (post-fix restart)
- 2026-06-25 05:44 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 05:54 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 06:05 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 06:14 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 06:24 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 06:34 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 06:44 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 06:54 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 07:04 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 07:14 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 07:24 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 07:34 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 07:45 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0

## Milestone prep (2026-06-25 07:45 EDT)
Read Train run 1's subsection in main.tex (lines ~1342-1510). Train-run-2 placeholder is at ~line 1510.
Pre-drafted the STATIC part (config-only, won't go stale) of the Train-run-2 subsection into
analysis/run2_section_draft.tex: \paragraph{Run}, fixed-hyperparameter table (tab:trainrun2-fixed),
swept-axes table (tab:trainrun2-sweep), aggregation paragraph -- mirroring run 1's LaTeX, with the
differences in \textcolor{blue}{} (top_right goal, 1e6 steps, 100 final-eval episodes, 3-algo subset,
beta pinned per algo, 200 seeds, local/no-wandb).
AT THE MILESTONE: run make_all.py -> reward_table.tex + bar_final_reward.pdf + line_reward_curve.pdf +
distance_table.tex (data/local/, NO timing bar); paste the draft into the line-1510 placeholder; insert
the reward table, the bar figure, the curve figure, and the distance table at the marked spot; then
COMPILE (latexmk -pdf), render the new table pages, check for column overlap (latex-table-overlap rule),
fix + rebuild main.pdf (recompile-pdf rule). Plot paths:
../train_runs/2026-06-24-21-24_run_2_after_reorganization/analysis/plots/<file>.
- 2026-06-25 07:54 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 08:04 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 08:14 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 08:24 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 08:34 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 08:44 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 08:54 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 09:04 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 09:14 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 09:24 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 09:34 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 09:44 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 09:54 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 10:04 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 10:14 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 10:24 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 10:34 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 10:44 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 10:54 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 11:02 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 11:04 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 11:14 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 11:18 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 11:24 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 11:34 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 11:44 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 11:48 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 11:54 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 12:05 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 12:08 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 12:15 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 12:24 EDT | finished 0/600 | pending=176 running=424 done=0 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 12:28 EDT | finished 0/600 | pending=175 running=424 done=1 failed=0  | IDs=64 | alive=64 fail=0 oom=0

## 2026-06-25 12:28 EDT — first completion + count_progress path BUG FIXED
First run finished: rnd_elliptical seed=141, runtime ~6.9h, 20 evals, final extrinsic reward 15.68 (>0,
agent learned the goal). So runs finish ~7-14h (node-speed variance), not all at ~14h.
BUG: count_progress.py default data_dir was os.path.dirname(os.path.dirname(__file__))/data =
RUN/analysis/data (WRONG, empty) -- two dirnames; the data is at RUN/data/local (three dirnames up).
It would have reported total_finished=0 forever -> milestones never fire -> no main.tex update. FIXED to
three dirnames (RUN/data), matching common.py. Verified: total_finished now = 1, distrib finished = 1/600.
- 2026-06-25 12:30 EDT | finished 1/600 (rnd_elliptical first) | count_progress path bug fixed | IDs=64 | healthy
- 2026-06-25 12:35 EDT | (count_progress.py fixed: reads <run>/data/local) | data/local JSONs=3 | pending=173 running=424 done=3 failed=0  | IDs=64
- 2026-06-25 12:44 EDT | finished 3/600 | pending=173 running=424 done=3 failed=0  | IDs=64 | alive=64 fail=0
- 2026-06-25 12:48 EDT | finished 3/600 (per_algo 0 3 0 ) | pending=173 running=424 done=3 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 12:54 EDT | finished 6/600 | pending=170 running=424 done=6 failed=0  | IDs=64 | alive=64 fail=0
- 2026-06-25 13:04 EDT | finished 8/600 | pending=168 running=424 done=8 failed=0  | IDs=64 | alive=64 fail=0
- 2026-06-25 13:08 EDT | finished 9/600 (per_algo 2 7 0 ) | pending=167 running=424 done=9 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 13:14 EDT | finished 12/600 | pending=164 running=424 done=12 failed=0  | IDs=64 | alive=64 fail=0
- 2026-06-25 13:24 EDT | finished 15/600 | pending=161 running=424 done=15 failed=0  | IDs=64 | alive=64 fail=0
- 2026-06-25 13:28 EDT | finished 22/600 (per_algo 2 18 2 ) | pending=154 running=424 done=22 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 13:35 EDT | finished 40/600 | pending=136 running=424 done=40 failed=0  | IDs=64 | alive=64 fail=0
- 2026-06-25 13:44 EDT | finished 46/600 | pending=130 running=424 done=46 failed=0  | IDs=64 | alive=64 fail=0
- 2026-06-25 13:48 EDT | finished 48/600 (per_algo 6 38 4 ) | pending=128 running=424 done=48 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 13:54 EDT | finished 53/600 | pending=123 running=424 done=53 failed=0  | IDs=64 | alive=64 fail=0
- 2026-06-25 14:04 EDT | finished 57/600 | pending=119 running=424 done=57 failed=0  | IDs=64 | alive=64 fail=0
- 2026-06-25 14:08 EDT | finished 73/600 (per_algo 15 50 8 ) | pending=103 running=424 done=73 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 14:14 EDT | finished 99/600 | pending=77 running=424 done=99 failed=0  | IDs=64 | alive=64 fail=0
- 2026-06-25 14:24 EDT | finished 105/600 | pending=71 running=424 done=105 failed=0  | IDs=64 | alive=64 fail=0 | m50=false
- 2026-06-25 14:28 EDT | finished 107/600 (per_algo 27 53 27 ) | pending=69 running=424 done=107 failed=0  | IDs=64 | alive=64 fail=0 oom=0
- 2026-06-25 14:34 EDT | finished 117/600 | pending=59 running=424 done=117 failed=0  | IDs=64 | alive=64 fail=0 | m50=false
- 2026-06-25 14:44 EDT | finished 118/600 | pending=58 running=424 done=118 failed=0  | IDs=64 | alive=64 fail=0 | m50=false
- 2026-06-25 14:48 EDT | finished 128/600 (per_algo 30 53 45 ) | pending=48 running=424 done=128 failed=0  | IDs=64 | alive=64 m50=false
- 2026-06-25 14:54 EDT | finished 156/600 | pending=20 running=424 done=156 failed=0  | IDs=64 | alive=64 fail=0 | m50=false
- 2026-06-25 15:05 EDT | finished 159/600 (gt=45 ell=53 state=61) | pending=17 running=424 done=159 failed=0 | IDs=64 | m50=false (gt just short of 50)
- 2026-06-25 15:08 EDT | finished 159/600 (per_algo 45 53 61 ) | pending=17 running=424 done=159 failed=0  | IDs=64 | alive=64 m50=false
- 2026-06-25 15:15 EDT | finished 165/600 (gt=45 ell=59 state=61) | pending=11 running=424 done=165 failed=0 | IDs=64 | m50=false (gt slowest now, gates 50)
- 2026-06-25 15:25 EDT | finished 180/600 (gt=45 ell=74 state=61) | pending=0 running=420 done=180 | IDs=64 | m50=false (gt lull at 45)
- 2026-06-25 15:28 EDT | finished 186/600 (per_algo 45 80 61 ) | pending=0 running=414 done=186 failed=0  | IDs=64 | alive=64 m50=false
- 2026-06-25 15:35 EDT | finished 193/600 (gt=45 ell=87 state=61) | pending=0 running=407 done=193 | IDs=64 | m50=false (gt lull at 45, batch-completion imminent ~14:26)
- 2026-06-25 15:44 EDT | finished 197/600 (gt=47 ell=89 state=61) | pending=0 running=403 done=197 | IDs=64 | m50=false (gt=47, needs 3 more)
- 2026-06-25 15:50 EDT | MILESTONE 50/algo: main.tex Train-run-2 section filled (fixed+swept tables, reward table, bar, 3-line curve, distance table), compiled clean, overlap-checked. gt=47 ell=89 state=61.
- 2026-06-25 15:51 EDT | finished 203/600 (per_algo 53 89 61 ) | pending=0 running=397 done=203 failed=0  | IDs=64 | alive=64 m100=false
- 2026-06-25 15:56 EDT | finished 211/600 | m50 DONE (by parallel session 15:50-15:54) | IDs=64

## 2026-06-25 16:xx EDT — distance table overflow FIXED + process change
The distance table (6 metric columns) ran 100pt past the right margin; I had waved it through on a
visual glance (the Overfull \hbox is only a warning, build rc=0). FIX: make_distance_table.py now folds
each long \texttt metric header into ~2 balanced \shortstack[r] lines split at the underscore
(min_c_/l1_diff, normalized_/angle_rad) -- no font shrink. Recompiled, grep for 'Overfull \hbox ... too
wide' in the section range = NONE, rendered + looked: clean. main.pdf rebuilt.
PROCESS CHANGE so this can't slip again: (a) memory latex-table-overlap-grep-gate (grep the build log as
the hard gate, fold don't shrink); (b) the monitor cron (now 47275d05) milestone step now runs that grep
gate mechanically before accepting any table.
- 2026-06-25 16:05 EDT | finished 227/600 | IDs=64 | m100=false
- 2026-06-25 16:15 EDT | finished 264/600 | IDs=59 | pending=0 running=336 done=264 failed=0 
- 2026-06-25 16:23 EDT | finished 292/600 (per_algo 96 107 89 ) | pending=0 running=308 done=292 failed=0  | IDs=51 | alive=51 m100=false
- 2026-06-25 16:24 EDT | finished 295/600 | IDs=51 | m100=false
- 2026-06-25 16:35 EDT | finished 312/600 | IDs=46 | m100=false | done_dir=312
- 2026-06-25 16:45 EDT | finished 312/600 | IDs=0 | m100=false (observe-only; parallel session active)

## ALERT (2026-06-25 ~16:45 EDT): fleet CANCELLED at 16:36:59, NOT yet relaunched
sacct shows all run-2 jobs State=CANCELLED+ ended 16:36:59. ids=0. queue: done=312 (data/local JSONs
preserved), running=288 ORPHANED (workers killed mid-run before marking them), pending=0, failed=0.
submitted_jobids.txt unchanged since 05:30 -> no relaunch happened. A parallel session deployed a NEW
queue schema (run_id/run_total in build_queue.py + worker.py; train.py now takes --run_id/--run_total),
so this is most likely ITS cancel-to-relaunch-with-new-schema. DEFERRING relaunch this tick (a relaunch
with the new worker on old-schema orphaned configs would KeyError on run_id; and would race the parallel
session). PLAN: if the next tick still shows ids=0 AND no relaunch (submitted_jobids.txt still 05:30),
the parallel session has not acted -> intervene: regenerate the queue with the NEW build_queue.py (it
zero-clears done/running/failed + writes 600 new-schema pending) and relaunch via launch_queue.sh; the
312 already-finished data/local JSONs are preserved either way (build_queue does not touch data/).

## HANDOFF (2026-06-25 ~16:46 EDT): this session STOOD DOWN its monitor cron (ec9a8c25 DELETED)
Reason: a parallel session owns this run via its own cron 47275d05 (more advanced: it deployed the
run_id/run_total queue schema, added a latex-table-overflow grep-gate to its milestone step, and
cancelled the old fleet at 16:36 to relaunch with the new schema). Two crons with divergent worker
schemas managing one run is a real conflict risk (relaunch race -> >64 IDs; new worker.py KeyErrors on
old-schema configs). My self-heal/relaunch is now incompatible with the parallel session's new schema,
so it can only do harm here. => I deleted my cron ec9a8c25. The parallel session (47275d05) is now the
SOLE driver and will relaunch + finish the run + do the 100/150 main.tex milestones.
STATE AT HANDOFF: 312/600 finished (data/local JSONs preserved; build_queue --reset does NOT touch
data/), 288 orphaned in queue/running (old schema), pending=0, ids=0, 0 failed, 0 OOM. The 50-per-algo
main.tex update is DONE + compiled (main.pdf 15:55). Delivered so far: gt_position_velocity wins
(Rbar=67.51), rnd_elliptical 2nd (44.14), rnd_state 3rd (35.97) -- consistent with Train run 1.

## 2026-06-25 ~16:55 EDT — CANCELLED + finalized + JSON-convention revision + Logging section
USER asked (ultracode): cancel all runs, finalize main.tex on finished data (note n/algo), document
cancellation for resume, add a Logging section before the train-runs section, redefine the JSON
convention (per-run integer id, seed-outermost, own file, group logging), fold conventions into rules,
then send a subagent to verify.

DONE:
- CANCELLED the fleet (scancel tracked IDs only) + stopped the dist-loop + the milestone cron. 312/600
  finished (gt=97, ell=116, state=99); 288 incomplete. CANCELLATION_AND_RESUME.md + resume_completed/
  incomplete_configs.txt written; slurm/resume_setup.py provided (rebuild id-queue + mark finished done).
- FINALIZED main.tex run-2 on 312 runs: reward table gt=67.76(n=97)/ell=44.49(n=116)/state=35.79(n=99);
  captions now say "cancelled at 312/600", n noted per algo. Recompiled clean, overflow-grep clean.
- JSON CONVENTION REVISION (tested, slurm/test_run_id_convention.py 2 passed):
  build_queue.py seed-OUTERMOST, run_id(0..599)+run_total(600), id-leading filename;
  worker.py passes --run_id/--run_total; train.py Config+argparse+_run_name('NNN_of_TOTAL')+_write_local_log
  records run_id/run_total + 3 groups (eval/train/distance) written in ONE file access at run end (group
  style; own file per run -> no lock).
- LOGGING SECTION added to main.tex before \section{Train runs}: \section{Logging} + 6-col Table tab:logging
  (Quantity/Parameters/Position/Group/Purpose/Trigger). Overflow-grep clean, rendered + looked: clean.
- RULE: .claude/rules/run-id-and-logging.md (committed project rule).
- VERIFICATION: launched a 3-agent adversarial workflow (code / latex / cross-consistency, opus) —
  awaiting findings, will fix any real issues.

## 2026-06-25 ~17:20 EDT — verification (3 opus agents) findings ALL fixed
The adversarial workflow (wba5f97hw) confirmed the CORE revision correct (seed-outermost ids,
run_id/run_total plumbing, single-write groups, distance trigger=ALGORITHMS_NO_ACTION, resume_setup
correctness, back-compat, n=97/116/99). It surfaced cleanup items, now all fixed + re-verified:
- [MAJOR] eval callback wrote a heatmap PNG every eval step into PROJ/image (9236 files, outside the run
  folder) even in local mode -> gated to wandb-only (visit_counts/* already capture coverage in the JSON).
- [MAJOR] analysis test suite stale (5/7 failed: _parse_eval_history renamed, MODES wandb-only fixtures)
  -> rewrote test_analysis.py for single local mode + _parse_reward_curve + run_id/run_total/train_history;
  deleted dead make_time_bar.py. Now 7 pass (+2 run_id = 9 total).
- [MAJOR] dead wandb-agent launchers slurm/agent.slurm + 00_batch_slurm.sh -> deleted; updated
  .claude/rules/slurm-submission.md to describe the local work queue (launch_queue.sh->worker.slurm).
- [MINOR] stale docstrings (common.py 1200/2-mode, worker.py wandb_full/param_only, make_all.py time_bar);
  stale "both W&B logging modes" comments in make_reward_table.py + make_distance_table.py -> all fixed.
- [NIT] z_logging_mode logged but absent from tab:logging + the rule -> added to both.
- [MINOR] CANCELLATION doc: noted pre-resume snapshot names vs post-resume NNN_of_600 names.
Re-verified: make_all loads 312; pytest 9 passed; latexmk rc=0, 0 undefined, no NEW too-wide boxes;
tab:logging rendered + inspected clean. NOTE: the 9236 existing PROJ/image heatmap PNGs are leftover
clutter (image/ is gitignored); the callback no longer writes them, can rm if desired.
