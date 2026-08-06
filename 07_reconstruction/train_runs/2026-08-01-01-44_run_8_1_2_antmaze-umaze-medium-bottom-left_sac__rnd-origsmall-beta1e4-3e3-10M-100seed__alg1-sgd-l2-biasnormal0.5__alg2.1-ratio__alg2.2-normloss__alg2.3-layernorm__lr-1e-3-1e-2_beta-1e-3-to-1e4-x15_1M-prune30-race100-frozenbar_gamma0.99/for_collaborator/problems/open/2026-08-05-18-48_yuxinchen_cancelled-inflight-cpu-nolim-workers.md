# 2026-08-05 18:48 — yuxinchen cancelled all of their cpu and nolim workers of this (stopped) sweep

**This is a notice, not a request.** There are deliberately **no `MARKER:` lines** in this report:
this sweep is stopped, so nothing here should be re-pended. Read it, then move it to
`problems/resolved/`.

## What happened

At the user's explicit instruction (given after being shown the cost), I cancelled every one of my
still-running cpu and nolim worker jobs of sweep `2026-08-01-02-03_run812`, to hand my cpu (400) and
nolim (80) pools to the follow-up run
`../../2026-08-05-16-05_run_5_8_1_2_addendum_rnd-origsmall-adam-lr1e-3__...__cpu-nolim`, which
`SWEEP_STOPPED.md` names as the reason this sweep was stopped at 15:45 today.

- 19 jobs, **444 worker slots**, cancelled at 18:44 by exact id from my own id file
  `submitted_jobids_2026-08-01-02-03_run812_yuxinchen.txt` (refreshed immediately before). No job
  outside that file was touched; no blanket cancel was used. Verified before cancelling: all 19 were
  in my id file and every running cpu/nolim job of mine was in that set.
- Every one of the 444 workers was genuinely computing at the moment of cancellation (probed with
  `srun --overlap`: cortado07 load 30.10/30 tasks, puma01 load 114.20/114 tasks, slurm3 load
  30.14/30 tasks — about 100% CPU per worker).
- The runs had been claimed at about **12:55 today** (all 19 jobs started 12:53–12:54), so each was
  roughly **5.8 h into a 15–20 h, 1,000,000-step run** — about 2,570 core-hours of in-flight work
  ended.

## Consequences for this sweep's state

- **444 markers stay in `queue/2026-08-01-02-03_run812/running/` and will never advance.** They are
  orphans now. The markers carry no claiming job id, so they cannot be mapped back to these job ids
  from the marker files; the mapping is by time (claimed ~12:55, per-run JSON stopped updating at
  18:44) and by node (table below).
- Their per-run JSONs under `data/2026-08-01-02-03_run812/local/` are partial records with
  `completed: false`, last written at their most recent 50,000-step checkpoint. They are usable as
  truncated curves and must not be counted in any final-reward aggregate.
- If you ever resume this sweep, archive those partials to a `killed_attempts_<ts>/` folder before
  re-pending, per the run's own convention. I did **not** archive them myself — a collaborator never
  touches the queue or the data.

## My jobs that were cancelled

| job id | partition | worker slots | node |
|---|---|---|---|
| 6528898 | cpu | 14 | panther01 |
| 6528899 | cpu | 14 | puma01 |
| 6528900 | cpu | 14 | puma01 |
| 6528901 | cpu | 14 | puma01 |
| 6528902 | cpu | 14 | puma01 |
| 6528903 | cpu | 14 | puma01 |
| 6528908 | cpu | 16 | puma01 |
| 6528909 | cpu | 14 | puma01 |
| 6533719 | cpu | 30 | affogato05 |
| 6533720 | cpu | 30 | affogato02 |
| 6533721 | cpu | 30 | affogato01 |
| 6533722 | cpu | 30 | cortado07 |
| 6533723 | cpu | 30 | cortado08 |
| 6533724 | cpu | 30 | cortado09 |
| 6533725 | cpu | 30 | cortado10 |
| 6533726 | cpu | 30 | cortado01 |
| 6533727 | cpu | 30 | cortado02 |
| 6533728 | nolim | 30 | slurm3 |
| 6533729 | nolim | 30 | slurm2 |

## What was NOT cancelled

My **gpu (352 slots)** and **gnolim (62 slots)** workers of this sweep are untouched and still
running their in-flight runs to completion — the addendum run takes cpu and nolim only, so there was
no reason to end them. Two gpu jobs of mine (6533707, 6533708) are still pending; when they start
they will find `pending/` empty and exit immediately.
