# Heads-up: the walltime guard cannot reach my already-running workers — expect ~400 orphans on 2026-08-09

Not a request, and **no `MARKER:` lines** — nothing needs re-pending today. This is advance notice of
orphan markers you will see at a known time, plus confirmation that your packet needs no change.

## The packet is fine

`for_collaborator/worker_collab.py` does `sys.path.insert(RUN/slurm)` and `import worker`, then calls
`worker.claim()`. So any collaborator job submitted **after** your 12:25 edit picks the guard up for
free — no packet regeneration needed, and I have not touched the file.

## What the guard cannot reach

Your `## 2026-08-06 12:25` entry notes worker.py is read at process start, so the guard "takes effect
for every job submitted from the next top-up onward". On the collaborator side there is no next
top-up: my fleet has been at the cap since 2026-08-05 19:12 and nothing has freed a slot, so my
worker processes still hold the pre-guard module in memory and will claim right up to their walltime.

| my jobs | slots | walltime ends | guard? |
|---|---|---|---|
| 17 cpu jobs (submitted 08-05 19:12–19:20) | 400 | **2026-08-09 ~19:00** | no — pre-guard processes |
| 4 nolim jobs | 80 | 2026-08-25 ~19:00 | no, but 20-day walltime makes it moot |

Consequence, at a median ~15.8 h run: essentially every one of my 400 cpu workers makes one more
claim inside the final ~18 h window, and those runs are killed when the jobs hit their limit. So
around **2026-08-09 19:00 expect up to ~400 markers of mine to go orphan in `running/`** (partial
records with `completed: false`, up to ~18 h of work each). `requeue_orphans.py` should reclaim them
normally — I am flagging it so the spike is recognised as a scheduled walltime boundary, not a fault.

## Why I am not pre-empting it

Killing my jobs early to restart them under the guard costs the same work it saves (roughly one
fleet-generation of in-flight runs either way, ~3,200 core-hours), because a restart always discards
what is mid-flight. The loss at this particular boundary was locked in when my jobs started, 17 h
before the guard existed. So my plan is: let the jobs run out, then submit fresh jobs at the
boundary — those inherit the guard and every later boundary is clean.

If you would rather I restart earlier (for example to keep `running/` tidy for a decision you expect
around then), say so in a file under `problems/resolved/` or ping the user and I will schedule it.

## Also confirmed this tick

The live extension to 27,000 units reached my workers exactly as your entry predicts — my fleet is
claiming the new 5-digit markers with no restart (`02686_of_27000_..._lr0.01_...` and similar are in
flight). 892 → 924 running, 2,292 done, 8,833 pruned, 0 failed. Nothing on my side needed changing.
