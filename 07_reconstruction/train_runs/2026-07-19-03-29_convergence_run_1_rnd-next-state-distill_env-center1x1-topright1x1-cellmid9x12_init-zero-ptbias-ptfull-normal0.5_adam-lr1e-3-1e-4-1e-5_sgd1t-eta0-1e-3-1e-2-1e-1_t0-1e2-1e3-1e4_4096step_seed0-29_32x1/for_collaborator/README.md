# Add workers to Convergence run 1 — collaborator handbook

You are adding COMPUTE CAPACITY to an already-running sweep: your jobs run extra workers under YOUR
OWN Slurm caps, on the same shared work queue. That is your whole role — you never build, repair,
prune, or cancel anything on the shared side. Everything corrective is automated on the owner's
side: **sl5nw's monitor runs every 10 minutes and actively fixes problems** (killed jobs' claimed
work is reclaimed automatically; reported problems are picked up from `problems/open/`). You never
need to repair anything — if in doubt, write a problem file (step 7) and stop.

**This sweep is small and may already be finished.** The whole run is about 18 core-hours (4320
runs of ~15 s each) and the owner's own fleet is already draining it, so the queue can empty within
the hour of launch. Every script here self-gates: **if the launcher reports there is nothing to do,
you are done** — that is the expected clean outcome, not an error.

## 1. What this run is

Convergence run 1 measures how fast the RND intrinsic bonus converges, in isolation from the RL
loop: each run forwards a fixed set of maze points through the RND target and predictor, takes one
full-batch distillation step, repeats for 4096 steps, and records the per-point bonus at log-spaced
checkpoints. Configurations are ranked by how close the fitted bonus-decay slope is to -1/2. The
sweep is 3 point sets x 4 initialization schemes x 12 optimizer configurations x 30 seeds = 4320
runs. Sweep id: `2026-07-19-03-32_convergence-run1`.

How the queue works: every run is a JSON marker in `queue/<sweep>/pending/`. A worker atomically
claims one (renames it into `running/`), runs `convergence_train.py`, and moves the marker to
`done/` (success) or `failed/`. Workers from different users interleave safely; when the queue
empties, workers exit on their own. **There is no pruning controller** — every configuration runs
all 30 seeds, so `pending/` only ever shrinks because workers claim markers, never because a
controller removes them.

All commands below assume you are IN this folder first:

```bash
cd "$(dirname <this README's path>)"   # i.e. the for_collaborator/ folder of the run
```

## 2. One-time setup check (nothing to install)

The shared environment is already built. Verify you can run it:

```bash
source packet_env.sh && "$PY" -V        # expect: Python 3.11.13
```

If that errors, stop and report (step 7) — do not build your own environment.

## 3. Smoke test (no Slurm job, does not touch the queue)

```bash
bash smoke_test.sh
```

It first runs non-mutating permission probes on the queue dirs, the data dir, the log dir, and one
pending marker file, then runs 6 short canary invocations of `convergence_train.py` (~15 s each,
covering all 3 point sets, all 4 inits, and both optimizers) into an isolated
`smoke_data/<you>/<timestamp>/` tree. It prints a checklist: every permission probe PASS, every
canary "OK rc=0, completed:true", the queue untouched, and `problems/open/` empty. All pass ->
continue. Any FAIL -> step 7.

## 4. Start your passive monitor, then the first wave

```bash
nohup bash monitor_collaborator.sh >> logs/monitor_collaborator.log 2>&1 &
DRY=1 bash launch_workers_collaborator.sh    # preview the sbatch plan (submits nothing)
bash launch_workers_collaborator.sh          # submit
```

The launcher sizes buckets live (your caps with 16-CPU headroom, node availability, and the
REMAINING unclaimed queue depth) toward the first-wave targets in `packet_env.sh`. Because this
sweep is tiny, the queue-depth guard will often trim the plan to nothing — if the DRY preview or the
submit prints "not submitting" / "nothing to do", the queue is already covered and you are finished.
Your monitor then auto-refills every 10 minutes (only ever adding jobs while there is uncovered
work) and keeps your id file current (a Slurm requeue gives a job a NEW id; the id file is your only
legal cancel source).

Deciding beyond the standard buckets: use the shared rule
`/p/rlprojects/.claude/skills/submit-cpu-sweep/SKILL.md` plus `resource_facts.md` here — you may
write NEW worker slurm scripts for node classes the packet does not cover (keep them in this folder,
source `packet_env.sh` by absolute path, run `worker_collab.py`, follow the shape rules).

## 5. Am I done?

You are done when `optuna/SWEEP_COMPLETE` exists in the run folder (the launcher and your monitor
check it and stop by themselves), OR when the launcher keeps reporting there is no uncovered work.
Check progress any time:

```bash
ls ../queue/2026-07-19-03-32_convergence-run1/pending | wc -l    # work left
ls ../queue/2026-07-19-03-32_convergence-run1/done | wc -l       # finished runs
ls ../queue/2026-07-19-03-32_convergence-run1/running | wc -l    # in flight now
```

## 6. Cancelling your own jobs (rare)

```bash
bash refresh_my_ids.sh          # ALWAYS refresh first (requeued jobs get new ids)
scancel <ids from YOUR id file only>
```

Never cancel by user, state, or name (`scancel -u`, `-t`, `-n` are forbidden — they can kill other
people's jobs). You cannot cancel sl5nw's jobs and sl5nw cannot cancel yours.

## 7. Something looks wrong -> write a problem file and stop

Create ONE new file (never append to an existing one):

```bash
cat > problems/open/$(date +%Y-%m-%d-%H-%M)_${USER}_<short-slug>.md << 'EOF'
What I saw: ...
What I ran: ...
EOF
```

The owner's monitor surfaces it within 10 minutes. (The fail-fast guard in your workers writes these
automatically if your environment is broken — it stops your workers after 3 consecutive fast
failures, before they can churn the shared queue, and names the affected markers for the owner to
requeue.)

## 8. The DON'Ts (hard rules)

1. Never run anything in `../slurm/` — those are OWNER-ONLY scripts (they refuse to run for you, but
   do not try): `launch_queue.sh`, `build_queue.py`, `requeue_orphans.py`, `worker.py`.
2. Never blanket-cancel (`scancel -u/-t/-n`); only ids from YOUR id file, after `refresh_my_ids.sh`.
3. Never use a reservation or `--qos=csresnolim` — you are not on the owner's reservation.
4. Never rename/move files in `queue/`, `data/`, or `optuna/` by hand — the single-writer rule: your
   side writes only YOUR id file, YOUR logs (`for_collaborator/logs/`), and new files in
   `problems/open/`.
5. Never point a job at `/u/sl5nw/...` paths (unreachable for you) — everything you need is in
   `packet_env.sh`.

Conventions reference: `/p/rlprojects/.claude/skills/submit-cpu-sweep/SKILL.md` (the single CPU
submission rule for this cluster) and `/p/rlprojects/.claude/CLAUDE.md`.
