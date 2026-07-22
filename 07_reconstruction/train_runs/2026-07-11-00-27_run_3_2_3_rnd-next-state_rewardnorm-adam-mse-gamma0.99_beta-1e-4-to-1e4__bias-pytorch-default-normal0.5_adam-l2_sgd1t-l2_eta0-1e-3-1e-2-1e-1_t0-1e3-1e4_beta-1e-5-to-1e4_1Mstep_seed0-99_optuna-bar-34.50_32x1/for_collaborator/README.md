# Add workers to train run 3.2.3 — collaborator handbook

You are adding COMPUTE CAPACITY to an already-running sweep: your jobs run extra workers under
YOUR OWN Slurm caps, on the same shared work queue. That is your whole role — you never build,
repair, prune, or cancel anything on the shared side. Everything corrective is automated on the
owner's side: **sl5nw's monitor runs every 10 minutes and actively fixes problems** (killed jobs'
claimed work is reclaimed automatically; reported problems are picked up from `problems/open/`).
You never need to repair anything — if in doubt, write a problem file (step 7) and stop.

## 1. What this run is

Train run 3.2.3 tests two RND remedies on `rnd_next_state` (reward normalization, and nonzero
bias initialization) — 149 configurations × up to 100 seeds, scored on the final training-episode
reward. Sweep id: `2026-07-11-00-57_init-rewardnorm-bar`. Each run takes ~10–14 h on one CPU. A
pruning controller stops configurations whose results rule out the reference bar, so
**`pending/` files vanish over time — that is normal operation, not an error.**

How the queue works: every pending config is a JSON marker in `queue/<sweep>/pending/`. A worker
atomically claims one (rename into `running/`), runs `train.py`, and moves the marker to `done/`
or `failed/`. Workers from different users interleave safely; when the queue empties, workers
exit on their own.

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

## 3. Smoke test (one small job first)

```bash
bash smoke_test.sh
```

It submits ONE 16-worker job and prints five checks (AllocCPUS == 16, workers claiming, running
markers growing, first JSON records, no fast-failure reports). All five pass → continue. Any
check fails → step 7.

## 4. Start your passive monitor, then the first wave

```bash
nohup bash monitor_collaborator.sh >> logs/monitor_collaborator.log 2>&1 &
DRY=1 bash launch_workers_collaborator.sh    # preview the sbatch plan
bash launch_workers_collaborator.sh          # submit
```

The launcher sizes buckets live (your caps, node availability, remaining queue depth) toward the
first-wave targets in `packet_env.sh` (~300 cpu / ~200 gpu / ~80 nolim partition CPUs). Your
monitor then auto-refills every 10 minutes — it only ever adds jobs while the queue is deep, and
it keeps your id file current (a Slurm requeue gives a job a NEW id; the id file is your only
legal cancel source).

Deciding beyond the standard buckets: use the shared rule
`/p/rlprojects/.claude/skills/submit-cpu-sweep/SKILL.md` plus `resource_facts.md` here — you may
write NEW worker slurm scripts for node classes the packet does not cover (keep them in this
folder, source `packet_env.sh`, run `worker_collab.py`, follow the shape rules).

## 5. Am I done?

You are done when `optuna/SWEEP_COMPLETE` exists in the run folder (the launcher and your monitor
check it and stop by themselves). Progress any time:

```bash
ls ../queue/2026-07-11-00-57_init-rewardnorm-bar/pending | wc -l    # work left
ls ../queue/2026-07-11-00-57_init-rewardnorm-bar/done | wc -l       # finished runs
cat ../optuna/progress.txt    # the controller's full snapshot (refreshed every 10 minutes)
```

(Do not run `../slurm/progress.py` — it REWRITES the owner's snapshot file; reading
`progress.txt` gives you the same information without writing anything.)

## 6. Cancelling your own jobs (rare)

```bash
bash refresh_my_ids.sh          # ALWAYS refresh first (requeued jobs get new ids)
scancel <ids from YOUR id file only>
```

Never cancel by user, state, or name (`scancel -u`, `-t`, `-n` are forbidden — they can kill other
people's jobs). You cannot cancel sl5nw's jobs and sl5nw cannot cancel yours.

## 7. Something looks wrong → write a problem file and stop

Create ONE new file (never append to an existing one):

```bash
cat > problems/open/$(date +%Y-%m-%d-%H-%M)_${USER}_<short-slug>.md << 'EOF'
What I saw: ...
What I ran: ...
EOF
```

The owner's monitor surfaces it within 10 minutes. (The fail-fast guard in your workers writes
these automatically if your environment is broken — it stops your workers before they can damage
the shared queue.)

## 8. The DON'Ts (hard rules)

1. Never run anything in `../slurm/` — those are OWNER-ONLY scripts (they refuse to run for you,
   but do not try): `launch_queue.sh`, `controller_submit.sh`, `monitor.sh`, `build_queue.py`.
2. Never blanket-cancel (`scancel -u/-t/-n`); only ids from YOUR id file, after `refresh_my_ids.sh`.
3. Never use a reservation or `--qos=csresnolim` — you are not on the owner's reservation.
4. Never rename/move files in `queue/`, `data/`, or `optuna/` by hand — the single-writer rule:
   your side writes only YOUR id file, YOUR logs (`for_collaborator/logs/`), and new files in
   `problems/open/`.
5. Never point a job at `/u/sl5nw/...` paths (unreachable for you) — everything you need is in
   `packet_env.sh`.

Conventions reference: `/p/rlprojects/.claude/skills/submit-cpu-sweep/SKILL.md` (the single CPU
submission rule for this cluster) and `/p/rlprojects/.claude/CLAUDE.md`.
