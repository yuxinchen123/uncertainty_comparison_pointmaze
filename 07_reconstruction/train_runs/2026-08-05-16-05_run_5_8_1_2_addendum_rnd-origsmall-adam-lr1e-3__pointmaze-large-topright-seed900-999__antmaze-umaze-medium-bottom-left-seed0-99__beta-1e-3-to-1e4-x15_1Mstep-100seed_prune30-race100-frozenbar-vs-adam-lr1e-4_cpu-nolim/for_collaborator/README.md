# Collaborator packet — original RND at Adam learning rate 1e-3

You (any rlprojects member; the known collaborator is `yuxinchen`) can add worker jobs to this
already-launched sweep, under YOUR OWN Slurm caps, to finish it faster. You never build the queue,
never truncate a configuration, never requeue anything, and never touch another user's files. You
add workers and, if your environment breaks, you write a problem report.

**What this run tests, in one line:** the project's faithful translation of the original RND has
always used the paper's predictor learning rate of 1e-4; this run reruns exactly that stack at 1e-3
AND at 1e-2 on the three environments the writeup measures (PointMaze Large top-right, AntMaze UMaze
and AntMaze Medium starting bottom-left), sweeping 15 bonus weights per environment and racing each
configuration to 300 seeds against a bar frozen from the matching 1e-4 result.

> **The sweep grew on 2026-08-06 and NOTHING is needed from you.** It started as one learning rate
> at 100 seeds (4,500 runs) and is now two learning rates at 300 seeds (27,000 runs). The extra work
> was added to the same live queue, so your workers pick it up on their own next claim — no
> resubmission, no new scripts, no change to anything you run. Your jobs, your id file and your
> commands are all unchanged. The run folder's name still says `lr1e-3` and `100seed` because
> renaming it would break every running worker.

Full context: `resource_facts.md` here, and `../experiment_background.md`.

## How the queue works

A **work queue** lives in the run folder: `queue/<sweep id>/{pending,running,done,failed,pruned}/`,
one small JSON marker per run (27,000 after the 2026-08-06 extension). Each of your workers
atomically claims a marker
(renames it from `pending/` into `running/`), runs `train.py` to completion, then moves it to `done/`
(exit code 0) or `failed/` (anything else), and claims the next one. Every run is the same shape:
1,000,000 env steps on one cpu core, about **17–20 hours**.

There is ONE pending pool — no pool priority and no walltime guard to think about.

The owner's controller decides, every 20 minutes, whether a configuration has fallen far enough
behind its environment's frozen bar to stop; when it does, that configuration's remaining markers
move to `pruned/` and your workers simply never see them again. Runs already in flight always finish.

## Setup check, then the smoke test

```bash
FC=/p/rlprojects/RND/07_reconstruction/train_runs/2026-08-05-16-05_run_5_8_1_2_addendum_rnd-origsmall-adam-lr1e-3__pointmaze-large-topright-seed900-999__antmaze-umaze-medium-bottom-left-seed0-99__beta-1e-3-to-1e4-x15_1Mstep-100seed_prune30-race100-frozenbar-vs-adam-lr1e-4_cpu-nolim/for_collaborator
bash $FC/smoke_test.sh
```

The smoke test probes every permission a worker of yours needs (including reading a marker FILE, not
just its directory), then runs five short trainings through the real entry point — covering all
three environments — each with a hard 20-minute timeout, writing only into
`smoke_data/$USER/<timestamp>/`. It never touches the queue. It prints `SMOKE PASS` or `SMOKE FAIL`.

**Do not submit worker jobs until the smoke passes.**

## Submitting workers

```bash
DRY=1 bash $FC/launch_workers_collaborator.sh    # preview the plan, submit nothing
bash $FC/launch_workers_collaborator.sh          # submit
```

The launcher submits worker jobs to the open **cpu** and **nolim** partitions only — no gpu, no
gnolim, no reservation (the owner's instruction for this sweep). It sizes every job from the live
cluster state, for YOUR uid:

- `--ntasks` never exceeds a node's physical core count (sbatch rejects a larger ask outright) or its
  free allocatable threads (`CPUEfctv − CPUAlloc`);
- memory per worker is at most 2 GB and at most 92% of `RealMemory / CPUEfctv`, so a high-core-count
  node cannot be oversubscribed on memory;
- your own per-user pool caps (cpu 400, nolim 80) minus 16 threads of headroom for your own
  interactive jobs;
- `--ntasks-per-core=2` only on nodes that have two threads per core;
- reserved nodes excluded, `--time` set to the partition's own maximum;
- it stops planning once worker slots would outnumber the runs left to claim, and refuses entirely
  once the owner's `SWEEP_COMPLETE` sentinel exists;
- it submits in batches of ten with a 30-second pause, and appends every returned job id to YOUR id
  file the instant sbatch returns it.

After the first job starts, check the shape landed as asked:

```bash
sacct -j <first id> --format=JobID,JobName,AllocCPUS,ReqMem,NodeList,Timelimit
```

`AllocCPUS` must equal the job's `--ntasks`. If it is double, stop and tell the owner — the
hyperthreading flag was lost and half the allocation is being wasted.

## Monitoring

```bash
nohup bash $FC/monitor_collaborator.sh > $FC/logs/monitor.log 2>&1 &
```

A passive 10-minute loop: refresh your id file, print a queue snapshot, top your workers back up
through the self-gating launcher, exit when the sweep completes. It never renames a queue marker and
never repairs anything.

## Am I done?

When `../SWEEP_COMPLETE` exists. The owner's monitor writes it when `pending/` and `running/` are
empty and all 90 configurations have a verdict (45 per predictor learning rate).

## Cancelling

Only ever from your own id file:

```bash
bash $FC/refresh_my_ids.sh          # ALWAYS refresh first
scancel <id> <id> ...               # ids from submitted_jobids_<sweep>_$USER.txt only
```

**Never** `scancel -u <user>`, `scancel -t PD`, `scancel -t R`, or `scancel -n <name>`. The uid space
is shared with other people, other sessions, and jobs started by hand; a blanket cancel has destroyed
other people's running work before. A job *name* can be reused by another sweep, so cancelling by
name is unsafe too — that is why this packet's job names are random 8-letter prefixes.

## If something breaks: report, do not repair

Write ONE new file under `problems/open/`, named `<timestamp>_<your user>_<short slug>.md`. If the
problem concerns specific runs, add one machine-readable line per marker:

```
MARKER: 00123_of_27000_AntMaze_UMaze-v5_origsmall_lr0.001_b30_seed23.json
```

The owner's monitor picks the report up within 20 minutes, re-pends those markers, and moves the
report to `problems/resolved/` with a footer saying what it did. Your worker wrapper writes such a
report by itself if three claimed runs in a row fail in under ten minutes each — that pattern means
your environment is broken, not the configurations, so the worker stops rather than burning through
the queue.

## The coordination contract

- **One writer per file.** Your id file is `submitted_jobids_<sweep>_$USER.txt` here; your logs go in
  `logs/` here; the owner's live in the run folder's `slurm/` and `logs/`. Each problem report is its
  own new file, never an append to a shared log.
- **Two monitors, disjoint write sets.** The owner's loop is active (controller, orphan reclaim,
  problem handling, sentinel) every 20 minutes; yours is passive (own ids, gated top-up, reporting).
  They cannot fight.
- **When in doubt, write a problem file and stop.**

## The don'ts

1. Do not run `../slurm/build_queue.py`, `../slurm/truncation_controller.py`, `../slurm/monitor.sh`
   or `../slurm/launch_queue.sh` — they refuse to run for you anyway.
2. Do not rename any queue marker by hand.
3. Do not use a reservation or a qos you do not hold.
4. Do not write the owner's id file or log into the owner's `logs/`.
5. Do not blanket-cancel anything.
6. Do not point a job at any `/u/<user>` private path — everything here is under `/p/rlprojects`.
