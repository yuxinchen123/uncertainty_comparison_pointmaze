# Adding workers to this run

**What this run is.** Thirty seeds of CleanRL's PPO + Random Network Distillation on the Atari game
Montezuma's Revenge, at the authors' own hyperparameters, with a bug in their environment handling
corrected. One run = one seed = 2,000,000,000 environment steps, about five days on a current GPU.

**Your role is capacity only.** You submit worker jobs under your own Slurm caps. Workers claim runs
from a queue the owner already built. You never build, repair, prune, or cancel anything shared. If
something looks wrong, you write a problem report and stop — you do not fix it.

---

## 1. Setup check (one minute)

```bash
source <this folder>/packet_env.sh
echo "$RUN_DIR"; ls "$RUN_DIR/queue/pending" | wc -l    # unclaimed runs
"$PY" -c "import torch, envpool, gym; print(torch.__version__, envpool.__version__, gym.__version__)"
```

Expect `2.6.0+cu124 0.6.6 0.23.1`. If the import fails, stop and write a problem report — do not
try to fix the environment.

## 2. Smoke test (about ten minutes) — required before you submit anything

```bash
bash <this folder>/smoke_test.sh
```

It checks you can read and write the queue directories and read a queue marker, then runs **five**
short trainings into your own folder, each with a hard 20-minute timeout. It never touches the queue.
It prints a checklist and ends with `SMOKE PASSED` or `SMOKE FAILED`.

If it fails, write a problem report (section 6) and stop.

## 3. Submitting workers

```bash
DRY=1 bash <this folder>/launch_workers_collaborator.sh    # preview, submits nothing
bash <this folder>/launch_workers_collaborator.sh          # after you add your submit calls
```

The launcher refuses to run if the queue is empty or the run is finished, and never submits more
worker slots than there are unclaimed runs. It writes only **your own** id file,
`submitted_jobids_ppo-rnd-atari_<your-user>.txt`, and tags every job with
`--comment=ppo-rnd-atari_<your-user>` so a job can be recovered if a submission is interrupted.

**Sizing.** Eight cpus per run, one run per GPU, about 6 GB of host memory and 5 GB of GPU memory per
run. Both numbers are measured, not guessed — see `resource_facts.md`. Do not raise the cpu count:
throughput measured flat past 8 to 16 cpus, so extra cores buy nothing and cost you run slots.

**Which nodes.** A submission script exists under `slurm/submission_script/` for every node class
this run supports. **The existence of that file is the compatibility verdict.** If a class has no
script, this run cannot use it. One class is deliberately absent: `nekomata01` and `nekomata02`
(RTX 5080, compute capability 12.0) — the torch build has no kernels for them and every run there
dies at the first kernel launch.

## 4. Watching your jobs

```bash
bash <this folder>/refresh_my_ids.sh
```

Refreshes your id file and prints the state of each of your jobs. Run it before any `scancel`.

## 5. Cancelling

**Cancel only ids that appear in your own id file.** Never `scancel -u`, never `scancel -t`, never
`scancel -n`. Other people's jobs and other sessions' jobs share this cluster, and a blanket cancel
has destroyed running work here before.

```bash
bash <this folder>/refresh_my_ids.sh          # first
scancel <id from your id file>                # then, one id at a time
```

## 6. When something goes wrong

Write **one new file** in `problems/open/`, named `<YYYYMMDD>_<HHMM>_<your-user>_<short-slug>.md`,
saying what you saw, which job ids, which nodes, and what you had run. Then stop.

The owner's monitoring loop checks `problems/open/` every twenty minutes, fixes the owner-side cause,
and moves the file to `problems/resolved/`. Do not repair shared state yourself, and do not requeue
anything.

## 7. Am I done?

The run is finished when `SWEEP_COMPLETE` appears in the run folder. Until then, `queue/pending`
holding zero just means every run is claimed — not that the work is over.

## The short list of things not to do

- Do not run the owner's launcher, `build_queue.py`, or the monitor.
- Do not move, rename, or delete queue markers.
- Do not write to anyone else's id file, or to the owner's `logs/`.
- Do not blanket-cancel.
- Do not edit the shared environment or the training code.
- Do not raise the cpus per run above 8, or pack more than one run per GPU on the open partitions —
  both are measured settings, and the reasoning is in `resource_facts.md`.
