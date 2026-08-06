# Adding workers to this run

**What this run is.** A five-arm ablation of CleanRL's PPO + Random Network Distillation on the Atari
game Montezuma's Revenge. One run = one arm at one seed = 2,000,000,000 environment steps, about six
days on a current GPU. 5 arms x 30 seeds = **150 runs**.

The three departures from CleanRL all move it toward this project's own RND:

| arm | what it changes from arm 1 |
|---|---|
| `arm1_original` | nothing — CleanRL as published, the reference |
| `arm2_no_rnd_grad_clip` | the RND predictor's gradient is not clipped (the policy still is) |
| `arm3_update_proportion_1` | the predictor trains on the whole batch, not a random quarter |
| `arm4_shallower_predictor` | the predictor is one block deeper than the target, not two |
| `arm5_all` | all three at once |

The policy is clipped at a global norm of 0.5 in **every** arm. CleanRL clips the policy and the
predictor together through one call, so arms 2 and 5 take the predictor out of that call and clip the
policy on its own norm instead.

**Your role is capacity only.** You submit worker jobs under your own Slurm caps. Workers claim runs
from a queue the owner already built. You never build, repair, prune, or cancel anything shared. If
something looks wrong, you write a problem report and stop — you do not fix it.

**The queue is dynamic.** Workers claim by atomic rename, picking at random among the first 32
unclaimed runs by id. So you can add workers at any time and they pick up whatever is free; no
coordination beyond the queue itself. Run ids are ordered seed-outermost and arm-inner, which is why
the random-within-a-window rule matters: it keeps the campaign's coverage balanced across arms even
if it never finishes.

---

## 1. Setup check (one minute)

```bash
source <this folder>/packet_env.sh
echo "$RUN_DIR"; ls "$RUN_DIR/queue/pending" | wc -l    # unclaimed runs
PYTHONNOUSERSITE=1 "$PY" -c "import torch, envpool, gym; print(torch.__version__, envpool.__version__, gym.__version__)"
```

Expect `2.6.0+cu124 0.6.6 0.23.1`. If the import fails, stop and write a problem report — do not try
to fix the environment.

`PYTHONNOUSERSITE=1` is not decoration. If you have a torch in `~/.local`, Python's user site
directory takes precedence over the shared environment's packages, and you would be told a version
this run does not use. Every job sets it too, for the same reason — and because a torch 2.10+cu128
build has no Pascal kernels, which would kill every run on `gnolim`.

## 2. Smoke test (about five minutes) — required before you submit anything

**Send it to a GPU node; do not run it on the login node.** The trainer falls back to the CPU when no
GPU is visible, so on the login node it would run five heavy CPU trainings that hit their timeout and
report a failure that is not real.

```bash
sbatch -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=8G -t 00:40:00 \
       --output="$LOGDIR/smoke_${USER}_%j.out" --wrap "bash <this folder>/smoke_test.sh"
```

It checks you can read and write the queue directories and read a queue marker, then runs **one short
training per arm** — all five — into your own folder, each with a hard 20-minute timeout. It never
touches the queue. It prints a checklist and ends with `SMOKE PASSED` or `SMOKE FAILED`. In practice
it takes about five minutes; the 20-minute timeouts are a ceiling, not an estimate.

If it fails, write a problem report (section 6) and stop.

## 3. Submitting workers

```bash
bash <this folder>/launch_workers_collaborator.sh                      # list free GPUs, submit nothing
DRY=1 NODES="cheetah08:4 ai07:3" bash <this folder>/launch_workers_collaborator.sh   # print the sbatch lines
NODES="cheetah08:4 ai07:3" bash <this folder>/launch_workers_collaborator.sh         # submit them
```

**You pass the nodes; the launcher works out everything else** — which submission script serves that
node, how many runs per GPU it packs, and therefore how many cpus and how much memory to request. You
never edit this file, and you never need to know the node-to-script mapping.

With no `NODES` it prints the free GPUs **with their partition**, which matters: a node belongs to one
partition, and pairing a node with the wrong `-p` means the job simply never runs.

The launcher refuses to run if the queue is empty or the run is finished; refuses a node this run
cannot use, rather than letting you submit an incompatible one; refuses a node inside the owner's
Slurm reservation, where your job would pend forever with no error; and never submits more worker
slots than there are unclaimed runs, counting the slots you already hold **by exact job id from your
own id file**. It writes only **your own** id file and tags every job with
`--comment=<sweep id>_<your user>` so a job can be recovered if a submission is interrupted.

**Sizing: 8 cpus per run, one run per GPU**, about 6 GB of host memory and 6 GB of GPU memory per run.
Measured, not guessed — see `resource_facts.md`. Two things not to change:

- **Do not raise the cpus per run.** Throughput was measured flat past 8 to 16 cpus on the fast nodes,
  so extra cores mostly buy nothing and cost you run slots, which are the scarce thing.
- **Do not pack more than one run per GPU on the open partitions.** Packing raises throughput per GPU
  but lowers it per cpu, and in `gpu` and `gnolim` the cpu is what runs out first. The launcher
  already applies this: it reads the packing factor from the submission script, so the only packed
  class is the owner's reserved node, which you cannot submit to anyway.

**Which nodes.** A submission script exists under `slurm/submission_script/` for every node class this
run supports. **The existence of that file is the compatibility verdict.** If a class has no script,
this run cannot use it. One class is deliberately absent: `nekomata01` and `nekomata02` (RTX 5080,
compute capability 12.0) — the torch build has no kernels for them and every run there dies at the
first kernel launch. All the Pascal cards, including the whole of `gnolim`, do work.

## 4. Watching your jobs

```bash
bash <this folder>/refresh_my_ids.sh
```

Refreshes your id file and prints the state of each of your jobs. Run it before any `scancel`.

To keep topping up on your own, without watching it:

```bash
NODES="cheetah08:4 ai07:3" nohup bash <this folder>/monitor_collaborator.sh &
```

Every ten minutes it refreshes your id file, prints your jobs, and refills those nodes only while
unclaimed runs remain. It stops by itself when `SWEEP_COMPLETE` appears. It is passive: it never
repairs, requeues, prunes or cancels anything.

## 5. Cancelling

**Cancel only ids that appear in your own id file.** Never `scancel -u`, never `scancel -t`, never
`scancel -n`. Other people's jobs and other sessions' jobs share this cluster, and a blanket cancel has
destroyed running work here before.

```bash
bash <this folder>/refresh_my_ids.sh          # first
scancel <id from your id file>                # then, one id at a time
```

## 6. When something goes wrong

Write **one new file** in `problems/open/`, named `<YYYYMMDD>_<HHMM>_<your-user>_<short-slug>.md`,
saying what you saw, which job ids, which nodes, and what you had run. Then stop.

The owner checks `problems/open/` on a twenty-minute cycle while a session is active, fixes the
owner-side cause, and moves the file to `problems/resolved/`. That cycle is not a daemon — if the
owner's session has ended, a report waits until the next one. Do not repair shared state yourself,
and do not requeue anything.

## 7. Am I done?

The run is finished when `SWEEP_COMPLETE` appears in the run folder — the owner's tick writes it once
`queue/pending` and `queue/running` are both empty. Until then, `queue/pending` holding zero just
means every run is claimed, not that the work is over.

**The campaign is on course to finish, which was not the original expectation.** Measured on
2026-08-06 with 105 runs in flight across both our uids: 326,000 steps per second in aggregate, so
the remaining work is about **10.5 days** of wall clock (26,800 GPU-hours in total, 179 per run).
That holds only while the capacity does — it assumes roughly 105 runs stay in flight, and it is your
jobs that supply half of them.

**A run that stops early is still expected**, because a job's walltime is shorter than a run: `gpu`
jobs are capped at 4 days and a run needs about 7.5 GPU-days. Every run checkpoints hourly and
resumes, so a job ending is a handover rather than a loss — but it does mean jobs must be resubmitted
as they expire. The ids are ordered so that whatever finishes holds every arm at the same seed count,
which is why you should not reorder or hand-pick runs.

## The short list of things not to do

- Do not run the owner's launcher, the queue builder, or the monitor.
- Do not move, rename, or delete queue markers, and do not hand-pick which runs to claim.
- Do not write to anyone else's id file, or to the owner's `logs/`.
- Do not blanket-cancel.
- Do not edit the shared environment or the training code.
- Do not change the cpus per run or pack runs onto a GPU — both are measured settings, and the
  reasoning is in `resource_facts.md`.
