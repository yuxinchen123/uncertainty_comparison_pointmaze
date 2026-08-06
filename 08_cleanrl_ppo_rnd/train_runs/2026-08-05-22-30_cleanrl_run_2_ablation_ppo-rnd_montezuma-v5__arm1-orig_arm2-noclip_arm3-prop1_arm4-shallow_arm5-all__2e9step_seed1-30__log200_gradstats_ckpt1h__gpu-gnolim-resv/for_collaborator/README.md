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
echo "$RUN_DIR"; ls "$RUN_DIR/queue/$SWEEP_ID/pending" | wc -l    # unclaimed runs
"$PY" -c "import torch, envpool, gym; print(torch.__version__, envpool.__version__, gym.__version__)"
```

Expect `2.6.0+cu124 0.6.6 0.23.1`. If the import fails, stop and write a problem report — do not try
to fix the environment.

## 2. Smoke test (about ten minutes) — required before you submit anything

```bash
bash <this folder>/smoke_test.sh
```

It checks you can read and write the queue directories and read a queue marker, then runs **one short
training per arm** — all five — into your own folder, each with a hard 20-minute timeout. It never
touches the queue. It prints a checklist and ends with `SMOKE PASSED` or `SMOKE FAILED`.

If it fails, write a problem report (section 6) and stop.

## 3. Submitting workers

```bash
DRY=1 bash <this folder>/launch_workers_collaborator.sh    # preview, submits nothing
bash <this folder>/launch_workers_collaborator.sh          # after you add your submit calls
```

The launcher refuses to run if the queue is empty or the run is finished, and never submits more
worker slots than there are unclaimed runs. It writes only **your own** id file and tags every job
with `--comment=<sweep id>_<your user>` so a job can be recovered if a submission is interrupted.

**Sizing: 8 cpus per run, one run per GPU**, about 6 GB of host memory and 6 GB of GPU memory per run.
Measured, not guessed — see `resource_facts.md`. Two things not to change:

- **Do not raise the cpus per run.** Throughput was measured flat past 8 to 16 cpus on the fast nodes,
  so extra cores mostly buy nothing and cost you run slots, which are the scarce thing.
- **Do not pack more than one run per GPU on the open partitions.** Packing raises throughput per GPU
  but lowers it per cpu, and in `gpu` and `gnolim` the cpu is what runs out first.

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

The owner's monitoring loop checks `problems/open/` every twenty minutes, fixes the owner-side cause,
and moves the file to `problems/resolved/`. Do not repair shared state yourself, and do not requeue
anything.

## 7. Am I done?

The run is finished when `SWEEP_COMPLETE` appears in the run folder. Until then, `queue/<sweep
id>/pending` holding zero just means every run is claimed — not that the work is over.

**A run that stops early is expected here.** 150 runs at six days each is more than the cluster will
give in one stretch, and that is planned for: the ids are ordered so that whatever finishes holds
every arm at the same seed count. Partial coverage is fine; unbalanced coverage is not, which is why
the claim order matters and why you should not reorder or hand-pick runs.

## The short list of things not to do

- Do not run the owner's launcher, the queue builder, or the monitor.
- Do not move, rename, or delete queue markers, and do not hand-pick which runs to claim.
- Do not write to anyone else's id file, or to the owner's `logs/`.
- Do not blanket-cancel.
- Do not edit the shared environment or the training code.
- Do not change the cpus per run or pack runs onto a GPU — both are measured settings, and the
  reasoning is in `resource_facts.md`.
