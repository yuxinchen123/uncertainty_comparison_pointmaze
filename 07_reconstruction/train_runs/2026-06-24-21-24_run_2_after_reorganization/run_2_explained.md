# Run 2 — how it works, and what went wrong, explained from scratch

This document explains, in plain language, how Run 2 is set up and the problems we hit while
launching it. It assumes you have **not** used Slurm or Weights & Biases (wandb) before, and that
"memory" and "the cluster" are fuzzy ideas. Read it top to bottom; each part builds on the last.

There is a short **glossary at the very end** if you want a quick definition of a single term.

---

## 1. What Run 2 is trying to do (the goal, in one paragraph)

We train a reinforcement-learning agent (SAC) to navigate a maze (PointMaze), over and over, under
different settings, and compare the results. The full set of settings is:

- **3 algorithms** (three different ways of computing the "curiosity" bonus): `gt_position_velocity`,
  `rnd_elliptical`, `rnd_state`. Each is pinned to its own best strength (its "beta").
- **2 logging modes** (explained in Part 3): `wandb_full` and `wandb_param_only`.
- **200 random seeds** (a "seed" is the starting random number; 200 seeds = 200 independent repeats
  so the averages are trustworthy).

Multiply them: 3 × 2 × 200 = **1200 separate training runs**. Each run trains for 1,000,000 steps and
takes roughly 30–60 minutes. One computer doing them one at a time would take weeks, so we run many
at once on a shared cluster. That is what Slurm is for.

---

## 2. Slurm — running 1200 trainings on a shared cluster

### 2.1 What a "cluster" and "Slurm" are

A **cluster** is a room full of computers (called **nodes**) that many people share. You do not log
into a node and run things directly — that would let people stomp on each other. Instead you submit a
**job** to a traffic controller called **Slurm**, and Slurm decides which node runs your job and when.
Think of Slurm as a restaurant host: you don't pick your own table, you tell the host what you need
and they seat you when a table is free.

### 2.2 The pieces of a job

When you submit a job you tell Slurm how big it is. The numbers that matter here:

| Term | Plain meaning | Our value |
|---|---|---|
| **node** | one physical computer in the cluster | varies (see partitions) |
| **partition** | a named pool of nodes you're allowed to ask for | `cpu`, `nolim`, `gpu` |
| **CPU** | one processor core; the unit of computing power Slurm hands out | — |
| **task** (`ntasks`) | one running program inside the job | 8 per job |
| **cpus-per-task** | how many cores each task gets | 2 |
| **memory** (`--mem`) | how much working memory (RAM) the job may use | see Part 4 |

We make each **job** run **8 tasks**, and each task is **1 "agent"** (a worker that trains one run at
a time — explained in Part 3). So **1 job = 8 agents**, and each agent gets 2 cores. One job therefore
uses 8 × 2 = **16 cores**.

We submit **many jobs** so that many agents run at once. Right now: **32 jobs × 8 agents = 256
agents** training in parallel. (We tried 64 jobs / 512 agents and it caused a problem — Part 5.)

### 2.3 Partitions and the "reservation"

Different nodes live in different **partitions** (pools). We spread our jobs across three open pools
(`cpu`, `nolim`, `gpu`) so we're not hogging any single one. There is also a special case:

- A **reservation** is a block of nodes the cluster admins set aside **just for you** for a while. We
  currently hold a reservation on two nodes, **jaguar03** and **puma01**. Jobs that want those nodes
  must say `--reservation=<name>`; ordinary jobs are not allowed onto reserved nodes. We fill our own
  reserved nodes first (they're ours and reliable), then add a few jobs on the open partitions.

The exact split right now:

| Where | How many jobs | Agents | Note |
|---|---|---|---|
| Reserved node jaguar03 | 14 | 112 | our reserved node, 224 cores, filled |
| Reserved node puma01 | 10 | 80 | our reserved node, 160 cores, filled |
| Open `nolim` partition | 2 | 16 | shared pool |
| Open `cpu` partition | 4 | 32 | shared pool |
| Open `gpu` partition | 2 | 16 | (we ask for 0 GPUs; the work is CPU-only) |
| **Total** | **32** | **256** | |

### 2.4 Why per-user limits matter

The cluster caps how much one person can grab in each open partition (about 400 cores per partition).
Reserved nodes don't count against that cap (they're already yours). This is why we fill the
reservation fully but keep the open-partition jobs modest.

---

## 3. wandb — what it is and how a "sweep" runs the 1200 jobs

### 3.1 What wandb is

**Weights & Biases (wandb)** is a website that records experiments. While a training run is going, it
sends its numbers (reward, distance-to-target, etc.) up to wandb, and you watch live charts in a
browser. It is the live dashboard for the project.

### 3.2 What a "sweep" is, and what an "agent" is

We do not want to launch 1200 runs by hand. wandb has a feature called a **sweep**: you give it the
list of all 1200 settings once, and it acts as a dispatcher. Then you start **agents** — small worker
programs — and each agent does this on a loop:

1. Ask the sweep dispatcher: "what should I run next?"
2. The dispatcher hands back one of the 1200 settings.
3. The agent runs the training (`train.py`) with that setting until it finishes.
4. Go back to step 1 for the next setting.

So **256 agents** chew through the **1200 settings** a few at a time until all are done. The sweep
makes sure each setting is handed out. (One important catch, Part 5: if a run **crashes**, the sweep
treats that setting as "attempted" and does **not** automatically retry it — so crashes lose settings.)

### 3.3 The three logging modes (the actual research question of Run 2)

Every run has to **start** by contacting wandb to ask the dispatcher for its setting (this contact is
called `wandb.init`). What differs is what it does **after** that:

| Mode | Asks wandb for its setting? | Sends its numbers to wandb during training? | Why |
|---|---|---|---|
| `wandb_full` | yes | **yes** (live charts) | the normal way |
| `wandb_param_only` | yes | **no** — saves numbers to a local file only | to test if sending to wandb slows training |
| (plain no-wandb) | no | no | not used in this sweep |

The whole point of Run 2 is to compare `wandb_full` vs `wandb_param_only` **timing**: does constantly
uploading numbers to wandb make a run slower than just writing them to a local file? We record each
run's wall-clock time and will show the two side by side in a bar chart.

---

## 4. Problem #1 — "Out Of Memory" (the silent killer)

### 4.1 What memory is

**Memory (RAM)** is a computer's short-term scratch space — separate from disk. A program can only use
as much as it's allowed. On the cluster, **Slurm gives each job a memory limit**, and if the job tries
to use more, the operating system **kills it instantly** — no warning, no error message.

### 4.2 Why our runs needed a lot of memory

The SAC training algorithm keeps a **replay buffer**: a big table of the agent's past experiences that
it learns from. Ours holds 1,000,000 experiences, and — this is the key part — it **reserves the whole
table at the very start**, before training even begins. We measured it: about **1 gigabyte (1 GB) per
run**.

### 4.3 What went wrong

Our first jobs **never told Slurm how much memory they needed**, so Slurm handed them its tiny default.
The moment each run reserved its 1 GB buffer, it blew past the limit and the operating system killed
it — in about **0.4 seconds**, with **no error message**. From the outside it looked baffling: runs
appeared to "start" and then vanish, and wandb still showed them as "running" (because a killed program
never gets to say "I crashed"). This is why it was hard to diagnose — a silent death looks like many
different bugs.

### 4.4 The fix

We added one line to the job: **`--mem-per-cpu=2G`**. With 2 cores per agent that's 4 GB per run —
plenty for the 1 GB buffer. We confirmed the fix by running one training by hand on jaguar03 with
memory requested: it trained normally and finished, and we measured its peak memory at 1.03 GB. Fixed.

---

## 5. Problem #2 — the wandb "429" init burst (it crashed the runs another way)

### 5.1 What a "rate limit" / "429" is

Websites protect themselves from being flooded. If too many requests arrive too fast, the server
replies with error code **429**, which means "**too many requests, slow down**." wandb does this
**per project**: across *all* our runs combined, only so many wandb requests per second are allowed.

### 5.2 What went wrong

After fixing the memory bug we scaled up to **512 agents at once**. They **all called `wandb.init` at
the same instant** (every one needs its setting from the dispatcher the moment it starts). That flood
tripped wandb's 429 limit, and about **27% of the runs failed at `wandb.init`** — again dying in ~0.4
seconds. Worse, an agent that fails to start its run a few times in a row **gives up and exits**, so
the burst didn't just crash ~288 runs, it **killed ~300 of the agents themselves**, leaving the fleet
limping along at a fraction of its size while still holding all the cluster slots.

Important nuance: this was a **one-time burst at the simultaneous start**. Once the agents are spread
out and running, they finish at different times and start their *next* run at different times, so the
requests are naturally spread out and 429 stops being fatal. The problem is purely the **synchronized
stampede at launch**.

### 5.3 The two fixes

1. **Stagger the start (jitter).** Each agent now waits a small random amount of time (a few seconds
   to about a minute and a half) before it calls `wandb.init`. So instead of 512 requests in one
   instant, they're smeared over a couple of minutes — gentle enough that wandb doesn't trip. This
   lives in `slurm/agent.slurm`.
2. **Run a moderate fleet.** We launched **256 agents** (not 512). Combined with the jitter, the start
   is calm. We can add more later, and because new agents are also jittered, adding them is safe.

### 5.4 Backfill (cleaning up the lost settings)

Because a wandb sweep does **not** re-run a setting whose run crashed, the burst **permanently lost**
those settings from the sweep. The monitor (Part 7) handles this: when the sweep finishes but some
`(algorithm, mode, seed)` combinations are short of 200 repeats, it submits a small **backfill** sweep
listing exactly the missing combinations, so every combination reaches a full 200.

---

## 6. Problem #3 — never cancel jobs you didn't start (a safety rule)

This one is about not breaking *other people's* work.

On this cluster, you and other running sessions (other automated assistants, and you-by-hand) **all
share one username** (`sl5nw`). Slurm has a command, `scancel -u sl5nw`, that cancels **every** job
under that username at once. Early on, that command was used to clean up — and it **killed another
session's evaluation jobs** that had nothing to do with this project.

The rule now, written into three separate rule files so it can't be forgotten:

- **Never** use a blanket cancel (`scancel -u`, or cancel-by-state, or cancel-by-name — names can be
  reused by other sessions).
- **Every job we submit, we record its ID** into `slurm/submitted_jobids.txt`.
- **To cancel, we only ever cancel the IDs in that file** — guaranteed to be ours and nobody else's.
- We never ask you for job IDs (you leave after launching); the file is the source of truth.

This held on every single cancel during the debugging — your other session's jobs were never touched.

---

## 7. The monitor — a loop that watches the run while you're away

You don't want to babysit this. A **monitor loop** runs automatically every 10 minutes (it's a small
scheduled task; the current one is named `0267a443`). Each time it wakes up it:

1. Counts how many of the 1200 runs have finished.
2. Checks the jobs are healthy — specifically watching for the two failure signatures above (Out Of
   Memory, or a 429 crash burst). If it sees them it reacts (more memory, or lower the agent count).
3. At progress milestones, it regenerates the tables and plots and **writes the "Train run 2" section
   of the paper** (`development_document/main.tex`) for you:
   - at **50 finished per algorithm**: reward tables + plots,
   - at **100 finished per algorithm per mode**: adds the full-vs-`param_only` timing bar chart,
   - at **150 finished per algorithm per mode**: refreshes everything.
4. At the very end, backfills any settings lost to the burst (Part 5.4), so all 1200 truly complete.

---

## 8. The debugging story, in order (why it took several tries)

Both bugs above produce the *same* symptom — a run that dies in under a second with no error — which is
exactly why it was confusing. The honest sequence:

1. Launched 512 agents → all crashed at startup. First guess (wandb's start-up helper failing under
   CPU overload) led to a thread-count fix, which helped that specific error but not the real one.
2. Runs still died. Wrongly concluded "wandb can't handle 500 agents." Reduced to a small fleet to test.
3. The small fleet *still* died on the reserved nodes — which disproved the wandb theory, because a
   handful of agents can't overload wandb.
4. Ran one training **by hand on jaguar03** and finally saw the real message: **Out Of Memory.** Fixed
   it with `--mem`. Confirmed a run now trains and finishes.
5. Scaled back up to 512 → a **new** problem appeared: the 429 init burst (Part 5). Diagnosed it as a
   one-time synchronized stampede, added the start jitter, and dropped to a calmer 256 agents.

Lesson for next time (saved to the assistant's memory): a run dying in under a second with no traceback
is almost always **a kill from outside the program** — Out Of Memory or an init-time rate limit — not a
bug in the code itself. Check those two first.

---

## 9. Current state and the files involved

**Right now:** a clean fleet of **256 agents** is training on sweep **`rzx004n1`** (sweep #5; the four
earlier sweeps were abandoned as we fixed the bugs). Both real bugs are fixed (memory requested; init
jittered). First runs complete in 30–60 minutes; the monitor takes over from there.

The files that make it run (all under this run folder):

| File | What it does |
|---|---|
| `config/sweep_run2.yaml` | the full list of 1200 settings (the sweep definition) |
| `config/sweep_id.txt` | the ID of the currently active sweep (`rzx004n1`) |
| `slurm/00_batch_slurm.sh` | the launcher: submits the jobs, requests memory, tracks job IDs |
| `slurm/agent.slurm` | what each job runs: the start jitter + the 8 agents |
| `slurm/submitted_jobids.txt` | the list of *our* job IDs (the only things we're allowed to cancel) |
| `analysis/code/` | scripts that turn finished runs into tables and plots |
| `analysis/code/count_progress.py` | counts finished runs and flags milestones |
| `.milestones_done` | records which paper updates have already been written |
| `experiment_background.md` | the short standard description of this run |

The durable record is this folder; the live charts are on wandb (project
`catresearch/rnd_07_reconstruction`).

---

## 10. Glossary (one-line definitions)

- **Cluster** — a shared room of computers (nodes) you submit work to.
- **Node** — one physical computer in the cluster.
- **Slurm** — the scheduler that decides which node runs your job and when.
- **Job** — one unit of work you submit to Slurm.
- **Partition** — a named pool of nodes you may request (`cpu`, `nolim`, `gpu`).
- **Reservation** — nodes set aside just for you (ours: jaguar03, puma01).
- **CPU / core** — one processor; the unit of computing Slurm hands out.
- **Task** — one running program inside a job. We run 8 tasks (= 8 agents) per job.
- **Memory (RAM)** — short-term working space; exceeding the job's limit causes an instant kill (OOM).
- **OOM (Out Of Memory)** — the operating system killing a program for using too much RAM.
- **wandb (Weights & Biases)** — the website that records and charts experiments.
- **Sweep** — wandb's dispatcher that hands out the list of settings to agents.
- **Agent** — a worker that repeatedly asks the sweep for a setting and runs it.
- **`wandb.init`** — the call a run makes at startup to contact wandb for its setting.
- **429 / rate limit** — the server saying "too many requests at once, slow down."
- **Jitter** — a small random delay added so many agents don't act at the exact same instant.
- **Replay buffer** — SAC's big table of past experiences (reserved up front, ~1 GB per run).
- **Seed** — the starting random number; different seeds = independent repeats.
- **Backfill** — re-running settings that were lost to crashes, so every combination reaches 200.
- **Milestone** — a progress point (50/100/150 finished) that triggers a paper update.
