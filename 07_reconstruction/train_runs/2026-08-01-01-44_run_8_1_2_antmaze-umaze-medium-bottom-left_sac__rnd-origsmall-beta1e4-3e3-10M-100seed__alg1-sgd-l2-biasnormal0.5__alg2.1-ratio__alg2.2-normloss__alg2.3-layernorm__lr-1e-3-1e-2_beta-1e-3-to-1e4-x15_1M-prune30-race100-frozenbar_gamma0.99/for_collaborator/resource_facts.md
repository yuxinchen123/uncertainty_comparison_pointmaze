# Run 8.1.2 (AntMaze UMaze + Medium) — resource facts for collaborators

- **SWEEP_ID**: `2026-08-01-02-03_run812` (also in `packet_env.sh` and `../slurm/SWEEP_ID.txt`).
- **Run total**: 24,200 runs = 242 configurations x 100 seeds, split over TWO pending pools:

| pool | task | runs | steps per run | who claims it |
|---|---|---|---|---|
| `pending_1m` | S — the four new RND arms | 24,000 (240 configs x 100 seeds) | 1,000,000 | your cpu / nolim / gpu-partition shapes, and your gnolim shapes as fallback |
| `pending_10m` | R — the RND baseline | 200 (2 configs x 100 seeds) | 10,000,000 | your gnolim shapes first (only with enough walltime), and the owner's GPU jobs |

- **The 10M walltime guard (nothing to configure).** The run has NO checkpoints, so a run that
  cannot finish inside its Slurm job must never start. The run's own `worker.py` therefore allows a
  `pending_10m` claim only when the job has at least 170 h of walltime REMAINING on a cpu worker
  (a 10M cpu run measures about 154 h). A gnolim job submitted with a short walltime — for example
  before the 2026-08-05 maintenance window — automatically skips the 10M pool and serves task S
  instead; a 20-day gnolim job submitted after the maintenance takes 10M runs first.
- **Racing / why `pending_1m` shrinks faster than runs finish**: the OWNER's stage-1 controller
  prunes a task-S configuration at the first tick where its 99% upper confidence bound
  (mean + 2.576 s / sqrt(n) over completed seeds, n >= 30) falls below the frozen bar — the run-8.1
  RND winner's mean at 1M for that environment. Pruned configurations' pending markers move to
  `pruned/`. That is the owner's job. A worker of yours that happens to claim a marker pruned
  in-flight is HARMLESS (it just runs one extra seed); you never react to it. Task-R markers are
  never pruned.
- **Per-run cost**: a 1M-step task-S run takes about 15.4 h on one cpu worker. A 10M-step task-R run
  takes about 154 h on cpu (about 64-70 h on a GPU worker — owner-only). Memory: about 1.0-1.2 GiB
  resident per worker at 1M steps; every shape asks `--mem-per-cpu=2G`, which covers it with margin.
- **Partitions you may use** (open partitions of YOUR own pools only):
  - `cpu` (per-user cap 400 CPUs) — task S.
  - `nolim` (per-user cap 80 CPUs) — task S.
  - `gpu` partition, **CPU-ONLY** jobs (`--gpus-per-node=0`), on the LEAST capable GPU nodes first
    (the launcher reads the shared node catalog and fills low-tier GPU nodes so their otherwise-idle
    CPUs are used while better GPUs stay free for real GPU jobs) — per-user gpu-pool cap 400 CPUs.
    The launcher also EXCLUDES the nodes the owner's task-R GPU jobs run on (cheetah02/04/08/09,
    jaguar01/02, adriatic01-06), so your CPU jobs can never block a 10M GPU run.
  - `gnolim` (per-user cap 80 CPUs), **CPU-ONLY** jobs (`--gpus-per-node=0`) — gnolim's Pascal and
    Maxwell GPUs cannot run this run's torch 2.10+cu128 build, so for this run gnolim is a CPU pool.
    These are the only shapes of yours that may take task-R 10M runs (walltime guard above).
  - **NOT**: the owner's reservation (puma01/jaguar03), and no GPU-USING job. `device=cuda`
    submission for task R is OWNER-ONLY in this run.
- **Job shapes**: `30x1` (about a 32-core node), `28x1` (about a 30-core node or a big fragment),
  `14x1` (16-core class — the adriatic/affogato class rejects `16x1` outright), `12x1` (12-core
  class such as jinx / titanx03, and small fragments). gnolim has its own two wrappers
  (`worker_gnolim_14x1_collab.slurm`, `worker_gnolim_12x1_collab.slurm`) because gnolim jobs claim
  both pools and must pass `--gpus-per-node=0`. All shapes use
  `--cpus-per-task=1 --ntasks-per-core=2 --mem-per-cpu=2G` and `srun --wait=0`.
  **Verify `AllocCPUS == ntasks`** (30 / 28 / 14 / 12) on your first submit of each shape; a doubled
  `AllocCPUS` means `--ntasks-per-core=2` was lost — write a problem report and stop that shape.
- **`--ntasks` never exceeds a node's PHYSICAL CORE count** (sbatch validates against cores, not
  hardware threads): 16-core nodes take at most `14x1` here, 12-core nodes at most `12x1`.
- **Fragments**: if the launcher's whole `30x1` cpu jobs sit PENDING while the partition holds idle
  fragments (`sinfo -p cpu -N` shows nodes with 12-30 idle threads and no competing pending jobs),
  apply the fragment-conversion procedure of `submit-cpu-sweep` SKILL.md section 10 IMMEDIATELY:
  cancel YOUR OWN pending big-shape ids (from your id file, each re-checked still PENDING) and
  resubmit the same budget as unpinned fragment fillers — the largest shape each live fragment takes
  (`28x1` / `14x1` / `12x1`) — keeping your pool total at or under cap minus the 16-CPU headroom.
- **Env**: `/p/rlprojects/RND/.venvs/exploration/bin/python` (shared canonical; Python 3.11,
  torch 2.10.0+cu128, SB3 2.7.1, gymnasium 1.2.3, gymnasium-robotics 1.3.1, mujoco 3.1.6).
- **Time limit**: `launch_workers_collaborator.sh` sets `--time` for you — the smallest of the
  partition limit (4 days for cpu/gpu, 20 days for nolim/gnolim) and the time to the next
  maintenance window minus 30 minutes. The current window starts **2026-08-05 07:30**, so every job
  submitted now ends before it. You never set `--time` by hand.
- **Owner**: `sl5nw`. The owner runs the queue build, the stage-1 prune controller, requeue-orphans,
  the GPU task-R jobs, and the 20-minute monitor. You only add worker jobs and (if your environment
  breaks) write a problem report.
- **Your caps are independent**: each user has their own cpu 400 / nolim 80 / gpu 400 / gnolim 80.
  Adding your workers stacks on top of the owner's fleet.
