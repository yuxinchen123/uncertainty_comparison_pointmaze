# Run 8.1 (point maze + ant maze) — resource facts for collaborators

- **SWEEP_ID**: `2026-07-23-02-05_pm-am-run1` (also in `packet_env.sh`).
- **Run total**: 92,400 runs = 308 configs x up to 300 seeds (0-299). Each run = 1,000,000 SAC steps.
  Configs: SAC-alone (`no_exploration`, 8), SAC+RND (`rnd_next_state`, 15 betas x 8 env = 120),
  and three ground-truth visit-count bonuses (`gt_position_velocity` PointMaze-only 60,
  `gt_position_maze_cell` AntMaze-only 60, `gt_position_1m` AntMaze-only 60). Full detail in the run
  folder's `experiment_background.md`.
- **Racing / why pending shrinks over time**: per-(env, algorithm) cell racing (the shared
  `sweep_prune` design, this run's overrides). Prune floor 20 seeds, winner-only continuation at 100,
  maximum 300. The OWNER's prune controller moves losing configs' pending markers from `pending/` to
  `pruned/`, so the pending count drops faster than workers finish runs. A collaborator worker that
  claims a marker pruned in-flight is HARMLESS (it just runs one extra seed) — you never prune, and
  you never need to react to it.
- **Per-run cost (estimate; superseded by the run-level probe, see below)**: on CPU a 1M-step run is
  roughly 9-25 h (PointMaze end of that range faster; AntMaze is heavier — 29-d state, MuJoCo Ant
  dynamics — and runs longer per step). Memory: PointMaze ~630 MB average / ~765 MB peak RSS per
  worker; AntMaze somewhat larger. The `--mem-per-cpu=2G` in every worker shape covers both.
- **Partitions you may use** (open partitions of YOUR own pools only):
  - `cpu` (per-user cap 400 CPUs)
  - `nolim` (per-user cap 80 CPUs)
  - `gpu` partition, **CPU-ONLY** jobs (`--gpus-per-node=0`), on the LEAST capable GPU nodes first
    (the launcher reads the shared node catalog and fills low-tier GPU nodes so their otherwise-idle
    CPUs are used while better GPUs stay free for real GPU jobs) — per-user gpu-pool cap 400 CPUs.
  - **NOT**: the owner's reservation (puma01/jaguar03), `gnolim`, or any GPU-using job. The sweep's
    GPU packing count W is still being probed by the owner, so `device=cuda` submission stays
    owner-only for now.
- **Job shapes**: `30x1` (big, ~32-core node with 2-core headroom), `28x1` (~30-core), `14x1`
  (~16-core class; the 16-core adriatic/affogato class rejects 16x1 outright), `12x1` (~14-core /
  small fragments). All use `--cpus-per-task=1 --ntasks-per-core=2 --mem-per-cpu=2G` and
  `srun --wait=0`. **Verify `AllocCPUS == ntasks`** (30/28/14/12) on your first submit of each shape;
  a doubled AllocCPUS means `--ntasks-per-core=2` was lost — write a problem report.
- **Fragments**: if the launcher's whole `30x1` cpu jobs sit PENDING while the partition holds idle
  fragments (`sinfo -p cpu -N` shows nodes with 12-30 idle threads and no competing pending jobs),
  apply the fragment-conversion procedure of `submit-cpu-sweep` SKILL.md section 10 IMMEDIATELY:
  cancel YOUR OWN pending big-shape ids (from your id file, each re-checked still PENDING) and
  resubmit the same budget as unpinned fragment fillers — the largest shape each live fragment takes
  (`28x1` / `14x1` / `12x1`) — keeping your pool total at or under cap minus the 16-CPU headroom.
- **Env**: `/p/rlprojects/RND/.venvs/exploration/bin/python` (shared canonical; Python 3.11).
- **Time limit**: `launch_workers_collaborator.sh` sets `--time` for you = min(4 days, next
  maintenance start minus 30 min). You never set it by hand.
- **Owner**: `sl5nw`. The owner runs the queue build, prune controller, requeue-orphans, and the
  20-minute monitor. You only add workers and (if your env breaks) write a problem report.
- **Your caps are independent**: each user has their own cpu 400 / nolim 80 / gpu 400. Adding your
  workers stacks on top of the owner's fleet.

## The measured probe supersedes these estimates

The run-level `resource_facts.md` (written by the owner's CPU/GPU resource probe) appears at the run
folder root — `../resource_facts.md` — once the probe finishes. It carries the MEASURED per-family
runtime and RSS (PointMaze vs AntMaze). **Check it and let it override the estimates above** before
sizing anything unusual. Until it exists, the `--mem-per-cpu=2G` shapes and the ~9-25 h/run estimate
are the working numbers.
