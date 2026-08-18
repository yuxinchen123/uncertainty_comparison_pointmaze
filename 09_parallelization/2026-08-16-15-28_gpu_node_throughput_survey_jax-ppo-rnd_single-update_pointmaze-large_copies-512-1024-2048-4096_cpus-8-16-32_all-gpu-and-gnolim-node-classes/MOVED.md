# This survey has moved

Moved 2026-08-17 10:13 PT to

```
/p/rlprojects/.claude/skills/rnd-jax-submission/gpu_node_throughput_survey/
```

where it is the data behind the shared skill `rnd-jax-submission`, the way
`school_compute_resource/` is the data behind `uva-submit-gpu-sweep`. Everything is there
unchanged — `experiment_background.md`, `code/`, `data/probe/`, `data/throughput/`, `plots/`,
`results.md`, `results.html`, `slurm/`. All 923 files were verified present at the new path, and
the `data/` tree's checksum matched, before this path was emptied.

`slurm/logs/` moved with the rest and is on disk at the new path, but it is not committed to git
(1.7 MB of raw job output; an ignore entry in `/p/rlprojects/.claude/.gitignore` covers it). Every
measurement it produced is in `data/` and `results.md`, which are committed.

## The state the survey finished in

25 of the cluster's 27 gpu and gnolim node classes are measured. The last three jobs —
`gputhr-jaguar02-c8` (6538407), `gputhr-jaguar05-c8` (6538409) and `gputhr-cheetah08-09-c32`
(6538417) — were cancelled on 2026-08-17 at the user's instruction that those cards no longer
needed measuring; they had been queued behind other users with a projected start about 20 hours
out. The rows they would have filled are carried in the skill as **interpolated** values, marked
as such with their basis stated. Nothing else was cancelled.

`gputhr-serval03-c8` (6538322), which had been blocked behind the `cs_admin_maint` reservation,
started and completed on its own at 2026-08-17 09:47 EDT, so the H100 NVL of `serval03` is
measured rather than inherited.

## Why the old path is kept

Other sessions' notes, the run's `experiment_background.md` history and earlier reports name this
folder. The directory stays with this file in it so those references resolve instead of hitting a
missing path.
