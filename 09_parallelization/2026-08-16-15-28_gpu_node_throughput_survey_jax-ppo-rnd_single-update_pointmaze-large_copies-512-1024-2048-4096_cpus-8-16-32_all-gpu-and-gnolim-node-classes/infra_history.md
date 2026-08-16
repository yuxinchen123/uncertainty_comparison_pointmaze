# Infrastructure history

One block per monitoring tick that had something to record. Times are Pacific; the cluster's
machines run Eastern and are converted here.

## 2026-08-16 15:28 PT — first probe wave failed on every node it reached

All 23 probe jobs that started died in 3 to 5 seconds with
`RuntimeError: Unable to load CUDA. Is it installed?`, while `nvidia-smi` in the same job printed
a healthy card and driver on every one of them.

Cause: `pip install "jax[cuda12]"` had resolved the NVIDIA runtime packages as already satisfied,
from the user-site directory `~/.local`, so none of them were installed into
`/p/rlprojects/RND/.venvs/jax_gpu`. Every job sets `PYTHONNOUSERSITE=1` (it must — a user-site
torch otherwise shadows the environment's own), which hid the only copies that existed.

Fix: `PYTHONNOUSERSITE=1 pip install "jax-cuda12-plugin[with-cuda]==0.10.2"` into the environment,
then resubmit. Recorded in the environment's `PROVENANCE.md` so the next environment built here
does not repeat it. No measurement was affected — the failure was before any timing.

## 2026-08-16 16:05 PT — one measurement discarded: two of this survey's jobs shared jaguar03

Two ticks in quick succession both submitted `jaguar03__cpus-8`, because `sacct` had not yet
indexed the first submission and the advancer read only `sacct`. Both jobs ran on jaguar03 at the
same time, on different cards but sharing the machine's memory bandwidth.

Action: both ids cancelled, the partial result deleted, the cell re-run alone. Its numbers in
`results.md` are from the clean re-run.

Fix: the advancer now reads `squeue` as well as `sacct` (`squeue` shows a submission at once;
`sacct` lags by seconds), and every advance runs under `flock` through `code/advance.sh`, so the
20-minute monitor and a hand-run pass cannot overlap.

## 2026-08-16 16:20 PT — four jobs queued behind a host-memory request they never used

`cheetah08-09`, `jaguar02`, `jaguar05` and `affogato11` sat pending on `(Resources)` while their
nodes had a free card. The jobs asked for a quarter of the node's system memory — 32 GB on a
125 GiB node, 64 GB on a serval — and the nodes had less than that left.

Measured peak host memory over the first 51 jobs (`sacct` MaxRSS): 7.7 GiB. The request is now a
flat 16 GB. `cheetah08-09` and `affogato11` started within seconds of resubmission; the submission
scripts also now pick whichever node of a class has the most free capacity, which moved
`cheetah08-09` from the full `cheetah08` to `cheetah09`.

## 2026-08-16 17:05 PT — a queued job held a node pin that had gone stale

`gputhr-ai05_ai10-c30` (job 6538379) sat on `(Resources)`: it was pinned to `ai05`, which had 28
of the 30 processors it needs, while `ai10` — the same node class, identical hardware — was
completely idle. Slurm stores the node list at submit time, so regenerating the submission script
does not move a job that is already queued.

Action: that one id cancelled, with the reason recorded, and resubmitted from the current script.
It started on `ai10` immediately.

Worth knowing for the remaining queued jobs: `jaguar02`, `jaguar05` and `affogato11` are each the
only node of their class, so there is no sibling to move them to — they wait for the other user
holding those processors. `serval03` is in maintenance.
