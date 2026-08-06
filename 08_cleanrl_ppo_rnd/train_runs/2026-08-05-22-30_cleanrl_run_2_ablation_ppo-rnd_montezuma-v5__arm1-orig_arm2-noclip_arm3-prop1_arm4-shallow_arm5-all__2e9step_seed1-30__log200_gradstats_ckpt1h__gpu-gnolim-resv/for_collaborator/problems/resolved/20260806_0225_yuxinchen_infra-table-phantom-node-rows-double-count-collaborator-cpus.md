# infra_history's collaborator CPU total is ~2x too high: five phantom "node" rows named 0–4

- **Reported by:** yuxinchen (collaborator), 2026-08-06 02:25
- **Where:** `infra_history.md`, blocks `2026-08-06 02:08` and `2026-08-06 02:09`
- **Nothing is wrong with the runs.** This is an accounting defect in the table only. All 21 of my
  jobs are RUNNING and healthy.

## What the table shows

Below the real per-node rows, the 02:09 block carries five extra grey rows whose node cell is a bare
integer:

```
| 0 (yuxinchen) | not recorded | ... | not recorded/2   | ... |
| 1 (yuxinchen) | not recorded | ... | not recorded/48  | ... |
| 2 (yuxinchen) | not recorded | ... | not recorded/16  | ... |
| 3 (yuxinchen) | not recorded | ... | not recorded/264 | ... |
| 4 (yuxinchen) | not recorded | ... | not recorded/64  | ... |
```

They land in the subtotal: **TOTAL (collaborators) reads `207.0/762` CPU threads**, while my true
allocation is **396** (`gpu` 320 + `gnolim` 76, from `squeue -u yuxinchen -t R -o %C`). The five
phantom values sum to 394 — my whole allocation counted a second time.

## Cause: `monitor.py` reads column 3 of a collaborator id file as a node name

The two id-file formats differ, and the phantom keys are exactly the distinct values of **column 3**
of mine:

- **Owner's** `slurm/submitted_jobids_<sweep_id>.txt` — one bare job id per line, no columns.
- **Collaborator's**, written by the packet's own `launch_workers_collaborator.sh`:
  `id node g 1 cpus_per_run sweep_id timestamp` — so column 3 is `g`, the GPU/slot count of that job,
  which for my jobs takes the values 0, 1, 2, 3, 4.

Grouping my RUNNING jobs by column 3 and summing `sacct AllocCPUS` reproduces the five rows exactly:

| column 3 (`g`) | summed AllocCPUS | phantom row in the table |
|---|---|---|
| 0 (my monitor job) | 2 | `0 → 2` |
| 1 | 48 | `1 → 48` |
| 2 | 16 | `2 → 16` |
| 3 | 264 | `3 → 264` |
| 4 | 64 | `4 → 64` |

Five for five. The real named rows (`cheetah01 (yuxinchen)` etc.) come from the node_status/squeue
path and are correct; the phantom rows are a second, id-file-derived path that treats `g` as a node.

## Why it matters beyond cosmetics

The table is what a top-up decision reads. An inflated collaborator subtotal says the sweep already
has ~762 collaborator CPU threads when it has 396, which understates how much capacity is still
worth adding. It also makes `TOTAL` (1202 asked) unreliable as a campaign-wide figure.

## Suggested fix

Parse a collaborator id line by taking `fields[0]` as the job id only, and derive the node from
squeue/sacct as the owner's own path already does — never from a positional column, since the two id
files legitimately have different shapes. If a positional read is wanted, key on the header the
packet writes rather than an index.

## Note on my id file

Two of my lines use non-numeric column 3 by design and are worth keeping in mind for the parser:
`... cheetah02 SMOKE ...` and `... cheetah02 SMOKE2 ...` (the smoke-test jobs), plus
`... MONITOR 0 ...` (my passive monitor job on gnolim, 1 cpu). I have left the file exactly as the
packet's launcher writes it and have not hand-edited any line.

I changed nothing owner-side. Related: `20260806_0205_yuxinchen_worker-slot-died-on-stale-nfs-handle.md`.

---

## Resolution (owner, 2026-08-06 02:35)

**Confirmed, and the diagnosis was right down to the five-for-five table.** The phantom keys are the
distinct values of a positional column, read as node names.

**The root cause is one field further back than the report placed it, and it is the owner's.** The
documented id-file line is **eight** fields:

```
<jobid> <node_class> <node> <G> <W> <c_used> <group> <submit_ts>
```

The packet's `launch_workers_collaborator.sh` wrote **seven** — it omitted `node_class`. So every
field shifted left by one: the node landed in `node_class`, and `G` landed in `node`. That is why the
phantom keys are exactly your GPU counts. `monitor.py` was reading the documented position correctly;
it was being handed a line of the wrong width.

Both sides are fixed, because either alone would have left a sharp edge:

1. **The launcher now writes the documented eight fields**, so new lines parse correctly everywhere.
2. **`monitor.py` now takes the node from Slurm, not from the column**, for every running job. Slurm
   is the authority on where a job is; the column is a convenience. This also repairs your *existing*
   lines without anyone editing your id file — which matters, because one writer per file is the rule
   and the owner must not rewrite it.

Fixing the launcher's line width also broke its own depth guard, which read the seven-field shape. It
now parses **by field count**, so an old seven-field line and a new eight-field one are both read
correctly rather than one being silently misread as the other. Your `SMOKE`, `SMOKE2` and `MONITOR`
rows are skipped by a numeric check on the `G` and `W` fields — thank you for flagging them, they
would otherwise have produced an arithmetic error.

**Verified on the newest infra block:** no phantom rows, and the totals reconcile against `sacct`:

| | from sacct | in the table |
|---|---|---|
| your 21 running jobs | 394 threads | 370 |
| owner's 11 running jobs | 440 threads | 464 |
| total | **834** | **834** |

The grand total is exact. The 24-thread difference in the split is the table's documented mixed-node
convention, not an error: `adriatic04` and `adriatic05` carry both our jobs, and a mixed node counts
into the owner subtotal. Your figure of 396 at 02:25 matches the 394 measured now.

**On the wider point you made — that the table is what a top-up decision reads — you were right, and
it was worse than the doubling.** Chasing this turned up a second defect in the same file: `monitor.py`
read only `slurm/submitted_jobids.txt`, while this sweep uses the shared per-sweep convention, so the
**owner's** jobs were invisible to the monitor entirely. A marker whose claiming job is never queried
can never be judged orphaned, so an owner job hitting its 4-day walltime would have left its runs in
`running/` forever. Fixed in the same pass.

Three reports, three real defects, two of them in shared tooling that every sweep uses. This is
exactly what the problem channel is for.
