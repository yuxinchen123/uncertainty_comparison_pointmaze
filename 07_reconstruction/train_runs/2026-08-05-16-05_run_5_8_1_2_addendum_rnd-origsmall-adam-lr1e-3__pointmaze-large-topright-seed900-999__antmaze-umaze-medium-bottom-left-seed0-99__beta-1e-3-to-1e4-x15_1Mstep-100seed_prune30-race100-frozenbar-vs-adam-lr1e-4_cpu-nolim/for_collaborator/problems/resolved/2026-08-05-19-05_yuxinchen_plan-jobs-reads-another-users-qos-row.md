# plan_jobs.py sizes a pool from the FIRST user row of the QOS record, not the running user's

No queue markers are affected — **no `MARKER:` lines in this report**. This is a sizing bug that
silently zeroes a collaborator's pool.

## Symptom

Running `DRY=1 bash for_collaborator/launch_workers_collaborator.sh` as `yuxinchen` at 18:50, with
**zero** nolim jobs of mine running or pending:

```
[nolim] cap 80, in use 64, headroom 16 -> 0 threads to fill
```

The 64 is `sl5nw`'s nolim usage, not mine. My real room was 64 slots, and the launcher would have
left the whole nolim pool empty for as long as the owner sat at 64.

## Cause

`slurm/plan_jobs.py`, `pool_room()`:

```python
m = re.search(rf"MaxTRESPU=cpu={cap}\((\d+)\)", out)
```

`scontrol show assoc_mgr qos=<qos> flags=qos` prints one `MaxTRESPU=` line per user under
`User Limits` — every user of the QOS, in an order this code does not control:

```
    User Limits
      sl5nw(2618919)
        MaxTRESPU=cpu=80(64),...      <-- re.search stops here
      bcw3zj(15367167)
        MaxTRESPU=cpu=80(0),...
      ...
      yuxinchen(...)
        MaxTRESPU=cpu=80(0),...       <-- the row that is actually mine
```

`re.search` returns the first match in the whole record, so whoever appears first wins. The cap
(`cpu=80`) is identical on every row, so the regex cannot tell them apart.

Why it never bit the owner: `sl5nw`'s row is first in both the cpu and nolim records, so for the
owner the first match *is* the right row. Only a collaborator reads someone else's number. The
`max(qos_used, my_cpus_in(partition))` guard added at 17:00 does not help here — it takes the
larger, so a foreign row that is too HIGH propagates straight through as phantom usage.

Direction of the error, for the record: too-high is the safe direction (under-submit, never
over-submit), so nothing was over-run at any point. The cost is an idle pool.

## Fix

Anchor the search to the running user's own row — the `User Limits` block is keyed by
`<user>(<uid>)`:

```python
user = getpass.getuser()
m = re.search(rf"^\s+{re.escape(user)}\(\d+\)\s*\n(?:.*\n)*?\s+MaxTRESPU=cpu={cap}\((\d+)\)",
              out, re.M)
qos_used = int(m.group(1)) if m else 0
```

or, more robustly, drop the QOS record for this purpose and read the user's own row with
`sacctmgr -nP show assoc user=$USER qos=<qos>`. Either way the existing `max(..., my_cpus_in(...))`
guard stays — it is what catches the opposite failure (the cpu counter reading `400(0)` while 384
CPUs of the same user were running, your 17:00 entry).

Worth checking the same pattern in `slurm/monitor.sh` and anywhere else the owner's tooling parses
`assoc_mgr` output, and in the collab-handbook skill if the packet generator carries this code.

## What I did in the meantime

I did not edit `plan_jobs.py` (owner's file). I submitted my nolim workers with a small script of my
own that computes room from **my** row plus `squeue -u $USER -p nolim`, and emits byte-for-byte the
same sbatch shape `plan_jobs.py` would have emitted (same worker script, `--ntasks-per-core=2`,
`srun --wait=0` via the packet script, 2 GB per worker, `--time=20-00:00:00`, ids appended to my own
id file). The cpu pool was planned and submitted by your launcher unchanged — its first row happened
to read 0, so it sized my cpu room correctly.

---
RESOLVED 2026-08-05T19:10 by the sweep owner (sl5nw). Confirmed and fixed exactly as diagnosed.
`pool_room()` in `slurm/plan_jobs.py` now anchors on the running user's own `<user>(<uid>)` heading
inside the `User Limits` block:

    m = re.search(rf"^\s+{re.escape(user)}\(\d+\)\s*$.*?MaxTRESPU=cpu={cap}\((\d+)\)", out, re.M|re.S)

Verified live at the moment of the fix: the old unanchored search returned 0 for the cpu QOS and 64
for nolim (both `sl5nw`'s rows); the anchored search returns 398/64 for `sl5nw` and 382/64 for
`yuxinchen` — each user's own number. The `max(qos_used, my_cpus_in(partition))` guard is unchanged,
as you recommended. No queue markers were involved and none were requeued.
