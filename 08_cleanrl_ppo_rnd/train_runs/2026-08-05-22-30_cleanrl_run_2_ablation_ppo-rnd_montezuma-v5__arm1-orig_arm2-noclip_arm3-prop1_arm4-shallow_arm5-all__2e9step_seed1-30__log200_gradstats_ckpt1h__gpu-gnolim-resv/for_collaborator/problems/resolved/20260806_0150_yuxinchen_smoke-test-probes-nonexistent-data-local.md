# smoke_test.sh fails on a data path this run never creates (`$RUN_DIR/data/local`)

- **Reported by:** yuxinchen (collaborator), 2026-08-06 01:50
- **Job id:** 6534099 (gpu, cheetah02, FAILED 1:0 at 00:00:00)
- **Log:** `for_collaborator/logs/smoke_yuxinchen_6534099.out`

## What I saw

`bash for_collaborator/smoke_test.sh` exits at section 1 before running any training:

```
  ok  rwx .../queue/pending
  ok  rwx .../queue/running
  ok  rwx .../queue/done
  ok  rwx .../queue/failed
  FAIL     .../data/local
  ok  rwx .../for_collaborator/logs
  ok  read a queue marker
  ok  shared env python is executable by you
PERMISSION PROBES FAILED - write a problem report, do not submit
```

## Cause (owner side, one line in the packet)

The probe list in `smoke_test.sh` checks `$RUN_DIR/data/local`. This run's data layout nests the
sweep id one level deeper:

```
data/2026-08-05-22-30_ablation/local/38_of_150.json
```

`$RUN_DIR/data/local` does not exist and is never created — `ls -d data/local` →
`No such file or directory`. The queue markers agree: their output path is under
`data/$SWEEP_ID/local`, and the owner's own workers are writing there now.

So this is a stale path in the probe list, not a permission fault. The permission the probe
means to check is fine — I verified it directly under my uid:

```
touch data/2026-08-05-22-30_ablation/local/.perm_probe_yuxinchen_257020   -> ok
rm    data/2026-08-05-22-30_ablation/local/.perm_probe_yuxinchen_257020   -> ok
```

(Directory is `drwxrwsr-x+ sl5nw rlprojects`, so group `rlprojects` can write.)

## Suggested fix

In `smoke_test.sh` section 1, change the probed path to `"$RUN_DIR/data/$SWEEP_ID/local"`
(`SWEEP_ID` is already exported by `packet_env.sh`).

## What I did, since the gate is required before submitting

I did not edit `smoke_test.sh` or anything else owner-side. I ran the substantive half of the
gate — the five per-arm trainings, identical arms, flags, timeouts and output location as the
packet's section 2 — from my own sbatch script, with the probe list corrected to
`data/$SWEEP_ID/local`. Job id 6534100, outputs under
`for_collaborator/smoke_data/yuxinchen/`. I submit workers only if that passes 5/5.

No shared state was modified.

---

## Resolution (owner, 2026-08-06 02:10)

**Confirmed exactly as reported, and fixed.** The diagnosis was right in every particular: the probe
list named `$RUN_DIR/data/local`, this run nests the sweep id one level deeper, and the permission
the probe meant to check was never in question.

`for_collaborator/smoke_test.sh` line 11 now probes `"$RUN_DIR/data/$SWEEP_ID/local"`. `SWEEP_ID` was
already exported by `packet_env.sh`, so nothing else changed. Re-ran the probe section under the
owner's uid: all six probes pass.

**Cause on the owner's side.** The packet's probe list was written against the generic layout in the
collaborator-handbook skill, where the run outputs sit at `data/local`. This run follows the RND
project's own work-queue convention, which scopes outputs by sweep id — `data/<sweep_id>/local/` —
so that two sweeps in one run folder cannot collide. The packet was not re-checked against the
layout the queue markers actually carry. The markers were the authority the whole time: every one of
them names `data/2026-08-05-22-30_ablation/local/<run_id>_of_150.json`.

**Please re-run `bash for_collaborator/smoke_test.sh`** — it should now reach section 2 and run the
five per-arm trainings. Your own corrected gate (job 6534100) tests the same thing, so if that
already passed 5/5, you are clear to submit workers either way.

**Two things done right on your side, for the record.** You did not edit anything owner-side, and you
ran the substantive half of the gate rather than skipping it — which is exactly the intended
behaviour when the packet itself is at fault. Nothing shared was modified.

**Note on capacity when you do submit.** Both of the owner's pools are currently at their caps
(`gpu` 384 of 400 CPU threads, `gnolim` 64 of 80), so the free GPUs you can see are not reachable by
the owner's uid. That is the whole reason for a second submitter — plan to the full per-user caps
under your own uid, with no headroom subtraction, per the handbook.

Added to the collaborator-handbook skill's `references/common_problems.md` so the next packet's probe
list is generated from the sweep's real data path rather than the generic one.
