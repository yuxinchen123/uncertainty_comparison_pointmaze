# Walltime kill and resume: rehearsed before the real one, and it took three attempts

Run on 2026-08-06, four days before all nine `gpu` jobs of the CleanRL RND ablation hit their
walltime at once. Isolated sweep folder with its own queue, data directory and id file, so nothing
here could touch the campaign.

## The chain being tested

Walltime approaches → Slurm signals early → the batch shell forwards it → `worker_manager` forwards
it to each child → the trainer checkpoints at the next iteration boundary and exits cleanly → its
marker stays in `running/` → the monitor requeues it → a new job claims it → the trainer resumes.

Every link had been checked alone. The chain had not, and run 79 of the campaign was direct evidence
it could break.

## Three attempts, two defects

| attempt | job | what happened |
|---|---|---|
| 1 | 6534144 | Ran 24 minutes on a 12-minute limit, still training. **The trainers never received a signal at all.** |
| 2 | 6534146 | Signal arrived, both trainers caught it, **neither finished its checkpoint**; job exited 15:0 three minutes early. |
| 3 | 6534147 | **COMPLETED 0:0.** Both trainers checkpointed and exited cleanly. |

**Defect 1 — the warning was delivered to nothing.** `--signal=<sig>@<time>` without a prefix targets
*job steps*. These jobs have no `srun` step; `worker_manager` is a plain child of the batch shell. So
Slurm had nothing to signal. `--signal=B:TERM@600` sends it to the batch shell instead, and because
bash does not pass a signal to a foreground child, the manager has to run in the background under a
trap that forwards it.

**Defect 2 — `wait` returns when the trap fires.** Bash interrupts `wait` to run a trap and returns
128+15 immediately; it does not wait for the child. The script then fell off the end and Slurm tore
the job down while the trainers were writing the checkpoint the signal existed to let them write. The
fix is to wait again until the child is genuinely gone:

```bash
wait "$MANAGER_PID"
while kill -0 "$MANAGER_PID" 2>/dev/null; do wait "$MANAGER_PID"; done
```

Both defects were in code I had already called fixed. The first was introduced as a fix two ticks
earlier and was inert from the moment it was written; the second was in a pattern I had called proven,
which was proven only for the case where the signal never arrives.

## What attempt 3 shows, in the logs

```
[slurm] walltime approaching; forwarding SIGTERM to worker_manager 81910
[signal] caught signal 15; will checkpoint at the next iteration boundary
[signal] termination requested; checkpointing at update 252
[signal] checkpointed at global_step 4128768; exiting
manager done on lynx05 (job 6534147)
```

and the checkpoints on disk agree with what the trainers said they wrote:

| run | checkpoint | trainer reported |
|---|---|---|
| 0 | format v2, global_step 4,128,768, update 252, 12,203 episodes | global_step 4,128,768 |
| 1 | format v2, global_step 4,358,144, update 266, 11,652 episodes | global_step 4,358,144 |

Both markers were left in `running/` for the owner's requeue, which is the designed behaviour rather
than a failure.

## The resume half passed on its own

Attempt 3's runs began by resuming attempt 2's checkpoints without being told to —
`[resume] continuing at update 142 of 122070, global_step 2310144`, with the episode sidecar cut to
match and nothing dropped past the checkpoint. So claim → run → kill → requeue → re-claim → resume
works end to end under the manager.

## What this cost, and what it would have cost

The decoupled checkpoint cadence carried the whole thing: checkpoints landed every ~90 s as asked,
and in the two failed attempts the last one was 30 s before the kill. So even a completely broken
shutdown loses less than one checkpoint interval. On the campaign's hourly cadence that is at most an
hour of one run, roughly 105 GPU-hours across a full kill wave, or 0.4% of the campaign — the reason
this was worth fixing carefully rather than urgently.
