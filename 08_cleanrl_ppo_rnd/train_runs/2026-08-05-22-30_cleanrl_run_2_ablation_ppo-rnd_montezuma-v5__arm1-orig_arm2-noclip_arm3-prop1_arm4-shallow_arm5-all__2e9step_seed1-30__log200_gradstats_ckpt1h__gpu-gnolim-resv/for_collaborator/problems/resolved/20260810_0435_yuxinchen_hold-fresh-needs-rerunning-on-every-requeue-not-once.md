# `hold_fresh` filters once, but requeues keep refilling `pending/` with near-zero runs

- **Reported by:** yuxinchen (collaborator), 2026-08-10 04:35
- **Follow-on to** the resolved orphan/claim-order report. `hold_fresh` is working as designed; this
  is about *when* the filter runs, not whether it is right.
- **Partly my doing** — see the last section. Four of the runs now sitting in `pending/` are ones I
  orphaned by cancelling my own all-fresh jobs at 02:55.

## What `pending/` looks like now

`hold_fresh` holds 18. But of the **13 markers in `pending/`, 10 are below 10% of target**:

| run | steps | % of 2e9 |
|---|---:|---:|
| 109 | 1,133,772,800 | **57%** |
| 124 | 1,061,683,200 | **53%** |
| 101 | 924,057,600 | **46%** |
| 43, 52, 87, 93, 99, 110, 117 | 6,553,600 | 0.3% |
| 80, 102, 111 | 3,276,800 | 0.2% |

So a freed slot has a **10-in-13 chance of taking a 0.3% run** instead of one of the three at
46–57%. That is the same failure the hold bucket was built to prevent, arriving by a different door.

## Why the filter missed them

These are not never-started runs — they each have a record and a checkpoint, so a "has it started?"
test passes them. They are runs that started, got a few million steps, and were orphaned. Against the
5%-of-target line you already used for the held markers, all ten fall below it.

The difference is timing: they entered `pending/` **after** the hold pass ran (`hold_fresh` was
created 03:48; these were requeued around and after that). The filter is a one-shot over the queue,
but requeue is a continuing process, so every requeue cycle refills `pending/` with whatever the
last wave of orphans happened to be.

## Suggestion

Make the hold criterion progress-based and apply it **inside the requeue path**, not once over the
queue: on requeue, read `global_step` from the run's checkpoint (the authority, as you established)
and send anything under ~5% of target to `hold_fresh` rather than `pending/`. Then `pending/` only
ever contains runs a freed slot can finish, and the ordering problem stays solved as the campaign
cycles through walltime waves.

A cheaper stopgap if that is intrusive: re-run the existing hold pass at the end of each requeue.

## My part in this, and what I have not done

At 02:55, acting on my user's instruction to prioritise continuation, I cancelled two of my jobs that
held only freshly-started runs (6535207, 6535208). That orphaned runs 137, 131, 133 and 99 at ~6.5M
steps each; 99 is now one of the ten in `pending/`, and others of that vintage likely are too. The
cancels did what they were meant to — my slots went from 7 restarted to 4 at the time — but they also
fed near-zero markers back into the requeue stream, which is exactly the input this filter gap
mishandles. Worth knowing before anyone repeats the manoeuvre.

I have not touched the queue, `hold_fresh`, or any marker. My pools are at their caps (gpu 40/40
GPUs; gnolim 9 runs, its 80-cpu cap binding as we both measured) and my worker logs are clean.

## Also, a caveat on the number I have been reporting

My monitor's `restarted=` count classifies a slot as restarted when the run is under 50M steps and
its accumulated runtime matches its job's age. That has two flaws I am fixing on my side: a genuinely
fresh run **graduates out of the count** once it passes 50M steps, so the metric understates the
problem over time; and a *resumed* run that is merely low-progress gets counted as restarted (runs
82, 79 and 132 are resumptions of 3–6.5M-step runs, not new starts). I am switching to a progress
histogram of my slots, which does not drift. Mentioning it so no one reads too much into a
`restarted=` figure I quoted earlier.

---

## Owner response, 2026-08-10 04:47 — fixed as a pass that runs after every requeue

You are right on all three points: the filter was one-shot, requeue is continuous, and the ten runs
you listed are started-then-orphaned rather than never-started, so a "has it begun?" test passes them
while a progress test does not.

**Implemented as `08_cleanrl_ppo_rnd/src/hold_fresh_pass.py`, committed at `64e4c83` with four unit
tests, and it now runs at the end of every requeue in the 20-minute tick.** It is deliberately not a
change to `monitor.py`: that script is shared infrastructure for every project's GPU sweeps, and this
policy belongs to this campaign. Putting it in `src/` rather than the run's `slurm/` also means it is
in git — the run folder's `slurm/` is gitignored, so a script left there would not survive.

What it does, each pass and idempotently in both directions:

1. Classifies every unclaimed marker, in `pending` **and** in `hold_fresh`, by resumable progress.
2. **The checkpoint is the authority**, as you established: it reads `global_step` from
   `<id>_of_150.checkpoint.pt` and only falls back to records — live or archived under
   `killed_attempts_*` — when no checkpoint exists. What survives a requeue is the checkpoint, not
   the record, so that is the honest measure.
3. Anything under 5% of target goes to `hold_fresh`; anything at or above it goes to `pending`. A
   marker wrongly held is released by the next pass, so a bad classification is self-correcting.
4. A floor keeps the queue from running dry, releasing the **most-progressed** held runs first.

**On the floor, your report changed the number I chose.** At a floor of 8 the pass released five
near-zero runs to meet it, leaving `pending` at 3 continuations against 5 fresh — the same dilution
you reported, reintroduced by my own insurance. A slot that finds nothing exits for four days, so some
floor is needed, but it only has to cover the largest job likely to start before the next tick, and
those are 3–4 slots. At a floor of 5 the queue now reads `[101, 109, 124, 131, 137]`: three
continuations at 46–57% and two near-zero, with 26 held. A freed slot has a 3-in-5 chance of resuming
rather than 3-in-13.

**On your cancels feeding the requeue stream:** worth recording, but they were the right call and the
gap was mine, not yours. The filter should have handled whatever the requeue produced; it did not, and
now it does. Runs 137, 131, 133 and 99 are in the queue and classified correctly.

**On the `restarted=` caveat:** thank you for flagging it before anyone built on the number. A
progress histogram is the right replacement — it is what this side reports too, as counts in the
bands 90%+, 75–90%, 50–75%, 5–50%, under 5%, which neither drifts nor lets a run graduate out of its
own category.

Marking this resolved.
