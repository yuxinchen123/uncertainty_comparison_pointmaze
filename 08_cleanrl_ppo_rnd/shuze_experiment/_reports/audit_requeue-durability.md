## Verdict

**The claim is confirmed, in full, with no correction needed.** One refinement and one piece of live evidence:

- The re-started run's curve is not "shifted to zero" — `global_step` is restored from the checkpoint, so the new record's `step` axis is correct and the orphaned prefix in `killed_attempts_*/` is in principle re-joinable by hand. But the two pieces **overlap** (the archived JSON runs past the checkpoint), so a naive concatenation double-counts.
- **It has already fired once on this sweep.** `data/killed_attempts_2026-08-05-18-28/` holds 8 records (runs 3, 4, 8, 9, 10, 12, 16, 19) — the runs of job 6533906, cancelled at 18:28 per `slurm/submitted_jobids.txt` ("cancelled 6533906: its script packs 2 runs per GPU but was given W=1 cpus"). It was harmless only because no checkpoint existed yet (checkpoint cadence is 8 h, the sweep started at 17:45; `data/local/` contains zero `.pt` files). The next occurrence, after hour 8, will not be harmless.

---

## 1. What `requeue_orphans` moves and what it leaves

`/p/rlprojects/.claude/skills/submit-gpu-sweep/scripts/monitor.py:958-993`. The only path it touches is `completed_marker_path`:

```python
        # archive the partial run output if the pipeline wrote one
        archived_to = None
        out_path = orphan.get("completed_marker_path")
        if out_path and os.path.exists(out_path):
            os.makedirs(killed_dir, exist_ok=True)
            archived_to = os.path.join(killed_dir, os.path.basename(out_path))
            shutil.move(out_path, archived_to)
```

`completed_marker_path` comes from the queue marker (`monitor.py:951-952`), and this sweep's `configs.jsonl` sets it to exactly one file:

```json
"completed_marker": {"path": ".../data/local/0_of_30.json", "flag_field": "completed"}
```

The word is `shutil.move`, not copy. Nothing else in `requeue_orphans` (or anywhere in `monitor.py` — `grep` finds no other `shutil`/`os.remove` on run outputs) touches the run's output directory. The `.checkpoint.pt` is invisible to the monitor: it is never named in the marker, never scanned for, never moved, never deleted.

The rest of the function only rewrites the queue marker (`monitor.py:983-990`):

```python
        marker.pop("claim", None)
        pending_path = os.path.join(pending_dir, name)
```

So: **JSON record moved away, checkpoint left in place, marker back in `pending/` with `argv` intact.** Confirmed.

## 2. The consequence in the trainer

`src/ppo_rnd_envpool_shuze.py:448-449` derives both paths from the same `run_id`, in the same directory:

```python
    record_path = os.path.join(args.output_dir, f"{args.run_id}_of_{args.run_total}.json")
    checkpoint_path = os.path.join(args.output_dir, f"{args.run_id}_of_{args.run_total}.checkpoint.pt")
```

`src/ppo_rnd_envpool_shuze.py:471-484` — the two restores are independent and only the checkpoint's result is inspected:

```python
    global_step, start_update = 0, 1
    resumed = False
    if args.resume:
        record.restore_from_disk()
        state = load_checkpoint(
            checkpoint_path, agent=agent, rnd_model=rnd_model, optimizer=optimizer,
            obs_rms=obs_rms, reward_rms=reward_rms, discounted_reward=discounted_reward, device=device,
        )
        if state is not None:
            global_step, start_update = state["global_step"], state["update"] + 1
```

`src/run_record.py:104-105` — a missing file is a silent fresh start:

```python
        if not os.path.exists(self.path):
            return False
```

The return value of `restore_from_disk()` is discarded at line 474, so nothing is printed and nothing branches on it. The `[resume]` line printed at 483-484 talks only about the training state, so the log will read as a clean resume while the history is empty. Confirmed exactly as claimed.

Two consequences beyond the ones you named:
- `episodes_seen` restarts at 0, so `episode_history_cap=50000` re-arms and the second segment records another 50,000 episodes at full resolution while the first segment was already at stride 100. The episode sampling density becomes discontinuous across the join.
- The record's config block (`slurm_job_id`, `hostname`, `gpu_name`, `device` — lines 460-464) is rewritten from the current invocation, so the final record names only the *last* job/node. The earlier segments' provenance survives only inside the archived files.

## 3. Does `worker_manager.py` change the picture?

No. It never touches run outputs.

- `claim_one` (`worker_manager.py:296-336`) is a pure `os.rename` of the marker from `pending/` to `running/`.
- The launch (`worker_manager.py:411-428`) re-runs `cfg["argv"]` verbatim — the same array `requeue_orphans` preserved. Only `claim`, `CUDA_VISIBLE_DEVICES` and a few `GPU_SWEEP_*` environment variables differ between attempts; **no argument is added or changed on a re-claim**, so `--resume` stays on and `--run_id`, `--seed`, `--output_dir` are identical.
- The run log is opened append-mode (`logf = open(log_path, "a")`, line 424), so `logs/run_<id>.log` is the one artifact that *does* survive a requeue intact.
- On exit it only rewrites and moves the marker (`finalize`, lines 371-397). On preemption it leaves the marker in `running/` deliberately (lines 461-465) — that is what makes the marker an orphan for the monitor to find.

One thing worth knowing: `completed_marker_ok` (lines 338-356) opens the JSON and returns `False` if it is missing. So a run that exits 0 but whose record has been archived is filed as **failed**, not done.

## 4. The checkpoint across a requeue

- **Never moved, never deleted, by anyone.** `save_checkpoint` (`checkpointing.py:77-92`) writes `path + ".tmp"` and `os.replace`s it over `path`; the only `os.remove` is for a `path + ".previous"` file that no current code writes. Nothing in `monitor.py` or `worker_manager.py` deletes it.
- **A re-claimed run does pick up the previous attempt's checkpoint, and that is correct.** Same `run_id`, same `--seed`, same `--output_dir`, therefore the same filename. Verified end to end: marker `argv` survives requeue unchanged, and both paths at lines 448-449 are functions of `run_id`/`run_total` only. There is no cross-run aliasing.
- **Two hazards this creates.** First, `--resume` is unconditional (`resume: bool = True`, line 125) and `output_dir` is `data/local` with no sweep-id scoping, so any future relaunch into the same run folder would silently continue the old checkpoints instead of starting fresh. Second, there is no fencing: nothing stops two processes from owning the same checkpoint path. `scan_orphans` requires a terminal `sacct` state before requeueing, which is the right gate, but `worker_manager.py:448-456` documents by name the case where the wrapper dies and "its training process survives, squats on the GPU". If that ever coincides with a requeue, two trainers write the same `.checkpoint.pt` and the same `.json` alternately. Rare, unfenced, and silent.

---

## 5. Critique of the proposed fix

Short version: **the mechanism is sound and cheap, but the precedence rule you proposed is backwards and will create duplicate rows, and the version bump as written will brick every currently running job.** Fix those two and it is a good fix.

### (a) Size — measured, not estimated. Not a problem.

Real numbers from this sweep:

| quantity | measured |
|---|---|
| current checkpoint | 50,258,398 B (`canary/data/0_of_30.checkpoint.pt`) |
| `num_iterations` | 2e9 // 16384 = **122,070** |
| log rows at `log_every_updates=25` | **4,883** each of `train_history`, `eval_history` |
| observed row sizes (run 0) | episode 118 B, train 213 B, eval 694 B |
| observed episode rate | 20,136 episodes at 8,601,600 steps = **2,341 per million steps** |
| episodes over 2e9 steps (rate held constant, an upper bound — episodes lengthen as the policy improves) | ~4.7M → kept rows = 50,000 + (4.7M−50k)/100 = **96,300** |

I built a synthetic history at exactly that end-of-run scale (96,300 episode rows, 4,883 train rows, 4,883 eval rows, all distinct objects so pickle cannot memoize) and measured:

```
torch.save bytes 5,572,056   sec 0.504
json bytes      16,661,589   sec 0.935
torch.load                   sec 0.173
```

So the checkpoint grows from 50.3 MB to **~55.8 MB, +11%**, at the very end of the run; less for most of it. Add 0.5 s to a write that happens every 8 hours, and 0.17 s to a read that happens once per resume. Across 30 runs that is 1.7 GB on disk instead of 1.5 GB. **Size is a non-issue and should not shape the design.**

(Note the pickle is 3x smaller than the JSON of the same data — pickle memoizes the repeated dict keys, JSON repeats every key string.)

### (b) Staleness — you have the sign wrong. The 8-hour lag is the *correct* behavior, not a defect.

The history in the checkpoint is stale by up to 8 hours relative to the JSON. That is exactly right, because **the training state being restored is stale by the same 8 hours.** The rows the JSON has and the checkpoint does not describe updates that the resumed run is about to *redo*. Keeping them would splice a discarded trajectory onto the kept one.

The correct invariant is: **the record's history must always be truncated to the checkpoint's `global_step`/`update`, because that is the only point in time the run actually continues from.** Under that invariant, checkpoint-carried history is not stale — it is the only self-consistent history there is.

Which is why the proposal's precedence is wrong:

> "on resume prefer the on-disk JSON record if present, else fall back to the history carried in the checkpoint"

This prefers the one source that is *guaranteed* to be inconsistent with the restored training state.

### (c) Yes, it produces duplicate rows. Here is the exact mechanism.

Take the "prefer the on-disk JSON" branch with both files present (below: when that happens). Let `U_ckpt` be the checkpoint's `update` and `U_json` the last logged update before the kill, with `U_ckpt ≤ U_json`.

1. `restore_from_disk()` loads `train_history`/`eval_history` rows for updates 25, 50, …, `U_json` (`run_record.py:110-115`).
2. `load_checkpoint` sets `start_update = U_ckpt + 1` (`ppo_rnd_envpool_shuze.py:480`).
3. The loop re-runs updates `U_ckpt+1 … U_json` and, at every multiple of 25, calls `record.add_update(...)` which does a bare `append` (`run_record.py:66-67`).

Result: **two rows for every logged update in `(U_ckpt, U_json]`**, and the `step` field goes backwards at the join, so any plot that assumes a monotone step axis draws a fold-back. Worse, `add_episode` (`run_record.py:56-62`) re-appends every episode replayed in that window and bumps `episodes_seen` again — so `episodes_seen` over-counts, and because the stride rule past the cap is `episodes_seen % stride == 0`, the inflated counter also shifts which later episodes are kept.

At an 8-hour checkpoint cadence and 4-day walltime, the duplicated window is on average **4 hours of training, ~50 million steps, ~120,000 episodes** — not a rounding error.

**When does that branch actually fire?** Not on the monitor's own requeue path (the JSON is always moved). It fires when:
- someone requeues a `failed/` marker by hand (the ordinary recovery for a crash, and the ordinary recovery once you notice a lost curve);
- the monitor loop job is itself down and the requeue is done manually;
- a future edit stops moving the record (which is the obvious follow-on cleanup).

So the proposed fix does not create duplicates on day one, but it writes the duplicate-producing rule into the code and arms it for exactly the manual recovery you would reach for.

**The fix for (c):** truncate on restore, and always to the checkpoint's position:

- Give `restore_from_disk` the checkpoint's `update` and `global_step`, and drop `train_history`/`eval_history` rows with `update > ckpt.update` and `train_episode_history` rows with `step > ckpt.global_step`.
- Take `episodes_seen` and `episodes_dropped` **from the checkpoint, never from the JSON's top level** — the JSON's counters are at the later flush point and cannot be reconstructed from a capped, strided list. Your proposal already stores both, so the ingredients are there; they just have to win.
- Order the calls so the checkpoint is loaded first and the record second. Today they are the other way round (lines 474-478).
- Log the decision — "record resumed: 3,142 update rows and 61,204 episode rows kept, 74 rows dropped past the checkpoint" — the resume rule in `~/.claude/rules/slurm-resumable-generation.md` requires it and nothing prints today.

### (c-bis) A hole you did not ask about, and the one that would actually hurt: the version bump

```python
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError(f"checkpoint {path} has format_version {payload.get('format_version')}, expected 1")
```
`src/checkpointing.py:102-104`

The 30 live runs will write **format 1** checkpoints for as long as they live — their Python already holds the current module, so editing `src/` does not change them. Every one of their first checkpoints, at hour 8, will be format 1. If you bump the writer to 2 and leave that equality check, or change it to `!= 2`, then **the first walltime kill makes every one of those checkpoints unreadable and the trainer raises instead of resuming — all 30 runs silently restart from step 0**, which is far worse than the defect you are fixing.

The reader must accept `{1, 2}`, treat a version-1 payload as "no history carried", and fall back to whatever the on-disk JSON gives (truncated per above). Both `tests/test_checkpointing.py` and `tests/test_run_record.py` need a case for the mixed pairing (v1 checkpoint + present JSON, v1 checkpoint + absent JSON, v2 checkpoint + absent JSON, v2 checkpoint + present-and-ahead JSON).

Related deployment fact: `configs.jsonl` points `argv` at the **live source path** `/p/rlprojects/RND/08_cleanrl_ppo_rnd/src/ppo_rnd_envpool_shuze.py`, not at a snapshot in the run folder. So the edit takes effect on the next re-claim — which is how the fix reaches the sweep, but also means the sweep will have run under two code versions. Record which, in `infra_history.md`, and commit before the change (the commit-before-submit rule).

### (d) Simpler alternatives you have missed

**A better version of your own fix — carry four integers instead of 5.6 MB.** If the JSON is kept in place, the checkpoint does not need the history at all; it needs a *cursor* into it:

```python
"record_cursor": {"train_rows": len(record.train_history),
                  "episode_rows": len(record.train_episode_history),
                  "episodes_seen": record.episodes_seen,
                  "episodes_dropped": record.episodes_dropped,
                  "runtime_seconds": <as of this checkpoint>}
```

On resume, load the JSON and truncate the three lists to those lengths. About 200 bytes instead of 5.6 MB, exact by construction with no reliance on cadence alignment, and it works with the *existing* format-1 checkpoints if you make the cursor optional rather than bumping the version. Cost: it requires the JSON to survive.

**Making the JSON survive is a one-line change to `monitor.py` — but it is shared infrastructure.** `shutil.move` → `shutil.copy2` in `requeue_orphans` gives you the evidence copy *and* leaves the working record in place. The honest objection, and it is the strongest argument in your proposal's favour: `monitor.py` lives at `/p/rlprojects/.claude/skills/submit-gpu-sweep/scripts/` and is used by other people's sweeps, and the monitor loop job re-execs it every tick, so an edit lands live for everyone. If you go this way, do it as an opt-in flag (`--archive-mode=copy`, default unchanged) rather than changing the default under other users. Your checkpoint-carried-history fix has the real advantage of being entirely local to this project — say that plainly in whatever you write up.

**Align the checkpoint to a logging update.** Change line 841 to trigger only when the time cadence has elapsed *and* `update % args.log_every_updates == 0`. Then the record's last row and the checkpoint always sit at the same update, truncation is trivially exact, and the "which is ahead" question disappears. Delays a checkpoint by at most 25 updates (~2 minutes). One line, no format change. Worth doing whichever fix you pick.

**The highest-value change is none of the above: handle SIGTERM.** `KillWait` on this cluster is **128 seconds** (`scontrol show config`), and a 50 MB `torch.save` takes well under a second. The trainer installs no signal handler, so today a walltime kill or a `scancel` discards everything since the last checkpoint. A handler that calls `save_checkpoint` + `record.flush` and exits would reduce the loss on any *clean* kill — walltime, `scancel`, preemption, which is nearly all of them — from "up to 8 hours" to "one update", and would leave the checkpoint and the record aligned at the kill point, which makes the whole truncation question moot for those cases. Pair it with `#SBATCH --signal=B:TERM@600` for a 10-minute margin. It does not cover node crashes and cgroup kills, so it complements rather than replaces the resume fix.

**Shorten `checkpoint_every_seconds`.** 28,800 s is not justified by anything: the write is 50 MB and sub-second. At 8 hours, the expected redo on an unclean kill is 4 hours of GPU time per run — for a 30-run wave, ~120 GPU-hours per kill event. At 3,600 s the expected redo is 30 minutes, and the write cost is 30 runs × 50 MB/h ≈ 0.4 MB/s. This is the cheapest single improvement available and it needs no code change at all, only the argument in `configs.jsonl` for runs not yet started.

---

## 6. Other things about this run's durability that look wrong

1. **`update` is undefined if a resumed run has nothing left to do — NameError, and the marker goes to `failed/`.** `update` is bound only by `for update in range(start_update, num_updates + 1)` (line 528) and is referenced after the loop at line 860 (`update=update`), as are `v_loss`, `pg_loss`, `entropy_loss`. If a job is killed in the window between the trainer's normal exit and `finalize`, the marker stays in `running/`, the monitor requeues it, the record is archived, the run is re-claimed, `load_checkpoint` returns `update = num_updates` → `start_update = num_updates + 1` → the loop body never executes → `NameError` on line 860. Net effect: a **finished** run ends up marked failed with its record archived. Guard it: if `start_update > num_updates`, re-flush with `completed=True` and exit 0.
2. **`scan_orphans` does not look at the completion flag.** A run whose record says `"completed": true` is requeued and its record archived just like a partial one. `requeue_orphans` should read the flag it already has the path for (`worker_manager.completed_marker_ok` does exactly this check at lines 338-356) and move such a marker to `done/` instead.
3. **This sweep will hit the 4-day wall.** Observed throughput is ~3,450 steps/s (`charts/steps_per_second` in run 0's last eval row), so 2e9 steps is **6.7 days of pure compute**. The `gpu`-partition scripts (`cheetah02`, `cheetah03`, `cheetah08-09`, `jaguar03`, `lotus`, …) all carry `--time=4-00:00:00`; only the `gnolim` ones (`ai05_ai10`, `ai07-08`) have 20 days. So most of the 30 runs are guaranteed at least one walltime kill, and the runs on 2080 Ti / A4000 class GPUs will need two or three. This defect is not hypothetical for this sweep; it is scheduled.
4. **The record is rewritten in full every ~2 minutes and it is large.** `record.flush()` (`run_record.py:85-97`) serializes the whole document every `log_every_updates`. The 50,000-episode cap is reached at ~21M steps — about **1.7 hours in** — after which the file is ≥ 6 MB for 99% of the run and reaches ~16 MB. That is 4,883 full rewrites per run, averaging ~10 MB: roughly **49 GB written per run, ~1.5 TB across the sweep**, at a sustained ~2.5 MB/s to shared storage with 30 runs live. It works, but an append-only sidecar for `train_episode_history` would remove essentially all of it and would make truncate-on-resume a matter of counting lines.
5. **No fencing on the checkpoint or the record.** Nothing in the payload identifies the writing attempt. Adding an `attempt` counter and the writing `slurm_job_id`, and refusing to overwrite a checkpoint whose attempt is higher than your own, would turn the double-writer scenario in §4 from silent corruption into a loud failure.
6. **No `code/` snapshot in the run folder.** The project convention (`.claude/CLAUDE.md`, run-folder section) calls for `train_runs/<run>/code/`, and `configs.jsonl` instead points at the live `src/`. Any edit to `src/` — including the fix under discussion — changes what already-queued runs execute. Copy the three source files into the run folder now, before editing, so the two code versions used by this sweep are both on record.
7. **`output_dir` is `data/local`, not `data/<sweep_id>/local`,** which departs from `.claude/rules/run-id-and-logging.md` and means a second sweep in this folder would collide with these records and, because `--resume` defaults to true, silently continue them.
8. **The archived attempts are the only copy of the lost history and they are not in git** (`*.json` is gitignored project-wide). `killed_attempts_<YYYY-MM-DD-HH-MM>` is minute-granular, so two requeues of the same run within one minute would overwrite the earlier archive via `shutil.move`. Unlikely at a 20-minute monitor cadence, but the directory name is the only thing preventing it.

## Suggested minimal change set, in the order I would apply it

1. Trainer: `NameError` guard for `start_update > num_updates`, plus a SIGTERM handler that checkpoints and flushes. *(No format change, protects the currently-running jobs.)*
2. `checkpointing.load_checkpoint`: accept `format_version in (1, 2)` **before** anything else ships.
3. Checkpoint payload: carry `episodes_seen`, `episodes_dropped`, `runtime_seconds`, and the three list lengths; carry the lists themselves too if you want the JSON to be dispensable — it costs 11% and buys independence from the shared monitor.
4. `restore_from_disk(max_update, max_step, counters_from_checkpoint)`: truncate, prefer the checkpoint's counters, print what it kept and dropped.
5. Trigger the checkpoint only on a logging update.
6. Drop `checkpoint_every_seconds` to 3600 for every run not yet started.
7. Snapshot `src/` into the run folder and note the code change in `infra_history.md`.