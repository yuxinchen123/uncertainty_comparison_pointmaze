All data gathered. Here is the audit.

---

# Configuration audit — CleanRL PPO+RND 30-seed sweep

Sweep: `/p/rlprojects/RND/08_cleanrl_ppo_rnd/train_runs/2026-08-05-17-45_cleanrl_train_run_1_ppo-rnd_montezuma-revenge-v5__envpool-autoreset-fixed__128env-128step-2e9step__int-coef1-ext-coef2-updateproportion0.25__seed1-30__checkpoint-8h-resumable__gpu-gnolim`
Audit taken 2026-08-05 18:29–18:34 EDT. Read-only; nothing submitted, cancelled or modified.

**A repair was in flight while I audited.** Job `6533906` (cheetah02) was cancelled at 18:28:10 and replaced by `6533915` at 18:29:10 with 64 cpus. That fixes the known cheetah02 defect. Everything below is what is *still* wrong.

## 1. CPU sizing per job

Every script bakes `--cpus_per_run 8`. Actual cpus per run = cpus given ÷ (W × G).

| job id | script file | W baked in | cpus_per_run baked in | GPUs given | cpus given | slots = W×G | **actual cpus/run** | flag |
|---|---|---|---|---|---|---|---|---|
| 6533903 | `lotus__ppo-rnd-atari.slurm` (canary) | 1 | 8 | 5 | 40 | 5 | 8.0 | ok |
| 6533905 | `lotus__ppo-rnd-atari.slurm` | 1 | 8 | 8 | 64 | 8 | 8.0 | ok |
| ~~6533906~~ | `cheetah02__ppo-rnd-atari.slurm` | 2 | 8 | 4 | 32 | 8 | **4.0** | known, cancelled 18:28 |
| 6533915 | `cheetah02__ppo-rnd-atari.slurm` | 2 | 8 | 4 | 64 | 8 | 8.0 | ok (the repair) |
| 6533907 | `cheetah08-09__ppo-rnd-atari.slurm` | 1 | 8 | 4 | 32 | 4 | 8.0 | ok |
| 6533908 | `cheetah08-09__ppo-rnd-atari.slurm` | 1 | 8 | 4 | 32 | 4 | 8.0 | ok |
| 6533909 | `cheetah03__ppo-rnd-atari.slurm` | 1 | 8 | 2 | 16 | 2 | 8.0 | ok |
| **6533910** | `jaguar03__ppo-rnd-atari.slurm` | **3** | 8 | **8** | **64** | **24** | **2.67** | **same defect, three times worse** |
| 6533911 | `ai07-08__ppo-rnd-atari.slurm` | 1 | 8 | 2 | 16 | 2 | 8.0 | ok (but see §3) |
| 6533912 | `ai07-08__ppo-rnd-atari.slurm` | 1 | 8 | 2 | 16 | 2 | 8.0 | ok (but see §3) |
| 6533913 | `monitor_loop.slurm` | — | — | 0 | 2 | — | n/a | monitoring only |

**Finding 1.1 — job 6533910 on jaguar03 has the identical defect, worse.** Its script is W=3 and it was given all 8 GPUs, so its manager provisioned 24 worker slots against 64 cpus: 2.67 cpus per run, not 8. It is not currently hurting anything only by luck — the five jobs submitted 30 seconds earlier claimed 26 of the 30 runs, so jaguar03 got only 4. Its remaining 20 slots called `claim_one()`, found an empty queue, and exited permanently (`worker_manager.py`, `worker_slot`: "exits when work runs out"). The 4 survivors have 16 cpus each and are the fastest in the sweep (4,350–4,564 steps/s).

**Finding 1.2 — the id file itself records the wrong W, which is why this stayed invisible.** `slurm/submitted_jobids.txt` uses the schema `<jobid> <node_class> <node> <G> <W> <c_used> <group> <ts>` (`monitor.py` line 274). Line 3 records cheetah02 as W=1 and line 7 records jaguar03 as W=1, when the scripts are W=2 and W=3. `monitor.py` recomputes asked-memory and asked-cpus from that file (`asked_mb_from_jobids`, line 1007), so the infra_history tables print "cpu busy/asked 17.2/**64**" for jaguar03 and never notice 64 should have been 192. The monitor is structurally blind to the bug because it reads the same wrong number the submitter wrote. The new line for 6533915 correctly records W=2; **6533910 is now the only remaining wrong line.**

**Finding 1.3 — nothing downstream can detect the mismatch.** `worker_manager.py` line 421 exports `GPU_SWEEP_CPUS_PER_RUN` from the *declared* `--cpus_per_run`, and `resolve_env_threads()` (`src/ppo_rnd_envpool_shuze.py` line 367) trusts it. Every record on the starved cheetah02 job reads `env_threads: 8` (checked `data/killed_attempts_2026-08-05-18-28/3_of_30.json`) while only 4 cpus existed — 8 runs × 8 envpool threads inside a 32-cpu cgroup. The manager knows both W and the token count and could compute the true share; it does not.

## 2. Memory sizing vs `slurm/submission_script/PACKING.md`

PACKING.md's `mem/job` = W × G × per-slot memory, where per-slot is 7,500 MB (the raw 6,000 plus the 25% room) except on `W=1-branch` rows, which use the raw 6,000. Every job was submitted with a flat `mem = G × 6000` — W dropped **and** the 25% room dropped.

| job id | node | W×G slots | PACKING mem/job (MB) | mem given (MB) | mem per slot given | shortfall | measured peak RSS/run | flag |
|---|---|---|---|---|---|---|---|---|
| 6533905 | lotus | 8 | 60,000 | 48,000 | 6,000 | −12,000 (20%) | 2,050 | under PACKING, safe |
| ~~6533906~~ | cheetah02 | 8 | 60,000 | 24,000 | 3,000 | −36,000 (60%) | 2,050 | ran at 68% of its cap |
| 6533915 | cheetah02 | 8 | 60,000 | 48,000 | 6,000 | −12,000 (20%) | 2,050 | under PACKING, safe |
| 6533907 | cheetah08 | 4 | 30,000 | 24,000 | 6,000 | −6,000 (20%) | 2,050 | under PACKING, safe |
| 6533908 | cheetah09 | 4 | 30,000 | 24,000 | 6,000 | −6,000 (20%) | 2,050 | under PACKING, safe |
| 6533909 | cheetah03 | 2 | 15,000 | 12,000 | 6,000 | −3,000 (20%) | 2,050 | under PACKING, safe |
| **6533910** | **jaguar03** | **24** | **180,000** | **48,000** | **2,000** | **−132,000 (73%)** | **2,050** | **would have been OOM-killed** |
| 6533911/12 | ai07 / ai08 | 2 | 12,000 | 12,000 | 6,000 | 0 | — | the only correct pair, by coincidence |

**Finding 2.1 — job 6533910 is the one under-provisioned on memory in a way that kills.** 48,000 MB ÷ 24 provisioned slots = **2,000 MB per slot, below the measured per-run peak of 2,050 MB** (max across all 30 runs, from `slurm/resource_usage/per_run/*.jsonl`; the monitor's own 95th percentile is 2,043.8 MB in `slurm/resource_usage/estimate_vs_actual.md`). Had the queue held 24 runs for it, the cgroup would have hit 24 × 2,050 = 49,200 MB against a 48,000 MB limit and the job would have been OOM-killed a few minutes after the last slot reached steady state — not hours in, but with all 24 runs lost at once. It is safe today only because 4 slots run (8,200 MB of 48,000).

**Finding 2.2 — the collaborator packet ships the same bug.** `for_collaborator/launch_workers_collaborator.sh`:

```bash
local cpus=$(( g * CPUS_PER_RUN )) mem=$(( g * 6000 ))
...
( flock 9; echo "$id $node $g 1 $CPUS_PER_RUN $SWEEP_ID $(date -Is)" >> "$IDFILE" )
```

`g` is the GPU count; W never appears. A collaborator submitting `cheetah02__` (W=2), `jaguar03__` (W=3), `cheetah04__` (W=3), `serval06-09__` (W=3) or `jaguar06__` (W=2) reproduces the defect exactly, and the hardcoded `1` in the id-file line reproduces the blindness. `for_collaborator/README.md` line 98 compounds it — "Do not ... pack more than one run per GPU on the open partitions" — while five of the shipped scripts do exactly that.

**Finding 2.3 — the monitor is actively advising the wrong correction.** `slurm/resource_usage/estimate_vs_actual.md` says host memory is "0.34 of estimate — **room to raise the packing count W (recompute)**". That verdict comes from a 120-second sampler that reads steady-state `VmRSS`. The 6,000 MB estimate came from `sacct MaxRSS` of 4,727 MB and 4,367 MB on the profiling jobs (`for_collaborator/resource_facts.md`). Acting on "raise W" while the submitter's formula still drops W is how a 73%-short job like 6533910 gets built again. In the same file the GPU row says "1.22 — over the estimate, lower W for new submissions now": 6,000 MB observed peak × W=3 on jaguar03's 20,470 MB A4500 leaves 12% headroom.

## 3. Provisioned slots vs slots actually running

| job id | node | GPUs held | W | slots provisioned | runs running now | **idle slots** | GPUs with no run |
|---|---|---|---|---|---|---|---|
| 6533905 | lotus | 8 | 1 | 8 | 8 | 0 | 0 |
| 6533915 | cheetah02 | 4 | 2 | 8 | 8 | 0 | 0 |
| 6533907 | cheetah08 | 4 | 1 | 4 | 4 | 0 | 0 |
| 6533908 | cheetah09 | 4 | 1 | 4 | 4 | 0 | 0 |
| 6533909 | cheetah03 | 2 | 1 | 2 | 2 | 0 | 0 |
| **6533910** | **jaguar03** | **8** | **3** | **24** | **4** | **20** | **5** |
| 6533911 | ai07 | 2 | 1 | 2 | 0 | 2 | job exited at 00:00:00 |
| 6533912 | ai08 | 2 | 1 | 2 | 0 | 2 | job exited at 00:00:00 |
| | | **30** | | **54** | **30** | **24** | |

Sources: `queue/running/*.json` claim blocks (8 lotus, 8 cheetah02, 4+4 cheetah08/09, 4 jaguar03, 2 cheetah03) and `infra_history.md` 18:24, which shows jaguar03 GPU2 and GPU4–7 at 0.0 GB and 0% utilization.

**Finding 3.1 — job 6533910 holds 5 idle GPUs for 4 days.** That is 480 GPU-hours of held-but-unused capacity, 12.5% of the user's whole 40-GPU gpu-partition ceiling, plus 32 unused cpus and 40 GB of unused memory. The idle slots will not refill: `worker_slot` returns permanently when `claim_one` finds an empty queue.

**Finding 3.2 — the two gnolim jobs claimed nothing and this is the sweep's most expensive mistake.** `6533911`/`6533912` were submitted at 17:43:23, 46 seconds after the gpu-partition jobs at 17:42:37. By then all 30 runs were claimed, so both managers printed `manager done on ai07 (job 6533911)` and exited with `Elapsed=00:00:00`. The gnolim partition allows **20 days**; the gpu partition allows **4**. With the gnolim per-user cap of 80 cpus and 20 GPUs, 10 of the 30 runs could have sat on 20-day jobs and never been killed by a walltime at all. Instead zero did, and all 30 runs are on 4-day jobs for a workload that needs 5.1–6.8 days (below). The generic "open partitions first, reserved last" ordering is backwards for a workload longer than the open partition's walltime.

**Finding 3.3 — no job is on the active reservation.** `sl5nw_156` covers jaguar03 and puma01 until 2026-08-19 (`Flags=OVERLAP,IGNORE_JOBS,SPEC_NODES`). Job 6533910 runs on jaguar03 without `--reservation=`. This costs nothing on walltime (the partition `MaxTime` still binds at 4 days) but it means the reservation is not protecting the sweep's jaguar03 slot.

## 4. Run records under `data/local/` — clean

| check | result |
|---|---|
| distinct `run_id` values | 30 (0–29), no duplicates, none missing |
| `seed == run_id + 1` | holds for all 30 in `configs.jsonl` and as `a_seed` in every record |
| `--seed` / `--run_id` inside each `argv` vs the config's own fields | agree for all 30 |
| record config vs `configs.jsonl` (`total_timesteps`, `log_every_updates`, `checkpoint_every_seconds`, `fix_envpool_autoreset`, `opt_cudnn_benchmark`, `opt_amp_fp16`, `env_id`) | no disagreement in any of the 30 |
| records advancing | all 22 live records had mtimes 18:31–18:33 against a 18:33 reading; `global_step` advanced on every run between my 18:29 and 18:33 samples |
| canary leakage into `data/local` | none — the canary's five records are under `canary/data/`, ids disjoint from nothing since they are in a separate tree |

**Finding 4.1 — `data/local/` holds 22 of 30 records right now, not 30.** The 8 cheetah02 runs (ids 3, 4, 8, 9, 10, 12, 16, 19) were moved to `data/killed_attempts_2026-08-05-18-28/` at 18:28 and their replacements have not reached their first flush. Any analysis run in this window silently sees 22 seeds.

**Finding 4.2 — the 18:28 repair cost 5.3 M steps per run and could not have avoided it.** `slurm/logs/run_3.log` shows a second `Start to initialize observation normalization parameter` with no `[resume] continuing at update ...` line: the 8 runs restarted from step 0. No checkpoint existed (45 minutes elapsed against an 8-hour cadence), so a resume was impossible either way. Moving the records aside was the right call — see next.

**Finding 4.3 — record restore and checkpoint restore are independent, and the mismatch is scheduled to hit all 30 runs.** In `src/ppo_rnd_envpool_shuze.py` line 473 onward, `record.restore_from_disk()` runs unconditionally and `load_checkpoint()` runs separately; `resumed` depends only on the checkpoint. Two failure modes follow:

- *Record present, checkpoint absent* (a kill inside the first 8 hours, if the record is left in place): the record reloads the old history while `global_step` restarts at 0, producing one file with two overlapping step sequences and an inflated `episodes_seen`. The operator dodged this at 18:28 by moving the records; nothing in the code enforces it.
- *Checkpoint present, record moved away* — **this is what will happen to every run at the 4-day walltime.** `monitor.py` `requeue_orphans` (line 958) moves `completed_marker_path`, i.e. `data/local/<id>_of_30.json`, into `data/killed_attempts_<ts>/`, and never touches `data/local/<id>_of_30.checkpoint.pt`. So each resumed run restores the correct weights and `global_step` (~1.18e9) but starts with an **empty** history: `episodes_seen` back to 0, `runtime_seconds` back to 0, and the 50,000-entry cap-and-stride restarted, so the second segment keeps its first 50,000 episodes at full resolution while the first segment's sit in a different file. Each run's final history will be split across two files with an ~8-hour hole at the seam (the checkpoint rewind), and nothing documented stitches them.

**Finding 4.4 — the monitor never submits, so the sweep stalls on 2026-08-09.** `slurm/monitor_loop.slurm` calls `monitor.py --requeue` and nothing else; `monitor.py` has no `sbatch`. At the walltime all six gpu jobs die together, all 30 markers are requeued to pending, and no worker job exists to claim them. It will not falsely write `SWEEP_COMPLETE` (pending=30 ≠ 0), so the failure is a silent stall, not a false success. Monitor job 6533913 runs to 2026-08-25, so coverage lasts.

## 5. Checkpoint cadence vs the 4-day walltime

`--checkpoint_every_seconds 28800` (8 h). `last_checkpoint_time` is set at line 525, after the observation-normalizer priming — measured at 198 s on cheetah09 (`charts/obs_norm_init_seconds`). Partition `gpu` `MaxTime=4-00:00:00`; every gpu job carries `TimeLimit=4-00:00:00` and ends 2026-08-09T17:42–18:29. No maintenance reservation exists.

**345,600 s ÷ 28,800 s = 12.0 exactly.** Checkpoints land at 198 + k×28,800 s. The 12th falls at 345,798 s — 198 s *after* the kill. The last surviving checkpoint is the 11th at 316,998 s (88.06 h).

> **Worst-case loss per walltime kill = 345,600 − 316,998 = 28,602 s = 7 h 57 m**, i.e. essentially the full 8-hour interval, and because 4 days is an exact multiple of 8 hours this is not a tail case — it is what happens every time.

| node | measured steps/s (18:33) | steps lost per kill | % of 2e9 | time to 2e9 | walltime kills needed |
|---|---|---|---|---|---|
| cheetah09 (6533908) | 3,426 | 98.0 M | 4.90% | 6.75 d | 1 |
| cheetah08 (6533907) | 3,427 | 98.0 M | 4.90% | 6.75 d | 1 |
| lotus (6533905) | 3,715 | 106.3 M | 5.31% | 6.23 d | 1 |
| cheetah02 (6533915, repaired) | ~3,500 est. | 100.1 M | 5.01% | ~6.6 d | 1 |
| cheetah03 (6533909) | 3,960 | 113.3 M | 5.66% | 5.85 d | 1 |
| jaguar03 (6533910), fastest run 18 | 4,564 | 130.5 M | 6.53% | 5.07 d | 1 |

**Worst case in the sweep: 7 h 57 m of wall time and 130.5 M steps (6.5% of the run), on job 6533910.** Resume overhead itself is small — the 198-second observation-normalizer priming is skipped on a resume (line 495).

Sum across all 30 runs at one kill each: roughly **3.2 billion steps and 238 node-hours discarded**, against a total sweep budget of 60 billion steps. A 1-hour cadence would cut the worst case to ~16 M steps (0.8%) at a cost of 96 checkpoint writes of 50.2 MB per 4-day segment.

**The deeper point:** every run needs 5.1–6.8 days and every job has a 4-day limit, so the 4-day kill is not a risk — it is a certainty for all 30 runs, and the 20-day partition that would have avoided it sat empty (§3.2).

## 6. Multi-day growth

**Record JSON projection.** Measured on `data/local/0_of_30.json`: 20,963 episodes at 9,011,200 steps (430 steps/episode), `eval_history` 694 B/entry, `train_history` 213 B/entry, `train_episode_history` 118 B/entry. `episode_history_cap=50000`, `episode_history_stride=100`; past the cap the list keeps growing at 1/100.

| component | count at 2e9 steps | bytes | size |
|---|---|---|---|
| `train_episode_history` | 50,000 + (4.65 M − 50,000)/100 = 96,000 | 118 | 11.3 MB |
| `eval_history` | 2e9/16,384/25 = 4,883 | 694 | 3.39 MB |
| `train_history` | 4,883 | 213 | 1.04 MB |
| config header | — | ~2 KB | ~0 |
| **total per run** | | | **≈ 15.7 MB** |
| **× 30 runs** | | | **≈ 472 MB** |

Currently 2.4–3.8 MB each. No problem.

**Finding 6.1 — write amplification is the real cost, not size.** `record.flush()` rewrites the *whole* file at every logged update (`ppo_rnd_envpool_shuze.py` line 834), i.e. every 25 updates = 409,600 steps ≈ 110 s at 3,715 steps/s, each write `fsync`'d (`run_record.py` `flush`). Per run: 4,883 rewrites averaging ~7.9 MB = **≈ 39 GB written**; across 30 runs **≈ 1.16 TB of fsync'd writes to the shared NFS mount** over the sweep. Sustained load ≈ 2 MB/s early rising to ~4 MB/s late, with 30 fsyncs every ~2 minutes. `json.dump` of 96,000 entries also costs ~0.3–0.6 s of CPU per flush by the end — under 0.5% of throughput, but it grows linearly and is charged to the training process.

**Finding 6.2 — disk projection is comfortable; the filesystem's headroom is not.**

| item | at completion |
|---|---|
| `data/local/*.json` (30 records) | 472 MB |
| `data/local/*.checkpoint.pt` (30 × 50.2 MB, replaced in place) | 1.51 GB (+1.51 GB transient during `.tmp` writes) |
| `data/killed_attempts_2026-08-05-18-28/` (existing) | 15 MB |
| `data/killed_attempts_2026-08-09-*/` (the walltime wave, records at ~59%) | ~350 MB |
| `slurm/logs/run_*.log` (4,883 lines × ~105 B per segment, append mode) | ~20 MB |
| `slurm/resource_usage/` (one 200 B line per run per 120 s) | ~60 MB |
| `canary/` (static, includes 5 × 50.2 MB checkpoints) | 244 MB |
| **total** | **≈ 2.7 GB** |

`/p/rlprojects` is at **749 G used of 1.0 T, 276 G available (74%)**. This sweep is not the threat, but every additional `killed_attempts_<ts>/` wave adds another full copy of all 30 records, and by the end of the run that is ~470 MB per wave.

**Finding 6.3 — per-run logs are append-mode across restarts.** `worker_manager.py` opens `run_{id}.log` with `"a"`, so `slurm/logs/run_3.log` already holds two segments back to back. Any parser reading the last `update=` line gets a stale value from the killed segment — my first pass showed run 3 at step 5,324,800 when its live process was at 0. Growth itself is trivial (~600 KB per run).

## Summary — what to fix, in order

1. **Job 6533910 (jaguar03)** — the same cpu bug as cheetah02 and a 73% memory shortfall: 24 provisioned slots on 64 cpus and 48,000 MB. Harmless today (4 slots), fatal on any resubmission that meets a full queue. Fix the id-file line's W=1 → W=3 at the same time so `monitor.py` stops printing a false "asked" figure.
2. **Every run is on a 4-day partition for a 5.1–6.8-day workload** while the 20-day gnolim partition sits empty because jobs `6533911`/`6533912` arrived 46 seconds late. Cost as configured: one forced kill per run, ~8 h and 98–130 M steps each.
3. **`for_collaborator/launch_workers_collaborator.sh`** computes `cpus = g * CPUS_PER_RUN` and `mem = g * 6000` and writes a hardcoded `1` for W — the bug is shipped to collaborators, and `README.md` line 98 tells them packing never exceeds one run per GPU when five scripts do.
4. **`monitor.py` `requeue_orphans` moves the record but leaves the checkpoint**, so at the 2026-08-09 walltime all 30 runs resume with correct weights and empty history, splitting each run's history across two files with an 8-hour hole.
5. **No auto-resubmission** — the monitor requeues to pending and stops; the sweep stalls silently on 2026-08-09 until a human submits.
6. **`estimate_vs_actual.md` advises raising W** from a 120-second sampler that misses the initialization peak the 6,000 MB estimate was sized for.
7. **8-hour checkpoints divide the 4-day walltime exactly**, so the loss is pinned at the maximum every time rather than averaging half an interval.