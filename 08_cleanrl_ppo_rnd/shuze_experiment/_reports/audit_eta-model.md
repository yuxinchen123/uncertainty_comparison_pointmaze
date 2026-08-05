## Snapshot

Measured at **2026-08-05 18:44:40 EDT**, ~60 min into the sweep. Two readings 809 s apart (18:31:11 and 18:44:40) plus the full `eval_history` of all 30 records.

Two things about the current state, because they change the numbers:

1. **Every run is on the `gpu` partition at a 4-day walltime.** The folder slug says `gpu-gnolim`, but the two `gnolim` spare jobs (6533911/6533912 on ai07/ai08) exited at 17:43:24 with zero elapsed — all 30 runs had already been claimed by the six `gpu` jobs that started 40 s earlier. **The 20-day `gnolim` walltime protects nothing here.**
2. **cheetah02 was cancelled and resubmitted at 18:29:13.** Its job had 32 threads for 8 runs (4 cpus each, ~2,311 steps/s). The requeue archived the 8 partial records and, since no checkpoint exists before the 8-hour mark, those 8 runs restarted **from step 0**. They now have 8 cpus each and are 46 min behind.

---

## 1. Cumulative vs instantaneous rate

Three rates per run. $r_{\text{inst}} = \dfrac{16384}{\frac{1}{5}\sum_{j=n-4}^{n}\texttt{iteration-seconds}_j}$, against the record's own in-segment cumulative counter and against a naive steps-over-wall-clock rate.

| comparison | median | range |
|---|---|---|
| instantaneous vs `charts/steps_per_second` | **+0.4%** | −2.9% .. +8.1% |
| instantaneous vs `step / runtime_seconds` | **+7.9%** | +1.1% .. +28.5% |
| ...mature runs only (≥20 eval rows, n=22) | +1.2% | — |
| ...young runs (<20 rows, n=8, all cheetah02) | −1.6% | — |

The two "cumulative" measures diverge because `charts/steps_per_second` is `(global_step − steps_at_start) / (time since training start)` — it **excludes** the observation-normalizer warm-up. `step / runtime_seconds` includes it, and that warm-up is 89–197 s of a 60-minute run, which is why the naive rate is up to 28.5% low.

**The instantaneous rate is the better predictor, for three reasons:**

1. **What it excludes is one-time.** Process start-up (55–80 s) plus the warm-up (89–197 s) is ~300 s against a ~160-hour run — 0.05%. Baking a fixed cost into a rate used to project the remaining 99.4% is just wrong.
2. **The early iterations were genuinely slower and are not representative** — cuDNN autotuning, cold cache, and all runs on a node warming up at once. Drift from the first third to the last third of each run's history: +2.9% (lotus) to +10.8% (jaguar03).
3. **The drift has now stopped**, which is what licenses using a short window. Last-5 mean vs last-10 mean is −1.3% to +0.4% across all six nodes. The window is sampling a stationary rate.

**Cross-check:** the step delta between the two snapshots (809 s apart) gives 3,546 / 4,053 / 4,560 steps/s per node. It is quantized to the 409,600-step eval cadence, so it is coarse, but it brackets the last-5 means. The two methods agree.

**The honest caveat, and the honest headline.** `charts/iteration_seconds` is a *single* iteration sample logged every 25 updates, so the last-5 mean is noisy: per-run coefficient of variation is 4.2–5.8% on five nodes but **12.7% on jaguar03**, giving a standard error of 1.4–2.2% normally and **3.8% on jaguar03**. And the headline: **at 60 minutes this choice is worth only ~1%** against the in-segment cumulative counter (it was worth 5–16% at 20 minutes). The uncertainty in this forecast does not live in the rate estimator. It lives in §4 and §5.

---

## 2. By node

`cpus per run` = job `AllocCPUS` ÷ runs on that node. These are **hardware threads** (`ThreadsPerCore=2`), so 8 threads = 4 physical cores. Sorted by projected days, worst first.

| node | GPU | runs | GPUs held | runs/GPU | cpus/run | inst. steps/s per run | steps done | steps remaining | pure days | **wall days** |
|---|---|---|---|---|---|---|---|---|---|---|
| cheetah02 | RTX 4000 Ada | 8 | 4 | 2 | 8 | 3,062 | 2,048,000 | 1,997,952,000 | 7.55 | **7.89** |
| cheetah08 | RTX A4000 | 4 | 4 | 1 | 8 | 3,476 | 11,878,400 | 1,988,121,600 | 6.62 | **6.96** |
| cheetah09 | RTX A4000 | 4 | 4 | 1 | 8 | 3,484 | 11,878,400 | 1,988,121,600 | 6.61 | **6.94** |
| lotus | Quadro RTX 6000 | 8 | 8 | 1 | 8 | 3,771 | 12,953,600 | 1,987,046,400 | 6.10 | **6.43** |
| cheetah03 | RTX 2080 Ti | 2 | 2 | 1 | 8 | 4,070 | 13,926,400 | 1,986,073,600 | 5.65 | **5.98** |
| jaguar03 | RTX A4500 | 4 | 8 | 0.5 | 16 | 4,286 | 14,848,000 | 1,985,152,000 | 5.36 | **5.70** |

"pure days" = remaining ÷ instantaneous rate. "wall days" adds the walltime-kill penalty from §4.

**jaguar03 holds 8 GPUs and runs 4 seeds** — 4 A4500 slots idle, on the fastest node, under a reservation. It started 26 s after the others and only 4 runs were left unclaimed. Rebalancing cheetah02's 8 packed runs onto them is the one lever that moves the long pole, and it is cheap right now precisely because cheetah02 has no checkpoint yet.

---

## 3. Campaign completion = max over nodes

**The campaign finishes when the last of 30 runs finishes, so the maximum is the only correct aggregate.**

- **Long pole: cheetah02** — mean-rate finish 2026-08-13 16:03; its **slowest run** (run 19, 2,957 steps/s) finishes **2026-08-14 06:30**.
- Mean over nodes would be **6.98 days → 2026-08-12**, understating by **1.5 days**.

cheetah02 is the long pole for three compounding reasons: it is the only node packing **2 runs per GPU**, it carries **8 of the 30 runs** (so its worst-of-8 is worse than a worst-of-2), and it **restarted from zero at 18:29**.

---

## 4. Effect of the 4-day walltime

**The critical arithmetic: 4 days = 96 h is an exact multiple of the 8-hour checkpoint interval.** Checkpoints are counted from *training start* (`last_checkpoint_time` is set after the warm-up), while the walltime runs from *job start*. So the 12th checkpoint lands 3–6 minutes **after** the kill and never happens. Every kill discards almost a full 8 hours.

| node | first kill | checkpoints written | last checkpoint | **progress lost** |
|---|---|---|---|---|
| lotus | 08-09 17:42:41 | 11 | 08-09 09:46:50 | **7.93 h** |
| cheetah08 | 08-09 17:42:41 | 11 | 08-09 09:47:07 | **7.93 h** |
| cheetah09 | 08-09 17:42:41 | 11 | 08-09 09:47:09 | **7.93 h** |
| cheetah03 | 08-09 17:42:41 | 11 | 08-09 09:46:39 | **7.93 h** |
| jaguar03 | 08-09 17:43:07 | 11 | 08-09 09:45:30 | **7.96 h** |
| cheetah02 | 08-09 18:29:13 | 11 | 08-09 10:33:38 | **7.93 h** |

This is the worst possible alignment, not bad luck averaging to 4 h. It is also cheap to fix: at 50.2 MB a checkpoint, an hourly interval would cap the loss at 1 h for ~0.05% throughput cost.

**Kills per run:** 27 of 30 runs are killed **once**; 3 runs (cheetah02 runs 19, 4, 8) are killed **twice**.

**Cost per resume — smaller than assumed.** The 79–154 s figure in the profiling report is the *observation-normalizer warm-up*, and a resume **does not pay it**: `src/ppo_rnd_envpool_shuze.py:495` guards it with `if not resumed:` and restores `obs_rms` from the checkpoint. A resume pays only process/CUDA/envpool init plus a 50 MB checkpoint load — measured **55–80 s** here. Against 7.93 h of discarded progress this is noise.

**What it adds:** +0.33 days per kill → **+0.33 d for 27 runs, +0.66 d for 3 runs**. Total campaign penalty ≈ 0.6 days.

**But there is no automatic resubmission, and this is the real exposure.** `monitor.py --requeue` moves an orphaned run back to `pending` and archives its partial record; it never calls `sbatch`. `worker_manager.py` exits when its slot finds no pending work. So on **2026-08-09 ~17:43 all 30 runs stop and sit in `pending`** until a human or the standing 20-minute agent loop submits fresh worker jobs. That gap is unbounded and is not in any number above.

---

## 5. What could make this estimate wrong

1. **Nobody resubmits on 2026-08-09.** The largest risk by far, and the only unbounded one. Every run stops that evening; each idle day is a day added.
2. **cheetah02 sits on a segment boundary.** A run needs a 3rd segment above 183.9 h of pure compute. cheetah02's mean is 181.3 h — **1.5% of margin**, and run 16 has **1.6 hours**. A ~1% slowdown flips runs across the boundary and each crossing adds ~8 h discontinuously. Three of its runs are already on the far side. This is why the cheetah02 row should be read as "7.9 or 8.5 days", not 7.9.
3. **Contention from other users.** Right now we hold *every* GPU on all six nodes, so the measured rates are contention-free — which means the estimate can only get worse, not better. Free CPU headroom remains on lotus (64/80 alloc), cheetah03 (16/72), cheetah08/09 (32/40), cheetah02 (64/72), so another user's CPU job can land beside ours and slow the envpool rollout, which is the CPU-bound half of the iteration.
4. **The rate estimator's own noise.** ±1.4–2.2% per node, ±3.8% on jaguar03. jaguar03 also climbed +10.8% over the first hour; the climb has stopped, but a node that moved once can move again.
5. **Node failure.** A dead node costs up to 8 h per run plus resubmission. jaguar03 in particular ran at a quarter speed for three days in late July before anyone noticed — the same silent-degradation failure would corrupt the rates this estimate is built on, not just delay it.
6. **The jaguar03 reservation ends 2026-08-19.** Its runs project to finish **2026-08-10 to 08-12**, so there is ~8 days of margin and the reservation is **not** currently binding. It becomes binding only if jaguar03 runs slip past 08-19 — which needs a roughly 60% slowdown or a multi-day resubmission gap. Worth a note, not a worry.
7. **Structural, not on the critical path:** after a kill the requeue archives the run's JSON while keeping the `.pt`, so training resumes correctly from `global_step` but each run's reward history restarts and the earlier segment lands in `data/killed_attempts_*/`. Analysis will need stitching. It does not affect timing.

Note that risks 1, 3, 5 and 6 are all one-directional: none of them can make the run finish earlier.

---

## 6. When will this finish

**Most likely 2026-08-13 to 2026-08-15, with the last seed on cheetah02 landing 2026-08-14 (±1 day) — provided someone resubmits all 30 runs promptly after the walltime kill on 2026-08-09 ~17:43; if that resubmission slips, the finish date slips day for day.**

Sensitivity, holding the resubmission assumption: 0% slower → 08-14; 5% → 08-14; 10% → 08-15; 20% → 08-16.

Files: analysis scripts at `/tmp/claude-2618919/-p-rlprojects-RND/12f4bbb4-d71e-4c5d-8c09-aae9a5c45be7/scratchpad/{final.py,analyze.py,snapA.json,snapB.json}`; records read from `/p/rlprojects/RND/08_cleanrl_ppo_rnd/train_runs/2026-08-05-17-45_cleanrl_train_run_1_ppo-rnd_montezuma-revenge-v5__envpool-autoreset-fixed__128env-128step-2e9step__int-coef1-ext-coef2-updateproportion0.25__seed1-30__checkpoint-8h-resumable__gpu-gnolim/data/local/`. Read-only throughout; nothing in the run folder was modified.