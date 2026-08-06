Probe files cleaned up. Here is the analysis.

---

# Logging cadence for the 150-run, 2e9-step PPO+RND campaign

**Recommendation: `log_every_updates = 200`, with every field aggregated over the full 200-update interval instead of sampled at its end.** The cadence change is the small part. The aggregation change is the win.

Run geometry: `batch_size = 128 envs x 128 steps = 16,384` steps per update, so 2e9 steps is **122,070 updates**; each update runs `update_epochs x num_minibatches = 4 x 4 = 16` optimizer steps. At 3,500 steps/s a run is **6.61 days** (9.65 days at 2,400, 4.72 days at 4,900).

---

## 1. What each logged field actually averages over

Read from the record-writing block at `ppo_rnd_envpool_shuze.py:825-864`. Three fields are real averages, eight are single samples or running totals, and one important field is not logged at all.

### Real averages

**`train/mean_extrinsic_reward` — a mean over at most 20 episodes.**
```python
avg_returns = deque(maxlen=20)                                   # line 465
...
"train/mean_extrinsic_reward": float(np.mean(avg_returns)) if avg_returns else None,   # line 832
```
The deque is fed only on a true game-over (`if d and info["lives"][idx] == 0`, line 619). At any cadence this row reports the mean of the last 20 game-overs. At `C=200`, **7,671 episodes finish inside the interval and 20 of them are reported — 0.26%.** The window is also wall-clock-varying: 20 episodes span about 8,500 env steps, so the row describes the last ~0.5 update out of 200.

**`train/mean_intrinsic_reward` — a mean over the current update's rollout only.**
```python
raw_curiosity_mean = float(curiosity_rewards.mean().item())      # line 637, before normalisation
curiosity_rewards /= np.sqrt(reward_rms.var)                     # line 638
```
That is a clean mean over 128 x 128 = 16,384 bonus values, but only of **update N**. Updates N-199 through N-1 contribute nothing. `train/mean_normalized_intrinsic_reward` (line 834) is the same 16,384 values after the division.

### Single noisy samples — this is the real problem

**All six `losses/*` come from one minibatch.** `v_loss`, `pg_loss`, `entropy_loss`, `forward_loss`, `approx_kl`, `old_approx_kl` are Python names bound inside the innermost loop at `748-802`; the values read at lines 857-862 are whatever the **last minibatch of the last epoch** left behind. With `opt_fixed_minibatch_shape` the minibatch is 3,840 rows of 16,384. So each logged loss is:

- 1 of 16 optimizer steps in that update,
- 1 of **3,200** optimizer steps in a 200-update interval — **0.03% coverage**,
- computed on 23% of one rollout's rows.

`approx_kl` is the worst case: it is exactly the quantity you would use to detect a policy update going wrong, and it is sampled from one gradient step in 3,200.

**Timing and drop fields are one-update snapshots.** `charts/iteration_seconds`, `charts/rollout_seconds`, `charts/update_seconds` and `charts/burned_rows_dropped` all describe the single update that happened to trigger the log.

**The per-episode intrinsic reward is a one-step sample.** In the episode row, `"train/intrinsic_reward": float(step_curiosity[idx])` (line 624) is the RND bonus at the single step the episode ended, for that one env — not an episode mean.

### Running totals that are not interval quantities

- `charts/steps_per_second` = `(global_step - steps_at_start) / elapsed` (line 827): the **cumulative** mean since this process segment started, reset by every resume. Late in a run it cannot show a node slowing down — a 50% slowdown for the last day moves a 6-day cumulative average by a few per cent.
- `charts/gpu_memory_peak_mb` = `torch.cuda.max_memory_allocated()`, never reset. It is a monotone staircase that stops carrying information after the first few rows.

### Not logged at all

`nn.utils.clip_grad_norm_(combined_parameters, args.max_grad_norm)` at lines 811 and 817 **returns the pre-clip total norm and the return value is discarded.** With `max_grad_norm = 0.5` active, nothing in the record says whether clipping fired on 1% or 90% of optimizer steps. For a 5-arm exploration comparison that is the single most useful missing diagnostic.

### One thing the cadence does not control

`record.add_episode` is called inside the rollout loop (line 621), so **the per-episode history is unaffected by `log_every_updates`.** Episode returns keep full per-episode resolution at any cadence. This is the fact that makes a sparse update-log safe.

---

## 2. Cadence table

`C` = `log_every_updates`. Wall-clock at 3,500 steps/s.

| C | env steps per row | minutes per row | rows per run | updates per row | optimizer steps per row | episodes per row |
|---|---|---|---|---|---|---|
| 25 (current) | 409,600 | 1.95 | 4,883 | 25 | 400 | 959 |
| 50 | 819,200 | 3.90 | 2,442 | 50 | 800 | 1,918 |
| 100 | 1,638,400 | 7.80 | 1,221 | 100 | 1,600 | 3,835 |
| **200** | **3,276,800** | **15.60** | **611** | **200** | **3,200** | **7,671** |
| 400 | 6,553,600 | 31.21 | 306 | 400 | 6,400 | 15,342 |
| 800 | 13,107,200 | 62.42 | 153 | 800 | 12,800 | 30,684 |

Minutes per row on the slowest and fastest cards: C=200 gives 22.8 min at 2,400 steps/s and 11.1 min at 4,900 steps/s.

The "optimizer steps per row" and "episodes per row" columns are the headline: they are what an interval-averaged row *could* average over. Today every one of those rows reports 1 optimizer step and 20 episodes regardless of the column value.

---

## 3. Record size

Using the measured row sizes (694 B eval, 213 B train, 118 B episode; my independent probe on a synthetic record gave 657 / 225 / 131 B, agreeing within 6%). Config block measures 1,479 B.

### Episode history — the dominant term, and cadence-independent

At 2,341 episodes per million steps, a 2e9-step run finishes **4,682,000 episodes**. The cap-and-stride rule in `RunRecord.add_episode` (cap 50,000, stride 100) keeps:

- the first **50,000** episodes, which are exhausted at **21.4 M steps — 1.07% of the run, about 1.7 hours in**;
- then every 100th of the remaining 4,632,000, which is **46,320** rows.

**Total kept: 96,320 rows = 11.37 MB. Dropped: 4,585,680 episodes (97.9%).**

Past the cap, one surviving episode row per 42,717 env steps — **2.6 policy updates**. That is finer than every cadence in the table, which is why the reward curve's resolution is set by the episode history and not by the update log.

### Per-run and campaign size

| C | update rows | update history | full record | 150 runs (json) | checkpoints | **total** |
|---|---|---|---|---|---|---|
| 25 | 4,883 | 4.43 MB | 15.80 MB | 2.37 GB | 9.87 GB | 12.24 GB |
| 50 | 2,442 | 2.21 MB | 13.58 MB | 2.04 GB | 9.54 GB | 11.57 GB |
| 100 | 1,221 | 1.11 MB | 12.47 MB | 1.87 GB | 9.37 GB | 11.24 GB |
| **200** | **611** | **0.55 MB** | **11.92 MB** | **1.79 GB** | **9.29 GB** | **11.08 GB** |
| 400 | 306 | 0.28 MB | 11.64 MB | 1.75 GB | 9.25 GB | 10.99 GB |
| 800 | 153 | 0.14 MB | 11.51 MB | 1.73 GB | 9.23 GB | 10.95 GB |

The checkpoint column is `(50 MB tensors + the live record history) x 150` — `save_checkpoint` embeds `record_history` in the format-2 payload, so a late-run checkpoint is about 62 MB, not 50.

**Disk is not the constraint.** The whole campaign is 11 GB against 276 GB free on `/p/rlprojects` (NFS, 1.0 TB, 74% used). Going from C=25 to C=800 saves 1.3 GB out of 12.2 — 11%. **Do not choose the cadence on disk grounds.**

---

## 4. Recommendation: C = 200, averaged over the whole interval

**At `log_every_updates = 200`, one row covers 3,276,800 env steps, 200 policy updates, 3,200 optimizer steps, and 7,671 finished episodes.**

The reasoning:

1. **A sparse row that averages beats a dense row that samples.** Today a C=25 row carries one minibatch's `approx_kl`. A C=200 row that accumulates carries the mean over 3,200 minibatches. That is 3,200x the sample count per row, at one eighth the row count — and the standard error of each plotted point falls by roughly `sqrt(3200) ≈ 57` relative to a single-minibatch sample. Eight times fewer points, each about fifty times less noisy: strictly better data for less disk and less wall clock.
2. **Nothing is lost on the reward curve.** Episode rows are written per episode inside the rollout loop, so extrinsic return keeps one row per 42,717 steps whatever `C` is. The update log is a diagnostics channel, not the learning curve.
3. **611 points is more than enough resolution** for a 2e9-step x-axis; at 150 runs that is 91,650 rows to plot, already at the limit of what is comfortable to load and aggregate.
4. **Monitoring stays responsive.** One row and one printed line every 15.6 min (22.8 min worst card). A stalled run is visible within one interval.
5. **The checkpoint stays aligned.** The checkpoint only fires on a logging update (`due and update % args.log_every_updates == 0`, line 876), so `C` sets the granularity of the 8-hour checkpoint. At C=200 the checkpoint lands at most 15.6 min late (8.26 h worst-case work lost to a hard crash). At C=800 it lands up to 62 min late — 9.04 h at risk, a 13% inflation. **This is the ceiling argument against going past 400.**

### The averaging window: the whole interval, not the last update

Every field accumulates across all 200 updates and resets at the row.

**Critical implementation detail: accumulate on the GPU.** Calling `.item()` per minibatch to build a running sum would add 16 device-host synchronisations per update at *every* update — which would genuinely slow the run and defeat the purpose. Instead keep running sums as device tensors (`pg_loss_sum += pg_loss.detach()`, `gn_max = torch.maximum(gn_max, gn)`) and call `.item()` once per logged row. That is a handful of tiny device ops added to a 4.68 s iteration — immeasurable.

Replace the 20-episode deque as the reported statistic: accumulate the returns of **all** episodes finishing in the interval (they are already on the host, as numpy from `info["r"]`, so this is free) and report mean, count, max, and p10/p50/p90. Keep the deque for the printed line if you like — it is CleanRL's — but do not let it be the recorded number.

---

## 5. What should be a max, not a mean

A mean over 3,200 optimizer steps is the right summary for a quantity whose typical value matters. It is the wrong summary for a quantity whose *worst* value is what breaks the run. Log both where the tail carries the meaning.

1. **Gradient norm — add it, and log mean, max, and clip rate.** Currently discarded. Capture the return of `clip_grad_norm_`. The mean tells you the scale; the **max** tells you whether a single step tried to move the weights a hundred times further than usual; the **clip rate** (fraction of the 3,200 steps exceeding `max_grad_norm = 0.5`) tells you whether the clip is a safety net or the actual optimizer. With RND's non-episodic intrinsic return there is no terminal mask to bound the advantage, so this is a live risk, not a hypothetical.
2. **Intrinsic reward — log the max alongside the mean.** The RND bonus is spiky *by design*: a genuinely novel state is a large single-step bonus. Averaging 3.28M rollout values per row buries exactly the event the whole method exists to produce. Log `curiosity_rewards.max()` over the interval and a high quantile (p99). Two arms with identical mean bonus and different max bonus are doing completely different things, and the mean cannot tell them apart.
3. **`approx_kl` — max.** PPO is broken by one bad update, not by the average update. A mean KL of 0.008 with a max of 0.4 is a run in trouble; the mean alone reads as healthy.
4. **`iteration_seconds` — max, and `steps_per_second` — min.** These find a node going bad or a filesystem stall. A mean absorbs a 30-second stall into a 200-update average and hides it.
5. **`burned_rows_dropped` — max, plus a count of intervals exceeding 1,024.** `minibatch_drop_allowance = 1024` is the assumption that keeps the minibatch shape constant. If the drop count ever exceeds it, the code silently takes the ragged-shape path (line 736) and cuDNN re-benchmarks every convolution — the measured cost of shape drift in the comment at line 724 is the update phase going 1.28 s to 5.32 s. A mean of ~412 would never reveal an occasional excursion to 1,100.
6. **`gpu_memory_peak_mb` — keep it a max, but reset it per interval.** Call `torch.cuda.reset_peak_memory_stats()` at each row. Right now it is a lifetime max that saturates in the first hour and then carries no information for six days. Per-interval peak is what tells you whether packing another run onto the card is safe.

Fields that should stay plain means: `losses/value_loss`, `losses/policy_loss`, `losses/entropy`, `losses/fwd_loss`, `learning_rate` (deterministic anyway).

---

## 6. Throughput cost of logging, measured

`RunRecord.flush` rewrites the entire record — config, all update rows, all 96,320 episode rows — into a temp file, fsyncs, chmods, and renames. **Cost grows with accumulated history, so the last flush of a run is the most expensive one.** I measured it on the actual NFS mount (`corezfs02:/p/rlprojects`) at the sizes a real run passes through:

| record contents | file size | flush, as coded | flush, `dumps`+`write` | speedup |
|---|---|---|---|---|
| 50,000 ep rows, 100 update rows (cap just filled) | 6.69 MB | 0.349 s | 0.141 s | 2.5x |
| 73,160 ep rows, 306 update rows (mid-run) | 9.93 MB | 0.510 s | 0.185 s | 2.8x |
| 96,320 ep rows, 611 update rows (end, C=200) | 13.25 MB | 0.684 s | 0.255 s | 2.7x |
| 96,320 ep rows, 4,883 update rows (end, C=25) | 17.03 MB | 0.852 s | 0.312 s | 2.7x |

Stage breakdown at 13.25 MB: `json.dumps` 0.195 s, `write` 0.046 s, `fsync` under 1 ms, chmod+rename 1 ms. **The cost is Python serialisation, not the disk.**

### Full-run cost, integrating the real file-size trajectory

| C | flushes per run | bytes rewritten per run | across 150 runs | cost as coded | cost with `dumps`+`write` |
|---|---|---|---|---|---|
| 25 | 4,883 | 52.7 GB | **7.90 TB** | 44.6 min (0.469%) | 17.0 min (0.178%) |
| 50 | 2,442 | 23.6 GB | 3.55 TB | 20.1 min (0.211%) | 7.8 min (0.081%) |
| 100 | 1,221 | 11.2 GB | 1.67 TB | 9.5 min (0.100%) | 3.7 min (0.039%) |
| **200** | **611** | **5.4 GB** | **0.81 TB** | **4.6 min (0.049%)** | **1.8 min (0.019%)** |
| 400 | 306 | 2.7 GB | 0.40 TB | 2.3 min (0.024%) | 0.9 min (0.009%) |
| 800 | 153 | 1.3 GB | 0.20 TB | 1.1 min (0.012%) | 0.4 min (0.005%) |

**Answering the owner's worry directly: logging is not slowing the run down much even today.** At C=25 the flush costs 0.47% of a 6.6-day run — 45 minutes. That is a real cost but not the reason to change cadence. The reason to change cadence is the data quality in section 1, and the shared-filesystem load below.

### Is rewriting the whole record a problem at 150 runs? Yes — but on the shared filesystem, not per-run

Sustained aggregate write load with 150 runs in flight:

| C | flush interval per run | flushes/s across 150 | sustained MB/s | total NFS writes |
|---|---|---|---|---|
| 25 | 2.0 min | 1.28 | 11.1 | 7.90 TB |
| 100 | 7.8 min | 0.32 | 2.8 | 1.67 TB |
| **200** | **15.6 min** | **0.16** | **1.4** | **0.81 TB** |
| 400 | 31.2 min | 0.08 | 0.7 | 0.40 TB |

At C=25 the campaign writes **7.9 TB to a 1.0 TB NFS share, sustained for a week, to produce 2.37 GB of final data** — a write amplification of about 3,300x, against a filesystem that other rlprojects members are using at the same time. That is the genuine problem with the current setup, and it is caused by the full-rewrite design rather than by the cadence.

### Two fixes, in order of value per line changed

**Fix 1 (one line, do it regardless of cadence).** In `RunRecord.flush`, replace
```python
json.dump(self.as_dict(completed), f)
```
with a serialise-then-write:
```python
blob = json.dumps(self.as_dict(completed))
f.write(blob)
```
Measured 2.7x faster at every size, identical output bytes, no change to the atomic temp-and-rename.

**Fix 2 (the structural one).** Move `train_episode_history` out of the rewritten JSON into an append-only `<run_id>_of_<total>.episodes.jsonl` sidecar. The episode history is 11.37 MB of the 11.92 MB record and it is append-only by nature — it is rewritten hundreds of times for no reason. Measured: appending a C=200 interval's worth of episode rows costs 5.0 ms pre-cap (7,671 rows, 1.0 MB) and 1.5 ms post-cap (77 rows). The record that still gets rewritten drops to 0.56 MB and flushes in **8 ms instead of 684 ms**.

Effect at C=200 across the campaign: rewritten bytes fall from **0.81 TB to 25.6 GB**, and per-run logging cost falls from 4.6 min to a few seconds. The cap-and-stride logic moves unchanged into the append path; `restore_for_resume` truncates the sidecar by `step <= resume_step` exactly as it already truncates the in-memory list.

**Optional fix 3.** Decouple flushing from row-appending with a `--flush_every_rows` knob. Note that `restore_for_resume` **prefers the checkpoint's embedded history** and the checkpointing docstring records that the sweep requeue moves the JSON aside when a worker dies — so between checkpoints the JSON's freshness serves human monitoring only, never the resume. Flushing every 2 rows at C=200 (about every 31 min) halves the remaining cost and risks nothing. With fixes 1 and 2 applied this is no longer worth the extra knob.

---

## 7. Three fully specified options

Shared by all three (apply regardless of which cadence is chosen):

- `json.dumps` + `write` in `RunRecord.flush` (fix 1).
- Episode history to an append-only sidecar (fix 2).
- All interval statistics accumulated as **device tensors**, `.item()` called once per row.
- Add the gradient norm: capture the `clip_grad_norm_` return; log interval mean, max, and clip rate.
- `charts/steps_per_second` becomes an **interval** rate (steps and seconds since the previous row), with the interval min also logged. Keep a separate `charts/steps_per_second_cumulative` if the lifetime number is wanted.
- `torch.cuda.reset_peak_memory_stats()` at each row so the memory peak is per-interval.
- Episode-history cap and stride unchanged at 50,000 / 100.

### Option A — dense

| | |
|---|---|
| `log_every_updates` | **100** |
| interval | 1,638,400 steps; 7.8 min (5.6–11.4 min across cards) |
| rows per run | 1,221 (183,150 across 150 runs) |
| averaged into one row | 100 updates, **1,600 optimizer steps**, ~3,835 episodes |
| record per run | 12.47 MB (0.56 MB json + 11.37 MB sidecar) |
| campaign disk | 1.87 GB json + 9.37 GB checkpoints = **11.24 GB** |
| logging cost | ~10 s per run after fixes 1+2 (0.002%) |
| checkpoint granularity | 8 h + up to 7.8 min |
| pick this if | you expect instability and want to catch a diverging arm within 8 minutes |

### Option B — balanced (recommended)

| | |
|---|---|
| `log_every_updates` | **200** |
| interval | 3,276,800 steps; 15.6 min (11.1–22.8 min across cards) |
| rows per run | 611 (91,650 across 150 runs) |
| averaged into one row | 200 updates, **3,200 optimizer steps**, ~7,671 episodes |
| record per run | 11.92 MB (0.56 MB json + 11.37 MB sidecar) |
| campaign disk | 1.79 GB json + 9.29 GB checkpoints = **11.08 GB** |
| logging cost | ~5 s per run after fixes 1+2 (0.001%) |
| checkpoint granularity | 8 h + up to 15.6 min |
| max-not-mean fields | grad norm, intrinsic reward, `approx_kl`, `iteration_seconds`, `burned_rows_dropped`, gpu memory peak |
| pick this if | you want the best noise-per-row for the plots and a monitoring signal every quarter hour |

### Option C — sparse

| | |
|---|---|
| `log_every_updates` | **400** |
| interval | 6,553,600 steps; 31.2 min (22.3–45.5 min across cards) |
| rows per run | 306 (45,900 across 150 runs) |
| averaged into one row | 400 updates, **6,400 optimizer steps**, ~15,342 episodes |
| record per run | 11.64 MB (0.28 MB json + 11.37 MB sidecar) |
| campaign disk | 1.75 GB json + 9.25 GB checkpoints = **10.99 GB** |
| logging cost | ~3 s per run after fixes 1+2 (0.0005%) |
| checkpoint granularity | 8 h + up to 31.2 min |
| caveat | add a cheap heartbeat `print` every 50 updates (no flush) so the Slurm log does not go quiet for 45 min on the slowest card |
| pick this if | monitoring latency does not matter and you want the smoothest possible curves |

**Do not go to 800.** It saves 0.02 GB of disk and 0.7 s of run time over Option C, while pushing the worst-case checkpoint delay to 62 min (91 min on the slowest card) and leaving the Slurm log silent for over an hour. The costs it saves are already rounding errors; the risk it adds is not.

**Files read:** `/p/rlprojects/RND/08_cleanrl_ppo_rnd/src/ppo_rnd_envpool_shuze.py`, `/p/rlprojects/RND/08_cleanrl_ppo_rnd/src/run_record.py`, `/p/rlprojects/RND/08_cleanrl_ppo_rnd/src/checkpointing.py`. Flush timings measured on `corezfs02:/p/rlprojects` with `/p/rlprojects/RND/.venvs/exploration/bin/python`; probe files removed.