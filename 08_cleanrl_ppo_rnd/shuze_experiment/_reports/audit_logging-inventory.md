# What one training run in this sweep writes to disk

Run folder: `/p/rlprojects/RND/08_cleanrl_ppo_rnd/train_runs/2026-08-05-17-45_cleanrl_train_run_1_ppo-rnd_montezuma-revenge-v5__envpool-autoreset-fixed__128env-128step-2e9step__int-coef1-ext-coef2-updateproportion0.25__seed1-30__checkpoint-8h-resumable__gpu-gnolim/`
Trainer: `/p/rlprojects/RND/08_cleanrl_ppo_rnd/src/ppo_rnd_envpool_shuze.py`
Record helper: `/p/rlprojects/RND/08_cleanrl_ppo_rnd/src/run_record.py`
Checkpoint helper: `/p/rlprojects/RND/08_cleanrl_ppo_rnd/src/checkpointing.py`
Live record cross-checked: `.../data/local/0_of_30.json` (run_id 0, seed 1, node cheetah09, NVIDIA RTX A4000).

Settings for this sweep: `--num_envs 128 --num_steps 128` (batch 16,384 environment steps per policy update), `--total_timesteps 2000000000`, `--log_every_updates 25`, `--checkpoint_every_seconds 28800`, `--episode_history_cap 50000`, `--episode_history_stride 100`. The last two are not on the command line; they are the dataclass defaults (`ppo_rnd_envpool_shuze.py:117-120`) and the live record confirms them (`episode_history_cap: 50000`, `episode_history_stride_past_cap: 100`). `num_iterations = 2000000000 // 16384 = 122070` (record: `num_iterations: 122070`), so the last step of a finished run is 1,999,994,880 — 5,120 short of 2e9.

---

## 1. Everything a run writes, with its trigger

### 1.1 Written by the trainer process itself

| What | Path | Written when | Code |
|---|---|---|---|
| The JSON record (whole file, all four groups, rewritten each time) | `data/local/<run_id>_of_30.json` | every 25 policy updates, and on the final update | `:794` condition, `:834` flush |
| Same file | same | immediately after every checkpoint write | `:847` |
| Same file, this time with `completed` true | same | once, after the update loop ends | `:862` |
| Transient copy during a flush | `data/local/<run_id>_of_30.json.tmp` | inside every flush, renamed over the target | `run_record.py:89-97` |
| The checkpoint | `data/local/<run_id>_of_30.checkpoint.pt` | when 28,800 s of wall clock have passed since the last one, tested once per update | `:841-846` |
| The checkpoint again | same | once, after the update loop ends | `:857-861` |
| Transient copy during a checkpoint write | `...checkpoint.pt.tmp` | inside every checkpoint write, renamed over the target | `checkpointing.py:77-84` |
| Stdout progress line | `slurm/logs/run_<run_id>.log` | every 25 updates, and on the final update | `:835-838` |
| Stdout checkpoint line | same log | after every checkpoint write | `:849-850` |
| Stdout resume line | same log | only when a checkpoint was found and loaded | `:483-484` |
| Stdout normaliser lines | same log | only when no checkpoint was loaded | `:496`, `:520` |
| Stdout final line | same log | once, after the loop ends | `:864` |

Notes on the triggers:

1. **The flush condition is `update % args.log_every_updates == 0 or update == num_updates`** (`:794`). With `log_every_updates 25` that is updates 25, 50, …, 122050, plus the final update 122070. Nothing at all is written before update 25 — the `RunRecord` constructor only creates the directory (`run_record.py:50`), it does not write a file. A run that dies inside its first ~5 minutes (about 3 minutes of observation-normaliser warm-up plus 2 minutes of training) leaves no JSON.
2. **The flush rewrites the entire file every time**, top-level fields and all three history lists (`run_record.py:69-83`, `85-97`). It is written to a `.tmp` sibling, fsynced, `chmod 0o660`, then `os.replace`d, so a reader always sees a complete file, either the old one or the new one.
3. **`completed` is `false` on every flush except the last** (`:834`, `:847` pass `completed=False`; `:862` passes `completed=args.profile_iterations == 0`). A record with `completed: false` is a normal in-progress or killed run, not a damaged file.
4. **The checkpoint cadence is wall clock, not update count** (`:841`), and the condition is tested only after a full update finishes, so the write lands within one update (~4.8 s) of the 8-hour mark.
5. **Both files live in `--output_dir`**, which for this sweep is `data/local` (`:448-449`). There is no separate checkpoint directory. Exactly one checkpoint file per run exists at any time; the rename overwrites the previous one (`checkpointing.py:83-92`).
6. **Stdout and stderr of the trainer both go to `slurm/logs/run_<run_id>.log`**, opened in append mode by the sweep's worker manager (`/p/rlprojects/.claude/skills/submit-gpu-sweep/scripts/worker_manager.py:423-427`), so a restarted run appends a second segment to the same file. The Slurm `%x_%j.out` files under `slurm/logs/` hold the worker manager's own output, not the trainer's.

### 1.2 Written around the run by the sweep machinery, not by the trainer

| What | Path | Cadence |
|---|---|---|
| Queue claim marker, rewritten with the claim block, then moved to `done/` or `failed/` | `queue/running/<NN>_of_30.json` | at claim, at exit |
| Per-run resource sample (GPU memory, host memory, busy threads) | `slurm/resource_usage/per_run/<run_id>.jsonl` | one line every 120 s |
| Whole-node status | `slurm/resource_usage/node_status/<node>.jsonl` | one line every 120 s |

---

## 2. Every key in the record, group by group

Verified two ways: read from the code, and read back from all live records. Every one of the 22 record files present at the time of reading has the identical top-level key tuple, and each of the three history lists has exactly one distinct key tuple across every row of every file — no row has an extra or missing key.

### 2.1 Top-level fields

Built at `:450-469` as `config` and extended at `run_record.py:69-82`. Order in the file is: the config dict, then the five bookkeeping fields, then the three history lists.

Identity and machine, all captured at process start:

| Key | Meaning |
|---|---|
| `run_id` | this run's index in the sweep, 0 to 29 |
| `run_total` | 30 |
| `algorithm` | the fixed string `ppo_rnd` |
| `implementation` | `cleanrl ppo_rnd_envpool.py @ fe8d8a0, envpool auto-reset fixed` |
| `env_id` | `MontezumaRevenge-v5` |
| `a_seed` | the value of `--seed`; run_id 0 is seed 1, run_id k is seed k+1 |
| `profile_tag` | empty for these runs; a label for profiling runs |
| `device` | `cuda` |
| `gpu_name` | the GPU this process got, e.g. `NVIDIA RTX A4000` |
| `env_threads` | envpool worker threads, 8 here, from `GPU_SWEEP_CPUS_PER_RUN` (`:353-370`) |
| `slurm_job_id` | the Slurm job of this process |
| `hostname` | `SLURMD_NODENAME` of this process |

Then every field of `Args` except `run_id`, `run_total`, `seed`, `env_id`, which are excluded by the splat at `:465` because they already appear under their own names: `exp_name`, `torch_deterministic`, `cuda`, `total_timesteps`, `learning_rate`, `num_envs`, `num_steps`, `anneal_lr`, `gamma`, `gae_lambda`, `num_minibatches`, `update_epochs`, `norm_adv`, `clip_coef`, `clip_vloss`, `ent_coef`, `vf_coef`, `max_grad_norm`, `target_kl`, `update_proportion`, `int_coef`, `ext_coef`, `int_gamma`, `num_iterations_obs_norm_init`, `fix_envpool_autoreset`, `output_dir`, `log_every_updates`, `episode_history_cap`, `episode_history_stride`, `checkpoint_every_seconds`, `resume`, `opt_env_threads`, `opt_fused_policy_pass`, `opt_rnd_no_grad`, `opt_uint8_obs`, `opt_gpu_obs_rms`, `opt_gpu_norm_stats`, `opt_no_sync_update`, `opt_fast_obs_norm_init`, `opt_cudnn_benchmark`, `opt_fixed_minibatch_shape`, `minibatch_drop_allowance`, `opt_amp_fp16`, `opt_channels_last`, `opt_matmul_tf32`, `opt_torch_compile`, `profile_iterations`, `batch_size`, `minibatch_size`, `num_iterations`.

Then five fields added at flush time (`run_record.py:74-79`):

| Key | Meaning |
|---|---|
| `completed` | true only in the final flush of a non-profiling run |
| `episodes_seen` | every complete game finished, including the ones the stride rule discarded |
| `episodes_dropped_from_history` | how many of those were not kept as rows |
| `episode_history_cap` | 50000, the point at which the stride rule starts |
| `episode_history_stride_past_cap` | 100 |
| `runtime_seconds` | seconds carried over from earlier segments plus seconds in this one |

### 2.2 `eval_history` — one row per logged update

Built at `:806-832`. Despite the name there is no evaluation in it: it holds throughput and optimisation diagnostics (`run_record.py:15`).

| Key | Meaning |
|---|---|
| `step` | `global_step` at that update, counting all 128 environments (a rollout adds 16,384) |
| `update` | the policy-update index, 1 to 122070 |
| `charts/steps_per_second` | steps done in this job segment divided by seconds in this job segment — a running average, not an instantaneous rate |
| `charts/iteration_seconds` | seconds for this one update: rollout, intrinsic-reward normalisation, advantage computation, and the optimisation, measured from `:529` to `:813` |
| `charts/rollout_seconds` | seconds of the 128-step rollout alone (`:540-596`) |
| `charts/update_seconds` | seconds of the advantage computation plus the four optimisation epochs (`:609-791`) |
| `charts/obs_norm_init_seconds` | seconds spent priming the observation normaliser at process start; 0.0 on every row written after a resume (`:522`) |
| `charts/learning_rate` | the annealed learning rate in force for this update |
| `charts/burned_rows_dropped` | rows of this update's 16,384 that envpool burned auto-resetting and the fix removed |
| `charts/gpu_memory_peak_mb` | `torch.cuda.max_memory_allocated()`, the largest allocation since this process started, never reset |
| `charts/gpu_memory_reserved_mb` | `torch.cuda.max_memory_reserved()`, the largest reservation since this process started |
| `losses/value_loss` | extrinsic plus intrinsic value loss of the last minibatch of the last epoch |
| `losses/policy_loss` | clipped policy-gradient loss of that same last minibatch |
| `losses/entropy` | mean policy entropy on that same last minibatch |
| `losses/fwd_loss` | RND predictor loss on that same last minibatch, over the quarter of rows the update-proportion mask kept |
| `losses/approx_kl` | the `(r-1)-log r` estimate of the policy change on that same last minibatch |
| `losses/old_approx_kl` | the `-log r` estimate on that same last minibatch |

### 2.3 `train_history` — one row per logged update, same rows as `eval_history`

Built at `:798-805`. Added in the same call (`run_record.py:64-67`), so the two lists always have the same length and the same `(step, update)` values.

| Key | Meaning |
|---|---|
| `step` | as above |
| `update` | as above |
| `train/mean_extrinsic_reward` | mean game score over the last 20 complete games finished by any of the 128 environments; `null` before the first game finishes |
| `train/mean_intrinsic_reward` | mean raw curiosity bonus over the 16,384 rows of this rollout, before normalisation (`:606`, taken before the division at `:607`) |
| `train/mean_normalized_intrinsic_reward` | the same mean after dividing by the standard deviation of the discounted intrinsic return, which is the value the optimiser actually sees |
| `train/n_episodes_averaged` | how many games are in the 20-slot window; 20 once warm |

Live values from `0_of_30.json`, update 25: raw 30.17, normalised 0.01226, so the divisor was about 2,460 at that point.

### 2.4 `train_episode_history` — one row per complete game

Appended at `:590-595`, inside the rollout loop, under the condition `if d and info["lives"][idx] == 0` (`:588`).

| Key | Meaning |
|---|---|
| `step` | `global_step` at the rollout step on which the game ended |
| `train/extrinsic_reward` | the game's total score, unclipped |
| `train/intrinsic_reward` | the raw curiosity bonus of that single final step for that single environment, not a sum over the game |
| `train/episode_length` | environment steps in the game, with envpool's burned auto-reset steps excluded (`:209-212`) |

Two facts behind these, both verified against envpool 0.6.6 in this run's environment rather than assumed:

1. **`info["reward"]` is the unclipped reward while the reward fed to the optimiser is sign-clipped.** Measured on MsPacman with `reward_clip=True`: the step reward was 1.0 and `info["reward"]` was 10.0 at the same step. The wrapper accumulates `infos["reward"]` (`:205`), so `train/extrinsic_reward` is the game score. For Montezuma the first key is 100 points, so scored rows will read 100, 400 and so on, while the reward the policy is trained on is 1 per event.
2. **A row is one complete game, not one life.** With `episodic_life=True`, `done` fires on every life loss but `terminated` only at game over; measured on MsPacman, `done` at lives 2 and 1 had `terminated 0`, and `done` at lives 0 had `terminated 1`. The wrapper resets the return and length counters on `terminated` (`:215-216`), and the row is only appended when `lives == 0`. Montezuma starts with 6 lives, so one row covers 6 lives.

---

## 3. Cadences in real units at this run's settings

Taking 3,400 steps per second. Measured on run 0: `charts/steps_per_second` climbed from 3,276 at update 25 to 3,451 at update 525; per-node cumulative rates across the sweep were 3,385-3,468 on RTX A4000, 3,694-3,758 on Quadro RTX 6000, 3,956-3,964 on RTX 2080 Ti, 3,779-4,564 on RTX A4500.

| Quantity | Value |
|---|---|
| Steps per policy update | 16,384 |
| Seconds per policy update | 4.82 (16,384 / 3,400); measured mean of `charts/iteration_seconds` 4.77 |
| Steps between `eval_history` rows | 409,600 (25 updates) |
| **Seconds between `eval_history` rows** | **120.5 s, that is 2.01 minutes** |
| Rows per hour, per day | 29.9 and 717 |
| Whole-file flush cadence | the same 120.5 s, plus one extra flush per checkpoint |
| Checkpoint cadence | 28,800 s wall clock, about every 6,050 updates, about every 99.2M steps |
| Resource-sampler cadence (worker manager) | 120 s |
| **Wall clock for one full run** | **588,235 s = 163.4 h = 6.81 days** at 3,400 steps/s; 5.1 to 6.8 days across the node types above |
| Slurm segments needed | at least two, since the gpu partition caps a job at 4 days, so every run resumes at least once |
| Periodic checkpoints per run | about 20 |

**Rows in the record at the end of a finished run.** `floor(122070 / 25) = 4882` rows from the modulo condition, plus one for the final update 122070 which is not a multiple of 25:

1. `eval_history`: **4,883 rows**.
2. `train_history`: **4,883 rows**.

**Episodes.** Measured on run 0 at update 525: 20,136 complete games in 8,601,600 steps, one game per 427.1 steps, mean game length 418.3 environment steps, median 345, range 133 to 3,525. Across the sweep the first ~45 minutes produced 20,233 to 33,180 games per run.

1. **Games finished in a full run**: 1,999,994,880 / 427.1 = **about 4.68 million**, at roughly 8.0 games per second across the 128 environments. This assumes the current game length holds; if the agent starts scoring and surviving longer, both this count and the file size fall.
2. **Games kept as rows**: the first 50,000, then every 100th by `episodes_seen` index. Kept = `50000 + (floor(4682880/100) - floor(50000/100)) = 50000 + 46328` = **96,328 rows**.
3. **Games dropped**: **about 4,586,552**, counted in `episodes_dropped_from_history`.

The kept-row formula was verified by running `RunRecord` directly: cap 100, stride 10, 10,000 episodes gives 1,090 kept and 8,910 dropped, and the first rows kept past the cap are indices 109, 119, 129, matching `episodes_seen % stride == 0`. The unit test pins the same rule (`tests/test_run_record.py:38-47`).

---

## 4. What is not logged, and why

1. **No Weights and Biases.** The trainer never imports it. The module docstring states it was removed and replaced by the JSON record (`ppo_rnd_envpool_shuze.py:11-13`); upstream `cleanrl/cleanrl/ppo_rnd_envpool.py:252-262` still has the `wandb.init` block. There is nothing to look up on a wandb dashboard for this run.
2. **No tensorboard.** No `SummaryWriter` anywhere in the trainer; upstream has one at `cleanrl/cleanrl/ppo_rnd_envpool.py:18` and `:263`. There is no `runs/` event-file directory.
3. **No per-step data.** The finest granularity in the record is one row per completed game and one row per 25 updates. Nothing in the rollout loop writes per step. The reason is stated in `run_record.py:18-20`: a 2e9-step Atari run finishes millions of episodes, and an uncapped per-episode list already outgrows the filesystem, let alone a per-step one. The rollout buffer itself, 16,384 rows of 4x84x84 observations, is rebuilt every update and never saved.
4. **No evaluation episodes separate from training.** There is no evaluation environment, no deterministic-policy rollout, and no `n_eval_episodes` knob anywhere in the file. `eval_history` holds throughput and optimisation diagnostics only, which `run_record.py:15` says explicitly. Every reward number in the record comes from the training rollouts: the per-game rows and the 20-game trailing mean at `:801`. This differs from the project's SAC runs, where `eval_history` really does hold separate evaluation episodes.
5. **No distance-to-ground-truth metrics.** The record has no `distance_history` and no `distance_to_gt/*` keys. The reason is in the mapping table at `run_record.py:16`: there is no oracle visit-count field on Atari to measure a bonus field against, unlike the PointMaze runs where the maze can be gridded.
6. **Also absent, and worth knowing**: no wall-clock timestamp on any row (only `step`, `update`, and the single cumulative `runtime_seconds`); no environment index on an episode row, so a game cannot be traced to one of the 128 emulators; no room, level or per-event score breakdown; no gradient norm and no explained variance; no per-minibatch or per-epoch losses, only the last minibatch; no observation-normaliser or reward-normaliser statistics in the JSON, which live only inside the checkpoint; no record inside the JSON that a resume happened, which appears only as a `[resume]` line in the log; and no per-life episodes, only complete games.

---

## 5. Size projections

Measured byte counts from the live record `0_of_30.json`: top-level fields 1,960 bytes; `eval_history` 697.0 bytes per row; `train_history` 217.2 bytes per row; `train_episode_history` 124.1 bytes per row. A synthetic record built at the projected end-of-run row counts, with the step and update fields widened to their final digit counts, serialises to **16,785,223 bytes**.

| Item | At end of one run | 30 seeds |
|---|---|---|
| Top-level fields | 1,960 B | 59 KB |
| `eval_history` 4,883 rows | 3.42 MB | 103 MB |
| `train_history` 4,883 rows | 1.08 MB | 33 MB |
| `train_episode_history` 96,328 rows | 12.2 MB | 367 MB |
| **Record JSON total** | **16.8 MB** | **0.50 GB** |
| **Checkpoint** | **50.26 MB** | **1.51 GB** |
| **Record plus checkpoint** | **67.1 MB** | **2.01 GB** |
| Run log, about 4,890 lines of 96 B | 0.47 MB | 14 MB |
| Resource-usage lines, 4,900 of about 195 B | 0.96 MB | 29 MB |
| **Everything** | **68.5 MB** | **2.06 GB** |

Notes:

1. **The checkpoint size is fixed and already measured**, not projected: the five canary checkpoints are 50,258,142 to 50,258,526 bytes, that is 50.26 MB or 47.9 MiB. It holds the agent, the RND predictor, the frozen RND target, the Adam state over both, two normalisers, the reward forward filter, the counters, a 20-value return deque, and four random-number-generator states (`checkpointing.py:58-73`). None of those grow, so the size stays constant for the whole run.
2. **Transient peak is higher.** Both writes go to a `.tmp` sibling before the rename, so during a checkpoint write a run momentarily occupies 100.5 MB of checkpoint, and during a flush 33.6 MB of record. Across 30 runs, if checkpoints happened to coincide, the peak would be about 3.5 GB rather than 2.0 GB.
3. **The whole record is rewritten every 120 s, and it grows.** Measured cost at the projected final size, on the shared `/p` filesystem: 0.33 s to serialise plus 0.04 s to write and fsync, so about 0.38 s per flush against a 120 s interval, roughly 0.3% of wall clock. Not a problem, but it is not free either, and it grows over the run.
4. **The record size falls if the agent improves.** All of the above assumes today's 427 steps per game. Longer games mean fewer games, which shrinks the dominant term.

---

## 6. Things that would surprise a later reader

1. **`episode_history_cap` is not a cap on the list length.** Past 50,000 rows the list keeps growing, at one row per 100 games. The code falls through to the stride branch and appends (`run_record.py:57-62`), so the final list is about 96,328 rows, not 50,000. The name reads like a hard limit; it is the point at which thinning starts. The unit test at `tests/test_run_record.py:38-47` pins this behaviour: cap 10 and stride 5 over 30 episodes gives 14 rows, not 10.
2. **`train/intrinsic_reward` on an episode row is one step's bonus, not the game's total.** It is `step_curiosity[idx]` (`:586`, `:593`), the raw curiosity value at the single rollout step where the game ended, for that one environment. It is also the pre-normalisation value, so it is on a completely different scale from what the optimiser sees. This matches upstream, which logged the same quantity as `charts/episode_curiosity_reward` (`cleanrl/cleanrl/ppo_rnd_envpool.py:383-387`), but the name in the record invites reading it as an episode return.
3. **`train/extrinsic_reward` is the unclipped game score, while PPO trains on the clipped reward.** Verified empirically against envpool. Do not compare it against the reward stream the value heads fit.
4. **An episode row is a complete game across all 6 lives, while the policy is trained on per-life episodes.** `episodic_life=True` ends a training episode at each life loss; the record only writes a row when `lives == 0`.
5. **`charts/steps_per_second` is a cumulative average over the current Slurm segment, not an instantaneous rate.** It is `(global_step - steps_at_start) / elapsed` with `steps_at_start` and `start_time` both set after the resume block (`:523-524`, `:796`), so it starts low, drifts up as start-up cost is amortised, and resets at every resume. Run 0's values climbing 3,276 to 3,451 over 500 updates is that drift, not the run speeding up. For an instantaneous rate use `16384 / charts/iteration_seconds`. The observation-normaliser warm-up is excluded from the denominator and reported separately as `charts/obs_norm_init_seconds`.
6. **`train/mean_extrinsic_reward` is a mean over a 20-slot deque of the most recently finished games** (`avg_returns = deque(maxlen=20)` at `:444`, appended at `:589`, averaged at `:801`). At about 8 games per second across 128 environments, those 20 games span roughly 2.5 seconds of game-endings, which is a very short and very noisy window, and it is shared across all 128 environments rather than being one game per environment. `train/n_episodes_averaged` is 20 in every live row. It is not an evaluation score and it is not a fixed-size sample.
7. **The `losses/*` values are from the last minibatch of the last epoch of that one update**, not averages. Each update runs 4 epochs of 4 minibatches, so the logged numbers come from optimisation step 16 of 16. This is upstream's behaviour too.
8. **`runtime_seconds` spans resumes, including work that was thrown away.** `prior_runtime_seconds` is loaded from the record on disk and added to the current segment (`run_record.py:49`, `:79`, `:115`). Because the record is flushed every 2 minutes but the checkpoint is written only every 8 hours, the carried-over value includes up to 8 hours of training that the resume rolled back. So `runtime_seconds` measures wall clock spent, not progress made, and it is always larger than `final_step / steps_per_second`.
9. **After a resume the history lists overlap, and `episodes_seen` double-counts.** On resume the record is restored first (`:474`) with all rows up to the last flush, then the checkpoint sets `start_update = state["update"] + 1` (`:480`), which is up to 8 hours earlier. Those updates are re-run and their rows appended again, so `eval_history` and `train_history` go up to the pre-kill update, jump backwards, and climb again — the same `(update, step)` pair can appear twice. At the current rate that is up to about 239 duplicated update rows and about 230,000 double-counted games per resume. **Any analysis must deduplicate by `update`, keeping the last occurrence, and must not read `episodes_seen` as an exact count for a run that resumed.**
10. **If the record exists but the checkpoint does not, the run silently restarts from update 1 while keeping the old rows.** `record.restore_from_disk()` at `:474` runs unconditionally under `--resume`; `load_checkpoint` returns `None` for a missing file (`checkpointing.py:99-100`), leaving `global_step 0, start_update 1`. This is the case for any job that dies inside its first 8 hours. There is a live example in this sweep: job 6533906 on cheetah02 was cancelled at 18:28 for being under-provisioned on CPUs and resubmitted as 6533915, and the 8 records for runs 3, 4, 8, 9, 10, 12, 16 and 19 were deleted by hand at the same time — which is the correct cleanup, precisely because keeping them would have produced this restart-with-old-rows record. Their logs now hold two segments in one file, the second beginning with `Start to initialize observation normalization parameter.....` and no `[resume]` line, which is the reliable way to tell a clean restart from a resume.
11. **The top-level `minibatch_size: 4096` is not the size actually used.** It is `batch_size // num_minibatches`, computed once at `:377` and never updated. With `opt_fixed_minibatch_shape` on, each update trains on `((16384 - 1024) // 4) * 4 = 15360` rows in 4 minibatches of **3,840** (`:698-707`). The 1,024-row allowance exists so cuDNN does not re-tune its convolutions on a new shape every update.
12. **`charts/burned_rows_dropped` does not count every row dropped.** It is set at `:688`, before the fixed-shape trim at `:698`. It counts only the envpool auto-reset rows, 70 to 347 per update in the live record, that is 0.4% to 2.1%. The additional genuine rows discarded to hold the shape constant are `16384 - burned_rows_dropped - 15360`, roughly 700 to 950 per update, chosen at random by the shuffle, and they are not reported anywhere.
13. **`charts/obs_norm_init_seconds` reads 0.0 on every row written after a resume** (`:522`), while pre-resume rows in the same file carry the real value. It is 87 to 200 seconds depending on the node.
14. **`charts/gpu_memory_peak_mb` and `charts/gpu_memory_reserved_mb` are process-lifetime maxima and are never reset**, so they are constant after warm-up (4,544.9 and 4,788.0 in every row of every live record) and start a fresh maximum in each new process after a resume.
15. **`hostname`, `slurm_job_id`, `gpu_name` and `env_threads` describe only the most recent segment.** `restore_from_disk` deliberately does not restore the config, so the current invocation's values overwrite the old ones (`run_record.py:108-109`). A run that moves between nodes leaves no trace of its earlier nodes in the JSON; that history is only in the queue marker's `claim` block and in the log file.
16. **`seed` is not a top-level key; `a_seed` is.** The splat at `:465` excludes `seed`, `env_id`, `run_id` and `run_total` because they are written under their own names above it.
17. **The record file does not exist for the first few minutes of a run**, and a run killed before update 25 leaves nothing at all in `data/local`.