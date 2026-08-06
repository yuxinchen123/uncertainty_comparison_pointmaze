# Correcting what "no gradient clipping" means, and a measurement error it exposed

2026-08-05. Two changes, one of which invalidates a number reported earlier in the session.

## 1. The arm definition was wrong

Arms 2 and 5 were implemented as `max_grad_norm = 0.0`, which disables clipping for **both**
networks, because CleanRL clips them jointly:

```python
nn.utils.clip_grad_norm_(list(agent.parameters()) + list(rnd_model.predictor.parameters()), 0.5)
```

One call, one parameter list, one norm. Setting the threshold to zero removes the policy's clip as
well as the predictor's, so arms 2 and 5 would have differed from arm 1 in two ways at once and no
result from them could have been attributed to the predictor.

The intent is that the **policy is clipped at 0.5 in every arm** and only the predictor's clipping
changes. That is now expressed by two knobs rather than by a magic threshold value:

| knob | default | meaning |
|---|---|---|
| `joint_grad_clip` | `True` | clip the policy and the predictor together through one norm, as CleanRL does |
| `rnd_max_grad_norm` | `0.0` | when the clips are separate, the predictor's own threshold; `0` leaves it unclipped |

`max_grad_norm` keeps its meaning — the policy's threshold — and is `0.5` in all five arms.

| arm | `max_grad_norm` | `joint_grad_clip` | `rnd_max_grad_norm` |
|---|---|---|---|
| arm1_original | 0.5 | **True** | joint, no separate threshold |
| arm2_no_rnd_grad_clip | 0.5 | **False** | 0.0 (unclipped) |
| arm3_update_proportion_1 | 0.5 | **True** | joint, no separate threshold |
| arm4_shallower_predictor | 0.5 | **True** | joint, no separate threshold |
| arm5_all | 0.5 | **False** | 0.0 (unclipped) |

Separating the clips changes what the policy is clipped **on** — its own norm rather than the
combined one — which is unavoidable: there is no way to leave the predictor out of a joint norm and
still have that joint norm bound the policy. The policy's threshold is unchanged, and its own norm is
below the joint one, so the policy is clipped no harder in arms 2 and 5 than in arm 1.

Nine unit tests in `tests/test_arms_and_clipping.py` pin this, including that arms 1, 3 and 4 still
use the joint clip and that the queue's arm definitions match the trainer's.

## 2. The predictor's gradient norm was being measured after the clip, not before

Found while fixing the above. `clip_grad_norm_` scales the gradients in place, and the predictor's
own norm was computed inside `record()`, which the trainer called **after** the clip. Under a joint
clip that means the recorded norm was the post-clip value — the true norm multiplied by the clip's
scale factor.

The fix moves the clipping into `grad_stats.clip_gradients`, which measures first and clips second,
so the ordering cannot be got wrong from the call site.

**This changes a number reported earlier in the session.** Comparing arms 1 and 2 on a short run, I
said arm 2's predictor gradient norm was about three times arm 1's and called it the effect the
ablation isolates. It was not: arm 1's figure was post-clip. At matched settings the two are nearly
equal, which is what one would expect this early in training.

| arm | predictor norm, as reported before | predictor norm, measured before clipping | scale the clip applied |
|---|---|---|---|
| arm1_original | 0.0907 | **0.3023** | 0.302 |
| arm2_no_rnd_grad_clip | 0.3100 | 0.3100 | 1.000 (not clipped) |

0.3023 × 0.302 = 0.0913, which is the old figure — the discrepancy was entirely the clip's scaling.
Whether the two arms' predictor gradients diverge later in training is now an open question the run
will answer, rather than something already visible at 2,560 steps.

## 3. What each row now reports

Every logged row carries all three pre-clip norms — joint, policy, predictor — whichever clip mode
the arm uses. One is measured by `clip_grad_norm_`, one by `grad_stats`, and the third follows
exactly, since the squared norms partition: joint² = policy² + predictor². All of it stays on the GPU
and is read to the host once per logging interval.

New and renamed fields:

| field | meaning |
|---|---|
| `grad/joint_grad_clip`, `grad/policy_clip_threshold`, `grad/rnd_clip_threshold` | the arm's clip configuration, in every row |
| `grad/predictor_clip_fired_fraction`, `grad/mean_scale_applied_to_predictor` | how often the predictor's gradient was scaled, and by how much — zero and one by construction in arms 2 and 5 |
| `grad/policy_clip_fired_fraction`, `grad/mean_scale_applied_to_policy` | the same for the policy |
| `grad/mean_policy_norm_before_clipping` | new; previously only the joint and predictor norms were kept |
| `grad/predictor_share_of_squared_norm` | renamed from `grad/mean_predictor_share_of_squared_norm`, which was misleading: it is a ratio of interval sums, not a mean of per-step ratios |

Note that `predictor_clip_fired_fraction` under a joint clip counts every step where the **combined**
norm exceeded 0.5, whether or not the predictor caused it. That is the point of the ablation, so the
norm columns are what say who was responsible.

## 4. End-to-end check, all five arms

2,560 steps each on CPU, 10-update logging cadence:

| arm | policy clip | joint? | RND threshold | joint norm | policy norm | predictor norm | policy clip fires | predictor clip fires | scale on predictor |
|---|---|---|---|---|---|---|---|---|---|
| arm1_original | 0.5 | True | joint | 1.7275 | 1.6996 | 0.3023 | 1.00 | 1.00 | 0.302 |
| arm2_no_rnd_grad_clip | 0.5 | False | 0 | 1.8100 | 1.7820 | 0.3100 | 1.00 | 0.00 | 1.000 |
| arm3_update_proportion_1 | 0.5 | True | joint | 1.7423 | 1.7305 | 0.1985 | 1.00 | 1.00 | 0.300 |
| arm4_shallower_predictor | 0.5 | True | joint | 1.5140 | 1.4769 | 0.3185 | 1.00 | 1.00 | 0.383 |
| arm5_all | 0.5 | False | 0 | 1.6264 | 1.6127 | 0.2009 | 1.00 | 0.00 | 1.000 |

Reading it: the joint norm is far above 0.5 at every step this early, so the clip fires on every
optimizer step in the joint arms. The predictor contributes about 3% of the joint squared norm
(0.30² / 1.73²), so the policy is what pushes the joint norm over the threshold — and the predictor's
gradient is scaled to 30% of its size as a side effect of that. Arms 2 and 5 are the arms where that
side effect does not happen.

## 5. Files changed

- `src/grad_stats.py` — rewritten: `clip_gradients` owns the measure-then-clip ordering; two knobs;
  three pre-clip norms; per-network firing rates and scales.
- `src/ppo_rnd_envpool_shuze.py` — `joint_grad_clip` and `rnd_max_grad_norm` replace
  `clip_rnd_predictor`; both clip sites call `clip_gradients`; the statistics object is `None` when
  the statistics are off, which also skips measuring the predictor's norm.
- `src/grad_norm_stats.py` and `tests/test_grad_norm_stats.py` — deleted; superseded by `grad_stats`
  and reachable only from their own test.
- `tests/test_arms_and_clipping.py` — new, 10 tests.
- `train_runs/build_ablation_configs.py` — arm definitions; `configs.jsonl` rebuilt, 150 runs, every
  25-id prefix arm-balanced.
- `analysis/compute_run_metrics.py` — the renamed gradient fields.
- The run's `for_collaborator/README.md`, `smoke_test.sh`, and the progress folder's smoke and
  short-run scripts — the arm's new name.
- `07_reconstruction/development_document/RND_development_document.tex` — Table 78 splits the
  clipping row in two, so the table shows the policy clipped at 0.5 in all five arms and only the
  predictor's row changing. Builds at 141 pages with no overfull box in the table.
