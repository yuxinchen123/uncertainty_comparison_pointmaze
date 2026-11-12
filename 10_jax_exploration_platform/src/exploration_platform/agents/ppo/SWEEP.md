# Sweeping across copy groups

The copies normally differ only by their seed; a sweep splits them into groups and gives each
group its own learning rate, its own weight on the intrinsic reward, or both, so one run answers
"which setting is best" instead of one run per setting.

## Using it

```python
from exploration_platform.training.runner import Runner
from exploration_platform.training.sweep import sweep_config

# 3 learning rates x 11 intrinsic weights x 256 copies = 8,448 copies in one run
cfg = sweep_config(learning_rates=[1e-3, 1e-4, 1e-5],
                   betas=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5],
                   copies_per_group=256)
runner = Runner(cfg, bonus="rnd_next_state")
state, stats = runner.train(num_iterations=20000, history_every=50)
```

Either list may be left out: `sweep_config(learning_rates=[...], copies_per_group=128)` sweeps
only the rate, `sweep_config(betas=[...], copies_per_group=128)` only the weight. The groups are
the cross product with the **rate as the outer factor**, and every cell gets the same number of
copies — which is what makes paired seeding well defined, since copy k then exists in every group.

`stats` carries `learning_rate_per_copy`, `beta_per_copy`, `group_index`, `group_settings` and
`sweep_seed_mode`, so a group's curve is the mean over the copies whose group index matches.

## Per-copy progress recording

`train(..., history_every=N)` records, every N iterations, each copy's extrinsic reward, its mean
intrinsic reward, and — when `track_coverage=True` — the fraction of the maze it has visited. The
visited map is a per-copy boolean array kept on the device and updated inside the compiled
iteration; only the recorded iterations copy anything back to the host, so the recording does not
serialise the loop.

## What the groups share, and what they do not

`sweep_seed_mode="paired"` (the default) gives copy k of EVERY group the same initial weights, so
the only difference between group 0's copy 7 and group 3's copy 7 is the setting being swept. That
makes the comparison between groups paired. `sweep_seed_mode="distinct"` restores the ordinary
behaviour of one seed stream per copy.

The environment cannot contribute to the pairing any more: its position noise is zero, so every
copy of every group starts every episode at exactly the same point regardless of its seed. What
the seed still controls is the initial weights, and through them the actions the copy takes.

Nothing else is shared: each copy keeps its own networks, environments, running statistics,
optimizer moments and — for a bonus that keeps one — its own count table. The tests check that
changing one group's rate, or one group's intrinsic weight, leaves every other group's parameters
bitwise unchanged.

## How it works

Adam is elementwise, so C copies stacked on the leading axis are already C independent Adam
optimizers, and the learning rate was the only thing that was a scalar. It becomes a `[C]` device
constant, reshaped to `[C, 1, ...]` against each parameter inside the same elementwise update —
XLA folds the broadcast into the kernel it was already emitting.

The weight on the intrinsic reward appears in exactly one place,
`advantage = beta * intrinsic + ext_coef * extrinsic`, so it becomes a `[C]` constant reshaped to
`[1, C, 1]` against the `[T, C, N]` advantage buffers. Same idea, one line.

A knob that is NOT being swept keeps its scalar, so the compiled program for a run that does not
sweep it is exactly what it was. Sweeping the intrinsic weight alone therefore leaves the whole
optimizer path untouched, and one intrinsic weight everywhere reproduces the scalar program bit
for bit — which is what the test of that name checks.

The annealing schedule stays one scalar argument per iteration: the constant holds the base rates
and the argument carries the multiplier, so no per-iteration host-to-device copy is introduced.

## More settings per copy: named cells

Explicit cells may give each copy more than a rate and a weight. `setting_names` says what each
column of `parameter_settings` holds, and a name that is not listed stays the configuration's own
scalar, so the compiled program of a run that does not list it is what it was.

| name | where the per-copy value enters |
|---|---|
| `learning_rate` | Adam's step, `[C, 1, ...]` against each parameter |
| `int_coef` | the intrinsic advantage's weight, `[1, C, 1]` |
| `ent_coef`, `vf_coef` | the per-copy loss terms, `[C]` |
| `clip_coef` | the ratio clip and the extrinsic value clip, `[C, 1]` against `[C, M]` rows |
| `ext_reward_scale`, `ext_reward_offset` | the reward the learner sees, scale x reward + offset, `[1, C, 1]` |

```python
from exploration_platform.agents.ppo.config import PPOConfig

# two entropy coefficients x two reward settings (limit 400), 16 copies each on the same seeds
cfg = PPOConfig(n_copies=64, copies_per_group=16,
                setting_names=("ent_coef", "ext_reward_scale", "ext_reward_offset"),
                parameter_settings=((0.001, 1.0, 0.0), (0.001, 1 / 400, -1 / 400),
                                    (0.01, 1.0, 0.0), (0.01, 1 / 400, -1 / 400)))
```

`SweepVectors.setting(cfg, name, shape, transform)` returns the scalar or the per-copy array. A
derived constant such as `1 - clip_coef` is formed from the Python numbers before rounding to
float32, as the scalar program forms it, so one value for every copy reproduces the scalar program
bit for bit (`tests/agents/test_per_copy_settings.py`).

A further per-copy setting needs one row in `SETTING_RULES` of `training/sweep.py` and a call of
`sweep.setting` where its scalar is used. Settings that fix shapes or structure cannot differ
between copies: environments per policy, rollout length, epochs, minibatches, update style,
network widths, the map, the bonus family.
