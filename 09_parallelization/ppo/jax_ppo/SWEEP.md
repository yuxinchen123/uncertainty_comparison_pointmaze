# Sweeping a hyperparameter across copy groups (JAX)

The same feature as `../torch_ppo/SWEEP.md`, in the JAX trainer. The copies normally differ
only by their seed; a sweep splits them into groups and gives each group its own learning
rate, so one run answers "which rate is best" instead of one run per rate.

## Using it

```python
from jax_ppo_rnd import JaxPPORND, sweep_config

# 4 learning rates, 128 independent copies each -> 512 copies in one run
cfg = sweep_config(learning_rates=[1e-4, 3e-4, 1e-3, 3e-3], copies_per_rate=128)
trainer = JaxPPORND(cfg)
state, stats = trainer.train(num_iterations=20000, history_every=50)
```

`copies_per_rate` may also be a list, one count per rate, when the groups should differ in
size: `sweep_config([1e-4, 3e-4, 1e-3], copies_per_rate=[64, 256, 64])` gives 384 copies.

`stats` carries `learning_rate_per_copy`, `group_index` and `sweep_seed_mode`, so a group's
curve is the mean over the copies whose group index matches.

## Per-copy progress recording

`train(..., history_every=N)` records, every N iterations, each copy's extrinsic reward, its
mean intrinsic reward, and — when `track_coverage=True` — the fraction of the maze it has
visited. The visited map is a per-copy boolean array kept on the device and updated inside the
compiled iteration; only the recorded iterations copy anything back to the host, so the
recording does not serialise the loop.

## What the groups share, and what they do not

`sweep_seed_mode="paired"` (the default) gives copy k of EVERY group the same initial weights
and the same environment reset noise, so the only difference between group 0's copy 7 and
group 3's copy 7 is the learning rate. That makes the comparison between rates paired.
`sweep_seed_mode="distinct"` restores the ordinary behaviour of one seed stream per copy.

Nothing else is shared: each copy keeps its own networks, environments, running statistics and
optimizer moments. The tests check that changing one group's rate leaves every other group's
parameters bitwise unchanged.

## How it works

Adam is elementwise, so C copies stacked on the leading axis are already C independent Adam
optimizers, and the learning rate is the only thing that was a scalar. It becomes a `[C]`
device constant, reshaped to `[C, 1, ...]` against each parameter inside the same elementwise
update — XLA folds the broadcast into the kernel it was already emitting.

The annealing schedule stays one scalar argument per iteration: the constant holds the base
rates and the argument carries the multiplier, so no per-iteration host-to-device copy is
introduced. Without a sweep the optimizer arithmetic is exactly what it was.

## Sweeping something other than the learning rate

Every per-copy loss term already reduces over the copy axis, so the same mechanism extends to
any scalar knob that enters elementwise — the entropy coefficient, the value coefficient, the
clip range, the advantage weights, the gradient-norm limit. Each needs a `[C]` constant beside
`lr_per_copy` and the scalar replaced where it is used.
