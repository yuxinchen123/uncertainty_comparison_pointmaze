# Sweeping a hyperparameter across copy groups

The trainer normally runs C identical copies that differ only by their seed. A sweep splits
those copies into groups and gives each group its own learning rate, so one training run
answers "which rate is best" instead of one run per rate.

## Using it

```python
from torch_ppo_rnd import PPORND, sweep_config

# 4 learning rates, 128 independent copies each -> 512 copies in one run
cfg = sweep_config(learning_rates=[1e-4, 3e-4, 1e-3, 3e-3], copies_per_rate=128)
trainer = PPORND(cfg, device="cuda")
trainer.train(num_iterations=20000)
```

The two inputs are exactly the two the task asks for: the rates to try, and how many copies
each rate gets. `copies_per_rate` may also be a list, one count per rate, when the groups
should be different sizes:

```python
cfg = sweep_config([1e-4, 3e-4, 1e-3], copies_per_rate=[64, 256, 64])   # 384 copies
```

The training driver takes the same knobs:

```
python train_runs/run_final.py --outdir <run> --sweep-rates 1e-4 3e-4 1e-3 3e-3 \
    --copies-per-rate 128 --iterations 20000
```

Per-group results are in the run record: `group_index` gives every copy's group, and the
history rows carry per-copy rewards and coverage, so a group's curve is the mean over its
copies.

## What the groups share, and what they do not

`sweep_seed_mode="paired"` (the default for a sweep) gives copy k of EVERY group the same
initial weights and the same environment reset noise. Group 0's copy 7 and group 3's copy 7
start from the same place and meet the same maze; the only difference between them is the
learning rate. That makes the comparison between rates paired, which is what a sweep wants —
a rate that wins does so because of the rate, not because it drew easier seeds.

`sweep_seed_mode="distinct"` restores the ordinary behaviour: every copy is its own seed
stream. Use it when the groups are meant to be independent samples rather than a controlled
comparison.

Nothing else is shared. Each copy keeps its own networks, environments, running statistics
and optimizer moments, exactly as in a uniform run — the tests check that changing one
group's rate leaves every other group's parameters bitwise unchanged.

## How it works, and why it is laid out this way

Adam is elementwise, so C copies stacked on the leading axis are already C independent Adam
optimizers. The one thing `torch.optim.Adam` cannot express is a different learning rate per
copy: a rate belongs to a parameter group, and putting each copy in its own group would give
up the batched layout that makes many copies cheap. So a sweep switches to a hand-written
batched Adam using torch's formula (same bias correction, same epsilon placement) with the
learning rate as a `[C]` vector broadcast along the copy axis. It is all tensor operations
with a device-side step count, so it captures into the iteration graph like the rest.

Annealing multiplies every group's rate by one shared factor held in a device tensor, so the
schedule applies to all rates without rebuilding the graph.

The alternative layouts were measured (`benchmarks/bench_sweep.py`); the numbers are in the
report and in `progress_and_changes.md`.

## Sweeping something other than the learning rate

Every per-copy loss term already reduces over the copy axis, so the same mechanism extends to
any scalar knob that enters elementwise — the entropy coefficient, the value coefficient, the
clip range, the intrinsic and extrinsic advantage weights, the gradient-norm limit. Each needs
the same two lines the learning rate needed: a `[C]` tensor built beside `lr_per_copy`, and
the scalar replaced by that tensor where it is used. The learning rate is implemented because
it is what the task asked for; the others are deliberately left until they are wanted, rather
than added speculatively.
