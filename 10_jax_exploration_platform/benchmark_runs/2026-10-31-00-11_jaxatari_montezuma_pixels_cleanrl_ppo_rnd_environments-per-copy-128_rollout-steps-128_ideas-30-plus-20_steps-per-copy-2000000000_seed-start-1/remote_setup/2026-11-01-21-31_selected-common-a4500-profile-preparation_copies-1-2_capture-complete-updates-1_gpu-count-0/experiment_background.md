## Purpose

Prepare two selected-baseline kernel profile packets from the independently accepted one- and two-copy A4500 confirmations. This step reads saved evidence and stages metadata; it does not execute numerical work or submit jobs.

## Key hyperparameters

Each future capture contains one complete update after fifty original normalization-warmup rollouts, with 128 environments per policy, 128 rollout steps, four epochs, four minibatches and the full two billion transition schedule. Preserve seeds one and one through two, respectively.

## Code and config changes

Use the published optional common-confirmation helper closure, with unchanged numerical source and capture function. The only selected environment changes are cached room backgrounds and rendering the last two frames.

## Git state

Helper source: fdb35f886ad01c6d4ed387dc55b172d0221a9853. Numerical source: d28e6735e79788f618acb0ebfde8609db1f776af.
