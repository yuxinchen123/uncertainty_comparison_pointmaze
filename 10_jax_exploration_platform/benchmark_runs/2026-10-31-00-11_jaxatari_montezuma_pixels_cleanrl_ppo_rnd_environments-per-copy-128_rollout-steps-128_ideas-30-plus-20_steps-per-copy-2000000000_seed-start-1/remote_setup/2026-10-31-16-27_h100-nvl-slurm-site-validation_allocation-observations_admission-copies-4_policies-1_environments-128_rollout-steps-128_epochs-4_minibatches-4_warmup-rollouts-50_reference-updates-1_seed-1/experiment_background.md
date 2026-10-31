## Purpose

Verify one matching Slurm H100 NVL site before any separately reviewed worker placement. This site validation admits no workers and claims no campaign cases.

## Key hyperparameters

The original one-policy reference uses 128 environments, 128 rollout steps, four epochs, four minibatches, fifty warmup rollouts and one complete training update. The original full learner and renderer gates run first in fresh sequential children.

## Code and config changes

Keep the frozen numerical source, highest float32 neural arithmetic, float64 statistics, deterministic operations and original allocator settings. Require the reference model, memory, driver, backend and library identities. Resolve the allocated physical UUID inside srun and bind eight CPUs. Submit held. The parent must write and fsync the exact UVA job ID and submission receipt, then run scontrol release for only that saved ID. This helper never submits or releases a job.

## Git state

Numerical source: d28e6735e79788f618acb0ebfde8609db1f776af. Metadata helpers: b535e72461e52a035dbb89f8bccb6343f1f880af. Exact input identities are in site_validation_plan.json. No submission or current admission change is performed by preparation.
