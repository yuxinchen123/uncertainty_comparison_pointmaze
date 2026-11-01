## Purpose

Verify one matching Slurm H100 NVL site before any separately reviewed worker placement. This site validation admits no workers and claims no campaign cases.

## Key hyperparameters

The original one-policy reference uses 128 environments, 128 rollout steps, four epochs, four minibatches, fifty warmup rollouts and one complete training update. The original full learner and renderer gates run first in fresh sequential children.

## Code and config changes

Keep the frozen numerical source, highest float32 neural arithmetic, float64 statistics, deterministic operations and original allocator settings. Require the reference model, memory, backend and library identities, and the selected qualification driver. Resolve the allocated physical UUID inside srun and bind eight CPUs. Submit held. The parent must write and fsync the exact UVA job ID and submission receipt, then run scontrol release for only that saved ID. This helper never submits or releases a job.

The original direct-server reference driver 590.48.01 is retained. The explicit Slurm profile separately qualifies target driver 580.159.04. It trusts configured scheduler task/cgroup device isolation; live BPF device rules are not independently inspected, and a numeric Slurm-to-NVML/minor/PCI mapping is not reconstructed. Every parent and child must provide fresh own allocation and task evidence, and the reference probe must bind its actual JAX CUDA ordinal to the selected physical UUID and PCI address.

## Git state

Numerical source: d28e6735e79788f618acb0ebfde8609db1f776af. Metadata helpers: 3c8b28402a78535f05f00b23905c9249f5dbdf99. Exact input identities are in site_validation_plan.json. No submission or current admission change is performed by preparation.
