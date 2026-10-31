## Purpose

Validate the committed synchronized one-versus-eight RTX A4500 producer using the original pixel PPO with Random Network Distillation reference. These are resource and component checks. Both have zero final optimized-scaling credit and add zero research hypotheses.

## Key hyperparameters

One independent policy per card, 128 environments per policy, 128 rollout steps, four epochs, four minibatches, fifty normalization warmups, and seven common-wall rounds of ten complete updates. Single-card seed 1 also appears on the first card of the eight-card group; the remaining cards use distinct seeds 2 through 8. Networks and optimizer use float32, running statistics float64, and highest matrix precision with deterministic operations. Each card has eight OpenMP threads. The deliberate component-check walltime is 40 minutes.

## Code and config changes

The new committed metadata wrapper derives the exact reference from an already completed original case, proves the full Slurm allocation, prepares immutable producer records and records zero final-study credit before numerical startup. The numerical source and gates remain frozen. The one-card job requests 8 CPUs and 64 GiB; the eight-card job requests 64 CPUs and 512 GiB under one node job. The largest relevant owned initialization diagnostic used 24093384 KiB (22.98 GiB), below the 64 GiB per-card host allowance; completed two-policy original isolated capacity used 4729176 KiB. The eight-card job requires the new single-card control to succeed and region diagnostic 6777971 to finish.

## Git state

Numerical source: d28e6735e79788f618acb0ebfde8609db1f776af.

Immutable helper checkout: eebea5219d6cdb6adcea56ade3c71c7dcc73dc09.

The wrapper and its two standard-library tests were committed and pushed before submission. No optimization is selected, no formal gate is modified, and no final policy seed block is reserved.
