## Purpose

Test whether uneven initial state bonuses delay goal discovery while exact inverse-square-root count decay is fixed. Compare deterministic straight and snake corridors with tabular and neural policies.

## Key hyperparameters

One environment per independent policy. Four actions. Four update epochs and four minibatches. Raw bonuses primary; selected CleanRL normalization sensitivity. Development budgets 262144 and 1048576 transitions; confirmation budgets 4194304 and 16777216. Research deadline 96 hours after performance measurements, with final 24 hours reserved for confirmation.

## Code and config changes

New grid_initialization trainer and grid_initialization_campaign scripts. Thirty performance ideas: twenty initial ideas followed by ten based on measured timings. All measurements have an execution limit below 600 seconds. GPU batches are sized by measured peak memory and speed.

## Git state

Current source commit: `6912360f2cf6f710e5eb6785235159aeefbb4339`. Source changes and verification attempts are recorded in `infrastructure_events.jsonl`; each submitted job runs an immutable snapshot. Existing unrelated working-tree changes are excluded from this study.
