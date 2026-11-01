## Purpose

Measure an explicitly supplied combined configuration or one isolated policy-copy count. The producer chooses no hypotheses or configurations and submits no jobs.

## Key hyperparameters

Original CleanRL policy optimization with Random Network Distillation: 128 pixel environments per policy, 128 rollout steps, four epochs, four minibatches, fifty random warmup rollouts. 7 timing rounds contain 10 complete updates each (requested minimum 10). The correctness-only mode performs the supplied-input and complete-iteration gates without timing rounds. Neural and optimizer arithmetic uses float32; running statistics use float64; matrix precision is highest.

## Code and config changes

The exact frozen numerical core and Executor are imported after source/runtime/assets validation and card admission. Packing holds only one variant in a fresh process and samples actual device used/free/total memory as well as process memory. No checkpoint is saved.

## Git state

Numerical source: `d28e6735e79788f618acb0ebfde8609db1f776af`. Helper source: `e5afbf6b51a3d1666b46e71c0631a45d47080cd8`. Original source/helper hashes, full configurations and input receipts are recorded in measurement_plan.json.

## Separate target qualification

This explicit same-original path runs every unchanged learner/environment pytest gate, all original renderer fixtures, fifty full normalization warmup rollouts and one full reference update. It requires no accepted common baseline, completed-fifty proof or existing target admission. Its distinct target_qualification_result.json grants zero scientific, throughput, packing, case or idea credit. Controlled agent transitions and updates are counted separately from unquantified synthetic/partial test-fixture work; no whole-job transition total is claimed. Actual independent review remains required before any subsequent use.
