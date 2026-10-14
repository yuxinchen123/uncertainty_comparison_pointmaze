## Purpose

Compare ten execution changes after the intrinsic reset correction. Each idea has a 580-second limit including compilation.

## Key hyperparameters

Random network distillation, full batch, rollout 128, four environments per copy, coverage enabled, 512 copies except the two copy-count tests. Three eight-second timing blocks.

## Code and config changes

Test packed parameters, rollout unroll 1/4/8/32/64, 128/256 copies, and 8/32 iterations per dispatch. Baseline unroll is 16. Preserve rewards, return definitions, networks and update schedule.

## Git state

Commit recorded in manifest.json at launch. Other sessions have unrelated changes in 08_cleanrl_ppo_rnd and RLeXplore.
