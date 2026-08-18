## Purpose

The AntMaze family's first live training batch (development document train run 2.1): show the
MJX-fused AntMaze training end to end at scale on one card — environment, agent and bonus in
one compiled program — and read whether the distillation bonus drives a quadruped's
exploration at a smoke-scale budget. Two arms at one setting each: `rnd_next_state` against
`none`, on the umaze. Hypotheses and the what-is-swept table:
`development_document/platform_development_document.tex`, section 2.1.

## Key hyperparameters

- environment: `antmaze_umaze_cont700_nonoise@1` (spec: `src/exploration_platform/envs/antmaze/spec.md`)
- arms: `rnd_next_state` (unit-rnd_next_state), `none` (unit-none); one unit each, run one
  after the other on the same card
- copies: 1,024 per arm, `n_envs` 1, rollout 128 steps, update style `full_batch`
- iterations: 2,100 (268,800 env steps per copy; 275 M per arm in total)
- learning rate 3e-4 (single value, no sweep); intrinsic weight 1 (`int_coef`), `ext_coef` 2
- recording: one episode window per 175 iterations (one whole 128-step-rollout /
  700-step-episode clock cycle, so windows are phase-blocked); 12 windows
- coverage tracked (46-cell umaze grid mask: 7 open cells)
- seeds: `base_seed` 0, `run_seed` 0, every copy its own seed stream (no sweep pairing)

## Code and config changes

Runs on branch `fused-antmaze` (worktree `/p/rlprojects/RND_worktrees/fused-antmaze`), which
adds the MJX AntMaze family and makes the trainer environment-generic; nothing was changed for
this run beyond that branch's committed state. Launched through
`scripts/run_training.py --env antmaze-umaze` with the driver
`/p/rlprojects/RND_worktrees/.scratch/run_two_units.sh` (serial units; serval05 is held by this
session with one job at a time). Hardware: the H100 NVL of serval05 (recorded per unit in the
shard's job record). Throughput at this shape, measured before launch:
`benchmarks/end_to_end/results/2026-08-18_antmaze_train_h100nvl_serval05.json`.

## Git state

Branch `fused-antmaze`, commit `31a594bd185ef3b9fff4dadb05552d682f66338b`, working tree clean at launch.
