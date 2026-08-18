## Purpose

The Montezuma's Revenge family's first live training batch (development document train run
3.1), and the platform's first discrete-action training at scale: show the JAXAtari-fused game
training end to end — game logic, categorical actor, bonus and update in one compiled
program — and read whether the distillation bonus moves a smoke-scale agent on the game the
original random-network-distillation paper was built for. Two arms at one setting each:
`rnd_next_state` against `none`. Hypotheses and the what-is-swept table:
`development_document/platform_development_document.tex`, section 3.1.

## Key hyperparameters

- environment: `montezuma_oc4500_sticky@1` (spec:
  `src/exploration_platform/envs/atari_montezuma/spec.md`): object-centric 774-d observation,
  18 discrete actions, sticky actions 0.25, frame skip 4, cap 4,500 env steps, clipped reward
- arms: `rnd_next_state` (unit-rnd_next_state), `none` (unit-none); one unit each, run one
  after the other on the same card
- copies: 1,024 per arm, `n_envs` 1, rollout 128 steps, update style `full_batch`
- iterations: 16,000 (2,048,000 env steps per copy; 2.1 B per arm in total — still about
  three orders of magnitude below the original RND campaign's budget, deliberately)
- learning rate 3e-4 (single value, no sweep); intrinsic weight 1 (`int_coef`), `ext_coef` 2
- recording: one episode window per 400 iterations, 40 windows. Episodes end on the game's
  own terms at variable lengths, so no fixed episode clock exists and no window is
  phase-blocked; the run compares the two arms' window sums at equal step budgets
- coverage tracked: rooms visited, over the 24 playable rooms
- seeds: `base_seed` 0, `run_seed` 0, every copy its own key stream (no sweep pairing)

## Code and config changes

Runs on branch `fused-atari` (worktree `/p/rlprojects/RND_worktrees/fused-atari`), which adds
the Montezuma adapter over JAXAtari and the categorical-actor path to the agent; nothing was
changed for this run beyond that branch's committed state. Launched through
`scripts/run_training.py --env montezuma` with the driver
`/p/rlprojects/RND_worktrees/.scratch/run_two_units.sh` (serial units; serval05 is held by this
session with one job at a time). Hardware: the H100 NVL of serval05 (recorded per unit in the
shard's job record). Throughput at this shape, measured before launch:
`benchmarks/end_to_end/results/2026-08-18_montezuma_train_h100nvl_serval05.json`.

## Git state

Branch `fused-atari`, commit `0e982e7cf01acde5a39399833d17b71598b6883b`, working tree clean at launch.
