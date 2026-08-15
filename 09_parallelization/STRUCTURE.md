# 09_parallelization — folder structure and design decisions

Task: GPU-parallel PointMaze environments + batched multi-seed PPO+RND training, benchmarked and
profiled on the serval05 H100. Full task statement: `efficiency_improvement_user_prompt.md`.
Working method: the experiment loop of `reference_repo/autoresearch` (baseline → one change →
measure → keep or revert → log), one `progress_and_changes.md` per subtask.

## Folder layout

```
09_parallelization/
├── efficiency_improvement_user_prompt.md   # the task statement (review frequently)
├── STRUCTURE.md                            # this file
├── PROGRESS.md                             # top-level state ledger: what is done, what is next
├── reference_repo/                         # local clones, read-only references
│   ├── autoresearch/                       # karpathy/autoresearch — the working method
│   ├── nanoGPT/                            # performance tricks reference
│   ├── nanochat/                           # performance tricks reference
│   └── cleanrl/                            # ppo_rnd_envpool.py — the PPO+RND base
├── parallel_agents_skill/                  # unzipped /p/rlprojects/RLforOR/parallel-agents-skill.zip, adapted
├── locks/                                  # H100 single-user lock protocol (README + wrapper script)
├── pointmaze/                                    # ── Module 1: the environment ──
│   ├── common/                             # shared ground truth: maze maps, physics spec,
│   │                                       #   reference-trajectory fixtures from the MuJoCo env,
│   │                                       #   correctness checks every implementation must pass
│   ├── torch_env/                          # (1) PyTorch batched env  + progress_and_changes.md
│   ├── cuda_env/                           # (2) fused CUDA-kernel env + progress_and_changes.md
│   └── jax_env/                            # (3) JAX env               + progress_and_changes.md
├── ppo/                                    # ── Module 2: batched multi-copy PPO+RND ──
│   ├── torch_ppo/                          # PyTorch module + progress_and_changes.md
│   └── jax_ppo/                            # JAX module     + progress_and_changes.md
├── e2e/                                    # ── Module 3: end-to-end env+training ──
│   ├── torch_e2e/                          # best env + torch PPO + progress_and_changes.md
│   └── jax_e2e/                            # best env + jax PPO   + progress_and_changes.md
├── benchmarks/                             # benchmark harness scripts + raw result JSONs
│   └── results/                            # one JSON per benchmark run (machine-readable)
├── train_runs/                             # final deliverable training runs (8/16/32/64/128 copies)
└── report/                                 # the one unified final report
    └── <YYYY-MM-DD>-<name>/
```

## Design decisions

1. **Hardware**: everything measured on serval05 (direct ssh, 1x H100 NVL 95 GB, 128 CPUs,
   503 GB RAM). Only one GPU job at a time — every GPU-touching command runs through the lock
   wrapper in `locks/` (an flock on a serval05 local file), so two agents or sessions cannot
   overlap on the GPU.
2. **Environments (python)**: NFS makes python imports from `/p` slow on serval05, so the working
   virtual envs live on serval05 LOCAL disk (`/localtmp/sl5nw/venvs/`): `rnd09_torch` (torch cu12x
   + CUDA 12 toolchain for compiling the fused kernel) and `rnd09_jax` (jax[cuda12]) — separate
   envs because torch and jax pin conflicting CUDA library wheels. Code stays in this shared repo;
   only the interpreters are local. This deliberately deviates from the shared-env rule: these
   envs serve single-machine profiling on a non-Slurm box, no collaborator submits jobs with them.
   Exact package lists recorded in `pointmaze/common/serval05_envs.md` once built.
3. **Environment semantics (ground truth)**: the batched envs reimplement Gymnasium-Robotics
   `PointMaze_{UMaze,Open,Medium,Large}-v3` — a force-actuated point mass on a grid maze
   (MuJoCo). The exact physics constants (timestep, frame_skip, mass, damping, actuator gear,
   velocity clip, wall collision) are extracted from the installed gymnasium-robotics source into
   `pointmaze/common/physics_spec.md`, and every implementation must pass the correctness checks in
   `pointmaze/common/` (same-trajectory comparison against the MuJoCo env within tolerance, wall
   containment, reward/termination agreement). Wrapper semantics reproduced from
   `07_reconstruction/src/rnd_exploration/envs/`: fixed start/goal cells, reward shift −1,
   terminate at goal distance ≤ 0.45, velocity clip ±5, optional per-cell visit counts.
4. **Batching axes**: every env carries TWO batch axes — `n_copies` (independent training copies,
   the user's knob: independent seeds, each with its own networks) × `n_envs` (parallel envs per
   copy). All state lives on GPU; `step` takes and returns GPU tensors only.
5. **PPO+RND base**: `reference_repo/cleanrl/cleanrl/ppo_rnd_envpool.py` (already used by
   `08_cleanrl_ppo_rnd/`), adapted to continuous actions (Gaussian policy, as in cleanrl's
   `ppo_continuous_action.py`) because PointMaze actions are Box(−1,1,(2,)). Two update styles,
   built as two SEPARATE fused functions (no `if` inside the hot loop): (a) one full-batch update
   then discard; (b) 4 epochs × 4 shuffled minibatches of 128.
6. **Logging**: no wandb, no tensorboard. Sparse prints (~20 min cadence) + a small final JSON
   per run.
7. **Benchmark protocol**: every number in the report comes from a JSON in `benchmarks/results/`
   written by a script in `benchmarks/`, run under the GPU lock, with warmup iterations excluded
   and the measurement repeated; the JSON records git hash, env name, n_copies, n_envs, timings.
8. **The extra step** (`/p/rlprojects/RLforOR/inventory_management/joint_replenishment/efficiency`)
   is NOT read until Modules 1–3 have finished their first optimization loops, per the task
   statement; then its techniques are diffed against ours and the modules improved again.

## Progress files

- `PROGRESS.md` (top level): the single resume point — phase checklist, current state, next action.
- One `progress_and_changes.md` per subtask folder (torch_env, cuda_env, jax_env, torch_ppo,
  jax_ppo, torch_e2e, jax_e2e): autoresearch-style ledger — one row per attempted change with
  measured before/after and keep/revert.
