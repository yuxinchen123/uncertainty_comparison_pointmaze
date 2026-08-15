# 09_parallelization — top-level progress ledger

Single resume point for this multi-day task. Task statement:
`efficiency_improvement_user_prompt.md`. Layout: `STRUCTURE.md`. Update this file whenever a
phase advances; per-subtask experiment logs live in each subtask's `progress_and_changes.md`.

## Phase checklist

- [x] Phase 0 — scaffolding
  - [x] folder structure designed (`STRUCTURE.md`)
  - [x] reference repos cloned (autoresearch, nanoGPT, nanochat, cleanrl)
  - [x] parallel-agents skill unzipped (`parallel_agents_skill/`)
  - [x] MuJoCo dynamics probed; closed-form update rule verified EXACT (`pointmaze/common/physics_spec.md`)
  - [x] reference fixtures generated (`pointmaze/common/fixtures/`, 12 cases)
  - [x] serval05 local envs built (rnd09_torch, rnd09_jax on /localtmp/sl5nw)
  - [x] H100 lock protocol in place (`locks/`)
- [ ] Phase 1 — Module 1: batched GPU envs (each with its own autoresearch loop)
  - [x] pointmaze/common: physics spec (probe-verified EXACT incl. the MuJoCo contact law),
        fixtures, one-step + rollout checker, frozen cross-impl reset RNG
  - [x] torch_env: exact + fused (4.03e9 env-steps/s @1M envs compiled)
  - [ ] cuda_env: fork agent building it (fused single kernel)
  - [x] jax_env: exact (bit-identical resets to torch); 6.87e9 (jit) / 8.39e9 (scan) @1M
  - [ ] throughput curves: n_envs sweep on H100 (log-scale line), step time per env
- [ ] Phase 2 — Module 2: batched multi-copy PPO+RND (torch + jax)
  - [x] algorithm spec written (ppo/research/, incl. a real cleanrl bug found: intrinsic
        filter iterates the wrong axis; and a spec §11 gradient correction)
  - [x] torch trainer: correct (copy-isolation bitwise tests) + optimized 777 -> 45.6 ms/iter
        at C=128 (whole-rollout + whole-update CUDA-graph capture, both bitwise-verified)
  - [x] jax trainer (fork agent): all tests pass, cross-framework agreement 8.6e-7,
        74 iter/s (style A) / 50 iter/s (style B) at C=128
  - [ ] remaining optimization loop + learning sanity run
- [ ] Phase 3 — Module 3: end-to-end (each env variant x each trainer, fused where possible)
- [ ] Phase 4 — extra step: read /p/rlprojects/RLforOR/inventory_management/joint_replenishment/efficiency,
      diff techniques, improve modules 1-3 again (do NOT read it before this phase)
- [ ] Phase 5 — final deliverable: 8/16/32/64/128-copy training runs (pytorch), all tables,
      profiling breakdowns, throughput plots, unified report in `report/<date>-<name>/`

## State notes (newest first)

- 2026-08-15 ~00:30 session start. serval05 idle (H100 NVL 95GB). Dynamics rule verified exact:
  v' = (m*clip(v,±5) + h*g*clip(a,±1))/(m+h*d), q' = q + h*v'; hard-wall clamp within 1.4 mm
  of MuJoCo soft contact. Fixtures written. Next: serval05 env bootstrap + lock + research
  fan-out + torch env v0.
