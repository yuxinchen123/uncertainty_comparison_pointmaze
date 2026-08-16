# Deferred experiments — the code for changes that must not be applied to this sweep

Everything here is a real optimization that **changes numbers the project has already recorded**.
The user's constraint for this round is that run 5, run 6 and the 1M sweep stay valid, so none of it
is wired into `train.py`, `train4m.py` or `rnd.py`, and none of it is reachable from
`analysis/code/patched_entry.py` (the measurement entry point) — wiring it in is deliberately the
first step of adopting it, and that step belongs to a sweep that starts a fresh baseline.

The reasoning and the measured or estimated gain for each item is in `../analysis/future_changes.md`;
this folder holds what you would actually run.

| item | code | gain | why it is deferred |
|---|---|---|---|
| **Fused RND forwards** | `fused_rnd/fused_rnd.py` (+ its README) | ~3-5% | `RND.update()` moves the observation statistics *before* its own forward, so the bonus forward and the loss forward see different normalizations. Fusing forces one of them onto the other, which changes the trained predictor |
| **torch.compile the SAC update** | `torch_compile/torch_compile_patch.py` | ~1% (estimate; never measured on this workload) | compiled kernels may reassociate float operations |
| **Batched SAC updates** | no code needed: `--sac_train_freq=N --sac_gradient_steps=N`, both already in `Config` | +1.0% measured 2026-06-25 | the policy collecting the data goes stale by up to N-1 steps |
| **sbx / JAX SAC port** | prototype already on disk: `../../2026-06-25-run-profiling/experiments/2026-06-25-21-40-jax-ab/` | **×1.66 measured** on this exact task | a different SAC implementation, and it *under-trains* versus SB3 (run 4 reached 0.39-0.52× run 2's reward at 500K) — adopting it means re-running every baseline it is compared against |
| **Lean CleanRL-style trainer** | prototype already on disk: `../../2026-06-25-run-profiling/experiments/2026-06-25-19-55-fast-trainer/fast_sac_rnd.py` | +6.8% (RND), −11% (gt) | different rollout/plumbing path, so different transition ordering |
| **Drop the per-step intrinsic logging wrapper** | no code needed: remove `ComputeIntrinsicRewardWrapper` from `wrap_for_rollout` | one batch-1 forward per env step | `train/intrinsic_reward` is a recorded metric; dropping it removes a column the analysis plots |

Two items in `future_changes.md` have no code here on purpose: **GPU** (`--device=cuda` already
exists; it is a throughput *loss* for a sweep that packs 46 CPU workers per node) and **reduced
precision** (nothing to prototype until there is a reason to want it).

## How to try any of them later

Each patch module exposes a `patch_*()` function. Point `PYTHONPATH` at its folder and call it from
a copy of `patched_entry.py` — then run `analysis/code/run_conditions.py` exactly as this campaign
did. Expect `DIFFERS at record...` from the record check; for these items that is the correct
outcome, and the question becomes whether the new numbers are *better science*, which is a
convergence study, not a benchmark.
