# Experiment background — CleanRL PPO + RND on Atari, train run 1

## Purpose

Run CleanRL's `ppo_rnd_envpool.py` on `MontezumaRevenge-v5` for **30 seeds** at the authors'
own hyperparameters, to answer four questions:

1. **Does the published single-seed number hold across seeds?** CleanRL reports 7,100 on
   MontezumaRevenge from **one** seed, against Burda et al.'s 8,152 from three. Thirty seeds turn
   that single sample into a distribution.
2. **What does the environment auto-reset defect cost?** CleanRL's own documentation records the
   defect and asserts "it does not seem to impact performance" with no evidence behind it. This run
   uses the corrected code; the defect and its measured reach are written up in the development
   document's Section 9.
3. **Where does this RND differ from the project's own RND?** The knob-by-knob comparison is
   Table `tab:cleanrl-vs-project-rnd` and Figure `fig:cleanrl-rnd-pipelines` of the development
   document. This run supplies the Atari-side evidence for that comparison.
4. **What does the workload cost on this cluster?** Profiled separately in
   `../../shuze_experiment/2026-08-05_profilling/`.

This is a **reproduction, not a sweep**: no hyperparameter is varied. The only axis is the seed.

## Key hyperparameters

Every algorithm value is CleanRL's default, unchanged.

| Parameter | Value |
|---|---|
| implementation | `cleanrl/ppo_rnd_envpool.py` @ upstream `fe8d8a03c41a7ef5b523e2e354bd01c363e786bb` |
| environment | `MontezumaRevenge-v5` via envpool 0.6.6, `episodic_life=True`, `reward_clip=True`, `repeat_action_probability=0.25` |
| total timesteps per run | 2,000,000,000 |
| parallel environments | 128 |
| rollout length | 128 steps, batch 16,384 |
| minibatches / update epochs | 4 / 4 |
| learning rate | 1e-4, annealed linearly to 0, shared by the policy and the RND predictor |
| optimizer | Adam, eps 1e-5, gradient clip at global norm 0.5 |
| clip coefficient | 0.1, value loss clipped |
| entropy / value coefficients | 0.001 / 0.5 |
| extrinsic discount | 0.999 |
| intrinsic discount | 0.99, non-episodic |
| GAE lambda | 0.95 |
| intrinsic / extrinsic advantage weights | `int_coef` 1.0, `ext_coef` 2.0 |
| RND update proportion | 0.25 |
| observation-normalizer warm-up | 50 rollouts of a random agent (6,400 steps) |
| seeds | 30, `--seed` 1 through 30 |

## Code and config changes

The upstream file is never edited. The run executes
`08_cleanrl_ppo_rnd/src/ppo_rnd_envpool_shuze.py`, which starts from a copy of it and changes four
things. Each is behind a flag, and every flag's default is the value used here.

1. **The envpool auto-reset defect is corrected** (`--fix_envpool_autoreset`). envpool spends one
   extra `step()` call auto-resetting after an episode ends and discards the action on that call, so
   the row it writes into the rollout buffer describes a transition that never happened. The
   correction carries the advantage through that row and drops it from the update batch. This is the
   only change that alters what the optimizer sees. Measured on `MontezumaRevenge-v5` it removes
   about 1.2% of rows per update, but because the intrinsic return is deliberately non-episodic the
   defect was contaminating the intrinsic advantage of *every preceding row in the rollout*, decaying
   as roughly 0.94 per step.
2. **Weights and Biases and tensorboard are removed.** The run writes one JSON record per run in the
   project's train-run-8.1.2 format, carrying the extrinsic reward, the intrinsic reward, the losses
   and the throughput. Per-episode history is capped at 50,000 entries and then strided by 100, with
   the true episode count and both parameters stored in the record — a 2e9-step Atari run finishes on
   the order of a million episodes and an uncapped list would outgrow the filesystem.
3. **Checkpoint and resume** every 8 hours, one checkpoint per run replaced in place. Required: the
   `gpu` partition kills a job at 4 days and one run needs far longer. 50.2 MB per checkpoint, so all
   30 seeds are checkpointed. Saves the policy, both value heads, the RND predictor **and the frozen
   RND target** (randomly initialized, so it defines the bonus), the Adam state, both normalizers,
   the reward forward filter, the counters, and all four random streams. **envpool's emulator state
   cannot be saved** — a resumed run restarts its environments from a fresh reset, losing at most one
   partial episode per environment copy per resume.
4. **Throughput options that leave the arithmetic unchanged are on**; the options that change it
   (float16 autocast, channels-last, TF32 matrix multiplication, `torch.compile`) are measured in the
   profiling folder and are **off** here, because the point of the run is a faithful reproduction.
   One of these interacts with the bug fix and needed a fix of its own: dropping the auto-reset rows
   makes the batch size differ every iteration, and `cudnn.benchmark` re-autotunes on every new
   shape. Measured on a Quadro RTX 6000: 4,677 steps/s without the fix, 2,178 with the fix and a
   drifting shape, 4,553 with the fix and autotuning off. The run therefore holds the minibatch shape
   constant by reserving a fixed row allowance.

Environment: the shared `/p/rlprojects/RND/.venvs/cleanrl_rnd` (Python 3.10, envpool 0.6.6,
gym 0.23.1, numpy 1.24.4, torch 2.6.0+cu124). Every script exports `PYTHONNOUSERSITE=1`, without
which `/u/sl5nw/.local` shadows the environment with a torch build that cannot run Pascal GPUs.

## Git state

Commit: recorded at submission time, before the first job is queued, per the shared
commit-before-submit rule.
