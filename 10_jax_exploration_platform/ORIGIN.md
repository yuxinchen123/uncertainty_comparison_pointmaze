# Where this platform's code came from

Times below are Pacific with a `PT` marker; the servers run on Eastern, so machine timestamps were
converted for display.

## Source

| item | value |
|---|---|
| repository | `/p/rlprojects/RND` (remote `git@github.com:yuxinchen123/uncertainty_comparison_pointmaze.git`) |
| branch | `Use-RLexplore-RND` |
| source folder | `09_parallelization/` |
| commit copied from | `4bb74559c8afd8a87d7b5cc69f070dad5899ab88` |
| tag on that commit | `jax-rnd-baseline-v0.1.0` ("Frozen 09_parallelization JAX PPO+RND baseline") |
| copied on | 2026-08-16 16:10 PT |

`09_parallelization/` is finished and frozen at that tag. Nothing in it is edited from here — the
files below were **copied**, not moved, and the originals stay where they are. Version history of
the platform is git tags (`jax-platform-v0.1.0`, ...), never a second folder or a second copy of a
file under a different name.

## What was copied, file by file

| source (under `09_parallelization/`) | destination (under this folder) | changed on copy |
|---|---|---|
| `pointmaze/common/pm_common.py` | `src/exploration_platform/envs/pointmaze/pm_common.py` | nothing — byte identical |
| `pointmaze/jax_env/jax_pointmaze.py` | `src/exploration_platform/envs/pointmaze/jax_pointmaze.py` | one import-path line (`pm_common.py` now sits beside it) |
| `ppo/jax_ppo/jax_ppo_rnd.py` | `src/exploration_platform/agents/ppo/jax_ppo_rnd.py` | two import-path lines (the environment now lives at `../../envs/pointmaze`) |
| `ppo/jax_ppo/SWEEP.md` | `src/exploration_platform/agents/ppo/SWEEP.md` | nothing — byte identical |
| `ppo/jax_ppo/tests/*.py` (6 files) | `tests/agents/` | one import-path line each (two in `test_flat_params_equivalence.py`) |
| `benchmarks/bench_train_jax.py` | `benchmarks/harness/bench_train_jax.py` | one import-path line |
| `benchmarks/profile_jax_phases.py` | `benchmarks/harness/profile_jax_phases.py` | one import-path line |

Nothing else changed at copy time. That mattered: the golden-parity gate
(`tests/golden_09/test_golden_parity.py`) runs the platform's trainer and the original
`09_parallelization/ppo/jax_ppo/jax_ppo_rnd.py` side by side in one process and requires the two to
produce bit-identical outputs and bit-identical state, so any edit beyond an import path would show
up as a test failure.

## What has changed since the copy

`jax_ppo_rnd.py` no longer exists here. It was split into environment / agent / bonus /
composition on 2026-08-16, and the file was deleted rather than kept as a second copy of the same
code. The gate is what makes that safe: it still runs against the frozen single module in
`09_parallelization/`, and it still requires bit-identical results, so the split is checked against
the baseline rather than against itself.

| the single module held | it now lives in |
|---|---|
| `PPOConfig` | `src/exploration_platform/agents/ppo/config.py` |
| the actor and critic, their keyed initialisation, their forwards | `agents/ppo/networks.py` (the keyed draw itself is `exploration_platform/networks.py`, shared with the bonuses) |
| `_losses`, minus the bonus's term | `agents/ppo/losses.py` |
| `_clip_per_copy`, `_adam_step`, both update styles | `agents/ppo/update.py` |
| `RMSState`, `rms_init`, `rms_update` | `exploration_platform/statistics.py` (the agent and the bonus both keep such statistics) |
| the target and predictor networks, `_whiten`, `_rnd_features`, the intrinsic-reward line, the predictor loss | `bonuses/rnd/` |
| `TrainState`, `pack` / `unpack` | `training/state.py` |
| `_iterate_impl`, `_prime_impl` | `training/train_step.py` |
| the sweep's per-copy vectors, `sweep_config` | `training/sweep.py` |
| the `JaxPPORND` constructor's assembly and its two `jax.jit` calls | `training/compose.py` |
| `train`, `prime_obs_rms`, `lr_argument`, `coverage`, `main` | `training/runner.py` (as the class `Runner`) |

Two deliberate differences from the baseline, both visible in the state rather than in the
arithmetic:

- The run key and the iteration counter now live IN the state (`rng`, `step`), so one iteration is
  `iterate(state, learning_rate)` and the driver keeps no counter beside it. Iteration k folds k
  into the run's key exactly as the baseline's driver did, so the keys — and therefore every
  random draw — are the same.
- `obs_norm_init_iters` is now `prime_iterations`. The warm-up rollouts exist so that a bonus which
  needs statistics of the observations has them before it scores anything; that is the bonus's
  business, not the observation normaliser's, and a bonus with no warm-up hook runs none of them.

One knob's behaviour is slightly more consistent than the baseline's, and the gate does not cover
it: with `batch_stats_f32=True` the baseline reduced the warm-up rollouts' statistics in double
precision while reducing the training rollouts' in single, because its priming call did not pass
the flag. The platform passes it in both places. Every gate run uses the default, `False`, where
the two are identical.

## The environment specification this platform starts from

The name used in run manifests: **`pointmaze_large_cont400_nonoise@1`**. It is the default
`EnvConfig` of the copied `pm_common.py`, and the trainer's default `PPOConfig` supplies the
discount:

| knob | value |
|---|---|
| maze map | `large` (12 columns x 9 rows, walls on the border) |
| start cell | `(7, 1)` — row 7, column 1, row 0 at the top; world position (-4.5, -3.0) |
| goal cell | `(1, 10)` — world position (4.5, 3.0) |
| position noise | `0.0` — every episode begins exactly at the start cell's centre and the goal sits exactly at its own centre, so the only randomness left in an episode is the policy's own sampling |
| episode length cap | 400 steps |
| task type | continuing — an episode is never terminated at the goal, only truncated at the cap |
| goal radius | 0.45 m (a step whose distance to the goal is under this earns reward 1) |
| reward shift | 0.0 |
| observation | (x, y, vx, vy); the goal is not observed |
| discount, extrinsic | 0.999 |
| discount, intrinsic | 0.99 |

Position noise was set to zero in `09_parallelization` on 2026-08-16, before this copy. The draw is
still made and still costs the same time; it is multiplied by zero.

## What was deliberately NOT copied

| not copied | why |
|---|---|
| `ppo/torch_ppo/` (the PyTorch trainer and its tests) | the platform trains in JAX; the torch twin's job was to prove the JAX arithmetic, and it did that in 09 |
| `pointmaze/torch_env/`, `pointmaze/cuda_env/` (the CUDA kernel, its build, its tests) | same reason — the JAX stepper is the one the platform runs |
| `benchmarks/results/`, `*/results/`, every result JSON | measurements of 09, not of the platform; the platform measures its own and records them under `benchmark_runs/` and `device/` |
| `report/`, `analysis/`, `PROGRESS.md`, `extra_step_review.md` | write-ups of the finished 09 study |
| `train_runs/`, the 2026-08-16 graphics-card throughput survey folder | old campaigns; the survey's per-class numbers migrate into `device/` as data, not as a copied folder |
| `reference_repo/`, `parallel_agents_skill/` | external content, not versioned in this repository |
| the CUDA environment and its build scripts | no CUDA kernel here |

Two things stayed in 09 that a later stage may want:

- `pointmaze/common/code/check_against_fixtures.py` with `pointmaze/common/fixtures/` — the
  environment's physics check against the Gymnasium-Robotics reference. It is not copied because it
  is a validation script over binary fixtures rather than one of the trainer's tests; when the
  platform wants its own environment gate under `tests/envs/`, copy it together with its fixtures
  and add the matching negations to `.gitignore`.
- `tests/agents/test_forward_fixture.py` was copied, but it cannot run here as it stands: it
  compares against a fixture file written by the PyTorch trainer, which is not part of this folder.
  It is kept so the check is not lost.
