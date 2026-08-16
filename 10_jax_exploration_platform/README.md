# 10_jax_exploration_platform

The training platform built on the JAX infrastructure of `09_parallelization`. `09_parallelization`
is finished and frozen at the git tag `jax-rnd-baseline-v0.1.0`; this folder is where the training
work continues. Versions are git tags (`jax-platform-v0.1.0`, ...) — never a `_v2` folder, and never
a second copy of a file under a faster-sounding name. One accepted implementation per family; git
holds the history.

Where things live:

| folder | what is in it |
|---|---|
| `ORIGIN.md` | which repository, branch, commit and tag the copied code came from, file by file, and what was deliberately not copied |
| `src/exploration_platform/` | the accepted implementations: environments, agents, bonuses, the training composition, evaluation metrics |
| `configs/` | component configuration files (environment, agent, bonus, runtime) and whole experiments that name a combination of them |
| `tests/` | the golden-parity gate against `09_parallelization`, agreement with `07_reconstruction`, and per-component tests |
| `benchmarks/` | the measurement harness and the per-component speed measurements |
| `research/` | measurement rounds and their write-ups; never production code |
| `device/` | what each graphics card class was measured to do, and the copy count it can hold |
| `runs/` | training run folders, `YYYY-MM-DD-HH-MM_<slug>` |
| `benchmark_runs/`, `reports/`, `scripts/` | benchmark campaign outputs, written reports, command-line entry points |
| `development_document/` | the platform's LaTeX writeup (a later stage creates it) |

Environment: `/p/rlprojects/RND/.venvs/platform_jax` (python 3.12, `jax[cuda12]` 0.11.0), registered
as canonical for this folder in `/p/rlprojects/RND/.venvs/ENVS.md`. Every python invocation sets
`PYTHONNOUSERSITE=1`. Graphics-card work on serval05 goes through the lock:
`bash /p/rlprojects/RND/09_parallelization/locks/gpu_run.sh "<command>"`.
