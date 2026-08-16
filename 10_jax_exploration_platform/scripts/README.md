# scripts

Command-line entry points for the platform.

| script | what it does |
|---|---|
| `create_platform_jax_env.sh` | builds the shared environment `/p/rlprojects/RND/.venvs/platform_jax` — python 3.12 with `jax[cuda12]` 0.11.0 and numpy 2.5.2, `--copy` and group-readable. Run once; it is kept here so the environment can be rebuilt exactly |
| `run_training.py` | runs one training configuration and writes a run folder under `runs/` (the folder layout is described in `../runs/README.md`) |

The environment registry is `/p/rlprojects/RND/.venvs/ENVS.md`, and `platform_jax`'s own
`PROVENANCE.md` sits inside the environment directory. Neither is in git — the whole `.venvs/`
directory is ignored by this repository — so the build script here is the tracked record of how the
environment was made.

Every python invocation sets `PYTHONNOUSERSITE=1`, so a package in `~/.local` cannot shadow the
environment's own. Graphics-card work on serval05 goes through the lock, one command at a time:

```
bash /p/rlprojects/RND/09_parallelization/locks/gpu_run.sh "<command>"
```
