# serval05 local working environments (built 2026-08-15 by bootstrap_serval05.sh)

Local disk, because python imports from the NFS `/p` mount are very slow on this box.

| env | path | contents |
|---|---|---|
| rnd09_torch | `/localtmp/sl5nw/venvs/rnd09_torch` | Python 3.12, torch 2.13.0+cu130, numpy, ninja, nvidia-cuda-nvcc-cu12 wheels (NOTE: torch is a CUDA-13.0 build — install the cu13 nvcc wheel before compiling the fused kernel) |
| rnd09_jax | `/localtmp/sl5nw/venvs/rnd09_jax` | Python 3.12, jax 0.11.0 with CUDA 12 pip wheels (CudaDevice(0) verified on the H100 NVL) |

- uv binary: `/localtmp/sl5nw/uv/uv`; caches under `/localtmp/sl5nw/uv_cache`.
- H100 lock file: `/localtmp/sl5nw/locks/h100.lock` (see `../../locks/README.md`).
- Re-running `code/bootstrap_serval05.sh` is safe (idempotent).
- These envs are deliberately NOT in the shared `.venvs` registry: they serve single-machine
  profiling on a non-Slurm box; no collaborator or Slurm job uses them.
