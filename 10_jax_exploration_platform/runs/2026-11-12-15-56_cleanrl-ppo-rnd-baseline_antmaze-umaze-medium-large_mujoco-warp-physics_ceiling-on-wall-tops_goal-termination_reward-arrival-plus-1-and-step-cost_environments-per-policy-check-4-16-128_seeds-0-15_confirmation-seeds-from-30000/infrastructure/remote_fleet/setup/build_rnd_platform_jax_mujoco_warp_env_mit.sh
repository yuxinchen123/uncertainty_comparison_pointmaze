#!/bin/bash
#SBATCH --job-name=rnd-maze-warp-environment-build
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --open-mode=append
# Build the PointMaze and AntMaze study's environment on MIT Engaging, with JAXAtari in it, inside a job on a
# processor-only compute node (the login node is a 4-core, 10 GiB slice). The commands are those of the Purdue
# scripts build_rnd_platform_jax_mujoco_warp_env.sh and add_jaxatari.sh (setup folder of the run
# 2026-11-12-02-24_cleanrl-ppo-rnd-baseline_pointmaze-...) and of build_rnd_platform_jax_mujoco_warp_env_aces.sh
# beside this file, with the same requirement and constraint files, which must lie beside this script. Only the
# paths differ. Downloads and unpacks wheels only: nothing is compiled. The last step is the processor-mode import
# check (imports only, no training, no device).
#   sbatch --parsable -o <setup folder>/build_%j.out build_rnd_platform_jax_mujoco_warp_env_mit.sh <setup folder>
# The grid study's environment rnd_platform_jax beside it is left as it is.
set -euo pipefail
HERE=${1:?give the folder that holds this script and its requirement files}
SCRATCH=/orcd/scratch/orcd/008/shuzel
ENV_PATH=$SCRATCH/envs/rnd_platform_jax_mujoco_warp
# The conda program that built the grid study's environment on this cluster (miniforge 26.7.2-0).
CONDA=/orcd/software/core/001/pkg/miniforge/26.7.2-0/bin/conda
export PYTHONNOUSERSITE=1
# Module-provided CUDA libraries must not be seen by pip's own Python.
unset LD_LIBRARY_PATH
echo "cluster mit, job ${SLURM_JOB_ID:-none}, host $(hostname), start $(date -Is), git $(git --version)"
[ ! -e "$ENV_PATH" ] || { echo "$ENV_PATH exists already; it is not rebuilt in place"; exit 1; }
[ -x "$CONDA" ] || { echo "no conda at $CONDA"; exit 1; }
# Downloaded conda packages and pip's temporary files (about 3 GB of wheels) go to scratch: the home folder is
# 90% full (179 of 200 GB on 2026-11-13).
export CONDA_PKGS_DIRS=$SCRATCH/conda_pkgs
export TMPDIR=$SCRATCH/tmp_build_rnd_platform_jax_mujoco_warp_${SLURM_JOB_ID:-manual}
export PIP_CACHE_DIR=$TMPDIR/pip_cache
mkdir -p "$SCRATCH/envs" "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

# A self-contained Python 3.12.13 from conda-forge (--copy: no links into the package cache), as on UVA.
"$CONDA" create --copy --yes -p "$ENV_PATH" --override-channels -c conda-forge python=3.12.13
# The pinned packages, wheels only; every dependency is held to the version of UVA's environment.
"$ENV_PATH/bin/python" -m pip install --no-cache-dir --only-binary=:all: \
  -r "$HERE/requirements_rnd_platform_jax_mujoco_warp.txt" -c "$HERE/constraints_uva_platform_jax_pip_list.txt"
echo "pinned packages installed, $(date -Is)"
# JAXAtari at UVA's source commit: the trainer imports its Montezuma environment at its start, for every
# environment family. Pure Python (a py3-none-any wheel is built from the source); nothing is compiled.
"$ENV_PATH/bin/python" -m pip install --no-cache-dir --only-binary=:all: --no-binary=jaxatari \
  -c "$HERE/constraints_uva_platform_jax_pip_list.txt" \
  "jaxatari @ git+https://github.com/k4ntz/JAXAtari@035fc46bf149220a2f462c864873707ced334a4c"
"$ENV_PATH/bin/python" -m pip list --format=freeze > "$ENV_PATH.pip_freeze.txt"
"$ENV_PATH/bin/python" -m pip check
chmod -R g+rX "$ENV_PATH"
# The installed files' record, to compare with UVA's copy of the same commit.
sha256sum "$ENV_PATH"/lib/python3.12/site-packages/jaxatari-0.1.0.dist-info/RECORD
echo "built $ENV_PATH with $(wc -l < "$ENV_PATH.pip_freeze.txt") packages and $(find "$ENV_PATH" -xdev | wc -l) files, end $(date -Is)"
echo BUILD_COMPLETE

# Imports only, on this processor-only node: no device is used (jax is told to use the processor).
JAX_PLATFORMS=cpu "$ENV_PATH/bin/python" -I "$HERE/import_check_rnd_platform_jax_mujoco_warp.py" processor
echo "end $(date -Is)"
