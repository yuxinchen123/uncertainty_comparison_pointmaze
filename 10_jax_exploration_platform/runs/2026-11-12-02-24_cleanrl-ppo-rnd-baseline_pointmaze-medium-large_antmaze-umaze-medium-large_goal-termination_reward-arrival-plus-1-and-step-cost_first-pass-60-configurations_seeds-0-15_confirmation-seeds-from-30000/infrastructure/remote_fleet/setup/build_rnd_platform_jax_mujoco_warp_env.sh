#!/bin/bash
# Build the PointMaze and AntMaze study's environment on a Purdue cluster: a new environment beside the grid
# study's rnd_platform_jax, which is left as it is. Downloads and unpacks wheels only: nothing is compiled, and
# no numerical library is imported here (the worker job checks on its card that jax uses it).
#   bash build_rnd_platform_jax_mujoco_warp_env.sh <cluster>      (cluster: gautschi or gilbreth)
set -euo pipefail
CLUSTER=$1
BASE=/scratch/$CLUSTER/liu4869
ENV_PATH=$BASE/envs/rnd_platform_jax_mujoco_warp
OLD=$BASE/envs/rnd_platform_jax
HERE=$(dirname "$(readlink -f "$0")")
export PYTHONNOUSERSITE=1
# Module-provided CUDA libraries must not be seen by pip's own Python.
unset LD_LIBRARY_PATH
echo "cluster $CLUSTER, host $(hostname), start $(date -Is)"
[ ! -e "$ENV_PATH" ] || { echo "$ENV_PATH exists already; it is not rebuilt in place"; exit 1; }
# The conda program that built the grid study's environment on this cluster.
CONDA=$(grep -m1 '^# cmd:' "$OLD/conda-meta/history" | awk '{print $3}')
[ -x "$CONDA" ] || { echo "no conda at $CONDA"; exit 1; }
export CONDA_PKGS_DIRS=$BASE/conda_pkgs

# A self-contained Python 3.12.13 from conda-forge (--copy: no links into the package cache), as on UVA.
"$CONDA" create --copy --yes -p "$ENV_PATH" --override-channels -c conda-forge python=3.12.13
# The pinned packages, wheels only; every dependency is held to the version of UVA's environment.
"$ENV_PATH/bin/python" -m pip install --no-cache-dir --only-binary=:all: \
  -r "$HERE/requirements_rnd_platform_jax_mujoco_warp.txt" -c "$HERE/constraints_uva_platform_jax_pip_list.txt"
"$ENV_PATH/bin/python" -m pip list --format=freeze > "$ENV_PATH.pip_freeze.txt"
"$ENV_PATH/bin/python" -m pip check
chmod -R g+rX "$ENV_PATH"
echo "built $ENV_PATH with $(wc -l < "$ENV_PATH.pip_freeze.txt") packages, end $(date -Is)"
echo BUILD_COMPLETE
