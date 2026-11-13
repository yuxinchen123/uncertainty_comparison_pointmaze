#!/bin/bash
# Build the PointMaze and AntMaze study's environment on TAMU ACES, with JAXAtari in it. The commands are those of
# the Purdue scripts build_rnd_platform_jax_mujoco_warp_env.sh and add_jaxatari.sh (setup folder of the run
# 2026-11-12-02-24_cleanrl-ppo-rnd-baseline_pointmaze-...), with the same requirement and constraint files, which
# must lie beside this script. Only the paths differ. Downloads and unpacks wheels only: nothing is compiled, and no
# numerical library is imported here. It runs on the login node, which reaches the package servers without a proxy.
#   bash build_rnd_platform_jax_mujoco_warp_env_aces.sh
# The environment goes to the project's folder, not the user's: on 2026-11-12 the user's scratch folder held
# 202,620 of the 250,000 files it may hold, and this environment has about 36,700 files (count of Gilbreth's copy);
# the project's folder held 106,482 of 500,000. The grid study's rnd_platform_jax is left as it is.
set -euo pipefail
PROJECT=/scratch/group/p.cis261532.000
ENV_PATH=$PROJECT/envs/rnd_platform_jax_mujoco_warp
OLD=/scratch/user/u.sl359075/envs/rnd_platform_jax
HERE=$(dirname "$(readlink -f "$0")")
export PYTHONNOUSERSITE=1
# Module-provided CUDA libraries must not be seen by pip's own Python.
unset LD_LIBRARY_PATH
echo "cluster aces, host $(hostname), start $(date -Is)"
[ ! -e "$ENV_PATH" ] || { echo "$ENV_PATH exists already; it is not rebuilt in place"; exit 1; }
# The conda program that built the grid study's environment on this cluster.
CONDA=$(grep -m1 '^# cmd:' "$OLD/conda-meta/history" | awk '{print $3}')
[ -x "$CONDA" ] || { echo "no conda at $CONDA"; exit 1; }
# Downloaded conda packages and pip's temporary files (about 3 GB of wheels) also go to the project's folder,
# so they count neither against the user's file limit nor against the login node's /tmp.
export CONDA_PKGS_DIRS=$PROJECT/conda_pkgs
export TMPDIR=$PROJECT/tmp_build_rnd_platform_jax_mujoco_warp
mkdir -p "$PROJECT/envs" "$TMPDIR"
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
