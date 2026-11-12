#!/bin/bash
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 02:00:00
#SBATCH -J platform-jax-warp-env
# Build /p/rlprojects/RND/.venvs/platform_jax_warp: a self-contained copy of platform_jax plus
# warp-lang 1.14.0, the package MuJoCo MJX 3.11.0 names for its Warp implementation
# (mujoco_mjx-3.11.0.dist-info/METADATA). It exists only to measure speed idea 11 of the
# PointMaze and AntMaze study (the ant's physics through MJX's Warp implementation).
# platform_jax itself is never changed: every earlier number was measured on it.
#
# Run on a compute node:  sbatch scripts/create_platform_jax_warp_env.sh
set -euo pipefail
export PYTHONNOUSERSITE=1
CONDA=/sw/ubuntu2204/ebu082024/software/common/core/miniforge/24.7.1-py3.11/condabin/conda
SRC=/p/rlprojects/RND/.venvs/platform_jax
ENV=/p/rlprojects/RND/.venvs/platform_jax_warp
[ -e "$ENV" ] && { echo "$ENV already exists; remove it by hand before rebuilding"; exit 1; }
echo "host $(hostname) started $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT')"

# --copy: no hardlinks into a private package cache, so the env stays readable for every member.
$CONDA create --copy --yes --clone "$SRC" -p "$ENV"
"$ENV/bin/python" -m pip install --no-cache-dir --no-deps "warp-lang==1.14.0"
chmod -R g+rX "$ENV"

# Verification: the copied packages are at platform_jax's versions and warp imports.
"$ENV/bin/python" - <<'PY'
import importlib.metadata as md
import sys
print("python", sys.version.split()[0])
for package in ("jax", "jaxlib", "jax-cuda12-plugin", "numpy", "mujoco", "mujoco-mjx", "warp-lang"):
    print(package, md.version(package))
import jax
print("jax from this env:", jax.__file__.startswith("/p/rlprojects/RND/.venvs/platform_jax_warp"))
import warp
print("warp", warp.__version__)
PY
"$ENV/bin/python" -m pip check || true
cat > "$ENV/PROVENANCE.md" <<PROV
# platform_jax_warp

Built $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M PT') on $(hostname) by
10_jax_exploration_platform/scripts/create_platform_jax_warp_env.sh: a conda --copy clone of
/p/rlprojects/RND/.venvs/platform_jax plus warp-lang 1.14.0 (pip, --no-deps). For speed idea 11 of the PointMaze and
AntMaze study only (MuJoCo MJX 3.11.0's Warp implementation). Not canonical for anything.
PROV
echo "BUILD OK $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT')"
