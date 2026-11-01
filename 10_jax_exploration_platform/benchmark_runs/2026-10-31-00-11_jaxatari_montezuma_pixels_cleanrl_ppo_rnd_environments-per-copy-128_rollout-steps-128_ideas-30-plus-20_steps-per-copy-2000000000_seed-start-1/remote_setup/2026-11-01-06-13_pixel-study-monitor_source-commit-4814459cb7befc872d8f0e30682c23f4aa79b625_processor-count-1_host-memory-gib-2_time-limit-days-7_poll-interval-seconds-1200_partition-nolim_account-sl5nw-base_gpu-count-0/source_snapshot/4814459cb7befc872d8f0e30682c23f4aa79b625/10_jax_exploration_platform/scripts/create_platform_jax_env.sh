#!/bin/bash
# Build the shared environment for 10_jax_exploration_platform.
#
# Pinned to exactly what every 09_parallelization number was measured on: python 3.12 and
# jax[cuda12] 0.11.0 with numpy 2.5.2 (read off serval05's node-local env
# /localtmp/sl5nw/venvs/rnd09_jax on 2026-08-16). No newer release is tried here — the point of
# the pin is that the platform starts from a numerically identical stack, so a difference in a
# measurement is the platform's doing and not the library's.
#
# Kept separate from the canonical torch env `exploration` on purpose: jax[cuda12] brings its own
# nvidia-*-cu12 pip packages, and installing them beside torch's could break the torch install
# that every 07_reconstruction run depends on.
#
# Adapted from /p/rlprojects/RLforOR/inventory_management/joint_replenishment/setup/create_jax_env.sh
#
# Run:  bash create_platform_jax_env.sh 2>&1 | tee build_platform_jax.log
set -euo pipefail
export PYTHONNOUSERSITE=1
CONDA=/sw/ubuntu2204/ebu082024/software/common/core/miniforge/24.7.1-py3.11/condabin/conda
ENV=/p/rlprojects/RND/.venvs/platform_jax

# --copy: no hardlinks into a private package cache, so the env is self-contained and stays
# readable for every rlprojects member.
$CONDA create --copy --yes -p "$ENV" python=3.12

"$ENV/bin/pip" install --no-cache-dir "jax[cuda12]==0.11.0"
"$ENV/bin/pip" install --no-cache-dir "numpy==2.5.2"

# group-readable and group-traversable, so a collaborator's job can run this interpreter
chmod -R g+rX "$ENV"

# Verification: jax resolves inside the env, at the pinned version, and the cuda plugin is there.
# A build-time host has no graphics card, so only the processor devices are listed here; the
# graphics-card check is the golden-parity run on serval05.
"$ENV/bin/python" - <<'EOF'
import sys
import jax
import numpy
print("python", sys.version.split()[0])
print("jax", jax.__version__, "numpy", numpy.__version__)
print("jax from this env:", jax.__file__.startswith("/p/rlprojects/RND/.venvs/platform_jax"))
print("devices at build time (processor only is expected here):", jax.devices("cpu"))
import importlib.metadata as md
for pkg in ("jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt"):
    print(pkg, md.version(pkg))
EOF
echo "BUILD OK"
