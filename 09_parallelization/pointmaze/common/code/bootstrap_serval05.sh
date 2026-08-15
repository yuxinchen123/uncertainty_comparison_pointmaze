#!/bin/bash
# One-time bootstrap of the serval05 LOCAL working environments for 09_parallelization.
# Local disk (not NFS) because python imports from /p are very slow on this box.
# Creates:  /localtmp/sl5nw/uv            — the uv package manager (static binary)
#           /localtmp/sl5nw/venvs/rnd09_torch — python 3.12 + torch cu128 + CUDA 12.8 nvcc wheels
#           /localtmp/sl5nw/venvs/rnd09_jax   — python 3.12 + jax[cuda12]
# Idempotent: safe to re-run; uv skips satisfied installs.
set -euxo pipefail

export UV_INSTALL_DIR=/localtmp/sl5nw/uv
export UV_CACHE_DIR=/localtmp/sl5nw/uv_cache
export UV_PYTHON_INSTALL_DIR=/localtmp/sl5nw/uv_pythons
mkdir -p /localtmp/sl5nw/venvs /localtmp/sl5nw/locks /localtmp/sl5nw/tmp

if [ ! -x /localtmp/sl5nw/uv/uv ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
UV=/localtmp/sl5nw/uv/uv

# torch env: cu128 build (H100 = sm_90), plus the pip CUDA toolchain for compiling the fused kernel
$UV venv /localtmp/sl5nw/venvs/rnd09_torch --python 3.12
$UV pip install --python /localtmp/sl5nw/venvs/rnd09_torch/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple \
  torch numpy ninja \
  nvidia-cuda-nvcc-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-cccl-cu12

# jax env: self-contained CUDA via pip wheels
$UV venv /localtmp/sl5nw/venvs/rnd09_jax --python 3.12
$UV pip install --python /localtmp/sl5nw/venvs/rnd09_jax/bin/python \
  "jax[cuda12]" numpy

echo "=== verify torch ==="
/localtmp/sl5nw/venvs/rnd09_torch/bin/python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(),
      "device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
EOF
echo "=== verify jax ==="
/localtmp/sl5nw/venvs/rnd09_jax/bin/python - <<'EOF'
import jax
print("jax", jax.__version__, "devices", jax.devices())
EOF
echo BOOTSTRAP_DONE
