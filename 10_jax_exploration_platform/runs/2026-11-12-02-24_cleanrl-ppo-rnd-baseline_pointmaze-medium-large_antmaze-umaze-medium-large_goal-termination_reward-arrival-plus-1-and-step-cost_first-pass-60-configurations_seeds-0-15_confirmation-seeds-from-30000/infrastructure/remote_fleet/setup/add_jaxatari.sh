#!/bin/bash
# Add JAXAtari to the environment rnd_platform_jax_mujoco_warp: the trainer imports its Montezuma environment at
# its start, for every environment family. The same source commit as UVA's platform_jax; its dependencies are held
# to UVA's versions by the constraints file. JAXAtari is pure Python (a py3-none-any wheel is built from the
# source); nothing is compiled, and no numerical library is imported here.
#   bash add_jaxatari.sh <cluster>
set -euo pipefail
CLUSTER=$1
ENV_PATH=/scratch/$CLUSTER/liu4869/envs/rnd_platform_jax_mujoco_warp
HERE=$(dirname "$(readlink -f "$0")")
export PYTHONNOUSERSITE=1
unset LD_LIBRARY_PATH
echo "cluster $CLUSTER, host $(hostname), start $(date -Is)"
"$ENV_PATH/bin/python" -m pip install --no-cache-dir --only-binary=:all: --no-binary=jaxatari \
  -c "$HERE/constraints_uva_platform_jax_pip_list.txt" \
  "jaxatari @ git+https://github.com/k4ntz/JAXAtari@035fc46bf149220a2f462c864873707ced334a4c"
"$ENV_PATH/bin/python" -m pip list --format=freeze > "$ENV_PATH.pip_freeze.txt"
"$ENV_PATH/bin/python" -m pip check
chmod -R g+rX "$ENV_PATH"
# The installed files' record, to compare with UVA's copy of the same commit.
sha256sum "$ENV_PATH"/lib/python3.12/site-packages/jaxatari-0.1.0.dist-info/RECORD
echo "now $(wc -l < "$ENV_PATH.pip_freeze.txt") packages, end $(date -Is)"
echo ADD_COMPLETE
