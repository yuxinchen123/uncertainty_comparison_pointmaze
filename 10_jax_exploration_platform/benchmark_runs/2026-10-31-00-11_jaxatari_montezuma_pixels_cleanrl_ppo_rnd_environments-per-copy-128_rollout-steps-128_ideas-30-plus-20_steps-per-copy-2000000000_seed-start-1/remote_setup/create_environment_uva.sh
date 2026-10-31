#!/bin/bash
# Build the pixel study environment without changing the historical platform environment.
set -euo pipefail
export PYTHONNOUSERSITE=1
study_conda=/sw/ubuntu2204/ebu082025/software/common/core/miniforge/26.7.2/condabin/conda
study_environment=/p/rlprojects/RND/.venvs/atari_pixels_jax
study_source=/p/rlprojects/RND/reference_repos/jaxtari_pixel_study
test "$(git -C "$study_source" rev-parse HEAD)" = 04ab9aec318c72162e2e53503a594a489d7e162b
"$study_conda" create --yes --offline --copy --clone /p/rlprojects/RND/.venvs/platform_jax --prefix "$study_environment"
"$study_environment/bin/python" -m pip install --no-deps "$study_source"
"$study_environment/bin/python" -m pip install pytest==8.4.2
"$study_environment/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0
"$study_environment/bin/python" -m pip check
"$study_environment/bin/python" -m pip freeze > "$study_environment/pip_freeze.txt"
chmod -R g+rX "$study_environment"
