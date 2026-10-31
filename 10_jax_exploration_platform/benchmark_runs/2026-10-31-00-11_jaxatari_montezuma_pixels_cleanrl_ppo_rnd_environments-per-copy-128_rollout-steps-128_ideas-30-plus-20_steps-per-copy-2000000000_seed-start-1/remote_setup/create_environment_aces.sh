#!/usr/bin/env bash
# Create this study's own environment; leave the source cs_jax environment unchanged.
set -euo pipefail
export PYTHONNOUSERSITE=1
unset LD_LIBRARY_PATH
setup_directory=/scratch/user/u.sl359075/rnd_atari_pixels/remote_setup
source_environment=/scratch/user/u.sl359075/envs/cs_jax
study_environment=/scratch/user/u.sl359075/envs/rnd_atari_pixels_jax
export PIP_CACHE_DIR=/scratch/user/u.sl359075/pip_cache
export CONDA_PKGS_DIRS=/scratch/user/u.sl359075/conda_pkgs
mkdir -p "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS"
[ ! -e "$study_environment" ] || { echo "Refusing to replace existing $study_environment"; exit 1; }
"$source_environment/bin/python" -m pip list --format=freeze > "$setup_directory/source_versions.txt"
/sw/eb/sw/Miniforge3/26.3.2-2/bin/conda create --yes --copy --clone "$source_environment" --prefix "$study_environment"
mkdir -p "$setup_directory/jaxtari_source"
tar -xzf "$setup_directory/jaxtari_04ab_source.tar.gz" -C "$setup_directory/jaxtari_source"
"$study_environment/bin/python" -m pip install --only-binary=:all: --constraint "$setup_directory/source_versions.txt" --requirement "$setup_directory/requirements_additional.txt"
"$study_environment/bin/python" -m pip install --constraint "$setup_directory/source_versions.txt" "$setup_directory/jaxtari_source"
"$study_environment/bin/python" -m pip check
"$study_environment/bin/python" -m pip list --format=freeze > "$setup_directory/study_versions.txt"
"$study_environment/bin/python" - <<'VERIFY'
# Read distribution metadata only; do not initialize JAX or execute numerical code here.
from importlib.metadata import version
from pathlib import Path
source = dict(line.strip().split('==',1) for line in Path('/scratch/user/u.sl359075/rnd_atari_pixels/remote_setup/source_versions.txt').read_text().splitlines() if '==' in line)
for name in ['jax','jaxlib','numpy','optax']:
    actual=version(name)
    expected=source[name]
    assert actual == expected, (name,actual,expected)
    print(name, actual)
print('ISOLATED_ENVIRONMENT_READY')
VERIFY
chmod -R g+rX "$study_environment"
