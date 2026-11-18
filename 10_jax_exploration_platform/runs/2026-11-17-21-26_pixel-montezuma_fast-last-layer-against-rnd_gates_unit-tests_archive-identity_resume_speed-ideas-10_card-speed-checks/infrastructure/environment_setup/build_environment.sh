#!/bin/bash
# Build the pixel Montezuma study's environment atari_pixels_jax on one remote server, with the package versions of
# UVA's /p/rlprojects/RND/.venvs/atari_pixels_jax (Python 3.12.13, JAX 0.11.0 with the CUDA 12 plugin, the study's
# JAXtari at commit 04ab9aec). It installs files only and imports nothing numerical, so it may run on a login node.
#
#   bash build_environment.sh <conda> <environment> <setup folder> <x86_64|aarch64> <conda package folder> <pip cache>
#
# x86_64: the conda packages are UVA's own files (conda_explicit_linux-64.txt); aarch64 (NCSA DeltaAI): the same
# versions from conda-forge's linux-aarch64 channel. Every pip package is installed at UVA's version with
# --no-deps, so nothing is resolved anew: wheels from PyPI, torch 2.6.0 (CPU) from PyTorch's CPU index on x86_64
# (PyPI's aarch64 torch 2.6.0 is a CPU build), JAXAtari from its git commit, JAXtari from the source archive.
set -euo pipefail
CONDA=$1 ENVIRONMENT=$2 SETUP=$3 PLATFORM=$4
export CONDA_PKGS_DIRS=$5 PIP_CACHE_DIR=$6
export PYTHONNOUSERSITE=1
unset LD_LIBRARY_PATH PYTHONPATH PYTHONHOME
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR"
[ ! -e "$ENVIRONMENT" ] || { echo "refusing to replace the existing $ENVIRONMENT"; exit 1; }
LOG=$SETUP/build_$(hostname -s)_$(date +%Y%m%d-%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
echo "build of $ENVIRONMENT on $(hostname) ($PLATFORM), start $(date -Is)"

# 1. Python 3.12.13 and the packages conda provides (pip, setuptools, wheel, packaging)
if [ "$PLATFORM" = x86_64 ]; then
  "$CONDA" create --yes --copy --prefix "$ENVIRONMENT" --file "$SETUP/conda_explicit_linux-64.txt"
else
  "$CONDA" create --yes --copy --prefix "$ENVIRONMENT" --override-channels -c conda-forge \
    python=3.12.13 pip=26.2.1 setuptools=84.0.0 wheel=0.48.0 packaging=26.3
fi
PY=$ENVIRONMENT/bin/python

# 2. every wheel at UVA's version; torch alone from PyTorch's CPU index on x86_64
grep -v '^torch==' "$SETUP/requirements_wheels_$PLATFORM.txt" > "$SETUP/requirements_wheels_${PLATFORM}_without_torch.txt"
"$PY" -m pip install --no-deps --only-binary=:all: -r "$SETUP/requirements_wheels_${PLATFORM}_without_torch.txt"
TORCH=$(grep '^torch==' "$SETUP/requirements_wheels_$PLATFORM.txt")
if [ "$PLATFORM" = x86_64 ]; then
  "$PY" -m pip install --no-deps --only-binary=:all: --index-url https://download.pytorch.org/whl/cpu "$TORCH"
else
  "$PY" -m pip install --no-deps --only-binary=:all: "$TORCH"
fi

# 3. the two packages built from source: JAXAtari at its git commit, the study's JAXtari at 04ab9aec
"$PY" -m pip install --no-deps -r "$SETUP/requirements_from_source.txt"
rm -rf "$SETUP/jaxtari_source"
mkdir -p "$SETUP/jaxtari_source"
tar -xzf "$SETUP/jaxtari_04ab9aec_source.tar.gz" -C "$SETUP/jaxtari_source"
"$PY" -m pip install --no-deps "$SETUP/jaxtari_source/jaxtari_pixel_study"
rm -rf "$SETUP/jaxtari_source"

# 4. the record of what was installed: dependency check, freeze, and the file digests compared with UVA's
"$PY" -m pip check
"$PY" -m pip freeze > "$SETUP/pip_freeze_$(hostname -s).txt"
"$PY" "$SETUP/distribution_records.py" "$ENVIRONMENT/lib/python3.12/site-packages" > "$SETUP/distribution_records_$(hostname -s).json"
"$PY" "$SETUP/distribution_records.py" --compare "$SETUP/uva_distribution_records.json" \
  "$SETUP/distribution_records_$(hostname -s).json" || true
du -sh "$ENVIRONMENT"
find "$ENVIRONMENT" -type f | wc -l
echo "build of $ENVIRONMENT ended $(date -Is)"
