#!/bin/bash
# Everything a JAX job of this run must set before it touches a graphics card.
# Sourced by every submission script and by the serval05 command; never run on its own.
# The settings and the reasons are the rnd-jax-submission skill's section 7.

# the platform's canonical environment, by full path, as registered in .venvs/ENVS.md
export PLATFORM_PYTHON=/p/rlprojects/RND/.venvs/platform_jax/bin/python
export PLATFORM_ROOT=/p/rlprojects/RND/10_jax_exploration_platform

# ~/.local holds a different jax and a different torch that shadow the environment's own packages;
# a job that does not hide them fails with "Unable to load CUDA" on a healthy card
export PYTHONNOUSERSITE=1

# one worker per card, allocating on demand: a preallocated pool would take the whole card and a
# co-located slot would then fail while initialising cuDNN rather than with an out-of-memory message
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75

# the driver libraries on the older nodes are not on the default path
export LD_LIBRARY_PATH=/usr/lib/nvidia:${LD_LIBRARY_PATH:-}

# a repair for hosts whose own ptxas predates CUDA 12: the environment carries its own
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/p/rlprojects/RND/.venvs/platform_jax/lib/python3.12/site-packages/nvidia/cuda_nvcc"

# node-local compiled-program cache. Stage C measured it cutting the 8,448-copy compile from about
# 500 s to about 110 s, so a canary on the same node pays the long compile and the real unit does
# not. It is node-local on purpose: the shared filesystem is far slower to read a cache entry from.
export JAX_COMPILATION_CACHE_DIR=/localtmp/${USER}/platform_jax_cache
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

# so the job log shows progress as it happens instead of at exit
export PYTHONUNBUFFERED=1
