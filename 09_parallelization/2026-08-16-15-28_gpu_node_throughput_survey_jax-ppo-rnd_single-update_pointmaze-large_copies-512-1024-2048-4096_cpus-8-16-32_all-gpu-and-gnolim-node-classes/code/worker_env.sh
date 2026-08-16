#!/bin/bash
# Sourced by every submission script before the benchmark starts.

# the user-site directory holds a different torch/jax build that would shadow the shared
# environment's own packages on sys.path; every job on this cluster must switch it off
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

# the shared project environment, invoked by full path — no conda activation needed
export SURVEY_PYTHON=/p/rlprojects/RND/.venvs/jax_gpu/bin/python

# the card is ours alone for the job, so the allocator takes it up front: a preallocated pool
# keeps the timing free of allocator growth between iterations, and the peak-in-use figure the
# report quotes is still the memory actually touched, not the size of the pool
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

# the driver on the older nodes needs the system library path spelled out
export LD_LIBRARY_PATH=/usr/lib/nvidia:${LD_LIBRARY_PATH}
