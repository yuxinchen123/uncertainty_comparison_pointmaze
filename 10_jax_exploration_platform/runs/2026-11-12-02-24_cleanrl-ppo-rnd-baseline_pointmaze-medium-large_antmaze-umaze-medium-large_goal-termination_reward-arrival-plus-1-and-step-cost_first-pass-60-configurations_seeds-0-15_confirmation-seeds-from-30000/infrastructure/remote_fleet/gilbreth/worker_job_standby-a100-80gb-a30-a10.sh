#!/bin/bash
# Worker job of the maze study on gilbreth, pool standby-a100-80gb-a30-a10. Written by scripts/maze_campaign/remote_fleet.py:
# change that, not this copy. One card per job. It checks the card, then runs the study's worker.py, which
# takes units from queue/pending one at a time until none is left or too little of the job is left.
# No speed test or benchmark is run.
RUN=/scratch/gilbreth/liu4869/rnd_maze/runs/2026-11-12-02-24_cleanrl-ppo-rnd-baseline
PY=/scratch/gilbreth/liu4869/envs/rnd_platform_jax_mujoco_warp/bin/python
CODE=/scratch/gilbreth/liu4869/rnd_maze/code/source_6ad2fa4643e7bc7dfd156472fe43b689a517e229/10_jax_exploration_platform
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# Module-provided CUDA libraries would be found before the environment's own (Gilbreth, DeltaAI).
unset LD_LIBRARY_PATH
JOB=${SLURM_JOB_ID:?this script runs only as a Slurm job}
cd "$RUN" || exit 2
# (no card guard on this server)
echo "job $JOB on $(hostname), start $(date -Is), card class a100_80gb_or_a30_or_a10_gilbreth"

# Nothing to do: give the card back before starting Python.
if ! ls queue/pending/*.json > /dev/null 2>&1; then echo "nothing pending; this job ends"; exit 0; fi

# run_training.py records the commit of the folder above the code copy and stops where git cannot read it.
git -C "$CODE/.." rev-parse HEAD > /dev/null \
  || { echo "git cannot read a commit above the code copy; this job ends without taking a unit"; exit 5; }

# The environment must use the card and hold every library the trainer imports at its start, or every unit taken
# would fail: the modules run_training.py imports are imported here first, before a unit is claimed.
PYTHONPATH="$CODE/src:$CODE/scripts" "$PY" -c 'import jax, experiment_configuration
from exploration_platform.training import checkpoint, episode_log, sweep
from exploration_platform.training.runner import Runner
from exploration_platform.agents.ppo import config, presets
from exploration_platform.bonuses import registry
from exploration_platform.envs.antmaze import am_common
from exploration_platform.envs.pointmaze import pm_common
print("jax devices:", jax.devices()); assert jax.default_backend() == "gpu"' \
  || { echo "jax does not use the card, or a module the trainer imports is missing; this job ends without taking a unit"; exit 3; }

exec "$PY" "$CODE/scripts/maze_campaign/worker.py" --run-dir "$RUN" --remote-records --minimum-seconds 1800
