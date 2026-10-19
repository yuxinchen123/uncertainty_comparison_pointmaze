#!/bin/bash
# Canary for the screening's job shape: a 512-copy RND unit and a 128-copy no-bonus unit on the
# medium map with diagnostics, 600 and 300 iterations, written to canary/ (not the run itself).
set -euo pipefail
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/usr/lib/nvidia:${LD_LIBRARY_PATH:-}
P=/p/rlprojects/RND/.venvs/platform_jax/bin/python
T=/p/rlprojects/RND/10_jax_exploration_platform/scripts/run_training.py
C=/p/rlprojects/RND/10_jax_exploration_platform/runs/2026-10-19-16-18_map-screening_umaze-cap150_medium-cap250_medium-cap400_large-cap400_rnd-weights-1-10-100-1000_visit-count-sqrt-weight-1_no-bonus_learning-rate-1e-3_steps-30M-large-50M_seeds-0-127/canary
$P -u $T --run-dir $C --unit-id canary-medium-cap250_rnd --bonus rnd_next_state --parameter-settings /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-10-19-16-18_map-screening_umaze-cap150_medium-cap250_medium-cap400_large-cap400_rnd-weights-1-10-100-1000_visit-count-sqrt-weight-1_no-bonus_learning-rate-1e-3_steps-30M-large-50M_seeds-0-127/code/parameters/medium-cap250_rnd_next_state.json --copies-per-cell 128 --iterations 600 --window-iterations 250 --track-coverage --diagnostics --no-aggregate --map medium --start-cell 6,1 --goal-cell 1,6 --episode-steps 250
$P -u $T --run-dir $C --unit-id canary-umaze-cap150_none --bonus none --parameter-settings /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-10-19-16-18_map-screening_umaze-cap150_medium-cap250_medium-cap400_large-cap400_rnd-weights-1-10-100-1000_visit-count-sqrt-weight-1_no-bonus_learning-rate-1e-3_steps-30M-large-50M_seeds-0-127/code/parameters/umaze-cap150_none.json --copies-per-cell 128 --iterations 300 --window-iterations 250 --track-coverage --diagnostics --no-aggregate --map umaze --start-cell 3,1 --goal-cell 1,1 --episode-steps 150
# the summarizer's own test, on this compute node
cd /p/rlprojects/RND/10_jax_exploration_platform && $P tests/integration/test_summarize_episode_records.py
$P scripts/summarize_episode_records.py $C --output $C/summary_canary.json
