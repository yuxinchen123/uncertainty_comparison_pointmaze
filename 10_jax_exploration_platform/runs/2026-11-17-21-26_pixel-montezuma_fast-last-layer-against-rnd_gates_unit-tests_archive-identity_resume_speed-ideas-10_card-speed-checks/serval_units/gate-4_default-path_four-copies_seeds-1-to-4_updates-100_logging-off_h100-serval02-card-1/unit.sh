#!/bin/bash
# Gate 4: the default path against the archive with four copies (archived seeds 1 to 4, 100 updates, H100 settings),
# on serval02 card 1, the card that produced the archive. Measured earlier: the archived run took about 5.5 minutes to
# start and 2.5 s per update there, so this unit takes about 10 minutes.
# EXPECTED_MINUTES=12
set -u
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true
export XDG_DATA_HOME=/p/rlprojects/RND/reference_repos/jaxtari_pixel_study_assets_montezuma_room_geometry_from_png
export JAXTARI_SKIP_SPRITE_UPDATE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
PY=/p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python
G=/p/rlprojects/RND/10_jax_exploration_platform/runs/2026-11-17-21-26_pixel-montezuma_fast-last-layer-against-rnd_gates_unit-tests_archive-identity_resume_speed-ideas-10_card-speed-checks
cd /p/rlprojects/RND/10_jax_exploration_platform
export PYTHONPATH=/p/rlprojects/RND/10_jax_exploration_platform/src
stamp() { TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT'; }
echo "start $(stamp) host $(hostname) card $CUDA_VISIBLE_DEVICES"
DATA="$G/data/gate-4_default-path_four-copies_seeds-1-to-4_updates-100_logging-off_h100-serval02-card-1"
ARCHIVED="/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/runs/2026-11-06-01-14_pixel-montezuma_corrected-room-geometry_episode-limit-27000-steps_cleanrl-ppo-rnd-final-configuration_checks-then-steps-per-copy-2000000000_h100-nvl-serval02-card-1-copies-4-seeds-1-to-4_rtx-a4500-jaguar03-8-cards-copies-1-seeds-5-to-12/data/h100-serval02-card-1_final-training_seeds-1-to-4/run/metrics.jsonl"
# A gate run starts from nothing: an existing run folder would be resumed instead.
[ -e "$DATA/run" ] && { echo "$DATA/run exists already; this gate needs a new folder"; exit 2; }
mkdir -p "$DATA"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.used --format=csv,noheader | tee "$DATA/hardware.txt"
timeout --signal=TERM 1500 $PY -u scripts/atari_pixel_campaign/train_final.py --run-dir "$DATA/run" --copies 4 --seed-start 1 --target-updates 100 --observation-warmup-iterations 50 --checkpoint-seconds 1200 --learner '{"convolution_algorithm": "space_to_depth", "cache_target_features": true, "predictor_selected_rows": 1280, "iteration_compiler_options": [["xla_gpu_enable_while_loop_double_buffering", "true"]]}' --environment '{"separable_rectangle_drawing": true, "unroll_fixed_render_loops": true, "masked_conditional_draws": true, "two_way_masked_draws": true, "render_final_two_together": true, "max_episode_steps": 27000, "reset_lane_budget": 16}'
rc=$?
echo "$(stamp) trainer exit $rc"
[ $rc -eq 0 ] || exit $rc
$PY scripts/atari_pixel_campaign/compare_metric_prefix.py "$ARCHIVED" "$DATA/run/metrics.jsonl" "$DATA/comparison_with_archive.json" --lines 100
rc=$?
$PY "$G/slurm/seconds_per_update.py" "$DATA/run/metrics.jsonl" --last-line 100 | tee "$DATA/seconds_per_update.json"
read MIG USED <<<"$(nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=mig.mode.current,memory.used --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' ')"
echo "end $(stamp) comparison exit $rc; card memory used after the run ${USED} MiB"
exit $rc
