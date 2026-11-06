#!/bin/bash
# Idea 51 (corrected game): run NOOP/FIRE resets only on the lanes that ended, gathered into a fixed
# batch (reset_lane_budget), instead of computing the reset for every lane whenever any lane ended.
# Fifteen-minute paired screens at each card's selected packing; the reference is the final configuration.
set -eu
cd "$(dirname "$0")"
H100_LEARNER='{"convolution_algorithm": "space_to_depth", "cache_target_features": true, "predictor_selected_rows": 1280, "iteration_compiler_options": [["xla_gpu_enable_while_loop_double_buffering", "true"]]}'
A4500_LEARNER='{"convolution_batching": "unrolled", "cache_target_features": true, "predictor_selected_rows": 1280}'
ENVIRONMENT='{"separable_rectangle_drawing": true, "unroll_fixed_render_loops": true, "masked_conditional_draws": true, "two_way_masked_draws": true, "render_final_two_together": true}'
python3 make_unit.py a4500-jaguar03 idea-51_resets-gathered-for-ended-lanes_budget-8 1 \
  "idea 51: NOOP/FIRE resets only for ended lanes, gathered 8 at a time (exact; per-lane reset beyond 8)" \
  --environment '{"reset_lane_budget": 8}' --reference-learner "$A4500_LEARNER" --reference-environment "$ENVIRONMENT"
python3 make_unit.py h100-serval02-card-1 idea-51_resets-gathered-for-ended-lanes_budget-16 4 \
  "idea 51: NOOP/FIRE resets only for ended lanes, gathered 16 at a time (exact; per-lane reset beyond 16)" \
  --environment '{"reset_lane_budget": 16}' --reference-learner "$H100_LEARNER" --reference-environment "$ENVIRONMENT"
