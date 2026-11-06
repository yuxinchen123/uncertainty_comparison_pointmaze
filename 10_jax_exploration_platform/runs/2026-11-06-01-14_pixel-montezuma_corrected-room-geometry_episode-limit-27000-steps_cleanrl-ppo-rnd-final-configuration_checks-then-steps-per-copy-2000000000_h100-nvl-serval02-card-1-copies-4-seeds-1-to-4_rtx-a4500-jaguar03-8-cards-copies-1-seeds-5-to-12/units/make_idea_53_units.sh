#!/bin/bash
# Idea 53 (corrected game): game resets as the stored start state with the new key (the pinned reset
# is deterministic apart from the key it stores), screened against gathered resets (budget 8), A4500.
set -eu
cd "$(dirname "$0")"
A4500_LEARNER='{"convolution_batching": "unrolled", "cache_target_features": true, "predictor_selected_rows": 1280}'
python3 make_unit.py a4500-jaguar03 idea-53_constant-game-reset_on-gathered-resets-budget-8 1 \
  "idea 53: game reset as the stored start state with the new key, on gathered resets (budget 8)" \
  --environment '{"constant_game_reset": true}' --reference-learner "$A4500_LEARNER" \
  --reference-environment '{"separable_rectangle_drawing": true, "unroll_fixed_render_loops": true, "masked_conditional_draws": true, "two_way_masked_draws": true, "render_final_two_together": true, "reset_lane_budget": 8}'
