#!/bin/bash
# Idea 52 (corrected game): gathered resets whose padding entries skip their NOOP frames, at budgets 4
# and 8, screened against the final configuration without gathering (A4500, one copy).
set -eu
cd "$(dirname "$0")"
A4500_LEARNER='{"convolution_batching": "unrolled", "cache_target_features": true, "predictor_selected_rows": 1280}'
ENVIRONMENT='{"separable_rectangle_drawing": true, "unroll_fixed_render_loops": true, "masked_conditional_draws": true, "two_way_masked_draws": true, "render_final_two_together": true}'
for budget in 4 8; do
  python3 make_unit.py a4500-jaguar03 idea-52_gathered-resets-padding-without-noops_budget-$budget 1 \
    "idea 52: resets only for ended lanes, gathered $budget at a time, padding entries without NOOP frames (exact)" \
    --environment "{\"reset_lane_budget\": $budget}" --reference-learner "$A4500_LEARNER" --reference-environment "$ENVIRONMENT"
done
