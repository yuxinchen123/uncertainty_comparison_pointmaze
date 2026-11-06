#!/bin/bash
# Timing-only runs of the H100 final configuration with gathered resets at four copies (budgets 16 and 32).
set -eu
cd "$(dirname "$0")"
H100_LEARNER='{"convolution_algorithm": "space_to_depth", "cache_target_features": true, "predictor_selected_rows": 1280, "iteration_compiler_options": [["xla_gpu_enable_while_loop_double_buffering", "true"]]}'
for budget in 16 32; do
  python3 make_unit.py h100-serval02-card-1 timing-final-configuration-gathered-resets_budget-$budget 4 \
    "timing of the H100 final configuration with gathered resets (budget $budget) on the corrected game" \
    --reference-learner "$H100_LEARNER" --reference-environment "{\"separable_rectangle_drawing\": true, \"unroll_fixed_render_loops\": true, \"masked_conditional_draws\": true, \"two_way_masked_draws\": true, \"render_final_two_together\": true, \"reset_lane_budget\": $budget}" --profile --timing-only
done
