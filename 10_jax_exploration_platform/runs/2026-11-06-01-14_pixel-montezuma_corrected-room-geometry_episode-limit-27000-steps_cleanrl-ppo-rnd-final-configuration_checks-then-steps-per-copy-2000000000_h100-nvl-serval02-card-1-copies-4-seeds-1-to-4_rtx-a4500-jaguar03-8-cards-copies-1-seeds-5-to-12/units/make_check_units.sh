#!/bin/bash
# Checks of the final configuration on the corrected game, each at most 15 minutes (user rule 2026-11-05):
# timing confirmation at the selected packing, learning check (original reference against final
# configuration, same seeds), and the stop-and-resume comparison. Run once; prints unit paths.
set -eu
cd "$(dirname "$0")"
H100_LEARNER='{"convolution_algorithm": "space_to_depth", "cache_target_features": true, "predictor_selected_rows": 1280, "iteration_compiler_options": [["xla_gpu_enable_while_loop_double_buffering", "true"]]}'
A4500_LEARNER='{"convolution_batching": "unrolled", "cache_target_features": true, "predictor_selected_rows": 1280}'
ENVIRONMENT='{"separable_rectangle_drawing": true, "unroll_fixed_render_loops": true, "masked_conditional_draws": true, "two_way_masked_draws": true, "render_final_two_together": true}'
BODY=$(mktemp)

# Timing confirmations (the final configuration is the "reference" side of a timing-only screen).
python3 make_unit.py a4500-jaguar03 confirmation-final-configuration_rounds-11_updates-20 1 \
  "confirmation of the final RTX A4500 configuration on the corrected game: up to 11 rounds of 20 updates" \
  --reference-learner "$A4500_LEARNER" --reference-environment "$ENVIRONMENT" --profile --timing-only
python3 make_unit.py h100-serval02-card-1 confirmation-final-configuration_rounds-11_updates-20 4 \
  "confirmation of the final H100 configuration on the corrected game: up to 11 rounds of 20 updates" \
  --reference-learner "$H100_LEARNER" --reference-environment "$ENVIRONMENT" --profile --timing-only

# Learning check: four seeds of each side, one copy per A4500 card, up to 300 updates.
for seed in 1 2 3 4; do
  echo "timeout 840 \$TRAIN --run-dir \$DATA/run --copies 1 --seed-start $seed --target-updates 300 --time-limit-seconds 760" > "$BODY"
  python3 make_train_unit.py a4500-jaguar03 learning-check_original-reference_seed-$seed \
    "learning check on the corrected game, original reference, seed $seed, up to 300 updates" "$BODY"
  echo "timeout 840 \$TRAIN --run-dir \$DATA/run --copies 1 --seed-start $seed --target-updates 300 --time-limit-seconds 760 --learner '$A4500_LEARNER' --environment '$ENVIRONMENT'" > "$BODY"
  python3 make_train_unit.py a4500-jaguar03 learning-check_final-configuration_seed-$seed \
    "learning check on the corrected game, final configuration, seed $seed, up to 300 updates" "$BODY"
done
echo "timeout 840 \$TRAIN --run-dir \$DATA/run --copies 4 --seed-start 1 --target-updates 300 --time-limit-seconds 760 --learner '$H100_LEARNER' --environment '$ENVIRONMENT'" > "$BODY"
python3 make_train_unit.py h100-serval02-card-1 learning-check_final-configuration_seeds-1-to-4 \
  "learning check on the corrected game, final H100 configuration, seeds 1 to 4 packed, up to 300 updates" "$BODY"

# Stop and resume against uninterrupted, bit for bit.
for spec in "a4500-jaguar03 1 $A4500_LEARNER" "h100-serval02-card-1 4 $H100_LEARNER"; do
  gpu=${spec%% *}; rest=${spec#* }; copies=${rest%% *}; learner=${rest#* }
  cat > "$BODY" <<BODY_END
timeout 330 \$TRAIN --run-dir \$DATA/uninterrupted --copies $copies --target-updates 40 --observation-warmup-iterations 2 --learner '$learner' --environment '$ENVIRONMENT' || exit 1
timeout 250 \$TRAIN --run-dir \$DATA/resumed --copies $copies --target-updates 20 --observation-warmup-iterations 2 --learner '$learner' --environment '$ENVIRONMENT' || exit 1
timeout 250 \$TRAIN --run-dir \$DATA/resumed --copies $copies --target-updates 40 --observation-warmup-iterations 2 --learner '$learner' --environment '$ENVIRONMENT' || exit 1
\$COMPARE \$DATA/uninterrupted \$DATA/resumed \$DATA/comparison.json
BODY_END
  python3 make_train_unit.py $gpu resume-check_final-configuration_copies-${copies}_updates-20-plus-20 \
    "stop-and-resume check on the corrected game, final configuration, $copies copies" "$BODY"
done
rm -f "$BODY"
