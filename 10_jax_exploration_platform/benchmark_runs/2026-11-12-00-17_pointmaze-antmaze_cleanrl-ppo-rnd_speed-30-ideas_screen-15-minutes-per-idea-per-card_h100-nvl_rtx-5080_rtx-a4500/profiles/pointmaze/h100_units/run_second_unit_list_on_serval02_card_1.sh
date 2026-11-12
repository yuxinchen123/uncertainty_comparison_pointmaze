#!/bin/bash
# Run ON serval02 (under nohup): the second list of PointMaze H100 NVL units (the ideas after idea 14, and the
# earlier ideas at 256 policies), one after another, each under the card's lock, which is free between two units.
# A unit starts as soon as card 1 and our lock are free; while they are busy it is tried again every 30 seconds.
# All units together wait at most 90 minutes; after that the list stops and names the units not run.
# Nothing is ever stopped. The waiting process's id is in logs/h100-nvl_units_second_list.pid.
RUN=$(cd "$(dirname "$0")/../../.." && pwd)
S=/p/rlprojects/.claude/skills/serval05-queue/scripts
U="$RUN/profiles/pointmaze/h100_units/h100_unit.sh"
R="$RUN/profiles/pointmaze"
C=benchmarks/harness/compare_study_arms.py
pt() { TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT'; }
echo $$ > "$RUN/logs/h100-nvl_units_second_list.pid"
TRIES_LEFT=180
unit() {
    # unit "<name>" <record file> <python script> <arguments ...>
    while [ "$TRIES_LEFT" -gt 0 ]; do
        bash $S/serval02_card1_run.sh --kind profile --session 9ca0cb45-e2aa-43c6-9353-3a3d17cddb5f -- bash "$U" "$@" > /dev/null
        rc=$?
        [ $rc -ne 75 ] && { echo "$(pt) unit \"$1\": ran, exit code $rc"; return; }
        TRIES_LEFT=$((TRIES_LEFT - 1))
        [ $((TRIES_LEFT % 10)) -eq 0 ] && echo "$(pt) unit \"$1\": card 1 or its lock busy; $TRIES_LEFT tries of 30 s left"
        sleep 30
    done
    echo "$(pt) unit \"$1\": not run, the 90 minutes of waiting are used up"
}
idea() {
    # idea <slug> <record suffix> <shapes> <arm> [<arm> ...]
    slug=$1; suffix=$2; shapes=$3; shift 3
    arms=(); for arm in "$@"; do arms+=(--arm "$arm"); done
    unit "idea $slug at $shapes" "$R/ideas/idea-${slug}_h100-nvl_$suffix.json" $C --idea "$slug" --shapes "$shapes" \
        --budget-seconds 780 "${arms[@]}"
}
idea 17_intrinsic-sums-by-matrix-product environments-per-policy-4_policies-256 4:256 \
    "128-step loops for the filter and both advantages" \
    "intrinsic filter and advantage as matrix products:intrinsic_sums_by_matrix_product=True" \
    "loops unrolled 16 steps:advantage_scan_unroll=16" \
    "matrix products and the extrinsic loop unrolled 16 steps:intrinsic_sums_by_matrix_product=True,advantage_scan_unroll=16"
idea 20_target-features-once-per-iteration environments-per-policy-4_policies-256-and-1024 4:256,4:1024 \
    "target features computed in every gradient step" \
    "target features computed once per iteration:target_features_once_per_iteration=True"
idea 28_several-iterations-per-host-call environments-per-policy-4_policies-256 4:256 "one iteration per host call" \
    "8 iterations per host call:iterations_per_host_call=8" "32 iterations per host call:iterations_per_host_call=32"
idea 13_rollout-unroll environments-per-policy-4_policies-256 4:256 "rollout unroll 16 (today's choice)" \
    "rollout unroll 32:scan_unroll=32" "rollout unroll 64:scan_unroll=64" "rollout unroll 8:scan_unroll=8"
idea 21_actor-and-critic-hidden-layers-as-one-batched-network environments-per-policy-4-and-16_policies-256 4:256,16:256 \
    "actor and critic forwards one after the other" "hidden layers of both as one batched network:actor_critic_joint_trunk=True"
idea 25_whole-sweep-in-one-program environments-per-policy-4_policies-960 4:960 \
    "1 cell (every policy has the same learning rate, entropy coefficient and intrinsic weight)" \
    "8 cells of different learning rate, entropy coefficient and intrinsic weight:sweep_cells=8" \
    "60 cells of different learning rate, entropy coefficient and intrinsic weight:sweep_cells=60"
idea 22_update-loop-unroll environments-per-policy-4_policies-256 4:256 "update unroll 2 (today's value)" \
    "update unroll 1:update_unroll=1" "update unroll 4:update_unroll=4"
idea 14_next-values-from-shifted-values environments-per-policy-4_policies-256 4:256 \
    "second critic pass over the final observations" "next values read from the following step's values:next_values_from_shifted_values=True"
idea 15_no-reset-draws-at-zero-noise environments-per-policy-4_policies-256 4:256 \
    "four keyed draws per environment and step" "no draws at zero noise:skip_reset_draws_at_zero_noise=True"
idea 16_action-noise-from-the-compiler-bit-generator environments-per-policy-4_policies-256 4:256 \
    "action noise from the default generator" "action noise from the compiler's bit generator:action_noise_generator=rbg"
unit "time breakdown at 4 environments per policy and 256 policies" \
    "$R/time_breakdown/time-breakdown_pointmaze-large_environments-per-policy-4_policies-256_h100-nvl.json" \
    benchmarks/harness/profile_study_iteration_pieces.py --copies 256 --envs-per-copy 4 --budget-seconds 720
echo "$(pt) the list is done"
rm -f "$RUN/logs/h100-nvl_units_second_list.pid"
