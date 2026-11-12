#!/bin/bash
# Run ON serval02 (under nohup): the third list of PointMaze H100 NVL units, started when the second list has
# ended: the kept keys timed together, the one-iteration state comparison of the later keys, and idea 27 (two
# processes on one card). Same rules as the second list: a unit starts when card 1 and our lock are free, is
# tried every 30 seconds, all units together wait at most 60 minutes, and nothing is ever stopped.
RUN=$(cd "$(dirname "$0")/../../.." && pwd)
S=/p/rlprojects/.claude/skills/serval05-queue/scripts
U="$RUN/profiles/pointmaze/h100_units/h100_unit.sh"
R="$RUN/profiles/pointmaze"
pt() { TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT'; }
while [ -f "$RUN/logs/h100-nvl_units_second_list.pid" ]; do sleep 30; done
echo $$ > "$RUN/logs/h100-nvl_units_third_list.pid"
TRIES_LEFT=120
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
    echo "$(pt) unit \"$1\": not run, the 60 minutes of waiting are used up"
}
PAIR="intrinsic_sums_by_matrix_product=True,advantage_scan_unroll=16"
unit "kept keys together at 4 environments and 256 policies" \
    "$R/ideas/idea-00_kept-keys-together_h100-nvl_environments-per-policy-4_policies-256.json" \
    benchmarks/harness/compare_study_arms.py --idea 00_kept-keys-together --shapes 4:256 --budget-seconds 780 \
    --arm "every speed key off" \
    --arm "matrix products and the loops unrolled 16 steps:$PAIR" \
    --arm "the same, and target features once per iteration:$PAIR,target_features_once_per_iteration=True" \
    --arm "the same, and joint hidden layers, no reset draws, next values from shifted values, packed-key permutation:$PAIR,target_features_once_per_iteration=True,actor_critic_joint_trunk=True,skip_reset_draws_at_zero_noise=True,next_values_from_shifted_values=True,permutation_draw=sort_packed_keys"
unit "state after one iteration with each later key" \
    "$R/state_checks/state-check_later-speed-keys_environments-per-policy-4_policies-64_iterations-1_h100-nvl.json" \
    benchmarks/harness/compare_study_states_on_card.py --copies 64 --envs-per-copy 4 --iterations 1 \
    --arm "reference" --arm "reference built again" \
    --arm "intrinsic filter and advantage as matrix products:intrinsic_sums_by_matrix_product=True" \
    --arm "loops unrolled 16 steps:advantage_scan_unroll=16" \
    --arm "hidden layers of actor and critic as one batched network:actor_critic_joint_trunk=True" \
    --arm "no reset draws at zero noise:skip_reset_draws_at_zero_noise=True" \
    --arm "target features computed once per iteration:target_features_once_per_iteration=True"
unit "idea 27 two processes on one card" \
    "$R/ideas/idea-27_two-processes-per-card_h100-nvl_environments-per-policy-4_policies-256-per-process.json" \
    benchmarks/harness/two_processes_on_one_card.py --envs-per-copy 4 --copies-per-process 256 \
    --lead-seconds 200 --also-one-process-alone
echo "$(pt) the list is done"
rm -f "$RUN/logs/h100-nvl_units_third_list.pid"
