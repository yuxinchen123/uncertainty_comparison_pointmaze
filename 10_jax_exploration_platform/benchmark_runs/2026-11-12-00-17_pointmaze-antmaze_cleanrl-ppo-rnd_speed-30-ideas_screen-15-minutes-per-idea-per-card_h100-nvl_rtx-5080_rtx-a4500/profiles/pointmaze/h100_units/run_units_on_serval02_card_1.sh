#!/bin/bash
# Run ON serval02 (under nohup): the PointMaze H100 NVL units, one after another, each under the card's lock,
# which is free between two units. A unit starts as soon as card 1 and our lock are free; while they are busy
# it is tried again every 30 seconds for at most 25 minutes, then skipped. Nothing is ever stopped.
# One line per attempt and per unit goes to standard output. The waiting process's id is in logs/h100-nvl_units.pid.
RUN=$(cd "$(dirname "$0")/../../.." && pwd)
S=/p/rlprojects/.claude/skills/serval05-queue/scripts
U="$RUN/profiles/pointmaze/h100_units/h100_unit.sh"
R="$RUN/profiles/pointmaze"
pt() { TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT'; }
echo $$ > "$RUN/logs/h100-nvl_units.pid"
unit() {
    # unit "<name>" <record file> <python script> <arguments ...>
    for attempt in $(seq 1 50); do
        bash $S/serval02_card1_run.sh --kind profile --session 9ca0cb45-e2aa-43c6-9353-3a3d17cddb5f -- bash "$U" "$@"
        rc=$?
        [ $rc -ne 75 ] && { echo "$(pt) unit \"$1\": ran, exit code $rc"; return; }
        echo "$(pt) unit \"$1\" attempt $attempt: card 1 or its lock busy, waiting 30 s"
        sleep 30
    done
    echo "$(pt) unit \"$1\": card 1 stayed busy for 25 minutes; not run"
}
C=benchmarks/harness/compare_study_arms.py
unit "policies per process at 4 environments per policy" \
    "$R/policies_per_process/idea-26_policies-per-process_h100-nvl_environments-per-policy-4_policies-512-1024-2048-4096.json" \
    $C --idea 26_policies-per-process --shapes 4:512,4:1024,4:2048,4:4096 --budget-seconds 780 --arm "the study's learner"
unit "time breakdown at 4 environments per policy and 512 policies" \
    "$R/time_breakdown/time-breakdown_pointmaze-large_environments-per-policy-4_policies-512_h100-nvl.json" \
    benchmarks/harness/profile_study_iteration_pieces.py --copies 512 --envs-per-copy 4 --budget-seconds 720
unit "idea 20 target features once per iteration" \
    "$R/ideas/idea-20_target-features-once-per-iteration_h100-nvl_environments-per-policy-4_policies-512-and-2048.json" \
    $C --idea 20_target-features-once-per-iteration --shapes 4:512,4:2048 --budget-seconds 780 \
    --arm "target features computed in every gradient step" \
    --arm "target features computed once per iteration:target_features_once_per_iteration=True"
unit "idea 25 whole sweep in one program at 960 policies" \
    "$R/ideas/idea-25_whole-sweep-in-one-program_h100-nvl_environments-per-policy-4_policies-960.json" \
    $C --idea 25_whole-sweep-in-one-program --shapes 4:960 --budget-seconds 780 \
    --arm "1 cell (every policy has the same learning rate, entropy coefficient and intrinsic weight)" \
    --arm "8 cells of different learning rate, entropy coefficient and intrinsic weight:sweep_cells=8" \
    --arm "60 cells of different learning rate, entropy coefficient and intrinsic weight:sweep_cells=60"
echo "$(pt) all units tried"
rm -f "$RUN/logs/h100-nvl_units.pid"
