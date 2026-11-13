#!/bin/bash
# The gate with trained policies for an AntMaze physics configuration, on an H100 NVL: the policies of a
# training checkpoint are rolled out, without learning, in the environment that a recorded configuration file
# describes (read by the training command's loader, built by the environment's own code), and the record
# says whether the physics held: environment steps, physics-fault respawns, non-finite respawns, the largest
# torso speed and angular speed after any physics step, and the share of steps with the torso above 1.5 m.
# Every ant gate runs this besides the gates with random torques: only trained policies drove the physics
# to the runaway of 2026-11-12 (notes.md of this folder).
#   trained_policy_gate.sh <unit name> <recorded configuration file> [rollout seconds, default 300]
# The checkpoint is the diagnosis's copy (128 policies x 128 environments of the U-maze, iteration 336 of
# 1,024, trained with the physics of before 2026-11-12); CHECKPOINT and UNIT_CONFIGURATION name another one.
# At most 14.5 minutes of wall time. Where it may run: serval05 through its queue as kind profile, or
# serval02 card 1 through serval02_card1_run.sh --kind profile (there the copy under
# /localtmp/sl5nw/antmaze_blow_up is read when it exists, because serval02 reads network storage slowly).
# Records: records/<unit name>.json; log: logs/<unit name>.log. Exit code 90: another process was on the
# card, the measurement is discarded.
set -uo pipefail
UNIT=${1:?unit name}; FILE=${2:?recorded configuration file}; SECONDS_OF_ROLLOUT=${3:-300}
HERE=$(cd "$(dirname "$0")" && pwd)
NAME=rollout-shape-check_antmaze-umaze_reward-arrival-plus-one_cleanrl-ppo-rnd-with-128-environments-per-policy_cells-33-to-40-of-48_seeds-0-to-15.npz
LOCAL=/localtmp/sl5nw/antmaze_blow_up
if [ -z "${CHECKPOINT:-}" ] && [ -s "$LOCAL/$NAME" ] && [ -s "$LOCAL/unit_configuration_cells-33-to-40.toml" ]; then
    export CHECKPOINT="$LOCAL/$NAME" UNIT_CONFIGURATION="$LOCAL/unit_configuration_cells-33-to-40.toml"
fi
# solver and line-search iterations 0: the configuration file's values; no training iteration is timed
exec bash "$HERE/run_rollout_unit.sh" "$UNIT" 0 0 "$SECONDS_OF_ROLLOUT" 0 --environment-from "$FILE" --max-events 0 --max-moderate 0
