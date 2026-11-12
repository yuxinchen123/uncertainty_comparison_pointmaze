#!/bin/bash
# One unit of the training-iteration measurement with the Warp AntMaze physics: at most 15 minutes of wall
# time on one card, startup and kernel compilation included.
#   run_cells.sh "<unit name>" <card slug> "<card name>" <policies>x<environments per policy> [<cell> ...]
# Each cell is one Python process (benchmarks/env/profile_antmaze_warp_training.py), run one after the other,
# so the memory a process holds on the card belongs to one cell. A cell gets an equal part of the time that
# is left; a cell that cannot start before the unit's end gets a record that says so.
# Output is appended to logs/<unit name>.log of the profile folder; records go to records/; the Markdown
# table antmaze_warp_training.md is written again at the end.
# The interpreter is platform_jax_warp (platform_jax plus warp-lang 1.14.0). WARP_CACHE_PATH is not set: the
# environment chooses the node-local kernel folder itself, as a training unit does.
# Where it may run: serval02 card 1 (started through serval02_card1_run.sh, which pins the card by its UUID),
# serval05 (through its queue), or a Slurm job that holds one card.
set -uo pipefail
START=$(date +%s)
UNIT=${1:?unit name}; CARD_SLUG=${2:?card slug}; CARD_NAME=${3:?card name}; shift 3
[ $# -gt 0 ] || { echo "no cells given" >&2; exit 1; }
P=/p/rlprojects/RND/10_jax_exploration_platform
OUT=$P/benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/profiles/antmaze_warp_training
PY=/p/rlprojects/RND/.venvs/platform_jax_warp/bin/python
SCRIPT=$P/benchmarks/env/profile_antmaze_warp_training.py
SLUG=$(echo "$UNIT" | tr -c 'A-Za-z0-9\n' '-')
exec >> "$OUT/logs/$SLUG.log" 2>&1
pt() { TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT'; }

# GOAL: name the machine and card; on serval02 only card 1, pinned by its UUID, is ever used.
CARD1=GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3
case "$(hostname -s)" in
    serval05) MACHINE="serval05";;
    serval02) MACHINE="serval02 card 1"
              [ "${CUDA_VISIBLE_DEVICES:-}" = "$CARD1" ] || { echo "$(pt) refused: on serval02 the unit must be started through serval02_card1_run.sh (card 1 only)"; exit 1; };;
    *) [ -n "${SLURM_JOB_ID:-}" ] || { echo "$(pt) refused: $(hostname -s) is used through a Slurm job only"; exit 1; }
       MACHINE="$(hostname -s), Slurm job $SLURM_JOB_ID"
       # on the Slurm card nodes JAX finds the card's libraries with this folder on the path (infrastructure/profile_job.sh)
       export LD_LIBRARY_PATH=/usr/lib/nvidia:${LD_LIBRARY_PATH:-};;
esac
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
echo "$(pt) unit \"$UNIT\" on $MACHINE, cells: $*"
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv,noheader
echo "processes on the cards before the unit:"; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader

# GOAL: one process per cell, stopped 14.5 minutes after the unit started whatever it is doing.
cell_arguments() { echo --records "$OUT/records" --policies "${1%x*}" --environments-per-policy "${1#*x}" --card-slug "$CARD_SLUG" ; }
worst=0; cells_left=$#
for cell in "$@"; do
    left=$((START + 870 - $(date +%s)))
    if [ $left -lt 150 ]; then
        echo "$(pt) not started, $left s of the unit are left: $cell"
        $PY $SCRIPT $(cell_arguments "$cell") --machine "$MACHINE" --unit "$UNIT" --card-name "$CARD_NAME" \
            --not-measured-reason "$left s were left of the 15-minute unit when this cell's turn came"
        cells_left=$((cells_left - 1)); continue
    fi
    budget=$((left / cells_left)); cells_left=$((cells_left - 1))
    echo "$(pt) cell $cell: $budget s of the $left s that are left"
    timeout -s TERM $left $PY -u $SCRIPT $(cell_arguments "$cell") --machine "$MACHINE" --unit "$UNIT" --unit-seconds $budget
    rc=$?
    if [ $rc -eq 76 ] && [ -z "${SLURM_JOB_ID:-}" ]; then
        echo "$(pt) JAX did not load the card; trying once more with /usr/lib/nvidia on LD_LIBRARY_PATH"
        export LD_LIBRARY_PATH=/usr/lib/nvidia${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
        timeout -s TERM $((START + 870 - $(date +%s))) $PY -u $SCRIPT $(cell_arguments "$cell") --machine "$MACHINE" --unit "$UNIT" --unit-seconds $budget
        rc=$?
    fi
    [ $rc -eq 124 ] && echo "$(pt) the cell was stopped at the unit's time limit"
    [ $rc -eq 77 ] && echo "$(pt) the card's memory ran out in this cell; its record says so"
    echo "$(pt) cell $cell ended with exit code $rc"
    [ $rc -gt $worst ] && worst=$rc
done
echo "processes on the cards after the unit:"; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader

# GOAL: the table is written again from every record, also when a cell was stopped.
$PY $SCRIPT --render --records "$OUT/records" --markdown "$OUT/antmaze_warp_training.md"
echo "$(pt) unit \"$UNIT\" ended with exit code $worst after $(( $(date +%s) - START )) s"
exit $worst
