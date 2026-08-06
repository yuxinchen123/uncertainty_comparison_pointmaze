#!/bin/bash
# Submit worker jobs for this run under YOUR OWN Slurm caps. It only adds workers; it never builds,
# repairs, prunes or cancels anything. DRY=1 previews without submitting.
set -uo pipefail
source /p/rlprojects/RND/08_cleanrl_ppo_rnd/train_runs/2026-08-05-22-30_cleanrl_run_2_ablation_ppo-rnd_montezuma-v5__arm1-orig_arm2-noclip_arm3-prop1_arm4-shallow_arm5-all__2e9step_seed1-30__log200_gradstats_ckpt1h__gpu-gnolim-resv/for_collaborator/packet_env.sh
DRY="${DRY:-0}"

# The owner runs their own launcher; this one refuses to run for them so the two can never both
# submit against the same caps by accident.
[ "$USER" = "sl5nw" ] && { echo "This is the collaborator launcher. Owner: use the owner's path."; exit 1; }

# Stop conditions, checked before anything is submitted.
[ -f "$RUN_DIR/SWEEP_COMPLETE" ] && { echo "SWEEP_COMPLETE exists - the run is finished. Nothing to do."; exit 0; }
pending=$(ls "$RUN_DIR/queue/pending" 2>/dev/null | wc -l)
[ "$pending" -eq 0 ] && { echo "queue pending = 0 - every run is claimed. Nothing to do."; exit 0; }
echo "queue has $pending unclaimed run(s)"

# Never submit more worker slots than there are unclaimed runs, counting slots already in flight
# from every submitter.
live=$( { squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -c -E "^($PREFIX_GPU|$PREFIX_GNOLIM)" ; } || echo 0)
want=$(( pending - live ))
[ "$want" -le 0 ] && { echo "you already have $live worker job(s) live for $pending unclaimed runs. Nothing to do."; exit 0; }
echo "you may add up to $want worker slot(s)"

mkdir -p "$LOGDIR" "$(dirname "$IDFILE")"; touch "$IDFILE"
submit () {  # partition node gres gpus script [extra...]
  local part=$1 node=$2 gres=$3 g=$4 script=$5; shift 5
  local cpus=$(( g * CPUS_PER_RUN )) mem=$(( g * 6000 ))
  local name="$(prefix_for_partition "$part")$g"
  local tl; [ "$part" = "gnolim" ] && tl=20-00:00:00 || tl=4-00:00:00
  if [ "$DRY" = "1" ]; then echo "  DRY sbatch -p $part --nodelist=$node --gres=$gres --cpus-per-task=$cpus --mem=${mem}M -t $tl $script $g"; return; fi
  local id
  id=$(sbatch --parsable --partition="$part" --nodelist="$node" --gres="$gres" \
       --cpus-per-task="$cpus" --mem="${mem}M" --time="$tl" --job-name="$name" \
       --comment="${SWEEP_ID}_${USER}" \
       --output="$LOGDIR/%x_%j.out" --error="$LOGDIR/%x_%j.out" \
       "$SUBMISSION_SCRIPT_DIR/$script" "$g" 2>&1)
  if [[ "$id" =~ ^[0-9]+$ ]]; then
    # flock so two of your own shells appending at once cannot interleave a line.
    ( flock 9; echo "$id $node $g 1 $CPUS_PER_RUN $SWEEP_ID $(date -Is)" >> "$IDFILE" ) 9>>"$IDFILE.lock"
    echo "  submitted $node ($g slot(s)) -> $id"
  else
    echo "  sbatch refused for $node: $id"
  fi
}

echo
echo "Pick nodes with free GPUs from the line below, then edit the submit calls at the bottom."
echo "A submission script exists only for node classes this run supports - if there is no script"
echo "for a class, this run cannot use it, and that is the compatibility verdict. Classes with no"
echo "script here: nekomata (compute capability 12.0, above what the torch build covers)."
echo
sinfo -p gpu,gnolim -N -h -o "%n %G %C" | head -50
echo
echo "Example calls (uncomment and adjust the node and gpu count to what is actually free):"
echo "  submit gpu    cheetah08 gpu:nvidia_rtx_a4000:2 2 cheetah08-09__ppo-rnd-atari.slurm"
echo "  submit gnolim ai07      gpu:nvidia_geforce_gtx_1080_ti:2 2 ai07-08__ppo-rnd-atari.slurm"
echo
echo "Nothing was submitted. Edit this script's last section to add your submit calls."
