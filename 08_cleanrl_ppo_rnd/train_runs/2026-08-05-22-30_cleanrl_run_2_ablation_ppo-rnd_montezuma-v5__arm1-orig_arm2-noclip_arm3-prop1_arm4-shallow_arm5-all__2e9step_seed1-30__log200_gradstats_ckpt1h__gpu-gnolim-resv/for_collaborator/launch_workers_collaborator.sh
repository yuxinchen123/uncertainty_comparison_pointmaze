#!/bin/bash
# Submit worker jobs for this run under YOUR OWN Slurm caps. It only adds workers; it never builds,
# repairs, prunes or cancels anything.
#
#   bash launch_workers_collaborator.sh                      # show what is free, submit nothing
#   NODES="cheetah08:4 ai07:3" bash launch_workers_collaborator.sh          # submit those
#   DRY=1 NODES="cheetah08:4" bash launch_workers_collaborator.sh          # print the sbatch lines only
#
# You pass the nodes; the script works out everything else — which submission script serves that
# node, how many runs per GPU it packs, and therefore how many cpus and how much memory to ask for.
# You never have to edit this file, and you never have to know the node-to-script mapping.
set -uo pipefail
source /p/rlprojects/RND/08_cleanrl_ppo_rnd/train_runs/2026-08-05-22-30_cleanrl_run_2_ablation_ppo-rnd_montezuma-v5__arm1-orig_arm2-noclip_arm3-prop1_arm4-shallow_arm5-all__2e9step_seed1-30__log200_gradstats_ckpt1h__gpu-gnolim-resv/for_collaborator/packet_env.sh
DRY="${DRY:-0}"
NODES="${NODES:-}"

# The owner runs their own launcher; this one refuses to run for them so the two can never both
# submit against the same caps by accident.
[ "$USER" = "sl5nw" ] && { echo "This is the collaborator launcher. Owner: use the owner's path."; exit 1; }

# Stop conditions, checked before anything is submitted.
[ -f "$RUN_DIR/SWEEP_COMPLETE" ] && { echo "SWEEP_COMPLETE exists - the run is finished. Nothing to do."; exit 0; }
pending=$(ls "$RUN_DIR/queue/pending" 2>/dev/null | wc -l)
[ "$pending" -eq 0 ] && { echo "queue pending = 0 - every run is claimed. Nothing to do."; exit 0; }

# How many worker slots you already have in flight. Counted from YOUR OWN id file, cross-checked
# against squeue by exact job id — never by job name. A name can be reused by another session, and
# a name-matching count also counts jobs rather than run slots, which is wrong by up to the packing
# factor: one job on the reserved node holds 24 slots, not 1.
# The id file may hold lines of two widths: the documented eight fields
#   <jobid> <node_class> <node> <G> <W> <c_used> <group> <submit_ts>
# and seven-field lines written before that was fixed, which omit <node_class>. Both are read by
# field COUNT rather than by a fixed index, so an old line is not silently misread as a new one —
# reading G and W from the wrong positions would miscount the slots this guard exists to bound.
live=0
if [ -s "$IDFILE" ]; then
  while read -r -a f; do
    [ "${#f[@]}" -ge 7 ] || continue
    case "${f[0]}" in ''|\#*) continue;; esac
    [[ "${f[0]}" =~ ^[0-9]+$ ]] || continue
    if [ "${#f[@]}" -ge 8 ]; then g="${f[3]}"; w="${f[4]}"; else g="${f[2]}"; w="${f[3]}"; fi
    [[ "$g" =~ ^[0-9]+$ && "$w" =~ ^[0-9]+$ ]] || continue   # smoke and monitor rows carry labels
    state=$(squeue -h -j "${f[0]}" -o "%T" 2>/dev/null)
    case "$state" in RUNNING|PENDING|COMPLETING|CONFIGURING) live=$(( live + g * w ));; esac
  done < "$IDFILE"
fi
want=$(( pending - live ))
echo "queue has $pending unclaimed run(s); you hold $live worker slot(s) in flight"
[ "$want" -le 0 ] && { echo "your slots already cover every unclaimed run. Nothing to do."; exit 0; }
echo "you may add up to $want worker slot(s)"

# Which submission script serves a node. The node-to-class map is derived from the generated script
# names, so a node this run cannot use resolves to nothing and is refused rather than mis-submitted.
# That is what makes the excluded classes an enforced rule and not just a sentence in the README.
script_for () {
  local node="$1" f cls
  for f in "$SUBMISSION_SCRIPT_DIR"/*.slurm; do
    cls="$(basename "$f")"; cls="${cls%%__*}"
    case "_${cls}_" in *"_${node}_"*) echo "$f"; return;; esac
    if [[ "$cls" == "$node" ]]; then echo "$f"; return; fi
    if [[ "$cls" =~ ^([a-z]+)([0-9]+)-([0-9]+)$ ]]; then
      local base="${BASH_REMATCH[1]}" lo="${BASH_REMATCH[2]}" hi="${BASH_REMATCH[3]}"
      if [[ "$node" =~ ^${base}([0-9]+)$ ]]; then
        local n="${BASH_REMATCH[1]}"
        if (( 10#$n >= 10#$lo && 10#$n <= 10#$hi )); then echo "$f"; return; fi
      fi
    fi
  done
}

# Nodes inside the owner's Slurm reservation. A job of yours pinned there without the reservation
# pends forever with no error, so it is refused up front.
RESERVED="$(scontrol show reservation -o 2>/dev/null | grep -oP 'Nodes=\K\S+' | tr ',\n' '  ')"

mkdir -p "$LOGDIR"; touch "$IDFILE"
submitted=0; slots=0
for spec in $NODES; do
  node="${spec%%:*}"; g="${spec##*:}"
  script="$(script_for "$node")"
  if [[ -z "$script" ]]; then
    echo "  $node: NO SUBMISSION SCRIPT — this run cannot use it (wrong GPU generation). Skipped."
    continue
  fi
  case " $RESERVED " in *" $node "*)
    echo "  $node: inside the owner's Slurm reservation — your job would pend forever there. Skipped."
    continue;; esac

  # Every number comes from the script's own header, so the packing decision has one home and this
  # launcher never recomputes it. W is the runs-per-GPU factor: without it, a packed node is asked
  # for a third of the cpus and memory it will actually use, which oversubscribes the cpus and
  # invites a memory kill.
  part="$(grep -oP '^#SBATCH --partition=\K\S+' "$script")"
  gres_type="$(grep -oP '^#SBATCH --gres=gpu:\K[^:]+' "$script")"
  total_g="$(grep -oP '^#SBATCH --gres=gpu:[^:]+:\K[0-9]+' "$script")"
  W="$(grep -oP '^  --slots_per_gpu \K[0-9]+' "$script")"
  C="$(grep -oP '^  --cpus_per_run \K[0-9]+' "$script")"
  mem_total="$(grep -oP '^#SBATCH --mem=\K[0-9]+' "$script")"
  mem_per_slot=$(( mem_total / (W * total_g) ))
  cpus=$(( W * g * C )); mem=$(( W * g * mem_per_slot ))
  tl=4-00:00:00; [ "$part" = "gnolim" ] && tl=20-00:00:00
  name="$(prefix_for_partition "$part")$g"

  cmd=(sbatch --parsable --partition="$part" --nodelist="$node" --gres=gpu:"$gres_type":"$g"
       --cpus-per-task="$cpus" --mem="${mem}M" --time="$tl" --job-name="$name"
       --comment="${SWEEP_ID}_${USER}"
       --output="$LOGDIR/%x_%j.out" --error="$LOGDIR/%x_%j.out" "$script" "$g")
  if [ "$DRY" = "1" ]; then
    printf '  DRY %s   [%d slot(s): %d GPU x %d run/GPU]\n' "${cmd[*]}" "$(( W * g ))" "$g" "$W"
    slots=$(( slots + W * g )); continue
  fi
  id="$("${cmd[@]}" 2>&1)"
  if [[ "$id" =~ ^[0-9]+$ ]]; then
    # flock so two of your own shells appending at once cannot interleave a line.
    # The id-file line must be the documented eight fields, in order:
    #   <jobid> <node_class> <node> <G> <W> <c_used> <group> <submit_ts>
    # A line of the wrong width does not merely look odd — the monitor reads these by position, so a
    # missing field shifts every later one and the node column ends up holding a GPU count, which
    # then appears in the infra table as a phantom node and double-counts its cpus.
    node_class="$(basename "$script")"; node_class="${node_class%%__*}"
    ( flock 9; echo "$id $node_class $node $g $W $C $SWEEP_ID $(date -Is)" >> "$IDFILE" ) 9>>"$IDFILE.lock"
    printf "  %-12s %d GPU x %d run/GPU = %2d slot(s), %3d cpus, %6sM, %-6s -> %s\n" \
      "$node" "$g" "$W" "$(( W * g ))" "$cpus" "$mem" "$part" "$id"
    submitted=$(( submitted + 1 )); slots=$(( slots + W * g ))
  else
    echo "  $node: sbatch refused: $id"
  fi
done

if [ -z "$NODES" ]; then
  echo
  echo "Free GPUs right now. The partition column matters: a node is in ONE partition, and pairing"
  echo "a node with the wrong -p means the job simply never runs."
  # sinfo rejects a '-' left-justify flag in its format string, so the columns are laid out here
  # rather than by sinfo. Its own error ("Invalid node format specification: -") is printed once per
  # node and buries the table, which is the first thing a collaborator sees.
  printf '  %-12s %-8s %-36s %s\n' NODE PARTITION GPUS "CPUS alloc/idle/other/total"
  sinfo -p gpu,gnolim -N -h -o "%n %P %G %C" | sort -u \
    | awk '{printf "  %-12s %-8s %-36s %s\n", $1, $2, $3, $4}'
  echo
  echo "Then re-run with the nodes you want, for example:"
  echo "  NODES=\"cheetah08:4 ai07:3\" bash \$0"
  echo "Nothing was submitted."
  exit 0
fi
echo "submitted $submitted job(s), $slots worker slot(s); id file holds $(grep -cE '^[0-9]+' "$IDFILE") id(s)"
