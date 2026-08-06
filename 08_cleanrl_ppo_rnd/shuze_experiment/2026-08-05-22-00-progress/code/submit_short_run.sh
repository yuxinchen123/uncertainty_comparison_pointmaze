#!/bin/bash
# Launch twenty short runs at once: 4 seeds x 5 arms, one GPU each, spread over three GPU classes.
# Run ids follow the real sweep's ordering (seed outermost, arm inner) so this also exercises that
# a prefix of the queue is arm-balanced.
set -uo pipefail
PROG=/p/rlprojects/RND/08_cleanrl_ppo_rnd/shuze_experiment/2026-08-05-22-00-progress
IDFILE="$PROG/slurm/submitted_jobids_short_run.txt"; touch "$IDFILE"
ARMS=(arm1_original arm2_no_rnd_grad_clip arm3_update_proportion_1 arm4_shallower_predictor arm5_all)
# Twenty GPU slots: node and its typed gres, repeated once per GPU taken on that node.
SLOTS=(
  "lotus:quadro_rtx_6000" "lotus:quadro_rtx_6000" "lotus:quadro_rtx_6000" "lotus:quadro_rtx_6000"
  "lotus:quadro_rtx_6000" "lotus:quadro_rtx_6000" "lotus:quadro_rtx_6000" "lotus:quadro_rtx_6000"
  "cheetah08:nvidia_rtx_a4000" "cheetah08:nvidia_rtx_a4000" "cheetah08:nvidia_rtx_a4000" "cheetah08:nvidia_rtx_a4000"
  "cheetah09:nvidia_rtx_a4000" "cheetah09:nvidia_rtx_a4000" "cheetah09:nvidia_rtx_a4000" "cheetah09:nvidia_rtx_a4000"
  "cheetah02:nvidia_rtx_4000_ada_generation" "cheetah02:nvidia_rtx_4000_ada_generation"
  "cheetah02:nvidia_rtx_4000_ada_generation" "cheetah02:nvidia_rtx_4000_ada_generation"
)
STEPS=${STEPS:-13107200}     # 800 policy updates = 4 logged rows at cadence 200
n=0
for seed_index in 0 1 2 3; do
  for arm_index in 0 1 2 3 4; do
    run_id=$(( seed_index * 5 + arm_index ))
    seed=$(( seed_index + 1 ))
    arm=${ARMS[$arm_index]}
    slot=${SLOTS[$n]}; node=${slot%%:*}; gres=${slot##*:}
    id=$(sbatch --parsable --partition=gpu --nodelist="$node" --gres=gpu:${gres}:1 \
         --cpus-per-task=8 --mem=16G \
         --export=ALL,PROG_DIR="$PROG",RUN_ARM="$arm",RUN_ID="$run_id",RUN_SEED="$seed",RUN_STEPS="$STEPS" \
         --output="$PROG/logs/short_${run_id}_%j.log" --error="$PROG/logs/short_${run_id}_%j.log" \
         "$PROG/code/short_run.slurm" 2>&1)
    if [[ "$id" =~ ^[0-9]+$ ]]; then
      echo "$id" >> "$IDFILE"
      printf "  run %2d  seed %d  %-26s %-10s -> %s\n" "$run_id" "$seed" "$arm" "$node" "$id"
    else
      printf "  run %2d  %-26s %-10s REFUSED: %s\n" "$run_id" "$arm" "$node" "$id"
    fi
    n=$((n+1))
  done
done
echo "submitted $(wc -l < "$IDFILE") jobs; ids in $IDFILE"
