#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
export OMP_NUM_THREADS=8
B=/p/rlprojects/RND/08_cleanrl_ppo_rnd/bench_scratch
PY=/p/rlprojects/RND/.venvs/exploration/bin/python
TAG=$1
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
mkdir -p $B/results/$TAG
for C in baseline bench_only cudnn_tf32_only matmul_tf32_only tf32_bench fused_tf32_bench \
         tf32_bench_channels_last amp_bf16 amp_bf16_channels_last amp_fp16 amp_fp16_channels_last \
         tf32_bench_cudagraph tf32_bench_compile_rollout tf32_bench_compile_update ; do
  echo "### $C"
  timeout 900 $PY -u $B/bench_isolate.py --config $C --out $B/results/$TAG/$C.json 2>&1 | tail -40
done
