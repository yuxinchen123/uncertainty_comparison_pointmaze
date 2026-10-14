#!/bin/bash
set -euo pipefail
umask 0002
TASK_RUN=/p/rlprojects/RND/10_jax_exploration_platform/runs/2026-10-13-22-13_pointmaze_full-batch_steps-100-million_seeds-0-127_rollout-128_environments-4_random-network-distillation_weights-1-10-100-1000_visit-count-square-root_weights-1-10_visit-count-inverse_weight-1_no-bonus_learning-rates-1e-3-1e-4
TASK_CPUS=${1:?CPU threads per program required}
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
export OMP_NUM_THREADS="$TASK_CPUS" OPENBLAS_NUM_THREADS="$TASK_CPUS" MKL_NUM_THREADS="$TASK_CPUS"
export TASK_JOB_DIR="$TASK_RUN/slurm/jobs/${SLURM_JOB_ID}_${SLURMD_NODENAME}"
mkdir -p "$TASK_JOB_DIR"
scontrol write batch_script "$SLURM_JOB_ID" "$TASK_JOB_DIR/job.sbatch"
/usr/bin/python3 - <<'META'
import json,os,subprocess,socket
from pathlib import Path
folder=Path(os.environ['TASK_JOB_DIR'])
tokens=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
card=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).splitlines()[0]
(folder/'hardware.json').write_text(json.dumps({'gpu':str(len(tokens))+' '+card,'host':socket.gethostname(),'gpu_tokens':tokens,'job_id':os.environ['SLURM_JOB_ID']},indent=2))
(folder/'assignment.json').write_text(json.dumps({'mode':'claim from shared queue','gpu_tokens':tokens},indent=2))
META
if [ -f "$TASK_RUN/SWEEP_COMPLETE" ]; then exit 0; fi
exec /usr/bin/python3 /p/rlprojects/.claude/skills/uva-submit-gpu-sweep/scripts/worker_manager.py --sweep_dir "$TASK_RUN" --slots_per_gpu 1 --cpus_per_run "$TASK_CPUS" --sample_seconds 120
