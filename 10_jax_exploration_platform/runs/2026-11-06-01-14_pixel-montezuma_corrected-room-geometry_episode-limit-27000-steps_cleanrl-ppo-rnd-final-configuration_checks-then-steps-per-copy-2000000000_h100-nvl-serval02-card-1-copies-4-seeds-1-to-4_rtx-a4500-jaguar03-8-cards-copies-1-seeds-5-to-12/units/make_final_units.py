"""Write the final-training units: one Slurm job per RTX A4500 seed and one serval02 card-1 supervisor.

Every unit runs train_final.py to 122,070 updates (two billion training steps per copy), checkpoints
every 20 minutes, and resumes from its checkpoint when started again. Prints the paths written.

A4500 (jaguar03, reservation sl5nw_168): one copy per card, seeds 5 to 12, a 4-day job each. The
trainer stops and checkpoints 20 minutes before the time limit; a failed attempt is retried in the
same job up to three times, each resuming from the last checkpoint.

H100 NVL (serval02 card 1, study exception of plan section 6): four packed copies, seeds 1 to 4. The
supervisor holds the card-1 lock for the whole run. Once a minute it looks for any other process on
card 1; if one appears it stops the trainer (which checkpoints on SIGTERM), waits until the card is
free again, and resumes. It never touches another user's process.
"""
import shlex
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
PLATFORM = RUN.parents[1]
PYTHON = "/p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python"
ASSETS = "/p/rlprojects/RND/reference_repos/jaxtari_pixel_study_assets_montezuma_room_geometry_from_png"
CARD = "GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3"
ENVIRONMENT = ('{"separable_rectangle_drawing": true, "unroll_fixed_render_loops": true, '
               '"masked_conditional_draws": true, "two_way_masked_draws": true, "render_final_two_together": true, '
               '"max_episode_steps": 27000}')
LEARNER = {
    "h100": ('{"convolution_algorithm": "space_to_depth", "cache_target_features": true, "predictor_selected_rows": 1280, '
             '"iteration_compiler_options": [["xla_gpu_enable_while_loop_double_buffering", "true"]]}'),
    "a4500": '{"convolution_batching": "unrolled", "cache_target_features": true, "predictor_selected_rows": 1280}',
}
TARGET_UPDATES = 122070
JOB_SECONDS = 4 * 24 * 3600
COMMON = f"""set -u
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAXTARI_SKIP_SPRITE_UPDATE=1 XDG_DATA_HOME={ASSETS} XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export PYTHONPATH={PLATFORM}/src
cd {PLATFORM}
stamp() {{ TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT'; }}"""


def train_command(gpu, copies, seed, data, limit_seconds):
    """The resumable training command for one unit."""
    limit = f" --time-limit-seconds {limit_seconds}" if limit_seconds else ""
    return (f"{PYTHON} -u scripts/atari_pixel_campaign/train_final.py --run-dir {data}/run --copies {copies} "
            f"--seed-start {seed} --target-updates {TARGET_UPDATES} --observation-warmup-iterations 50 "
            f"--checkpoint-seconds 1200{limit} --learner {shlex.quote(LEARNER[gpu])} "
            f"--environment {shlex.quote(ENVIRONMENT)}")


def a4500_job(seed):
    """One Slurm job: one copy on one jaguar03 card."""
    unit = f"a4500-jaguar03_final-training_seed-{seed}"
    data, jobs = RUN / "data" / unit, RUN / "slurm" / "jobs" / unit
    jobs.mkdir(parents=True, exist_ok=False)
    command = train_command("a4500", 1, seed, data, JOB_SECONDS - 1200)
    script = f"""#!/bin/bash
#SBATCH --job-name=pixel-final-seed-{seed}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --partition=gpu
#SBATCH --nodelist=jaguar03
#SBATCH --account=sl5nw_base
#SBATCH --qos=csresnolim
#SBATCH --reservation=sl5nw_168
#SBATCH --gres=gpu:nvidia_rtx_a4500:1
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output={jobs}/%j.out
# Final training, RTX A4500, one copy, seed {seed}, 122,070 updates; resumes from {data}/run/checkpoint.pkl.
{COMMON}
export JAX_COMPILATION_CACHE_DIR=/tmp/sl5nw_pixel_final_cache_a4500
mkdir -p {data} "$JAX_COMPILATION_CACHE_DIR"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.used --format=csv,noheader >> {data}/hardware.txt
for attempt in 1 2 3 4; do
  echo "$(stamp) attempt $attempt job $SLURM_JOB_ID host $(hostname)" >> {data}/unit.log
  {command} >> {data}/unit.log 2>&1
  rc=$?
  echo "$(stamp) attempt $attempt exit $rc" >> {data}/unit.log
  grep -q '"completed": true' {data}/run/progress.json 2>/dev/null && exit 0
  [ $rc -eq 0 ] && exit 0
  sleep 60
done
exit 1
"""
    (jobs / "job.sbatch").write_text(script)
    return jobs / "job.sbatch"


def h100_supervisor():
    """The serval02 card-1 supervisor for the four packed H100 copies."""
    unit = "h100-serval02-card-1_final-training_seeds-1-to-4"
    data = RUN / "data" / unit
    command = train_command("h100", 4, 1, data, 0)
    script = f"""#!/bin/bash
# Final training, H100 NVL serval02 card 1, four packed copies, seeds 1 to 4, 122,070 updates.
# Holds the card-1 lock for the whole run (study exception, plan section 6); yields the card if any
# other process appears on it and resumes from the checkpoint once it is free again.
{COMMON}
export JAX_COMPILATION_CACHE_DIR=/tmp/sl5nw_pixel_final_cache_h100
export CUDA_VISIBLE_DEVICES={CARD}
U={CARD}
DATA={data}
mkdir -p $DATA "$JAX_COMPILATION_CACHE_DIR"
echo $$ > $DATA/supervisor.pid
card_free() {{
  read MIG USED <<<"$(nvidia-smi -i $U --query-gpu=mig.mode.current,memory.used --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' ')"
  PROCS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$U")
  CC=$(nvidia-smi conf-compute -f | grep -c 'CC status: OFF')
  [ "$MIG" = Disabled ] && [ "$USED" -lt 1024 ] && [ "$PROCS" -eq 0 ] && [ "$CC" -eq 1 ]
}}
others_on_card() {{
  # Processes on card 1 other than our trainer ($1).
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep "$U" | tr -d ' ' | cut -d, -f2 | grep -v "^$1\\$" | wc -l
}}
exec 9>/tmp/sl5nw_serval02_card1.lock
flock -w 7200 9 || {{ echo "$(stamp) card-1 lock not obtained" >> $DATA/unit.log; exit 4; }}
nvidia-smi -i $U --query-gpu=name,uuid,driver_version --format=csv,noheader >> $DATA/hardware.txt
failures=0
while ! grep -q '"completed": true' $DATA/run/progress.json 2>/dev/null; do
  until card_free; do echo "$(stamp) card 1 not free; waiting" >> $DATA/unit.log; sleep 300; done
  echo "$(stamp) starting trainer (failures so far $failures)" >> $DATA/unit.log
  {command} >> $DATA/unit.log 2>&1 &
  trainer=$!
  echo $trainer > $DATA/trainer.pid
  yielded=0
  while kill -0 $trainer 2>/dev/null; do
    sleep 60
    if [ "$(others_on_card $trainer)" -gt 0 ]; then
      echo "$(stamp) another process appeared on card 1; stopping our trainer (it checkpoints)" >> $DATA/unit.log
      nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | grep "$U" >> $DATA/unit.log
      kill -TERM $trainer; yielded=1; break
    fi
  done
  wait $trainer; rc=$?
  echo "$(stamp) trainer exit $rc" >> $DATA/unit.log
  if [ $yielded -eq 0 ] && [ $rc -ne 0 ]; then
    failures=$((failures + 1))
    [ $failures -ge 5 ] && {{ echo "$(stamp) five failures; supervisor stops" >> $DATA/unit.log; exit 1; }}
    sleep 120
  fi
done
echo "$(stamp) training complete" >> $DATA/unit.log
rm -f $DATA/supervisor.pid $DATA/trainer.pid
"""
    path = RUN / "units" / f"{unit}.sh"
    path.write_text(script)
    path.chmod(0o755)
    return path


if __name__ == "__main__":
    print(h100_supervisor())
    for seed in range(5, 13):
        print(a4500_job(seed))
