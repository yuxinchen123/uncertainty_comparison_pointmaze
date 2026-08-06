#!/bin/bash
# One short run of one arm at one seed, on one GPU. Twenty of these run at once so the trainer is
# exercised under the concurrency the real sweep will have, and so the records can be compared
# across arms and seeds before committing 150 runs. Logging is at the production cadence of 200
# updates, so this rehearses the real configuration rather than an easier one.
#SBATCH --job-name=rndshort
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
set -uo pipefail
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
P=/p/rlprojects/RND/08_cleanrl_ppo_rnd

# Slurm's pre-walltime warning goes to this shell, and bash does not pass a signal to a foreground
# child. So the trainer runs in the background and the trap forwards the signal to it, which is what
# lets it checkpoint at the next update boundary instead of being killed outright. The real sweep
# depends on exactly this path, so the rehearsal has to use it too.
/p/rlprojects/RND/.venvs/cleanrl_rnd/bin/python -u "$P/src/ppo_rnd_envpool_shuze.py" \
  --arm "$RUN_ARM" --env_id MontezumaRevenge-v5 \
  --total_timesteps "$RUN_STEPS" --num_iterations_obs_norm_init 2 \
  --log_every_updates 200 --checkpoint_every_seconds 300 \
  --output_dir "$PROG_DIR/data/short_run" \
  --run_id "$RUN_ID" --run_total 20 --seed "$RUN_SEED" \
  --opt_env_threads "${SLURM_CPUS_PER_TASK:-8}" --resume &
CHILD=$!
trap 'echo "[slurm] forwarding SIGTERM to $CHILD"; kill -TERM $CHILD' TERM
wait $CHILD
echo "exit=$?"
