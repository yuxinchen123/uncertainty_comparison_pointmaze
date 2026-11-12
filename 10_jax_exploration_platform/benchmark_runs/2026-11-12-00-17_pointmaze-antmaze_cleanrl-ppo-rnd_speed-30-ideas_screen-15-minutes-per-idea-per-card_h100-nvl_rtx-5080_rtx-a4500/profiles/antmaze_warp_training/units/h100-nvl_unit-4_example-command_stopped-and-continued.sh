#!/bin/bash
# EXPECTED_MINUTES=10
# The training command with the Warp example file on the H100 NVL (serval02 card 1), as a user would start it:
# 16 policies x 4 environments, 400 iterations, reward 1 on arrival. The unit is stopped by a 45-second time
# limit, and the same command then continues it from its checkpoint to the end. Checks on the card what the
# processor tests check: the kernel folder in the log, the pool use in the records, save and restore of the
# Warp state. At most 10 minutes of wall time.
set -uo pipefail
START=$(date +%s)
P=/p/rlprojects/RND/10_jax_exploration_platform
OUT=/p/rlprojects/RND/10_jax_exploration_platform/benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/profiles/antmaze_warp_training
RUN=$OUT/example_command_run
PY=/p/rlprojects/RND/.venvs/platform_jax_warp/bin/python
exec >> "$OUT/logs/h100-nvl-unit-4-example-command.log" 2>&1
pt() { TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT'; }
[ "$(hostname -s)" = serval02 ] && [ "${CUDA_VISIBLE_DEVICES:-}" = GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3 ] || {
    echo "$(pt) refused: this unit runs on serval02 card 1 only, started through serval02_card1_run.sh"; exit 1; }
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
echo "$(pt) unit 4 on serval02 card 1"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
rm -rf "$RUN"
COMMAND=($PY $P/scripts/run_training.py --run-dir "$RUN" --unit-id example
         --configuration-file $P/configs/experiments/antmaze_umaze_warp-physics_cleanrl-ppo-rnd_goal-termination_reward-arrival-plus-one_example_steps-204800.toml
         --checkpoint-seconds 30)
echo "$(pt) first attempt, stopped by a time limit of 45 seconds: ${COMMAND[*]} --time-limit-seconds 45"
timeout -s TERM 280 "${COMMAND[@]}" --time-limit-seconds 45; first=$?
echo "$(pt) first attempt ended with exit code $first; checkpoint: $(ls -la "$RUN/checkpoints" 2>/dev/null | tail -1)"
echo "$(pt) second attempt, the same command without the time limit"
timeout -s TERM $((START + 570 - $(date +%s))) "${COMMAND[@]}"; second=$?
echo "$(pt) second attempt ended with exit code $second"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
echo "$(pt) records: $(grep -c . "$RUN/data/example.jsonl") lines; kinds: $(grep -o '"record": "[a-z_]*"' "$RUN/data/example.jsonl" | sort | uniq -c | tr '\n' ' ')"
[ $first -eq 1 ] && [ $second -eq 0 ]
