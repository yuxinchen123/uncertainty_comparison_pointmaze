#!/bin/bash
# Combined per-node probe: measure (1) clean memory latency and (2) real single-thread RND training
# step-rate BACK TO BACK on the same 1-cpu allocation, so the two are directly comparable and see the
# same contention. Prints one line: host lat_ns train_rate_steps_per_min. Used to test whether training
# slowdown tracks the latency ratio, and whether puma01 (high latency) trains normally.
set -u
EXP="$(cd "$(dirname "$0")" && pwd)"
PY="/p/rlprojects/RND/.venvs/exploration/bin/python"
PROJ="/p/rlprojects/RND/07_reconstruction"
TRAIN="$PROJ/train.py"
H="$(hostname)"

# (1) memory latency (single dependent-load pointer chase)
gcc -O3 -o "/tmp/lat_$$" "$EXP/latency.c" 2>/dev/null
LAT="$("/tmp/lat_$$" 256 2>/dev/null | grep -oP '[0-9.]+(?= ns)')"

# (2) single-thread training: the run-5 origsmall arm, fast space_sample warmup so the window is training
D="/tmp/probe_${H}_$$"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$PROJ" || exit 1
"$PY" "$TRAIN" \
  --algorithm=rnd_next_state --beta=1000 --rnd_optimizer=adam --rnd_bonus_readout=mse_mean \
  --rnd_lr=0.0001 --rnd_update_proportion=1.0 --rnd_activation=leaky_relu \
  --rnd_predictor_extra_layers=1 --rnd_obs_warmup_mode=space_sample --rnd_obs_warmup_steps=200 \
  --rnd_reward_norm=True --rnd_reward_norm_gamma=0.99 --rnd_bias_init=zero --rnd_weight_init=orthogonal \
  --total_timesteps=100000000 --eval_freq=50 --n_eval_episodes=3 --z_logging_mode=local --use_wandb=False \
  --a_seed=0 --run_id=0 --run_total=1 --local_log_dir="$D" >/dev/null 2>&1 &
TPID=$!

# read the JSON's max eval step (excludes startup by differencing two reads 140s apart)
read_step() { "$PY" -c "import json,glob
f=glob.glob('$D/local/*.json')
d=json.load(open(f[0])) if f else {}
eh=d.get('eval_history',[])
print(eh[-1].get('step',0) if eh else 0)" 2>/dev/null; }

sleep 60;  A="$(read_step)"          # step after warmup+60s
sleep 140; B="$(read_step)"          # step 140s later -> steady-state rate excludes startup
kill "$TPID" 2>/dev/null; pkill -9 -P "$TPID" 2>/dev/null; kill -9 "$TPID" 2>/dev/null
RATE="$("$PY" -c "print(round((($B)-($A))/140*60))" 2>/dev/null)"
rm -rf "$D" "/tmp/lat_$$"
echo "RESULT host=$H lat_ns=$LAT steps60=$A steps200=$B train_rate_per_min=$RATE"
