#!/bin/bash
# serval02 card 1 (while serval05 is busy): four-law bench of batch 2's ideas, one unit per lock.
U=GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3
P=/p/rlprojects/RND/10_jax_exploration_platform
export PYTHONPATH=$P/src PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
for part in "$@"; do
  flock /tmp/sl5nw_serval02_card1.lock bash -c "
    read MIG USED <<<\"\$(nvidia-smi -i $U --query-gpu=mig.mode.current,memory.used --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' ')\"
    PROCS=\$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c $U)
    if [ \"\$MIG\" != Disabled ] || [ \"\$USED\" -ge 1024 ] || [ \"\$PROCS\" -ne 0 ]; then echo 'serval02 card 1 busy; unit not run'; exit 75; fi
    date; echo 'machine: serval02 card 1'
    CUDA_VISIBLE_DEVICES=$U timeout 1750 /p/rlprojects/RND/.venvs/platform_jax/bin/python -u $P/scripts/grid_initialization_campaign/rnd_decay_bench.py --ideas /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-11-09-19-42_online-tabular-rnd-auto-research_reward-normalization-on_even-start-and-decay-exponent-near-minus-0.5_straight-32-snake-16x2_tabular-ppo-coordinate-ppo_five-days/batches/02_stable-shrink-target-exponents-and-count-rate/bench_ideas.json --output /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-11-09-19-42_online-tabular-rnd-auto-research_reward-normalization-on_even-start-and-decay-exponent-near-minus-0.5_straight-32-snake-16x2_tabular-ppo-coordinate-ppo_five-days/bench/batch_02_four_laws --only \$(cat /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-11-09-19-42_online-tabular-rnd-auto-research_reward-normalization-on_even-start-and-decay-exponent-near-minus-0.5_straight-32-snake-16x2_tabular-ppo-coordinate-ppo_five-days/batches/02_stable-shrink-target-exponents-and-count-rate/bench_names_part_$part.txt)
  " > /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-11-09-19-42_online-tabular-rnd-auto-research_reward-normalization-on_even-start-and-decay-exponent-near-minus-0.5_straight-32-snake-16x2_tabular-ppo-coordinate-ppo_five-days/logs/bench_serval02_batch2_part_$part.log 2>&1
  echo "batch 2 part $part exit $?" >> /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-11-09-19-42_online-tabular-rnd-auto-research_reward-normalization-on_even-start-and-decay-exponent-near-minus-0.5_straight-32-snake-16x2_tabular-ppo-coordinate-ppo_five-days/logs/bench_serval02_runner.log
done
