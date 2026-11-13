#!/bin/bash
# EXPECTED_MINUTES=9
# Blow-up diagnosis, unit 14: the gate with trained policies for the physics the AntMaze baseline runs with
# since 2026-11-12, through the recorded configuration file and the environment's own code (no patched model
# builder): time step 0.005 s, 10 physics steps per environment step, solver 4 and line search 8, the fault
# guard at 30 m/s and 100 rad/s. The 128 policies of the diagnosis's checkpoint are rolled out for 300 s.
exec bash /p/rlprojects/RND/10_jax_exploration_platform/benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/profiles/antmaze_warp_blow_up/units/trained_policy_gate.sh serval02-card-1_unit-14_trained-policy-gate_time-step-0-005_frame-skip-10_fault-guard-30-metres-and-100-radians-per-second /p/rlprojects/RND/10_jax_exploration_platform/configs/experiments/antmaze_umaze_warp-physics_time-step-0.005_frame-skip-10_fault-guard-30-metres-and-100-radians-per-second_cleanrl-ppo-rnd_goal-termination_reward-arrival-plus-one_example_steps-204800.toml 300
