"""
Check that 04_many_exploration_method.py runs to completion for every algorithm.
Runs the script via CLI with minimal env (few timesteps, CPU, no wandb).
Uses the same algorithm list as distance_to_GT.algorithm_vector.ALGORITHM_NAMES.
"""
import os
import subprocess
import sys

import pytest

# Mirror ALGORITHM_NAMES to avoid importing distance_to_GT (which may pull in other deps)
ALGORITHM_NAMES = [
    "no_exploration",
    "gt_position",
    "gt_position_velocity",
    "rnd_next_state",
    "rnd_next_state_position_only",
    "rnd_state",
    "rnd_state_action",
    "rnd_state_action_next_state",
    "rnd_linear_next_state",
    "rnd_elliptical",
]


# Minimal run: 500 steps, eval once at 400, 1 eval episode, CPU, no wandb
MINIMAL_ARGS = [
    "--total_timesteps", "500",
    "--eval_freq", "400",
    "--n_eval_episodes", "1",
    "--device", "cpu",
    "--use_wandb", "false",
    "--beta", "0.01",
    "--env_max_episode", "50",
]


@pytest.mark.parametrize("algorithm", ALGORITHM_NAMES)
def test_04_runs_with_algorithm(algorithm):
    """Run 04_many_exploration_method.py with each algorithm; must finish with exit 0."""
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    program = os.path.join(script_dir, "04_many_exploration_method.py")
    assert os.path.isfile(program), f"Program not found: {program}"
    cmd = [
        sys.executable,
        program,
        "--algorithm", algorithm,
        *MINIMAL_ARGS,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = script_dir + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        cmd,
        cwd=script_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"algorithm={algorithm} failed (exit {result.returncode}). "
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
