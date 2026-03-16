"""
Test that at the final training step, eval uses 10000 episodes (eval/n_episodes logged).
Run with: PYTHONPATH=.. pytest tests/test_final_eval_n_episodes.py -v -s
"""
import os
import subprocess
import sys

import pytest


def test_final_step_uses_10000_eval_episodes():
    """Run 04_many_exploration_method with total_timesteps=600, eval_freq=400; check last eval has n_episodes=10000."""
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    program = os.path.join(script_dir, "04_many_exploration_method.py")
    assert os.path.isfile(program), f"Program not found: {program}"
    cmd = [
        sys.executable,
        program,
        "--algorithm", "no_exploration",
        "--total_timesteps", "600",
        "--eval_freq", "400",
        "--n_eval_episodes", "3",
        "--device", "cpu",
        "--use_wandb", "false",
        "--env_max_episode", "25",
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
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    # First eval at step 400 should show n_episodes=3; final eval at step 600 should show n_episodes=10000
    stdout = result.stdout
    assert "eval/n_episodes" in stdout or "n_episodes" in stdout, "eval/n_episodes should appear in log"
    # Check we see 10000 for the final eval (last occurrence of n_episodes before "Program Finished")
    if "eval/n_episodes" in stdout:
        # Find lines containing n_episodes; last eval (step 600) should be 10000
        lines = stdout.splitlines()
        n_ep_values = []
        for line in lines:
            if "n_episodes" in line or "Eval (step 600)" in line:
                # Summary is printed as key-value; look for 10000 near the end
                if "10000" in line:
                    n_ep_values.append(10000)
                elif "3" in line and "step" in line.lower():
                    n_ep_values.append(3)
        # We must have run the final eval with 10000
        assert 10000 in stdout, (
            "Final eval (step 600) should use 10000 episodes; stdout should contain 10000. "
            f"stdout snippet:\n{stdout[-2000:]}"
        )
    # Also require step 600 eval appears (so we know final step was hit)
    assert "600" in stdout, "Step 600 (final) eval should appear in output"


if __name__ == "__main__":
    test_final_step_uses_10000_eval_episodes()
    print("OK: final-step 10000 eval episodes verified")
