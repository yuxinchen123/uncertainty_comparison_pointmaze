"""At the final training step, eval uses n_eval_episodes_final (distinct from the per-eval count)."""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PY = PROJECT_ROOT / "train.py"


def test_final_step_uses_n_eval_episodes_final():
    """Run train.py to the final step with n_eval_episodes_final=7 (regular n_eval_episodes=2); the
    step-600 eval block must report eval/n_eval_episodes = 7."""
    # total_timesteps must be a multiple of eval_freq for the eval to fire at the final step
    # (the callback only evaluates when num_timesteps % eval_freq == 0); 800 = 2 * 400.
    cmd = [
        sys.executable, str(TRAIN_PY),
        "--algorithm", "no_exploration",
        "--total_timesteps", "800",
        "--eval_freq", "400",
        "--n_eval_episodes", "2",
        "--n_eval_episodes_final", "7",
        "--device", "cpu",
        "--use_wandb", "false",
        "--env_max_episode", "25",
    ]
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    out = result.stdout
    # the final eval fires at the final step (800 = total_timesteps); locate that block
    marker = "Eval (step 800)"
    assert marker in out, f"final-step eval block missing.\nstdout:\n{out[-2000:]}"
    final_block = out.split(marker, 1)[1][:400]
    # the final eval must use n_eval_episodes_final (7), not the regular per-eval count (2)
    assert "eval/n_eval_episodes" in final_block and "7" in final_block, (
        f"final eval should report n_eval_episodes=7.\nblock:\n{final_block}"
    )
    assert "Program Finished" in out
