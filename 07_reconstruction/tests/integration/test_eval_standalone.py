"""Standalone eval is OFF by default (the run-3.1.1 standard): a default train.py run logs only the
cheap visit-count coverage + the training-episode reward, and --eval_standalone True restores the
deterministic eval rollout that reports the eval/* extrinsic-reward keys. Since 2026-07-04 the
console is also silent by default (sb3_verbose=0): the per-eval blocks print only with
--sb3_verbose 1, so these stdout-parsing tests pass that flag; the per-run JSON is unaffected."""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PY = PROJECT_ROOT / "train.py"


def _run(extra_args):
    """Run a tiny no-exploration train.py job to 800 steps (eval cadence 400); return its stdout."""
    # total_timesteps must be a multiple of eval_freq so the eval callback fires at the final step
    cmd = [
        sys.executable, str(TRAIN_PY),
        "--algorithm", "no_exploration",
        "--total_timesteps", "800",
        "--eval_freq", "400",
        "--n_eval_episodes", "3",
        "--device", "cpu",
        "--use_wandb", "false",
        "--env_max_episode", "25",
        *extra_args,
    ]
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    return result.stdout


def test_eval_standalone_off_by_default():
    """Default run + verbose console: the deterministic eval rollout is skipped (no eval/* reward
    keys), though the eval callback still logs cheap visit-count coverage; the training-episode
    stats block is still logged."""
    # eval_standalone defaults to False -> the rollout never runs; sb3_verbose=1 prints the blocks
    out = _run(["--sb3_verbose", "1"])
    assert "eval/mean_extrinsic_reward" not in out      # no rollout -> no eval extrinsic reward
    assert "eval/n_eval_episodes" not in out
    assert "visit_counts/coverage_pct" in out           # the cheap coverage metric is still logged
    assert "Train episode stats (step 800)" in out      # training-episode reward is logged instead
    assert "Program Finished" in out


def test_eval_standalone_on_restores_rollout():
    """--eval_standalone True restores the rollout; the final eval reports n_eval_episodes (3), with no
    final-eval special case (n_eval_episodes_final is retired)."""
    out = _run(["--eval_standalone", "true", "--sb3_verbose", "1"])
    # the final eval fires at the final step (800 = total_timesteps); locate that block
    marker = "Eval (step 800)"
    assert marker in out, f"final-step eval block missing.\nstdout:\n{out[-2000:]}"
    final_block = out.split(marker, 1)[1][:400]
    assert "eval/mean_extrinsic_reward" in final_block
    assert "eval/n_eval_episodes" in final_block and "3" in final_block, (
        f"final eval should report n_eval_episodes=3.\nblock:\n{final_block}"
    )
    assert "Program Finished" in out


def test_console_silent_by_default():
    """Default sb3_verbose=0: no per-eval block, no train-stats block, no SB3 rollout/train table —
    only the one-time setup block and the finish line reach the console."""
    out = _run([])
    # the blocks the 2026-07-04 silencing removed from the default console
    assert "Eval (step" not in out
    assert "Train episode stats" not in out
    assert "ep_rew_mean" not in out                     # the SB3 periodic table
    assert "Program Finished" in out                    # the run itself still completes normally
