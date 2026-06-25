"""Every registered algorithm runs train.py to completion (CLI smoke test, CPU, few timesteps)."""
import subprocess
import sys
from pathlib import Path

import pytest

from rnd_exploration.methods import ALGORITHM_NAMES  # single source of truth (the registry)

# repo root holds train.py: this file is tests/integration/test_algorithms.py -> parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PY = PROJECT_ROOT / "train.py"

# minimal run: 500 steps, one eval, CPU, no wandb, short episodes
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
def test_train_runs_with_algorithm(algorithm):
    """Run train.py with each algorithm; it must finish with exit code 0."""
    # editable install makes `import rnd_exploration` resolve regardless of cwd, so no PYTHONPATH hack
    assert TRAIN_PY.is_file(), f"train.py not found: {TRAIN_PY}"
    cmd = [sys.executable, str(TRAIN_PY), "--algorithm", algorithm, *MINIMAL_ARGS]
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, (
        f"algorithm={algorithm} failed (exit {result.returncode}).\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "Program Finished" in result.stdout, f"algorithm={algorithm} did not reach the end"
