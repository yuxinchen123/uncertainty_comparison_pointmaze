"""The three gates of tests/envs/test_antmaze_parity_cpu_mujoco.py for the models of gates_models.json.

Runs on processors inside a Slurm job (never on a login node), through the test runner of this
benchmark run, which calls a function named `main`:
  bash <run folder>/infrastructure/submit_tests.sh -o "python_functions=test_* main" -s \
      <this file, relative to 10_jax_exploration_platform>
gates_models.json beside this folder lists the models in their command-line form
(`field=value,field=value`, or `default`). The results go to gates_on_processors.json there; a
model that was run before is replaced, the others are kept.
"""
import json
import sys
from pathlib import Path

FOLDER = Path(__file__).resolve().parent.parent
PLATFORM = FOLDER.parents[3]


def main():
    """Run the gates for every listed model and write the record."""
    sys.path.insert(0, str(PLATFORM / "benchmarks" / "env"))
    import profile_antmaze_mjx_settings as settings
    import exploration_platform  # noqa: F401  (64-bit statistics on, as the gates expect)
    models = [settings.parse_changes(text) for text in json.loads((FOLDER / "gates_models.json").read_text())]
    settings.measure_gates(models, FOLDER / "gates_on_processors.json")


if __name__ == "__main__":
    main()
