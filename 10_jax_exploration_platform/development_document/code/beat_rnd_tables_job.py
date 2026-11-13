"""Run beat_rnd_tables.py inside the test helper's Slurm job (numpy never runs on a login node), and its tests.

  bash "benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/infrastructure/submit_tests.sh" \
      -s development_document/code/beat_rnd_tables_job.py development_document/code/test_beat_rnd_tables.py

`-s` prints every spliced block into the job's log, so the numbers can be read without opening the .tex.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import beat_rnd_tables  # noqa: E402


def test_generate_beat_rnd_section():
    """Splice every table and number of the section and draw its figures; print what was written."""
    result = beat_rnd_tables.main()
    for name, block in result['blocks'].items():
        print(f'\n===== {name} =====\n{block}')
    print(f"\nfigures: {result['figures']}")
    print(f"ant budget rule: {result['budgets']}")
