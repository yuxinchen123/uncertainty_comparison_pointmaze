"""Generate a run folder's standing page inside the test helper's Slurm job, since numpy never runs on a login node.

  bash "<speed-study benchmark run>/infrastructure/submit_tests.sh" -s \
      scripts/maze_campaign/analysis/standing_job.py --run-dir=runs/<run folder>

The helper runs pytest in the platform folder, so the work is done by this file's one function. `-s` prints what it
found into the job's log. It writes <run folder>/analysis/standing.md and standing.html from the settings in
<run folder>/analysis/standing_settings.json.
"""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from maze_campaign.analysis import standing  # noqa: E402


def test_generate_standing_page(request):
    """Generate the standing page of the run folder named on the command line and print its tables' rows."""
    run_dir = request.config.getoption('--run-dir')
    if not run_dir:
        pytest.skip('name a run folder with --run-dir=<run folder>')
    result = standing.generate(Path(run_dir).resolve())
    print(f"\nwrote {result['page']}")
    print('sections: ' + ', '.join(f'{name} ({state})' for name, state in result['sections'].items()))

    # Every unit's state per case and block, then the page's headings and table rows, for the job's log.
    for case in result['cases'].values():
        for block, read in (('first block', case['first']), ('second block', case['second'])):
            for side in (read['sides'].values() if read is not None else ()):
                print(f"units | {case['words']} | {block} | {side.name} | {len(side.groups)} of {side.planned} configurations read")
    text = result['page'].read_text()
    print('\n'.join(line for line in text.splitlines() if line.startswith(('| ', '#', '- **', 'Not complete'))))
