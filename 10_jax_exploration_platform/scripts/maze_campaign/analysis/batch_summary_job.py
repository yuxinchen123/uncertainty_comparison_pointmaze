"""Generate a batch's summary inside the test helper's Slurm job, since numpy and plotting never run on a login node.

  bash "<speed-study benchmark run>/infrastructure/submit_tests.sh" -s \
      scripts/maze_campaign/analysis/batch_summary_job.py --run-dir=runs/<run folder> --batch=<batch folder name>

The helper runs pytest in the platform folder, so the work is done by this file's one function. `-s` prints what it
found into the job's log. It writes <run folder>/batches/<batch>/summary.md, summary.html and plots/.
"""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from maze_campaign.analysis import batch_summary  # noqa: E402


def test_generate_batch_summary(request):
    """Generate the summary of the batch named on the command line and print its units and its screen rows."""
    run_dir, batch = request.config.getoption('--run-dir'), request.config.getoption('--batch')
    if not run_dir or not batch:
        pytest.skip('name a run folder and a batch with --run-dir=<run folder> --batch=<batch folder name>')
    result = batch_summary.generate(Path(run_dir).resolve(), batch)
    screen = result['screen']
    print(f"\nwrote {result['summary']}")
    print('sections: ' + ', '.join(f'{name} ({state})' for name, state in result['sections'].items()))

    # Every unit's state, then every complete method at its best weight, for the job's log.
    for unit in screen.units:
        print(f"unit | {unit['name']} | {unit['state']} | windows {unit.get('windows', 0)}")
    for slug in batch_summary.row_order(screen):
        method, difference = screen.methods[slug], screen.differences.get(slug)
        against = ('' if difference is None else
                   f" | difference in W {batch_summary.report.interval_text(difference['whole_curve_mean'])}")
        ratios = [value for value in method.grid['start_ratio_by_weight'].values() if value is not None]
        print(f"method | {method.name} | best weight {method.best_weight:g} | "
              f"W {method.measures['whole_curve_mean']:.3f} | L {method.measures['final_level']:.3f}{against} | "
              f"median start ratio per weight from {min(ratios):.2f} to {max(ratios):.2f} | slopes fitted at the best "
              f"weight for {method.grid['decay_slope_pairs']} of {method.grid['pairs']} pairs")
