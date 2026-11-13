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

    # What is read beside the screen: the table rows of the validation block at the best weights, of the
    # development sweeps and of the discovery checks, as the summary shows them.
    text = result['summary'].read_text()
    for heading in ('### Each method at its best intrinsic weight against the control at its best weight\n',
                    '## Development sweeps against the baseline, case by case\n', '## Discovery checks beyond the screen\n',
                    '## Validation seeds, case by case\n', '## Predictions\n'):
        if heading in text:
            body = text.split(heading)[1].split('\n## ')[0]
            print(f'\n{heading.strip()}')
            print('\n'.join(line for line in body.splitlines() if line.startswith(('| ', '#', '**', 'Observed', '- ')) or 'confirmation seeds' in line
                            or 'at an end of its grid' in line or 'Not read yet' in line))

    # The secondary metric: every complete method at its best weight for M, in the order of its table.
    report = batch_summary.report
    for slug in batch_summary.discovery_order(screen):
        method, difference, row = screen.methods[slug], screen.discovery_differences.get(slug), screen.methods[slug].discovery_measures
        against = ('' if difference is None or difference['ahead'] is None else
                   f" | difference in M {report.discovery_interval_text(difference['capped_first_goal_step'])}, "
                   f"{difference['verdict']}")
        end = batch_summary.end_of_grid(method.weights, method.discovery_weight)
        print(f"discovery | {method.name} | best weight for M {method.discovery_weight:g}"
              f"{f' ({end} of {len(method.weights)} weights)' if end else ''} | "
              f"M {report.thousands_with_error(row['capped_first_goal_step'], row['capped_first_goal_step_error'])} "
              f"thousand steps | policies that reach the goal {round(row['reached_goal_share'] * row['seeds'])} of "
              f"{row['seeds']} | W {row['whole_curve_mean']:.3f} | L {row['final_level']:.3f}{against}")
