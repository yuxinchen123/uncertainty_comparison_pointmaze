"""Generate a run folder's report inside the test helper's Slurm job, since numpy and plotting never run on a login node.

  bash "<speed-study benchmark run>/infrastructure/submit_tests.sh" -s \
      scripts/maze_campaign/analysis/report_job.py --run-dir=runs/<run folder> [--at-steps=N] [--latex-check-dir=<folder>]

The helper runs pytest in the platform folder, so the work is done by this file's one function. `-s` prints its
summary into the job's log. `--latex-check-dir` also compiles analysis/results.tex inside a small document of the
development document's text width, fails on any overfull box, and writes each page as a PNG for reading by eye.
"""

import os
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from maze_campaign.analysis import measures, report  # noqa: E402

WRAPPER = r"""\documentclass{article}
\usepackage[T1]{fontenc}
\usepackage[textwidth=13.97cm]{geometry}
\usepackage{booktabs,float,array,graphicx,amsmath,amssymb}
\newcommand{\MazeCampaignRunPath}{RUN}
\begin{document}
\input{RUN/analysis/results.tex}
\end{document}
"""


def compile_results(run_dir, check_dir):
    """Compile the run folder's results.tex, refuse overfull boxes, and write the pages as PNG files."""
    check_dir.mkdir(parents=True, exist_ok=True)
    (check_dir / 'check.tex').write_text(WRAPPER.replace('RUN', str(run_dir)))
    tex_bin = Path.home() / '.TinyTeX/bin/x86_64-linux'
    if not (tex_bin / 'latexmk').exists():
        raise FileNotFoundError(f'No latexmk under {tex_bin}; install TinyTeX (skill latex-build-tinytex).')
    environment = {**os.environ, 'PATH': f"{tex_bin}:{os.environ['PATH']}"}
    subprocess.run(['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error', 'check.tex'], cwd=check_dir,
                   env=environment, check=True, capture_output=True)
    overfull = [line for line in (check_dir / 'check.log').read_text(errors='replace').splitlines()
                if line.startswith('Overfull \\hbox')]
    subprocess.run(['gs', '-q', '-dSAFER', '-dBATCH', '-dNOPAUSE', '-sDEVICE=png16m', '-r110',
                    f'-sOutputFile={check_dir}/page-%02d.png', str(check_dir / 'check.pdf')], check=True)
    return overfull


def test_generate_report(request):
    """Generate the report of the run folder named on the command line and print what it found."""
    run_dir = request.config.getoption('--run-dir')
    if not run_dir:
        pytest.skip('name a run folder with --run-dir=<run folder> to generate its report')
    run_dir = Path(run_dir).resolve()
    result = report.generate(run_dir, request.config.getoption('--at-steps'))
    built = result['report']
    print(f"\nwrote {result['analysis']}")
    print('sections: ' + ', '.join(f'{name} ({state})' for name, state in result['sections'].items()))

    # Every development configuration at the step count its table used, for the job's log; while some runs of a
    # table are still training, all of its configurations again at the steps every policy has reached.
    tables = report.grouped(built.groups['development'], lambda group: (
        group.environment, group.reward_setting, group.planned_steps, group.method))
    for (environment, setting, planned, _), groups in tables.items():
        steps, ready, waiting = built.evaluation(groups)
        common = measures.common_steps([policy for group in groups for policy in group.policies])
        blocks = [(steps, ready)] + ([(common, groups)] if waiting and built.at_steps is None else [])
        for at, members in blocks:
            print(f'{environment}, {report.REWARD_SETTING_NAMES[setting]}, at {at:,} of {planned:,} planned steps:')
            for group in members:
                row = built.measured(group, at)
                print(f"  {report.row_name(group, report.MAIN_SETTINGS, True, True)} | "
                      f"{measures.common_steps(group.policies):,} steps so far | "
                      f"W {report.with_error(row['whole_curve_mean'], row['whole_curve_mean_error'])} | "
                      f"L {report.with_error(row['final_level'], row['final_level_error'])} | "
                      f"policies that reached the goal {100 * row['reached_goal_share']:.0f}% | seeds {row['seeds']}")

    # The secondary metric: each development sweep's choice by M, where it was read, and its confirmation.
    for key, group in built.discovery_choices.items():
        environment, setting, planned, method = key
        notes, row = built.discovery_notes[key], built.measured(group, built.discovery_notes[key]['steps'])
        capped = report.thousands_with_error(row['capped_first_goal_step'], row['capped_first_goal_step_error'])
        print(f"discovery choice | {environment}, {report.REWARD_SETTING_NAMES[setting]}, {method}, {planned:,} planned "
              f"steps | {notes['words']} | M {capped} thousand steps at {notes['steps']:,} steps | policies that "
              f"reached the goal {100 * row['reached_goal_share']:.0f}% | "
              f"W {report.with_error(row['whole_curve_mean'], row['whole_curve_mean_error'])} | "
              f"L {report.with_error(row['final_level'], row['final_level_error'])} | {notes['configurations']} "
              f"configurations from units {notes['units']} | confirmation: "
              f"{built.discovery_status.get(key, 'no confirmation unit of this case and training length')}")

    # What the budget rule proposes from each finished sweep of the baseline.
    for proposal in built.budgets:
        reached = '; '.join(f"{report.row_name(group, report.MAIN_SETTINGS, False, False)}: "
                            f"{'never' if step is None else format(step, ',')}"
                            for group, step in proposal['configurations'])
        print(f"budget rule, {proposal['case']}: first steps at 95% of the final level: {reached} | proposed budget "
              f"{proposal['budget']:,} of {proposal['run_steps']:,} run steps"
              + (' (capped at the run\'s length)' if proposal['capped'] else ''))

    check_dir = request.config.getoption('--latex-check-dir')
    if check_dir:
        overfull = compile_results(run_dir, Path(check_dir).resolve())
        print(f'results.tex compiled in {check_dir}; overfull boxes: {len(overfull)}')
        assert not overfull, overfull
