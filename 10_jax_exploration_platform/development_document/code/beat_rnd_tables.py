"""Tables, figures and prose numbers of the section "PointMaze and AntMaze: beat CleanRL PPO-RND".

The section is development_document/beat_rnd_pointmaze_antmaze.tex. Every number in it is read from the run
records of three run folders by the analysis modules of scripts/maze_campaign/analysis (the same reading,
choosing and resampling as the generated standing page and batch summaries, so a number here equals the number
there when both read the same comparison):

- the auto research folder (AUTO): validation blocks 1 and 2 of every arm, and the discovery checks;
- the PointMaze RND baseline folder (BASELINE): RND's confirmation runs on the seeds from 30000;
- the AntMaze baseline folder (ANT): the environment-count check, the longer budget runs, the arm screens.

Only the blocks between `% >>> AUTO-GENERATED TABLE START: <name>` and `% <<< AUTO-GENERATED TABLE END: <name>`
are rewritten; captions, labels and prose stay as written. A unit that is not complete stops the generator with
an error naming it, so no table is ever read from part of a block.

numpy runs only on a compute node, so this is run through the speed study's test helper (from the platform folder):

  bash "benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/infrastructure/submit_tests.sh" \
      -s development_document/code/beat_rnd_tables_job.py
"""

import fnmatch
import json
import re
from pathlib import Path
import sys

PLATFORM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLATFORM / 'scripts'))
from maze_campaign.analysis import (case_sweeps, compare, figures, measures, records, report,  # noqa: E402
                                    same_configuration, validation_seeds)

RUNS = PLATFORM / 'runs'
AUTO = RUNS / ('2026-11-12-00-22_pointmaze-medium-large_antmaze-umaze-medium-large_beat-cleanrl-ppo-rnd-auto-research'
               '_reward-arrival-plus-1-and-step-cost_bonus-5-days_other-parts-5-days')
BASELINE = RUNS / ('2026-11-12-02-24_cleanrl-ppo-rnd-baseline_pointmaze-medium-large_antmaze-umaze-medium-large'
                   '_goal-termination_reward-arrival-plus-1-and-step-cost_first-pass-60-configurations_seeds-0-15'
                   '_confirmation-seeds-from-30000')
ANT = RUNS / ('2026-11-12-15-56_cleanrl-ppo-rnd-baseline_antmaze-umaze-medium-large_mujoco-warp-physics'
              '_ceiling-on-wall-tops_goal-termination_reward-arrival-plus-1-and-step-cost'
              '_environments-per-policy-check-4-16-128_seeds-0-15_confirmation-seeds-from-30000')
TEX = PLATFORM / 'development_document' / 'beat_rnd_pointmaze_antmaze.tex'
PLOTS = PLATFORM / 'development_document' / 'code' / 'beat_rnd_plots'

CONTROL = 'control-cleanrl-ppo-rnd'
RND_NAME = 'CleanRL PPO-RND'
# Each arm as (its method on the first validation block, its method on the second): unit names have a length limit,
# so some arms are spelled differently on the two blocks.
FIRST_ARRIVAL_HOLD = ('divisor-held-from-first-arrival', 'held-from-first-arrival')
FIXED_HOLD = ('divisor-held-after-one-sixteenth', 'divisor-held-one-sixteenth')
FAST_STEP_003 = ('fast-step-0-point-03-with-hold', 'fast-step-0-point-03-with-hold')
FAST_DISCOVERY = ('fast-last-layer-step-0-point-1', 'fast-layer-step-0-point-1')
FAST_STEP_001 = ('fast-step-0-point-01-with-hold', 'fast-step-0-point-01-with-hold')
# The method (the user decided on 2026-11-13 that the divisor held from each policy's first arrival counts as part
# of it): that divisor, plus the fast last layer at the step with the highest whole-curve mean on the first block
# among 0, 0.01, 0.03 and 0.1 (the session's rule, fixed before the second blocks were read): 0 on PointMaze Medium
# with reward 1 on arrival, 0.01 on Medium with the step cost, 0.03 on the two Large cases.
CASES = (
    {'environment': 'pointmaze-medium', 'reward_setting': 'arrival_plus_one', 'method': FIRST_ARRIVAL_HOLD,
     'step': '0', 'hold_alone': None},
    {'environment': 'pointmaze-medium', 'reward_setting': 'step_cost', 'method': FAST_STEP_001,
     'step': '0.01', 'hold_alone': FIRST_ARRIVAL_HOLD},
    {'environment': 'pointmaze-large', 'reward_setting': 'arrival_plus_one', 'method': FAST_STEP_003,
     'step': '0.03', 'hold_alone': FIRST_ARRIVAL_HOLD},
    {'environment': 'pointmaze-large', 'reward_setting': 'step_cost', 'method': FAST_STEP_003,
     'step': '0.03', 'hold_alone': FIRST_ARRIVAL_HOLD},
)
# Arms of the first validation block that did not become the method, for the table of what did not work.
OTHER_ARMS = (
    ('error length as the bonus, divisor held after one sixteenth of the run', 'error-length-with-held-divisor'),
    ('slower predictor (own optimizer at 0.25 times $10^{-4}$), divisor held after one sixteenth',
     'slower-predictor-with-held-divisor'),
    ('predictor at half speed (own optimizer at 0.5 times $10^{-4}$), divisor held after one sixteenth',
     'predictor-half-speed-held-divisor'),
    ('all three parts: fast last layer (step 0.1), slower predictor, divisor held after one sixteenth',
     'three-parts-held-one-sixteenth'),
    ('fast last layer (step 0.1) until the first arrival, then the divisor of the fast scores held',
     'fast-until-arrival-then-hold'),
    ('fast last layer (step 0.1) until the first arrival, then the divisor of the fast scores left running',
     'fast-until-arrival-no-hold'),
    ('fast last layer (step 0.1) until the first arrival, then the predictor\'s own divisor, held',
     'fast-then-predictor-divisor-held'),
    ('fast last layer at step 0.1 throughout, divisor held from the first arrival', 'fast-layer-first-arrival-hold'),
)
NAMES = ('int_coef', 'learning_rate', 'ent_coef')
SHORT_WORDS = {'int_coef': 'intrinsic weight', 'learning_rate': 'learning rate', 'ent_coef': 'entropy'}
ANT_CASE = ('antmaze-umaze', 'arrival_plus_one')
ANT_COUNTS = (4, 16, 128)


# ---- splicing ---------------------------------------------------------------------------------------------------

def inject_table(tex_path, table_name, block):
    """Replace the text between the named markers of `tex_path` with `block`; refuse a missing or repeated marker."""
    begin = f'% >>> AUTO-GENERATED TABLE START: {table_name}'
    end = f'% <<< AUTO-GENERATED TABLE END: {table_name}'
    text = tex_path.read_text()
    if text.count(begin) != 1 or text.count(end) != 1:
        raise SystemExit(f'Marker mismatch for {table_name!r} in {tex_path}: START {text.count(begin)} times, '
                         f'END {text.count(end)} times; exactly one of each is needed.')
    before, _, rest = text.partition(begin)
    _, _, after = rest.partition(end)
    tex_path.write_text(f'{before}{begin}\n{block}\n{end}{after}')


# ---- LaTeX text of numbers --------------------------------------------------------------------------------------

def styled(text, mark):
    """A cell's text bold ('best'), underlined ('second') or as it is (None)."""
    if mark == 'best':
        return r'{\bfseries\boldmath ' + text + '}'
    return r'\underline{' + text + '}' if mark == 'second' else text


def math(text):
    """A number written by Python as LaTeX math: thousands commas kept as commas, minus as a minus sign."""
    # before: '-1,784' and '[-0.0067, +0.0059]' ; after: '$-1{,}784$' and '$[-0.0067, +0.0059]$' (a thousands
    # comma is braced so that math mode puts no space after it; a comma between two numbers keeps its space)
    return '$' + re.sub(r'(?<=\d),(?=\d{3})', '{,}', text) + '$'


def mean_cell(value, error, digits, mark=None):
    """A mean over policies with its standard error under it, in a paragraph column."""
    return styled(math(f'{value:.{digits}f}'), mark) + r' \newline ' + math(f'\\pm {error:.{digits}f}')


def difference_cell(found, digits, mark=None):
    """A paired difference with its 95% interval under it."""
    low, high = found['interval']
    return styled(math(f"{found['mean']:+.{digits}f}"), mark) + r' \newline ' + math(f'[{low:+.{digits}f}, {high:+.{digits}f}]')


def steps_cell(value, error, mark=None):
    """A mean capped first-goal step in thousands of steps, with its standard error under it."""
    return styled(math(report.thousands(value)), mark) + r' \newline ' + math(r'\pm ' + report.thousands(error))


def steps_difference_cell(found, mark=None):
    """A paired difference of first-goal steps in thousands of steps, with its 95% interval under it."""
    low, high = found['interval']
    return (styled(math(report.thousands(found['mean'], True)), mark) + r' \newline '
            + math(f'[{report.thousands(low, True)}, {report.thousands(high, True)}]'))


def round_steps(steps):
    """A step count rounded for the prose and the tables: millions with three figures."""
    # before: 8388608 ; after: '8.39M'
    return f'{steps / 1e6:.3g}M'


def configuration_text(group):
    """A configuration by its intrinsic weight, learning rate and entropy coefficient."""
    # before: settings (('ent_coef', 0.01), ('int_coef', 4.0), ('learning_rate', 0.006), ...)
    # after:  'intrinsic weight 4, learning rate 0.006, entropy 0.01'
    return ', '.join(f'{SHORT_WORDS[name]} {group.setting(name):g}' for name in NAMES)


def marks_of(values, digits, higher_better):
    """Best and second marks of one column, ranked on the printed text so that printed ties tie."""
    return report.column_marks(values, [None if value is None else f'{value:.{digits}f}' for value in values],
                               higher_better)


def banner(words, columns):
    """A row that names the case of the rows below it."""
    return r'\midrule' + '\n' + r'\multicolumn{' + str(columns) + r'}{@{}l}{\textbf{' + words + r'}} \\' + '\n'


# ---- reading ------------------------------------------------------------------------------------------------------

def case_key(case):
    """(environment, reward setting) of a case."""
    return case['environment'], case['reward_setting']


def arms_of(case, with_fast=False):
    """The arms of one case for validation_seeds: the control first, then the method and the references."""
    arms = [{'name': RND_NAME, 'first_block': CONTROL, 'second_block': CONTROL, 'control': True},
            {'name': 'the method', 'first_block': case['method'][0], 'second_block': case['method'][1]},
            {'name': 'divisor held after one sixteenth of the run', 'first_block': FIXED_HOLD[0],
             'second_block': FIXED_HOLD[1]}]
    if case['hold_alone'] is not None:
        arms.append({'name': 'divisor held from the first arrival alone', 'first_block': case['hold_alone'][0],
                     'second_block': case['hold_alone'][1]})
    if with_fast:
        arms.append({'name': 'fast last layer at step 0.1', 'first_block': FAST_DISCOVERY[0],
                     'second_block': FAST_DISCOVERY[1]})
    return arms


def read_block(case, first, second, arms, specifications):
    """One case on both blocks through validation_seeds, refused while a unit of an arm is not complete."""
    block = {'first_block': first, 'second_block': second, 'arms': arms, 'cases': [dict(zip(
        ('environment', 'reward_setting'), case_key(case)))]}
    read = validation_seeds.read_case(AUTO, block, *case_key(case), specifications)
    # Every unit of every arm must be complete on both blocks: a best configuration read from part of a side can change.
    waiting = (validation_seeds.unfinished(read['first']['sides'], *(arm['first_block'] for arm in arms if 'first_block' in arm))
               + validation_seeds.unfinished(read['second']['sides'], *(arm['second_block'] for arm in arms if 'second_block' in arm)))
    if waiting:
        raise ValueError(f"{read['words']}: units not complete: {'; '.join(waiting)}.")
    return read


def read_cases():
    """Every PointMaze case read three ways: for W (the phases of the standing page), for M with RND's discovery
    configurations among the control's (the phases of both checks), and the discovery check (batch 9's phases)."""
    specifications = case_sweeps.read_specifications(AUTO)
    found = {}
    for case in CASES:
        found[case_key(case)] = {
            'case': case,
            'whole': read_block(case, 'validation-block', 'validation-block-two', arms_of(case), specifications),
            'discovery': read_block(case, ['validation-block', 'discovery-validation'],
                                    ['validation-block-two', 'discovery-block-two'], arms_of(case), specifications),
            'check': read_block(case, 'discovery-validation', 'discovery-block-two',
                                [{'name': RND_NAME, 'first_block': CONTROL, 'second_block': CONTROL, 'control': True},
                                 {'name': 'fast last layer at step 0.1', 'first_block': FAST_DISCOVERY[0],
                                  'second_block': FAST_DISCOVERY[1]}], specifications),
            'others': read_block(case, 'validation-block', 'validation-block-two',
                                 arms_of(case) + [{'name': 'fast last layer at step 0.01, divisor held from the first arrival',
                                                   'first_block': FAST_STEP_001[0], 'second_block': FAST_STEP_001[1]}]
                                 + [{'name': name, 'first_block': slug} for name, slug in OTHER_ARMS
                                                  if any(case_sweeps.select(specifications, ['validation-block'], slug,
                                                                            *case_key(case)))],
                                 specifications)}
        # The discovery check's control on the second block must be the RND row of the M table.
        mine = found[case_key(case)]['discovery']['second']['metrics']['discovery']['fixed']
        theirs = found[case_key(case)]['check']['second']['metrics']['discovery']['fixed']
        if mine is None or theirs is None or mine.settings != theirs.settings:
            raise ValueError(f"{case_sweeps.case_words(*case_key(case))}: RND's configuration for M on the second block "
                             'differs between the two readings.')
    return found


def second_row(read, metric, slug):
    """An arm's row on the second block for one metric, with its comparison under equal rules."""
    row = read['second']['metrics'][metric]['rows'].get(slug)
    if row is None or row['equal'] is None:
        raise ValueError(f"{read['words']}: no second-block comparison of {slug} for {metric}.")
    return row


def first_entry(read, metric, slug):
    """An arm at its best on the first block against the control at its own, for one metric."""
    entry = read['first']['comparisons'].get(slug)
    if entry is None or entry[metric]['result'] is None:
        raise ValueError(f"{read['words']}: no first-block comparison of {slug} for {metric}.")
    return entry[metric]


# ---- the protocol's two primary tables ---------------------------------------------------------------------------

def arm_words(case, which):
    """The words of an arm's row in the result tables."""
    if which == 'method':
        return (r'\textbf{the method}: divisor held from the first arrival, fast last layer at step '
                + case['step'] + (' (off)' if case['step'] == '0' else ''))
    return {'fixed': 'divisor held after one sixteenth of the run (reference)',
            'alone': 'divisor held from the first arrival alone (reference)',
            'fast': 'fast last layer at step 0.1, RND\'s running divisor (discovery arm)'}[which]


def whole_rows(found):
    """The W table's rows of one case: (row words, configuration, second-block measures, first-block difference,
    second-block difference) for the method and its references, then RND."""
    case, read = found['case'], found['whole']
    arms = [('method', case['method'])] + ([('alone', case['hold_alone'])] if case['hold_alone'] else []) + [('fixed', FIXED_HOLD)]
    rows = []
    for which, (first, second) in arms:
        row = second_row(read, 'whole', second)
        if row['how'] != 'first':
            raise ValueError(f"{read['words']}: {second} does not stand at its first-block best on the second block.")
        rows.append({'words': arm_words(case, which), 'group': row['group'], 'measures': row['measures'],
                     'first': first_entry(read, 'whole', first)['result'], 'second': row['equal'],
                     'favouring': row['favouring']})
    # The arms from the highest W on; RND, the baseline, in its own block below them.
    rows.sort(key=lambda row: -row['measures']['whole_curve_mean'])
    control = read['second']['metrics']['whole']['fixed']
    rows.append({'words': RND_NAME + ' (baseline)', 'group': control,
                 'measures': measures.configuration_measures(control.policies, read['second']['budget']),
                 'first': None, 'second': None, 'favouring': None})
    return rows


def whole_table(cases, setting):
    """The W and L table of one reward setting on the second block, with the first block's best-against-best
    difference beside it."""
    lines = [r'\begin{tabular}{@{}>{\raggedright\arraybackslash}p{3.0cm}'
             r'>{\raggedright\arraybackslash}p{1.25cm}>{\raggedright\arraybackslash}p{2.45cm}'
             r'>{\raggedright\arraybackslash}p{2.45cm}>{\raggedright\arraybackslash}p{1.25cm}'
             r'>{\raggedright\arraybackslash}p{2.45cm}@{}}',
             r'\toprule',
             r'arm \newline configuration fixed on the first block & $W$ $\uparrow$, second block & '
             r'$W$ minus RND, first block, each at its best & $W$ minus RND, second block & '
             r'$L$, second block & $L$ minus RND, second block \\']
    for key, found in ((key, found) for key, found in cases.items() if key[1] == setting):
        rows = whole_rows(found)
        lines.append(banner(case_sweeps.case_words(*key), 6).rstrip('\n'))
        digits = 4
        # Marks per column within the case: the arms' differences, and every row's W and L.
        whole = marks_of([row['measures']['whole_curve_mean'] for row in rows], digits, True)
        final = marks_of([row['measures']['final_level'] for row in rows], digits, True)
        gaps = {(key_, quantity): marks_of([row[key_][quantity]['mean'] if row[key_] else None for row in rows], digits, True)
                for key_ in ('first', 'second') for quantity in ('whole_curve_mean', 'final_level')}
        for index, row in enumerate(rows):
            m = row['measures']
            cells = [row['words'] + r' \newline \emph{' + configuration_text(row['group']) + '}',
                     mean_cell(m['whole_curve_mean'], m['whole_curve_mean_error'], digits, whole[index]),
                     difference_cell(row['first']['whole_curve_mean'], digits, gaps[('first', 'whole_curve_mean')][index])
                     if row['first'] else 'baseline',
                     difference_cell(row['second']['whole_curve_mean'], digits, gaps[('second', 'whole_curve_mean')][index])
                     if row['second'] else 'baseline',
                     mean_cell(m['final_level'], m['final_level_error'], digits, final[index]),
                     difference_cell(row['second']['final_level'], digits, gaps[('second', 'final_level')][index])
                     if row['second'] else 'baseline']
            lines.append((r'\cmidrule(l{0pt}r{0pt}){1-6}' + '\n' if index == len(rows) - 1 else '') + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)


def discovery_rows(found):
    """The M table's rows of one case: the method and its references, the discovery arm, then RND at its M configuration."""
    case, read, check = found['case'], found['discovery'], found['check']
    rows = []
    arms = [('method', case['method'])] + ([('alone', case['hold_alone'])] if case['hold_alone'] else []) + [('fixed', FIXED_HOLD)]
    for which, (first, second) in arms:
        row = second_row(read, 'discovery', second)
        rows.append({'words': arm_words(case, which), 'group': row['group'], 'how': row['how'], 'measures': row['measures'],
                     'first': first_entry(read, 'discovery', first)['result'], 'second': row['equal']})
    row = second_row(check, 'discovery', FAST_DISCOVERY[1])
    rows.append({'words': arm_words(case, 'fast'), 'group': row['group'], 'how': row['how'], 'measures': row['measures'],
                 'first': first_entry(check, 'discovery', FAST_DISCOVERY[0])['result'], 'second': row['equal']})
    # The arms from the smallest M on; RND, the baseline, in its own block below them.
    rows.sort(key=lambda row: row['measures']['capped_first_goal_step'])
    control = read['second']['metrics']['discovery']['fixed']
    rows.append({'words': RND_NAME + ' (baseline)', 'group': control, 'how': 'first',
                 'measures': measures.configuration_measures(control.policies, read['second']['budget']),
                 'first': None, 'second': None})
    return rows


def discovery_table(cases, setting):
    """The M table of one reward setting on the second block, each arm at its first-block best for M, with W at
    that configuration."""
    lines = [r'\begin{tabular}{@{}>{\raggedright\arraybackslash}p{3.6cm}'
             r'>{\raggedright\arraybackslash}p{1.45cm}>{\raggedright\arraybackslash}p{2.2cm}'
             r'>{\raggedright\arraybackslash}p{2.2cm}>{\raggedright\arraybackslash}p{1.45cm}'
             r'>{\raggedright\arraybackslash}p{1.3cm}@{}}',
             r'\toprule',
             r'arm \newline configuration with its smallest $M$ on the first block & $M$ $\downarrow$, second block & '
             r'$M$ minus RND, first block, each at its best & $M$ minus RND, second block & '
             r'$W$ at this configuration, second block & policies that reach the goal \\']
    for key, found in ((key, found) for key, found in cases.items() if key[1] == setting):
        rows = discovery_rows(found)
        lines.append(banner(case_sweeps.case_words(*key), 6).rstrip('\n'))
        # Marks per column within the case; M, its differences and the arrivals rank the arms, W is shown only.
        capped = report.column_marks([row['measures']['capped_first_goal_step'] for row in rows],
                                     [report.thousands(row['measures']['capped_first_goal_step']) for row in rows], False)
        gaps = {key_: report.column_marks([row[key_]['capped_first_goal_step']['mean'] if row[key_] else None for row in rows],
                                          [report.thousands(row[key_]['capped_first_goal_step']['mean'], True) if row[key_] else None
                                           for row in rows], False) for key_ in ('first', 'second')}
        for index, row in enumerate(rows):
            m = row['measures']
            reached = round(m['reached_goal_share'] * m['seeds'])
            dagger = '' if row['how'] == 'first' else r'$^{\dagger}$'
            cells = [row['words'] + r' \newline \emph{' + configuration_text(row['group']) + '}' + dagger,
                     steps_cell(m['capped_first_goal_step'], m['capped_first_goal_step_error'], capped[index]),
                     steps_difference_cell(row['first']['capped_first_goal_step'], gaps['first'][index]) if row['first'] else 'baseline',
                     steps_difference_cell(row['second']['capped_first_goal_step'], gaps['second'][index]) if row['second'] else 'baseline',
                     math(f"{m['whole_curve_mean']:.4f}"), f"{reached} of {m['seeds']}"]
            lines.append((r'\cmidrule(l{0pt}r{0pt}){1-6}' + '\n' if index == len(rows) - 1 else '') + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)


def others_table(cases):
    """Every other arm on the first block at its best against RND at its best: W difference per case."""
    keys = list(cases)
    lines = [r'\begin{tabular}{@{}>{\raggedright\arraybackslash}p{3.4cm}'
             + r'>{\raggedright\arraybackslash}p{2.45cm}' * len(keys) + '@{}}',
             r'\toprule',
             'arm & ' + ' & '.join(r'\shortstack[l]{' + case_sweeps.case_words(*key).replace(', ', r',\\') + '}'
                                   for key in keys) + r' \\',
             r'\midrule']
    for name, slug in OTHER_ARMS:
        cells = []
        for key in keys:
            entry = cases[key]['others']['first']['comparisons'].get(slug)
            cells.append(difference_cell(entry['whole']['result']['whole_curve_mean'], 4)
                         if entry is not None and entry['whole']['result'] is not None else 'not run')
        lines.append(name + ' & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)


# ---- RND on the confirmation seeds --------------------------------------------------------------------------------

def baseline_sides(environment, reward_setting, specifications, excluded):
    """RND's confirmation rows in one case: the configuration chosen for W, the one for M, CleanRL's defaults, no bonus."""
    keep = lambda found: [spec for spec in found if not any(fnmatch.fnmatchcase(spec['unit_id'], pattern) for pattern in excluded)]
    side = lambda phase, slug, name: case_sweeps.read_side(
        BASELINE, keep(case_sweeps.select(specifications, [phase], slug, environment, reward_setting)), slug, name)
    rows = {'whole': side('confirmation', 'cleanrl-ppo-rnd', RND_NAME),
            'defaults': side('confirmation', 'cleanrl-ppo-rnd-at-cleanrl-defaults', "RND at CleanRL's defaults"),
            'none': side('confirmation', 'ppo-without-a-bonus', 'PPO without a bonus'),
            'discovery': side('confirmation-for-discovery', 'cleanrl-ppo-rnd', RND_NAME)}
    for key, found in rows.items():
        if len(found.groups) < found.planned or not found.groups:
            raise ValueError(f'{environment}, {reward_setting}: confirmation units of {key} not complete.')
    for key in ('whole', 'defaults', 'none'):
        if len(rows[key].groups) != 1:
            raise ValueError(f'{environment}, {reward_setting}: {len(rows[key].groups)} confirmation configurations of {key}.')
    budget = rows['whole'].budget
    # RND's configuration for M: the smallest M of its development sweeps at the budget, found among the confirmations.
    development = case_sweeps.read_side(BASELINE, case_sweeps.select(
        specifications, ['development-at-the-budget*', 'development-for-discovery*'], 'cleanrl-ppo-rnd', environment,
        reward_setting, budget), 'cleanrl-ppo-rnd', RND_NAME)
    chosen = case_sweeps.best_configuration(development.groups, budget, 'discovery')
    matched = [group for group in rows['discovery'].groups
               if same_configuration.cell_key(group) == same_configuration.cell_key(chosen)]
    if len(matched) != 1:
        raise ValueError(f'{environment}, {reward_setting}: the development choice for M ({configuration_text(chosen)}) '
                         f'is held by {len(matched)} confirmation configurations.')
    return {'budget': budget, 'whole': rows['whole'].groups[0], 'discovery': matched[0],
            'defaults': rows['defaults'].groups[0], 'none': rows['none'].groups[0]}


def baseline_table(setting):
    """RND on the confirmation seeds in the two cases of one reward setting: its configuration for W, its
    configuration for M, and the two reference rows."""
    specifications = case_sweeps.read_specifications(BASELINE)
    excluded = json.loads((BASELINE / 'report_settings.json').read_text()).get('exclude_units', [])
    lines = [r'\begin{tabular}{@{}>{\raggedright\arraybackslash}p{5.4cm}'
             r'>{\raggedright\arraybackslash}p{1.5cm}>{\raggedright\arraybackslash}p{1.5cm}'
             r'>{\raggedright\arraybackslash}p{1.6cm}>{\raggedright\arraybackslash}p{1.3cm}@{}}',
             r'\toprule',
             r'row \newline configuration & $W$ $\uparrow$ & $L$ & $M$, thousand steps & policies that reach the goal \\']
    found = {}
    for case in (case for case in CASES if case['reward_setting'] == setting):
        rows = baseline_sides(*case_key(case), specifications, excluded)
        found[case_key(case)] = rows
        budget = rows['budget']
        lines.append(banner(f"{case_sweeps.case_words(*case_key(case))}, budget {round_steps(budget)} steps", 5).rstrip('\n'))
        order = [('whole', 'RND, configuration for $W$'), ('discovery', 'RND, configuration for $M$'),
                 ('defaults', "RND at CleanRL's defaults (reference)"), ('none', 'PPO without a bonus (reference)')]
        # Every row is a baseline or a reference, so all four are one block, from the highest W on.
        values = [measures.configuration_measures(rows[key].policies, budget) for key, _ in order]
        order, values = zip(*sorted(zip(order, values), key=lambda pair: -pair[1]['whole_curve_mean']))
        whole = marks_of([value['whole_curve_mean'] for value in values], 3, True)
        final = marks_of([value['final_level'] for value in values], 3, True)
        capped = report.column_marks([value['capped_first_goal_step'] for value in values],
                                     [report.thousands(value['capped_first_goal_step']) for value in values], False)
        for index, ((key, words), value) in enumerate(zip(order, values)):
            reached = round(value['reached_goal_share'] * value['seeds'])
            cells = [words + r' \newline \emph{' + configuration_text(rows[key]) + '}',
                     mean_cell(value['whole_curve_mean'], value['whole_curve_mean_error'], 3, whole[index]),
                     mean_cell(value['final_level'], value['final_level_error'], 3, final[index]),
                     steps_cell(value['capped_first_goal_step'], value['capped_first_goal_step_error'], capped[index]),
                     f"{reached:,} of {value['seeds']:,}"]
            lines.append(' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines), found


# ---- AntMaze ----------------------------------------------------------------------------------------------------------

def ant_side(phase, count, specifications):
    """RND with one count of environments per policy on the U-maze in one phase of the AntMaze folder."""
    slug = f'cleanrl-ppo-rnd-with-{count}-environments-per-policy'
    return case_sweeps.read_side(ANT, case_sweeps.select(specifications, [phase], slug, *ANT_CASE), slug,
                                 f'{count} environments per policy')


def ant_count_table():
    """The environment-count check on the U-maze: each count's best W and its best M, 48 configurations each."""
    specifications = case_sweeps.read_specifications(ANT)
    lines = [r'\begin{tabular}{@{}>{\raggedright\arraybackslash}p{1.6cm}>{\raggedright\arraybackslash}p{3.55cm}'
             r'>{\raggedright\arraybackslash}p{1.35cm}>{\raggedright\arraybackslash}p{1.35cm}'
             r'>{\raggedright\arraybackslash}p{3.55cm}>{\raggedright\arraybackslash}p{1.3cm}@{}}',
             r'\toprule',
             r'environments per policy & configuration with the highest $W$ & $W$ & $L$ & '
             r'configuration with the smallest $M$ & $M$, thousand steps \\',
             r'\midrule']
    found = {}
    rows = []
    for count in ANT_COUNTS:
        side = ant_side('rollout-shape-check-stable-physics', count, specifications)
        if len(side.groups) < side.planned:
            raise ValueError(f'AntMaze check, {count} environments per policy: {case_sweeps.read_words(side)}.')
        whole = case_sweeps.best_configuration(side.groups, side.budget, 'whole')
        discovery = case_sweeps.best_configuration(side.groups, side.budget, 'discovery')
        rows.append((count, side, whole, measures.configuration_measures(whole.policies, side.budget), discovery,
                     measures.configuration_measures(discovery.policies, side.budget)))
        found[count] = {'budget': side.budget, 'configurations': len(side.groups), 'seeds': len(whole.policies)}
    whole_marks = marks_of([row[3]['whole_curve_mean'] for row in rows], 4, True)
    final_marks = marks_of([row[3]['final_level'] for row in rows], 4, True)
    capped_marks = report.column_marks([row[5]['capped_first_goal_step'] for row in rows],
                                       [report.thousands(row[5]['capped_first_goal_step']) for row in rows], False)
    for index, (count, side, whole, w, discovery, d) in enumerate(rows):
        lines.append(' & '.join([str(count), configuration_text(whole),
                                 mean_cell(w['whole_curve_mean'], w['whole_curve_mean_error'], 4, whole_marks[index]),
                                 mean_cell(w['final_level'], w['final_level_error'], 4, final_marks[index]),
                                 configuration_text(discovery),
                                 steps_cell(d['capped_first_goal_step'], d['capped_first_goal_step_error'], capped_marks[index])])
                     + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines), found


def ant_budget_rule():
    """The plan's budget rule on the longer U-maze runs, per count of environments per policy; a run whose unit is
    not complete on UVA (its records are copied back only when it ends) is marked so."""
    specifications = case_sweeps.read_specifications(ANT)
    found = {}
    for count in (16, 128):
        side = ant_side('budget-probe', count, specifications)
        if len(side.groups) < side.planned:
            found[count] = {'complete': False, 'run_steps': side.budget}
            continue
        proposal = compare.propose_budget(side.groups, side.budget)
        best = case_sweeps.best_configuration(side.groups, side.budget, 'whole')
        found[count] = {'complete': True, 'run_steps': side.budget, 'proposal': proposal,
                        'best': best, 'best_measures': measures.configuration_measures(best.policies, side.budget)}
    return found


def ant_screen_groups():
    """The AntMaze groups of batches 10 and 14's same-configuration settings, in their order."""
    groups = []
    for batch in ('10_*', '14_*'):
        folder = next((AUTO / 'batches').glob(batch))
        settings = json.loads((folder / 'summary_settings.json').read_text())
        groups += [group for group in settings['same_configuration'] if group['title'].startswith('AntMaze')]
    return groups


def ant_screen_table():
    """Each bonus arm on the U-maze against RND at the same configurations and development seeds."""
    groups = same_configuration.read(AUTO, ant_screen_groups())
    lines = [r'\begin{tabular}{@{}>{\raggedright\arraybackslash}p{3.8cm}>{\raggedright\arraybackslash}p{1.5cm}'
             r'>{\raggedright\arraybackslash}p{2.2cm}>{\raggedright\arraybackslash}p{2.0cm}'
             r'>{\raggedright\arraybackslash}p{1.2cm}>{\raggedright\arraybackslash}p{1.9cm}@{}}',
             r'\toprule',
             r'arm & environments per policy & $W$: ahead, level, behind RND (configurations) & '
             r'$W$ minus RND, smallest to largest & best $W$ of the arm & $M$: sooner, level, later \\',
             r'\midrule']
    found = []
    for group in groups:
        read = group['cases'][ANT_CASE]
        sides = [read['arm']] + read['others']
        waiting = [f'{side.name}: {case_sweeps.read_words(side)}' for side in sides if len(side.groups) < side.planned]
        if waiting:
            raise ValueError(f"{group['title']}: not complete: {'; '.join(waiting)}.")
        # Only the comparison with RND (the environment-count check's units) is tabled; batch 14 also compares
        # each step with batch 10's arm.
        rnd = [slug for slug in read['pairs'] if slug.startswith('cleanrl-ppo-rnd-with-')]
        if len(rnd) != 1:
            raise ValueError(f"{group['title']}: {len(rnd)} RND sides compared, one expected.")
        pairs, missing = read['pairs'][rnd[0]]
        # Counts of pairs per side of zero, per metric.
        # before: 8 pairs whose W intervals lie [above, within x 6, below]; after: {'ahead': 1, 'level': 6, 'behind': 1}
        count = lambda quantity: {side: sum(pair['result'][quantity]['side'] == side for pair in pairs)
                                  for side in ('ahead', 'level', 'behind')}
        whole, discovery = count('whole_curve_mean'), count('capped_first_goal_step')
        differences = sorted(pair['result']['whole_curve_mean']['mean'] for pair in pairs)
        best = case_sweeps.best_configuration(read['arm'].groups, read['budget'], 'whole')
        best_w = measures.configuration_measures(best.policies, read['budget'])['whole_curve_mean']
        environments = read['arm'].groups[0].environments_per_policy
        arm = read['arm'].name
        found.append({'arm': arm, 'environments': environments, 'whole': whole, 'discovery': discovery,
                      'pairs': len(pairs), 'missing': len(missing), 'range': (differences[0], differences[-1]), 'best': best_w})
        lines.append(' & '.join([arm, str(environments), f"{whole['ahead']}, {whole['level']}, {whole['behind']} of {len(pairs)}",
                                 math(f'{differences[0]:+.4f}') + ' to ' + math(f'{differences[-1]:+.4f}'),
                                 math(f'{best_w:.4f}'),
                                 f"{discovery['ahead']}, {discovery['level']}, {discovery['behind']} of {len(pairs)}"]) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines), found


# ---- figures ----------------------------------------------------------------------------------------------------------

def curve_figures(cases):
    """Second-block training curves of RND and the method, one figure per reward setting, one panel per maze."""
    PLOTS.mkdir(parents=True, exist_ok=True)
    names = {}
    for setting in ('arrival_plus_one', 'step_cost'):
        panels = []
        for (environment, reward_setting), found in cases.items():
            if reward_setting != setting:
                continue
            read, case = found['whole'], found['case']
            budget = read['second']['budget']
            method = second_row(read, 'whole', case['method'][1])['group']
            control = read['second']['metrics']['whole']['fixed']
            series = []
            # RND in solid black (the reference), the method in the first series hue.
            for group, color, order in ((control, '#000000', 3), (method, figures.series_color(0), 4)):
                steps, mean, error = measures.mean_curve(group.policies, budget)
                series.append({'steps': steps, 'mean': mean, 'error': error, 'color': color, 'width': 1.2,
                               'band': True, 'order': order})
            panels.append({'title': records.ENVIRONMENT_NAMES[tuple(environment.split('-', 1))], 'series': series})
        legend = [{'label': f'{RND_NAME}, the baseline', 'color': '#000000', 'width': 1.2, 'band': True},
                  {'label': 'the method (divisor held from the first arrival, fast last layer at the step of each case)',
                   'color': figures.series_color(0), 'width': 1.2, 'band': True}]
        name = f'second-validation-block_training-return_{setting.replace("_", "-")}'
        names[setting] = figures.curve_figure(panels, legend, PLOTS, name, 'training return')
    return names


# ---- numbers in the prose -----------------------------------------------------------------------------------------

def macro(name, value):
    """One prose number as a LaTeX command."""
    return r'\newcommand{\BeatRnd' + name + '}{' + value + '}'


def prose_numbers(cases, baseline, counts, budgets):
    """The numbers the prose quotes that no table shows."""
    lines = []
    # The method against RND at RND's best on the second block (favouring RND), per case.
    for key, found in cases.items():
        row = second_row(found['whole'], 'whole', found['case']['method'][1])
        label = ''.join(part.capitalize() for part in (key[0].split('-')[1], *key[1].split('_')))
        lines.append(macro('Favouring' + label, difference_text(row['favouring']['whole_curve_mean'])))
    # PPO without a bonus against RND on PointMaze Medium with the step cost, confirmation seeds.
    medium = baseline[('pointmaze-medium', 'step_cost')]
    result = compare.compare_with_baseline(medium['none'].policies, medium['whole'].policies, medium['budget'],
                                           'beat-rnd section', 'no bonus against RND', 'pointmaze-medium', 'step_cost')
    lines.append(macro('NoBonusMediumStepCost', difference_text(result['whole_curve_mean'], 3)))
    lines.append(macro('ConfirmationSeeds', f"{len(medium['whole'].policies):,}".replace(',', '{,}')))
    # The AntMaze check's size and the budget rule.
    lines.append(macro('AntCheckSteps', round_steps(counts[16]['budget'])))
    lines.append(macro('AntCheckConfigurations', str(counts[16]['configurations'])))
    for count, found in budgets.items():
        words = {16: 'Sixteen', 128: 'HundredTwentyEight'}[count]
        lines.append(macro(f'AntRunSteps{words}', round_steps(found['run_steps'])))
        if found['complete']:
            proposal = found['proposal']
            # The rule's value before it is capped at the run's own length, and the latest rise it read.
            lines.append(macro(f'AntBudgetRule{words}', round_steps(proposal['rounded'])))
            lines.append(macro(f'AntLatestRise{words}', round_steps(proposal['largest'])))
            lines.append(macro(f'AntBestW{words}', f"{found['best_measures']['whole_curve_mean']:.4f}"))
            lines.append(macro(f'AntBestL{words}', f"{found['best_measures']['final_level']:.4f}"))
            lines.append(macro(f'AntBestConfiguration{words}', configuration_text(found['best'])))
    # PointMaze Medium with the step cost: the step 0.01 against the step 0 (the hold alone), each at its best on the
    # first block, and the step 0.01 on the second block against RND under equal rules.
    medium = cases[('pointmaze-medium', 'step_cost')]['others']
    small = medium['first']['comparisons'][FAST_STEP_001[0]]['whole']
    alone = medium['first']['comparisons'][FIRST_ARRIVAL_HOLD[0]]['whole']
    lines.append(macro('MediumStepCostStepSmallFirst', math(f"{small['measures']['whole_curve_mean']:.4f}")))
    lines.append(macro('MediumStepCostStepZeroFirst', math(f"{alone['measures']['whole_curve_mean']:.4f}")))
    row = second_row(medium, 'whole', FAST_STEP_001[1])
    lines.append(macro('MediumStepCostStepSmallSecond', difference_text(row['equal']['whole_curve_mean'])))
    return '\n'.join(lines)


def difference_text(found, digits=4):
    """A paired difference with its interval on one line, for the prose."""
    low, high = found['interval']
    return math(f"{found['mean']:+.{digits}f}") + ' ' + math(f'[{low:+.{digits}f}, {high:+.{digits}f}]')


# ---- the whole run --------------------------------------------------------------------------------------------------

def main():
    """Read every record the section uses, splice the tables and numbers into the section, draw the figures."""
    cases = read_cases()
    baseline_arrival, found_arrival = baseline_table('arrival_plus_one')
    baseline_step_cost, found_step_cost = baseline_table('step_cost')
    baseline_found = {**found_arrival, **found_step_cost}
    counts_table, counts = ant_count_table()
    budgets = ant_budget_rule()
    screen_table, screen = ant_screen_table()
    blocks = {'beat-rnd-baseline-arrival': baseline_arrival, 'beat-rnd-baseline-step-cost': baseline_step_cost,
              'beat-rnd-whole-arrival': whole_table(cases, 'arrival_plus_one'),
              'beat-rnd-whole-step-cost': whole_table(cases, 'step_cost'),
              'beat-rnd-discovery-arrival': discovery_table(cases, 'arrival_plus_one'),
              'beat-rnd-discovery-step-cost': discovery_table(cases, 'step_cost'),
              'beat-rnd-others': others_table(cases),
              'beat-rnd-ant-counts': counts_table, 'beat-rnd-ant-screen': screen_table,
              'beat-rnd-numbers': prose_numbers(cases, baseline_found, counts, budgets)}
    for name, block in blocks.items():
        inject_table(TEX, name, block)
    figures_written = curve_figures(cases)
    return {'blocks': blocks, 'figures': figures_written, 'budgets': budgets, 'counts': counts, 'screen': screen}


if __name__ == '__main__':
    main()
