"""Generate the standing page of a run folder: <run folder>/analysis/standing.md and standing.html.

The page reads the validation seeds of every case named in `<run folder>/analysis/standing_settings.json` and
shows, case by case, each arm against the control: on the first block each side at its best configuration there,
and on the second block, where it was run, at the configuration fixed on the first block (validation_seeds.py).
One table at the top gives a verdict word per case and family of arms.

`standing_settings.json`:

- `title`: the page's title;
- `validation_seeds`: the two blocks, the arms and the cases (keys in validation_seeds.py); an arm's `family`
  puts it into a row of the table at the top, and an arm without one is shown in its case only;
- `families`: the families of the table at the top, in its row order;
- `discovery` (optional): the check for the first-goal time: `family` (its row in the table at the top), `phase`,
  `control` (the control's method in that phase) and `method_names` (method -> name in words, the control among
  them). A case with units of that phase shows them, and the family's verdict is read from the mean capped
  first-goal time.

A verdict word is read from the 95% interval of a paired difference from the control: for the whole-curve mean
"ahead" when the interval is entirely above zero, "behind" when entirely below, else "level"; for the first-goal
time "ahead" when the interval is entirely below zero. A family's cell is for its arm with the highest
whole-curve mean on the first block. It rests on the second block under equal rules when an arm of the family was
run there, else on the first block.

Through the test helper's Slurm job (numpy never runs on a login node): see standing_job.py.
"""

import json
from pathlib import Path

from . import batch_summary, case_sweeps, contents, report, validation_seeds

SETTING_KEYS = {'title', 'validation_seeds', 'families', 'discovery'}
DISCOVERY_KEYS = {'family', 'phase', 'control', 'method_names'}


def read_settings(folder):
    """The page's settings, with their keys checked."""
    path = Path(folder) / 'standing_settings.json'
    settings = json.loads(path.read_text())
    unknown, missing = set(settings) - SETTING_KEYS, SETTING_KEYS - {'discovery'} - set(settings)
    if unknown or missing:
        raise ValueError(f'{path}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}.')
    if 'discovery' in settings:
        case_sweeps.check_keys(settings['discovery'], DISCOVERY_KEYS, DISCOVERY_KEYS, 'standing page, discovery')
    named = {arm['family'] for arm in settings['validation_seeds']['arms'] if 'family' in arm}
    if named - set(settings['families']):
        raise ValueError(f"{path}: the arms name the families {sorted(named - set(settings['families']))}, which `families` does not list.")
    return settings


def read_discovery(run_dir, block, environment, reward_setting, specifications):
    """The discovery check of one case: its sides, and each method at its best for M against the control at its own."""
    sides = {slug: case_sweeps.read_side(run_dir, case_sweeps.select(specifications, [block['phase']], slug, environment, reward_setting),
                                         slug, name) for slug, name in block['method_names'].items()}
    control = sides[block['control']]
    if not any(side.units for side in sides.values()):
        return None
    comparisons = {}
    if any(side.groups for side in sides.values()):
        budget = case_sweeps.one_budget(sides.values(), f"the phase {block['phase']}")
        comparisons = {slug: case_sweeps.compare_at_best(side, control, budget,
                                                         *case_sweeps.validation_name(block['phase'], environment, reward_setting))
                       for slug, side in sides.items() if slug != block['control']}
    return {'sides': sides, 'comparisons': comparisons}


# ---- the table at the top -------------------------------------------------------------------------------------------

def not_final(waiting):
    """What a cell and its sentence add when units of the family or of the control are not complete on the block."""
    # before: ['held divisor alone: 2 of 3 configurations read (2 of 3 units complete)']; after: the mark for the
    #         cell and the sentence that names what is waited for. Nothing is added when everything is read.
    if not waiting:
        return '', ''
    return '; not final, units still running', f". Not final: {'; '.join(waiting)}"


def family_cell(block, case, family):
    """One cell of the table at the top: (the cell's text, the sentence on which arm it is for; None without a comparison).

    The cell is for the family's arm with the highest W on the first block, so the arm too was fixed before the
    second block was read. It rests on the second block under equal rules when that arm, or another arm of the
    family, has that comparison; else on the first block. The sentence also gives the family's other arms on the
    block the cell rests on.
    """
    arms = [arm for arm in block['arms'] if arm.get('family') == family]
    # Each arm's first-block comparison and its second-block row under equal rules, where they are complete.
    first = {arm['name']: case['first']['comparisons'].get(arm.get('first_block')) for arm in arms}
    first = {name: entry['whole'] for name, entry in first.items()
             if entry is not None and entry['whole']['result'] is not None and entry['whole']['result']['seeds'] >= 2}
    second = case['second']
    rows = {arm['name']: second['metrics']['whole']['rows'].get(arm.get('second_block')) for arm in arms} if second is not None else {}
    rows = {name: row for name, row in rows.items() if row is not None and row['equal'] is not None and row['equal']['seeds'] >= 2}
    earlier = lambda name: first[name]['measures']['whole_curve_mean'] if name in first else float('-inf')
    words = lambda result: f"{case_sweeps.fine_difference(result, 4)} ({result['side']})"
    if rows:
        chosen = max(rows, key=earlier)
        equal, favouring = rows[chosen]['equal']['whole_curve_mean'], rows[chosen]['favouring']['whole_curve_mean']
        others = [f"{name} {words(row['equal']['whole_curve_mean'])}" for name, row in rows.items() if name != chosen]
        waiting = validation_seeds.unfinished(second['sides'], second['control'].slug,
                                              *(arm['second_block'] for arm in arms if 'second_block' in arm and second['sides'][arm['second_block']].units))
        favoured = ("the control's first-block best is also its best on the second block" if rows[chosen]['favouring'] is rows[chosen]['equal']
                    else f'with the control at its best on the second block the difference is {words(favouring)}')
        return (f"{equal['side']}: {case_sweeps.fine_difference(equal, 4)}, second block under equal rules" + not_final(waiting)[0],
                f"{chosen}; {favoured}"
                + (f". Other arms of the family on the second block under equal rules: {'; '.join(others)}" if others else '')
                + not_final(waiting)[1])
    if first:
        chosen = max(first, key=earlier)
        result = first[chosen]['result']['whole_curve_mean']
        others = [f"{name} {words(entry['result']['whole_curve_mean'])}" for name, entry in first.items() if name != chosen]
        sides = case['first']['sides']
        waiting = validation_seeds.unfinished(sides, case['first']['control'],
                                              *(arm['first_block'] for arm in arms if 'first_block' in arm and sides[arm['first_block']].units))
        return (f"{result['side']}: {case_sweeps.fine_difference(result, 4)}, first block" + not_final(waiting)[0],
                chosen + (f". Other arms of the family on the first block: {'; '.join(others)}" if others else '') + not_final(waiting)[1])
    # No comparison: units of the family exist on a block and are not complete (or the control's are not), or none exists.
    sides = [case['first']['sides'].get(arm.get('first_block')) for arm in arms]
    sides += [second['sides'].get(arm.get('second_block')) for arm in arms] if second is not None else []
    started = any(side is not None and side.units for side in sides)
    return ('no complete comparison yet' if started else 'not run'), None


def discovery_cell(found, block):
    """The cell of the first-goal-time family in one case: (text, which method it is for and how the others stand)."""
    if found is None:
        return 'not run', None
    entries = [(slug, entry['discovery']) for slug, entry in found['comparisons'].items()
               if entry is not None and entry['discovery']['result'] is not None and entry['discovery']['result']['seeds'] >= 2]
    if not entries:
        return 'no complete comparison yet', None
    slug, entry = min(entries, key=lambda pair: pair[1]['measures']['capped_first_goal_step'])
    result = entry['result']['capped_first_goal_step']
    words = lambda found: f"{report.discovery_interval_text(found)} thousand steps ({found['side']})"
    others = [f"{block['method_names'][other]} {words(found['result']['capped_first_goal_step'])}"
              for other, found in entries if other != slug]
    return (f"{result['side']}: {report.discovery_interval_text(result)} thousand steps, first block of the discovery check",
            block['method_names'][slug] + ', the method of the check with the smallest $M$'
            + (f". Other methods of the check: {'; '.join(others)}" if others else ''))


def top_table(settings, cases, discovery):
    """The table at the top (families in rows, cases in columns) and, under it, which arm each cell is for."""
    block = settings['validation_seeds']
    header = ['family of arms'] + [case['words'] for case in cases.values()]
    rows, notes = [], []
    for family in settings['families']:
        cells = []
        for key, case in cases.items():
            text, arm = family_cell(block, case, family)
            cells.append(text)
            if arm is not None:
                notes.append(f"- **{family}, {case['words']}:** {arm}.\n")
        rows.append([f'{family} (whole-curve mean)'] + cells)
    if 'discovery' in settings:
        cells = []
        for key, case in cases.items():
            text, name = discovery_cell(discovery[key], settings['discovery'])
            cells.append(text)
            if name is not None:
                notes.append(f"- **{settings['discovery']['family']}, {case['words']}:** {name}.\n")
        rows.append([f"{settings['discovery']['family']} (mean capped first-goal time)"] + cells)
    return report.markdown_table(header, rows) + 'Which arm each cell is for:\n\n' + ''.join(notes) + '\n'


# ---- the page -------------------------------------------------------------------------------------------------------

def command(run_dir):
    """The command that regenerates the page, from the platform folder."""
    platform = Path(run_dir).parents[1]
    return (f'cd {platform}\n'
            f'bash "{batch_summary.SUBMITTER}" -s scripts/maze_campaign/analysis/standing_job.py \\\n'
            f'    --run-dir=runs/{Path(run_dir).name}\n')


def build(run_dir, settings, cases, discovery):
    """The whole page in Markdown, section by section."""
    block = settings['validation_seeds']
    specifications = case_sweeps.read_specifications(run_dir) if 'discovery' in settings else None
    parts = [f"# {settings['title']}\n\n"
             f'Generated {contents.pacific(contents.now())} by the command below. This file is generated: regenerate '
             'it, do not edit it. Times are Pacific, converted from machines running Eastern. Run folder: '
             f'`{Path(run_dir).name}`.\n\n```bash\n{command(run_dir)}```\n\n',
             '## Standing in one table\n\n'
             'Each cell gives a verdict word, the paired difference from the control (CleanRL PPO-RND) with its 95% '
             'interval, and the block of validation seeds it rests on. For the whole-curve mean $W$: "ahead" when the '
             'interval is entirely above zero, "behind" when entirely below, else "level". For the mean capped '
             'first-goal time $M$: "ahead" when the interval is entirely below zero. A cell is for the family\'s arm '
             'with the highest $W$ on the first block, among its arms run on the block the cell rests on. It rests '
             'on the second block when an arm of the family was '
             'run there (the method at the configuration fixed on the first block, against the control at the '
             'configuration that was its best on the first block), else on the first block (each side at its best '
             'configuration there, chosen on those same seeds, which is a second selection step). "no complete '
             'comparison yet" means units exist and the method or the control has none complete; "not run" means '
             'no unit; "not final, units still running" means the cell is read from the complete units while other '
             'configurations of the family or of the control are not complete, so a best configuration and the '
             'verdict can still change. None of this is the comparison on the '
             'confirmation seeds. The families are in rows and the four cases in columns.\n\n'
             + top_table(settings, cases, discovery),
             case_sweeps.measures_section()]
    for (environment, reward_setting), case in cases.items():
        parts.append(validation_seeds.case_text(block, environment, reward_setting, case, level='##'))
        if 'discovery' in settings and discovery[(environment, reward_setting)] is not None:
            parts.append(case_sweeps.validation_part(
                run_dir, {key: settings['discovery'][key] for key in case_sweeps.VALIDATION_KEYS}, environment,
                reward_setting, specifications, ('discovery',),
                '### Discovery check: the leading configurations for the first-goal time\n\n',
                "these configurations were picked as the leading ones of each method's discovery sweep"))
    return ''.join(parts)


def generate(run_dir):
    """Write the run folder's analysis/standing.md and standing.html from its shards."""
    run_dir = Path(run_dir).resolve()
    folder = run_dir / 'analysis'
    settings = read_settings(folder)
    cases = validation_seeds.read_cases(run_dir, settings['validation_seeds'], 'standing page, validation_seeds')
    discovery = {}
    if 'discovery' in settings:
        specifications = case_sweeps.read_specifications(run_dir)
        discovery = {key: read_discovery(run_dir, settings['discovery'], *key, specifications) for key in cases}
    markdown = build(run_dir, settings, cases, discovery)
    states = contents.write_report(folder, markdown, settings['title'], name='standing')
    return {'page': folder / 'standing.md', 'sections': states, 'cases': cases, 'discovery': discovery}
