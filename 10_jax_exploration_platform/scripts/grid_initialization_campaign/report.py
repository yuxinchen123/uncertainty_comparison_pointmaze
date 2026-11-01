"""Generate initial-bonus results and behavior figures from completed run records."""

import argparse
import csv
from collections import defaultdict
import difflib
import hashlib
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys

import numpy as np

from common import PLATFORM, now, pacific, read_json, seed_for, write_json
from campaign import result_records

sys.path.insert(0, str(PLATFORM / 'src'))
from exploration_platform.grid_initialization.fields import ADVERSARIAL_FIELD_KINDS


WAVE_ORDER = ('calibration', 'field-search', 'diagnosis', 'fine-tuning', 'confirmation', 'long-confirmation')
LABELS = {'uniform': 'Uniform initial bonuses', 'none': 'No intrinsic bonus',
          'random_log_uniform': 'Random initial bonuses', 'random_network': 'Frozen coordinate networks'}
COLORS = ('#0072B2', '#E69F00', '#009E73', '#CC79A7', '#D55E00')
LINE_STYLES = ('solid', 'dashed', 'dashdot', 'dotted', (0, (3, 1, 1, 1, 1, 1)))


def capped_first_goal_steps(first_goal_steps, budget):
    """Count failures at the full common budget instead of dropping them."""
    # Before: [12,-1,30] at budget 40; after: [12,40,30].
    values = np.asarray(first_goal_steps, dtype=np.int64)
    if values.ndim != 1 or np.any(values < -1) or np.any(values > budget):
        raise ValueError('First-goal steps must be a vector of -1 or steps within the budget.')
    return np.where(values < 0, budget, values).astype(np.float64)


def grouped_mean_interval(values, group_ids, seed, repetitions=2000):
    """Resample independent field seeds while retaining all their policy seeds together."""
    # Before: policies [1,3] for one field and [8,10] for another; after: two field groups.
    values = np.asarray(values, dtype=np.float64)
    identities = np.asarray(group_ids)
    if values.ndim != 1 or identities.shape != values.shape or not len(values):
        raise ValueError('Uncertainty requires matching nonempty value and group vectors.')
    unique, inverse = np.unique(identities, return_inverse=True)
    sums = np.bincount(inverse, weights=values, minlength=len(unique))
    counts = np.bincount(inverse, minlength=len(unique))
    if len(unique) < 2:
        return [None, None]

    # Whole groups are sampled with replacement; unequal group sizes keep their original weights.
    generator = np.random.default_rng(seed)
    chosen = generator.integers(0, len(unique), size=(repetitions, len(unique)))
    means = sums[chosen].sum(axis=1) / counts[chosen].sum(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def field_label(specification):
    """Name a field by its literal initialization rule and relevant severity."""
    # Numerical exponents stay in words so table entries remain ordinary wrappable text.
    kind = specification['field_kind']
    label = LABELS.get(kind, kind.replace('_', ' ').capitalize())
    if kind in ADVERSARIAL_FIELD_KINDS or kind == 'random_log_uniform':
        label += f"; severity exponent {specification['severity_exponent']}"
    return label


def geometry_label(specification):
    """Describe a geometry without hiding its width or number of turns."""
    # A snake's width is its horizontal corridor width, not its total number of states.
    geometry = specification['geometry']
    return f"{geometry['kind'].capitalize()} grid, width {geometry['width']}, turns {geometry['turns']}"


def condition_variant(specification):
    """Keep sensitivity conditions separate while treating rollout length as a training setting."""
    # Different rollout lengths already appear in configuration; transforms and normalization do not.
    variant = specification.get('variant', 'primary')
    return 'primary' if variant.startswith('rollout-') else variant


def collect_rows(run_dir):
    """Combine seed batches only when their geometry, field, settings, and budget match."""
    # Completed summaries supply selection and metrics; behavior archives are loaded only for plots.
    grouped = {}
    for record in result_records(run_dir):
        specification = record['specification']
        for setting in record['settings']:
            configuration = setting['configuration']
            identity = json.dumps([specification['wave'], specification['geometry'],
                specification['policy_kind'], specification['field_kind'],
                specification['severity_exponent'], specification['total_timesteps'], configuration,
                condition_variant(specification)],
                sort_keys=True)
            if identity not in grouped:
                grouped[identity] = {'identity': identity, 'specification': specification,
                    'configuration': configuration, 'first_goal_steps': [], 'policy_seeds': [],
                    'field_seeds': [], 'coverage_sum': 0.0, 'sources': []}
            row = grouped[identity]
            count = len(setting['first_goal_steps'])
            for key in ('first_goal_steps', 'policy_seeds', 'field_seeds'):
                if len(setting[key]) != count:
                    raise ValueError(f"Seed metric lengths disagree for {specification['name']}.")
                row[key].extend(setting[key])
            row['coverage_sum'] += float(setting['coverage_mean']) * count
            row['sources'].append({'task': specification['name'], 'setting_index': setting['setting_index']})

    # Recompute the capped mean from every completed seed rather than averaging batch means.
    rows = []
    for row in grouped.values():
        specification = row['specification']
        budget = specification['total_timesteps']
        capped = capped_first_goal_steps(row['first_goal_steps'], budget)
        if len(set(zip(row['policy_seeds'], row['field_seeds']))) != len(capped):
            raise ValueError(f"Repeated policy/field seed pairs in {specification['name']}.")
        groups = row['field_seeds'] if specification['field_kind'] in (
            'random_network', 'random_log_uniform') else row['policy_seeds']
        successes = np.asarray(row['first_goal_steps']) >= 0
        row.update(seed_count=len(capped), independent_group_count=len(set(groups)),
                   capped_mean_first_goal_steps=float(capped.mean()),
                   capped_mean_interval=grouped_mean_interval(capped, groups,
                       seed_for('report', row['identity'], 'mean_steps')),
                   goal_fraction=float(successes.mean()),
                   goal_fraction_interval=grouped_mean_interval(successes, groups,
                       seed_for('report', row['identity'], 'goal_fraction')),
                   coverage_mean=row['coverage_sum'] / len(capped))
        rows.append(row)
    return rows


def select_reporting_rows(rows):
    """Use development minima but preserve every frozen confirmation comparison."""
    # Fresh confirmation data never choose a winning learning rate or bonus coefficient.
    selected = {}
    for row in rows:
        specification = row['specification']
        confirmation = specification['wave'] in ('confirmation', 'long-confirmation')
        comparison = row['configuration'].get('comparison', 'development best')
        key = json.dumps([specification['wave'], specification['geometry'], specification['policy_kind'],
                          specification['field_kind'], specification['severity_exponent'],
                          specification['total_timesteps'], comparison, condition_variant(specification)], sort_keys=True)
        if key in selected and confirmation:
            raise ValueError('A frozen confirmation comparison contains multiple different configurations.')
        if key not in selected or row['capped_mean_first_goal_steps'] < selected[key]['capped_mean_first_goal_steps']:
            selected[key] = row
    return sorted(selected.values(), key=lambda row: (
        WAVE_ORDER.index(row['specification']['wave']), geometry_label(row['specification']),
        row['specification']['policy_kind'], row['configuration'].get('comparison', ''),
        condition_variant(row['specification']),
        field_label(row['specification'])))


def select_focus_rows(rows, state):
    """Limit expensive presentation to maps chosen before confirmation outcomes are observed."""
    # Numerical tables and seed files retain all maps; this controls only main figures and LaTeX.
    if not rows:
        return []
    latest = max(WAVE_ORDER.index(row['specification']['wave']) for row in rows)
    available = [row for row in rows if WAVE_ORDER.index(row['specification']['wave']) == latest
                 and condition_variant(row['specification']) == 'primary']
    chosen = state.get('selected_comparisons', []) if state else []
    if chosen:
        allowed = {(json.dumps(item['geometry'], sort_keys=True), item['policy_kind']) for item in chosen}
    elif state and state.get('selected_maps'):
        allowed = {(json.dumps(geometry, sort_keys=True), policy)
                   for geometry in state['selected_maps'] for policy in ('tabular', 'neural')}
    elif state is not None:
        return []
    else:
        # Standalone schema checks have no campaign state; a fixed metadata order remains bounded.
        allowed = set(sorted({(json.dumps(row['specification']['geometry'], sort_keys=True),
                               row['specification']['policy_kind']) for row in available})[:8])
    return [row for row in available if (json.dumps(row['specification']['geometry'], sort_keys=True),
                                         row['specification']['policy_kind']) in allowed]


def _highlight_changed_html(current, previous):
    """Color only text inserted or replaced since the section was explicitly marked read."""
    class TextReader(HTMLParser):
        """Collect rendered text separately from its HTML elements."""

        def __init__(self):
            """Start an empty text stream."""
            super().__init__(convert_charrefs=True)
            self.parts = []

        def handle_data(self, data):
            """Retain visible text in its original order."""
            self.parts.append(data)

    # Compare visible text so an inserted link or image cannot break the surrounding HTML tags.
    old_reader, current_reader = TextReader(), TextReader()
    old_reader.feed(previous)
    current_reader.feed(current)
    changes = [(start, end) for action, _, _, start, end in difflib.SequenceMatcher(
        None, ''.join(old_reader.parts), ''.join(current_reader.parts), autojunk=False).get_opcodes()
        if action in ('insert', 'replace')]

    class ChangedTextWriter(HTMLParser):
        """Preserve HTML while coloring changed ranges of its text stream."""

        def __init__(self):
            """Start reconstruction at the first visible character."""
            super().__init__(convert_charrefs=True)
            self.parts, self.offset = [], 0

        def handle_starttag(self, tag, attributes):
            """Preserve the renderer's complete opening element."""
            self.parts.append(self.get_starttag_text())

        def handle_startendtag(self, tag, attributes):
            """Preserve images and other self-closing elements."""
            self.parts.append(self.get_starttag_text())

        def handle_endtag(self, tag):
            """Close the existing renderer element."""
            self.parts.append(f'</{tag}>')

        def handle_data(self, data):
            """Split at changed ranges without coloring unchanged words."""
            end = self.offset + len(data)
            boundaries = sorted({self.offset, end, *[max(self.offset, start) for start, stop in changes
                if start < end and stop > self.offset], *[min(end, stop) for start, stop in changes
                if start < end and stop > self.offset]})
            for start, stop in zip(boundaries, boundaries[1:]):
                content = html.escape(data[start-self.offset:stop-self.offset])
                changed = any(left <= start < right for left, right in changes)
                self.parts.append('<span class="changed">'+content+'</span>' if changed else content)
            self.offset = end

    writer = ChangedTextWriter()
    writer.feed(current)
    return ''.join(writer.parts)


def _write_report_with_history(analysis_dir, markdown):
    """Add section history and render read, unread, and updated states in a browser report."""
    from markdown_it import MarkdownIt

    # Section history records actual content changes, never file modification times or automatic reads.
    history_path = analysis_dir / 'section_history.json'
    history = read_json(history_path) if history_path.exists() else {'sections': {}}
    split = re.split(r'(?m)^## (.+)\n', markdown)
    renderer = MarkdownIt('commonmark', {'html': False}).enable('table')
    contents = ['## Contents\n\n| Section | First written | Last changed | Status |\n|---|---|---|---|\n']
    sections, html_sections = [], []
    timestamp = now()
    for index in range(1, len(split), 2):
        title, text = split[index], split[index+1]
        identifier = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')
        images = {name: hashlib.sha256((analysis_dir/name).read_bytes()).hexdigest()
                  for name in re.findall(r'!\[[^\]]*\]\(([^)]+)\)', text)}
        digest = hashlib.sha256((text+json.dumps(images, sort_keys=True)).encode()).hexdigest()
        saved = history['sections'].get(identifier, {})
        record = {**saved, 'title': title, 'content_digest': digest,
            'first_added': saved.get('first_added', timestamp),
            'last_changed': saved.get('last_changed', timestamp) if saved.get('content_digest') == digest else timestamp,
            'last_read_digest': saved.get('last_read_digest'), 'last_read_text': saved.get('last_read_text'),
            'last_read_figures': saved.get('last_read_figures', {}), 'current_figures': images,
            'current_text': text}
        status = 'unread' if not record['last_read_digest'] else ('read' if record['last_read_digest'] == digest else 'updated')
        record['status'] = status
        history['sections'][identifier] = record
        contents.append(f'| [{title}](#{identifier}) | {pacific(record["first_added"])} | {pacific(record["last_changed"])} | {status} |\n')
        body = renderer.render(text)
        if status == 'updated':
            body = _highlight_changed_html(body, renderer.render(record['last_read_text'] or ''))
        for image_name, image_digest in images.items():
            if record['last_read_digest'] and image_digest != record['last_read_figures'].get(image_name):
                body = re.sub(r'(<img[^>]*src="'+re.escape(image_name)+r'"[^>]*>)',
                              r'\1<small class="figure-updated">updated</small>', body)
        sections.append(f'## {title}\n'+text)
        html_sections.append(f'<section id="{identifier}" class="{status}"><h2>{html.escape(title)}</h2>{body}</section>')

    # The HTML companion provides colors; Markdown retains ordinary readable links and explicit statuses.
    contents_markdown = ''.join(contents)+'\n[Read with section colors](analysis.html). No section marks itself read.\n\n'
    (analysis_dir/'analysis.md').write_text(split[0]+contents_markdown+''.join(sections))
    html_contents = renderer.render(''.join(contents))
    for identifier, record in history['sections'].items():
        html_contents = html_contents.replace(f'href="#{identifier}"', f'class="{record["status"]}" href="#{identifier}"')
    style = '''body{font:16px system-ui;max-width:1150px;margin:32px auto;padding:0 20px;line-height:1.5}
table{border-collapse:collapse;width:100%;margin:18px 0}th,td{padding:6px;border-bottom:1px solid #bbb;text-align:left;overflow-wrap:anywhere}
img{max-width:100%;height:auto}section{margin-top:32px}.unread,.unread a{color:#2452cc}.changed{color:#852D0F}.figure-updated{display:block;color:#2452cc}
@media(prefers-color-scheme:dark){body{background:#171717;color:#eee}.unread,.unread a{color:#8bb4ff}.changed{color:#e7a48f}a{color:#adc7ff}}
@media print{body{max-width:none;margin:0}table{font-size:10pt}img{break-inside:avoid}}'''
    (analysis_dir/'analysis.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Initial bonus research</title><style>'+style+
        '</style>'+renderer.render(split[0])+html_contents+''.join(html_sections)+'</html>')
    write_json(history_path, history)


def paired_confirmation_comparisons(rows, repetitions=8000):
    """Estimate paired adverse-minus-uniform effects with four-comparison adjusted intervals."""
    # Primary comparisons exclude shuffled fields, flattened fields, normalization, and path controls.
    confirmed = [row for row in rows if row['specification']['wave'] in (
        'confirmation', 'long-confirmation') and condition_variant(row['specification']) == 'primary']
    references = {}
    for row in confirmed:
        spec = row['specification']
        if spec['field_kind'] == 'uniform':
            key = (spec['wave'], json.dumps(spec['geometry'], sort_keys=True), spec['policy_kind'],
                   spec['total_timesteps'], row['configuration'].get('comparison', 'development best'))
            references[key] = row
    comparisons = []
    for row in confirmed:
        spec = row['specification']
        requested = spec['field_kind'] in ADVERSARIAL_FIELD_KINDS if spec['geometry']['kind'] == 'straight' else spec['field_kind'] == 'random_network'
        if not requested:
            continue
        key = (spec['wave'], json.dumps(spec['geometry'], sort_keys=True), spec['policy_kind'],
               spec['total_timesteps'], row['configuration'].get('comparison', 'development best'))
        if key not in references:
            continue
        reference = references[key]
        reference_indices = {int(seed): index for index, seed in enumerate(reference['policy_seeds'])}
        candidate_indices = {int(seed): index for index, seed in enumerate(row['policy_seeds'])}
        paired_seeds = sorted(reference_indices.keys() & candidate_indices.keys())
        if not paired_seeds:
            continue

        # Align raw outcomes by policy seed before applying censoring or forming differences.
        first_reference = np.asarray([reference['first_goal_steps'][reference_indices[seed]] for seed in paired_seeds])
        first_candidate = np.asarray([row['first_goal_steps'][candidate_indices[seed]] for seed in paired_seeds])
        reference_steps = capped_first_goal_steps(first_reference, spec['total_timesteps'])
        candidate_steps = capped_first_goal_steps(first_candidate, spec['total_timesteps'])
        if reference_steps.mean() <= 0:
            raise ValueError('A first-goal ratio requires a positive reference first-goal time.')
        groups = [row['field_seeds'][candidate_indices[seed]] for seed in paired_seeds] if spec['field_kind'] == 'random_network' else paired_seeds
        unique, inverse = np.unique(groups, return_inverse=True)
        reference_sums = np.bincount(inverse, weights=reference_steps, minlength=len(unique))
        candidate_sums = np.bincount(inverse, weights=candidate_steps, minlength=len(unique))
        counts = np.bincount(inverse, minlength=len(unique))
        difference_interval = ratio_interval = [None, None]
        if len(unique) >= 2:
            generator = np.random.default_rng(seed_for('paired_confirmation', *key, spec['field_kind']))
            sampled = generator.integers(0, len(unique), (repetitions, len(unique)))
            total_reference = reference_sums[sampled].sum(axis=1)
            total_candidate = candidate_sums[sampled].sum(axis=1)
            total_counts = counts[sampled].sum(axis=1)
            # Two-sided 98.75% intervals allocate 5% across four environment/policy comparisons.
            difference_interval = np.quantile((total_candidate-total_reference)/total_counts, [.00625, .99375]).tolist()
            ratio_interval = np.quantile(total_candidate/total_reference, [.00625, .99375]).tolist()
        reference_goal = float((first_reference >= 0).mean())
        candidate_goal = float((first_candidate >= 0).mean())
        label = 'No confirmed increase in first-goal time'
        if difference_interval[0] is not None and difference_interval[0] > 0:
            label = 'Increased first-goal time'
            if ratio_interval[0] >= 2 and reference_goal >= .75:
                label = 'At least twice the capped mean first-goal time'
        if reference_goal < .75:
            label += '; uniform initialization did not reach the 75 percent discovery requirement'
        comparisons.append({'wave': spec['wave'], 'geometry': spec['geometry'], 'policy_kind': spec['policy_kind'],
            'comparison': key[-1], 'field_kind': spec['field_kind'], 'field_label': field_label(spec),
            'budget': spec['total_timesteps'], 'paired_seed_count': len(paired_seeds),
            'independent_group_count': len(unique), 'uniform_capped_mean': float(reference_steps.mean()),
            'candidate_capped_mean': float(candidate_steps.mean()),
            'mean_difference': float((candidate_steps-reference_steps).mean()),
            'mean_difference_interval': difference_interval,
            'mean_ratio': float(candidate_steps.mean()/reference_steps.mean()),
            'mean_ratio_interval': ratio_interval, 'interval_confidence': .9875,
            'family_comparison_count': 4, 'uniform_goal_fraction': reference_goal,
            'candidate_goal_fraction': candidate_goal, 'effect_label': label,
            'all_current_seeds_paired': len(paired_seeds) == len(reference_indices) == len(candidate_indices)})
    return comparisons


def _load_behavior(run_dir, row):
    """Gather one selected setting's complete seed arrays and sampled movement history."""
    # Before: archives containing all settings; after: matching copies concatenated along seed axis.
    gathered = defaultdict(list)
    trajectories, trajectory_steps, maximum_history = [], [], []
    positions = grid = None
    for source in row['sources']:
        path = Path(run_dir) / 'data/research' / source['task'] / 'behavior.npz'
        with np.load(path, allow_pickle=False) as archive:
            selected = archive['setting_indices'] == source['setting_index']
            if positions is not None and (not np.array_equal(positions, archive['positions']) or
                                          not np.array_equal(grid, archive['grid'])):
                raise ValueError('A reporting group contains incompatible physical grids.')
            positions, grid = archive['positions'].copy(), archive['grid'].copy()
            for key in ('initial_fields', 'visit_counts', 'late_visit_counts', 'policy_probabilities',
                        'first_goal_steps', 'seed_ids', 'field_seed_ids'):
                gathered[key].append(archive[key][selected].copy())
            for copy in np.flatnonzero(selected):
                trajectories.append(archive['trajectory_states'][:, copy].copy())
                trajectory_steps.append(archive['trajectory_steps'].copy())
                maximum_history.append(archive['maximum_progress_history'][:, copy].copy())
    arrays = {key: np.concatenate(values, axis=0) for key, values in gathered.items()}
    arrays.update(positions=positions, grid=grid, trajectories=trajectories,
                  trajectory_steps=trajectory_steps, maximum_history=maximum_history)
    if len(arrays['seed_ids']) != row['seed_count']:
        raise ValueError('Behavior archive seed count disagrees with completed summary metrics.')
    return arrays


def representative_seed_index(first_goal_steps, budget, seed_ids, field_seed_ids):
    """Choose the seed nearest median capped first-goal time with fixed seed tie breaks."""
    # Failures all contribute the budget; equal candidates use increasing field and policy seeds.
    capped = capped_first_goal_steps(first_goal_steps, budget)
    median = np.median(capped)
    return min(range(len(capped)), key=lambda index: (
        abs(capped[index] - median), int(field_seed_ids[index]), int(seed_ids[index])))


def _draw_map(axis, values, grid, title, normalization=None, label_states=False):
    """Draw true grid cells with walls distinct from unvisited states in vector output."""
    # Before: one value per traversable state; after: a masked physical grid with dark walls.
    from matplotlib import colormaps
    from matplotlib.colors import LogNorm
    displayed = np.asarray(values)
    if isinstance(normalization, LogNorm):
        displayed = np.where(displayed == 0, normalization.vmin / 2, displayed)
    field = np.ma.masked_where(grid < 0, displayed[np.maximum(grid, 0)])
    colors = colormaps['viridis'].copy()
    colors.set_under('white')
    axis.set_facecolor('#505050')
    mesh = axis.pcolormesh(np.arange(grid.shape[1] + 1) - 0.5,
                          np.arange(grid.shape[0] + 1) - 0.5, field,
                          cmap=colors, norm=normalization, edgecolors='none')
    axis.set_xlim(-0.5, grid.shape[1] - 0.5)
    axis.set_ylim(grid.shape[0] - 0.5, -0.5)
    axis.set_title(title, fontsize=11)
    axis.set_xlabel('Column')
    axis.set_ylabel('Row')
    axis.scatter([0, grid.shape[1] - 1], [0, grid.shape[0] - 1], s=25,
                 marker='o', facecolors='none', edgecolors='#D55E00', linewidths=1.5)
    if label_states:
        for row_index, column_index in np.argwhere(grid >= 0):
            axis.text(column_index, row_index, str(grid[row_index, column_index]),
                      ha='center', va='center', fontsize=7)
    return mesh


def _save_figure(figure, directory, name):
    """Keep a publication PDF and a directly readable PNG for each behavior figure."""
    # Both formats are generated from the same figure with the same vector wall geometry.
    figure.savefig(directory / (name + '.pdf'), bbox_inches='tight')
    figure.savefig(directory / (name + '.png'), bbox_inches='tight', dpi=180)


def _write_sampled_position_animation(rows, data, directory, name):
    """Write a standalone browser animation of recorded training-position samples."""
    # Before: positions at recorded training times; after: a slider that never invents intermediate actions.
    conditions = []
    for row, values in zip(rows, data):
        index = representative_seed_index(values['first_goal_steps'], row['specification']['total_timesteps'],
                                         values['seed_ids'], values['field_seed_ids'])
        conditions.append({'label': field_label(row['specification']),
            'policy_seed': int(values['seed_ids'][index]), 'field_seed': int(values['field_seed_ids'][index]),
            'states': values['trajectories'][index].tolist(), 'steps': values['trajectory_steps'][index].tolist(),
            'visits': values['visit_counts'].mean(axis=0).tolist()})
    payload = json.dumps({'positions': data[0]['positions'].tolist(), 'shape': list(data[0]['grid'].shape),
                          'conditions': conditions}).replace('</', '<\\/')
    title = html.escape(geometry_label(rows[0]['specification']) + '; ' + rows[0]['specification']['policy_kind'] + ' policy')

    # All styles, geometry, seed identifiers and sampled positions are embedded; no network is needed.
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Sampled training positions</title>
<style>body{font:16px system-ui;margin:24px;color:#202020}h1{font-size:22px}button,input{font:inherit}
.cards{display:flex;flex-wrap:wrap;gap:16px}.card{border:1px solid #aaa;padding:12px;flex:1;min-width:260px}
svg{display:block;width:100%;height:220px;background:#505050}.note{max-width:900px}button{margin-right:12px}
input{width:min(700px,70vw)}.detail{font-size:14px}</style>
<h1>''' + title + '''</h1><p class="note">Recorded position samples during training. Intermediate actions are not shown.
Each condition uses the seed nearest its median capped first-goal time. Background colors show final mean visitation;
the red marker shows the selected seed's recorded cell. The marker may span several cells for visibility.</p>
<p><button id="play">Play</button><input id="sample" type="range" min="0" value="0"><span id="time"></span></p>
<div id="cards" class="cards"></div><script>
const evidence=''' + payload + ''';
const ns='http://www.w3.org/2000/svg';const markers=[];const details=[];
const make=(tag,attributes)=>{const item=document.createElementNS(ns,tag);for(const [key,value]of Object.entries(attributes))item.setAttribute(key,value);return item};
const maximum=Math.max(...evidence.conditions.flatMap(condition=>condition.visits));
for(const condition of evidence.conditions){const card=document.createElement('section');card.className='card';
const heading=document.createElement('h2');heading.style.fontSize='17px';heading.textContent=condition.label;card.append(heading);
const seed=document.createElement('p');seed.className='detail';seed.textContent=`Policy seed ${condition.policy_seed}; field seed ${condition.field_seed}`;card.append(seed);
const svg=make('svg',{viewBox:`-0.5 -0.5 ${evidence.shape[1]} ${evidence.shape[0]}`,preserveAspectRatio:'none'});
for(let state=0;state<evidence.positions.length;state++){const [row,column]=evidence.positions[state];
const fraction=Math.log1p(condition.visits[state])/Math.log1p(Math.max(1,maximum));
const shade=Math.round(255-190*fraction);svg.append(make('rect',{x:column-.5,y:row-.5,width:1,height:1,fill:`rgb(${shade},${shade},255)`}));}
const marker=make('circle',{r:Math.max(.35,evidence.shape[1]/100,evidence.shape[0]/100),fill:'#d55e00',stroke:'white','stroke-width':.1});
svg.append(marker);card.append(svg);const detail=document.createElement('p');detail.className='detail';card.append(detail);details.push(detail);markers.push(marker);document.querySelector('#cards').append(card);}
const slider=document.querySelector('#sample');slider.max=Math.max(...evidence.conditions.map(condition=>condition.states.length))-1;
const draw=()=>{const sample=Number(slider.value);for(let index=0;index<markers.length;index++){const condition=evidence.conditions[index];
const location=Math.min(sample,condition.states.length-1);const state=condition.states[location];const [row,column]=evidence.positions[state];
markers[index].setAttribute('cx',column);markers[index].setAttribute('cy',row);details[index].textContent=`Training transitions: ${condition.steps[location].toLocaleString()}; distance along path: ${state}`;}
document.querySelector('#time').textContent=` Sample ${sample+1} of ${Number(slider.max)+1}`;};slider.addEventListener('input',draw);
let timer=null;document.querySelector('#play').addEventListener('click',()=>{if(timer!==null){clearInterval(timer);timer=null;document.querySelector('#play').textContent='Play';return;}
document.querySelector('#play').textContent='Pause';timer=setInterval(()=>{slider.value=(Number(slider.value)+1)%(Number(slider.max)+1);draw();},350);});draw();
</script></html>'''
    (directory / (name + '_sampled_positions.html')).write_text(page)


def select_behavior_rows(rows):
    """Use the recorded reference fields and one development-selected adverse field in figures."""
    # Select at most one development adversary by the largest tuned capped mean, then freeze that view.
    priority = ('uniform', 'random_log_uniform', 'none', 'random_network')
    selected = [row for kind in priority for row in rows if row['specification']['field_kind'] == kind]
    adversaries = [row for row in rows if row['specification']['field_kind'] in ADVERSARIAL_FIELD_KINDS]
    if adversaries:
        selected.insert(min(2, len(selected)), max(adversaries, key=lambda row: row['capped_mean_first_goal_steps']))
    # Different random severities are tabled; the figure uses the strongest measured random condition.
    unique = {}
    for row in selected:
        kind = row['specification']['field_kind']
        if kind not in unique or row['capped_mean_first_goal_steps'] > unique[kind]['capped_mean_first_goal_steps']:
            unique[kind] = row
    return list(unique.values())


def _plot_comparison(run_dir, rows, plots_dir, figure_name):
    """Compare equal-budget initial fields, visits, movement, and learned action probabilities."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    # Tables and figures use the same presentation subset; all measured rows remain in the data.
    selected = select_behavior_rows(rows)
    data = [_load_behavior(run_dir, row) for row in selected]
    if not selected:
        return []

    # Every map in a row uses the same color scale and equal transition budgets.
    summaries = [[values[key].mean(axis=0) for values in data]
                 for key in ('initial_fields', 'visit_counts', 'late_visit_counts')]
    figure, axes = plt.subplots(3, len(selected), figsize=(4.5 * len(selected), 7.5), squeeze=False,
                                layout='constrained')
    for metric, fields in enumerate(summaries):
        maximum = max(float(field.max()) for field in fields)
        positive_minima = [float(field[field > 0].min()) for field in fields if np.any(field > 0)]
        minimum = min(positive_minima) if positive_minima else 1
        normalization = Normalize(vmin=0, vmax=max(1, maximum)) if metric == 0 else LogNorm(vmin=minimum, vmax=max(minimum * 2, maximum))
        for column, (row, values, field) in enumerate(zip(selected, data, fields)):
            label = field_label(row['specification']) if metric == 0 else ('Total visits' if metric == 1 else 'Late visits')
            mesh = _draw_map(axes[metric, column], field, values['grid'], label, normalization)
        figure.colorbar(mesh, ax=axes[metric].tolist(), label=('Initial bonus' if metric == 0 else 'Mean visits per seed'))
    figure.suptitle(geometry_label(selected[0]['specification']) + '; ' + selected[0]['specification']['policy_kind'] + ' policy', fontsize=12)
    _save_figure(figure, plots_dir, figure_name + '_maps')
    plt.close(figure)

    # Separate field profiles expose neural variation without the large adversarial map color range.
    figure, axes = plt.subplots(1, len(selected), figsize=(4.5 * len(selected), 3.5), squeeze=False,
                                layout='constrained')
    for column, (row, values) in enumerate(zip(selected, data)):
        axis = axes[0, column]
        fields = values['initial_fields']
        distance = np.linspace(0, 1, fields.shape[1])
        _, unique_indices = np.unique(values['field_seed_ids'], return_index=True)
        for number, index in enumerate(unique_indices):
            axis.plot(distance, fields[index], color=COLORS[column % len(COLORS)], alpha=.22,
                      linewidth=.8, label='One field seed' if number == 0 else None)
        axis.plot(distance, fields.mean(axis=0), color=COLORS[column % len(COLORS)], linewidth=2,
                  label='Mean initial bonus')
        axis.axhline(1, color='#555555', linestyle='dotted', linewidth=1, label='Uniform reference: 1')
        if np.all(fields > 0):
            axis.set_yscale('log')
        axis.set_xlim(0, 1)
        axis.set_xlabel('Normalized distance along path')
        axis.set_ylabel('Initial bonus; separate scale per panel')
        axis.set_title(field_label(row['specification']), fontsize=10)
        axis.legend(fontsize=7)
    _save_figure(figure, plots_dir, figure_name + '_initial_profiles')
    plt.close(figure)

    # Sampled training positions across all seeds show persistent concentration over actual training time.
    figure, axes = plt.subplots(2, len(selected), figsize=(4.5 * len(selected), 6), squeeze=False,
                                layout='constrained')
    representatives = []
    for column, (row, values) in enumerate(zip(selected, data)):
        steps = values['trajectory_steps'][0]
        if any(not np.array_equal(steps, other) for other in values['trajectory_steps']):
            raise ValueError('Training position samples have inconsistent global step coordinates.')
        trajectories = np.stack(values['trajectories'], axis=1)
        density = np.zeros((len(values['positions']), len(steps)), dtype=np.float64)
        for time_index in range(len(steps)):
            density[:, time_index] = np.bincount(trajectories[time_index], minlength=len(values['positions'])) / row['seed_count']
        axes[0, column].pcolormesh(steps, np.arange(len(values['positions'])), density, shading='nearest',
                                  cmap='viridis', vmin=0, vmax=1)
        index = representative_seed_index(values['first_goal_steps'], row['specification']['total_timesteps'],
                                          values['seed_ids'], values['field_seed_ids'])
        axes[0, column].plot(steps, values['maximum_history'][index], color='#D55E00', linewidth=1,
                             label='Farthest state, representative seed')
        axes[0, column].set_title(field_label(row['specification']), fontsize=10)
        axes[0, column].set_xlabel('Training transitions per policy')
        axes[0, column].set_ylabel('Distance along path')
        axes[0, column].legend(fontsize=7, loc='lower right')
        axes[1, column].plot(steps, values['trajectories'][index], linewidth=.8, color=COLORS[column % len(COLORS)],
                             label='Recorded position samples')
        axes[1, column].set_xlabel('Training transitions per policy')
        axes[1, column].set_ylabel('Representative sampled position')
        # Use the exact recorded goal arrival, since a sampled position can skip an immediate reset.
        first_goal = int(values['first_goal_steps'][index])
        for axis in axes[:, column]:
            axis.set_ylim(0, len(values['positions'])-1)
            axis.set_xlim(0, row['specification']['total_timesteps'])
            if first_goal >= 0:
                axis.axvline(first_goal, color='#D55E00', linestyle='dashed', linewidth=1,
                             label=f'Exact first goal: {first_goal:,} transitions')
            else:
                axis.text(.03, .95, 'Goal not reached within budget', transform=axis.transAxes,
                          fontsize=7, va='top', bbox={'facecolor': 'white', 'alpha': .8, 'edgecolor': 'none'})
            axis.legend(fontsize=7, loc='lower right')
        representatives.append({'condition': field_label(row['specification']),
                                'policy_seed': int(values['seed_ids'][index]),
                                'field_seed': int(values['field_seed_ids'][index]),
                                'first_goal_step': first_goal})
    figure.suptitle('Recorded training positions; goal arrivals reset to the start, including after the first goal', fontsize=11)
    _save_figure(figure, plots_dir, figure_name + '_movement')
    plt.close(figure)

    # Arrows show legal action probabilities; cell color shows probability assigned to blocked moves.
    figure, axes = plt.subplots(1, len(selected), figsize=(4.5 * len(selected), 3.5), squeeze=False,
                                layout='constrained')
    for column, (row, values) in enumerate(zip(selected, data)):
        positions, grid = values['positions'], values['grid']
        probabilities = values['policy_probabilities'].mean(axis=0)
        stay = np.zeros(len(positions))
        arrows = []
        for state, (physical_row, physical_column) in enumerate(positions):
            for action, (row_offset, column_offset) in enumerate(((-1, 0), (0, 1), (1, 0), (0, -1))):
                next_row, next_column = physical_row + row_offset, physical_column + column_offset
                legal = 0 <= next_row < grid.shape[0] and 0 <= next_column < grid.shape[1] and grid[next_row, next_column] >= 0
                if legal:
                    arrows.append((physical_column, physical_row, column_offset * probabilities[state, action] * .8,
                                   row_offset * probabilities[state, action] * .8, state))
                else:
                    stay[state] += probabilities[state, action]
        _draw_map(axes[0, column], stay, grid, field_label(row['specification']), Normalize(0, 1))
        center = int(np.argmax(values['late_visit_counts'].mean(axis=0)))
        window = positions[max(0, center - 16):min(len(positions), center + 17)]
        for x, y, dx, dy, state in arrows:
            if abs(state - center) <= 16:
                axes[0, column].quiver(x, y, dx, dy, angles='xy', scale_units='xy', scale=1,
                                      color='#D55E00', width=.008)
        axes[0, column].set_xlim(window[:, 1].min() - .75, window[:, 1].max() + .75)
        axes[0, column].set_ylim(window[:, 0].max() + .75, window[:, 0].min() - .75)
    figure.suptitle('Policy arrows near the most visited late state; color is blocked-action probability', fontsize=11)
    _save_figure(figure, plots_dir, figure_name + '_policy')
    plt.close(figure)

    # First-goal curves use all seeds, including failures remaining undiscovered at the budget.
    figure, axis = plt.subplots(figsize=(7, 4), layout='constrained')
    for index, row in enumerate(selected):
        budget = row['specification']['total_timesteps']
        times = np.linspace(0, budget, 129)
        goals = np.asarray(row['first_goal_steps'])
        fraction = ((goals[:, None] >= 0) & (goals[:, None] <= times[None, :])).mean(axis=0)
        axis.plot(times, fraction, color=COLORS[index % len(COLORS)], linestyle=LINE_STYLES[index % len(LINE_STYLES)],
                  label=field_label(row['specification']))
    axis.set_xlabel('Training transitions per policy')
    axis.set_ylabel('Fraction of seeds reaching the goal')
    axis.set_ylim(0, 1.02)
    axis.legend(fontsize=8)
    _save_figure(figure, plots_dir, figure_name + '_discovery')
    plt.close(figure)
    _write_sampled_position_animation(selected, data, plots_dir, figure_name)
    return representatives


def _format_interval(value, interval, percentage=False, decimals=1):
    """Render an estimate and its 95-percent interval in a compact plain-text cell."""
    # Single independent groups have no estimable interval and are marked explicitly.
    scale = 100 if percentage else 1
    estimate = f'{value * scale:.{decimals}f}'
    if interval[0] is None:
        return estimate + ' (interval N/A)'
    return estimate + f' [{interval[0] * scale:.{decimals}f}, {interval[1] * scale:.{decimals}f}]'


def _latex_escape(value):
    """Escape ordinary report labels for text inside generated LaTeX tables."""
    # Before: names containing underscores or percent signs; after: literal printable text.
    replacements = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
                    '_': r'\_\allowbreak ', '{': r'\{', '}': r'\}'}
    return ''.join(replacements.get(character, character) for character in str(value))


def _write_results_tex(analysis_dir, display_rows, figure_groups, paired_comparisons):
    """Write generated result tables and selected behavior figures for the development document."""
    # Keep results separate across budgets and shared versus independently tuned comparisons.
    pieces = ['% Generated by scripts/grid_initialization_campaign/report.py; edit that generator.\n']
    if not display_rows:
        pieces.append('No completed training records are available yet. No failure case has been established.\n')
    groups = defaultdict(list)
    for row in display_rows:
        spec = row['specification']
        identity = (spec['wave'], geometry_label(spec), spec['policy_kind'],
                    spec['total_timesteps'], row['configuration'].get('comparison', 'development best'),
                    condition_variant(spec))
        groups[identity].append(row)
    latest = max((WAVE_ORDER.index(row['specification']['wave']) for row in display_rows), default=-1)
    for group_index, (identity, rows) in enumerate(groups.items()):
        wave, geometry, policy, budget, comparison, variant = identity
        if WAVE_ORDER.index(wave) != latest:
            continue
        rows = select_behavior_rows(rows)
        pieces.append('\n\\begin{table}[H]\n\\centering\n\\setlength{\\tabcolsep}{3pt}\n'
                      '\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{4.0cm}'
                      '>{\\raggedright\\arraybackslash}p{3.7cm}'
                      '>{\\raggedright\\arraybackslash}p{2.8cm}r@{}}\n\\toprule\n'
                      'Initialization & Mean steps, 95\\% interval $\\downarrow$ & Goal found, \\% & Seeds \\\\\n\\midrule\n')
        ranking = sorted(set(row['capped_mean_first_goal_steps'] for row in rows))
        for row in rows:
            value = _format_interval(row['capped_mean_first_goal_steps'], row['capped_mean_interval'])
            if row['capped_mean_first_goal_steps'] == ranking[0]:
                value = r'\textbf{' + value + '}'
            elif len(ranking) > 1 and row['capped_mean_first_goal_steps'] == ranking[1]:
                estimate, interval = value.split(' ', 1)
                value = r'\underline{' + estimate + '} ' + interval
            goal = _format_interval(row['goal_fraction'], row['goal_fraction_interval'], percentage=True)
            pieces.append(_latex_escape(field_label(row['specification'])) + ' & ' + value + ' & ' +
                          _latex_escape(goal) + f" & {row['seed_count']} \\\\\n")
        pieces.append('\\bottomrule\n\\end{tabular}\n\\caption{' + _latex_escape(
            f'{geometry}; {policy} policy; {wave.replace("-", " ")}; {comparison.replace("-", " ")}; {variant.replace("-", " ")}. '
            f'Every failed seed contributes the full {budget:,}-transition budget. '
            'Intervals resample network or random-field seeds as groups, retaining their policy seeds; '
            'deterministic fields resample policy seeds. Development rows select the lowest development mean; '
            'confirmation rows retain frozen settings. All development fields and settings remain in the run records.') + '}\n'
            + f'\\label{{tab:grid-initialization-results-{group_index}}}\n\\end{{table}}\n')
    if paired_comparisons:
        pieces.append('\n\\paragraph{Paired confirmation comparisons.} The difference is adverse initialization minus uniform initialization; '
                      'the ratio is their capped means. The same policy seeds are paired. '
                      'Two-sided 98.75\\% intervals adjust for four geometry/policy comparisons within each comparison rule and budget.\n')
        for index, comparison in enumerate(paired_comparisons):
            description = geometry_label({'geometry': comparison['geometry']}) + '; ' + comparison['policy_kind'] + ' policy; ' + comparison['comparison']
            pieces.append('\\noindent ' + _latex_escape(description) + ': difference ' +
                _latex_escape(_format_interval(comparison['mean_difference'], comparison['mean_difference_interval'])) +
                ' transitions; ratio ' + _latex_escape(_format_interval(comparison['mean_ratio'], comparison['mean_ratio_interval'], decimals=3)) +
                '; ' + _latex_escape(comparison['effect_label']) + '.\\par\n')
    for figure in figure_groups:
        for suffix, caption in (('maps', 'Initial bonus, cumulative visitation, and final-window visitation. '
                'Each row uses a common color scale across initializations. Dark cells are walls; white visitation cells have zero visits.'),
                ('initial_profiles', 'Initial bonuses against normalized path distance. Faint curves are independent field seeds; '
                'thick curves are mean initial bonuses. Each panel has its own vertical scale, logarithmic when all bonuses are positive. '
                'The dotted line is the uniform reference of one.'),
                ('movement', 'Sampled training positions across all seeds, followed by a representative seed selected by median capped first-goal time. '
                'Orange lines show its farthest discovered state. Dashed vertical markers use the exact first-goal transition, '
                'rather than a sampled position. All panels span zero to the goal. Goal arrivals reset to the start; '
                'post-goal samples therefore need not remain near the goal.'),
                ('policy', 'Final legal-action probabilities near the most visited late state. Cell color is the probability assigned to blocked actions.'),
                ('discovery', 'Fraction of independent policy copies that have reached the goal by each transition budget, including failed copies.')):
            filename = analysis_dir / 'plots' / (figure['name'] + '_' + suffix + '.pdf')
            pieces.append('\n\\begin{figure}[H]\n\\centering\n\\includegraphics[width=\\linewidth]{\\detokenize{' +
                          str(filename) + '}}\n\\caption{' + _latex_escape(caption) +
                          '}\n\\end{figure}\n')
    (analysis_dir / 'results.tex').write_text(''.join(pieces))


def generate_report(run_dir):
    """Regenerate durable metrics and figures without launching or modifying training."""
    # This generator performs numerical analysis and must run on a compute node.
    run_dir = Path(run_dir).resolve()
    analysis_dir = run_dir / 'analysis'
    plots_dir = analysis_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(run_dir)
    display_rows = select_reporting_rows(rows)
    paired_comparisons = paired_confirmation_comparisons(display_rows)
    state_path = run_dir/'campaign_state.json'
    state = read_json(state_path) if state_path.exists() else None
    focus_rows = select_focus_rows(display_rows, state)
    pieces = ['<!-- Generated by scripts/grid_initialization_campaign/report.py; edit that generator. -->\n',
              '# Initial bonus differences and exploration\n\n', f'Updated {pacific()}.\n\n',
              '## Metrics and comparison rules\n\n',
              'For policy seed $i$, let $T_i$ be the number of training transitions before its first goal, '
              '$B$ its transition budget, and $K$ the number of completed independent copies. A failure contributes $B$. '
              'The primary metric is the mean capped first-goal time, lower being better.\n\n',
              '$$ M = \\frac{1}{K} \\sum_{i=1}^{K} \\min(T_i,B). $$\n\n',
              'A missing first goal is treated as $T_i$ exceeding $B$. Goal-found percentage counts every seed. '
              'Coverage is the mean fraction of traversable states visited. Brackets give 95% bootstrap intervals. '
              'For random initial fields, entire field seeds are resampled with their policy seeds kept together. '
              'For deterministic fields, policy seeds are resampled. One independent group gives no interval.\n\n',
              'Development rows choose settings using development results. Confirmation comparisons retain settings '
              'chosen before their fresh seeds ran. Shared and independently tuned settings appear separately. '
              'These results concern the recorded finite budget; they do not establish permanent trapping.\n\n']
    pieces.append('## Numerical results\n\nEvery measured map remains in these summaries and in the seed records. '
                  'Large figures and the main document use only the maps already chosen by the campaign.\n\n')
    if not display_rows:
        pieces.append('No completed training records are available. No failure case has been established.\n')
    previous = None
    for row in display_rows:
        specification = row['specification']
        heading = (specification['wave'], geometry_label(specification), specification['policy_kind'],
                   specification['total_timesteps'], row['configuration'].get('comparison', 'development best'),
                   condition_variant(specification))
        if heading != previous:
            pieces.append('\n### ' + '; '.join(map(str, heading)) + '\n\n'
                          '| Initialization | Mean steps [95% interval] | Goal found, % | Coverage, % | Seeds |\n'
                          '|---|---:|---:|---:|---:|\n')
            previous = heading
        pieces.append(f"| {field_label(specification)} | " +
            _format_interval(row['capped_mean_first_goal_steps'], row['capped_mean_interval']) + ' | ' +
            _format_interval(row['goal_fraction'], row['goal_fraction_interval'], percentage=True) +
            f" | {100 * row['coverage_mean']:.1f} | {row['seed_count']} |\n")
    if paired_comparisons:
        pieces.append('\n## Paired confirmation comparisons\n\n'
            'Differences are adverse initialization minus uniform initialization. Ratios divide their capped means. '
            'The same policy seeds are paired. Two-sided 98.75% bootstrap intervals adjust for four '
            'geometry/policy comparisons within each comparison rule and transition budget. '
            'Network-field comparisons resample whole field seeds; adversarial fields resample policy seeds. '
            'An unfinished confirmation uses the paired seeds currently available and remains interim.\n\n')
        for comparison in paired_comparisons:
            pieces.append('### ' + geometry_label({'geometry': comparison['geometry']}) + '; ' + comparison['policy_kind'] +
                ' policy; ' + comparison['comparison'] + '; ' + comparison['wave'] + '\n\n' +
                comparison['field_label'] + '. ' + comparison['effect_label'] + '.\n\n' +
                '- Paired seeds: ' + str(comparison['paired_seed_count']) + '; independent groups: ' + str(comparison['independent_group_count']) + '.\n' +
                '- Mean difference in transitions [98.75% interval]: ' + _format_interval(comparison['mean_difference'], comparison['mean_difference_interval']) + '.\n' +
                '- Ratio of capped means [98.75% interval]: ' + _format_interval(comparison['mean_ratio'], comparison['mean_ratio_interval'], decimals=3) + '.\n' +
                f"- Goal found: uniform {100*comparison['uniform_goal_fraction']:.1f}%; adverse {100*comparison['candidate_goal_fraction']:.1f}%.\n\n")

    # Refresh behavior figures for the latest completed phase while preserving older phase files.
    figure_groups = []
    render_history_path = analysis_dir/'behavior_render_history.json'
    render_history = read_json(render_history_path) if render_history_path.exists() else {}
    if focus_rows:
        latest = max(WAVE_ORDER.index(row['specification']['wave']) for row in focus_rows)
        grouped = defaultdict(list)
        for row in focus_rows:
            spec = row['specification']
            if WAVE_ORDER.index(spec['wave']) == latest:
                key = json.dumps([spec['geometry'], spec['policy_kind'], spec['total_timesteps'],
                    row['configuration'].get('comparison', 'development best'), condition_variant(spec)], sort_keys=True)
                grouped[key].append(row)
        pieces.append('\n## Behavior\n\nMaps show means over every completed seed at the displayed setting. '
                      'Movement examples use the seed nearest the median capped first-goal time, '
                      'with field seed and policy seed as fixed tie breaks. Policy arrows show legal actions; '
                      'blocked actions are represented by the background probability. '
                      'The policy view covers up to 33 states around the most visited late state.\n\n')
        for index, comparison_rows in enumerate(grouped.values()):
            name = f'{WAVE_ORDER[latest]}_comparison-{index}'
            # Immutable completed summaries identify unchanged figures without repeatedly drawing them.
            signature = hashlib.sha256((json.dumps(comparison_rows, sort_keys=True)+
                hashlib.sha256(Path(__file__).read_bytes()).hexdigest()).encode()).hexdigest()
            required = [plots_dir/f'{name}_{suffix}.{extension}' for suffix in (
                'maps', 'initial_profiles', 'movement', 'policy', 'discovery') for extension in ('png', 'pdf')]
            required.append(plots_dir/f'{name}_sampled_positions.html')
            if render_history.get(name, {}).get('digest') == signature and all(path.exists() for path in required):
                representatives = render_history[name]['representatives']
            else:
                representatives = _plot_comparison(run_dir, comparison_rows, plots_dir, name)
                render_history[name] = {'digest': signature, 'representatives': representatives}
            figure_groups.append({'name': name, 'representatives': representatives})
            pieces.append(f'### {geometry_label(comparison_rows[0]["specification"])}; '
                          f'{comparison_rows[0]["specification"]["policy_kind"]} policy\n\n')
            for suffix in ('maps', 'initial_profiles', 'movement', 'policy', 'discovery'):
                pieces.append(f'![{suffix.capitalize()}](plots/{name}_{suffix}.png)\n\n')
            pieces.append('Initial-profile panels use separate vertical scales so neural-field variation remains visible. '
                          'Movement panels share the full zero-to-goal range; exact first-goal markers use saved transition counts. '
                          'Goal arrivals reset to the start, including after the first goal.\n\n')
            pieces.append(f'[Play the recorded training-position samples](plots/{name}_sampled_positions.html). '
                          'Intermediate actions are not shown.\n\n')
            for example in representatives:
                pieces.append(f"- {example['condition']}: policy seed {example['policy_seed']}, field seed {example['field_seed']}.\n")
            pieces.append('\n')

    # Keep every seed's metric available independently of selection and illustrative figures.
    data_dir = analysis_dir / 'data'
    data_dir.mkdir(exist_ok=True)
    with (data_dir / 'seed_metrics.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['wave', 'geometry', 'policy', 'field', 'configuration', 'budget',
                         'policy_seed', 'field_seed', 'first_goal_steps', 'capped_first_goal_steps', 'variant'])
        for row in rows:
            spec = row['specification']
            capped = capped_first_goal_steps(row['first_goal_steps'], spec['total_timesteps'])
            for index in range(row['seed_count']):
                writer.writerow([spec['wave'], geometry_label(spec), spec['policy_kind'], field_label(spec),
                    json.dumps(row['configuration'], sort_keys=True), spec['total_timesteps'],
                    row['policy_seeds'][index], row['field_seeds'][index], row['first_goal_steps'][index], capped[index], condition_variant(spec)])
    pieces.append('\n## Records and visualization sources\n\n'
                  '[Every seed metric](data/seed_metrics.csv) is retained, including unselected development settings. '
                  'Every task retains its complete seed arrays in `data/research/<task>/behavior.npz`.\n\n'
                  'The figure choices follow [visitation maps and action arrows](https://arxiv.org/pdf/2001.00119), '
                  '[trajectory and coverage comparisons](https://proceedings.neurips.cc/paper_files/paper/2023/file/9c7900fac04a701cbed83256b76dbaa3-Paper-Conference.pdf), '
                  'and [initial random-network bonus maps](https://proceedings.mlr.press/v235/yang24w.html).\n')
    _write_report_with_history(analysis_dir, ''.join(pieces))
    write_json(render_history_path, render_history)
    _write_results_tex(analysis_dir, focus_rows, figure_groups, paired_comparisons)
    write_json(analysis_dir / 'data/aggregate_results.json', {'generated_at': pacific(),
        'completed_setting_groups': len(rows), 'reported_groups': display_rows, 'figures': figure_groups,
        'paired_confirmation_comparisons': paired_comparisons})
    return {'completed_setting_groups': len(rows), 'reported_rows': len(display_rows),
            'figure_comparisons': len(figure_groups), 'report': str(analysis_dir / 'analysis.md')}


def main():
    """Generate one run's numerical report from the command line."""
    # No training command is called; numerical plotting still belongs on a compute node.
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(generate_report(arguments.run_dir), indent=2), flush=True)


if __name__ == '__main__':
    main()
