"""Queue the online tabular-RND auto research's units.

An idea file maps idea names (words) to {"settings": {GridPPOConfig rnd_* keys}, "network": {NetworkOptions keys},
"reward_normalization": true}. Every such unit is online tabular-RND on the fixed-settings study's maps and PPO. An
idea holding {"count": {...}} is a count-rule row instead: on the uniform field, or with
"field": "frozen_tabular_rnd" on the untrained tabular-RND error of each cell (an uneven start).

Stages:
- screen: one idea x one case x one policy learning rate x one entropy per unit, 7 weights, seeds 0-31, stopped at
  2^22 transitions on the full learning-rate schedule;
- development: the fixed-settings study's own protocol, all 52 configurations on seeds 0-15 at the full budget;
- confirmation: each idea's configuration chosen per case from its finished development units by the study's own
  rule, on the confirmation seeds in units of 128.
"""

import argparse
import json
from pathlib import Path

from common import enqueue, read_json, write_json
from fixed_settings_campaign import (BASELINE_CONFIRMATION_SEEDS, BUDGET, CONFIRMATION_SEEDS_PER_UNIT,
                                     DEVELOPMENT_SEEDS, ENTROPIES, EVALUATION_EPISODES, GEOMETRIES, LEARNING_RATES,
                                     POLICIES, WEIGHTS, choose)

HERE = Path(__file__).resolve().parent
STAGES = {'screen': {'learning_rates': (1e-3,), 'entropies': (1e-3, 1e-2),
                     'weights': tuple(2.0 ** exponent for exponent in range(-6, 7, 2)),
                     'seed_start': 0, 'seed_count': 32, 'stop_after_timesteps': 2 ** 22},
          'development': {'learning_rates': LEARNING_RATES, 'entropies': ENTROPIES, 'weights': WEIGHTS,
                          'seed_start': DEVELOPMENT_SEEDS[0], 'seed_count': DEVELOPMENT_SEEDS[1],
                          'stop_after_timesteps': None}}


def case_name(geometry, policy):
    """Name one of the four cases as the unit names spell it."""
    return f"{geometry['kind']}-width-{geometry['width']}-turns-{geometry['turns']}_policy-{policy}"


def checked_idea(name, idea):
    """Refuse an idea with unknown keys; return whether it is a count rule and whether its field is the frozen one."""
    unknown = set(idea) - {'settings', 'network', 'reward_normalization', 'count', 'field'}
    if unknown:
        raise ValueError(f'{name}: unknown idea keys {sorted(unknown)}')
    # A "count" idea is a count rule in a diagnostic form, not an RND variant.
    counted = 'count' in idea
    if counted and (set(idea['count']) - {'count_update', 'count_rate', 'count_decay_exponent', 'count_mixture'}
                    or 'settings' in idea):
        raise ValueError(f'{name}: a count idea holds only count_update, count_rate, count_decay_exponent and count_mixture')
    # Its field is uniform, or the frozen tabular-RND field: the count rule on an uneven start.
    field = idea.get('field', 'uniform')
    if field not in {'uniform', 'frozen_tabular_rnd'} or ('field' in idea and not counted):
        raise ValueError(f'{name}: "field" is uniform or frozen_tabular_rnd, and only a count idea has one')
    return counted, field == 'frozen_tabular_rnd'


def specification(stage, batch, name, idea, geometry, policy, settings, seed_start, seed_count, stop, label):
    """Build one unit's specification: an idea in one case, with its settings and seeds."""
    counted, frozen = checked_idea(name, idea)
    reward = idea.get('reward_normalization', True)
    seeds = f'{seed_start}-{seed_start + seed_count - 1}'
    kind = ('frozen-tabular-rnd-count' if frozen else 'uniform-count') if counted else 'online-tabular-rnd'
    return {'name': f'{stage}-{batch}_{case_name(geometry, policy)}_{name}{label}_seeds-{seeds}',
            'study': 'fixed-settings', 'wave': f'fixed-{stage}-{batch}', 'protocol': None, 'geometry': geometry,
            'policy_kind': policy, 'idea': name, 'batch': batch,
            'condition': f'{kind}_{name}_reward-normalization-{"on" if reward else "off"}',
            'method': ('frozen-tabular-rnd' if frozen else 'uniform') if counted else 'online-tabular-rnd',
            'field_kind': 'uniform' if counted and not frozen else 'random_network', 'severity_exponent': 0,
            'intrinsic_source': 'count' if counted else 'online_rnd', 'state_normalization': False,
            'normalize_intrinsic_rewards': reward, 'rnd_input': 'coordinates' if counted and not frozen else 'one_hot',
            'rnd_settings': idea.get('settings', {}), 'network_options': idea.get('network', {}),
            **idea.get('count', {}),
            'settings': settings, 'setting_offset': 0, 'total_timesteps': BUDGET, 'stop_after_timesteps': stop,
            'seed_start': seed_start, 'seed_count': seed_count, 'evaluation_episodes': EVALUATION_EPISODES,
            'variant': 'primary',
            **({} if counted and 'count_decay_exponent' in idea['count'] else {'count_decay_exponent': 0.5})}


def unit_specifications(batch, ideas, stage='screen', learning_rates=None):
    """Build a screen's or a development's units: ideas x cases x policy learning rates x entropies."""
    # Before: {"control": {}} at the screen stage; after: 4 cases x 2 entropies = 8 units of 7 settings x 32 seeds.
    plan = dict(STAGES[stage], **({'learning_rates': tuple(learning_rates)} if learning_rates else {}))
    rows = []
    for name, idea in ideas.items():
        for geometry in GEOMETRIES:
            for policy in POLICIES:
                for rate in plan['learning_rates']:
                    for entropy in plan['entropies']:
                        settings = [{'learning_rate': rate, 'entropy_coefficient': entropy,
                                     'intrinsic_coefficient': weight, 'gamma_int': 0.99, 'rollout_steps': 128}
                                    for weight in plan['weights']]
                        rows.append(specification(stage, batch, name, idea, geometry, policy, settings,
                            plan['seed_start'], plan['seed_count'], plan['stop_after_timesteps'],
                            f'_learning-rate-{rate:g}_entropy-{entropy:g}'))
    return rows


def development_settings(run_dir, batch):
    """Read every finished development setting of a batch, keyed by (idea, case)."""
    # Before: summary.json files of development units; after: {(idea, case): [setting rows with their scores]}.
    table = {}
    for path in (run_dir/'data/research').glob(f'development-{batch}_*/summary.json'):
        summary = read_json(path)
        if not summary.get('completed'):
            continue
        spec = summary['specification']
        key = (spec['idea'], case_name(spec['geometry'], spec['policy_kind']))
        table.setdefault(key, []).extend(summary['settings'])
    return table


def confirmation_specifications(run_dir, batch, ideas, seed_start, seed_count, finished_cases_only=False,
                                seeds_per_unit=CONFIRMATION_SEEDS_PER_UNIT):
    """Choose each idea's configuration per case from its development units and build its confirmation units."""
    finished = development_settings(run_dir, batch)
    expected = len(LEARNING_RATES) * len(ENTROPIES) * len(WEIGHTS)
    rows, frozen = [], []
    for name, idea in ideas.items():
        for geometry in GEOMETRIES:
            for policy in POLICIES:
                candidates = finished.get((name, case_name(geometry, policy)), [])
                if len(candidates) != expected:
                    message = (f'{name} {case_name(geometry, policy)}: {len(candidates)} of {expected} '
                               'development settings finished')
                    if not finished_cases_only:
                        raise RuntimeError(message)
                    # A case whose grid is still training is left for a later call; the finished cases go ahead.
                    print('skipped, ' + message)
                    continue
                # The study's own rule: smallest mean capped first-goal time, then the larger discovery rate.
                best = choose(candidates)
                frozen.append({'idea': name, 'case': case_name(geometry, policy), 'configuration': best['configuration'],
                               'development_capped_mean_first_goal_steps': best['capped_mean_first_goal_steps'],
                               'development_goal_fraction': best['goal_fraction']})
                # Seeds per unit follow the card's measured knee: about 256 copies for a trained predictor,
                # about 2,048 for the exact fit or a count rule (infrastructure/profiles/README.md).
                for start in range(seed_start, seed_start + seed_count, seeds_per_unit):
                    count = min(seeds_per_unit, seed_start + seed_count - start)
                    rows.append(specification('confirmation', batch, name, idea, geometry, policy,
                                              [best['configuration']], start, count, None, ''))
    return rows, frozen


def without_split_again(rows, queued):
    """Drop confirmation units of an idea and case that already has units under another split of the seeds."""
    # Before: new rows "..._seeds-20000-20255" while "..._seeds-20000-20127" is queued; after: that case's rows are gone.
    new_names = {row['name'] for row in rows}
    kept = []
    for row in rows:
        prefix = row['name'].rsplit('_seeds-', 1)[0] + '_seeds-'
        if any(name.startswith(prefix) and name not in new_names for name in queued):
            continue
        kept.append(row)
    skipped = sorted({row['name'].rsplit('_seeds-', 1)[0] for row in rows} - {row['name'].rsplit('_seeds-', 1)[0] for row in kept})
    for prefix in skipped:
        print(f'skipped, already queued under another split of the seeds: {prefix}')
    return kept


def record_frozen(path, frozen):
    """Add the chosen configurations to the batch's record, replacing an earlier entry of the same idea and case."""
    # Before: a file with idea A's four cases, and new rows for idea B; after: the file holds A's and B's rows.
    earlier = read_json(path) if path.exists() else []
    fresh = {(entry['idea'], entry['case']) for entry in frozen}
    write_json(path, [entry for entry in earlier if (entry['idea'], entry['case']) not in fresh] + frozen)


def main():
    """Write and queue one batch's units, interleaved so every idea gets cards early."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--batch', required=True)
    parser.add_argument('--ideas', type=Path, required=True)
    parser.add_argument('--stage', default='screen', choices=sorted(STAGES) + ['confirmation'])
    parser.add_argument('--learning-rates', nargs='*', type=float, default=None,
                        help='policy learning rates, when they differ from the stage\'s own')
    parser.add_argument('--only', nargs='*', default=None, help='queue only these ideas of the file')
    parser.add_argument('--confirmation-seeds', nargs=2, type=int, default=list(BASELINE_CONFIRMATION_SEEDS),
                        metavar=('START', 'COUNT'), help='confirmation stage: first seed and number of seeds')
    parser.add_argument('--seeds-per-unit', type=int, default=CONFIRMATION_SEEDS_PER_UNIT,
                        help='confirmation stage: policy seeds trained by one process')
    parser.add_argument('--finished-cases-only', action='store_true',
                        help='confirmation stage: confirm the cases whose development grid is complete, skip the others')
    args = parser.parse_args()
    ideas = {name: idea for name, idea in json.loads(args.ideas.read_text()).items()
             if not args.only or name in args.only}
    if args.stage == 'confirmation':
        rows, frozen = confirmation_specifications(args.run_dir, args.batch, ideas, *args.confirmation_seeds,
                                                   finished_cases_only=args.finished_cases_only,
                                                   seeds_per_unit=args.seeds_per_unit)
        record_frozen(args.run_dir/f'confirmation_frozen_{args.batch}.json', frozen)
    else:
        rows = unit_specifications(args.batch, ideas, args.stage, args.learning_rates)
    # Round-robin over ideas: the queue order is case-major, so each idea's first unit is near the front.
    # A unit staged on a remote server or held aside sits in a queue folder that enqueue() does not look at.
    elsewhere = {path.stem for folder in (args.run_dir/'queue').iterdir() if folder.is_dir()
                 for path in folder.glob('*.json')}
    if args.stage == 'confirmation':
        rows = without_split_again(rows, elsewhere)
    if not rows:
        print(f'0 units queued for batch {args.batch} ({args.stage}): nothing is ready')
        return
    ideas = list(dict.fromkeys(row['idea'] for row in rows))
    per_idea = {idea: [row for row in rows if row['idea'] == idea] for idea in ideas}
    ordered = [per_idea[idea][index] for index in range(max(map(len, per_idea.values()))) for idea in ideas
               if index < len(per_idea[idea])]
    created = 0
    for row in ordered:
        if row['name'] in elsewhere:
            continue
        location = args.run_dir/'specifications'/(row['name']+'.json')
        write_json(location, row)
        created += enqueue(args.run_dir, {'name': row['name'], 'stage': 'research',
            'limit_seconds': 4*86400, 'minimum_device_memory_mb': 7000,
            'intrinsic_source': row['intrinsic_source'],
            'command': [str(HERE/'train_unit.py'), '--run-dir', str(args.run_dir), '--specification', str(location)]})
    print(f'{created} of {len(ordered)} units queued for batch {args.batch} ({args.stage})')


if __name__ == '__main__':
    main()
