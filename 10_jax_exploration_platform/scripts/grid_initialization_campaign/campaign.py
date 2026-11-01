"""Prepare and advance the bounded automatic initial-bonus research campaign."""

import argparse
from datetime import datetime, timedelta
import itertools
import json
from pathlib import Path
import subprocess

from common import PLATFORM, append_json, enqueue, now, queue_counts, read_json, write_json, queue_lock

HERE = Path(__file__).resolve().parent
ADVERSARIAL = ('high_start_cell', 'high_start_region', 'high_interior_pair',
               'smooth_interior_peak', 'decreasing_exponential', 'low_middle_interval',
               'repeated_low_intervals', 'repeated_smooth_peaks', 'alternating_high_low',
               'decreasing_steps')


def add_checks(run_dir):
    """Require numerical verification before admitting performance measurements."""
    return enqueue(run_dir, {'name': 'checks_exact-counts_cleanrl-updates_copy-isolation',
        'stage': 'checks', 'limit_seconds': 1100,
        'command': [str(HERE / 'run_checks.py'), '--run-dir', str(run_dir)]})


def add_measurement(run_dir, idea):
    """Store the precise candidate before queuing its bounded measurement."""
    spec = run_dir / 'infrastructure/specifications' / (idea['name'] + '.json')
    write_json(spec, idea)
    return enqueue(run_dir, {'name': 'measurement_' + idea['name'],
        'stage': 'infrastructure', 'limit_seconds': 590,
        'command': [str(HERE / 'benchmark.py'), '--run-dir', str(run_dir),
                    '--idea', idea['name'], '--idea-spec', str(spec),
                    '--policy-kind', 'all', '--geometry', 'all']})


def geometry_candidates():
    """Enumerate the prespecified development maps without assigning machines."""
    straight = [{'kind': 'straight', 'width': width, 'turns': 0}
                for width in (32, 64, 128, 256, 512, 1024)]
    snake = [{'kind': 'snake', 'width': width, 'turns': turns}
             for width in (16, 32, 64, 128) for turns in (2, 4, 8, 16)]
    return straight + snake


def research_task(run_dir, wave, geometry, policy, field, severity, settings,
                  budget=262144, seed_start=0, seed_count=16, variant="primary", **overrides):
    """Define a batch of fully independent settings and seeds for the shared queue."""
    name = (f'{wave}_{geometry["kind"]}-width-{geometry["width"]}-turns-{geometry["turns"]}'
            f'_policy-{policy}_bonus-{field}_severity-exponent-{severity}'
            f'_steps-{budget}_seeds-{seed_start}-{seed_start+seed_count-1}_variant-{variant}')
    specification = {'name': name, 'wave': wave, 'geometry': geometry, 'policy_kind': policy,
        'field_kind': field, 'severity_exponent': severity, 'settings': settings,
        'total_timesteps': budget, 'seed_start': seed_start, 'seed_count': seed_count, 'variant': variant, **overrides}
    location = run_dir / 'specifications' / (name + '.json')
    write_json(location, specification)
    return enqueue(run_dir, {'name': name, 'stage': 'research', 'limit_seconds': 10800,
        'minimum_device_memory_mb': 7000,
        'command': [str(HERE / 'train_unit.py'), '--run-dir', str(run_dir),
                    '--specification', str(location)]})


def coarse_settings():
    """Start tuning with a balanced twelve-setting power-of-two grid."""
    return [{'learning_rate': 2.0 ** learning, 'intrinsic_coefficient': 2.0 ** coefficient,
             'entropy_coefficient': 2.0 ** -10, 'rollout_steps': 128, 'gamma_int': 0.99}
            for learning in (-14, -10, -6) for coefficient in (-6, -2, 2, 6)]


def add_calibration(run_dir):
    """Measure baseline reachability before selecting demonstration geometries."""
    for geometry, policy, field in itertools.product(geometry_candidates(),
                                                    ('tabular', 'neural'), ('uniform', 'none')):
        research_task(run_dir, 'calibration', geometry, policy, field, 0, coarse_settings())


def result_records(run_dir, wave=None):
    """Read only complete result summaries for research decisions."""
    records = []
    for path in (run_dir / 'data/research').glob('*/summary.json'):
        record = read_json(path)
        if record.get('completed') and (wave is None or record['specification']['wave'] == wave):
            records.append(record)
    return records


def select_maps(records):
    """Prefer hard maps where uniform bonuses still discover the goal reliably."""
    selected = []
    for kind in ('straight', 'snake'):
        uniform = [record for record in records
                   if record['specification']['geometry']['kind'] == kind
                   and record['specification']['field_kind'] == 'uniform']
        # Retried budgets add records; retain each policy's best successful development result.
        scores = {}
        for record in uniform:
            geometry = record['specification']['geometry']
            identity = (geometry['width'], geometry['turns'])
            policy = record['specification']['policy_kind']
            best = best_setting(record)
            previous = scores.setdefault(identity, {}).get(policy)
            if previous is None or (-best['goal_fraction'], best['capped_mean_first_goal_steps']) < (
                    -previous['goal_fraction'], previous['capped_mean_first_goal_steps']):
                scores[identity][policy] = best
        eligible = [(identity, list(values.values())) for identity, values in scores.items()
                    if set(values) == {'tabular', 'neural'}
                    and min(value['goal_fraction'] for value in values.values()) >= .75]
        ordered = sorted(eligible,
                         key=lambda pair: sum(row['capped_mean_first_goal_steps'] for row in pair[1]),
                         reverse=True)
        for (width, turns), values in ordered[:2]:
            selected.append({'kind': kind, 'width': width, 'turns': turns})
    return selected


def add_field_search(run_dir, maps):
    """Test all ten adversarial hypotheses and independent network fields."""
    fields = [(field, severity) for field in ADVERSARIAL + ('random_log_uniform',)
              for severity in (2, 4, 6, 8)] + [('uniform', 0), ('random_network', 0)]
    for geometry, policy, (field, severity) in itertools.product(maps, ('tabular', 'neural'), fields):
        research_task(run_dir, 'field-search', geometry, policy, field, severity,
                      coarse_settings(), budget=1048576)


def best_setting(record):
    """Choose a successful setting before comparing its discovery time."""
    return min(record['settings'], key=lambda row: (-row['goal_fraction'], row['capped_mean_first_goal_steps']))


def choose_comparisons(records):
    """Require a successful matched uniform condition before choosing a slowdown."""
    chosen = []
    for kind, policy in itertools.product(('straight', 'snake'), ('tabular', 'neural')):
        candidates = []
        for record in records:
            spec = record['specification']
            relevant = spec['field_kind'] in ADVERSARIAL if kind == 'straight' else spec['field_kind'] == 'random_network'
            if spec['geometry']['kind'] != kind or spec['policy_kind'] != policy or not relevant:
                continue
            uniform = [item for item in records if item['specification']['geometry'] == spec['geometry']
                       and item['specification']['policy_kind'] == policy
                       and item['specification']['field_kind'] == 'uniform'
                       and item['specification']['total_timesteps'] == spec['total_timesteps']]
            if not uniform:
                continue
            control = min((best_setting(item) for item in uniform), key=lambda row: row['capped_mean_first_goal_steps'])
            if control['goal_fraction'] < .75:
                continue
            impaired = best_setting(record)
            ratio = impaired['capped_mean_first_goal_steps'] / max(control['capped_mean_first_goal_steps'], 1)
            candidates.append((ratio, record))
        if candidates:
            chosen.append(max(candidates, key=lambda item: item[0])[1]['specification'])
    return chosen


def sensitivity_settings():
    """Vary entropy and effective intrinsic horizons using powers of two."""
    return [dict(setting, entropy_coefficient=entropy, gamma_int=1-2.0**(-horizon))
            for setting in coarse_settings()[::3]
            for entropy in (0., 2.0**-14, 2.0**-10, 2.0**-6)
            for horizon in (6, 8, 10, 12)]


def add_diagnosis(run_dir, records, maps):
    """Use measured weak conditions to choose a bounded new development batch."""
    for geometry, policy in itertools.product(maps, ('tabular', 'neural')):
        relevant = [row for row in records if row['specification']['geometry'] == geometry
                    and row['specification']['policy_kind'] == policy
                    and row['specification']['field_kind'] in ADVERSARIAL]
        adversary = max(relevant, key=lambda row: best_setting(row)['capped_mean_first_goal_steps'])['specification'] if relevant else {'field_kind':'high_start_region', 'severity_exponent':8}
        for field, severity in [('uniform', 0), ('random_network', 0),
                                (adversary['field_kind'], adversary['severity_exponent'])]:
            research_task(run_dir, 'diagnosis', geometry, policy, field, severity,
                          sensitivity_settings(), budget=1048576)


def add_fine_tuning(run_dir, comparisons):
    """Give each main condition the same complete learning-rate/coefficient grid."""
    settings = [{'learning_rate': 2.0 ** learning, 'intrinsic_coefficient': 2.0 ** coefficient,
                 'entropy_coefficient': 2.0 ** -10, 'rollout_steps': 128, 'gamma_int': .99}
                for learning in range(-14, -5) for coefficient in range(-6, 7)]
    settings += sensitivity_settings()
    for spec in comparisons:
        adversaries = [record for record in result_records(run_dir, 'field-search')
                       if record['specification']['geometry'] == spec['geometry']
                       and record['specification']['policy_kind'] == spec['policy_kind']
                       and record['specification']['field_kind'] in ADVERSARIAL]
        adversary = max(adversaries, key=lambda record: best_setting(record)['capped_mean_first_goal_steps'])['specification'] if adversaries else spec
        for field, severity in [('uniform', 0), ('random_log_uniform', 4),
                                (adversary['field_kind'], adversary['severity_exponent']), ('random_network', 0)]:
            research_task(run_dir, 'fine-tuning', spec['geometry'], spec['policy_kind'], field,
                          severity, settings, budget=1048576)
            for rollout in (64, 256):
                structural = [dict(setting, rollout_steps=rollout) for setting in coarse_settings()]
                research_task(run_dir, 'fine-tuning', spec['geometry'], spec['policy_kind'], field,
                              severity, structural, budget=1048576, variant=f'rollout-{rollout}')


def add_confirmation(run_dir, comparisons, tuning, extended=False):
    """Freeze settings from development and use disjoint policy and field seeds."""
    frozen = []
    for spec in comparisons:
        matched = [record for record in tuning if
                   record['specification']['geometry'] == spec['geometry'] and
                   record['specification']['policy_kind'] == spec['policy_kind']]
        adversaries = [record for record in matched if record['specification']['field_kind'] in ADVERSARIAL]
        adversary = max(adversaries, key=lambda record: best_setting(record)['capped_mean_first_goal_steps'])['specification'] if adversaries else spec
        allowed = {('uniform', 0), ('random_log_uniform', 4), ('random_network', 0),
                   (adversary['field_kind'], adversary['severity_exponent'])}
        matched = [record for record in matched if
                   (record['specification']['field_kind'], record['specification']['severity_exponent']) in allowed
                   and record['specification'].get('variant', 'primary') in {'primary', 'rollout-64', 'rollout-256'}]
        uniform = [record for record in matched if record['specification']['field_kind'] == 'uniform']
        if not uniform or max(best_setting(record)['goal_fraction'] for record in uniform) < .75:
            continue
        shared = min((row for item in uniform for row in item['settings']), key=lambda row: (-row['goal_fraction'], row['capped_mean_first_goal_steps']))['configuration']
        seen = set()
        for record in matched:
            candidate = record['specification']
            identity = (candidate['field_kind'], candidate['severity_exponent'])
            if identity in seen:
                continue
            seen.add(identity)
            same_field = [row for item in matched if item['specification']['field_kind'] == identity[0] and item['specification']['severity_exponent'] == identity[1] for row in item['settings']]
            tuned = min(same_field, key=lambda row: (-row['goal_fraction'], row['capped_mean_first_goal_steps']))['configuration']
            configurations = [dict(shared, comparison='shared'), dict(tuned, comparison='independently-tuned')]
            groups = {}
            for configuration in configurations:
                groups.setdefault(configuration.get('rollout_steps', 128), []).append(configuration)
            for start in range(10000, 10128, 16):
                for rollout, group in groups.items():
                    research_task(run_dir, 'long-confirmation' if extended else 'confirmation',
                        candidate['geometry'], candidate['policy_kind'], candidate['field_kind'],
                        candidate['severity_exponent'], group,
                        budget=16777216 if extended else 4194304, seed_start=start,
                        variant=f'rollout-{rollout}')
            if not extended:
                for transform in ('reverse', 'shuffle', 'flatten'):
                    research_task(run_dir, 'confirmation', candidate['geometry'], candidate['policy_kind'],
                        candidate['field_kind'], candidate['severity_exponent'], [shared],
                        budget=4194304, seed_start=10000, seed_count=32,
                        variant=f'field-{transform}', field_transform=transform)
                research_task(run_dir, 'confirmation', candidate['geometry'], candidate['policy_kind'],
                    candidate['field_kind'], candidate['severity_exponent'], [shared],
                    budget=4194304, seed_start=10000, seed_count=32,
                    variant='cleanrl-normalization', normalize_intrinsic_rewards=True)
                if candidate['geometry']['kind'] == 'snake' and candidate['field_kind'] == 'random_network':
                    state_count = (candidate['geometry']['turns']+1)*candidate['geometry']['width']+candidate['geometry']['turns']
                    for geometry in (candidate['geometry'], {'kind':'straight', 'width':state_count, 'turns':0}):
                        research_task(run_dir, 'confirmation', geometry, candidate['policy_kind'],
                            'random_network', 0, [shared], budget=4194304, seed_start=10000,
                            seed_count=32, variant='shared-path-field', field_geometry={'kind':'straight', 'width':state_count, 'turns':0})
            frozen.append(dict(candidate, settings=configurations))
    write_json(run_dir / ('long_confirmation_frozen.json' if extended else 'confirmation_frozen.json'), frozen)


def stage_finished(run_dir, prefix):
    """Require every queued unit of a phase to have a terminal record."""
    active = list((run_dir / 'queue/pending').glob(prefix + '*.json'))
    active += list((run_dir / 'queue/running').glob(prefix + '*.json'))
    return not active


def advance(run_dir):
    """Advance known research phases without selecting from fresh confirmation results."""
    state_path = run_dir / 'campaign_state.json'
    state = read_json(state_path) if state_path.exists() else {'phase': 'checks', 'created_at': now()}
    phase = state['phase']
    if phase == 'checks':
        add_checks(run_dir)
        if list((run_dir / 'queue/done').glob('checks_*.json')):
            from ideas import initial_ideas
            for idea in initial_ideas():
                add_measurement(run_dir, idea)
            state['phase'] = 'initial-measurements'
    elif phase == 'initial-measurements' and stage_finished(run_dir, 'measurement_'):
        from ideas import choose_adaptive_ideas
        records = [read_json(path) for path in (run_dir / 'data/benchmarks').rglob('*.json')]
        if not records:
            state['blocked_reason'] = 'No valid performance records; inspect failed measurement logs.'
        else:
            ideas = choose_adaptive_ideas(records, count=10)
            write_json(run_dir / 'infrastructure/adaptive_ideas.json', ideas)
            for idea in ideas:
                add_measurement(run_dir, idea)
            state['phase'] = 'adaptive-measurements'
    elif phase == 'adaptive-measurements' and stage_finished(run_dir, 'measurement_'):
        state.update(phase='calibration', research_started_at=now(),
                     deadline=(datetime.now().astimezone()+timedelta(hours=96)).isoformat())
        add_calibration(run_dir)
    elif phase == 'calibration' and stage_finished(run_dir, 'calibration_'):
        maps = select_maps(result_records(run_dir, 'calibration'))
        if not maps:
            if not state.get('calibration_retry'):
                state['calibration_retry'] = True
                for geometry, policy in itertools.product(
                    [{'kind':'straight','width':width,'turns':0} for width in (16,32,64)] +
                    [{'kind':'snake','width':width,'turns':turns} for width in (8,16) for turns in (2,4)],
                    ('tabular','neural')):
                    research_task(run_dir, 'calibration', geometry, policy, 'uniform', 0,
                                  sensitivity_settings(), budget=1048576, variant='extended-baseline')
            else:
                state.update(phase='complete', outcome='No geometry met the prespecified uniform-success criterion.')
        else:
            state.update(phase='field-search', selected_maps=maps)
            add_field_search(run_dir, maps)
    elif phase == 'field-search' and stage_finished(run_dir, 'field-search_'):
        comparisons = choose_comparisons(result_records(run_dir, 'field-search'))
        state.update(phase='diagnosis', selected_comparisons=comparisons)
        add_diagnosis(run_dir, result_records(run_dir, 'field-search'), state['selected_maps'])
    elif phase == 'diagnosis' and stage_finished(run_dir, 'diagnosis_'):
        comparisons = choose_comparisons(result_records(run_dir, 'field-search') + result_records(run_dir, 'diagnosis'))
        if comparisons:
            state.update(phase='fine-tuning', selected_comparisons=comparisons)
            add_fine_tuning(run_dir, comparisons)
        else:
            state.update(phase='complete', outcome='Development produced no eligible matched comparison.')
    elif phase == 'fine-tuning' and stage_finished(run_dir, 'fine-tuning_'):
        add_confirmation(run_dir, state['selected_comparisons'], result_records(run_dir, 'fine-tuning'))
        state['phase'] = 'confirmation'
    elif phase == 'confirmation' and stage_finished(run_dir, 'confirmation_'):
        remaining = (datetime.fromisoformat(state['deadline'])-datetime.now().astimezone()).total_seconds()
        if remaining > 8*3600:
            add_confirmation(run_dir, state['selected_comparisons'], result_records(run_dir, 'fine-tuning'), True)
            state['phase'] = 'long-confirmation'
        else:
            state['phase'] = 'complete'
    elif phase == 'long-confirmation' and stage_finished(run_dir, 'long-confirmation_'):
        state['phase'] = 'complete'
    # Freeze the best available development records when the confirmation reserve starts.
    if state.get('deadline') and state['phase'] in ('calibration', 'field-search', 'diagnosis', 'fine-tuning'):
        remaining = (datetime.fromisoformat(state['deadline'])-datetime.now().astimezone()).total_seconds()
        if remaining < 24*3600:
            development = result_records(run_dir)
            comparisons = state.get('selected_comparisons') or choose_comparisons(development)
            if comparisons:
                add_confirmation(run_dir, comparisons, development)
                state.update(phase='confirmation', selected_comparisons=comparisons,
                             reserve_started_at=now())
                with queue_lock(run_dir):
                    destination = run_dir / 'queue/deferred'
                    destination.mkdir(exist_ok=True)
                    for path in (run_dir/'queue/pending').glob('*.json'):
                        task = read_json(path)
                        if task['stage'] == 'research' and not path.name.startswith('confirmation_'):
                            path.rename(destination/path.name)
            else:
                state.update(phase='complete', outcome='Confirmation reserve reached without an eligible matched comparison.')
    state['updated_at'] = now()
    write_json(state_path, state)
    if phase != state['phase']:
        append_json(run_dir / 'research_decisions.jsonl', {'at': now(), 'before': phase,
                    'after': state['phase'], 'reason': 'Completed recorded phase; next prespecified decision.'})
    if state['phase'] == 'complete':
        (run_dir / 'SWEEP_COMPLETE').write_text(now()+'\n')
    return state


def main():
    """Prepare or advance this campaign using lightweight record inspection only."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    print(json.dumps(advance(args.run_dir), indent=2), flush=True)


if __name__ == '__main__':
    main()
