"""Queue and advance the fixed CleanRL PPO-RND settings study on the straight and snake grids.

Phases: checks -> measurements (normalization speed screen on an RTX A4500) -> development (every
configuration on seeds 0-15) -> confirmation (each condition's chosen configuration and the
default cell's best intrinsic weight on fresh seeds 20000-20127) -> complete.
"""

import argparse
import json
from pathlib import Path

from common import append_json, enqueue, now, read_json, write_json

HERE = Path(__file__).resolve().parent
BUDGET = 2 ** 24
DEVELOPMENT_SEEDS = (0, 16)
CONFIRMATION_SEEDS = (20000, 128)
EVALUATION_EPISODES = 32
SETTINGS_PER_UNIT = 16
GEOMETRIES = ({'kind': 'straight', 'width': 32, 'turns': 0}, {'kind': 'snake', 'width': 16, 'turns': 2})
POLICIES = ('tabular', 'neural')
LEARNING_RATES = (1e-4, 1e-3)
ENTROPIES = (1e-3, 1e-2)
WEIGHTS = tuple(2.0 ** exponent for exponent in range(-6, 7))
DEFAULT_CELL = (1e-4, 1e-3)


def conditions():
    """List the 13 conditions of a case: method, field, strength and the two normalizations."""
    # Uniform and alternating fields have no RND network, so state normalization does not apply.
    rows = [{'method': 'no-bonus', 'field_kind': 'none', 'severity_exponent': 0,
             'intrinsic_source': 'count', 'state_normalization': False, 'reward_normalization': False}]
    for reward in (False, True):
        rows.append({'method': 'uniform', 'field_kind': 'uniform', 'severity_exponent': 0,
                     'intrinsic_source': 'count', 'state_normalization': False, 'reward_normalization': reward})
    for reward in (False, True):
        rows.append({'method': 'alternating-high-low', 'field_kind': 'alternating_high_low',
                     'severity_exponent': 6, 'intrinsic_source': 'count', 'state_normalization': False,
                     'reward_normalization': reward})
    for method, source in (('frozen-network', 'count'), ('online-rnd', 'online_rnd')):
        for state in (False, True):
            for reward in (False, True):
                rows.append({'method': method, 'field_kind': 'random_network', 'severity_exponent': 0,
                             'intrinsic_source': source, 'state_normalization': state,
                             'reward_normalization': reward})
    return rows


def condition_name(condition):
    """Name a condition in words for files and reports."""
    return (f"{condition['method']}_state-normalization-{'on' if condition['state_normalization'] else 'off'}"
            f"_reward-normalization-{'on' if condition['reward_normalization'] else 'off'}")


def settings_for(condition):
    """Every learning rate x entropy x intrinsic weight; no bonus has no intrinsic weight to sweep."""
    weights = (1.0,) if condition['method'] == 'no-bonus' else WEIGHTS
    return [{'learning_rate': rate, 'entropy_coefficient': entropy, 'intrinsic_coefficient': weight,
             'gamma_int': 0.99, 'rollout_steps': 128}
            for rate in LEARNING_RATES for entropy in ENTROPIES for weight in weights]


def add_units(run_dir, wave, geometry, policy, condition, settings, seed_start, seed_count, protocol=None):
    """Queue one condition's settings in units of at most sixteen settings."""
    created = False
    case = f"{geometry['kind']}-width-{geometry['width']}-turns-{geometry['turns']}_policy-{policy}"
    for start in range(0, len(settings), SETTINGS_PER_UNIT):
        selected = settings[start:start + SETTINGS_PER_UNIT]
        name = (f"fixed-{wave}{'-'+protocol if protocol else ''}_{case}_{condition_name(condition)}"
                f"_settings-{start}-{start+len(selected)-1}_seeds-{seed_start}-{seed_start+seed_count-1}")
        specification = {'name': name, 'study': 'fixed-settings', 'wave': f'fixed-{wave}', 'protocol': protocol,
            'geometry': geometry, 'policy_kind': policy, 'condition': condition_name(condition),
            'method': condition['method'], 'field_kind': condition['field_kind'],
            'severity_exponent': condition['severity_exponent'],
            'intrinsic_source': condition['intrinsic_source'],
            'state_normalization': condition['state_normalization'],
            'normalize_intrinsic_rewards': condition['reward_normalization'],
            'settings': selected, 'setting_offset': start, 'total_timesteps': BUDGET,
            'seed_start': seed_start, 'seed_count': seed_count, 'evaluation_episodes': EVALUATION_EPISODES,
            'variant': 'primary'}
        location = run_dir/'specifications'/(name+'.json')
        write_json(location, specification)
        created = enqueue(run_dir, {'name': name, 'stage': 'research', 'limit_seconds': 4*86400,
            'minimum_device_memory_mb': 7000, 'intrinsic_source': condition['intrinsic_source'],
            'command': [str(HERE/'train_unit.py'), '--run-dir', str(run_dir), '--specification', str(location)]}) or created
    return created


def add_development(run_dir):
    """Queue every configuration of every case on the development seeds, conditions interleaved."""
    for condition in conditions():
        for geometry in GEOMETRIES:
            for policy in POLICIES:
                add_units(run_dir, 'development', geometry, policy, condition, settings_for(condition),
                          *DEVELOPMENT_SEEDS)


def development_rows(run_dir):
    """Read every finished development setting with its case and condition."""
    rows = []
    for path in (run_dir/'data/research').glob('fixed-development_*/summary.json'):
        summary = read_json(path)
        if not summary.get('completed'):
            continue
        spec = summary['specification']
        for setting in summary['settings']:
            rows.append({'geometry': spec['geometry'], 'policy_kind': spec['policy_kind'],
                         'condition': spec['condition'], **setting})
    return rows


def choose(rows, cell=None):
    """Smallest mean capped first-goal time, then the larger goal discovery rate."""
    if cell is not None:
        rows = [row for row in rows if (row['configuration']['learning_rate'],
                                        row['configuration']['entropy_coefficient']) == cell]
    return min(rows, key=lambda row: (row['capped_mean_first_goal_steps'], -row['goal_fraction']))


def add_confirmation(run_dir):
    """Freeze each condition's chosen configuration and the default cell's best weight, then queue them."""
    rows = development_rows(run_dir)
    frozen = []
    for condition in conditions():
        for geometry in GEOMETRIES:
            for policy in POLICIES:
                candidates = [row for row in rows if row['geometry'] == geometry and
                              row['policy_kind'] == policy and row['condition'] == condition_name(condition)]
                expected = len(settings_for(condition))
                if len(candidates) != expected:
                    raise RuntimeError(f'{condition_name(condition)} {geometry} {policy}: '
                                       f'{len(candidates)} of {expected} development settings finished')
                best = choose(candidates)
                for protocol, chosen in (('chosen', best), ('default-cell', choose(candidates, DEFAULT_CELL))):
                    # When the chosen configuration is in the default cell, its confirmation serves both.
                    duplicate = protocol == 'default-cell' and chosen['configuration'] == best['configuration']
                    frozen.append({'same_as_chosen': duplicate, 'geometry': geometry, 'policy_kind': policy, 'condition': condition_name(condition),
                                   'protocol': protocol, 'configuration': chosen['configuration'],
                                   'development_capped_mean_first_goal_steps': chosen['capped_mean_first_goal_steps'],
                                   'development_goal_fraction': chosen['goal_fraction']})
                    if not duplicate:
                        add_units(run_dir, 'confirmation', geometry, policy, condition, [chosen['configuration']],
                                  *CONFIRMATION_SEEDS, protocol=protocol)
    write_json(run_dir/'confirmation_frozen.json', frozen)


def stage_finished(run_dir, prefix):
    """True when no unit with this prefix is pending or running."""
    return not any(any((run_dir/'queue'/state).glob(prefix+'*.json')) for state in ('pending', 'running'))


def advance(run_dir):
    """Move to the next phase when the current one has finished."""
    run_dir = Path(run_dir)
    path = run_dir/'campaign_state.json'
    state = read_json(path) if path.exists() else {'phase': 'checks', 'created_at': now()}
    phase = state['phase']
    if phase == 'checks':
        enqueue(run_dir, {'name': 'checks_fixed-settings', 'stage': 'checks', 'limit_seconds': 3000,
                          'command': [str(HERE/'run_fixed_checks.py'), '--run-dir', str(run_dir)]})
        if (run_dir/'checks.json').exists() and read_json(run_dir/'checks.json').get('passed'):
            enqueue(run_dir, {'name': 'measurement_normalization-speed-screen', 'stage': 'infrastructure',
                              'limit_seconds': 3000, 'command': [str(HERE/'fixed_settings_benchmark.py'),
                              '--run-dir', str(run_dir), '--gpu-class', 'nvidia_rtx_a4500', '--screen']})
            state['phase'] = 'measurements'
        elif list((run_dir/'queue/failed').glob('checks_*.json')):
            state['blocked_reason'] = 'Checks failed; read the checks log before continuing.'
    elif phase == 'measurements' and stage_finished(run_dir, 'measurement_'):
        add_development(run_dir)
        state.update(phase='development', development_started_at=now())
    elif phase == 'development' and stage_finished(run_dir, 'fixed-development_'):
        add_confirmation(run_dir)
        state.update(phase='confirmation', confirmation_started_at=now())
    elif phase == 'confirmation' and stage_finished(run_dir, 'fixed-confirmation'):
        state.update(phase='complete', completed_at=now())
    state['updated_at'] = now()
    write_json(path, state)
    if phase != state['phase']:
        append_json(run_dir/'research_decisions.jsonl', {'at': now(), 'before': phase, 'after': state['phase']})
    if state['phase'] == 'complete':
        (run_dir/'SWEEP_COMPLETE').write_text(now()+'\n')
    return state


def main():
    """Advance the campaign once from its records."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(advance(args.run_dir), indent=2), flush=True)


if __name__ == '__main__':
    main()
