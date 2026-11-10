"""Queue the online tabular-RND auto research's units: one idea x one case x one entropy per unit.

An idea file maps idea names (words) to {"settings": {GridPPOConfig rnd_* keys}, "network": {NetworkOptions keys},
"reward_normalization": true}. Every unit is online tabular-RND on the fixed-settings study's maps and PPO, with the
stage's seeds, budget and early stop.
"""

import argparse
import json
from pathlib import Path

from common import enqueue, write_json
from fixed_settings_campaign import BUDGET, EVALUATION_EPISODES, GEOMETRIES, POLICIES

HERE = Path(__file__).resolve().parent
# Screen: learning rate 1e-3 x entropy {1e-3, 1e-2} x weights 2^-6, 2^-4, ..., 2^6, seeds 0-31, stopped at 2^22.
STAGES = {'screen': {'learning_rates': (1e-3,), 'entropies': (1e-3, 1e-2),
                     'weights': tuple(2.0 ** exponent for exponent in range(-6, 7, 2)),
                     'seed_start': 0, 'seed_count': 32, 'stop_after_timesteps': 2 ** 22}}


def unit_specifications(batch, ideas, stage='screen', learning_rates=None):
    """Build every unit specification of a batch: ideas x cases x policy learning rates x entropies."""
    # Before: {"control": {}}; after: 4 cases x 2 entropies = 8 specifications of 7 settings x 32 seeds.
    plan = dict(STAGES[stage], **({'learning_rates': tuple(learning_rates)} if learning_rates else {}))
    rows = []
    for name, idea in ideas.items():
        unknown = set(idea) - {'settings', 'network', 'reward_normalization', 'count'}
        if unknown:
            raise ValueError(f'{name}: unknown idea keys {sorted(unknown)}')
        # A "count" idea is the uniform count field in a diagnostic form, not an RND variant.
        counted = 'count' in idea
        if counted and (set(idea['count']) - {'count_update', 'count_rate', 'count_decay_exponent', 'count_mixture'}
                        or 'settings' in idea):
            raise ValueError(f'{name}: a count idea holds only count_update, count_rate, count_decay_exponent and count_mixture')
        for geometry in GEOMETRIES:
            for policy in POLICIES:
                for rate in plan['learning_rates']:
                    for entropy in plan['entropies']:
                        settings = [{'learning_rate': rate, 'entropy_coefficient': entropy,
                                     'intrinsic_coefficient': weight, 'gamma_int': 0.99, 'rollout_steps': 128}
                                    for weight in plan['weights']]
                        case = f"{geometry['kind']}-width-{geometry['width']}-turns-{geometry['turns']}_policy-{policy}"
                        seeds = f"{plan['seed_start']}-{plan['seed_start'] + plan['seed_count'] - 1}"
                        unit = f"{stage}-{batch}_{case}_{name}_learning-rate-{rate:g}_entropy-{entropy:g}_seeds-{seeds}"
                        reward = idea.get('reward_normalization', True)
                        rows.append({'name': unit, 'study': 'fixed-settings', 'wave': f'fixed-{stage}-{batch}',
                            'protocol': None, 'geometry': geometry, 'policy_kind': policy, 'idea': name, 'batch': batch,
                            'condition': (f'{"uniform-count" if counted else "online-tabular-rnd"}_{name}'
                                          f'_reward-normalization-{"on" if reward else "off"}'),
                            'method': 'uniform' if counted else 'online-tabular-rnd',
                            'field_kind': 'uniform' if counted else 'random_network', 'severity_exponent': 0,
                            'intrinsic_source': 'count' if counted else 'online_rnd', 'state_normalization': False,
                            'normalize_intrinsic_rewards': reward, 'rnd_input': 'coordinates' if counted else 'one_hot',
                            'rnd_settings': idea.get('settings', {}), 'network_options': idea.get('network', {}),
                            **idea.get('count', {}),
                            'settings': settings, 'setting_offset': 0, 'total_timesteps': BUDGET,
                            'stop_after_timesteps': plan['stop_after_timesteps'], 'seed_start': plan['seed_start'],
                            'seed_count': plan['seed_count'], 'evaluation_episodes': EVALUATION_EPISODES,
                            'variant': 'primary', **({} if counted and 'count_decay_exponent' in idea['count'] else
                                                     {'count_decay_exponent': 0.5})})
    return rows


def main():
    """Write and queue one batch's units, interleaved so every idea gets cards early."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--batch', required=True)
    parser.add_argument('--ideas', type=Path, required=True)
    parser.add_argument('--stage', default='screen', choices=sorted(STAGES))
    parser.add_argument('--learning-rates', nargs='*', type=float, default=None,
                        help='policy learning rates, when they differ from the stage\'s own')
    parser.add_argument('--only', nargs='*', default=None, help='queue only these ideas of the file')
    args = parser.parse_args()
    ideas = {name: idea for name, idea in json.loads(args.ideas.read_text()).items()
             if not args.only or name in args.only}
    rows = unit_specifications(args.batch, ideas, args.stage, args.learning_rates)
    # Round-robin over ideas: the queue order is case-major, so each idea's first unit is near the front.
    ideas = list(dict.fromkeys(row['idea'] for row in rows))
    per_idea = {idea: [row for row in rows if row['idea'] == idea] for idea in ideas}
    ordered = [per_idea[idea][index] for index in range(max(map(len, per_idea.values()))) for idea in ideas
               if index < len(per_idea[idea])]
    created = 0
    # A unit staged on a remote server or held aside sits in a queue folder that enqueue() does not look at.
    elsewhere = {path.stem for folder in (args.run_dir/'queue').iterdir() if folder.is_dir()
                 for path in folder.glob('*.json')}
    for specification in ordered:
        if specification['name'] in elsewhere:
            continue
        location = args.run_dir/'specifications'/(specification['name']+'.json')
        write_json(location, specification)
        created += enqueue(args.run_dir, {'name': specification['name'], 'stage': 'research',
            'limit_seconds': 4*86400, 'minimum_device_memory_mb': 7000,
            'intrinsic_source': specification['intrinsic_source'],
            'command': [str(HERE/'train_unit.py'), '--run-dir', str(args.run_dir), '--specification', str(location)]})
    print(f'{created} of {len(ordered)} units queued for batch {args.batch}')


if __name__ == '__main__':
    main()
