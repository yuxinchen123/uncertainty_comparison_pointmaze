"""Run tiny real units of the fixed-settings study, including a resume, on an allocated card."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from common import PYTHON, now, read_json, write_json
from fixed_settings_campaign import condition_set, conditions, condition_name

HERE = Path(__file__).resolve().parent


def run_unit(run_dir, spec):
    """Write one specification and run the production unit driver on it."""
    location = run_dir/'specifications'/(spec['name']+'.json')
    write_json(location, spec)
    subprocess.run([str(PYTHON), str(HERE/'train_unit.py'), '--run-dir', str(run_dir),
                    '--specification', str(location)], check=True)
    return run_dir/'data/research'/spec['name']


def main():
    """Every condition once at a short budget; one online unit stopped after a shard and resumed."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Unit checks require an allocated compute node.')
    results = []
    with tempfile.TemporaryDirectory() as folder:
        scratch = Path(folder)
        (scratch/'infrastructure').mkdir()
        # Two copies per process, so four copies make two shards.
        write_json(scratch/'infrastructure'/('batch_limit_'+os.environ.get('GRID_GPU_CLASS', 'default')+'.json'), {'copies': 2})
        settings = [{'learning_rate': 1e-3, 'entropy_coefficient': 1e-3, 'intrinsic_coefficient': 1.0,
                     'gamma_int': 0.99, 'rollout_steps': 128}, {'learning_rate': 1e-4, 'entropy_coefficient': 1e-2,
                     'intrinsic_coefficient': 4.0, 'gamma_int': 0.99, 'rollout_steps': 128}]
        listed = conditions(condition_set(args.run_dir))
        for condition in listed:
            for policy in ('tabular', 'neural'):
                spec = {'name': f'check_{policy}_{condition_name(condition)}', 'study': 'fixed-settings',
                        'wave': 'fixed-check', 'geometry': {'kind': 'snake', 'width': 4, 'turns': 2},
                        'policy_kind': policy, 'field_kind': condition['field_kind'],
                        'severity_exponent': condition['severity_exponent'],
                        'intrinsic_source': condition['intrinsic_source'],
                        'state_normalization': condition['state_normalization'],
                        'normalize_intrinsic_rewards': condition['reward_normalization'],
                        'count_decay_exponent': condition.get('count_decay_exponent', 0.5),
                        'rnd_input': condition.get('rnd_input', 'coordinates'),
                        'settings': settings, 'total_timesteps': 2048, 'seed_start': 0, 'seed_count': 2,
                        'evaluation_episodes': 4}
                folder_path = run_unit(scratch, spec)
                summary = read_json(folder_path/'summary.json')
                behavior = np.load(folder_path/'behavior.npz')
                assert len(summary['copy_records']) == 4
                assert behavior['visit_count_history'].shape == (2, 4, 14)
                assert behavior['evaluation_sampled_goal_steps'].shape == (4, 4)
                if condition['intrinsic_source'] == 'online_rnd':
                    assert behavior['rnd_error_history'].shape == (2, 4, 14)
                    assert np.all(behavior['rnd_error_initial'] > 0)
                results.append({'unit': spec['name'], 'passed': True,
                                'capped_means': [row['capped_mean_first_goal_steps'] for row in summary['settings']]})
        # Resume: remove the summary and the second shard, then rerun; two copies must be reused.
        last_online = [condition for condition in listed if condition['intrinsic_source'] == 'online_rnd'][-1]
        folder_path = scratch/'data/research'/f'check_neural_{condition_name(last_online)}'
        before = read_json(folder_path/'summary.json')['copy_records']
        (folder_path/'summary.json').unlink()
        shards = sorted(folder_path.glob('completed_copies_*.json'))
        for path in (shards[-1], folder_path/read_json(shards[-1])['behavior_file']):
            path.unlink()
        spec = read_json(scratch/'specifications'/(folder_path.name+'.json'))
        run_unit(scratch, spec)
        after = read_json(folder_path/'summary.json')
        assert after['resumed_copy_count'] == 2 and len(after['copy_records']) == 4
        same = sorted(before, key=lambda row: (row['setting_index'], row['policy_seed'])) == sorted(
            after['copy_records'], key=lambda row: (row['setting_index'], row['policy_seed']))
        results.append({'unit': 'resume', 'passed': True, 'rerun_identical_records': same})
    write_json(args.run_dir/'checks_fixed_units.json', {'at': now(), 'passed': True, 'results': results})
    print('fixed-settings unit checks passed', flush=True)


if __name__ == '__main__':
    main()
