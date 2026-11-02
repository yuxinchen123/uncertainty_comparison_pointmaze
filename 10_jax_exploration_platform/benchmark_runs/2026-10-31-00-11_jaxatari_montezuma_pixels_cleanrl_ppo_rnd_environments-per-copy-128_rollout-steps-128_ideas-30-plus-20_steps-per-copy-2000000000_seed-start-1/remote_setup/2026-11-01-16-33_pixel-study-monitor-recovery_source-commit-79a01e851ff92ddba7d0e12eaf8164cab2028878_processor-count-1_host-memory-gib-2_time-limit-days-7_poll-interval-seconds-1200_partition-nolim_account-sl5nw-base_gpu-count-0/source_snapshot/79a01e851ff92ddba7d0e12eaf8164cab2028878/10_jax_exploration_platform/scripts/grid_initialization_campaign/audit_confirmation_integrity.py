"""Verify completed confirmation artifacts and save the actual integrity checks with provenance."""

import argparse
import collections
import hashlib
import importlib
import json
import os
from pathlib import Path
import socket
import sys
import time
import zipfile

from common import now, read_json, write_json


def require_compute_allocation():
    """Keep array audits on an allocated compute node."""
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith(('portal', 'login')):
        raise RuntimeError('Confirmation array audits require an allocated compute node.')


def audit(run_dir, waves, expected_source=None):
    """Check CRCs, exact visit histories, goal timing, frozen fields and complete seed groups."""
    require_compute_allocation()
    import numpy as np

    started = time.monotonic()
    source = read_json(run_dir/'source.json')
    package = Path(source['snapshot'])/'10_jax_exploration_platform'
    sys.path.insert(0, str(package/'scripts/grid_initialization_campaign'))
    sys.path.insert(0, str(package/'src'))
    campaign = importlib.import_module('campaign')
    archive_helper = importlib.import_module('initial_field_archive')
    make_geometry = importlib.import_module('exploration_platform.grid_initialization.environments').make_geometry
    records = [record for record in campaign.result_records(run_dir)
               if record['specification']['wave'] in waves]
    errors, artifacts, canonical, sources, groups, repeat_records = [], [], {}, collections.Counter(), {}, []
    counters = collections.Counter()

    def check(condition, name, path):
        """Keep failed checks visible without repairing an artifact."""
        counters[name] += 1
        if not condition:
            errors.append({'check': name, 'path': str(path)})

    for record in records:
        specification = record['specification']
        folder = run_dir/'data/research'/specification['name']
        path = folder/'behavior.npz'
        with zipfile.ZipFile(path) as compressed:
            check(compressed.testzip() is None, 'behavior_zip_crc', path)
        artifacts.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        with np.load(path, allow_pickle=False) as values:
            steps, counts = values['visit_count_history_steps'], values['visit_count_history']
            budget = specification['total_timesteps']
            expected = [1 << exponent for exponent in range(10, budget.bit_length())]
            if not expected or expected[-1] != budget:
                expected.append(budget)
            check(np.array_equal(steps, expected), 'exact_checkpoint_steps', path)
            check(counts.shape == (len(steps), len(values['seed_ids']), len(values['positions'])),
                  'visit_history_shape', path)
            check(bool(np.all(counts.sum(axis=2) == steps[:, None])), 'visit_count_sum_equals_step', path)
            check(bool(np.all(counts >= 0) and np.all(np.diff(counts, axis=0) >= 0)), 'visit_counts_monotonic', path)
            check(np.array_equal(counts[-1], values['visit_counts']), 'final_counts_match', path)
            first = values['first_goal_steps']
            expected_goal_seen = (first[None, :] >= 0) & (first[None, :] <= steps[:, None])
            check(np.array_equal(counts[:, :, -1] > 0, expected_goal_seen), 'goal_counts_iff_first_goal_by_checkpoint', path)
            check(bool(np.all((first == -1) | ((first >= 1) & (first <= budget)))), 'first_goal_range', path)
            check(set(values['seed_ids']).issubset(range(10000, 10128)), 'fresh_policy_seed_range', path)
            check(set(values['field_seed_ids']).issubset(range(2500, 2532)), 'fresh_field_seed_range', path)
            counters['completed_copies'] += len(first)
            counters['goal_checkpoint_copy_comparisons'] += expected_goal_seen.size
            counters['visitation_history_cells'] += counts.size
        task = read_json(run_dir/'queue/done'/(specification['name']+'.json'))
        submission = read_json(run_dir/'slurm/submissions'/(task['job_id']+'.json'))
        source_commit = submission['source_commit']
        sources[source_commit] += 1
        if expected_source:
            check(source_commit.startswith(expected_source), 'expected_data_source_commit', path)
        for shard_path in folder.glob('completed_copies_*.json'):
            shard = read_json(shard_path)
            duplicate = collections.defaultdict(dict)
            for copy in shard['copies']:
                duplicate[copy['policy_seed']][copy['setting_index']] = copy['first_goal_step']
            if specification['field_kind'] == 'uniform' and specification['policy_kind'] == 'tabular':
                pairs = [(seed, values) for seed, values in duplicate.items() if len(values) == 2]
                repeat_records.append({'task': specification['name'], 'job_id': task['job_id'],
                    'hardware': task['result']['hardware'], 'shard': str(shard_path),
                    'duplicate_seed_pairs': len(pairs),
                    'different_first_goal_times': sum(len(set(values.values())) > 1 for _, values in pairs)})
            for metadata in shard['field_metadata']:
                reference = metadata.get('canonical_archive')
                check(reference is not None, 'canonical_field_reference_present', shard_path)
                if reference is None:
                    continue
                field_path = Path(reference['path'])
                if field_path not in canonical:
                    with np.load(field_path, allow_pickle=False) as field_values:
                        manifest = json.loads(str(field_values['manifest_json'].item()))
                    identity = manifest['generation']
                    descriptor = identity['field_geometry']
                    base = make_geometry(descriptor['kind'], descriptor['width'], descriptor['turns'])
                    required = archive_helper.generation_identity(base, identity['field_kind'],
                        identity['severity_exponent'], identity['field_seed'])
                    check(identity == required, 'canonical_generation_source_and_inputs', field_path)
                    validated = archive_helper.load_archive(field_path, required, base.coordinates)
                    canonical[field_path] = validated.metadata['canonical_archive']
                check(reference['array_sha256'] == canonical[field_path]['array_sha256'],
                      'canonical_field_array_checksums', field_path)
        variant = specification.get('variant', 'primary')
        if variant == 'primary' or variant.startswith('rollout-'):
            for setting in record['settings']:
                identity = json.dumps([specification['wave'], specification['geometry'],
                    specification['policy_kind'], specification['field_kind'], specification['severity_exponent'],
                    setting['configuration']], sort_keys=True)
                group = groups.setdefault(identity, {'policies': [], 'fields': []})
                group['policies'].extend(setting['policy_seeds'])
                group['fields'].extend(setting['field_seeds'])
    complete = sum(len(group['policies']) == 128 and len(set(group['policies'])) == 128
                   and len(set(group['fields'])) == 32 for group in groups.values())
    return {'passed': not errors, 'at': now(), 'command': sys.argv,
        'execution': {'hostname': socket.gethostname(), 'slurm_job_id': os.environ['SLURM_JOB_ID'],
                      'slurm_step_id': os.environ.get('SLURM_STEP_ID')},
        'audit_script': {'path': str(Path(__file__).resolve()),
                         'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        'source': source, 'expected_data_source_commit': expected_source,
        'completed_data_source_counts': dict(sources), 'waves': waves,
        'completed_units': len(records), 'canonical_field_archives_checked': len(canonical),
        'main_groups_with_results': len(groups), 'main_groups_complete_128_policies_32_fields': complete,
        'check_counts': dict(counters), 'errors': errors, 'behavior_archive_checksums': artifacts,
        'tabular_duplicate_seed_shards': repeat_records,
        'wall_seconds': time.monotonic()-started}


def main():
    """Write one reproducible audit record inside the existing study's verification directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--waves', nargs='+', default=['confirmation'])
    parser.add_argument('--expected-source')
    args = parser.parse_args()
    result = audit(args.run_dir, args.waves, args.expected_source)
    write_json(args.output, result)
    print(json.dumps({name: result[name] for name in ('passed', 'completed_units',
        'canonical_field_archives_checked', 'main_groups_complete_128_policies_32_fields', 'check_counts', 'errors', 'wall_seconds')}), flush=True)
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
