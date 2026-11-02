"""Check that memory retries preserve scientific work and respect retry bounds."""

from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import read_json, write_json
from worker import retry_memory_failure


def test_memory_retry_keeps_seed_records_and_halves_only_physical_batch(tmp_path):
    """An out-of-memory retry must retain finished copies and keep its original settings."""
    for name in ('failed', 'pending', 'attempt_history'):
        (tmp_path/'queue'/name).mkdir(parents=True)
    task = {'name': 'neural_seed-0-15', 'stage': 'research', 'command': ['train_unit.py', '--same-settings']}
    write_json(tmp_path/'queue/failed'/(task['name']+'.json'), task)
    completed = tmp_path/'data/completed_seed_0.json'
    write_json(completed, {'policy_seed': 0, 'completed': True})
    with patch.dict('os.environ', {'GRID_GPU_CLASS': 'small_gpu'}):
        assert retry_memory_failure(tmp_path, task)
    assert read_json(tmp_path/'infrastructure/batch_limit_small_gpu.json')['copies'] == 16
    queued = read_json(tmp_path/'queue/pending'/(task['name']+'.json'))
    assert queued['command'] == task['command']
    assert queued['memory_retries'] == 1
    assert read_json(completed)['completed']


def test_measurements_and_exhausted_retries_are_not_requeued(tmp_path):
    """Preserve failed measurements and stop reducing already exhausted research batches."""
    assert not retry_memory_failure(tmp_path, {'stage': 'infrastructure'})
    assert not retry_memory_failure(tmp_path, {'stage': 'research', 'memory_retries': 4})
