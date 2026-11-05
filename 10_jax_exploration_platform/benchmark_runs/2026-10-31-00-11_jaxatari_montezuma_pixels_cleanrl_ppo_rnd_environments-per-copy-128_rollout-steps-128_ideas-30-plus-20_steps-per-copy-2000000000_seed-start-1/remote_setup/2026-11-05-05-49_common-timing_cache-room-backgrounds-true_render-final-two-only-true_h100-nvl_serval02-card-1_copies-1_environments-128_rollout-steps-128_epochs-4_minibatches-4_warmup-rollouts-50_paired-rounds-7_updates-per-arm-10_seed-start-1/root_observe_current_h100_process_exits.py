"""Observe only the two exact processes of the current completed H100 benchmark."""
import datetime
import json
import os
from pathlib import Path
import socket

expected_uid = 2618919
actual_host = socket.gethostname()
if os.getuid() != expected_uid or actual_host.split('.')[0] != 'serval02':
    raise RuntimeError('the exact authorized host and owner are required')
observations = []
for pid, expected_birth in [(386202, '84810007'), (386669, '84815419')]:
    directory = Path('/proc') / str(pid)
    row = {'pid': pid, 'expected_birth_counter': expected_birth}
    try:
        owner = directory.stat().st_uid
    except FileNotFoundError:
        row.update(proc_entry_absent=True, original_process_ended=True)
    else:
        row['proc_entry_absent'] = False
        if owner != expected_uid:
            row.update(original_process_ended=False, owner_matches=False,
                       observation='present process has another owner; no process files read')
        else:
            try:
                raw = (directory / 'stat').read_text()
            except FileNotFoundError:
                row.update(proc_entry_absent=True, original_process_ended=True)
            else:
                actual_birth = raw.rsplit(')', 1)[1].split()[19]
                row.update(actual_birth_counter=actual_birth,
                           original_process_ended=actual_birth != expected_birth, owner_matches=True)
    observations.append(row)
print(json.dumps({'kind': 'actual_current_h100_controller_and_child_exit_observation',
    'observed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'host': actual_host.split('.')[0], 'actual_hostname': actual_host, 'uid': os.getuid(),
    'all_original_processes_ended': all(row['original_process_ended'] is True for row in observations),
    'signals_sent': 0, 'gpu_or_standing_lock_operations': 0, 'global_process_enumeration': False,
    'observations': observations}))
