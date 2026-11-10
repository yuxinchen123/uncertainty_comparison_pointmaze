"""Run this study's training units on remote servers: stage units, submit worker jobs, pull verified results.

Standard library only. It runs on the UVA login node and does no numerical work: file moves, ssh, rsync and JSON.

A unit's queue file sits in exactly one UVA queue folder at any time. While a server holds the unit, that folder is
`queue/remote_<server>/`. The unit reaches `queue/done/` only after its records were pulled to UVA and verified.
Nothing is deleted on a server. Jobs are cancelled only by exact id from the server's ledger and only while PENDING.

    python remote_fleet.py --run-dir RUN stage --server S --units N [--from-end] [--include-deferred] [--match TEXT]
    python remote_fleet.py --run-dir RUN submit --server S --jobs K [--pool P]
    python remote_fleet.py --run-dir RUN tick
    python remote_fleet.py --run-dir RUN status
    python remote_fleet.py --run-dir RUN cancel-pending --server S
    (stage, submit and tick take --dry-run: print what would happen and change nothing)
"""

import argparse
from contextlib import contextmanager
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time
from zoneinfo import ZoneInfo

from common import now, queue_lock, read_json, write_json

SCRIPTS = Path(__file__).resolve().parent
REMOTE_STATES = ('incoming', 'pending', 'running', 'done', 'failed', 'withdrawn')
REMOTE_FOLDERS = tuple(f'queue/{state}' for state in REMOTE_STATES) + (
    'specifications', 'infrastructure/packing', 'data/research', 'logs', 'slurm', 'first_unit_check')
PACKING_KEYS = ('tabular_online_rnd', 'neural_online_rnd', 'tabular_count', 'neural_count')
PACKING_COPIES = 512
PACKING_SOURCE = 'calculated from the serval05 H100 NVL profile of 2026-11-09 (no probe on this server, user rule)'
JOB_PREFIX = 'rndfleet'
MARK = '@@fleet:'
# Units still unclaimed this long after staging, with no job start on the server since, go back to UVA.
WAIT_LIMIT_SECONDS = 2 * 3600
# A unit whose job died is put back in the server's queue this many times; after that it goes back to UVA.
RECOVERY_LIMIT = 2
FAILED_CALL_LIMIT = 3
CARD_GUARD = Path('/p/rlprojects/.claude/skills/mit-submit-gpu-sweep/scripts/card_guard.sh')
CARD_GUARD_TORCH = "import torch; print(torch.cuda.device_count())"
CARD_GUARD_JAX = 'import jax; print(len(jax.devices("gpu")))'

WORKER_TEMPLATE = r'''#!/bin/bash
# Worker job of the RND grid study on @SERVER@, pool @POOL@. Written by remote_fleet.py: change that, not this copy.
# One card per job. It checks the card, compares one small processor-backend training with UVA's (once per server),
# then runs units from queue/pending one at a time until none is left or too little walltime remains.
# No speed test or benchmark is run.
RUN=@RUN@
PY=@PYTHON@
CODE=@CODE@
UNIT_SECONDS=@UNIT_SECONDS@
LIMIT_SECONDS=@LIMIT_SECONDS@
export GRID_GPU_CLASS=@GPU_CLASS@
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
# Module-provided CUDA libraries would be found before the environment's own (Gilbreth, DeltaAI).
unset LD_LIBRARY_PATH
JOB=${SLURM_JOB_ID:?this script runs only as a Slurm job}
cd "$RUN" || exit 2
@CARD_GUARD@
# Record the start, the node and the card.
echo "$JOB $(hostname) $(date -Is) restart=${SLURM_RESTART_COUNT:-0}" >> slurm/job_starts.txt
echo "job $JOB on $(hostname), start $(date -Is), class $GRID_GPU_CLASS"
CARD=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -n 1)
echo "card: $CARD MiB"

# A requeued job starts again under the same id: a unit claimed by its earlier start goes back to pending.
for claim in queue/running/*.job; do
  [ -e "$claim" ] || continue
  grep -q "^job=$JOB " "$claim" || continue
  unit=$(basename "$claim" .job)
  mv "queue/running/$unit.json" "queue/pending/$unit.json" && echo "released $unit, claimed by an earlier start of this job"
  mv "$claim" "logs/${unit}_${JOB}_released_$(date +%s).job"
done

# Nothing to do: give the card back before starting Python.
if ! ls queue/pending/*.json > /dev/null 2>&1; then echo "nothing pending; this job ends"; exit 0; fi

# The environment must use the card, or every unit taken would fail.
"$PY" -c 'import jax; print("jax devices:", jax.devices()); assert jax.default_backend() == "gpu"' \
  || { echo "jax does not use the card; this job ends without taking a unit"; exit 3; }

# First-unit check, once per server: the same small training as on UVA, on the processor backend, must give the
# same arrays (floats within a relative difference of 1e-4, integers equal). A failure leaves every unit unclaimed.
if ! grep -q '"passed": true' first_unit_check.json 2> /dev/null; then
  DUMP="first_unit_check/dump_$JOB.npz"
  JAX_PLATFORMS=cpu PYTHONPATH="$CODE/src" "$PY" "$CODE/scripts/grid_initialization_campaign/check_online_path_unchanged.py" --dump "$DUMP" \
    && "$PY" first_unit_check/remote_first_unit_compare.py first_unit_check/reference_uva.npz "$DUMP" \
         --output first_unit_check.json --server @SERVER@
  if ! grep -q '"passed": true' first_unit_check.json 2> /dev/null; then
    echo "the first-unit check did not pass; this job ends without taking a unit"; exit 4
  fi
fi

END=${SLURM_JOB_END_TIME:-$(( $(date +%s) + LIMIT_SECONDS ))}
STATUS=0
while :; do
  # Stop while a whole unit still fits, so no unit is cut off by the time limit.
  LEFT=$(( END - $(date +%s) ))
  if [ "$LEFT" -lt "$UNIT_SECONDS" ]; then echo "walltime left $LEFT s is below $UNIT_SECONDS s; this job ends"; break; fi
  # Claim the first pending unit: the move succeeds for exactly one job.
  unit=""
  for file in queue/pending/*.json; do
    [ -e "$file" ] || continue
    name=$(basename "$file" .json)
    if mv "$file" "queue/running/$name.json" 2> /dev/null; then unit=$name; break; fi
  done
  if [ -z "$unit" ]; then echo "nothing pending; this job ends"; break; fi
  printf 'job=%s host=%s gpu=%s started=%s\n' "$JOB" "$(hostname)" "$(echo "$CARD" | tr ' ' '_')" "$(date -Is)" > "queue/running/$unit.job"
  echo "$(date -Is) start $unit"
  BEGAN=$(date +%s)
  "$PY" "$CODE/scripts/grid_initialization_campaign/train_unit.py" --run-dir "$RUN" \
    --specification "$RUN/specifications/$unit.json" > "logs/${unit}_${JOB}.log" 2>&1
  CODE_OF_UNIT=$?
  printf 'exit=%s wall_seconds=%s ended=%s\n' "$CODE_OF_UNIT" "$(( $(date +%s) - BEGAN ))" "$(date -Is)" >> "queue/running/$unit.job"
  # The claim record moves first, so a unit seen in done/ or failed/ always has its record beside it.
  TO=failed; [ "$CODE_OF_UNIT" -eq 0 ] && TO=done
  mv "queue/running/$unit.job" "queue/$TO/$unit.job"
  mv "queue/running/$unit.json" "queue/$TO/$unit.json"
  echo "$(date -Is) $TO $unit (exit $CODE_OF_UNIT)"
  # One fault must not empty the queue into failed/: after a failed unit this job takes no more.
  if [ "$CODE_OF_UNIT" -ne 0 ]; then echo "a unit failed; this job ends"; STATUS=1; break; fi
done
echo "job $JOB ends $(date -Is)"
exit $STATUS
'''


class ServerUnreachable(Exception):
    """An ssh or rsync call did not reach the server, or its shared connection is closed."""


class RemoteCommandFailed(Exception):
    """A command reached the server and ended with an error."""


def pacific_now():
    """Return the current time for a log line, in Pacific time."""
    return datetime.now().astimezone(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d %H:%M:%S PT')


def epoch(timestamp):
    """Turn a timestamp that carries its offset into seconds since 1970."""
    # Before: "2026-11-09T22:36:50-06:00"; after: 1794285410.0.
    return datetime.fromisoformat(timestamp).timestamp()


def log(run_dir, text, dry_run=False):
    """Print one decision and, unless this is a dry run, append it to the fleet log."""
    line = f'{pacific_now()} | {"DRY RUN | " if dry_run else ""}{text}'
    print(line, flush=True)
    if not dry_run:
        with (run_dir / 'slurm/fleet_log.txt').open('a') as handle:
            handle.write(line + '\n')


@contextmanager
def fleet_lock(run_dir):
    """Let one remote_fleet command at a time change the fleet's records."""
    with (run_dir / 'slurm/remote_fleet.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def table_path(run_dir):
    """Name the server table."""
    return run_dir / 'infrastructure/remote_fleet_servers.json'


def assignment_path(run_dir):
    """Name the record of which server holds which unit."""
    return run_dir / 'slurm/remote_assignment.json'


def ledger_path(run_dir, name):
    """Name the UVA copy of a server's job-id ledger."""
    return run_dir / 'slurm' / f'submitted_jobids_{name}.txt'


def ledger_ids(run_dir, name):
    """Read every job id this run ever submitted on a server."""
    path = ledger_path(run_dir, name)
    return [value for value in path.read_text().split() if value.isdecimal()] if path.exists() else []


def unit_names(folder):
    """List the unit names of one queue folder in sorted order."""
    return sorted(path.stem for path in folder.glob('*.json'))


def load_assignment(run_dir):
    """Read the unit-to-server record, empty before the first stage."""
    path = assignment_path(run_dir)
    return read_json(path) if path.exists() else {}


def packing_record():
    """Give every policy and bonus kind the copies per process calculated for these cards."""
    # After: {"tabular_online_rnd": {"copies": 512}, ..., "source": "calculated from the serval05 ..."}.
    return {**{key: {'copies': PACKING_COPIES} for key in PACKING_KEYS}, 'source': PACKING_SOURCE}


def limit_seconds(time_limit):
    """Turn a Slurm time limit into seconds."""
    # Before: "12:00:00" or "2-00:00:00"; after: 43200 or 172800.
    days, _, clock = time_limit.rpartition('-')
    hours, minutes, seconds = (int(part) for part in clock.split(':'))
    return (int(days) if days else 0) * 86400 + hours * 3600 + minutes * 60 + seconds


def card_guard_text():
    """Return the shared MIT card guard, with its framework count taken from jax (this environment has no torch)."""
    text = CARD_GUARD.read_text()
    if text.count(CARD_GUARD_TORCH) != 1:
        raise RuntimeError(f'{CARD_GUARD} no longer holds the torch count line this tool replaces.')
    return ('NEED_CARDS=1\nRUN_DIR="$RUN"\nGUARD_ACTION=${GUARD_ACTION:-requeue}\n'
            + text.replace(CARD_GUARD_TORCH, CARD_GUARD_JAX))


def worker_script(name, entry, pool_name, pool):
    """Write the bash job script of one pool of one server."""
    values = {'@SERVER@': name, '@POOL@': pool_name, '@RUN@': shlex.quote(entry['remote_run']),
              '@PYTHON@': shlex.quote(entry['python']), '@CODE@': shlex.quote(entry['code']),
              '@UNIT_SECONDS@': str(int(entry['unit_seconds'])),
              '@LIMIT_SECONDS@': str(limit_seconds(pool['time_limit'])), '@GPU_CLASS@': pool['gpu_class'],
              '@CARD_GUARD@': card_guard_text() if pool.get('card_guard') else '# (no card guard on this server)'}
    text = WORKER_TEMPLATE
    for placeholder, value in values.items():
        text = text.replace(placeholder, value)
    return text


def choose_units(pending, deferred, count, from_end, match):
    """Pick the units to stage and say which UVA folder each one leaves."""
    # Before: pending ['a', 'c'], deferred ['b', 'd'], count 2, from_end True, match None.
    # After: [('c', 'pending'), ('d', 'deferred')] (the last two of the sorted names a, b, c, d).
    origin = {**{unit: 'deferred' for unit in deferred}, **{unit: 'pending' for unit in pending}}
    names = sorted(unit for unit in origin if match is None or match in unit)
    chosen = names[-count:] if from_end else names[:count]
    return [(unit, origin[unit]) for unit in chosen] if count > 0 else []


def fleet_jobs(name, status, ledger):
    """Select this run's live jobs on a server: ids from the ledger, and jobs carrying the fleet's name."""
    prefix = f'{JOB_PREFIX}-{name}-'
    return {job: row for job, row in status['jobs'].items() if job in ledger or row['name'].startswith(prefix)}


def pool_of(name, job_name):
    """Read the pool from a fleet job's name."""
    # Before: server "mit", job name "rndfleet-mit-preemptable-20261109-210501-1"; after: "preemptable".
    prefix = f'{JOB_PREFIX}-{name}-'
    return job_name[len(prefix):].rsplit('-', 3)[0] if job_name.startswith(prefix) else None


def allowed_jobs(requested, max_jobs, live_in_pool, pending_units, waiting_jobs):
    """Limit a submission to the pool's room and to the units no waiting job will take."""
    # Before: 8 requested, cap 12, 2 live in the pool, 5 units pending, 1 job still waiting; after: 4.
    return max(0, min(requested, max_jobs - live_in_pool, pending_units - waiting_jobs))


def status_script(name, entry, assigned):
    """Write the one bash script that reads a server's queue, jobs, budget and failed logs."""
    # Each marker is printed after an empty line, so it starts a line even when a file lacks its final newline.
    run = shlex.quote(entry['remote_run'])
    lines = [f'cd {run} 2> /dev/null || {{ echo; echo "{MARK}norun"; echo; echo "{MARK}end"; exit 0; }}',
             f'echo; echo "{MARK}queue"',
             f'for state in {" ".join(REMOTE_STATES)}; do for file in queue/$state/*.json; do '
             '[ -e "$file" ] && echo "$state $(basename "$file" .json)"; done; done',
             f'echo; echo "{MARK}claims"',
             'for state in running done failed; do for file in queue/$state/*.job; do '
             '[ -e "$file" ] && echo "$state $(basename "$file" .job) $(tr \'\\n\' \' \' < "$file")"; done; done',
             f'echo; echo "{MARK}jobs"',
             'squeue -h -u "$(id -un)" -o "%i|%T|%r|%j|%M|%P"; listed=$?',
             f'echo; echo "{MARK}squeue_exit $listed"',
             f'echo; echo "{MARK}starts"', 'tail -n 300 slurm/job_starts.txt 2> /dev/null',
             f'echo; echo "{MARK}ledger"', f'cat slurm/submitted_jobids_{name}.txt 2> /dev/null',
             f'echo; echo "{MARK}check"', 'cat first_unit_check.json 2> /dev/null',
             f'echo; echo "{MARK}budget"', entry.get('budget_command') or 'true']
    # Card counts of running jobs, where the scheduler is known to give a wrong number of cards (MIT).
    if any(pool.get('card_guard') for pool in entry['pools'].values()):
        lines += [f'echo; echo "{MARK}cards"',
                  'sacct -X -n -P -u "$(id -un)" -s RUNNING -o JobID,JobName,ReqTRES,AllocTRES 2> /dev/null']
    # The last log lines of failed units this run still counts as staged here.
    if assigned:
        lines += [f'for unit in {" ".join(shlex.quote(unit) for unit in assigned)}; do',
                  '  [ -e "queue/failed/$unit.json" ] || continue',
                  f'  echo; echo "{MARK}tail $unit"',
                  '  tail -n 40 "$(ls -t logs/${unit}_*.log 2> /dev/null | head -n 1)" 2> /dev/null',
                  'done']
    return '\n'.join(lines + [f'echo; echo "{MARK}end"', 'exit 0']) + '\n'


def parse_status(text):
    """Split one status call's output into the queue listing, claims, jobs, job starts, budget and log tails."""
    # Before: "@@fleet:queue\npending unit_a\n@@fleet:jobs\n12|RUNNING|None|rndfleet-aces-h100-x-1|5:02|gpu\n
    #          @@fleet:squeue_exit 0\n@@fleet:end"
    # After: {'queue': {'pending': ['unit_a'], ...}, 'jobs': {'12': {'state': 'RUNNING', ...}}, 'squeue_ok': True, ...}
    sections, current = {}, None
    for line in text.splitlines():
        if line.startswith(MARK):
            current = line[len(MARK):].strip()
            sections[current] = []
        elif current is not None and line.strip():
            sections[current].append(line)
    # A cut-off answer must never be read as an empty queue.
    if 'end' not in sections:
        raise RemoteCommandFailed('the status answer is incomplete (no end marker)')
    status = {'prepared': 'norun' not in sections, 'queue': {state: [] for state in REMOTE_STATES}, 'claims': {},
              'jobs': {}, 'squeue_ok': False, 'starts': [], 'ledger': [], 'check': None, 'budget': [], 'cards': [],
              'tails': {}}
    for line in sections.get('queue', []):
        state, unit = line.split(' ', 1)
        status['queue'][state].append(unit)
    # Before: "done unit_a job=12 host=n1 gpu=NVIDIA_H100,_81559 started=... exit=0 wall_seconds=4000 ended=..."
    # After: claims[('done', 'unit_a')] = {'job': '12', 'host': 'n1', ..., 'exit': '0', 'wall_seconds': '4000'}.
    for line in sections.get('claims', []):
        state, unit, *fields = line.split()
        status['claims'][(state, unit)] = dict(field.split('=', 1) for field in fields if '=' in field)
    for line in sections.get('jobs', []):
        parts = line.split('|')
        if len(parts) == 6:
            status['jobs'][parts[0]] = dict(zip(('state', 'reason', 'name', 'elapsed', 'partition'), parts[1:]))
    status['squeue_ok'] = any(header == 'squeue_exit 0' for header in sections)
    # Before: "2331041 ac053 2026-11-09T22:24:58-06:00 restart=0"; after: ('2331041', 1794284698.0).
    for line in sections.get('starts', []):
        parts = line.split()
        if len(parts) >= 3:
            status['starts'].append((parts[0], epoch(parts[2])))
    status['ledger'] = [value for line in sections.get('ledger', []) for value in line.split() if value.isdecimal()]
    if ''.join(sections.get('check', [])).strip():
        status['check'] = json.loads('\n'.join(sections['check']))
    status['budget'] = [line for line in sections.get('budget', []) if line.strip()]
    status['cards'] = [line for line in sections.get('cards', []) if line.strip()]
    status['tails'] = {header.split(' ', 1)[1]: lines for header, lines in sections.items()
                       if header.startswith('tail ')}
    return status


def budget_text(entry, lines):
    """Shorten a server's budget answer to the balance."""
    # Before (ACES): "|158590754315|  2027|  Y|   145000.00|   55667.82|    89332.18|Liu, Shuze Daniel   |", field 6,
    # unit "SUs"; after: "89332.18 SUs".
    if not entry.get('budget_command'):
        return entry.get('budget_note', 'not read')
    if not lines:
        return 'no answer'
    if 'budget_field' in entry:
        return f"{lines[0].split('|')[entry['budget_field']].strip()} {entry.get('budget_unit', '')}".strip()
    return ' '.join(' '.join(lines).split())


def card_mismatches(name, lines):
    """List running fleet jobs whose allocated card count differs from the requested one."""
    # Before: "28556900|rndfleet-mit-advanced-x-1|billing=4,cpu=4,gres/gpu=1,mem=16G|cpu=4,gres/gpu=2,mem=16G"
    # After: [('28556900', '1', '2')].
    wrong = []
    for line in lines:
        job, job_name, requested, allocated = line.split('|')
        if not job_name.startswith(f'{JOB_PREFIX}-{name}-'):
            continue
        counts = [dict(item.split('=', 1) for item in text.split(',') if '=' in item).get('gres/gpu')
                  for text in (requested, allocated)]
        if counts[0] != counts[1]:
            wrong.append((job, counts[0], counts[1]))
    return wrong


def plan_tick(name, assigned, status, assignment, ledger, moment):
    """Decide what one tick does with a server's units, from one status reading."""
    queue = {state: set(units) for state, units in status['queue'].items()}
    live = fleet_jobs(name, status, ledger)
    plan = {'place': sorted(set(assigned) & queue['incoming']), 'pull': sorted(set(assigned) & queue['done']),
            'failed': sorted(set(assigned) & queue['failed']), 'recover': [], 'pull_back': [], 'withdraw': [],
            'missing': sorted(set(assigned) - set().union(*queue.values())), 'reason': {}}
    # Without a job listing nothing can be called dead or not started.
    if not status['squeue_ok']:
        return plan
    check_failed = status['check'] is not None and not status['check'].get('passed')
    running_now = any(row['state'] == 'RUNNING' for row in live.values())
    for unit in sorted(set(assigned) & queue['running']):
        # A claim file written a moment after the move may not exist yet: look again at the next tick.
        claim = status['claims'].get(('running', unit))
        if claim is None or 'job' not in claim:
            continue
        # A unit is orphaned when its job is gone, or waits again after a preemption (same id, PENDING).
        job = status['jobs'].get(claim['job'])
        if job is None or job['state'] == 'PENDING':
            exhausted = assignment[unit].get('recoveries', 0) >= RECOVERY_LIMIT
            plan['pull_back' if exhausted else 'recover'].append(unit)
            plan['reason'][unit] = f"job {claim['job']} is {'waiting again' if job else 'no longer live'}"
    for unit in sorted(set(assigned) & queue['pending']):
        staged = epoch(assignment[unit]['staged_at'])
        started_since = any(start >= staged for _, start in status['starts'])
        # A server whose first-unit check failed takes no unit; one that started nothing in two hours is too slow.
        if check_failed:
            plan['withdraw'].append(unit)
            plan['reason'][unit] = 'the first-unit check failed on this server'
        elif moment - staged > WAIT_LIMIT_SECONDS and not running_now and not started_since:
            plan['withdraw'].append(unit)
            plan['reason'][unit] = f'unclaimed {(moment - staged) / 3600:.1f} h with no job start on this server'
    return plan


def move_script(entry, plan, stamp):
    """Write the bash script that carries out a tick's queue moves on the server and reports each success."""
    lines = [f'cd {shlex.quote(entry["remote_run"])} || exit 3']
    for action, source, target in (('place', 'incoming', 'pending'), ('recover', 'running', 'pending'),
                                   ('pull_back', 'running', 'withdrawn'), ('withdraw', 'pending', 'withdrawn')):
        for unit in plan[action]:
            quoted = shlex.quote(unit)
            # Only a move that succeeded counts: a worker job may have taken or finished the unit meanwhile.
            lines.append(f'if mv queue/{source}/{quoted}.json queue/{target}/{quoted}.json 2> /dev/null; then '
                         f'mv queue/{source}/{quoted}.job logs/{quoted}_{action}_{stamp}.job 2> /dev/null; '
                         f'[ -d data/research/{quoted} ] && data=yes || data=no; '
                         f'echo "{MARK}moved {action} {quoted} $data"; fi')
    return '\n'.join(lines + [f'echo "{MARK}end"', 'exit 0']) + '\n'


def parse_moves(text):
    """Read which queue moves succeeded on the server, and whether each unit has records there."""
    # Before: "@@fleet:moved withdraw unit_a no\n@@fleet:end"; after: [('withdraw', 'unit_a', False)].
    if f'{MARK}end' not in text:
        raise RemoteCommandFailed('the answer to the queue moves is incomplete (no end marker)')
    rows = [line[len(MARK):].split() for line in text.splitlines() if line.startswith(f'{MARK}moved ')]
    return [(action, unit, data == 'yes') for _, action, unit, data in rows]


def verify_unit(run_dir, unit):
    """Return None when a unit's pulled records are complete, otherwise the reason they are not."""
    folder = run_dir / 'data/research' / unit
    if not (folder / 'summary.json').exists():
        return 'summary.json is missing'
    summary = read_json(folder / 'summary.json')
    specification = read_json(run_dir / 'specifications' / f'{unit}.json')
    expected = len(specification['settings']) * specification['seed_count']
    if summary.get('completed') is not True:
        return 'summary.json does not say completed'
    if len(summary.get('copy_records', [])) != expected:
        return f"{len(summary.get('copy_records', []))} copy records, expected {expected}"
    if not (folder / 'behavior.npz').exists() or (folder / 'behavior.npz').stat().st_size == 0:
        return 'behavior.npz is missing or empty'
    return None


def result_block(name, claim, summary=None):
    """Build the result a finished or failed remote unit carries in its UVA queue file."""
    # Before: claim {'job': '12', 'host': 'n1', 'gpu': 'NVIDIA_H100,_81559', 'exit': '0', 'wall_seconds': '4000'}.
    # After: {'server': 'aces', 'job_id': '12', 'exit_code': 0, 'wall_seconds': 4000.0, 'pulled_at': '2026-...'}.
    block = {'server': name, 'job_id': claim.get('job'), 'hostname': claim.get('host'), 'gpu': claim.get('gpu'),
             'exit_code': int(claim['exit']) if 'exit' in claim else None, 'timed_out': False,
             'wall_seconds': float(claim['wall_seconds']) if 'wall_seconds' in claim else None,
             'pulled_at': now()}
    if summary is not None:
        block['training_wall_seconds'] = summary.get('wall_seconds')
    return block


def move_unit(run_dir, unit, source, target, extra=None):
    """Move one unit's queue file between two UVA queue folders, adding fields to it first when given."""
    with queue_lock(run_dir):
        path, destination = (run_dir / 'queue' / folder / f'{unit}.json' for folder in (source, target))
        if destination.exists():
            raise RuntimeError(f'{unit} is already in queue/{target}; a unit must be in one queue folder only.')
        # The file is rewritten in place and then renamed, so it is never in two folders.
        if extra:
            write_json(path, dict(read_json(path), **extra))
        destination.parent.mkdir(exist_ok=True)
        os.rename(path, destination)


def run_command(command, script, timeout):
    """Run one local command; return (exit code or None after a timeout, output, error output)."""
    try:
        result = subprocess.run(command, input=script, stdin=None if script is not None else subprocess.DEVNULL,
                                capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, '', f'no answer within {timeout} s'
    return result.returncode, result.stdout, result.stderr


def note_call(run_dir, table, name, succeeded, dry_run):
    """Count a server's failed calls in a row; at its limit mark it not ready and say so."""
    entry = table[name]
    before = entry.get('failed_calls_in_a_row', 0)
    entry['failed_calls_in_a_row'] = 0 if succeeded else before + 1
    limit = entry.get('failed_call_limit', FAILED_CALL_LIMIT)
    if not succeeded and entry['failed_calls_in_a_row'] >= limit and entry['ready']:
        entry['ready'] = False
        entry['not_ready_reason'] = f'{limit} failed call(s) in a row, the last at {pacific_now()}'
        log(run_dir, f'{name} | NOT READY from now on: {entry["not_ready_reason"]}', dry_run)
    if not dry_run and entry['failed_calls_in_a_row'] != before:
        write_json(table_path(run_dir), table)


def call(run_dir, table, name, command, script=None, timeout=180, dry_run=False):
    """Run one ssh or rsync command for a server and return its output."""
    entry = table[name]
    # A command sent after a shared connection closed would be a new login (ACES, MIT): check it first.
    if entry.get('check') and run_command(entry['check'], None, 60)[0] != 0:
        note_call(run_dir, table, name, False, dry_run)
        raise ServerUnreachable(f'{name}: the shared connection is closed ({" ".join(entry["check"])})')
    code, output, error = run_command(command, script, timeout)
    # ssh and rsync answer 255 when the connection itself failed.
    if code is None or code == 255:
        note_call(run_dir, table, name, False, dry_run)
        raise ServerUnreachable(f'{name}: {" ".join(command[:4])} ...: {error.strip()[-500:]}')
    note_call(run_dir, table, name, True, dry_run)
    if code:
        raise RemoteCommandFailed(f'{name}: {" ".join(command[:4])} ... ended with exit {code}: {error.strip()[-1500:]}')
    return output


def remote(run_dir, table, name, script, timeout=180, dry_run=False):
    """Run a bash script on a server, sent on standard input so that no quoting is needed."""
    return call(run_dir, table, name, table[name]['ssh'] + ['bash', '-s'], script, timeout, dry_run)


def rsync(run_dir, table, name, arguments, timeout=900):
    """Copy files to or from a server over the same ssh route."""
    shell = shlex.join(table[name]['ssh'][:-1])
    return call(run_dir, table, name, ['rsync', '-rt', '-e', shell] + arguments, None, timeout)


def remote_path(entry, path=''):
    """Spell a path inside a server's run folder the way rsync takes it."""
    # Before: "data/research/unit_a"; after: "aces:/scratch/user/u.sl359075/rnd_grid/runs/auto_research/data/...".
    return f"{entry['ssh'][-1]}:{entry['remote_run']}/{path}"


def require_ready(table, name):
    """Return a server's entry, refusing a server that is not ready."""
    if name not in table:
        raise SystemExit(f'{name} is not in the server table ({", ".join(table)}).')
    if not table[name]['ready']:
        raise SystemExit(f'{name} is marked not ready: {table[name].get("not_ready_reason", "no reason recorded")}')
    return table[name]


def prepare(run_dir, table, name):
    """Create a server's run folders and send its job scripts, packing files and first-unit check files."""
    entry = table[name]
    reference = run_dir / 'infrastructure/remote_fleet/first_unit_reference_uva.npz'
    if not reference.exists():
        raise RuntimeError(f'The first-unit reference {reference} does not exist; produce it on a UVA compute node.')
    # Build the server's files locally, laid out as in its run folder, so that one copy sends them all.
    tree = run_dir / 'infrastructure/remote_fleet' / name
    for folder in REMOTE_FOLDERS:
        (tree / folder).mkdir(parents=True, exist_ok=True)
    for pool_name, pool in entry['pools'].items():
        (tree / f'worker_job_{pool_name}.sh').write_text(worker_script(name, entry, pool_name, pool))
        write_json(tree / 'infrastructure/packing' / f"{pool['gpu_class']}.json", packing_record())
    shutil.copy2(SCRIPTS / 'remote_first_unit_compare.py', tree / 'first_unit_check/remote_first_unit_compare.py')
    # The reference is large: a hard link keeps one copy on UVA and an unchanged time, so it is sent only once.
    linked = tree / 'first_unit_check/reference_uva.npz'
    if not (linked.exists() and linked.samefile(reference)):
        linked.unlink(missing_ok=True)
        os.link(reference, linked)
    remote(run_dir, table, name, f'mkdir -p {shlex.quote(entry["remote_run"])}\n')
    rsync(run_dir, table, name, [f'{tree}/', remote_path(entry)])


def read_status(run_dir, table, name, dry_run=False):
    """Make the one status call of a server and return (assigned units, parsed answer)."""
    assigned = unit_names(run_dir / 'queue' / f'remote_{name}')
    return assigned, parse_status(remote(run_dir, table, name, status_script(name, table[name], assigned),
                                         dry_run=dry_run))


def stage(run_dir, table, args):
    """Move units from the UVA queue to a server's queue."""
    name, entry = args.server, require_ready(table, args.server)
    held = run_dir / 'queue' / f'remote_{name}'
    if args.dry_run:
        chosen = choose_units(unit_names(run_dir / 'queue/pending'),
                              unit_names(run_dir / 'queue/deferred') if args.include_deferred else [],
                              args.units, args.from_end, args.match)
        log(run_dir, f'{name} | stage | would move {len(chosen)} unit(s) to queue/remote_{name}/ and to the server', True)
        for unit, origin in chosen:
            print(f'    {unit}  (from queue/{origin})')
        return
    # The server must be reachable and set up before any UVA file moves.
    prepare(run_dir, table, name)
    with queue_lock(run_dir):
        chosen = choose_units(unit_names(run_dir / 'queue/pending'),
                              unit_names(run_dir / 'queue/deferred') if args.include_deferred else [],
                              args.units, args.from_end, args.match)
        held.mkdir(exist_ok=True)
        for unit, origin in chosen:
            os.rename(run_dir / 'queue' / origin / f'{unit}.json', held / f'{unit}.json')
        assignment = load_assignment(run_dir)
        assignment.update({unit: {'server': name, 'staged_at': now(), 'origin': origin, 'recoveries': 0,
                                  'state': 'staged'} for unit, origin in chosen})
        write_json(assignment_path(run_dir), assignment)
    if not chosen:
        log(run_dir, f'{name} | stage | no unit matched; nothing moved')
        return
    units = [unit for unit, _ in chosen]
    try:
        # Specifications and any records UVA already holds go first; queue files go to incoming/, out of the
        # workers' sight, so a copy that fails half-way leaves nothing a worker could take.
        rsync(run_dir, table, name, [str(run_dir / 'specifications' / f'{unit}.json') for unit in units]
              + [remote_path(entry, 'specifications/')])
        records = [str(run_dir / 'data/research' / unit) for unit in units if (run_dir / 'data/research' / unit).is_dir()]
        if records:
            rsync(run_dir, table, name, records + [remote_path(entry, 'data/research/')])
        rsync(run_dir, table, name, [str(held / f'{unit}.json') for unit in units]
              + [remote_path(entry, 'queue/incoming/')])
    except (ServerUnreachable, RemoteCommandFailed) as error:
        # Nothing reached the server's pending folder: the units go back where they came from.
        with queue_lock(run_dir):
            for unit, origin in chosen:
                os.rename(held / f'{unit}.json', run_dir / 'queue' / origin / f'{unit}.json')
            assignment = load_assignment(run_dir)
            for unit in units:
                del assignment[unit]
            write_json(assignment_path(run_dir), assignment)
        log(run_dir, f'{name} | stage | FAILED to copy, {len(units)} unit(s) moved back on UVA: {error}')
        raise
    # One call makes the units visible to the workers. If it fails, the units stay assigned here and the next
    # tick finishes the move from incoming/ (or reports them missing).
    moved = parse_moves(remote(run_dir, table, name, move_script(entry, {
        'place': units, 'recover': [], 'pull_back': [], 'withdraw': []}, int(time.time()))))
    log(run_dir, f'{name} | stage | {len(units)} unit(s) moved to queue/remote_{name}/, {len(moved)} now pending on '
        f'the server (from the end: {args.from_end}, deferred included: {args.include_deferred}, match: {args.match})')
    for unit, origin in chosen:
        log(run_dir, f'{name} | stage | {unit} (from queue/{origin})')


def choose_pool(entry, requested):
    """Return the pool a submission goes to; a server with several pools needs it named."""
    if requested is None and len(entry['pools']) > 1:
        raise SystemExit(f'This server has several pools; name one with --pool ({", ".join(entry["pools"])}).')
    pool_name = requested or next(iter(entry['pools']))
    if pool_name not in entry['pools']:
        raise SystemExit(f'Unknown pool {pool_name}; this server has {", ".join(entry["pools"])}.')
    return pool_name, entry['pools'][pool_name]


def submit(run_dir, table, args):
    """Submit worker jobs to one pool, within its cap and the units still unclaimed."""
    name, entry = args.server, require_ready(table, args.server)
    pool_name, pool = choose_pool(entry, args.pool)
    _, status = read_status(run_dir, table, name, args.dry_run)
    if not status['prepared'] or not status['squeue_ok']:
        raise SystemExit(f'{name}: the run folder is not set up or the job listing failed; nothing submitted.')
    # Room is the pool's cap minus its live jobs; work is the pending units no waiting job will take.
    live = fleet_jobs(name, status, set(ledger_ids(run_dir, name)))
    live_in_pool = sum(pool_of(name, row['name']) == pool_name for row in live.values())
    waiting = sum(row['state'] != 'RUNNING' for row in live.values())
    pending = len(status['queue']['pending']) + len(status['queue']['incoming'])
    count = allowed_jobs(args.jobs, pool['max_jobs'], live_in_pool, pending, waiting)
    log(run_dir, f'{name} | submit | pool {pool_name}: {args.jobs} asked, {count} allowed (cap {pool["max_jobs"]}, '
        f'{live_in_pool} live in the pool, {pending} unit(s) pending, {waiting} job(s) of this run still waiting)',
        args.dry_run)
    # Every option stands before the script: sbatch hands anything after the script to the script itself.
    command = (f'sbatch --parsable -o {shlex.quote(entry["remote_run"] + "/logs/job_%j.out")} -t {pool["time_limit"]} '
               f'{shlex.join(pool["sbatch_options"])}')
    script = shlex.quote(entry['remote_run'] + f'/worker_job_{pool_name}.sh')
    if args.dry_run or count == 0:
        print(f'    command: {command} -J {JOB_PREFIX}-{name}-{pool_name}-<time>-<n> {script}')
        return
    prepare(run_dir, table, name)
    # Each job gets its own name, and its id goes to the server's ledger in the same command that created it.
    stamp = datetime.now().astimezone(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%d-%H%M%S')
    lines = [f'cd {shlex.quote(entry["remote_run"])} || exit 3']
    for index in range(1, count + 1):
        job_name = f'{JOB_PREFIX}-{name}-{pool_name}-{stamp}-{index}'
        # The id is the last line sbatch prints ("123" or "123;cluster"); anything else is the scheduler's refusal.
        lines += [f'out=$({command} -J {job_name} {script} 2>&1); code=$?',
                  'id=$(echo "$out" | tail -n 1); id=${id%%;*}',
                  f'if [ "$code" -eq 0 ] && [ -n "$id" ] && [ -z "${{id//[0-9]/}}" ]; then '
                  f'echo "$id" >> slurm/submitted_jobids_{name}.txt; echo "{MARK}submitted {job_name} $id"; '
                  f'else echo "{MARK}refused {job_name} $(echo "$out" | tr "\\n" " ")"; fi', 'sleep 1']
    answer = remote(run_dir, table, name, '\n'.join(lines + [f'echo "{MARK}end"', 'exit 0']) + '\n', timeout=300)
    for line in answer.splitlines():
        if line.startswith(f'{MARK}submitted '):
            _, job_name, job = line.split()
            with ledger_path(run_dir, name).open('a') as handle:
                handle.write(job + '\n')
            log(run_dir, f'{name} | submit | job {job} ({job_name}) submitted')
        elif line.startswith(f'{MARK}refused '):
            log(run_dir, f'{name} | submit | REFUSED by the scheduler: {line[len(MARK) + 8:]}')
    if f'{MARK}end' not in answer:
        log(run_dir, f'{name} | submit | the answer was cut off; the next tick reads the server ledger for lost ids')


def record_unlisted_jobs(run_dir, name, status):
    """Add to the UVA ledger any job of this run that only the server knows (a submit answer lost on the way)."""
    known = set(ledger_ids(run_dir, name))
    named = [job for job, row in status['jobs'].items() if row['name'].startswith(f'{JOB_PREFIX}-{name}-')]
    for job in sorted((set(status['ledger']) | set(named)) - known):
        with ledger_path(run_dir, name).open('a') as handle:
            handle.write(job + '\n')
        log(run_dir, f'{name} | tick | job {job} was missing from the UVA ledger and is now recorded')


def finish_assignment(run_dir, units, state):
    """Record how the stay of some units on a server ended."""
    assignment = load_assignment(run_dir)
    for unit in units:
        assignment[unit].update(state=state, ended_at=now())
    write_json(assignment_path(run_dir), assignment)


def bring_back(run_dir, table, name, returned, command, folder='pending'):
    """Put units the server gave up back in a UVA queue folder (pending, or deferred to hold them), with their records."""
    # A unit going back to UVA brings the records it has, so finished copies are not trained again.
    entry = table[name]
    with_records = [unit for unit, data in returned if data]
    if with_records:
        rsync(run_dir, table, name, [remote_path(entry, f'data/research/{with_records[0]}')]
              + [f':{entry["remote_run"]}/data/research/{unit}' for unit in with_records[1:]]
              + [str(run_dir / 'data/research') + '/'])
    for unit, _ in returned:
        move_unit(run_dir, unit, f'remote_{name}', folder)
    finish_assignment(run_dir, [unit for unit, _ in returned], 'returned')
    for unit, _ in returned:
        log(run_dir, f'{name} | {command} | {unit} is back in queue/{folder} on UVA')


def take_back(run_dir, table, args):
    """Take units still waiting on a server back to UVA's pending folder, so they can be staged elsewhere."""
    name = args.server
    assigned, status = read_status(run_dir, table, name, False)
    if not status['prepared']:
        raise SystemExit(f'{name} is not prepared; nothing to take back.')
    # Only units no job has claimed: a unit a worker took meanwhile fails its move on the server and stays there.
    waiting = sorted(unit for unit in set(assigned) & set(status['queue']['pending'])
                     if args.match is None or args.match in unit)
    chosen = (waiting[-args.units:] if args.from_end else waiting[:args.units]) if args.units > 0 else []
    if not chosen:
        log(run_dir, f'{name} | take-back | no waiting unit matched; nothing moved')
        return
    plan = {'place': [], 'recover': [], 'pull_back': [], 'withdraw': chosen}
    moved = parse_moves(remote(run_dir, table, name, move_script(table[name], plan, int(time.time()))))
    returned = [(unit, data) for _, unit, data in moved]
    log(run_dir, f'{name} | take-back | {len(returned)} of {len(chosen)} waiting unit(s) taken back '
        f'(match: {args.match}, from the end: {args.from_end}, held: {args.hold})')
    # Held units go to queue/deferred, where no UVA worker takes them; "stage --include-deferred" sends them on.
    bring_back(run_dir, table, name, returned, 'take-back', 'deferred' if args.hold else 'pending')


def tick_server(run_dir, table, name, dry_run):
    """Bring one server's finished, failed, orphaned and long-waiting units up to date on UVA."""
    entry = table[name]
    assigned, status = read_status(run_dir, table, name, dry_run)
    if not status['prepared']:
        return status, assigned
    if not dry_run:
        record_unlisted_jobs(run_dir, name, status)
    assignment = load_assignment(run_dir)
    plan = plan_tick(name, assigned, status, assignment, set(ledger_ids(run_dir, name)), time.time())
    for job, requested, allocated in card_mismatches(name, status['cards']):
        log(run_dir, f'{name} | tick | WARNING job {job} asked for {requested} card(s) and holds {allocated}', dry_run)
    if not status['squeue_ok']:
        log(run_dir, f'{name} | tick | the job listing failed; no unit is recovered or withdrawn this tick', dry_run)
    for unit in plan['missing']:
        log(run_dir, f'{name} | tick | WARNING {unit} is staged here on UVA but in no queue folder on the server', dry_run)
    if dry_run:
        for action in ('place', 'pull', 'failed', 'recover', 'pull_back', 'withdraw'):
            for unit in plan[action]:
                log(run_dir, f'{name} | tick | would {action} {unit} {plan["reason"].get(unit, "")}', True)
        return status, assigned
    # (a) Finished units: pull the records, verify them on UVA, and only then mark the unit done.
    if plan['pull']:
        rsync(run_dir, table, name, [remote_path(entry, f'data/research/{plan["pull"][0]}')]
              + [f':{entry["remote_run"]}/data/research/{unit}' for unit in plan['pull'][1:]]
              + [str(run_dir / 'data/research') + '/'])
    for unit in plan['pull']:
        problem = verify_unit(run_dir, unit)
        if problem:
            log(run_dir, f'{name} | tick | {unit} is done on the server but NOT verified on UVA ({problem}); left staged')
            continue
        claim = status['claims'].get(('done', unit), {})
        block = result_block(name, claim, read_json(run_dir / 'data/research' / unit / 'summary.json'))
        move_unit(run_dir, unit, f'remote_{name}', 'done', {
            'job_id': claim.get('job'), 'hostname': claim.get('host'), 'finished_at': now(), 'result': block})
        finish_assignment(run_dir, [unit], 'done')
        log(run_dir, f'{name} | tick | {unit} verified and moved to queue/done (job {claim.get("job")}, '
            f'{block["wall_seconds"]} s)')
    # (b) Failed units: keep the end of the log on UVA and show it.
    for unit in plan['failed']:
        claim = status['claims'].get(('failed', unit), {})
        tail = status['tails'].get(unit, [])
        (run_dir / 'logs' / f'remote_{name}_{unit}.log').write_text('\n'.join(tail) + '\n')
        move_unit(run_dir, unit, f'remote_{name}', 'failed', {
            'job_id': claim.get('job'), 'hostname': claim.get('host'), 'finished_at': now(),
            'result': result_block(name, claim)})
        finish_assignment(run_dir, [unit], 'failed')
        log(run_dir, f'{name} | tick | {unit} FAILED on the server (job {claim.get("job")}, exit {claim.get("exit")}); '
            f'moved to queue/failed, last log lines in logs/remote_{name}_{unit}.log')
        print('\n'.join('    ' + line for line in tail[-12:]))
    # (c), (d) Orphaned and long-waiting units: one call moves them on the server and reports each success.
    if plan['place'] or plan['recover'] or plan['pull_back'] or plan['withdraw']:
        moved = parse_moves(remote(run_dir, table, name, move_script(entry, plan, int(time.time()))))
        returned = [(unit, data) for action, unit, data in moved if action in ('pull_back', 'withdraw')]
        assignment = load_assignment(run_dir)
        for action, unit, _ in moved:
            if action == 'recover':
                assignment[unit]['recoveries'] = assignment[unit].get('recoveries', 0) + 1
            log(run_dir, f'{name} | tick | {action} {unit}: {plan["reason"].get(unit, "copied earlier, now visible")}')
        write_json(assignment_path(run_dir), assignment)
        bring_back(run_dir, table, name, returned, 'tick')
    # A server that failed its first-unit check takes no more units until someone has looked.
    if status['check'] is not None and not status['check'].get('passed') and entry['ready']:
        entry['ready'] = False
        entry['not_ready_reason'] = (f'the first-unit check failed (largest relative difference '
                                     f'{status["check"].get("largest_relative_difference")})')
        write_json(table_path(run_dir), table)
        log(run_dir, f'{name} | tick | NOT READY from now on: {entry["not_ready_reason"]}')
    return status, assigned


def status_rows(run_dir, table, readings):
    """Format the status table: one row per pool, then the live jobs with the reason each one waits."""
    units_title = 'units on server (pending/running/done/failed/withdrawn)'
    header = (f'{"server":<10}{"ready":<7}{"pool":<19}{"live jobs":<28}{units_title:<56}{"staged":<8}'
              f'{"balance":<24}first-unit check')
    rows, jobs = [header, '-' * len(header)], []
    for name, entry in table.items():
        if name not in readings:
            rows.append(f'{name:<10}{"yes" if entry["ready"] else "NO":<7}'
                        f'not read: {entry.get("not_ready_reason", "not reachable in this call")}')
            continue
        status, assigned = readings[name]
        if not status['prepared']:
            rows.append(f'{name:<10}{"yes" if entry["ready"] else "NO":<7}reachable; its run folder is not set up yet')
            continue
        live = fleet_jobs(name, status, set(ledger_ids(run_dir, name)))
        check = ('not run yet' if status['check'] is None else
                 f"{'passed' if status['check'].get('passed') else 'FAILED'} "
                 f"(largest relative difference {status['check'].get('largest_relative_difference') or 0.0:.2e})")
        counts = '/'.join(str(len(status['queue'][state])) for state in REMOTE_STATES[1:])
        counts += f" (+{len(status['queue']['incoming'])} incoming)" if status['queue']['incoming'] else ''
        for pool_name in entry['pools']:
            states = [row['state'] for row in live.values() if pool_of(name, row['name']) == pool_name]
            summary = ', '.join(f'{states.count(state)} {state}' for state in sorted(set(states))) or 'none'
            rows.append(f'{name:<10}{"yes" if entry["ready"] else "NO":<7}{pool_name:<19}{summary:<28}{counts:<56}'
                        f'{len(assigned):<8}{budget_text(entry, status["budget"]):<24}{check}')
        jobs += [f'    {name} job {job} {row["name"]}: {row["state"]} ({row["reason"]}), elapsed {row["elapsed"]}, '
                 f'partition {row["partition"]}' for job, row in sorted(live.items())]
    return rows + jobs


def tick(run_dir, table, args, act=True):
    """Read every ready server once, act on what changed (unless only the status is wanted), print the table.

    The table shows each server as it was read, before this tick's own moves; those are in the lines above it.
    """
    readings = {}
    for name, entry in table.items():
        if not entry['ready']:
            staged = len(unit_names(run_dir / 'queue' / f'remote_{name}'))
            if staged:
                print(f'WARNING {name} is not ready and still holds {staged} staged unit(s).')
            continue
        # An unreachable server is reported and the others are still read.
        try:
            readings[name] = (tick_server(run_dir, table, name, args.dry_run) if act
                              else read_status(run_dir, table, name, True)[::-1])
        except (ServerUnreachable, RemoteCommandFailed) as error:
            log(run_dir, f'{name} | {"tick" if act else "status"} | NOT READ: {error}', args.dry_run or not act)
    print('\n'.join(status_rows(run_dir, table, readings)), flush=True)
    for name, (status, _) in readings.items():
        if act and not args.dry_run:
            log(run_dir, f'{name} | tick | balance: {budget_text(table[name], status["budget"])}; units on server '
                + ', '.join(f'{len(status["queue"][state])} {state}' for state in REMOTE_STATES[1:]))


def cancel_pending(run_dir, table, args):
    """Cancel this run's jobs on one server that are still PENDING, each by its exact id from the ledger."""
    name, entry = args.server, table[args.server]
    ids = ledger_ids(run_dir, name)
    # The state is read and the job cancelled in the same call, one id at a time; a running job is never touched.
    lines = [f'for id in {" ".join(ids)}; do',
             '  [ "$(squeue -h -j "$id" -o %T 2> /dev/null)" = PENDING ] && scancel "$id" && '
             f'echo "{MARK}cancelled $id"', 'done', f'echo "{MARK}end"', 'exit 0']
    answer = remote(run_dir, table, name, '\n'.join(lines) + '\n') if ids else f'{MARK}end'
    cancelled = [line.split()[1] for line in answer.splitlines() if line.startswith(f'{MARK}cancelled ')]
    log(run_dir, f'{name} | cancel-pending | cancelled {len(cancelled)} waiting job(s) by id: {" ".join(cancelled) or "none"}')


def main():
    """Read the command line and run one subcommand under the fleet lock."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-dir', type=Path, required=True)
    commands = parser.add_subparsers(dest='command', required=True)
    staging = commands.add_parser('stage')
    staging.add_argument('--server', required=True)
    staging.add_argument('--units', type=int, required=True)
    staging.add_argument('--from-end', action='store_true')
    staging.add_argument('--include-deferred', action='store_true')
    staging.add_argument('--match')
    submitting = commands.add_parser('submit')
    submitting.add_argument('--server', required=True)
    submitting.add_argument('--jobs', type=int, required=True)
    submitting.add_argument('--pool')
    ticking = commands.add_parser('tick')
    for command in (staging, submitting, ticking):
        command.add_argument('--dry-run', action='store_true')
    commands.add_parser('status')
    commands.add_parser('cancel-pending').add_argument('--server', required=True)
    taking = commands.add_parser('take-back')
    taking.add_argument('--server', required=True)
    taking.add_argument('--units', type=int, required=True)
    taking.add_argument('--from-end', action='store_true')
    taking.add_argument('--match')
    taking.add_argument('--hold', action='store_true',
                        help='put the units in queue/deferred, out of the UVA workers\' reach, to stage them elsewhere')
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    table = read_json(table_path(run_dir))
    if args.command == 'status':
        args.dry_run = True
        tick(run_dir, table, args, act=False)
        return
    with fleet_lock(run_dir):
        {'stage': stage, 'submit': submit, 'tick': tick, 'cancel-pending': cancel_pending,
         'take-back': take_back}[args.command](
            run_dir, table, args)


if __name__ == '__main__':
    main()
