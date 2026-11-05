"""Root-only held launch for one actual full profile qualification controller.

Derived from the reviewed configured-native diagnostic one-shot launcher. Metadata
preparation does not submit; every real action requires immutable actual inputs,
genuine independent reviews and exact locally recorded published Git ancestry.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess

PACKET = Path(__file__).resolve(strict=True).parent
RUN = PACKET.parent.parent
INTEGRATION = RUN.parents[2]
OWNER_UID = 2618919
PREFIX = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"
KIND = "full_original_profile_source_target_qualification"
PARENT_KIND = "profile_target_qualification_controller"
PROTOCOL = "full-original-profile-source-target-qualification-version-1"
BUDGET = {"schema_version":1,"total_seconds":10800,"work_seconds":10200,"cleanup_seconds":600}
STANDARD = ("manifest.yaml","config_resolved.yaml","command.txt","experiment_background.md")
ORIGIN_URL = "git@github.com:yuxinchen123/uncertainty_comparison_pointmaze.git"
PUBLICATION_REF = "refs/remotes/origin/merge-atari-cleanrl"
ORIGIN_FETCH = "+refs/heads/*:refs/remotes/origin/*"
ZERO = {"scientific_credit":0,"profile_credit":0,"case_credit":0,"idea_credit":0,
    "packing_credit":0,"final_training_credit":0,"worker_admitted":False,"admission_granted":False,
    "physical_gres_mapping_verified":False,"kernel_device_enforcement_verified":False}
LIMIT = 4 * 1024 * 1024


def require(value, message):
    """Refuse unsupported or previously attempted actions without a fallback."""
    if not value:
        raise ValueError(message)


def now():
    """Record an actual UTC timestamp rather than deriving one from a filename."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def new_path(path):
    """Preserve every completed, partial or dangling earlier publication attempt."""
    path = Path(path)
    require(not os.path.lexists(path) and not os.path.lexists(path.with_name(path.name + ".prepared")),
            "preserve the previous attempt: " + path.name)


def publish(path, value):
    """Publish a complete artifact atomically without overwriting an earlier one."""
    path = Path(path)
    new_path(path)
    require(path.parent.resolve(strict=True) == path.parent, "the artifact parent must be exact")
    raw = value if isinstance(value, bytes) else (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary = path.with_name(path.name + ".prepared")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    os.link(temporary, path, follow_symlinks=False)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(directory)
        temporary.unlink()
        os.fsync(directory)
    finally:
        os.close(directory)
    return reference(path)


def same(left, right):
    """Require exact JSON types, including integer rather than Boolean counts."""
    if type(left) is not type(right): return False
    if type(right) is dict: return left.keys() == right.keys() and all(same(left[key], right[key]) for key in right)
    if type(right) is list: return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return left == right


def actor():
    """Require the campaign owner's actual user identity before any scheduler call."""
    require(os.getuid() == os.geteuid() == OWNER_UID, "only the campaign owner may act on this packet")


def scheduler(command, stem):
    """Retain a single actual native command return, with no automatic retry."""
    actor()
    for suffix in ("_stdout.raw", "_stderr.raw", "_return.json"):
        new_path(PACKET / (stem + suffix))
    # Before: SBATCH_RESERVATION could override an omitted option; after: only recorded arguments and headers apply.
    environment = {key: value for key, value in os.environ.items() if not key.startswith("SBATCH_")}
    environment["TZ"] = "UTC"
    started = now()
    try:
        result = subprocess.run(command, capture_output=True, env=environment, timeout=30)
    except subprocess.TimeoutExpired as error:
        publish(PACKET / (stem + "_stdout.raw"), error.stdout or b"")
        publish(PACKET / (stem + "_stderr.raw"), error.stderr or b"")
        publish(PACKET / (stem + "_return.json"), {"command": command, "started_at_utc": started,
            "finished_at_utc": now(), "returncode": None, "timed_out": True, "automatic_retry_permitted": False})
        raise
    publish(PACKET / (stem + "_stdout.raw"), result.stdout)
    publish(PACKET / (stem + "_stderr.raw"), result.stderr)
    publish(PACKET / (stem + "_return.json"), {"command": command, "started_at_utc": started,
        "finished_at_utc": now(), "returncode": result.returncode, "automatic_retry_permitted": False,
        "inherited_SBATCH_variables_removed": sorted(key for key in os.environ if key.startswith("SBATCH_"))})
    require(type(result.returncode) is int and result.returncode == 0, "the actual native command failed; preserve this attempt")
    return result


def append_member(path, member):
    """Append and fsync one new exact ledger member through an owned pinned descriptor."""
    path = Path(path)
    before = ordinary_bytes(path)
    require(not before or before.endswith(b"\n"), "an existing ledger lacks its final newline")
    require(member not in before.decode().splitlines(), "the returned identifier or exact row is already recorded")
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
    try:
        status = os.fstat(descriptor)
        require(status.st_uid == OWNER_UID and stat.S_ISREG(status.st_mode), "the ledger is not an owned ordinary file")
        with os.fdopen(descriptor, "a", closefd=False) as stream:
            stream.write(member + "\n"); stream.flush(); os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    require(ordinary_bytes(path) == before + (member + "\n").encode(), "a ledger changed during its single append")
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return reference(path)


def ordinary_bytes(path):
    """Refuse aliases/nonregular files before a bounded pinned ordinary-file read."""
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path
        and stat.S_ISREG(path.lstat().st_mode), "an exact regular launch artifact is required")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and before.st_uid == OWNER_UID and 0 <= before.st_size <= LIMIT,
            "an owned bounded launch artifact is required")
        with os.fdopen(descriptor, 'rb', closefd=False) as stream:
            raw = stream.read(LIMIT + 1)
        after, named = os.fstat(descriptor), path.lstat()
        identity = lambda row:(row.st_dev,row.st_ino,row.st_uid,row.st_mode,row.st_size,row.st_mtime_ns,row.st_ctime_ns)
        require(len(raw) == before.st_size <= LIMIT and identity(before) == identity(after) == identity(named),
            "the launch artifact changed during its bounded read")
        return raw
    finally:
        os.close(descriptor)


def reference(path):
    """Match the unchanged runtime's exact two-field receipt schema."""
    return {'path':str(path),'sha256':hashlib.sha256(ordinary_bytes(path)).hexdigest()}


def document(path):
    """Read finite, duplicate-safe metadata without loading scientific modules."""
    def unique(rows):
        value = {}
        for key,item in rows:
            require(key not in value,'duplicate metadata key');value[key] = item
        return value
    def invalid(value):raise ValueError('nonfinite metadata number: '+value)
    return json.loads(ordinary_bytes(path),object_pairs_hook=unique,parse_constant=invalid)


def checked(item):
    """Validate an actual immutable reference before its bounded metadata read."""
    require(type(item) is dict and set(item) == {'path','sha256'} and reference(Path(item['path'])) == item,
        'an exact actual metadata reference is required')
    return document(Path(item['path']))


def git(*arguments):
    """Read only local Git with optional locking disabled; never fetch or authenticate."""
    environment = {**os.environ,'GIT_OPTIONAL_LOCKS':'0'}
    return subprocess.check_output(['git','-C',str(INTEGRATION),*arguments],env=environment,stderr=subprocess.PIPE,timeout=30)


def require_published_commit(commit):
    """Require actual configured origin ancestry in its exact locally recorded branch."""
    require(type(commit) is str and re.fullmatch('[0-9a-f]{40}',commit),'a real full publication commit is required')
    try:
        origin = git('config','--get-all','remote.origin.url').decode().splitlines()
        fetch = git('config','--get-all','remote.origin.fetch').decode().splitlines()
        tip = git('rev-parse','--verify',PUBLICATION_REF+'^{commit}').decode().strip()
        require(origin == [ORIGIN_URL] and fetch == [ORIGIN_FETCH] and re.fullmatch('[0-9a-f]{40}',tip),
            'the actual configured origin/branch mapping is required')
        git('merge-base','--is-ancestor',commit,tip)
    except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as error:
        raise ValueError('the exact publication commit is not in the recorded origin branch') from error
    return tip


def resources(scheduling):
    """Use one exact approved three-hour shape, with explicitly selected scheduling."""
    require(scheduling in ('reservation','normal'),'select reviewed reservation or normal scheduling')
    return {'partition':'gpu','account':'sl5nw_base','qos':'csresnolim' if scheduling == 'reservation' else 'normal',
        'reservation':'sl5nw_168' if scheduling == 'reservation' else None,'node':'jaguar03','nodes':1,'tasks':1,
        'requested_cpus':8,'host_memory_gib':64,'gres':'gpu:nvidia_rtx_a4500:1','wall_seconds':10800,'requeue':False}


def canonical(parent_unit):
    """Join genuine parent/child plans and launch review without opening scientific results."""
    require(PACKET.parent == RUN/'remote_setup' and PACKET.resolve(strict=True) == PACKET,'an exact campaign packet is required')
    parent = Path(parent_unit)
    require(parent.parent == RUN/'data' and parent.resolve(strict=True) == parent,'an exact actual parent unit is required')
    parent_plan = parent/'measurement_plan.json';p = document(parent_plan)
    child = RUN/'data'/PACKET.name;child_plan = child/'measurement_plan.json';c = document(child_plan)
    require(parent != child and p.get('kind') == PARENT_KIND and p.get('protocol') == PROTOCOL
        and p.get('unit_id') == parent.name and p.get('campaign_run_dir') == str(RUN) and p.get('profile_unit') == str(child)
        and type(p.get('controller_maximum_seconds')) is int and p['controller_maximum_seconds'] == 10800
        and same(p.get('qualification_time_budget'),BUDGET),'the actual full qualification controller differs')
    require(c.get('kind') == KIND and c.get('unit_id') == PACKET.name and c.get('campaign_run_dir') == str(RUN)
        and c.get('hardware_class') == 'jaguar03_rtx_a4500' and c.get('gpu_uuid') is None and c.get('job_id') is None
        and 'direct_binding' not in c and same(c.get('diagnostic_time_budget'),BUDGET)
        and same(c.get('profile_target_qualification'),{'schema_version':1,'protocol':PROTOCOL,'parent':reference(parent_plan)})
        and c.get('configured_isolation',{}).get('owned_inputs_path') == str(child/'owned_allocation_inputs.json'),
        'the real A4500 qualification capture plan differs')
    binding_path = parent/'prepared_child_binding.json';binding = document(binding_path)
    command = ['/usr/bin/python3','-I','-S','-B',str(parent/'code/profile_target_qualification.py'),'--prepared-unit',str(parent)]
    require(binding.get('controller_plan') == reference(parent_plan) and binding.get('child_plan') == reference(child_plan)
        and binding.get('controller_source') == reference(parent/'code/profile_target_qualification.py')
        and binding.get('command') == shlex.split(ordinary_bytes(child/'command.txt').decode())
        and same(binding.get('helpers'),c.get('helpers')) and shlex.split(ordinary_bytes(parent/'command.txt').decode()) == command,
        'the actual prepared controller/child command binding changed')
    review_path = parent/'controller_launch_independent_review.json';review = document(review_path)
    require(review.get('kind') == 'independent_profile_target_qualification_controller_launch_review'
        and review.get('approved') is review.get('passed') is review.get('independent_of_producer') is True
        and review.get('controller_plan') == reference(parent_plan) and review.get('child_plan') == reference(child_plan),
        'a genuine exact bound controller/child launch review is required')
    helpers = c.get('helpers')
    require(type(helpers) is dict and type(helpers.get('source_commit')) is str
        and re.fullmatch('[0-9a-f]{40}',helpers['source_commit']) and type(helpers.get('files')) is dict
        and helpers['files'],'the actual full helper closure is required')
    inputs = {parent_plan,child_plan,binding_path,review_path}
    for relative,expected in helpers['files'].items():
        require(type(relative) is str and relative.startswith(PREFIX) and Path(relative).name == relative[len(PREFIX):]
            and relative.endswith('.py') and re.fullmatch('[0-9a-f]{64}',expected),'unsupported staged helper reference')
        path = child/'code'/Path(relative).name
        require(reference(path)['sha256'] == expected and hashlib.sha256(git('cat-file','blob',helpers['source_commit']+':'+relative)).hexdigest() == expected,
            'a staged capture helper differs from its declared committed source')
        inputs.add(path)
    for name in ('profile_target_qualification.py','direct_common_timing_controller.py'):
        path = parent/'code'/name
        require(reference(path)['sha256'] == hashlib.sha256(git('cat-file','blob',helpers['source_commit']+':'+PREFIX+name)).hexdigest(),
            'a parent helper differs from the exact source commit');inputs.add(path)
    for unit in (PACKET,parent,child):
        inputs.update(unit/name for name in STANDARD)
    for name in ('measurement_prepared.json',):
        path = child/name
        require(document(path) == {'plan_sha256':reference(child_plan)['sha256']},'the actual child preparation receipt differs')
        inputs.add(path)
    for unit,names in ((parent,('controller_intent.json','controller_spawn.json','controller_wait.json','controller_failure.json')),
            (child,('progress.json','result.json','failure.json','owned_allocation_inputs.json'))):
        require(not any(os.path.lexists(unit/name) for name in names),'preserve the attempted runtime unit; never reuse it')
    # Metadata references are hashed here; old scientific result and captured payload references are not followed.
    for path in inputs:ordinary_bytes(path)
    return p,c,command,inputs


def held_command(policy):
    """Return the exact held command, with every resource specified in recorded argv."""
    result = ['sbatch','--hold','--parsable','--no-requeue','--partition=gpu','--account=sl5nw_base',
        '--qos='+policy['qos'],'--nodelist=jaguar03','--nodes=1','--ntasks=1','--cpus-per-task=8','--mem=64G',
        '--gres=gpu:nvidia_rtx_a4500:1','--time=03:00:00']
    if policy['reservation'] is not None:result += ['--reservation='+policy['reservation']]
    return result+[str(PACKET/'job.sbatch')]


def expected_job_script(parent,policy):
    """Run the real metadata parent in one eight-CPU owned numeric step after envelope publication."""
    lines = ['#!/bin/bash','#SBATCH --job-name=profile-target-qualification','#SBATCH --nodes=1','#SBATCH --ntasks=1',
        '#SBATCH --cpus-per-task=8','#SBATCH --mem=64G','#SBATCH --time=03:00:00','#SBATCH --partition=gpu',
        '#SBATCH --nodelist=jaguar03','#SBATCH --account=sl5nw_base','#SBATCH --qos='+policy['qos'],
        '#SBATCH --gres=gpu:nvidia_rtx_a4500:1','#SBATCH --no-requeue']
    if policy['reservation'] is not None:lines += ['#SBATCH --reservation='+policy['reservation']]
    lines += ['#SBATCH --output='+str(RUN/'slurm/jobs/profile-target-%j.out'),
        '#SBATCH --error='+str(RUN/'slurm/jobs/profile-target-%j.err'),'set -euo pipefail','umask 077',
        'unset PYTHONPATH PYTHONHOME','export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1',
        'cd '+shlex.quote(str(RUN)), 'sha256sum --check --strict '+shlex.quote(str(PACKET/'source_checksums.sha256')),
        'SECONDS=0','while [ ! -f '+shlex.quote(str(RUN/'data'/PACKET.name/'owned_allocation_inputs.json'))+' ]; do',
        '  if [ "$SECONDS" -ge 60 ]; then exit 75; fi','  sleep 0.1','done',
        'exec '+shlex.join(['srun','--exact','--nodes=1','--ntasks=1','--cpus-per-task=8','--cpu-bind=cores',
            '--gres=gpu:nvidia_rtx_a4500:1','--kill-on-bad-exit=1','/usr/bin/python3','-I','-S','-B',
            str(parent/'code/profile_target_qualification.py'),'--prepared-unit',str(parent)])]
    return ('\n'.join(lines)+'\n').encode()


def prepare_request(parent_unit,scheduling):
    """Create only late launch records from actual approved prepared units, preserving all standards."""
    actor()
    for name in ('qualification_launch_request.json','job.sbatch','source_checksums.sha256','held_submission_command.json',
            'immutable_artifacts.json','prepared_launch_binding.json','submission_intent.json'):
        new_path(PACKET/name)
    p,c,command,inputs = canonical(parent_unit);parent = Path(parent_unit);policy = resources(scheduling)
    request = {'schema_version':1,'kind':'profile_target_qualification_aa_launch_request','parent_unit':str(parent),
        'child_unit':str(RUN/'data'/PACKET.name),'parent_plan':reference(parent/'measurement_plan.json'),
        'child_plan':reference(RUN/'data'/PACKET.name/'measurement_plan.json'),
        'prepared_child_binding':reference(parent/'prepared_child_binding.json'),
        'controller_launch_review':reference(parent/'controller_launch_independent_review.json'),
        'resources':policy,'actual_job_id':None,'actual_release_performed':False,**ZERO}
    publish(PACKET/'qualification_launch_request.json',request)
    publish(PACKET/'job.sbatch',expected_job_script(parent,policy))
    publish(PACKET/'held_submission_command.json',held_command(policy))
    checks = inputs|{PACKET/'qualification_launch_request.json'}
    publish(PACKET/'source_checksums.sha256',''.join(reference(path)['sha256']+'  '+str(path)+'\n' for path in sorted(checks)).encode())
    inputs |= {PACKET/name for name in ('qualification_launch_request.json','job.sbatch','held_submission_command.json','source_checksums.sha256')}
    publish(PACKET/'immutable_artifacts.json',{'files':{str(path):{'sha256':reference(path)['sha256'],
        'bytes':len(ordinary_bytes(path)),'mode':oct(path.stat().st_mode&0o777)} for path in sorted(inputs)}})
    return {'request':reference(PACKET/'qualification_launch_request.json'),'action_performed':False,**ZERO}


def prepared_records():
    """Recompute only actual small metadata/source bindings; never consume scientific results."""
    request = document(PACKET/'qualification_launch_request.json')
    require(request.get('kind') == 'profile_target_qualification_aa_launch_request' and type(request.get('schema_version')) is int
        and request['schema_version'] == 1 and request.get('actual_job_id') is None and request.get('actual_release_performed') is False,
        'an actual unsubmitted late launch request is required')
    parent = Path(request['parent_unit']);p,c,command,inputs = canonical(parent)
    policy = request['resources'];require(any(same(policy,resources(mode)) for mode in ('normal','reservation')),'unsupported resource policy')
    for key,path in (('parent_plan',parent/'measurement_plan.json'),('child_plan',RUN/'data'/PACKET.name/'measurement_plan.json'),
            ('prepared_child_binding',parent/'prepared_child_binding.json'),('controller_launch_review',parent/'controller_launch_independent_review.json')):
        require(request.get(key) == reference(path),'a late request actual reference changed')
    require(request.get('child_unit') == str(RUN/'data'/PACKET.name) and all(same(request.get(k),v) for k,v in ZERO.items()),
        'the child or zero-credit declaration changed')
    require(ordinary_bytes(PACKET/'job.sbatch') == expected_job_script(parent,policy)
        and document(PACKET/'held_submission_command.json') == held_command(policy),'the exact held script/command changed')
    checks = inputs|{PACKET/'qualification_launch_request.json'}
    require(ordinary_bytes(PACKET/'source_checksums.sha256') == ''.join(reference(path)['sha256']+'  '+str(path)+'\n' for path in sorted(checks)).encode(),
        'the source checksum closure changed')
    inputs |= {PACKET/name for name in ('qualification_launch_request.json','job.sbatch','held_submission_command.json','source_checksums.sha256')}
    inventory = document(PACKET/'immutable_artifacts.json')
    require(type(inventory) is dict and set(inventory) == {'files'} and set(inventory['files']) == {str(path) for path in inputs},
        'the complete actual launch inventory is required')
    for path in inputs:
        expected = inventory['files'][str(path)];raw = ordinary_bytes(path)
        require(type(expected) is dict and set(expected) == {'sha256','bytes','mode'} and type(expected['bytes']) is int
            and expected == {'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'mode':oct(path.stat().st_mode&0o777)},
            'a frozen launch artifact changed')
    binding = {'schema_version':1,'kind':'actual_profile_target_qualification_prepared_launch_binding',
        'request':reference(PACKET/'qualification_launch_request.json'),'parent_plan':request['parent_plan'],
        'plan':request['child_plan'],'data_unit':request['child_unit'],'source_commit':c['helpers']['source_commit'],
        'resources':policy,'immutable_artifacts':reference(PACKET/'immutable_artifacts.json'),
        'submission_command':held_command(policy),'actual_job_id':None,'actual_release_performed':False,**ZERO}
    return c,held_command(policy),binding


def prepare_binding_once():
    """Publish an immutable actual prepared binding without any scheduler operation."""
    actor()
    for name in ('prepared_launch_binding.json','submission_intent.json','submission_receipt.json'):new_path(PACKET/name)
    unused,unused_command,binding = prepared_records()
    return publish(PACKET/'prepared_launch_binding.json',binding)


def checked_packet():
    """Require the reviewed immutable later binding, never placeholder records."""
    plan,command,binding = prepared_records()
    require(same(document(PACKET/'prepared_launch_binding.json'),binding),'the actual prepared binding changed')
    return plan,command


def check_publication(commit,release=False):
    """Require reviewed actual source/records published in the exact recorded origin branch."""
    require_published_commit(commit)
    freeze_path = PACKET/'launcher_source_freeze.json';freeze = document(freeze_path)
    review_path = PACKET/'launcher_independent_review.json';review = document(review_path)
    require(review.get('kind') == 'independent_profile_target_qualification_prepared_launcher_review'
        and review.get('passed') is review.get('approved') is review.get('independent_of_producer') is True
        and review.get('source_freeze') == reference(freeze_path)
        and review.get('prepared_binding') == reference(PACKET/'prepared_launch_binding.json'),
        'the exact launcher and actual prepared binding need genuine independent approval')
    static = document(PACKET/'packet_static_verification.json')
    require(static.get('passed') is True and static.get('prepared_binding') == reference(PACKET/'prepared_launch_binding.json'),
        'the actual packet needs exact static verification')
    require(type(freeze.get('files')) is dict,'an exact immutable source/publication roster is required')
    paths = dict(freeze['files'])
    mandatory = set(document(PACKET/'immutable_artifacts.json')['files'])|{str(PACKET/name) for name in (
        'immutable_artifacts.json','prepared_launch_binding.json','packet_static_verification.json',
        'submit_profile_qualification_held_once.py','test_submit_profile_qualification_held_once.py')}
    require(mandatory <= set(paths),'the publication roster omitted a mandatory actual input')
    paths.update({str(path):reference(path)['sha256'] for path in (freeze_path,review_path)})
    if release:
        binding_path = PACKET/'release_binding.json';independent_path = PACKET/'release_independent_review.json'
        independent = document(independent_path)
        require(independent.get('passed') is independent.get('approved') is independent.get('independent_of_producer') is True
            and independent.get('kind') == 'independent_profile_target_qualification_release_review'
            and independent.get('binding') == reference(binding_path),'the fixed actual release needs independent approval')
        binding = document(binding_path)
        for path in (binding_path,independent_path,PACKET/'submission_receipt.json',PACKET/'submission_ledgers_fsynced.json',
                Path(binding['ledger']['snapshot']['path']),Path(binding['submissions']['snapshot']['path'])):
            paths[str(path)] = reference(path)['sha256']
        for name in ('release_preparation_query_stdout.raw','release_preparation_query_stderr.raw','release_preparation_query_return.json'):
            path = PACKET/name;paths[str(path)] = reference(path)['sha256']
    for name,expected in paths.items():
        path = Path(name)
        require(path.is_relative_to(INTEGRATION) and re.fullmatch('[0-9a-f]{64}',expected)
            and reference(path)['sha256'] == expected,'a reviewed actual source or packet input changed')
        require(hashlib.sha256(git('cat-file','blob',commit+':'+str(path.relative_to(INTEGRATION)))).hexdigest() == expected,
            'a reviewed actual input is not the published Git byte sequence')


def identifier(value):
    """Accept only an actual positive scheduler ID; ledger joins establish its ownership."""
    require(type(value) is str and re.fullmatch('[1-9][0-9]*',value),'invalid actual qualification job ID')
    return value


def submission_row(job,plan):
    """Match the unchanged configured-isolation reader's exact CHILD row."""
    return [identifier(job),KIND,plan['helpers']['source_commit'],str(PACKET/'job.sbatch')]


def ledgers():
    """Name only this campaign's three local UVA submission ledgers."""
    return (RUN/'slurm/submitted_jobids.txt',RUN/'slurm/submitted_jobids_uva.txt',RUN/'slurm/submissions.tsv')


def owned_submission(plan):
    """Prove actual identifier ownership and unique command before every exact job query."""
    value = document(PACKET/'submission_receipt.json');job = identifier(value.get('job_id'));row = submission_row(job,plan)
    require(value.get('kind') == 'owned_uva_configured_isolation_submission' and type(value.get('schema_version')) is int
        and value['schema_version'] == 1 and value.get('cluster') == 'uva' and type(value.get('returncode')) is int
        and value['returncode'] == 0 and value.get('plan_sha256') == reference(RUN/'data'/PACKET.name/'measurement_plan.json')['sha256']
        and value.get('data_unit') == str(RUN/'data'/PACKET.name) and value.get('submission_row') == row
        and value.get('held_command') == document(PACKET/'held_submission_command.json'),'the actual held receipt differs')
    for path,member in zip(ledgers(),(job,job,'\t'.join(row))):
        lines = ordinary_bytes(path).decode().splitlines()
        require(lines.count(member) == 1,'the actual ID/row lacks unique campaign ownership')
        if path.name != 'submissions.tsv':require(len(lines) == len(set(lines)),'duplicate owned identifiers')
        else:require([line for line in lines if line.split('\t',1)[0] == job] == [member],'conflicting owned submission row')
    synced = document(PACKET/'submission_ledgers_fsynced.json')
    require(synced.get('job_id') == job and synced.get('submission_row') == row and len(synced.get('ledgers',[])) == 3,
        'the three-ledger fsync receipt is required')
    return value


def native_fields(raw):
    """Reject ambiguous duplicate native key/value fields."""
    pairs = re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_/:]*)=(\S*)',raw.decode())
    require(len({key for key,value in pairs}) == len(pairs),'duplicate native fields')
    return dict(pairs)


def reservation_observation(policy,stem):
    """Observe only the explicit named reservation in UTC immediately before a real action."""
    if policy['reservation'] is None:return None
    result = scheduler(['scontrol','show','reservation','sl5nw_168','-o'],stem)
    fields = native_fields(result.stdout);current = datetime.datetime.now(datetime.timezone.utc)
    require(fields.get('ReservationName') == 'sl5nw_168' and fields.get('State') == 'ACTIVE',
        'the exact named reservation is not active')
    def utc(value):return datetime.datetime.fromisoformat(value).replace(tzinfo=datetime.timezone.utc)
    require(utc(fields['StartTime']) <= current and utc(fields['EndTime']) >= current+datetime.timedelta(seconds=10800),
        'the fresh UTC reservation lacks the full requested wall window')
    require('jaguar03' in fields.get('Nodes','').split(','),'the fresh reservation does not explicitly include jaguar03')
    return {'return':reference(PACKET/(stem+'_return.json')),'stdout':reference(PACKET/(stem+'_stdout.raw')),
        'stderr':reference(PACKET/(stem+'_stderr.raw'))}


def submit_held_once(commit):
    """Submit once held; fsync all three real ledger rows before the actual submission receipt."""
    plan,command = checked_packet();check_publication(commit);actor()
    for name in ('submission_intent.json','submission_receipt.json','submission_ledgers_fsynced.json'):new_path(PACKET/name)
    child = RUN/'data'/PACKET.name
    for name in ('configured_isolation_owned_job_ids.txt','configured_isolation_owned_submissions.tsv','owned_allocation_inputs.json'):
        new_path(child/name)
    with (RUN/'queue/metadata.lock').open('r+') as guard:
        fcntl.flock(guard.fileno(),fcntl.LOCK_EX)
        publish(PACKET/'submission_intent.json',{'kind':'one_profile_target_qualification_held_submission_attempt',
            'at_utc':now(),'publication':commit,'command':command,'plan':reference(child/'measurement_plan.json'),
            'automatic_retry_permitted':False})
        reservation = reservation_observation(document(PACKET/'qualification_launch_request.json')['resources'],'submission_reservation')
        actual = scheduler(command,'actual_submission')
        parsed = re.fullmatch(r'([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?',actual.stdout.decode().strip())
        require(parsed is not None,'actual parsable output lacks a unique job identifier')
        job = identifier(parsed.group(1));row = submission_row(job,plan)
        require(not any(line.split('\t',1)[0] == job for line in ordinary_bytes(ledgers()[2]).decode().splitlines()),
            'the returned identifier already has a submission row')
        durable = [append_member(path,member) for path,member in zip(ledgers(),(job,job,'\t'.join(row)))]
        publish(PACKET/'submission_ledgers_fsynced.json',{'kind':'actual_profile_target_qualification_three_ledgers_fsynced',
            'job_id':job,'submission_row':row,'ledgers':durable,'at_utc':now()})
        receipt = {'schema_version':1,'kind':'owned_uva_configured_isolation_submission','cluster':'uva',
            'job_id':job,'returncode':0,'plan_sha256':reference(child/'measurement_plan.json')['sha256'],
            'data_unit':str(child),'submission_row':row,'held_command':command,'submitted_held':True,
            'publication':commit,'at_utc':now(),'reservation_observation':reservation,**ZERO}
        publish(PACKET/'submission_receipt.json',receipt)
        publish(child/'configured_isolation_owned_job_ids.txt',ordinary_bytes(ledgers()[1]))
        publish(child/'configured_isolation_owned_submissions.tsv',ordinary_bytes(ledgers()[2]))
        job_directory = RUN/'slurm/jobs'/job
        require(not os.path.lexists(job_directory),'preserve the existing per-job directory');job_directory.mkdir()
        for name in (*STANDARD,'job.sbatch'):
            publish(job_directory/name,ordinary_bytes(PACKET/name))
        publish(job_directory/'assignment.json',{'job_id':job,'parent_unit':document(PACKET/'qualification_launch_request.json')['parent_unit'],
            'child_unit':str(child),'submission_row':row,'state':'held','actual_gpu_uuid':None,
            'submission':reference(PACKET/'submission_receipt.json'),**ZERO})
    return receipt


def query_held(job,stem,plan):
    """Query exactly the proven owned ID and require the recorded held resource shape."""
    require(owned_submission(plan)['job_id'] == identifier(job),'the query ID is not this actual owned job')
    result = scheduler(['scontrol','show','job','-o',job],stem);fields = native_fields(result.stdout)
    policy = document(PACKET/'qualification_launch_request.json')['resources']
    expected = {'JobId':job,'UserId':'sl5nw('+str(OWNER_UID)+')','JobState':'PENDING','Reason':'JobHeldUser',
        'Requeue':'0','Restarts':'0','TimeLimit':'03:00:00','Partition':'gpu','Account':'sl5nw_base',
        'QOS':policy['qos'],'ReqNodeList':'jaguar03','CPUs/Task':'8','NumTasks':'1','Command':str(PACKET/'job.sbatch')}
    require(all(fields.get(key) == value for key,value in expected.items())
        and fields.get('NumNodes') in ('1','1-1') and fields.get('Reservation','') == (policy['reservation'] or ''),
        'the actual held owner/shape/state differs')
    pairs = [item.split('=',1) for item in fields.get('ReqTRES','').split(',')]
    require(all(len(pair) == 2 for pair in pairs) and len({pair[0] for pair in pairs}) == len(pairs),'ambiguous requested resources')
    requested = dict(pairs)
    require(requested.get('cpu') in ('8','16') and fields.get('NumCPUs') in ('8','16')
        and requested.get('mem') == '64G' and requested.get('node') == '1' and requested.get('gres/gpu') == '1'
        and fields.get('TresPerNode') == 'gres/gpu:nvidia_rtx_a4500:1','the actual requested TRES differ')
    return {'return':reference(PACKET/(stem+'_return.json')),'stdout':reference(PACKET/(stem+'_stdout.raw')),
        'stderr':reference(PACKET/(stem+'_stderr.raw'))}


def snapshot_binding(job,plan):
    """Join immutable actual snapshots while preserving later live-ledger append operations."""
    child = RUN/'data'/PACKET.name;row = submission_row(job,plan);value = {}
    for key,name,source,member in (('ledger','configured_isolation_owned_job_ids.txt',ledgers()[1],job),
            ('submissions','configured_isolation_owned_submissions.tsv',ledgers()[2],'\t'.join(row))):
        path = child/name;lines = ordinary_bytes(path).decode().splitlines()
        require(lines.count(member) == 1 and (key != 'ledger' or len(lines) == len(set(lines))),'invalid actual owned snapshot')
        if key == 'submissions':require([line for line in lines if line.split('\t',1)[0] == job] == [member],'conflicting snapshot row')
        value[key] = {'snapshot':reference(path),'source_path':str(source)}
    return value


def prepare_release(commit):
    """Freeze one actual identifier after its held query and durable own snapshots exist."""
    plan,unused = checked_packet();check_publication(commit);new_path(PACKET/'release_binding.json')
    actual = owned_submission(plan);job = actual['job_id'];snapshots = snapshot_binding(job,plan)
    query = query_held(job,'release_preparation_query',plan)
    binding = {'schema_version':1,'kind':'actual_profile_target_qualification_fixed_identifier_release_binding',
        'job_id':job,'command':['scontrol','release',job],'submission':reference(PACKET/'submission_receipt.json'),
        'plan':reference(RUN/'data'/PACKET.name/'measurement_plan.json'),'held_query':query,
        'at_utc':now(),'actual_release_performed':False,**snapshots}
    return publish(PACKET/'release_binding.json',binding)


def release_once(commit):
    """Fsync the real successful release receipt before the atomic unchanged owned-input envelope."""
    plan,unused = checked_packet();check_publication(commit,release=True)
    child = RUN/'data'/PACKET.name
    for path in (PACKET/'release_intent.json',PACKET/'release_receipt.json',child/'owned_allocation_inputs.json'):new_path(path)
    actual = owned_submission(plan);job = actual['job_id'];binding = document(PACKET/'release_binding.json')
    require(binding.get('job_id') == job and binding.get('command') == ['scontrol','release',job]
        and binding.get('submission') == reference(PACKET/'submission_receipt.json')
        and binding.get('plan') == reference(child/'measurement_plan.json')
        and all(binding.get(key) == item for key,item in snapshot_binding(job,plan).items()),'the fixed release binding changed')
    query_held(job,'release_final_query',plan)
    reservation = reservation_observation(document(PACKET/'qualification_launch_request.json')['resources'],'release_reservation')
    publish(PACKET/'release_intent.json',{'kind':'one_actual_profile_target_qualification_release_attempt','job_id':job,
        'at_utc':now(),'publication':commit,'binding':reference(PACKET/'release_binding.json'),'automatic_retry_permitted':False})
    scheduler(binding['command'],'actual_release')
    receipt = {'schema_version':1,'kind':'owned_uva_configured_isolation_release','cluster':'uva','job_id':job,'returncode':0,
        'command':binding['command'],'submission_receipt_sha256':reference(PACKET/'submission_receipt.json')['sha256'],
        'plan_sha256':reference(child/'measurement_plan.json')['sha256'],'data_unit':str(child),
        'publication':commit,'at_utc':now(),'reservation_observation':reservation,**ZERO}
    release = publish(PACKET/'release_receipt.json',receipt)
    ownership = {'schema_version':1,'kind':'uva_configured_isolation_owned_inputs','job_id':job,
        'plan':reference(child/'measurement_plan.json'),'submission_row':submission_row(job,plan),
        'submission':reference(PACKET/'submission_receipt.json'),'release':release,
        **{key:binding[key] for key in ('ledger','submissions')}}
    final = publish(child/'owned_allocation_inputs.json',ownership)
    publish(PACKET/'release_completed.json',{'job_id':job,'release':release,'owned_allocation_inputs':final,
        'at_utc':now(),'released_once':True,**ZERO})
    return receipt


def main(argv=None):
    """Default to an actual publication check; explicit modes are the only action entry points."""
    parser = argparse.ArgumentParser(description=__doc__);operations = parser.add_mutually_exclusive_group()
    for name in ('prepare-request','prepare-binding-once','submit-held-once','prepare-release','release-once'):
        operations.add_argument('--'+name,action='store_true')
    parser.add_argument('--parent-unit',type=Path);parser.add_argument('--scheduling',choices=('normal','reservation'))
    parser.add_argument('--publication-commit');args = parser.parse_args(argv)
    if args.prepare_request:
        require(args.parent_unit is not None and args.scheduling is not None and args.publication_commit is None,
            'metadata request preparation needs actual parent and explicit scheduling, without a publication claim')
        value = prepare_request(args.parent_unit,args.scheduling)
    else:
        require(args.parent_unit is None and args.scheduling is None,'prepared operations never accept replacement parent/resources')
        if args.prepare_binding_once:
            require(args.publication_commit is None,'metadata binding makes no publication claim');value = prepare_binding_once()
        elif args.submit_held_once:value = submit_held_once(args.publication_commit)
        elif args.prepare_release:value = prepare_release(args.publication_commit)
        elif args.release_once:value = release_once(args.publication_commit)
        else:
            plan,command = checked_packet();check_publication(args.publication_commit)
            value = {'kind':'profile_target_qualification_launcher_default_check','passed':True,'action_performed':False,
                'held_submission_command':command,'plan':reference(RUN/'data'/PACKET.name/'measurement_plan.json'),**ZERO}
    print(json.dumps(value,indent=2,sort_keys=True))


if __name__ == '__main__':main()
