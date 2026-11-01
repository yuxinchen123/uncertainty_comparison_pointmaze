"""Release one adopted ACES job once after published, fresh held-monitor proof."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys

RUN = Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1')
WRAPPER = RUN / 'remote_setup/2026-11-01-07-50_aces-autodetect-held-submission_helper-bad2ee3736a1_gpu-count-1_processor-count-8_host-memory-gib-32_time-limit-hours-3'
JOB = '2275518'
SUBMISSION_PUBLICATION = '36b496bd77df3a9e145a0fa572402081b792e2a2'
HELPER = 'bad2ee3736a1451094d5d1644b36b8cfc2d91c38'
MONITOR_JOB = '6781087'
MONITOR_SOURCE = '4814459cb7befc872d8f0e30682c23f4aa79b625'
SUBMISSION_SHA = '358f16cc1eaa369638153ee7dc090572cca49652ed5ae1595714e2617ef8735f'
BOUNDARY = {'workers_permitted': False, 'campaign_claims_permitted': False,
    'scientific_hypothesis_credit': 0, 'comparison_credit': 0, 'idea_credit': 0, 'admission_credit': 0}
NUMERICAL = {'jax', 'jaxlib', 'numpy', 'torch', 'flax', 'optax', 'cupy', 'scipy'}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def contents(path, limit=4 * 1024 * 1024):
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file() and not path.is_symlink(),
        'exact ordinary release artifact required: ' + str(path))
    raw = path.read_bytes()
    require(len(raw) <= limit, 'release metadata exceeds its bound')
    return raw


def read(path):
    return json.loads(contents(path))


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


def write(path, raw):
    require(path.parent.resolve(strict=True) == path.parent and not path.exists() and not path.is_symlink(),
        'preserve every prior release artifact')
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    os.fsync(descriptor)
    os.close(descriptor)


def field(raw, name):
    values = re.findall(r'(?:^|\s)' + re.escape(name) + r'=([^\s]+)', raw)
    require(len(values) == 1, 'missing or duplicated owned job field: ' + name)
    return values[0]


def resource_values(raw, name):
    pairs = [part.split('=', 1) for part in field(raw, name).split(',')]
    require(all(len(pair) == 2 for pair in pairs) and len({p[0] for p in pairs}) == len(pairs),
        'invalid or duplicated owned job resources')
    return dict(pairs)


def validate_held_detail(raw, native_packet):
    """Check the requested ACES shape without deriving physical identity from GRES numbering."""
    expected = {'JobId': JOB, 'UserId': 'u.sl359075(56626)', 'Account': '158590754315',
        'Partition': 'gpu', 'QOS': 'normal', 'JobState': 'PENDING', 'Reason': 'JobHeldUser',
        'NumTasks': '1', 'CPUs/Task': '8', 'TimeLimit': '03:00:00', 'MinMemoryNode': '32G',
        'Command': str(native_packet / 'job.sbatch')}
    require(all(field(raw, key) == value for key, value in expected.items()), 'actual owned held job request changed')
    require(field(raw, 'NumNodes') in {'1', '1-1'} and field(raw, 'NumCPUs') == '8', 'actual requested nodes or CPUs changed')
    values = resource_values(raw, 'ReqTRES')
    required = {'cpu': '8', 'mem': '32G', 'node': '1', 'gres/gpu': '1', 'gres/gpu:h100': '1'}
    require(all(values.get(k) == v for k, v in required.items())
        and all(k in required for k in values if k.startswith('gres/')), 'actual requested typed H100 resources changed')
    require(field(raw, 'TresPerNode') == 'gres/gpu:h100:1', 'actual typed per-node H100 request changed')


# The native child uses these exact same strict parsers, not a separately relaxed implementation.
REMOTE_PREFIX = r'''
import base64,datetime,hashlib,json,os,pathlib,pwd,re,socket,subprocess,sys
JOB = '2275518'
BOUNDARY = {'workers_permitted':False,'campaign_claims_permitted':False,'scientific_hypothesis_credit':0,'comparison_credit':0,'idea_credit':0,'admission_credit':0}
def require(value,message):
 if not value:raise ValueError(message)
def digest(raw):return hashlib.sha256(raw).hexdigest()
'''
REMOTE_SUFFIX = r'''
def ordinary(path):
 require(path.is_absolute() and path.resolve(strict=True)==path and path.is_file() and not path.is_symlink(),'ordinary native file required')
 raw=path.read_bytes();require(len(raw)<=4*1024*1024,'native metadata exceeds bound');return raw
def put(path,raw):
 require(path.parent.resolve(strict=True)==path.parent and not path.exists() and not path.is_symlink(),'preserve existing native release evidence')
 descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(descriptor,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
 descriptor=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(descriptor);os.close(descriptor)
def save(path,value):put(path,(json.dumps(value,indent=2,allow_nan=False)+'\n').encode())
def native_main(x):
 require(type(x.get('schema_version')) is int and x['schema_version']==1 and x.get('kind')=='owned_aces_full_site_release_intent' and x.get('automatic_retry_permitted') is False and all(type(x.get(k)) is type(v) and x[k]==v for k,v in BOUNDARY.items()),'typed zero-credit native release intent required')
 require(not {'jax','jaxlib','numpy','torch','flax','optax','cupy','scipy'}.intersection(sys.modules),'numerical modules forbidden')
 require(os.getuid()==56626 and pwd.getpwuid(os.getuid()).pw_name=='u.sl359075' and socket.gethostname().split('.')[0]=='alogin1','native login owner or host changed')
 require(x['job_id']==JOB and x['monitor_job_id']=='6781087' and x['monitor_source_commit']=='4814459cb7befc872d8f0e30682c23f4aa79b625','exact owned job and current monitor required')
 require(re.fullmatch(r'[0-9a-f]{40}',x['published_release_commit']) is not None,'published release review required')
 packet=pathlib.Path(x['packet']);run=pathlib.Path(x['run'])
 require(packet.resolve(strict=True)==packet and run.resolve(strict=True)==run and packet.parent==run/'remote_setup','native packet must be exact ordinary campaign child')
 require(packet.name==x['submission_row'][1] and x['submission_row']==[JOB,packet.name,'bad2ee3736a1451094d5d1644b36b8cfc2d91c38',str(run)],'native source and campaign identity changed')
 raw=ordinary(packet/'submission_receipt.json');submission=json.loads(raw)
 require(digest(raw)==x['submission_receipt_sha256']=='358f16cc1eaa369638153ee7dc090572cca49652ed5ae1595714e2617ef8735f','native immutable submission bytes changed')
 require(type(submission.get('schema_version')) is int and submission['schema_version']==1 and submission.get('kind')=='owned_aces_full_site_qualification_submission' and submission.get('cluster')=='aces' and submission.get('job_id')==JOB and submission.get('submission_row')==x['submission_row'],'native submission identity changed')
 require(ordinary(run/'slurm/submitted_jobids_aces.txt').decode().splitlines().count(JOB)==1,'native owned ID missing or duplicated')
 require([r for r in ordinary(run/'slurm/submissions_aces.tsv').decode().splitlines() if r.split('\t',1)[0]==JOB]==['\t'.join(x['submission_row'])],'native owned TSV row changed')
 for name in ['release_intent.json','release_receipt.json','release_return.json','release_preflight_return.json','release_owned_job_before.raw','release_owned_job_before_stderr.raw','release_stdout.raw','release_stderr.raw']:
  require(not (packet/name).exists() and not (packet/name).is_symlink(),'native release attempt already exists')
 for path,expected in x['native_hashes'].items():require(digest(ordinary(pathlib.Path(path)))==expected,'native immutable file changed: '+path)
 save(packet/'release_intent.json',x)
 query=['scontrol','show','job','-o',JOB]
 try:
  checked=subprocess.run(query,capture_output=True,timeout=30,check=False)
 except BaseException as error:
  put(packet/'release_owned_job_before.raw',getattr(error,'stdout',None) or b'')
  put(packet/'release_owned_job_before_stderr.raw',getattr(error,'stderr',None) or b'')
  save(packet/'release_preflight_return.json',{'command':query,'returncode':None,'error_type':type(error).__name__,'error':str(error),'automatic_retry_permitted':False})
  raise
 put(packet/'release_owned_job_before.raw',checked.stdout);put(packet/'release_owned_job_before_stderr.raw',checked.stderr)
 save(packet/'release_preflight_return.json',{'command':query,'returncode':checked.returncode,'automatic_retry_permitted':False})
 require(checked.returncode==0 and checked.stderr==b'','exact owned held query failed')
 validate_held_detail(checked.stdout.decode(),packet)
 command=['scontrol','release',JOB]
 with (packet/'release_stdout.raw').open('xb') as out,(packet/'release_stderr.raw').open('xb') as err:
  try:returned=subprocess.run(command,stdout=out,stderr=err,timeout=30,check=False)
  except BaseException as error:
   out.flush();os.fsync(out.fileno());err.flush();os.fsync(err.fileno())
   save(packet/'release_return.json',{'command':command,'returncode':None,'error_type':type(error).__name__,'error':str(error),'automatic_retry_permitted':False})
   raise
  out.flush();os.fsync(out.fileno());err.flush();os.fsync(err.fileno())
 save(packet/'release_return.json',{'command':command,'returncode':returned.returncode,'automatic_retry_permitted':False})
 require(returned.returncode==0,'native release failed; never retry')
 result={'schema_version':1,'kind':'owned_aces_full_site_qualification_release','cluster':'aces','job_id':JOB,'command':command,'returncode':0,'stdout':ordinary(packet/'release_stdout.raw').decode(),'stderr':ordinary(packet/'release_stderr.raw').decode(),'released_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'submission_receipt_sha256':digest(raw),'published_release_commit':x['published_release_commit'],'held_monitor_verification_sha256':x['held_monitor_verification_sha256'],'persistent_monitor_job_id':x['monitor_job_id'],'persistent_monitor_source_commit':x['monitor_source_commit'],'owned_job_detail_before_release':checked.stdout.decode(),'release_source_sha256':x['release_source_sha256'],'automatic_retry_permitted':False,**BOUNDARY}
 save(packet/'release_receipt.json',result)
 names=['release_receipt.json','release_intent.json','release_owned_job_before.raw','release_owned_job_before_stderr.raw','release_preflight_return.json','release_stdout.raw','release_stderr.raw','release_return.json']
 print(json.dumps({'files':{name:base64.b64encode(ordinary(packet/name)).decode() for name in names}}))
if __name__=='__main__':native_main(json.load(sys.stdin))
'''


def remote_program():
    import inspect
    return REMOTE_PREFIX + '\n'.join(inspect.getsource(f) for f in (field, resource_values, validate_held_detail)) + REMOTE_SUFFIX


def check_boundary(value):
    require(all(type(value.get(k)) is type(v) and value[k] == v for k, v in BOUNDARY.items()), 'release scientific flags changed')


def checked_payload(publication_commit=None):
    """Validate saved metadata only; require exact published bytes before an executable action."""
    require(socket.gethostname().split('.')[0] == 'portal05' and os.getuid() == 2618919, 'local release requires portal05/sl5nw')
    require(not NUMERICAL.intersection(sys.modules), 'numerical modules forbidden')
    repository = RUN.parents[2]
    plan_path = WRAPPER / 'aces_full_site_held_submission_plan.json'
    wrapper = read(plan_path);packet = Path(wrapper['local_packet']);native = read(packet / 'site_validation_plan.json')
    require(wrapper['helpers']['source_commit'] == HELPER and native['helpers']['source_commit'] == HELPER and native['python'] == wrapper['private_python'], 'release native helper or selected private Python changed')
    submission_raw = contents(packet / 'submission_receipt.json');submission = json.loads(submission_raw)
    require(digest(submission_raw) == SUBMISSION_SHA and submission['job_id'] == JOB and submission['published_preparation_commit'] == SUBMISSION_PUBLICATION, 'exact adopted native submission required')
    row = [JOB, packet.name, HELPER, native['campaign_run_dir']]
    require(submission['submission_row'] == row, 'exact adopted submission row required')
    for scope in (RUN / 'slurm', WRAPPER / 'execution/received'):
        require(contents(scope / 'submitted_jobids_aces.txt').decode().splitlines().count(JOB) == 1
            and [r for r in contents(scope / 'submissions_aces.tsv').decode().splitlines() if r.split('\t', 1)[0] == JOB] == ['\t'.join(row)], 'local/native returned ACES ID and row required')
    require(contents(WRAPPER / 'execution/received/submission_receipt.json') == submission_raw, 'native/local receipt bytes differ')
    adoption = read(packet / 'submission_adoption_receipt.json')
    require(adoption['passed'] is True and adoption['job_id'] == JOB and adoption['local_ledgers_fsynced_before_receipt'] is True and adoption['native_receipt_bytes_preserved'] is True, 'exact durable local adoption required')
    proof_raw = contents(packet / 'held_monitor_verification.json');proof = json.loads(proof_raw)
    require(proof['kind'] == 'owned_aces_full_site_actual_held_monitor_verification' and proof['passed'] is True
        and proof['job_id'] == JOB and proof['monitor_job_id'] == MONITOR_JOB and proof['monitor_source_commit'] == MONITOR_SOURCE
        and type(proof['actual_monitor_returncode']) is int and proof['actual_monitor_returncode'] == 0
        and proof['zero_scientific_credit'] is True and proof['session_cron_verified_by_root'] is True
        and proof['cadence_seconds'] == 1200 and len(proof['held_rows']) == 1
        and proof['native_submission_receipt_sha256'] == SUBMISSION_SHA, 'actual owned held full-monitor and recurring proof required')
    relative = '10_jax_exploration_platform/scripts/atari_pixel_campaign/aces_site_monitor.py'
    monitor_packet = RUN / 'remote_setup/2026-11-01-06-13_pixel-study-monitor_source-commit-4814459cb7befc872d8f0e30682c23f4aa79b625_processor-count-1_host-memory-gib-2_time-limit-days-7_poll-interval-seconds-1200_partition-nolim_account-sl5nw-base_gpu-count-0'
    pinned = monitor_packet / 'source_snapshot' / MONITOR_SOURCE / relative
    expected = subprocess.check_output(['git', '-C', str(repository), 'cat-file', 'blob', MONITOR_SOURCE + ':' + relative])
    require(contents(pinned) == expected, 'current pinned monitor adapter changed')
    spec = importlib.util.spec_from_file_location('reviewed_release_monitor_2275518', pinned)
    adapter = importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
    inputs = adapter.packet_inputs(RUN, packet)
    require(inputs['release'] is None and inputs['identifier'] == JOB, 'release already recorded or wrong exact job')
    checked_row, checked_record = adapter.site_row(inputs)
    require(checked_record == proof['held_rows'][0] and checked_row['state'] == 'held' and checked_row['slurm_job_id'] == JOB and checked_row['workers_permitted'] is False, 'current raw snapshot does not reproduce actual held monitor row')
    now = datetime.now(timezone.utc)
    for label, value, maximum in [('snapshot', checked_row['observed_at_utc'], 1200), ('held proof', proof['at_utc'], 1200)]:
        observed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        require(observed.tzinfo is not None and observed.utcoffset().total_seconds() == 0 and 0 <= (now - observed).total_seconds() < maximum, label + ' is stale or not UTC')
    require(digest(contents(packet / 'remote_monitor_snapshot.json')) == proof['snapshot_sha256'], 'actual held snapshot bytes changed')
    existing = ['release_intent.json','release_receipt.json','release_transport_stdout.raw','release_transport_stderr.raw','release_transport_result.json','release_return.json']
    require(all(not (packet / name).exists() and not (packet / name).is_symlink() for name in existing), 'preserve prior local release attempt')
    for name, key in [('site_validation_plan.json','plan_sha256'),('manifest.yaml','manifest_sha256'),('job.sbatch','job_script_sha256')]:
        require(digest(contents(packet / name)) == submission[key], 'native prepared bytes changed')
    immutable = {str(Path(wrapper['native_packet']) / entry['relative_path']): entry['sha256'] for entry in wrapper['native_packet_files']}
    for key in ('helpers', 'numerical_source'):
        identity = native[key]
        for path, hashed in identity['files'].items():
            immutable[str(Path(identity['source_root']) / path)] = hashed
    cleanrl = native['cleanrl_reference']['file'];immutable[cleanrl['path']] = cleanrl['sha256']
    files = [Path(__file__).absolute(),plan_path,packet/'site_validation_plan.json',packet/'manifest.yaml',packet/'job.sbatch',packet/'software_at_prepare.json',packet/'submission_receipt.json',packet/'submission_adoption_receipt.json',packet/'held_monitor_verification.json',packet/'remote_monitor_snapshot.json']
    original_pins = {'submit_aces_full_site_held.py':'48ef07150dffb1d7f366e61ae77813c37934da11479d1e4b0e5b7b0baa6d84a6','aces_full_site_held_submission_plan.json':'50d36dc21fe5842ddace8d84b4d7d91d584e0013b9aaaf4f19e2da778e7c6bae','adopt_aces_full_site_held.py':'4710655837443f8305c0d76a7ea9ea44c8aef67b1d76181f580d38e293d92467'}
    for name, hashed in original_pins.items():
        path=WRAPPER/name;require(digest(contents(path))==hashed, 'published submission utility changed')
        require(contents(path)==subprocess.check_output(['git','-C',str(repository),'cat-file','blob',SUBMISSION_PUBLICATION+':'+str(path.relative_to(repository))]), 'original submission publication bytes changed')
    if publication_commit is not None:
        require(re.fullmatch(r'[0-9a-f]{40}', publication_commit) is not None, 'full published release commit required')
        subprocess.run(['git','-C',str(repository),'merge-base','--is-ancestor',publication_commit,wrapper['publication_ref']],check=True)
        for path in files:
            require(contents(path)==subprocess.check_output(['git','-C',str(repository),'cat-file','blob',publication_commit+':'+str(path.relative_to(repository))]), 'release input was not published: '+str(path))
    value = {'schema_version':1,'kind':'owned_aces_full_site_release_intent','job_id':JOB,'packet':wrapper['native_packet'],'run':native['campaign_run_dir'],'at_utc':now.isoformat(),'submission_row':row,'submission_receipt_sha256':SUBMISSION_SHA,'held_monitor_verification_sha256':digest(proof_raw),'native_hashes':immutable,'published_release_commit':publication_commit,'release_source_sha256':digest(contents(Path(__file__).absolute())),'remote_source_sha256':digest(remote_program().encode()),'monitor_job_id':MONITOR_JOB,'monitor_source_commit':MONITOR_SOURCE,'automatic_retry_permitted':False,**BOUNDARY}
    return packet, native, value


def accept_response(raw, value):
    response = json.loads(raw)
    names={'release_receipt.json','release_intent.json','release_owned_job_before.raw','release_owned_job_before_stderr.raw','release_preflight_return.json','release_stdout.raw','release_stderr.raw','release_return.json'}
    require(set(response)=={'files'} and set(response['files'])==names, 'release response roster changed')
    blobs={name:base64.b64decode(raw,validate=True) for name,raw in response['files'].items()}
    require(all(len(raw)<=4*1024*1024 for raw in blobs.values()), 'returned release metadata exceeds bound')
    released=json.loads(blobs['release_receipt.json']);check_boundary(released)
    require(type(released.get('schema_version')) is int and released['schema_version']==1 and released['kind']=='owned_aces_full_site_qualification_release' and released['cluster']=='aces' and released['job_id']==JOB and released['command']==['scontrol','release',JOB] and type(released['returncode']) is int and released['returncode']==0, 'typed native release receipt changed')
    require(released['submission_receipt_sha256']==SUBMISSION_SHA and released['published_release_commit']==value['published_release_commit'] and released['held_monitor_verification_sha256']==value['held_monitor_verification_sha256'] and released['persistent_monitor_job_id']==MONITOR_JOB and released['persistent_monitor_source_commit']==MONITOR_SOURCE and released['release_source_sha256']==value['release_source_sha256'] and released['automatic_retry_permitted'] is False, 'native release proof identity changed')
    require(json.loads(blobs['release_intent.json'])==value, 'native/local release intent changed')
    for name in ('release_preflight_return.json','release_return.json'):
        returned=json.loads(blobs[name]);expected_command=['scontrol','show','job','-o',JOB] if name=='release_preflight_return.json' else ['scontrol','release',JOB]
        require(returned.get('command')==expected_command and type(returned['returncode']) is int and returned['returncode']==0 and returned['automatic_retry_permitted'] is False, 'native query/release return changed')
    require(blobs['release_owned_job_before_stderr.raw']==b'' and blobs['release_stdout.raw']==released['stdout'].encode() and blobs['release_stderr.raw']==released['stderr'].encode() and blobs['release_owned_job_before.raw']==released['owned_job_detail_before_release'].encode(), 'native raw release bytes differ')
    validate_held_detail(blobs['release_owned_job_before.raw'].decode(), Path(value['packet']))
    return blobs


def execute_once(packet, native, value):
    require(value['published_release_commit'] is not None, 'published release required before execution')
    write(packet/'release_intent.json',encoded(value))
    command=['ssh','-T','-o','ControlMaster=no','-o','ProxyCommand=false','-o','BatchMode=yes','-o','NumberOfPasswordPrompts=0','-o','ConnectionAttempts=1','-o','ConnectTimeout=10','aces','exec '+shlex.join([native['python'],'-I','-B','-c',remote_program()])]
    with (packet/'release_transport_stdout.raw').open('xb') as out,(packet/'release_transport_stderr.raw').open('xb') as err:
        try:result=subprocess.run(command,input=encoded(value),stdout=out,stderr=err,timeout=120,check=False)
        except BaseException as error:
            out.flush();os.fsync(out.fileno());err.flush();os.fsync(err.fileno())
            write(packet/'release_transport_result.json',encoded({'returncode':None,'error_type':type(error).__name__,'error':str(error),'automatic_retry_permitted':False}));raise
        out.flush();os.fsync(out.fileno());err.flush();os.fsync(err.fileno())
    write(packet/'release_transport_result.json',encoded({'returncode':result.returncode,'automatic_retry_permitted':False}))
    require(result.returncode==0, 'release transport failed; preserve every artifact and never retry')
    blobs=accept_response(contents(packet/'release_transport_stdout.raw'),value)
    # Publish the native receipt last after every returned raw/return record is durable locally.
    for name in sorted(blobs):
        if name!='release_receipt.json':write(packet/('remote_release_intent.json' if name=='release_intent.json' else name),blobs[name])
    write(packet/'release_receipt.json',blobs['release_receipt.json'])
    return {'released':True,'job_id':JOB,'remote_and_local_receipt_bytes_saved':True,**BOUNDARY}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--publication-commit');parser.add_argument('--release',action='store_true');args=parser.parse_args()
    require(not args.release or args.publication_commit is not None, 'publication commit is mandatory for one release')
    packet,native,value=checked_payload(args.publication_commit)
    if args.release:print(json.dumps(execute_once(packet,native,value)))
    else:print(json.dumps({'validated':True,'job_id':JOB,'native_release_argv':['scontrol','release',JOB],'native_packet':value['packet'],'publication_commit':args.publication_commit,'ssh_calls':0,'release_calls':0}))

if __name__=='__main__':main()
