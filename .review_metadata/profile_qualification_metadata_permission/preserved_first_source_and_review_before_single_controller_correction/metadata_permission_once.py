"""Protect two owned metadata files before launch and restore only after recorded startup or exit."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import socket
import stat
import subprocess

OWNER=2618919
CAPABILITY='owned_campaign_metadata_permission_root_action_version_1'
ORIGIN='git@github.com:yuxinchen123/uncertainty_comparison_pointmaze.git'
PREFIX='10_jax_exploration_platform/scripts/atari_pixel_campaign/'
PROTOCOL='full-original-profile-source-target-qualification-version-1'


def require(value,message):
    """Refuse an incomplete identity or proof without changing permission bits."""
    if not value:raise ValueError(message)


def capture(path,descriptor=None):
    """Pin bounded owned regular bytes and their actual same-host identity."""
    path=Path(path).absolute();before=path.lstat()
    require(path.resolve(strict=True)==path and stat.S_ISREG(before.st_mode) and before.st_uid==OWNER
        and before.st_nlink==1 and before.st_size<=4*1024*1024,'an exact bounded singly linked owned file is required')
    own=descriptor is None
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK) if own else descriptor
    try:
        identity=lambda item:(item.st_dev,item.st_ino,item.st_uid,item.st_gid,item.st_mode,item.st_size,item.st_mtime_ns,item.st_ctime_ns,item.st_nlink)
        require(identity(os.fstat(fd))==identity(before),'file identity changed before capture')
        os.lseek(fd,0,os.SEEK_SET);raw=b''
        while len(raw)<=before.st_size:
            block=os.read(fd,min(65536,before.st_size+1-len(raw)))
            if not block:break
            raw+=block
        require(len(raw)==before.st_size and identity(os.fstat(fd))==identity(before)==identity(path.lstat()),'file changed during capture')
        return raw,{'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}, {
            'path':str(path),'hostname':socket.gethostname(),'device':before.st_dev,'inode':before.st_ino,
            'uid':before.st_uid,'gid':before.st_gid,'mode':stat.S_IMODE(before.st_mode),'nlink':before.st_nlink,
            'bytes':before.st_size,'mtime_ns':before.st_mtime_ns,'sha256':hashlib.sha256(raw).hexdigest()}
    finally:
        if own:os.close(fd)


def reference(path):
    """Return one actual path/hash reference without interpreting its data."""
    return capture(path)[1]


def document(path):
    """Reject duplicate JSON keys and nonfinite numbers in operational records."""
    def unique(items):
        value={}
        for key,item in items:
            require(key not in value,'duplicate metadata key');value[key]=item
        return value
    return json.loads(capture(path)[0],object_pairs_hook=unique,
        parse_constant=lambda value:(_ for _ in ()).throw(ValueError('nonfinite metadata')))


def checked(value):
    """Require the complete exact reference before consuming a saved fact."""
    require(type(value) is dict and set(value)=={'path','sha256'} and reference(value['path'])==value,'immutable metadata reference changed')
    return document(value['path'])


def write_once(path,value):
    """Fsync each real intent/result without replacing an earlier operation."""
    require(not os.path.lexists(path),'preserve the previous permission operation')
    raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return reference(path)


def actor(capability):
    """Require the explicit root-session action capability under sl5nw, never UID zero."""
    require(capability==CAPABILITY and os.getuid()==os.geteuid()==OWNER and os.getresuid()==(OWNER,OWNER,OWNER)
        and pwd.getpwuid(OWNER).pw_name=='sl5nw','only the explicit owner-session action is allowed')


def publication(commit,review_path,run):
    """Require actual source and independent review in the recorded shared origin branch."""
    require(type(commit) is str and re.fullmatch('[0-9a-f]{40}',commit),'an actual full publication is required')
    environment={key:value for key,value in os.environ.items() if not key.startswith('GIT_')}
    environment.update(GIT_OPTIONAL_LOCKS='0',GIT_TERMINAL_PROMPT='0')
    def git(root,*arguments):
        return subprocess.check_output(['git','-C',str(root),*arguments],env=environment,stderr=subprocess.PIPE,timeout=15)
    integration=run.parents[2]
    require(git(integration,'config','--get-all','remote.origin.url')==(ORIGIN+'\n').encode()
        and git(integration,'config','--get-all','remote.origin.fetch')==b'+refs/heads/*:refs/remotes/origin/*\n','the shared origin changed')
    git(integration,'merge-base','--is-ancestor',commit,'refs/remotes/origin/merge-atari-cleanrl')
    review=document(review_path)
    require(review.get('kind')=='independent_metadata_permission_utility_source_review'
        and review.get('approved') is review.get('passed') is review.get('independent_of_author') is True
        and review.get('source_sha256')==reference(Path(__file__).resolve())['sha256'],'the exact permission source needs independent approval')
    for path in (Path(__file__).resolve(),Path(review_path).resolve(strict=True)):
        root=Path(git(path.parent,'rev-parse','--show-toplevel').decode().strip())
        require(git(integration,'cat-file','blob',commit+':'+str(path.relative_to(root)))==capture(path)[0],
            'source or independent approval was not published with these bytes')
    return {'publication':commit,'source':reference(Path(__file__).resolve()),'independent_source_review':reference(Path(review_path))}


def inputs(run,packet,controllers):
    """Freeze only actual reviewed controller/capture plans; future units are refused."""
    run,packet=Path(run).absolute(),Path(packet).absolute()
    require(run.resolve(strict=True)==run and packet.parent==run/'remote_setup' and packet.resolve(strict=True)==packet,
        'the existing exact campaign packet is required')
    for name in ('manifest.yaml','config_resolved.yaml','command.txt','experiment_background.md'):capture(packet/name)
    require(1<=len(controllers)<=4 and len(controllers)==len(set(map(str,controllers))),'declare one to four actual controllers exactly once')
    rows=[]
    for controller in controllers:
        controller=Path(controller).absolute();require(controller.parent==run/'data' and controller.resolve(strict=True)==controller,'an actual controller unit is required')
        parent_ref=reference(controller/'measurement_plan.json');parent=document(controller/'measurement_plan.json')
        binding_ref=reference(controller/'prepared_child_binding.json');binding=document(controller/'prepared_child_binding.json')
        child=Path(parent['profile_unit']);child_plan=checked(binding['child_plan'])
        require(not os.path.lexists(run/'remote_setup'/child.name/'metadata_permission_no_submit_abort.json'),
            'an explicitly aborted prepared unit cannot be launched or protected again')
        review_ref=reference(controller/'controller_launch_independent_review.json');review=checked(review_ref)
        require(parent.get('kind')=='profile_target_qualification_controller' and parent.get('protocol')==PROTOCOL
            and parent['campaign_run_dir']==str(run) and child.parent==run/'data'
            and child_plan['kind']=='full_original_profile_source_target_qualification'
            and binding['controller_plan']==parent_ref and review.get('approved') is review.get('passed') is True
            and review.get('kind')=='independent_profile_target_qualification_controller_launch_review'
            and review.get('independent_of_producer') is True and review.get('controller_plan')==parent_ref
            and review.get('child_plan')==binding['child_plan'],'actual immutable controller review differs')
        rows.append({'controller_plan':parent_ref,'child_plan':binding['child_plan'],'prepared_child_binding':binding_ref,'launch_review':review_ref})
    paths=[run/'queue/metadata.lock',run/'processes.lock']
    snapshots=[capture(path)[2] for path in paths]
    require(all(row['mode'] in (0o640,0o660) and row['bytes']<=65536 for row in snapshots),'metadata files must retain the explicit owner modes and bounded bytes')
    return run,packet,rows,snapshots


def same_file(current,original,mode):
    """Scope device/inode identity to the same acting host and preserve every original byte."""
    return all(current[key]==original[key] for key in ('path','hostname','device','inode','uid','gid','nlink','bytes','mtime_ns','sha256')) and current['mode']==mode


def restore_evidence(protection,evidence):
    """Require every frozen controller to have released its exact child or ended without one."""
    require(type(evidence) is dict and set(evidence)=={'controllers'} and type(evidence['controllers']) is list,
        'explicit actual restoration evidence is required')
    rows=evidence['controllers'];expected={row['controller_plan']['path']:row for row in protection['controllers']}
    require(len(rows)==len(expected) and {row['controller_plan']['path'] for row in rows}==set(expected),'every protected controller needs actual evidence')
    saved=[]
    for row in rows:
        frozen=expected[row['controller_plan']['path']];require(row['controller_plan']==frozen['controller_plan'],'a bound controller changed')
        controller=Path(row['controller_plan']['path']).parent;child=Path(frozen['child_plan']['path']).parent
        if row['mode']=='child_released':
            spawn_ref=reference(controller/'controller_spawn.json');spawn=checked(spawn_ref)
            release_ref=reference(child/'profile_target_controller_release.json');release=checked(release_ref)
            startup_ref=reference(child/'profile_target_child_startup_observations.json');startup=checked(startup_ref)
            binding=checked(frozen['prepared_child_binding'])
            last=startup['observations'][-1] if startup.get('observations') else {}
            require(release.get('kind')=='actual_profile_target_controller_child_release' and release['parent_plan']==frozen['controller_plan']
                and release['child_plan']==frozen['child_plan'] and release['controller_spawn']==spawn_ref
                and release['child']==spawn['child'] and release['controller']==spawn['controller']
                and type(spawn['actual_popen_pid']) is int and spawn['actual_popen_pid']>0
                and spawn['actual_popen_pid']==spawn['child']['pid'] and spawn['child']['parent_pid']==spawn['controller']['pid']
                and startup['exact_command_ready'] is True and startup['failure'] is None
                and startup['actual_popen_pid']==spawn['actual_popen_pid'] and startup['observations']
                and startup['expected_command']==spawn['command']==binding['command']
                and last.get('process')==spawn['child']
                and set(last.get('predicates',{}))=={'pid','birth','parent','uid','affinity','raw_hash','raw_decode'}
                and all(value is True for value in startup['observations'][-1]['predicates'].values())
                and startup['observations'][-1]['expected_raw_command'] is True,
                'actual exact child startup, spawn and release are all required')
            saved.append({'controller_plan':frozen['controller_plan'],'mode':'child_released','spawn':spawn_ref,'release':release_ref,'startup':startup_ref})
        elif row['mode']=='terminal_without_child':
            review=checked(row['terminal_review']);failure_ref=reference(controller/'controller_failure.json');failure=checked(failure_ref)
            require(review.get('kind')=='independent_no_child_qualification_controller_terminal_review'
                and review.get('approved') is review.get('passed') is review.get('independent_of_producer') is True
                and review.get('controller_plan')==frozen['controller_plan'] and review.get('child_plan')==frozen['child_plan']
                and review.get('failure')==failure_ref and review.get('controller_finished') is True
                and review.get('actual_child_started') is False and review.get('unresolved_anomalies')==[]
                and review.get('terminal_state') in {'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'}
                and failure.get('actual_popen_pid') is None and failure.get('actual_identity') is None
                and not os.path.lexists(controller/'controller_spawn.json')
                and not os.path.lexists(child/'profile_target_controller_release.json'),
                'a genuine independent no-child terminal review is required')
            require(reference(review['raw_terminal_evidence']['path'])==review['raw_terminal_evidence']
                and capture(review['raw_terminal_evidence']['path'])[0], 'the independently reviewed actual terminal bytes are required')
            saved.append({'controller_plan':frozen['controller_plan'],'mode':'terminal_without_child','terminal_review':row['terminal_review'],'failure':failure_ref})
        elif row['mode']=='aborted_before_submission_or_launch':
            packet=controller.parent.parent/'remote_setup'/child.name
            require(row.get('root_aborted_before_submission_or_launch') is True
                and not any(os.path.lexists(packet/name) for name in ('submission_intent.json','actual_submission_return.json',
                    'actual_submission_stdout.raw','submission_receipt.json','release_intent.json','release_receipt.json',
                    'root_remote_launch_intent.json','root_transport_popen_attempt.json','root_actual_controller_spawn.json',
                    'root_transport_failure.json','root_transport_custody_inconclusive.json'))
                and not any(os.path.lexists(controller/name) for name in ('controller_intent.json','controller_spawn.json',
                    'controller_failure.json','controller_wait.json','controller_inconclusive.json'))
                and not any(os.path.lexists(child/name) for name in ('owned_allocation_inputs.json',
                    'profile_target_child_startup_observations.json','profile_target_controller_release.json','progress.json','result.json','failure.json')),
                'an attempted or ambiguous submission/launch cannot use the no-submit abort route')
            saved.append({'controller_plan':frozen['controller_plan'],'mode':'aborted_before_submission_or_launch',
                'root_aborted_before_submission_or_launch':True,'packet':str(packet),'all_attempt_markers_absent':True})
        else:raise ValueError('a pending or unknown controller cannot permit restoration')
    return saved


def change(run,packet,controllers,mode,capability,commit,review_path,evidence_path=None):
    """Perform exactly one explicit permission action, never a GPU or process action."""
    actor(capability);run,packet,rows,current=inputs(run,packet,controllers)
    published=publication(commit,review_path,run)
    if mode=='protect':
        intent=packet/'metadata_permission_protection_intent.json';result=packet/'metadata_permission_protection.json'
        require(not os.path.lexists(intent) and not os.path.lexists(result),'never repeat a protection attempt')
        originals=current;evidence=None;targets=[0o640,0o640]
    else:
        protection=checked(reference(packet/'metadata_permission_protection.json'))
        require(protection['controllers']==rows and protection['publication']==commit,'the original protection/controller declaration changed')
        originals=protection['original_files'];require(all(same_file(row,old,0o640) for row,old in zip(current,originals)),'same protected files and bytes required')
        evidence=restore_evidence(protection,document(evidence_path))
        intent=packet/'metadata_permission_restoration_intent.json';result=packet/'metadata_permission_restoration.json'
        require(not os.path.lexists(intent) and not os.path.lexists(result),'never repeat a restoration attempt')
        targets=[row['mode'] for row in originals]
    descriptors=[];changed=[]
    record={'schema_version':2,'kind':'owned_metadata_permission_'+mode,'root_action_capability':CAPABILITY,
        'hostname':socket.gethostname(),'uid':os.getuid(),'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'controllers':rows,'original_files':originals,'restoration_evidence':evidence,**published,
        'standing_gpu_descriptor_opened':False,'scientific_credit':0,'admission_granted':False}
    write_once(intent,record)
    if mode=='restore' and any(row['mode']=='aborted_before_submission_or_launch' for row in evidence):
        write_once(packet/'metadata_permission_no_submit_abort.json',{
            'schema_version':2,'kind':'root_aborted_before_submission_or_launch','controllers':evidence,
            'root_action_capability':CAPABILITY,'publication':commit,'future_launch_of_aborted_units_permitted':False,
            'scientific_credit':0,'admission_granted':False})
    try:
        for row in current:
            fd=os.open(row['path'],os.O_RDWR|os.O_NOFOLLOW|os.O_NONBLOCK);descriptors.append(fd)
            require(capture(row['path'],fd)[2]==row,'file changed before its permission action')
        for fd,row,target in zip(descriptors,current,targets):
            os.fchmod(fd,target);os.fsync(fd)
            after=capture(row['path'],fd)[2]
            require(same_file(after,row,target),'permission operation changed file identity or bytes')
            changed.append(after)
        return write_once(result,{**record,'completed':True,'actual_files_after':changed,'automatic_timeout_restoration':False})
    except BaseException as error:
        write_once(packet/('metadata_permission_'+mode+'_failure.json'),{**record,'completed':False,
            'error':repr(error),'actual_files_changed':changed,'automatic_restoration_attempted':False})
        raise
    finally:
        for fd in descriptors:os.close(fd)


def main(argv=None):
    """Default to strict non-action preflight; root explicitly chooses any actual chmod."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--packet',type=Path,required=True)
    parser.add_argument('--controller-unit',type=Path,action='append',required=True)
    action=parser.add_mutually_exclusive_group();action.add_argument('--protect-once',action='store_true');action.add_argument('--restore-once',action='store_true')
    parser.add_argument('--root-action-capability');parser.add_argument('--publication-commit');parser.add_argument('--independent-source-review',type=Path)
    parser.add_argument('--restoration-evidence',type=Path);args=parser.parse_args(argv)
    if not args.protect_once and not args.restore_once:
        run,packet,rows,files=inputs(args.run_dir,args.packet,args.controller_unit)
        require(all(row['mode']==0o640 for row in files),'preflight refuses launch: protect both exact owned metadata files to0640 before submission/release')
        print(json.dumps({'passed':True,'action_performed':False,'controller_count':len(rows),'files':files,'admission_granted':False}));return
    require(args.independent_source_review is not None and (args.restore_once==(args.restoration_evidence is not None)),
        'real actions need actual source review; restoration also needs actual evidence')
    result=change(args.run_dir,args.packet,args.controller_unit,'restore' if args.restore_once else 'protect',
        args.root_action_capability,args.publication_commit,args.independent_source_review,args.restoration_evidence)
    print(json.dumps({'result':result,'admission_granted':False}))


if __name__=='__main__':main()
