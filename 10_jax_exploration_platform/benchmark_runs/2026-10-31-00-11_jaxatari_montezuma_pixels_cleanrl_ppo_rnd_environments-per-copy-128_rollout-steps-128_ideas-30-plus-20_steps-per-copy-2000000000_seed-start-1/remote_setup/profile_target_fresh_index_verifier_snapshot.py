"""Narrow fresh real-index window, composed with the preserved failed full proof.

The historical full proof remains FAILED. Its pinned checkpoint proves that the
5916 source/roster/protected-file checks preceded one final index-preservation
failure. This script repeats no source/payload hash work. Root alone invokes it.
All real-index Git reads finish BEFORE the late index-byte snapshot. Subsequent
Git operations use copies through explicit GIT_INDEX_FILE only.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import time

I = Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl')
R = I / '10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1'
OLD = R / 'remote_setup/profile_target_union_incoming_root_2026-11-05-11-29'
SOURCE = Path(__file__).with_name('verify_profile_target_incoming.py')
PINS = {'_checkpoint.json':'e482bf0b7f35bbdf148f757e8a88147fdbd1650e0a6fa3c248738a35265ca9f3',
    '_before.json':'4ca56d3186368535742bcf541641616299efcb4441e2ddda24d6a8627e6e163e',
    '_root_stderr.raw':'90641ea3b13d540a5584947e01e31d9fe3118987b1a057558b33d503801365dc',
    '_root_return.json':'5fe4bcd163b1e8f48cb06349501e47bbab780ec3cc53eda4b412727d8ec186aa'}
SOURCE_SHA = 'db682ac94893b8671d97068583456f377c1456c6c617d3adfbc749d7c3041411'
STAGES = '7cda055ccd42674b8bcb3a3f73c5d858773d175b743443ea8b74ae9ede5212f4'
ENV = dict(os.environ,GIT_OPTIONAL_LOCKS='0')


def identity(path):
    value=Path(path).lstat()
    return {'device':value.st_dev,'inode':value.st_ino,'uid':value.st_uid,'mode':stat.S_IMODE(value.st_mode),
        'file_type':stat.S_IFMT(value.st_mode),'size_bytes':value.st_size,'mtime_ns':value.st_mtime_ns,
        'ctime_ns':value.st_ctime_ns,'nlink':value.st_nlink}


def sha(path,maximum=16*1024*1024):
    path=Path(path)
    if not stat.S_ISREG(path.lstat().st_mode) or path.stat().st_size>maximum:raise ValueError('ordinary bounded metadata/index required')
    digest=hashlib.sha256();count=0
    with path.open('rb') as stream:
        while block:=stream.read(1024*1024):
            count+=len(block)
            if count>maximum:raise ValueError('file exceeded its metadata/index bound')
            digest.update(block)
    return digest.hexdigest()


def publish(path,value):
    raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    with Path(path).open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    descriptor=os.open(Path(path).parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)
    return {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-head',required=True)
    parser.add_argument('--output-prefix',type=Path,required=True)
    args=parser.parse_args();started=time.monotonic();checks=[]
    def check(label,value):
        if not value:raise ValueError(label)
        if time.monotonic()-started>285:raise TimeoutError('narrow proof exceeded its original285second bound')
        checks.append(label)
    check('actual current ownedCPUallocation',os.environ.get('SLURM_JOB_ID')=='6825983' and os.environ.get('SLURM_STEP_ID','').isdigit())
    check('actual exactowner',os.getuid()==os.geteuid()==2618919)
    check('matching campaign ledger owns currentCPU',(R/'slurm/submitted_jobids_uva.txt').read_text().splitlines().count('6825983')==1)
    check('fresh outputprefix',args.output_prefix.parent==R/'remote_setup' and args.output_prefix.is_absolute()
        and all(not os.path.lexists(str(args.output_prefix)+s) for s in ('_late_before.json','_checkpoint.json','_review.json')))
    check('exact historical full program retained',sha(SOURCE,2500000)==SOURCE_SHA)
    check('exact old finalfailure sourcepoint',SOURCE.read_text().splitlines()[288].strip()==
        "check('real index bytes identity and stage entries unchanged',real_after == real_before)")
    prior={}
    for suffix,wanted in PINS.items():
        path=Path(str(OLD)+suffix);check('exact preserved failedfullproof '+suffix,sha(path,2500000)==wanted)
        prior[suffix]=path.read_bytes()
    old=json.loads(prior['_checkpoint.json']);before=json.loads(prior['_before.json']);returned=json.loads(prior['_root_return.json'])
    stderr=prior['_root_stderr.raw'].decode()
    check('old full proof is FAILED and remains so',type(returned['returncode']) is int and returned['returncode']==1
        and 'ValueError: real index bytes identity and stage entries unchanged' in stderr and 'line 289, in main' in stderr)
    check('all prior5916checks reachedexactcheckpoint',old['check_count']==5916 and before['protected_file_count']==662
        and before['frozen_static_file_count']==248 and before['index_before']==old['index_before'])
    check('old index bytesidentity reallydiffered; causestillunproved',old['index_before']!=old['index_after']
        and old['index_before']['stage_entry_sha256']==old['index_after']['stage_entry_sha256']==STAGES)
    check('same confirmed integrationHEAD',old['head']==before['head']==args.expected_head)
    handoff_path=Path(before['author_handoff']['path'])
    check('same actualauthorhandoff bytes',sha(handoff_path,2500000)==before['author_handoff']['sha256'])
    handoff=json.loads(handoff_path.read_bytes());paths=set(handoff['files'])
    check('same actual126path authorcommit',handoff['commit']=='d1469fd063a44167ec56760729eb0beea7ecf7d3' and len(paths)==126)
    def git(*arguments,private=None,allowed_failure=False):
        available=285-(time.monotonic()-started)
        if available<=0:raise TimeoutError('narrow proof deadlineexpired')
        result=subprocess.run(['git','-c','core.preloadIndex=false','-c','core.fsmonitor=false',*arguments],cwd=I,
            env=ENV if private is None else dict(ENV,GIT_INDEX_FILE=str(private)),capture_output=True,timeout=min(60,available))
        if result.returncode and not allowed_failure:raise subprocess.CalledProcessError(result.returncode,result.args,result.stdout,result.stderr)
        return result
    def split(raw):return [part for part in raw.decode().split('\0') if part]
    check('fresh actualHEAD beforelatewindow',git('rev-parse','HEAD').stdout.decode().strip()==args.expected_head)
    check('fresh actualauthorobject identities',git('rev-parse',handoff['commit']+'^').stdout.decode().strip()==handoff['parent']
        and git('rev-parse',handoff['commit']+'^{tree}').stdout.decode().strip()==handoff['tree'])
    predicted=old['predicted_tree']
    check('same existing predictedtree object',git('rev-parse',predicted+'^{tree}').stdout.decode().strip()==predicted)
    check('fresh no staged edits',git('diff','--cached','--name-only','-z').stdout==b'')
    stages_before=hashlib.sha256(git('ls-files','--stage','-z').stdout).hexdigest()
    check('freshstageentries equal oldstageentries',stages_before==STAGES)
    dirty=split(git('diff','--name-only','--no-ext-diff','--no-textconv','-z','HEAD').stdout)
    untracked=split(git('ls-files','--others','--exclude-standard','-z').stdout)
    check('fresh actualincoming disjointdirty/untracked',not paths.intersection(dirty+untracked))
    tracked=set()
    for entry in git('ls-tree','-r','-z',args.expected_head).stdout.split(b'\0'):
        if entry:tracked.add(entry.split(b'\t',1)[1].decode())
    for name in sorted(paths):
        target=I/name
        check('fresh ignored/untracked collision absent '+name,name in tracked or not os.path.lexists(target))
        parent=target.parent
        while parent!=I:
            if os.path.lexists(parent):check('fresh incoming parent ordinary '+str(parent),parent.is_dir() and not parent.is_symlink())
            parent=parent.parent
    changed=split(git('diff','--name-only','--no-renames','-z',args.expected_head,predicted).stdout)
    check('fresh predictedtree changesonlyexactincoming',set(changed)<=paths)
    index=Path(git('rev-parse','--git-path','index').stdout.decode().strip());index=index if index.is_absolute() else I/index
    refs=[Path(git('rev-parse','--git-path','HEAD').stdout.decode().strip())]
    symbolic=git('symbolic-ref','-q','HEAD',allowed_failure=True)
    check('HEAD is known symbolic or detached state',symbolic.returncode in (0,1))
    if symbolic.returncode==0:refs.append(Path(git('rev-parse','--git-path',symbolic.stdout.decode().strip()).stdout.decode().strip()))
    refs.append(Path(git('rev-parse','--git-path','packed-refs').stdout.decode().strip()))
    refs=[p if p.is_absolute() else I/p for p in refs]
    def ref_snapshot():return {str(p):{'identity':identity(p),'sha256':sha(p)} if p.exists() else {'absent':True} for p in refs}
    ref_before=ref_snapshot()
    def index_snapshot():
        first=identity(index);digest=sha(index)
        check('realindexstable duringnonGitsnapshot',identity(index)==first)
        return {'identity':first,'sha256':digest}
    # No more real-index Git reads after this point. All following Git commands
    # receive explicit private copies, including the final stage-entry read.
    late_before=index_snapshot()
    late_ref=publish(str(args.output_prefix)+'_late_before.json',{'head':args.expected_head,'predicted_tree':predicted,
        'real_index':late_before,'stage_entries_from_last_real_Git_read':stages_before,'head_ref_files':ref_before,
        'historical_fullproof_remains_failed':True,'reused_completed_full_checks':5916,
        'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    with tempfile.TemporaryDirectory(prefix='rnd_narrow_index_window_') as temporary:
        root=Path(temporary)
        for number in (1,2):
            copied=root/('actual-index-copy-'+str(number));shutil.copy2(index,copied)
            check('privatecopy exactlate realindex '+str(number),sha(copied)==late_before['sha256'])
            git('read-tree','-m','-u','--dry-run',args.expected_head,predicted,private=copied)
        after_dryruns=index_snapshot()
        post=root/'post-stage-only.index';shutil.copy2(index,post)
        check('poststagecopy exactactualindex',sha(post)==after_dryruns['sha256'])
        stages_after=hashlib.sha256(git('ls-files','--stage','-z',private=post).stdout).hexdigest()
    final=index_snapshot();ref_after=ref_snapshot()
    checkpoint={'head':args.expected_head,'predicted_tree':predicted,'real_index_late_before':late_before,
        'real_index_after_dryruns':after_dryruns,'real_index_final':final,'stage_entry_sha256_before':stages_before,
        'stage_entry_sha256_after_private_copy':stages_after,'HEAD_refs_before':ref_before,'HEAD_refs_after':ref_after,
        'fresh_copied_index_dryruns_completed':2,'historical_fullproof_remains_failed':True,'check_count':len(checks)}
    checkpoint_ref=publish(str(args.output_prefix)+'_checkpoint.json',checkpoint)
    check('fresh lateindex bytesandidentity unchanged',late_before==after_dryruns==final)
    check('fresh exactstageentry equivalence withoutrealindexGitread',stages_after==stages_before==STAGES)
    check('actualHEAD reffiles unchanged duringlatewindow',ref_after==ref_before)
    record={'schema_version':1,'kind':'preparatory_author_fresh_incoming_index_window_check','passed':True,
        'historical_fullproof_status':'FAILED; not relabeled; index replacement cause unproved',
        'historical_fullproof_reused_completed_checks':5916,'historical_fullproof_program':{'path':str(SOURCE),'sha256':SOURCE_SHA},
        'historical_checkpoint':{'path':str(OLD)+'_checkpoint.json','sha256':PINS['_checkpoint.json']},
        'historical_failure_return':{'path':str(OLD)+'_root_return.json','sha256':PINS['_root_return.json']},
        'historical_failure_stderr':{'path':str(OLD)+'_root_stderr.raw','sha256':PINS['_root_stderr.raw']},
        'late_before':late_ref,'checkpoint':checkpoint_ref,'integration_head':args.expected_head,
        'author_commit':handoff['commit'],'author_tree':handoff['tree'],'predicted_merge_tree':predicted,
        'exact_incoming_paths':126,'incoming_disjoint_fresh_dirty_untracked_and_ignored_paths':True,
        'real_index_late_before':late_before,'real_index_after':final,'stage_entry_sha256':stages_after,
        'fresh_actual_copied_real_index_dry_runs':2,'real_index_Git_reads_after_late_snapshot':0,
        'protected_source_payload_checks_repeated':False,'H1largepayload_or5_67GB_capture_read':False,
        'not_independent_source_or_scientific_approval':True,'real_merge_push_submit_GPU_or_remote_actions':0,
        'GIT_OPTIONAL_LOCKS_every_Git_call':'0','job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],
        'pid':os.getpid(),'uid':os.getuid(),'checks':checks,'check_count':len(checks),
        'seconds':time.monotonic()-started,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    out=publish(str(args.output_prefix)+'_review.json',record)
    print(json.dumps({'passed':True,'proof':out,'predicted_tree':predicted,'checks':len(checks),'seconds':record['seconds']}))


if __name__=='__main__':main()
