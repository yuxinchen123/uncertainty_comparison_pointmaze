"""Prepare a mechanically checked incoming tree without touching the real index/worktree.

Lineage: campaign remote_setup/independent_configured_profile_preparation_incoming_2026-11-05.py.
This adaptation is preparatory engineering by a source author, not independent
scientific/source approval. Root alone invokes it after confirming a stable HEAD.
Every Git subprocess disables optional locks. Two real-index copies receive the
actual read-tree -m -u --dry-run checks; the real index is never a write target.
"""
import argparse
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tempfile
import time

I = Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl')
R = I / '10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1'
BASE = '548cbcaef4c1fc28f80817ec9ba93f1e6b2fee4e'
BASE_PROGRAM = R / 'remote_setup/independent_configured_profile_preparation_incoming_2026-11-05.py'
PROTECTED = R / 'remote_setup/direct_nsight_incoming_independent_2026-11-04-20-01-56_protected_before.json'
PROTECTED_SHA = '3412f310f74e010aab08548a77813f811ad51e5a5b5ae9db1798ee21afe77430'
APPROVAL_SHA = '34cb02a3da0df49db2a5dc40f934e18888034c8c113a852a857dc84e45991fae'
FREEZE_SHA = 'dee047f4bd891c1523382658d8726c8ff360dc53f24f189bd6c2e3085e697d8f'
H1_REVIEW_SHA = '901a5805d7f768cd3b3159fe7e478fffdbee36280c2144f195ada1b3471bdf22'
LIMIT = 2500000
ENV = dict(os.environ, GIT_OPTIONAL_LOCKS='0')


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def identity(path):
    info = Path(path).lstat()
    return {'device':info.st_dev, 'inode':info.st_ino, 'uid':info.st_uid, 'mode':stat.S_IMODE(info.st_mode),
        'file_type':stat.S_IFMT(info.st_mode), 'size_bytes':info.st_size, 'mtime_ns':info.st_mtime_ns,
        'ctime_ns':info.st_ctime_ns, 'nlink':info.st_nlink}


def digest_file(path, maximum=None):
    """Hash one pinned ordinary descriptor; never call this for the accepted H1 payload."""
    path = Path(path)
    if path.resolve(strict=True) != path or not stat.S_ISREG(path.lstat().st_mode):raise ValueError('not an ordinary exact file: ' + str(path))
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or maximum is not None and before.st_size > maximum:raise ValueError('file exceeds read scope: ' + str(path))
        count, value = 0, hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:break
            count += len(block)
            if maximum is not None and count > maximum:raise ValueError('file grew beyond read scope')
            value.update(block)
        after, named = os.fstat(descriptor), path.lstat()
        fields = ('st_dev','st_ino','st_uid','st_mode','st_size','st_mtime_ns','st_ctime_ns')
        if count != before.st_size or any(getattr(before,k) != getattr(after,k) or getattr(before,k) != getattr(named,k) for k in fields):
            raise ValueError('file changed during pinned read: ' + str(path))
        return value.hexdigest()
    finally:os.close(descriptor)


def bounded_json(path, expected=None):
    path = Path(path)
    if path.stat().st_size > LIMIT:raise ValueError('metadata exceeds 2.5MB: ' + str(path))
    wanted = digest_file(path, LIMIT)
    raw = path.read_bytes()
    if sha(raw) != wanted or expected is not None and wanted != expected:raise ValueError('metadata changed or hash differs: ' + str(path))
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:raise ValueError('duplicate metadata key')
            out[key] = value
        return out
    return json.loads(raw, object_pairs_hook=unique, parse_constant=lambda value:(_ for _ in ()).throw(ValueError(value)))


def publish(path, value):
    raw = encoded(value)
    with Path(path).open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    descriptor = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)
    return {'path':str(path),'sha256':sha(raw)}


def relative(name):
    path = PurePosixPath(name)
    return (type(name) is str and not path.is_absolute() and str(path) == name and name not in ('','.','..')
        and '..' not in path.parts and '.git' not in path.parts and all(ord(c) >= 32 for c in name))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-head', required=True)
    parser.add_argument('--author-handoff', type=Path, required=True)
    parser.add_argument('--author-handoff-sha256', required=True)
    parser.add_argument('--output-prefix', type=Path, required=True)
    parser.add_argument('--maximum-seconds', type=int, default=285)
    args = parser.parse_args()
    started = time.monotonic();checks=[]
    def check(label, value):
        if not value:raise ValueError(label)
        if time.monotonic() - started > args.maximum_seconds:raise TimeoutError('incoming proof exceeded its original total bound')
        checks.append(label)
    check('bounded current CPU proof budget',0 < args.maximum_seconds <= 285)
    check('actual new owned CPU allocation',os.environ.get('SLURM_JOB_ID') == '6825983' and os.environ.get('SLURM_STEP_ID','').isdigit())
    check('exact reviewer UID',os.getuid() == os.geteuid() == 2618919)
    check('matching campaign ledger owns actual CPU allocation',(R/'slurm/submitted_jobids_uva.txt').read_text().splitlines().count('6825983') == 1)
    check('explicit actual expected HEAD',bool(re.fullmatch('[0-9a-f]{40}',args.expected_head)))
    check('explicit actual handoff hash',bool(re.fullmatch('[0-9a-f]{64}',args.author_handoff_sha256)))
    check('fresh campaign proof prefix',args.output_prefix.is_absolute() and args.output_prefix.parent == R/'remote_setup'
        and all(not os.path.lexists(str(args.output_prefix)+ending) for ending in ('_before.json','_checkpoint.json','_review.json','_failure.json')))
    def git(*arguments, private=None, data=None):
        environment = ENV if private is None else dict(ENV, GIT_INDEX_FILE=str(private))
        available = args.maximum_seconds - (time.monotonic() - started)
        if available <= 0:raise TimeoutError('the original proof deadline expired')
        return subprocess.check_output(['git','-c','core.preloadIndex=false','-c','core.fsmonitor=false',*arguments],
            cwd=I,env=environment,input=data,stderr=subprocess.PIPE,timeout=min(60,available))
    def split(raw):return [part for part in raw.decode('utf-8').split('\0') if part]
    def blob(revision,path):
        query = subprocess.run(['git','-c','core.preloadIndex=false','-c','core.fsmonitor=false','cat-file','-e',revision+':'+path],
            cwd=I,env=ENV,capture_output=True,timeout=min(30,args.maximum_seconds-(time.monotonic()-started)))
        return git('show',revision+':'+path) if query.returncode == 0 else None
    def index_snapshot(path):
        before=identity(path);value=digest_file(path)
        check('real index stable while snapshotting',identity(path) == before)
        return {'identity':before,'sha256':value,'stage_entry_sha256':sha(git('ls-files','--stage','-z'))}
    check('fresh integration HEAD matches root confirmation',git('rev-parse','HEAD').decode().strip() == args.expected_head)
    check('no preexisting staged changes',git('diff','--cached','--name-only','-z') == b'')
    index=Path(git('rev-parse','--git-path','index').decode().strip());index=index if index.is_absolute() else I/index
    real_before=index_snapshot(index)
    handoff=bounded_json(args.author_handoff,args.author_handoff_sha256)
    for key in ('commit','parent','tree'):check('exact author '+key,type(handoff.get(key)) is str and bool(re.fullmatch('[0-9a-f]{40}',handoff[key])))
    check('exact original detached source base',handoff['parent'] == BASE)
    check('actual author parent',git('rev-parse',handoff['commit']+'^').decode().strip() == handoff['parent'])
    check('actual author tree',git('rev-parse',handoff['commit']+'^{tree}').decode().strip() == handoff['tree'])
    base=git('merge-base',args.expected_head,handoff['commit']).decode().strip()
    check('actual common base is approved548',base == BASE)
    paths=split(git('diff','--name-only','--no-renames','-z',base,handoff['commit']))
    check('complete exact author path map',type(handoff.get('files')) is dict and set(paths) == set(handoff['files'])
        and len(paths) == len(set(paths)) and len(paths) > 0)
    tree_rows={}
    for row in git('ls-tree','-r','-z',handoff['commit']).split(b'\0'):
        if row:
            fields,name=row.split(b'\t',1);mode,kind,oid=fields.decode().split();tree_rows[name.decode()] = (mode,kind,oid)
    incoming={};total=0
    for name in paths:
        check('safe relative incoming path '+name,relative(name))
        row=handoff['files'][name]
        check('typed exact incoming metadata '+name,type(row) is dict and set(row) == {'sha256','size_bytes','git_blob','git_mode'}
            and type(row['size_bytes']) is int and 0 <= row['size_bytes'] <= LIMIT
            and type(row['sha256']) is str and bool(re.fullmatch('[0-9a-f]{64}',row['sha256']))
            and type(row['git_blob']) is str and bool(re.fullmatch('[0-9a-f]{40}',row['git_blob']))
            and row['git_mode'] in ('100644','100755'))
        check('actual Git mode and blob '+name,tree_rows.get(name) == (row['git_mode'],'blob',row['git_blob']))
        raw=git('show',handoff['commit']+':'+name);raw.decode('utf-8',errors='strict')
        check('exact bounded UTF8 incoming bytes '+name,len(raw) == row['size_bytes'] and sha(raw) == row['sha256']
            and b'\0' not in raw and Path(name).suffix.lower() not in ('.so','.pyc','.tar','.pdf','.png','.bin','.whl','.npy','.npz','.pkl','.sqlite','.db'))
        incoming[name]=raw;total+=len(raw)
    check('all incoming source and UTF8 proof inputs together <=2.5MB',total <= LIMIT)
    review_ref=handoff['independent_review'];freeze_ref=handoff['source_freeze']
    check('exact genuine independent approval reference',review_ref['sha256'] == APPROVAL_SHA and relative(review_ref['relative_path']))
    check('exact reviewed union source freeze reference',freeze_ref['sha256'] == FREEZE_SHA and relative(freeze_ref['relative_path']))
    review_raw=incoming[review_ref['relative_path']];freeze_raw=incoming[freeze_ref['relative_path']]
    check('approval bytes really committed',sha(review_raw) == APPROVAL_SHA)
    review=json.loads(review_raw);freeze=json.loads(freeze_raw)
    check('genuine independent source-only approval',review.get('passed') is True and review.get('approved') is True
        and review.get('independent_of_source_author') is True and review.get('actual_qualification_or_ordinary_profile_admission_granted') is False)
    check('all17 frozen source/test bytes really committed',sha(freeze_raw) == FREEZE_SHA and len(freeze['runtime_and_test_files']) == 17
        and all(name in incoming and sha(incoming[name]) == row['sha256'] and len(incoming[name]) == row['size_bytes']
            for name,row in freeze['runtime_and_test_files'].items()))
    check('protected baseline exact checksum',digest_file(PROTECTED,LIMIT) == PROTECTED_SHA)
    protected=bounded_json(PROTECTED)['files'];check('exact662protectedentries',len(protected) == 662)
    actions=bounded_json(R/'next_actions.json')['h100_common_timing_controller_preparation']
    h1_packet=Path(actions['copy_one_request']['packet']);h2_packet=Path(actions['copy_two_request']['packet'])
    h1_result=R/'data'/h1_packet.name/'result.json';h1_stat=identity(h1_result)
    check('accepted H1 large payload STAT ONLY exact known size',h1_stat['size_bytes'] == 53225495)
    check('genuine accepted H1 timing review unchanged',digest_file(h1_packet/'independent_actual_common_timing_review.json',LIMIT) == H1_REVIEW_SHA)
    live={};large_stats={str(h1_result):h1_stat}
    for packet in (h1_packet,h2_packet):
        sf=packet/'controller_source_freeze.json';live[str(sf)]=digest_file(sf,LIMIT)
        frozen=bounded_json(sf)
        for name,row in frozen['files'].items():
            expected=row['sha256'] if isinstance(row,dict) else row
            if Path(name) == h1_result:continue
            check('frozen source/static input bounded '+name,Path(name).stat().st_size <= LIMIT)
            live[name]=expected
        unit=R/'data'/packet.name;plan_path=unit/'measurement_plan.json';plan=bounded_json(plan_path)
        live[str(plan_path)]=digest_file(plan_path,LIMIT)
        for label in ('core','paired_source','helpers'):
            for name,row in plan[label]['files'].items():
                expected=row['sha256'] if isinstance(row,dict) else row
                live[str(Path(plan[label]['source_root'])/name)]=expected
    live_identities={name:identity(name) for name in live}
    protected_identities={name:identity(name) for name in protected}
    def verify_static():
        for name,row in protected.items():
            value=identity(name)
            check('protected bytes/mode/size '+name,digest_file(name) == row['sha256']
                and value['size_bytes'] == row['size_bytes'] and value['mode'] == row['mode'])
        for name,wanted in live.items():check('frozen H1/H2/cba69fD28 input '+name,
            digest_file(name,LIMIT) == wanted and identity(name) == live_identities[name])
        for name,wanted in protected_identities.items():check('protected actual file identity unchanged '+name,identity(name) == wanted)
        check('accepted H1 payload identity/stat unchanged without reading bytes',identity(h1_result) == h1_stat)
    verify_static()
    dirty=split(git('diff','--name-only','--no-ext-diff','--no-textconv','-z','HEAD'))
    untracked=split(git('ls-files','--others','--exclude-standard','-z'))
    check('incoming disjoint from current dirty/untracked',not set(paths).intersection(dirty+untracked))
    def collision_check():
        for name in paths:
            target=I/name;current=blob(args.expected_head,name)
            check('incoming ignored/untracked path collision absent '+name,current is not None or not os.path.lexists(target))
            parent=target.parent
            while parent != I:
                if os.path.lexists(parent):check('incoming parent ordinary directory '+str(parent),parent.is_dir() and not parent.is_symlink())
                parent=parent.parent
    collision_check()
    def generated(path):
        absolute=I/path
        return absolute.is_relative_to(R/'data') or absolute.is_relative_to(R/'20_mins_monitoring')
    def preserve(path):
        absolute=I/path
        if not os.path.lexists(absolute):return {'exists':False}
        value={'exists':True,'identity':identity(absolute)}
        if absolute.is_symlink():value['symlink_target']=os.readlink(absolute)
        elif absolute.is_file() and absolute.stat().st_size <= LIMIT and absolute != h1_result:value['sha256']=digest_file(absolute,LIMIT)
        else:value['stat_only']=True
        return value
    preexisting={name:preserve(name) for name in sorted(set(dirty+untracked))}
    before={'kind':'preparatory_author_incoming_before','head':args.expected_head,'index_before':real_before,
        'author_handoff':{'path':str(args.author_handoff),'sha256':args.author_handoff_sha256},
        'protected_file_count':len(protected),'frozen_static_file_count':len(live),'H1_payload_STAT_ONLY':h1_stat,
        'dirty':dirty,'untracked':untracked,'preexisting':preexisting,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    publish(str(args.output_prefix)+'_before.json',before)
    automatic=[]
    with tempfile.TemporaryDirectory(prefix='rnd_profile_target_incoming_') as temporary:
        temp=Path(temporary);private=temp/'predicted.index';git('read-tree',args.expected_head,private=private)
        for name in paths:
            original=blob(base,name);current=blob(args.expected_head,name);wanted=incoming[name]
            if current == original:merged=wanted
            elif current == wanted:merged=current
            elif current is not None and original is not None:
                names=[]
                for label,raw in (('current',current),('base',original),('incoming',wanted)):
                    path=temp/label;path.write_bytes(raw);names.append(str(path))
                merged=subprocess.check_output(['git','merge-file','-p',*names],env=ENV,stderr=subprocess.PIPE,timeout=30);automatic.append(name)
            else:raise ValueError('different simultaneous additions: '+name)
            oid=git('hash-object','-w','--stdin',data=merged).decode().strip()
            git('update-index','--add','--cacheinfo',handoff['files'][name]['git_mode']+','+oid+','+name,private=private)
        predicted=git('write-tree',private=private).decode().strip()
        for number in (1,2):
            copied=temp/('copied-real-index-'+str(number));shutil.copy2(index,copied)
            git('read-tree','-m','-u','--dry-run',args.expected_head,predicted,private=copied)
    changed=split(git('diff','--name-only','--no-renames','-z',args.expected_head,predicted))
    check('predicted tree changes only exact author paths',set(changed) <= set(paths))
    for name,wanted in freeze['protected_existing_files'].items():
        for revision in (args.expected_head,handoff['commit'],predicted):check('AA7/original protected Git blob '+revision+':'+name,sha(blob(revision,name)) == wanted)
    for filename,function,wanted in [('profile_selected.py','_capture_core',freeze['capture_core_ast_sha256'])]:
        name='10_jax_exploration_platform/scripts/atari_pixel_campaign/'+filename
        for revision in (args.expected_head,handoff['commit'],predicted):
            node=next(node for node in ast.parse(blob(revision,name)).body if isinstance(node,ast.FunctionDef) and node.name == function)
            check('protected numerical function AST '+revision+':'+function,sha(ast.dump(node,include_attributes=False).encode()) == wanted)
    verify_static();collision_check()
    generated_changes=[]
    for name,value in preexisting.items():
        after=preserve(name)
        if after != value and generated(name):generated_changes.append({'path':name,'before':value,'after':after,'cause':'not determined; generated data/report may change without any merge'})
        else:check('unrelated preexisting static file unchanged '+name,after == value)
    final_dirty=split(git('diff','--name-only','--no-ext-diff','--no-textconv','-z','HEAD'))
    final_untracked=split(git('ls-files','--others','--exclude-standard','-z'))
    check('incoming also disjoint from final dirty/untracked',not set(paths).intersection(final_dirty+final_untracked))
    check('real integration HEAD unchanged after all dryruns',git('rev-parse','HEAD').decode().strip() == args.expected_head)
    real_after=index_snapshot(index)
    checkpoint={'head':args.expected_head,'predicted_tree':predicted,'index_before':real_before,'index_after':real_after,
        'generated_changes_observed_without_attributing_to_merge':generated_changes,'check_count':len(checks)}
    publish(str(args.output_prefix)+'_checkpoint.json',checkpoint)
    check('real index bytes identity and stage entries unchanged',real_after == real_before)
    result={'schema_version':1,'kind':'preparatory_author_profile_target_incoming_git_preservation_check','passed':True,
        'not_an_independent_source_or_scientific_approval':True,'actual_independent_source_review':review_ref,
        'source_lineage':{'path':str(BASE_PROGRAM),'sha256':digest_file(BASE_PROGRAM,LIMIT)},
        'integration_head':args.expected_head,'author_commit':handoff['commit'],'author_parent':handoff['parent'],'author_tree':handoff['tree'],
        'author_handoff':{'path':str(args.author_handoff),'sha256':args.author_handoff_sha256},'actual_changed_paths':paths,
        'incoming_UTF8_bytes':total,'incoming_paths':len(paths),'frozen_runtime_test_paths':len(freeze['runtime_and_test_files']),
        'predicted_merge_tree':predicted,'automatic_three_way_paths':automatic,'predicted_changed_paths':changed,
        'actual_copied_real_index_read_tree_dry_runs':2,'protected_files':len(protected),'frozen_H1_H2_static_files':len(live),
        'accepted_H1_large_payload_STAT_ONLY':h1_stat,'old_H1_payload_or5_67GB_capture_rehashed':False,
        'real_index_before':real_before,'real_index_after':real_after,'real_HEAD_unchanged':True,
        'real_index_or_worktree_write_commands_performed':False,'private_object_database_hash_write_only':True,
        'generated_changes_observed_without_attributing_to_merge':generated_changes,'dirty_before':dirty,'dirty_after':final_dirty,
        'untracked_before':untracked,'untracked_after':final_untracked,'GIT_OPTIONAL_LOCKS_for_every_Git_call':'0',
        'job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],'pid':os.getpid(),'uid':os.getuid(),
        'native_library_GPU_NUM_or_remote_operations':0,'merges_pushes_submissions':0,'ordinary_profile_admission_granted':False,
        'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'seconds':time.monotonic()-started,
        'check_count':len(checks),'checks':checks}
    ref=publish(str(args.output_prefix)+'_review.json',result)
    print(json.dumps({'passed':True,'proof':ref,'predicted_merge_tree':predicted,'paths':len(paths),'incoming_bytes':total,
        'checks':len(checks),'seconds':result['seconds'],'job_id':result['job_id'],'step_id':result['step_id']}))


if __name__ == '__main__':main()
