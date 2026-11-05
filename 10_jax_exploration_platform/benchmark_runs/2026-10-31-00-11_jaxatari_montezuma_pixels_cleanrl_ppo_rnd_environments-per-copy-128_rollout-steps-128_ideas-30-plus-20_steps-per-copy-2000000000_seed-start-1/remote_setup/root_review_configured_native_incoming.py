"""Verify the exact reviewed incoming paths and real private-index merge checks."""
import ast,datetime,hashlib,json,os,pathlib,shutil,subprocess,tempfile,time
I=pathlib.Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl');R=I/'10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1'
INCOMING='c228cfc402b357ad9cf14ef1a7e0134eebdcf35b';EXPECTED_HEAD='d4221c2373847cede7b8681bf5f240867e392a2d';started=time.monotonic()
def git(*args,input=None,env=None):
    """Call one exact Git operation without a shell."""
    return subprocess.check_output(['git',*args],cwd=I,input=input,env=env)
def optional_blob(revision,path):
    """Read an existing committed file, keeping absence separate from empty bytes."""
    result=subprocess.run(['git','cat-file','-e',revision+':'+path],cwd=I,capture_output=True)
    return git('show',revision+':'+path) if result.returncode==0 else None
head=git('rev-parse','HEAD').decode().strip();assert head==EXPECTED_HEAD
assert subprocess.run(['git','diff','--cached','--quiet'],cwd=I).returncode==0
base=git('merge-base',head,INCOMING).decode().strip();paths=git('diff','--name-only','-z',base,INCOMING).decode().split('\0')[:-1];assert len(paths)==74
frozen=R/'remote_setup/direct_nsight_incoming_independent_2026-11-04-20-01-56_protected_before.json';assert hashlib.sha256(frozen.read_bytes()).hexdigest()=='3412f310f74e010aab08548a77813f811ad51e5a5b5ae9db1798ee21afe77430'
protected=json.loads(frozen.read_bytes())['files']
def verify_protected():
    """Verify every protected actual file without importing its code."""
    for name,row in protected.items():
        p=pathlib.Path(name);s=p.stat();assert s.st_size==row['size_bytes'] and s.st_mode&0o777==row['mode'],name
        assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],name
verify_protected();dirty=set(git('diff','--name-only','-z').decode().split('\0'));assert not dirty.intersection(paths)
handoff=pathlib.Path('/p/rlprojects/RND_worktrees/2026-11-05-06-20_a4500-native-process-control_python-3.12_total-seconds-300_work-seconds-240_cleanup-seconds-60_admission-closed/.review_metadata/configured_native_process/publication_handoff_actual.json')
assert hashlib.sha256(handoff.read_bytes()).hexdigest()=='1e8892146887479a0bc5387f2d5a30f14868dbdafec661562768971947ce8703'
roster=json.loads(handoff.read_bytes());assert set(paths)==set(roster['files']) and roster['commit']==INCOMING
for path,row in roster['files'].items():
    raw=git('show',INCOMING+':'+path);raw.decode('utf-8');assert hashlib.sha256(raw).hexdigest()==row['sha256'] and len(raw)==row['size_bytes'] and len(raw)<2000000
    assert git('ls-tree',INCOMING,'--',path).decode().split()[:2]==[row['mode'],'blob']
freeze=json.loads(git('show',INCOMING+':.review_metadata/configured_native_process/corrected_union_source_freeze.json'))
for path,row in freeze['files'].items():assert hashlib.sha256(git('show',INCOMING+':'+path)).hexdigest()==row['sha256']
review=git('show',INCOMING+':.review_metadata/configured_native_process/independent_corrected_union_source_and_proposal_review.json')
assert hashlib.sha256(review).hexdigest()=='61786869da556246513c58e847b05acbb6d49628c90dd77de14315dd28ae5fc4' and json.loads(review)['approved'] is True
profile='10_jax_exploration_platform/scripts/atari_pixel_campaign/profile_selected.py'
def core_at(revision):
    tree=ast.parse(git('show',revision+':'+profile));return ast.dump(next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='_capture_core'),include_attributes=False)
assert core_at(head)==core_at(INCOMING)
next_actions=json.loads((R/'next_actions.json').read_bytes());live=next_actions['h100_common_timing_controller_preparation']['copy_one_source_freeze']
live_path=pathlib.Path(live['path']);assert hashlib.sha256(live_path.read_bytes()).hexdigest()==live['sha256']
live_files=json.loads(live_path.read_bytes())['files']
for p in live_path.parent.iterdir():
    if p.is_file() and p.suffix in {'.py','.sh'}:live_files.setdefault(str(p),hashlib.sha256(p.read_bytes()).hexdigest())
def verify_live():
    for name,digest in live_files.items():assert hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()==digest,name
verify_live()

merges=[]
with tempfile.TemporaryDirectory(prefix='rnd_initial_identity_incoming_') as temp:
    temp=pathlib.Path(temp);env=dict(os.environ,GIT_INDEX_FILE=str(temp/'predicted.index'));git('read-tree',head,env=env)
    for path in paths:
        target=I/path;current=optional_blob(head,path);original=optional_blob(base,path);incoming=optional_blob(INCOMING,path);assert incoming is not None
        assert current is not None or not target.exists(),('incoming untracked/ignored collision',path)
        if current==original:result=incoming
        elif incoming==current:result=current
        elif original is not None and current is not None:
            files=[]
            for name,raw in [('current',current),('base',original),('incoming',incoming)]:
                f=temp/name;f.write_bytes(raw);files.append(str(f))
            result=subprocess.check_output(['git','merge-file','-p',*files]);merges.append(path)
        else:raise AssertionError(('independent additions differ',path))
        mode=git('ls-tree',INCOMING,'--',path).decode().split()[0];blob=git('hash-object','-w','--stdin',input=result).decode().strip();git('update-index','--add','--cacheinfo',mode+','+blob+','+path,env=env)
    tree=git('write-tree',env=env).decode().strip();actual_index=pathlib.Path(git('rev-parse','--git-path','index').decode().strip())
    if not actual_index.is_absolute():actual_index=I/actual_index
    for index in (1,2):
        private=temp/('dry-run-'+str(index)+'.index');shutil.copy2(actual_index,private);private_env=dict(os.environ,GIT_INDEX_FILE=str(private));git('read-tree','-m','-u','--dry-run',head,tree,env=private_env)
verify_protected();verify_live();assert git('rev-parse','HEAD').decode().strip()==head
record={'kind':'root_independent_configured_native_incoming_review','passed':True,'incoming_commit':INCOMING,'integration_head':head,'base':base,'predicted_merge_tree':tree,'changed_paths':paths,'automatic_three_way_paths':merges,'actual_private_index_read_tree_dry_runs':2,'protected_file_count':len(protected),'live_timing_static_file_count':len(live_files),'reviewed_source_freeze_count':len(freeze['files']),'all_incoming_paths_match_author_roster':True,'numerical_capture_function_unchanged':True,'dirty_and_untracked_ignored_collision_checks':True,'scientific_source_changed':False,'gpu_actions':0,'job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],'wall_seconds':time.monotonic()-started,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
p=R/'remote_setup/configured_native_incoming_root_2026-11-05.json'
with p.open('x') as f:json.dump(record,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps({key:record[key] for key in ['passed','incoming_commit','predicted_merge_tree','protected_file_count','actual_private_index_read_tree_dry_runs','wall_seconds']}))
