"""Verify the exact reviewed incoming paths and real private-index merge checks."""
import datetime,hashlib,json,os,pathlib,shutil,subprocess,tempfile,time
I=pathlib.Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl');R=I/'10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1'
INCOMING='d2253fa853290cb8fcd0a9212e332dd33a63f2f9';EXPECTED_HEAD='18c7fcba26903d59054527b21bfbe156592ac6b3';started=time.monotonic()
def git(*args,input=None,env=None):
    """Call one exact Git operation without a shell."""
    return subprocess.check_output(['git',*args],cwd=I,input=input,env=env)
def optional_blob(revision,path):
    """Read an existing committed file, keeping absence separate from empty bytes."""
    result=subprocess.run(['git','cat-file','-e',revision+':'+path],cwd=I,capture_output=True)
    return git('show',revision+':'+path) if result.returncode==0 else None
head=git('rev-parse','HEAD').decode().strip();assert head==EXPECTED_HEAD
assert subprocess.run(['git','diff','--cached','--quiet'],cwd=I).returncode==0
base=git('merge-base',head,INCOMING).decode().strip();paths=git('diff','--name-only','-z',base,INCOMING).decode().split('\0')[:-1];assert len(paths)==17
frozen=R/'remote_setup/direct_nsight_incoming_independent_2026-11-04-20-01-56_protected_before.json';assert hashlib.sha256(frozen.read_bytes()).hexdigest()=='3412f310f74e010aab08548a77813f811ad51e5a5b5ae9db1798ee21afe77430'
protected=json.loads(frozen.read_bytes())['files']
def verify_protected():
    """Verify every protected actual file without importing its code."""
    for name,row in protected.items():
        p=pathlib.Path(name);s=p.stat();assert s.st_size==row['size_bytes'] and s.st_mode&0o777==row['mode'],name
        assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],name
verify_protected();dirty=set(git('diff','--name-only','-z').decode().split('\0'));assert not dirty.intersection(paths)
roster=pathlib.Path('/p/rlprojects/RND_worktrees/2026-11-05-05-32_empty-command-startup_wait-seconds-30_preserve-owner-birth-parent-affinity_exact-argv-before-registration_gpu-count-0/.review_metadata/empty_command_startup/publication_roster.json');assert hashlib.sha256(roster.read_bytes()).hexdigest()=='02e66653442845c4cf12ac52acb383e80e9ee8fc846087086a38ba3e6a392778'
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
verify_protected();assert git('rev-parse','HEAD').decode().strip()==head
record={'kind':'root_independent_empty_only_startup_incoming_review','passed':True,'incoming_commit':INCOMING,'integration_head':head,'base':base,'predicted_merge_tree':tree,'changed_paths':paths,'automatic_three_way_paths':merges,'actual_private_index_read_tree_dry_runs':2,'protected_file_count':len(protected),'dirty_and_untracked_ignored_collision_checks':True,'scientific_source_changed':False,'gpu_actions':0,'job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],'wall_seconds':time.monotonic()-started,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
p=R/'remote_setup/empty_only_startup_incoming_root_2026-11-05.json'
with p.open('x') as f:json.dump(record,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps({key:record[key] for key in ['passed','incoming_commit','predicted_merge_tree','protected_file_count','actual_private_index_read_tree_dry_runs','wall_seconds']}))
