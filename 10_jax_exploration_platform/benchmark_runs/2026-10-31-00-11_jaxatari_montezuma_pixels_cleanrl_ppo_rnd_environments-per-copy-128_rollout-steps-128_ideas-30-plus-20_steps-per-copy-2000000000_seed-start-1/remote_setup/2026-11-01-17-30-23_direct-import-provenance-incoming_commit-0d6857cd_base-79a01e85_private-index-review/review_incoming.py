W="/p/rlprojects/RND_worktrees/2026-11-01-16-36_direct-qualification-metadata-import-provenance_preserve-frozen-numerical-helper-paths"
I="/p/rlprojects/RND_worktrees/merge-atari-cleanrl"
R="/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1"
D="/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-17-30-23_direct-import-provenance-incoming_commit-0d6857cd_base-79a01e85_private-index-review"
"""Verify one reviewed incoming commit using copies of the real index, without changing worktrees."""
import pathlib,subprocess,os,json,hashlib,stat,shutil,datetime,sys,traceback
w=pathlib.Path(W);i=pathlib.Path(I);r=pathlib.Path(R);d=pathlib.Path(D)
BASE='79a01e851ff92ddba7d0e12eaf8164cab2028878';HEAD='5f059abe1736680bbb4692ab65928b80841e062f';INCOMING='0d6857cdfcc4863ce764c67a4d8c694d03777972'
PREFIX='10_jax_exploration_platform/scripts/atari_pixel_campaign/'
PRODUCTION={PREFIX+'serval02_card1_binding.py',PREFIX+'tests/direct_metadata_import_order_probe.py',PREFIX+'tests/test_direct_metadata_import_provenance.py'}
commands=[]
def sha(raw):
 "Hash exact bytes for the review record."
 return hashlib.sha256(raw).hexdigest()
def read(path):
 "Read only an exact ordinary metadata or source file."
 p=pathlib.Path(path);assert p.is_absolute() and p.resolve(strict=True)==p and stat.S_ISREG(p.lstat().st_mode),str(p)
 return p.read_bytes()
def write(path,value):
 "Save a new durable review artifact without replacing prior evidence."
 raw=(json.dumps(value,indent=2,sort_keys=True)+'\n').encode()
 with pathlib.Path(path).open('xb') as out:out.write(raw);out.flush();os.fsync(out.fileno())
def git(args,root=i,index=None):
 "Run an explicit local Git query or a private-index operation."
 env=os.environ.copy();env['GIT_OPTIONAL_LOCKS']='0'
 if index is not None:env['GIT_INDEX_FILE']=str(index)
 command=['git','--no-optional-locks','-C',str(root),*args]
 out=subprocess.run(command,capture_output=True,env=env,timeout=90)
 commands.append({'command':command,'private_index':str(index) if index else None,'returncode':out.returncode,'stdout_bytes':len(out.stdout),'stdout_sha256':sha(out.stdout),'stderr':out.stderr.decode(errors='replace')})
 assert out.returncode==0,commands[-1]
 return out.stdout
def tree(ref):
 "Read the exact path, mode and blob map of one committed tree."
 entries={}
 for line in git(['ls-tree','-rz',ref]).split(b'\0'):
  if line:
   head,path=line.split(b'\t',1);mode,kind,blob=head.decode().split();assert kind in {'blob','commit'},(ref,path,kind)
   entries[path.decode()]={'mode':mode,'blob':blob}
 return entries
def snapshots(paths):
 "Record bytes and relevant file identity without using read access times."
 result={}
 for path in sorted(set(map(str,paths))):
  p=pathlib.Path(path);info=p.lstat()
  result[path]={'sha256':sha(read(p)),'mode':stat.S_IMODE(info.st_mode),'inode':info.st_ino,'device':info.st_dev,'size':info.st_size,'uid':info.st_uid,'gid':info.st_gid,'mtime_ns':info.st_mtime_ns,'ctime_ns':info.st_ctime_ns}
 return result
try:
 assert git(['rev-parse','HEAD']).decode().strip()==HEAD
 assert git(['rev-parse',INCOMING+'^']).decode().strip()==BASE
 assert git(['rev-parse',INCOMING+'^{tree}']).decode().strip()=='bca6001aebc8bebfbf28e957adb445b30a91934a'
 assert git(['rev-parse','refs/heads/direct-qualification-metadata-import-provenance-2026-11-01']).decode().strip()==INCOMING
 assert git(['rev-parse','refs/remotes/origin/direct-qualification-metadata-import-provenance-2026-11-01']).decode().strip()==INCOMING
 assert git(['merge-base',HEAD,INCOMING]).decode().strip()==BASE
 roster_path=w/'.review_metadata/import_provenance_publication_roster.json'
 roster_raw=read(roster_path);assert sha(roster_raw)=='3ac0138859ef080a33463b4f55404f3d1e16e4878ca591388748aaba4f008489'
 roster=json.loads(roster_raw);assert roster['file_count']==len(roster['files'])==25 and roster['base_commit']==BASE
 incoming_paths=git(['diff','--name-only','-z',BASE,INCOMING]).decode().split('\0')[:-1]
 expected_paths={record['path'] for record in roster['files']}|{'.review_metadata/import_provenance_publication_roster.json'}
 assert len(incoming_paths)==26 and set(incoming_paths)==expected_paths
 assert {path for path in incoming_paths if not path.startswith('.review_metadata/')}==PRODUCTION
 assert git(['status','--porcelain','--untracked-files=no','--',*incoming_paths],root=w)==b''
 base=tree(BASE);head=tree(HEAD);incoming=tree(INCOMING)
 root_changes=set(git(['diff','--name-only','-z',BASE,HEAD]).decode().split('\0')[:-1])
 overlaps=root_changes & set(incoming_paths);assert not overlaps,overlaps
 incoming_blobs=[]
 for relative in incoming_paths:
  record=next((row for row in roster['files'] if row['path']==relative),None)
  disk=read(w/relative);committed=git(['cat-file','blob',INCOMING+':'+relative])
  assert committed==disk
  if record:assert sha(disk)==record['sha256'] and len(disk)==record['bytes']
  assert incoming[relative]['mode'] in {'100644','100755'}
  assert bool((w/relative).stat().st_mode&stat.S_IXUSR)==(incoming[relative]['mode']=='100755')
  current=i/relative
  for parent in list(current.parents):
   if parent==i:break
   if parent.exists():assert parent.resolve(strict=True)==parent and parent.is_dir(),str(parent)
  if relative not in head:assert not current.exists() and not current.is_symlink(),str(current)
  else:assert read(current)==git(['cat-file','blob',HEAD+':'+relative]),relative
  incoming_blobs.append({'path':relative,'mode':incoming[relative]['mode'],'blob':incoming[relative]['blob'],'sha256':sha(disk),'bytes':len(disk),'already_tracked':relative in head})
 # Protect every original numerical file and both live A4500 staged/private helper closures.
 protected=[r/'SOURCE_AND_GATES_READY.json',i/PREFIX/'serval02_card1_binding.py',i/PREFIX/'measure_selected.py']
 live_units=[]
 for copies in (1,2):
  unit=next((r/'data').glob('*15-09_common*correctness_jaguar03_copies-'+str(copies)+'_*'))
  plan=json.loads(read(unit/'measurement_plan.json'));assert plan['source_commit']=='d28e6735e79788f618acb0ebfde8609db1f776af' and plan['helpers']['source_commit']=='e5afbf6b51a3d1666b46e71c0631a45d47080cd8'
  live_units.append({'unit':str(unit),'helper_source_root':plan['helpers']['source_root'],'helper_source_commit':plan['helpers']['source_commit'],'staged_helpers':len(plan['helpers']['files'])})
  protected += [unit/'measurement_plan.json']
  for relative,expected in plan['helpers']['files'].items():
   for path in [unit/'code'/pathlib.Path(relative).name,pathlib.Path(plan['helpers']['source_root'])/relative]:
    assert sha(read(path))==expected,str(path);protected.append(path)
  for relative,expected in plan['core']['files'].items():
   path=pathlib.Path(plan['source_root'])/relative;assert sha(read(path))==expected['sha256'];protected.append(path)
  for relative,expected in plan['paired_source']['files'].items():
   path=pathlib.Path(plan['source_root'])/relative;assert sha(read(path))==expected;protected.append(path)
 assert not {str(i/path) for path in incoming_paths}&set(map(str,protected))
 protected_before=snapshots(protected)
 assert sha(read(r/'SOURCE_AND_GATES_READY.json'))=='4a71ef2b245ae784933c55bb2b5159aa1bb7f8f626eb94b15390e2f9a304ee7c'
 index_path=pathlib.Path(git(['rev-parse','--git-path','index']).decode().strip())
 if not index_path.is_absolute():index_path=i/index_path
 index_before=read(index_path);index_sha=sha(index_before)
 # Each private index starts from actual cached stat entries; the real index is never refreshed.
 private_a=d/'checkout_dryrun.index';private_b=d/'three_way.index'
 private_a.write_bytes(index_before);private_b.write_bytes(index_before)
 head_tree=git(['rev-parse',HEAD+'^{tree}']).decode().strip()
 assert git(['write-tree'],index=private_a).decode().strip()==head_tree
 git(['read-tree','-m','-u','--dry-run',BASE,HEAD,INCOMING],index=private_a)
 assert git(['write-tree'],index=private_b).decode().strip()==head_tree
 git(['read-tree','-m','-i',BASE,HEAD,INCOMING],index=private_b)
 expected_tree=git(['write-tree'],index=private_b).decode().strip()
 expected_map=dict(head);expected_map.update({path:incoming[path] for path in incoming_paths})
 assert tree(expected_tree)==expected_map
 assert git(['rev-parse','HEAD']).decode().strip()==HEAD
 assert read(index_path)==index_before
 protected_after=snapshots(protected);assert protected_after==protected_before
 assert all(read(w/row['path'])==git(['cat-file','blob',INCOMING+':'+row['path']]) for row in incoming_blobs)
 value={'schema_version':1,'kind':'independent_direct_import_provenance_incoming_tree_review','approved':True,'passed':True,'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'integration_repository':str(i),'held_integration_head':HEAD,'base_commit':BASE,'incoming_commit':INCOMING,'incoming_tree':'bca6001aebc8bebfbf28e957adb445b30a91934a','expected_actual_merge_tree':expected_tree,'incoming_path_count':26,'incoming_blobs':incoming_blobs,'author_roster_sha256':sha(roster_raw),'overlapping_paths':sorted(overlaps),'ignored_or_untracked_collisions':[],'actual_checkout_dryrun_passed':True,'actual_private_index_three_way_passed':True,'private_indexes_initialized_from_actual_real_index':True,'real_index_path':str(index_path),'real_index_sha256_before':index_sha,'real_index_sha256_after':sha(read(index_path)),'real_head_unchanged':True,'live_a4500_units':live_units,'no_imported_or_hashed_live_source_path_changes':True,'protected_before':protected_before,'protected_after':protected_after,'protected_path_count':len(protected_before),'commands':commands,'numerical_imports':0,'gpu_operations':0,'job_actions':0,'original_checkout_fast_forward_performed':False,'scientific_credit':0,'admission_granted':False,'execution_scope':'local standard-library and Git metadata only; no Slurm attribution'}
 write(d/'review.json',value)
 print(json.dumps({'review':str(d/'review.json'),'sha256':sha(read(d/'review.json')),'expected_actual_merge_tree':expected_tree,'incoming_paths':26,'protected_paths':len(protected_before)}))
except BaseException as error:
 write(d/'failure.json',{'error':repr(error),'traceback':traceback.format_exc(),'commands':commands,'execution_scope':'local standard-library and Git metadata only','numerical_imports':0,'gpu_operations':0,'job_actions':0})
 raise
