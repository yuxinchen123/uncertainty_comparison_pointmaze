INTEGRATION='/p/rlprojects/RND_worktrees/merge-atari-cleanrl'
AUTHOR_WORKTREE='/p/rlprojects/RND_worktrees/2026-11-01-17-55_a4500-common-confirmation_prior-correctness-required_rounds-7_complete-updates-per-arm-10_copies-1-2'
RUN='/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1'
OUTPUT='/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-18-39_common-confirmation-incoming_author-commit-b1c7405c_target-commit-490504ff_private-source-preservation'
"""Check the reviewed incoming tree with private indexes and preserved private sources."""
import datetime,hashlib,json,os,pathlib,stat,subprocess,sys,tempfile
I=pathlib.Path(INTEGRATION);W=pathlib.Path(AUTHOR_WORKTREE);R=pathlib.Path(RUN);O=pathlib.Path(OUTPUT)
BASE='6066d9ea35fbc9c045855144550971c2ac5895f9'
TARGET='490504ffcd3ff0b6a8dd7b2e98e96f9d0bb5c313'
AUTHOR='b1c7405c2dce76270554973bdf81c87271216cd4'
assert os.environ['SLURM_JOB_ID']=='6783003' and os.environ['SLURM_STEP_ID'].isdigit()
commands=[]
envbase={**os.environ,'GIT_OPTIONAL_LOCKS':'0'}
def sha(raw):return hashlib.sha256(raw).hexdigest()
def save(name,v):
 p=O/name
 with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 return {'path':str(p),'sha256':sha(p.read_bytes())}
def git(*args,root=I,index=None,input=None,check=True):
 env=dict(envbase)
 if index is not None:env['GIT_INDEX_FILE']=str(index)
 argv=['git','-C',str(root),*args]
 v=subprocess.run(argv,input=input,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,check=False)
 commands.append({'argv':argv,'alternate_index':str(index) if index is not None else None,'returncode':v.returncode,'stdout_sha256':sha(v.stdout),'stderr_sha256':sha(v.stderr),'stdout':v.stdout.decode(errors='replace') if len(v.stdout)<5000 else None,'stderr':v.stderr.decode(errors='replace')})
 if check:assert v.returncode==0,(argv,v.stderr.decode())
 return v.stdout
def blob(commit,path):return git('cat-file','blob',commit+':'+path)
def head(root=I):return git('rev-parse','HEAD',root=root).decode().strip()
def index_image(root):
 p=pathlib.Path(git('rev-parse','--git-path','index',root=root).decode().strip())
 if not p.is_absolute():p=root/p
 s=p.stat()
 return {'path':str(p),'sha256':sha(p.read_bytes()),'device':s.st_dev,'inode':s.st_ino,'mode':stat.S_IMODE(s.st_mode),'size':s.st_size}
def tree(commit):
 raw=git('ls-tree','-r','-z',commit);items={}
 for entry in raw.split(b'\0'):
  if not entry:continue
  meta,name=entry.split(b'\t',1);mode,kind,oid=meta.decode().split()
  items[os.fsdecode(name)]={'mode':mode,'type':kind,'blob':oid}
 return items
def ordinary(path):
 s=path.lstat();assert stat.S_ISREG(s.st_mode) and not path.is_symlink() and path.resolve(strict=True)==path
 assert s.st_size<=16*1024*1024,('unexpected large source/proof read',str(path),s.st_size)
 return s,path.read_bytes()
def state(path):
 s,raw=ordinary(path)
 return {'sha256':sha(raw),'bytes':len(raw),'mode':stat.S_IMODE(s.st_mode),'uid':s.st_uid,'device':s.st_dev,'inode':s.st_ino}
def mode_matches(mode,actual):
 return (mode=='100755')==bool(actual&0o111)
try:
 assert head()==TARGET and head(W)==AUTHOR
 assert git('rev-parse',AUTHOR+'^').decode().strip()==BASE
 assert git('rev-parse',AUTHOR+'^{tree}').decode().strip()=='43d0735710649a3b0a019dfbb69c0e01dfbb519a'
 realindex_before=index_image(I)
 original=pathlib.Path('/p/rlprojects/RND')
 original_before={'head':head(original),'index':index_image(original)}
 roster_path=W/'.review_metadata/common_confirmation_publication_roster.json'
 roster=json.loads(roster_path.read_bytes());assert sha(roster_path.read_bytes())=='c20a6835b33d6b3eedfb8d78a412c04f3fdf742a3f531a3dc5a69bc1c8e79bc8'
 rows=roster['files']+[{'path':'.review_metadata/common_confirmation_publication_roster.json','sha256':sha(roster_path.read_bytes()),'bytes':roster_path.stat().st_size,'git_mode':'100644'}]
 incoming=git('diff-tree','--no-commit-id','--name-only','-r',BASE,AUTHOR).decode().splitlines()
 assert len(rows)==len(incoming)==48 and len({x['path'] for x in rows})==48 and {x['path'] for x in rows}==set(incoming)
 target_tree,author_tree=tree(TARGET),tree(AUTHOR)
 incoming_manifest=[]
 for row in rows:
  name=row['path'];entry=author_tree[name];s,raw=ordinary(W/name)
  assert entry['type']=='blob' and entry['mode']==row['git_mode'] and mode_matches(entry['mode'],s.st_mode)
  assert raw==blob(AUTHOR,name) and sha(raw)==row['sha256'] and len(raw)==row['bytes']
  incoming_manifest.append({**row,'git_blob':entry['blob']})
 freeze_path=W/'.review_metadata/common_confirmation_complete_source_freeze.json'
 frozen=json.loads(freeze_path.read_bytes())
 assert sha(freeze_path.read_bytes())=='27281757454d61f4717e8d335a30f770829907c2f1d6a6a4d698c723c709cb6c'
 assert len(frozen['files'])==6
 for name,expected in frozen['files'].items():
  rel=str(pathlib.Path(name).relative_to(W));assert sha(blob(AUTHOR,rel))==expected
 review=W/'.review_metadata/common_confirmation_independent_source_review.json'
 assert sha(review.read_bytes())=='bf2c6eebff276c834495f01e122ecd946d45f11b44b9a175023a778ee224a4c9'
 root_paths=set(git('diff-tree','--no-commit-id','--name-only','-r',BASE,TARGET).decode().splitlines())
 overlap=root_paths&set(incoming);assert not overlap,('incoming/operational overlap',sorted(overlap))
 def collision_check():
  ignored=git('ls-files','--others','--ignored','--exclude-standard','-z').split(b'\0')
  untracked=git('ls-files','--others','--exclude-standard','-z').split(b'\0')
  unknown={os.fsdecode(x) for x in ignored+untracked if x}
  collisions=[]
  for name in incoming:
   path=I/name
   if name not in target_tree and os.path.lexists(path):collisions.append(name)
   for parent in path.parents:
    if parent==I:break
    if os.path.lexists(parent) and (parent.is_symlink() or not parent.is_dir()):collisions.append(str(parent.relative_to(I)))
   if name in target_tree:
    s,raw=ordinary(path)
    assert raw==blob(TARGET,name) and mode_matches(target_tree[name]['mode'],s.st_mode),('dirty incoming tracked path',name)
   for x in unknown:
    if x==name or x.startswith(name+'/') or name.startswith(x+'/'):collisions.append(x)
  assert not collisions,('existing ignored/untracked collision',sorted(set(collisions)))
  return {'untracked_count':len([x for x in untracked if x]),'ignored_count':len([x for x in ignored if x]),'untracked_paths_sha256':sha(b'\0'.join(sorted(set(untracked)))),'ignored_paths_sha256':sha(b'\0'.join(sorted(set(ignored)))),'incoming_collisions':[]}
 collision_before=collision_check()
 protected={};labels={}
 def protect(path,label,expected=None):
  path=pathlib.Path(path);value=state(path)
  if expected is not None:assert value['sha256']==expected,('protected input differs',str(path))
  protected[str(path)]=value;labels.setdefault(label,[]).append(str(path))
 def protect_ref(ref,label):
  assert set(ref)=={'path','sha256'};protect(ref['path'],label,ref['sha256'])
 def plan_closure(unit,label):
  plan_path=unit/'measurement_plan.json';protect(plan_path,label);p=json.loads(plan_path.read_bytes())
  protect(unit/'measurement_prepared.json',label)
  for group in ['core','paired_source','helpers','qualification_sources']:
   if group not in p:continue
   rec=p[group]
   for relative,expected in rec['files'].items():
    expected=expected['sha256'] if isinstance(expected,dict) else expected
    src=pathlib.Path(rec['source_root'])/relative;protect(src,label,expected)
    assert sha(git('cat-file','blob',rec['source_commit']+':'+relative,root=pathlib.Path(rec['source_root'])))==expected
    if group=='helpers':protect(unit/'code'/pathlib.Path(relative).name,label,expected)
  for ref in p.get('target_qualification_inputs',{}).values():protect_ref(ref,label+'_qualification_proof_reference')
  return p
 n=json.loads((R/'next_actions.json').read_bytes())
 hpacket=pathlib.Path(n['h100_fresh_original_qualification_preparation']['packet']);hunit=R/'data'/hpacket.name
 hp=plan_closure(hunit,'actual_private_H100_6066')
 assert hp['helpers']['source_commit']==BASE
 for name in ['packet_metadata_freeze.json','controller_source_freeze.json','direct_qualification_controller.py','controller_root_independent_review.json']:
  protect(hpacket/name,'actual_private_H100_6066')
 hfreeze=json.loads((hpacket/'packet_metadata_freeze.json').read_bytes())
 for name,expected in hfreeze['files'].items():protect(name,'actual_private_H100_6066_packet',expected)
 for count in ['1','2']:
  unit=pathlib.Path(n['new_common_correctness_submissions']['jobs_actual'][count]['unit'])
  p=plan_closure(unit,'actual_private_A4500_E5_copies_'+count)
  assert p['helpers']['source_commit']=='e5afbf6b51a3d1666b46e71c0631a45d47080cd8'
  protect(unit/'result.json','actual_A4500_common_correctness_result')
 mpacket=pathlib.Path(n['persistent_monitor']['packet'])
 mp=json.loads((mpacket/'monitor_replacement_plan.json').read_bytes())
 manifest_path=mpacket/'source_snapshot.json';manifest=json.loads(manifest_path.read_bytes())
 protect(mpacket/'monitor_replacement_plan.json','actual_CPU_79a_monitor');protect(manifest_path,'actual_CPU_79a_monitor')
 assert manifest['source_commit']=='79a01e851ff92ddba7d0e12eaf8164cab2028878' and len(manifest['files'])==265
 for root in [pathlib.Path(manifest['source_root']),pathlib.Path(manifest['clean_detached_source_root'])]:
  for relative,expected in manifest['files'].items():
   path=root/relative;protect(path,'actual_CPU_79a_source',expected['sha256'])
   assert mode_matches(expected['git_mode'],path.stat().st_mode)
 protect(R/'SOURCE_AND_GATES_READY.json','frozen_source_admission','4a71ef2b245ae784933c55bb2b5159aa1bb7f8f626eb94b15390e2f9a304ee7c')
 before_ref=save('protected_private_sources_before.json',{'files':protected,'labels':{k:sorted(set(v)) for k,v in labels.items()},'library_asset_payloads_rehashed':False})
 with tempfile.TemporaryDirectory(prefix='common-confirmation-incoming-') as temp:
  tmp=pathlib.Path(temp);dry=tmp/'dry.index';merged=tmp/'merge.index'
  git('read-tree',TARGET,index=dry)
  dry_before=sha(dry.read_bytes())
  git('read-tree','--dry-run','-m','-u',BASE,TARGET,AUTHOR,index=dry)
  assert sha(dry.read_bytes())==dry_before
  git('read-tree','-m',BASE,TARGET,AUTHOR,index=merged)
  assert not git('ls-files','--unmerged','-z',index=merged)
  merged_tree=git('write-tree',index=merged).decode().strip()
  merged_items=tree(merged_tree)
  actual_changed=git('diff-tree','--no-commit-id','--name-only','-r',TARGET,merged_tree).decode().splitlines()
  assert set(actual_changed)==set(incoming) and len(actual_changed)==48
  for name in incoming:assert merged_items[name]==author_tree[name]
  for name in set(target_tree)-set(incoming):assert merged_items[name]==target_tree[name]
  assert set(merged_items)==set(target_tree)|set(incoming)
  dryrun={'exact_three_way_base':BASE,'target':TARGET,'author':AUTHOR,'dry_run_alternate_index_unchanged':True,'three_way_conflicts':0,'resulting_tree':merged_tree,'actual_changed_paths':actual_changed,'all_other_target_blobs_modes_preserved':True}
 collision_after=collision_check()
 after={path:state(pathlib.Path(path)) for path in protected};assert after==protected
 assert head()==TARGET and index_image(I)==realindex_before
 original_after={'head':head(original),'index':index_image(original)};assert original_after==original_before
 result={'schema_version':1,'kind':'independent_common_confirmation_incoming_tree_and_private_source_review','passed':True,'merge_approved':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':'6783003','step_id':os.environ['SLURM_STEP_ID'],'base_commit':BASE,'target_commit':TARGET,'author_commit':AUTHOR,'author_tree':'43d0735710649a3b0a019dfbb69c0e01dfbb519a','expected_merged_tree':merged_tree,'incoming_path_count':48,'incoming_manifest':incoming_manifest,'roster_sha256':sha(roster_path.read_bytes()),'approved_source_freeze_sha256':sha(freeze_path.read_bytes()),'independent_source_review_sha256':sha(review.read_bytes()),'operational_overlap_paths':[],'collision_before':collision_before,'collision_after':collision_after,'private_protection':before_ref,'protected_private_source_file_count':len(protected),'all_private_source_hashes_modes_uids_inodes_before_after_equal':True,'dual_actual_alternate_index_checks':dryrun,'real_integration_index_unchanged':True,'original_checkout':{'before':original_before,'after':original_after,'no_worktree_update_commands_targeted_original':True,'whole_original_content_hash_claimed':False},'actual_worktree_mutations':0,'actual_gpu_or_job_actions':0,'scientific_credit':0,'admission_granted':False,'remaining_root_action':'Merge this exact author into this exact held target and verify the resulting tree before push.'}
 save('commands.json',commands);save('review.json',result);print(json.dumps({'passed':True,'review':str(O/'review.json'),'review_sha256':sha((O/'review.json').read_bytes()),'expected_merged_tree':merged_tree,'protected_files':len(protected),'incoming_paths':48}))
except BaseException as error:
 save('commands_failure.json',commands);save('failure.json',{'kind':'incoming_tree_review_failure','type':type(error).__name__,'message':str(error),'scientific_credit':0});raise
