# Independently predict two reviewed source commits without changing the real index or worktree.
import ast,datetime,hashlib,json,os,shutil,stat,subprocess,tempfile,time
from pathlib import Path
I=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl');R=I/'10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1'
HEAD='90ea7a898d69ee23c51fff7cfbcfd0499a768e09';started=time.monotonic();checks=[]
ENV=dict(os.environ,GIT_OPTIONAL_LOCKS='0')
AUTHORS=[{'name':'observer','commit':'e43959c06e2610f20abac23c82d472e8820434c1','parent':HEAD,'tree':'8c442c13316d3c4e2980201cf2526a3d9966946c','roster':Path('/p/rlprojects/RND_worktrees/2026-11-05-07-23_prior-h100-exit-observation_publication_controller-pid-386202_numerical-pid-386669_gpu-operations-0/.review_metadata/prior_h100_exit_observer/actual_detached_publication_handoff.json'),'roster_sha':'456ac44e423f8b890c6c1141267fcee2799fb02e95cd5534bd9ba08b7074afa5','count':8},{'name':'controller_request','commit':'cba869200cb31131b354c985f77b43386f90f9b9','parent':'493c15234c2ad2be6a59fd226d977fd66e8c0296','tree':'3e46e317d1b4631eeb26604ce47119edab8ec55f','roster':Path('/p/rlprojects/RND_worktrees/2026-11-05-07-20_h100-timing-predecessor_actual-confirmation-schema_two-copy-request_read-only-exits_gpu-actions-0/.review_metadata/timing_predecessor_schema/prepared_local_commit_roster.json'),'roster_sha':'6b5567855a811e0fb997fcf802467ebf650a3afa3f12c7de1c7dcaec693570b5','count':23}]
def check(label,value):
 if not value:raise ValueError(label)
 checks.append(label)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  while True:
   raw=f.read(1024*1024)
   if not raw:break
   h.update(raw)
 return h.hexdigest()
def git(*args,env=None,input=None):return subprocess.check_output(['git','-c','core.preloadIndex=false',*args],cwd=I,env=ENV if env is None else env,input=input,timeout=90)
def optional(revision,path):
 q=subprocess.run(['git','cat-file','-e',revision+':'+path],cwd=I,env=ENV,capture_output=True,timeout=30)
 return git('show',revision+':'+path) if q.returncode==0 else None
def split(raw):return [p for p in raw.decode().split('\0') if p]
def fsidentity(path):
 s=Path(path).lstat();return {'device':s.st_dev,'inode':s.st_ino,'mode':stat.S_IMODE(s.st_mode),'size_bytes':s.st_size,'mtime_ns':s.st_mtime_ns}
assert os.environ['SLURM_JOB_ID']=='6821437';check('current exact review allocation owned',(R/'slurm/submitted_jobids_uva.txt').read_text().splitlines().count('6821437')==1)
check('actual integration HEAD unchanged before review',git('rev-parse','HEAD').decode().strip()==HEAD)
check('actual index contains no staged edits',subprocess.run(['git','diff','--cached','--quiet'],cwd=I,env=ENV).returncode==0)
index=Path(git('rev-parse','--git-path','index').decode().strip());index=index if index.is_absolute() else I/index;index_before={'identity':fsidentity(index),'sha256':sha(index),'stage_entry_sha256':hashlib.sha256(git('ls-files','--stage','-z')).hexdigest()}
initial_snapshot=R/'remote_setup/independent_controller_observer_combined_incoming_second_before.json'
with initial_snapshot.open('x') as f:json.dump({'integration_head':HEAD,'index_before':index_before,'job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
protected_path=R/'remote_setup/direct_nsight_incoming_independent_2026-11-04-20-01-56_protected_before.json';check('exact 662 protected baseline',sha(protected_path)=='3412f310f74e010aab08548a77813f811ad51e5a5b5ae9db1798ee21afe77430');protected=json.loads(protected_path.read_bytes())['files'];check('all 662 protected entries present',len(protected)==662)
def verify_protected():
 for path,row in protected.items():
  p=Path(path);s=p.stat();check('protected byte/mode/size '+path,sha(p)==row['sha256'] and s.st_size==row['size_bytes'] and stat.S_IMODE(s.st_mode)==row['mode'])
verify_protected()
P=next((R/'remote_setup').glob('2026-11-05-05-49_*'));U=R/'data'/P.name;plan=json.loads((U/'measurement_plan.json').read_bytes());live={}
source_freeze=P/'controller_source_freeze.json';sf=json.loads(source_freeze.read_bytes());live.update(sf['files'])
for path in P.iterdir():
 if path.is_file() and path.suffix in ('.py','.sh','.json','.yaml','.md','.txt') and path.stat().st_size<2500000:live[str(path)]=sha(path)
for path in U.iterdir():
 if path.is_file() and path.suffix in ('.json','.yaml','.md','.txt') and path.stat().st_size<2500000:live[str(path)]=sha(path)
for label in ('core','paired_source','helpers'):
 for relative,row in plan[label]['files'].items():
  p=Path(plan[label]['source_root'])/relative;expected=row['sha256'] if isinstance(row,dict) else row;check('actual immutable '+label+' '+relative,sha(p)==expected);live[str(p)]=expected
result_stat=fsidentity(U/'result.json');check('accepted H100 result actual size retained without reread',result_stat['size_bytes']==53225495)
check('accepted timing review unchanged',sha(P/'independent_actual_common_timing_review.json')=='901a5805d7f768cd3b3159fe7e478fffdbee36280c2144f195ada1b3471bdf22')
def verify_live():
 for path,expected in live.items():check('accepted H100 static file '+path,sha(path)==expected)
 check('accepted H100 result identity/stat unchanged without payload reread',fsidentity(U/'result.json')==result_stat)
verify_live()
dirty=split(git('diff','--name-only','-z','HEAD'));untracked=split(git('ls-files','--others','--exclude-standard','-z'))
preserved={path:{'identity':fsidentity(I/path),'sha256':sha(I/path)} for path in set(dirty+untracked) if (I/path).is_file() and not (I/path).is_symlink()}
all_paths=[];author_proofs=[]
for author in AUTHORS:
 check('actual author roster '+author['name'],sha(author['roster'])==author['roster_sha']);roster=json.loads(author['roster'].read_bytes())
 check('actual author commit and parent '+author['name'],roster['commit']==author['commit'] and git('rev-parse',author['commit']+'^').decode().strip()==author['parent'])
 check('actual author committed tree '+author['name'],git('rev-parse',author['commit']+'^{tree}').decode().strip()==author['tree'])
 base=git('merge-base',HEAD,author['commit']).decode().strip();paths=split(git('diff','--name-only','--no-renames','-z',base,author['commit']))
 check('actual exact author changed roster '+author['name'],len(paths)==author['count'] and set(paths)==set(roster['files']))
 check('author paths disjoint from dirty/untracked '+author['name'],not set(paths).intersection(dirty+untracked))
 for path in paths:
  check('ordinary safe relative incoming path '+path,not Path(path).is_absolute() and '..' not in Path(path).parts and '.git' not in Path(path).parts)
  raw=git('show',author['commit']+':'+path);row=roster['files'][path];raw.decode('utf-8');entry=git('ls-tree',author['commit'],'--',path).decode().split()
  check('exact UTF8 incoming mode/blob/hash/size '+path,entry[:2]==[row['mode'],'blob'] and entry[2]==row['git_blob'] and row['mode'] in ('100644','100755') and hashlib.sha256(raw).hexdigest()==row['sha256'] and len(raw)==row.get('bytes',row.get('size_bytes')) and len(raw)<=2500000)
  current=optional(HEAD,path);target=I/path
  check('incoming ignored/untracked collision absent '+path,current is not None or not os.path.lexists(target))
  ancestor=target.parent
  while ancestor!=I:
   if os.path.lexists(ancestor):check('incoming parent ordinary directory '+str(ancestor),ancestor.is_dir() and not ancestor.is_symlink())
   ancestor=ancestor.parent
 all_paths+=paths;author_proofs.append({'name':author['name'],'commit':author['commit'],'parent':author['parent'],'tree':author['tree'],'base':base,'paths':paths,'roster':{'path':str(author['roster']),'sha256':author['roster_sha']}})
check('both incoming rosters disjoint',len(all_paths)==len(set(all_paths))==31)
for author in AUTHORS:
 approval_path=('.review_metadata/prior_h100_exit_observer/' if author['name']=='observer' else '.review_metadata/timing_predecessor_schema/')+'independent_request_controller_observer_review.json'
 raw=git('show',author['commit']+':'+approval_path);review=json.loads(raw);check('actual independent approval committed '+author['name'],hashlib.sha256(raw).hexdigest()=='df55b20ea77ab5a4c775c2b32f286f3953655c2f06b3cf7f6489e5a93a3ce141' and review['approved'] is True and review['passed'] is True)
changed_source='10_jax_exploration_platform/scripts/atari_pixel_campaign/direct_common_timing_controller.py'
check('only approved incoming controller source bytes',hashlib.sha256(git('show',AUTHORS[1]['commit']+':'+changed_source)).hexdigest()=='1d8bb43df5c4c247e28bb3d588674e8d3cd7718363c62584bc57267d67af10e2')
check('only approved incoming observer source bytes',hashlib.sha256(git('show',AUTHORS[0]['commit']+':.review_metadata/prior_h100_exit_observer/observe_exact_prior_h100_exits.py')).hexdigest()=='87bdf73927b03341ebe6b5b3d3600e8b77e62820b7f8c07f78c9ff5936ba9540')
merges=[];stages=[]
with tempfile.TemporaryDirectory(prefix='rnd_review_controller_observer_incoming_') as temporary:
 temp=Path(temporary);env=dict(ENV,GIT_INDEX_FILE=str(temp/'predicted.index'));git('read-tree',HEAD,env=env);current_tree=HEAD
 for author,proof in zip(AUTHORS,author_proofs):
  for path in proof['paths']:
   original=optional(proof['base'],path);current=optional(current_tree,path);incoming=optional(author['commit'],path);check('ordinary incoming blob remains '+path,incoming is not None)
   if current==original:merged=incoming
   elif current==incoming:merged=current
   elif current is not None and original is not None:
    names=[]
    for name,raw in (('current',current),('base',original),('incoming',incoming)):
     file=temp/name;file.write_bytes(raw);names.append(str(file))
    merged=subprocess.check_output(['git','merge-file','-p',*names],env=ENV,timeout=30);merges.append(path)
   else:raise ValueError('different simultaneous additions '+path)
   mode=git('ls-tree',author['commit'],'--',path).decode().split()[0];blob=git('hash-object','-w','--stdin',input=merged).decode().strip();git('update-index','--add','--cacheinfo',mode+','+blob+','+path,env=env)
  current_tree=git('write-tree',env=env).decode().strip();stages.append({'after':author['commit'],'tree':current_tree})
 predicted=current_tree
 for number in (1,2):
  copied=temp/('copied-real-index-'+str(number));shutil.copy2(index,copied);private=dict(ENV,GIT_INDEX_FILE=str(copied));git('read-tree','-m','-u','--dry-run',HEAD,predicted,env=private)
check('predicted changed tree exact approved union',set(split(git('diff','--name-only','--no-renames','-z',HEAD,predicted)))==set(all_paths))
verify_protected();verify_live()
for path,row in preserved.items():check('preexisting dirty/untracked file preserved '+path,fsidentity(I/path)==row['identity'] and sha(I/path)==row['sha256'])
check('actual HEAD still unchanged',git('rev-parse','HEAD').decode().strip()==HEAD)
index_after={'identity':fsidentity(index),'sha256':sha(index),'stage_entry_sha256':hashlib.sha256(git('ls-files','--stage','-z')).hexdigest()}
checkpoint=R/'remote_setup/independent_controller_observer_combined_incoming_second_checkpoint.json'
with checkpoint.open('x') as f:json.dump({'integration_head':HEAD,'predicted_merge_tree':predicted,'stages':stages,'index_before':index_before,'index_after':index_after,'all_prior_checks_passed':True,'check_count':len(checks),'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
check('actual index bytes and identity still unchanged',index_after==index_before)
record={'kind':'independent_combined_controller_observer_incoming_review','passed':True,'integration_head':HEAD,'authors':author_proofs,'predicted_merge_tree':predicted,'predicted_sequential_stage_trees':stages,'changed_paths':all_paths,'automatic_three_way_paths':merges,'actual_copied_index_read_tree_dry_runs':2,'protected_file_count':len(protected),'accepted_H100_static_file_count':len(live),'accepted_H100_large_result_stat_preserved_without_payload_reread':result_stat,'dirty_paths_preserved':dirty,'untracked_paths_preserved':untracked,'preexisting_dirty_untracked_file_preservation_count':len(preserved),'real_index_before':index_before,'real_index_after':index_after,'all_incoming_files_exact_UTF8_at_most_2500000_bytes':True,'all_incoming_paths_disjoint_dirty_untracked_and_existing_ignored_collisions':True,'real_HEAD_index_worktree_not_mutated':True,'actual_numerical_GPU_or_native_library_operations':0,'historical_5_67GB_snapshot_or_captured_images_rehashed':False,'job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],'pid':os.getpid(),'uid':os.getuid(),'seconds':time.monotonic()-started,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'check_count':len(checks),'checks':checks}
out=R/'remote_setup/independent_controller_observer_combined_incoming_second_2026-11-05.json'
with out.open('x') as f:json.dump(record,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps({'passed':True,'proof':str(out),'proof_sha256':sha(out),'expected_head':HEAD,'predicted_merge_tree':predicted,'paths':len(all_paths),'protected_files':len(protected),'accepted_static_files':len(live),'copied_index_dry_runs':2,'checks':len(checks),'seconds':record['seconds'],'job_id':record['job_id'],'step_id':record['step_id']}))
