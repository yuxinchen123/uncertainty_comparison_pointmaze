import hashlib,json,os,shutil,stat,subprocess,tempfile,time
from pathlib import Path
I=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl');R=I/'10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1';HEAD='fd66474d4cc9c79e5e15a3b6ded388c9d038a77b';A='b51b37413dbf4d4dbc3f5a0461a7bc2be716a48a';BASE='d1469fd063a44167ec56760729eb0beea7ecf7d3';ENV=dict(os.environ,GIT_OPTIONAL_LOCKS='0');started=time.monotonic();checks=[]
def need(v,label):
 if not v:raise ValueError(label)
 checks.append(label)
def sha(b):return hashlib.sha256(b).hexdigest()
def git(*a,index=None,data=None):
 env=dict(ENV)
 if index is not None:env['GIT_INDEX_FILE']=str(index)
 return subprocess.check_output(['git','-c','core.preloadIndex=false','-c','core.fsmonitor=false',*a],cwd=I,env=env,input=data,timeout=45,stderr=subprocess.PIPE)
def names(b):return list(filter(None,b.decode().split('\0')))
def fs(p):
 s=p.stat();return {'device':s.st_dev,'inode':s.st_ino,'size':s.st_size,'mtime':s.st_mtime_ns,'ctime':s.st_ctime_ns,'mode':s.st_mode,'sha256':sha(p.read_bytes())}
need(os.environ.get('SLURM_JOB_ID')=='6825983' and os.environ.get('SLURM_STEP_ID','').isdigit(),'current owned CPU step');need('6825983' in (R/'slurm/submitted_jobids_uva.txt').read_text().splitlines(),'owned campaign ledger')
hp=Path('/p/rlprojects/RND_worktrees/2026-11-05-11-30_common-performance-prerequisite-fact_contexts-4_paired-rounds-7_updates-per-arm-10_profile-qualification-operational-preparation_gpu-actions-0/.review_metadata/profile_target_operational_preparation/h_publication_handoff.json');need(sha(hp.read_bytes())=='1816141298002d2fe300a444325a01b1092f8ff4609cabae3754f2b11fb803b0','actual author handoff');h=json.loads(hp.read_text());need(h['commit']==A and h['parent']==BASE,'actual commit/base');need(git('rev-parse','HEAD').decode().strip()==HEAD,'current actual integration head');need(git('diff','--cached','--name-only','-z')==b'','no staged edits');need(git('rev-parse',A+'^').decode().strip()==BASE and git('rev-parse',A+'^{tree}').decode().strip()==h['tree'],'actual Git author identity');paths=names(git('diff','--name-only','-z',BASE,A));need(len(paths)==40 and set(paths)==set(h['files']),'exact 40 path roster');need(all(p.startswith('.review_metadata/profile_target_operational_preparation/') and '..' not in Path(p).parts for p in paths),'metadata-only paths cannot change runtime helper files')
dirty=names(git('diff','--name-only','-z','HEAD'));untracked=names(git('ls-files','--others','--exclude-standard','-z'));need(not set(paths).intersection(dirty+untracked),'incoming disjoint from dirty/untracked');tracked=set(names(git('ls-files','-z')));incoming={}
for p in paths:
 row=h['files'][p];b=git('show',A+':'+p);b.decode('utf-8');need(len(b)==row['size_bytes'] and sha(b)==row['sha256'] and len(b)<2500000,'exact bounded approved bytes '+p);need(p in tracked or not os.path.lexists(I/p),'no ignored/untracked collision '+p);incoming[p]=b
review=h['independent_review'];rb=incoming[review['relative_path']];v=json.loads(rb);need(sha(rb)=='31b806c02b90d426a61a40c65f7b488e0c5ff92db1e7176cd3cf5aa8e9f271e7' and v['approved'] is True and v['passed'] is True,'genuine independent operational source approval');freeze=h['source_freeze'];need(sha(incoming[freeze['relative_path']])==freeze['sha256']=='094ccd75dfe0f1948d4c1085a2894b28cec0b454b9ba80d2db49f1fdfbc420ee','actual reviewed source freeze')
with tempfile.TemporaryDirectory(prefix='rnd_operational_metadata_merge_') as td:
 t=Path(td);private=t/'predicted.index';git('read-tree',HEAD,index=private)
 for p,b in incoming.items():
  need(p not in tracked,'all operational artifacts are new paths '+p);oid=git('hash-object','-w','--stdin',data=b).decode().strip();git('update-index','--add','--cacheinfo',h['files'][p]['git_mode']+','+oid+','+p,index=private)
 predicted=git('write-tree',index=private).decode().strip();need(set(names(git('diff','--name-only','-z',HEAD,predicted)))==set(paths),'predicted changes exactly approved metadata paths')
 index=Path(git('rev-parse','--git-path','index').decode().strip());index=index if index.is_absolute() else I/index;stages=sha(git('ls-files','--stage','-z'));head=git('rev-parse','HEAD').decode().strip();need(head==HEAD,'actual HEAD still unchanged');before=fs(index)
 for n in (1,2):
  copied=t/('real-index-copy-'+str(n));shutil.copy2(index,copied);need(fs(copied)['sha256']==before['sha256'],'exact copied index '+str(n));git('read-tree','-m','-u','--dry-run',HEAD,predicted,index=copied)
 after=fs(index);need(after==before,'real index bytes and identity unchanged during dry runs')
record={'kind':'root_operational_metadata_commit_incoming_review','passed':True,'integration_head':HEAD,'author_commit':A,'author_parent':BASE,'author_tree':h['tree'],'predicted_merge_tree':predicted,'paths':paths,'all_40_approved_UTF8_metadata_files_exact':True,'runtime_helper_files_changed':False,'actual_copied_index_dryruns':2,'index_before':before,'index_after':after,'stage_entry_sha256':stages,'source_approval':review,'numerical_GPU_or_remote_actions':0,'owned_job_id':os.environ['SLURM_JOB_ID'],'owned_step_id':os.environ['SLURM_STEP_ID'],'checks':checks,'check_count':len(checks),'seconds':time.monotonic()-started};out=R/'remote_setup/root_profile_operational_metadata_incoming_review.json'
with out.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
print(json.dumps({'passed':True,'review':str(out),'sha256':sha(out.read_bytes()),'predicted_tree':predicted,'checks':len(checks),'seconds':record['seconds']}))
