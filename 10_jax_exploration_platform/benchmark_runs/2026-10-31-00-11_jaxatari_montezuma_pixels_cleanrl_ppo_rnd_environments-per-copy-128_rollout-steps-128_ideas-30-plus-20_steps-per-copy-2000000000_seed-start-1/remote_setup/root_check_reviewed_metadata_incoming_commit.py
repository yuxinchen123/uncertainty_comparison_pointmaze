"""Check one independently approved metadata-only commit using private copies of the actual index."""
import argparse,hashlib,json,os,subprocess,tempfile,time,stat
from pathlib import Path
I=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl')
R=I/'10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1'
CHECKS=[]
def need(value,message):
    """Stop before any real mutation when a precise incoming condition fails."""
    if not value:raise ValueError(message)
    CHECKS.append(message)
def git(*args,env=None,required=True):
    """Read local Git metadata, using explicit alternate indexes for every merge dry run."""
    e={**os.environ,'GIT_OPTIONAL_LOCKS':'0'}
    if env:e.update(env)
    q=subprocess.run(['git',*args],cwd=I,env=e,capture_output=True,timeout=30)
    if required:need(q.returncode==0,'Git operation succeeded: '+str(args[:2]))
    return q
def file_state(path):
    """Observe ordinary actual file identity and content for the final no-write window."""
    s=path.lstat();need(stat.S_ISREG(s.st_mode),'ordinary Git metadata file')
    return [s.st_dev,s.st_ino,s.st_uid,s.st_gid,s.st_mode,s.st_size,s.st_mtime_ns,s.st_ctime_ns,hashlib.sha256(path.read_bytes()).hexdigest()]
def main():
    """Verify all actual changed blobs and two three-way private-index results before root merge."""
    ap=argparse.ArgumentParser();ap.add_argument('--incoming',required=True);ap.add_argument('--review',type=Path,required=True);ap.add_argument('--review-sha256',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();started=time.monotonic()
    need(os.environ.get('SLURM_JOB_ID')=='6825983' and os.environ.get('SLURM_STEP_ID','').isdecimal(),'actual owned current CPU allocation')
    reviewraw=a.review.read_bytes();need(hashlib.sha256(reviewraw).hexdigest()==a.review_sha256,'exact genuine independent review bytes');review=json.loads(reviewraw);need(review.get('approved') is review.get('passed') is True,'genuine review approved')
    head=git('rev-parse','HEAD').stdout.decode().strip();parent=git('rev-parse',a.incoming+'^').stdout.decode().strip();names=git('diff','--name-only',parent,a.incoming).stdout.decode().splitlines();need(0<len(names)<100,'bounded exact incoming path set')
    need(not git('diff','--cached','--name-only').stdout.strip(),'real staged entries empty before check')
    files=[]
    for name in names:
        need(name.startswith('.review_metadata/') and '..' not in Path(name).parts,'metadata-only incoming path')
        raw=git('cat-file','blob',a.incoming+':'+name).stdout;need(len(raw)<2500000,'bounded UTF-8 source or metadata');raw.decode()
        target=I/name;old=git('cat-file','blob',parent+':'+name,required=False);current=git('cat-file','blob',head+':'+name,required=False)
        if old.returncode==0:
            need(name=='.review_metadata/profile_target_operational_preparation/derive_one_copy_qualification_request.py','only the reviewed existing operational deriver may change')
            need(current.returncode==0 and current.stdout==old.stdout==target.read_bytes(),'changed operational file matches incoming base and actual root bytes')
        else:need(current.returncode!=0 and not os.path.lexists(target),'no tracked, untracked or ignored incoming collision')
        tree=git('ls-tree',a.incoming,'--',name).stdout.decode().strip();need(tree.startswith('100644 blob '),'ordinary non-executable incoming Git mode')
        files.append({'path':name,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'git_entry':tree})
    # Read every real-index Git fact before the narrow preserved-file window.
    index=Path(git('rev-parse','--git-path','index').stdout.decode().strip());index=index if index.is_absolute() else I/index
    headpath=Path(git('rev-parse','--git-path','HEAD').stdout.decode().strip());headpath=headpath if headpath.is_absolute() else I/headpath
    indexraw=index.read_bytes();before_index=file_state(index);before_head=file_state(headpath)
    predicted=[]
    with tempfile.TemporaryDirectory(prefix='reviewed_metadata_private_index_') as temporary:
        for number in range(2):
            private=Path(temporary)/('index_'+str(number));private.write_bytes(indexraw);e={'GIT_INDEX_FILE':str(private)}
            git('read-tree','-m',parent,head,a.incoming,env=e);predicted.append(git('write-tree',env=e).stdout.decode().strip())
    need(predicted[0]==predicted[1],'two actual copied-index read-tree results agree')
    need(file_state(index)==before_index and file_state(headpath)==before_head,'actual real index bytes identity times and HEAD file unchanged')
    result={'kind':'root_reviewed_metadata_incoming_commit_check','passed':True,'root_actual_merge_or_push':False,'head':head,'incoming':a.incoming,'incoming_parent':parent,'predicted_merge_tree':predicted[0],'actual_copied_index_dry_runs':2,'real_index_reads_after_late_snapshot':0,'independent_review':{'path':str(a.review),'sha256':a.review_sha256},'source':{'path':str(Path(__file__).resolve()),'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},'files':files,'checks':CHECKS,'check_count':len(CHECKS),'seconds':time.monotonic()-started,'actual_cpu_job':os.environ['SLURM_JOB_ID'],'actual_cpu_step':os.environ['SLURM_STEP_ID'],'NUM_GPU_device_or_library_operations':False}
    need(result['seconds']<285,'complete CPU metadata check below five minutes')
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    print(json.dumps({k:result[k] for k in ['passed','head','incoming','predicted_merge_tree','check_count','seconds','actual_cpu_step']}))
if __name__=='__main__':main()
