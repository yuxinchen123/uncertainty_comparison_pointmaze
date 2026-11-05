"""Check a synthetic non-direct parent exit/reap/reuse interleaving on owned CPU only."""
import hashlib,json,os,socket,sys,time
from pathlib import Path
from types import SimpleNamespace
W=Path(__file__).resolve().parents[1];S=W/'10_jax_exploration_platform/scripts/atari_pixel_campaign';R=W/'.review_metadata'
F=R/'direct_nsight_producer_source_freeze_author_4_release_and_kernel.json';f=json.loads(F.read_text())
assert os.environ['SLURM_JOB_ID']=='6783003' and os.environ['SLURM_STEP_ID'].isdecimal()
assert all(hashlib.sha256((W/p).read_bytes()).hexdigest()==h for p,h in f['files'].items())
sys.path.insert(0,str(S));import direct_tool_descendants as source
scope=source.OwnedToolScope.__new__(source.OwnedToolScope)
scope.pid=100;scope.uid=888;scope.cgroup='synthetic common owned cgroup'
scope.handles={101:'pinned tool root',102:'pinned original non-direct waiter'}
scope.nodes={101:{'initial':{'process_start_time':'20'},'last':{},'positive_parent_pid':100},102:{'initial':{'process_start_time':'30'},'last':{},'positive_parent_pid':101}}
scope.events=[];scope.failed=False
scope.observer=SimpleNamespace(handle_exited=lambda handle:False,open_process_handle=lambda pid:'synthetic new child handle',close_process_handle=lambda handle:None)
reads=[]
def lists(pid):
    """Only the non-direct parent may be reaped concurrently by its actual tool parent."""
    if pid==100:return {101},[{'fixture':'producer sees actual own tool root'}]
    if pid==101:return {102},[{'fixture':'tool root lists its original waiter before the waiter exits'}]
    if pid==102:
        reads.append({'known_non_direct_parent':102,'known_parent_birth':'30','actual_parent_birth_now':'99',
            'interleaving':'waiter exits; its still-live tool parent101 reaps it; PID102 is reused before child-list read',
            'new_same_uid_same_cgroup_child':303})
        return {303},[{'fixture':'new unrelated same-UID process at reused waiter PID102'}]
    return set(),[{'fixture':'no children'}]
def snapshot(pid):
    """The original identities remain valid until their positive edge was queued."""
    if pid in (101,102):return {'pid':pid,'parent_pid':100 if pid==101 else101,
        'process_start_time':'20' if pid==101 else'30','state':'S'}
    return {'pid':303,'parent_pid':102,'process_start_time':'100','state':'S','uids':[888]*4,'cgroup':scope.cgroup}
scope.child_lists=lists;scope.snapshot=snapshot
scope.scan()
record={'kind':'independent_non_direct_parent_reuse_interleaving_control','source_freeze':{'path':str(F),'sha256':hashlib.sha256(F.read_bytes()).hexdigest()},
    'job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],'hostname':socket.gethostname(),'reviewed_at_seconds':time.time(),
    'synthetic_interleaving':reads,'unrelated_same_uid_child_was_pinned':303 in scope.handles,'actual_processes_or_device_files_touched':0,
    'source_unchanged':all(hashlib.sha256((W/p).read_bytes()).hexdigest()==h for p,h in f['files'].items()),'gpu_operations':0,'scientific_credit':0,
    'note':'This corrects the first abbreviated control to place PID reuse below the tool root, where a different process can reap the known parent concurrently. No actual PID reuse or foreign process operation occurred.'}
p=R/('direct_scope_parent_birth_independent_'+os.environ['SLURM_JOB_ID']+'_'+os.environ['SLURM_STEP_ID']+'.json');p.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'receipt':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'foreign_same_uid_child_pinned':record['unrelated_same_uid_child_was_pinned']}))
assert record['source_unchanged'] and record['unrelated_same_uid_child_was_pinned']
