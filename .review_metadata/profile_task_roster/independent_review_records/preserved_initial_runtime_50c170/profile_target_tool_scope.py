"""Discard a failed child-list sample only after a pinned parent exit and fresh adoption proof.

The accepted component scope stays unchanged. New profile-source qualifications
and admitted profiles use this explicit adapter and its additional saved reader.
"""
import copy
import errno
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import sys
import time

NAME='profile_target_tool_scope.py'
PREFIX='10_jax_exploration_platform/scripts/atari_pixel_campaign/'
PROTOCOL='profile-target-pinned-parent-exit-observation-version-1'
TASK_PROTOCOL='profile-target-known-parent-task-roster-observation-version-1'
MAX_TASK_ROSTER_ATTEMPTS=4
MAX_TASK_ROSTER_RETRY_SECONDS=.5
KINDS={'full_original_profile_source_target_qualification','selected_configuration_profile'}
EXIT_REFUSALS={
    'the pinned known parent exited before child-list access',
    'the known parent handle exited during child-list access',
    'the known parent exited before its child lists could be accepted',
}
_CLASSES={}


def require(value,message):
    if not value:raise ValueError(message)


def same(first,second):
    return type(first) is type(second) and json.dumps(first,sort_keys=True,allow_nan=False)==json.dumps(second,sort_keys=True,allow_nan=False)


def source(plan,name):
    path=Path(__file__).with_name(name)
    raw=path.read_bytes();expected=plan['helpers']['files'].get(PREFIX+name)
    require(path.resolve(strict=True)==path and hashlib.sha256(raw).hexdigest()==expected,
        'the explicit profile scope source changed: '+name)
    return path,raw,expected


def adjacent(plan,name):
    path,raw,expected=source(plan,name+'.py')
    key='profile_target_'+hashlib.sha256((str(path)+expected).encode()).hexdigest()
    if key not in sys.modules:
        spec=importlib.util.spec_from_file_location(key,path);module=importlib.util.module_from_spec(spec)
        exec(compile(raw,str(path),'exec'),module.__dict__);module._profile_target_executed_sha256=expected;sys.modules[key]=module
    module=sys.modules[key]
    require(Path(module.__file__)==path and module._profile_target_executed_sha256==expected,'the staged scope namespace changed')
    return module


def base_module(plan):
    require(plan.get('kind') in KINDS,'only a separately qualified profile-source scope may use this adapter')
    source(plan,NAME)
    return adjacent(plan,'direct_tool_descendants')


def handle_state(plan,descriptor,pid,observer):
    """Record only this process's already retained pidfd, never reopen the departed PID."""
    require(type(descriptor) is int and descriptor>=0 and observer.handle_exited(descriptor) is True,
        'a positively retained exited process handle is required')
    before=os.fstat(descriptor)
    require(os.readlink('/proc/self/fd/'+str(descriptor))=='anon_inode:[pidfd]' and not os.get_inheritable(descriptor),
        'the retained process handle changed its kernel type or close-on-exec flag')
    with Path('/proc/self/fdinfo/'+str(descriptor)).open() as stream:raw=stream.read(16385)
    parsed=adjacent(plan,'linux_pidfd_process_control').parse_pidfd_fdinfo(raw,before.st_ino,pid)
    after=os.fstat(descriptor)
    require((before.st_dev,before.st_ino)==(after.st_dev,after.st_ino)
        and observer.handle_exited(descriptor) is True,'the retained exited handle changed during observation')
    return {'descriptor':descriptor,'device':before.st_dev,'inode':before.st_ino,'raw_fdinfo':raw,
        'fdinfo_sha256':hashlib.sha256(raw.encode()).hexdigest(),'pid':parsed['pid'],
        'close_on_exec':True,'handle_exited':True,'observed_at_seconds':time.time()}


class KnownPinnedParentExit(Exception):
    """A discarded sample, not an accepted empty child list."""
    def __init__(self,record):
        super().__init__('the pinned parent exited; its child-list sample is invalid')
        self.record=record


class EnumeratedTaskDisappeared(Exception):
    """A positively owned parent's partial task collection is invalid and must never be used."""
    def __init__(self,record):
        super().__init__('the enumerated task disappeared before its exact directory open')
        self.record=record


class VerifiedTaskRosterChanged(Exception):
    """A complete pass saw a different task roster and therefore proves no complete child list."""
    def __init__(self,record):
        super().__init__('the verified live parent task roster changed during collection')
        self.record=record


def open_enumerated_task(task_directory,name):
    """Open only this exact name from an already captured pinned-parent task roster."""
    return os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=task_directory)


def directory_identity(descriptor):
    """Retain actual directory inode/device/owner fields without interpreting a numeric PID."""
    value=os.fstat(descriptor)
    require(stat.S_ISDIR(value.st_mode),'the retained process/task descriptor is not a directory')
    return {'device':value.st_dev,'inode':value.st_ino,'uid':value.st_uid,'mode':value.st_mode,'directory':True}


def task_names(task_directory):
    """Keep one bounded canonical task roster read through the retained task directory."""
    rows=os.listdir(task_directory)
    require(0<len(rows)<=4096 and all(type(value) is str and value.isdecimal()
        and str(int(value))==value and int(value)>0 for value in rows),'the actual task roster is malformed or oversized')
    return sorted(rows,key=int)


def live_handle_state(plan,descriptor,pid,observer):
    """Read this retained own pidfd while it still denotes the positively pinned live parent."""
    require(type(descriptor) is int and descriptor>=0 and observer.handle_exited(descriptor) is False,
        'the task observation requires its retained live parent handle')
    before=os.fstat(descriptor)
    require(os.readlink('/proc/self/fd/'+str(descriptor))=='anon_inode:[pidfd]' and not os.get_inheritable(descriptor),
        'the retained live parent handle changed type or close-on-exec state')
    with Path('/proc/self/fdinfo/'+str(descriptor)).open() as stream:raw=stream.read(16385)
    parsed=adjacent(plan,'linux_pidfd_process_control').parse_pidfd_fdinfo(raw,before.st_ino,pid)
    after=os.fstat(descriptor)
    require(parsed['pid']==pid and (before.st_dev,before.st_ino)==(after.st_dev,after.st_ino)
        and observer.handle_exited(descriptor) is False,'the retained parent exited or changed during the task observation')
    return {'descriptor':descriptor,'device':before.st_dev,'inode':before.st_ino,'raw_fdinfo':raw,
        'fdinfo_sha256':hashlib.sha256(raw.encode()).hexdigest(),'pid':pid,'close_on_exec':True,
        'handle_exited':False,'observed_at_seconds':time.time()}


def scope_class(base):
    if base in _CLASSES:return _CLASSES[base]
    class ProfileToolScope(base.OwnedToolScope):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self._base_module=base
            self.parent_exit_records=[]
            self.parent_exit_failures=[]
            self.task_roster_records=[]
            self.task_roster_failures=[]
            self.task_roster_collection_sequence=0

        def task_failure(self,record,error):
            """Preserve an unsupported partial task observation without converting it to a child list."""
            self.failed=True
            if len(self.task_roster_failures)<32:
                value={**record,'outcome':'unverified_task_roster_observation','child_list_sample_valid':False,
                    'error_type':type(error).__name__,'error':str(error),'failed_at_seconds':time.time()}
                self.task_roster_failures.append(value)
                self.observer.publish(self.directory/('tool_task_roster_failure_'+str(len(self.task_roster_failures))+'.json'),value)

        def strict_task_child_lists_once(self,pid):
            """Read every current task, preserving original owner/birth checks and rejecting partial rosters."""
            started=time.time();birth,handle=self.known_parent(pid)
            directory=os.open(base.PROC_ROOT/str(pid),os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
            task_directory=None
            try:
                parent_directory=directory_identity(directory)
                require(parent_directory['uid']==self.uid,'the pinned parent directory changed owner before task collection')
                before=self.parent_identity(directory,pid,birth,handle)
                task_directory=os.open('task',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=directory)
                task_directory_identity=directory_identity(task_directory)
                require(task_directory_identity['uid']==self.uid,'the retained parent task directory changed owner')
                tids=task_names(task_directory)
                require(str(pid) in tids,'the live parent leader is absent from its actual task roster')
                rows,children,count=[],set(),0
                for index,name in enumerate(tids):
                    try:thread=open_enumerated_task(task_directory,name)
                    except FileNotFoundError as error:
                        # Only this exact enumerated task-directory open may have a disappearance outcome.
                        require(error.errno==errno.ENOENT and error.filename==name,'the failed task open is not the exact enumerated ENOENT')
                        initial=self.owner if pid==self.pid else self.nodes[pid]['initial']
                        record={'schema_version':1,'protocol':TASK_PROTOCOL,'outcome':'enumerated_task_open_requires_fresh_complete_read',
                            'parent_pid':pid,'parent_birth':birth,'initial_identity':copy.deepcopy(initial),
                            'parent_directory':parent_directory,'task_directory':task_directory_identity,
                            'parent_before':before,'task_roster_before':tids,'failed_task':int(name),
                            'failed_task_birth':None,'failed_task_owner_independently_observed':False,
                            'partial_rows':copy.deepcopy(rows),'partial_child_pids':sorted(children),
                            'failure':{'operation':'open_enumerated_task_directory','error_type':type(error).__name__,
                                'errno':error.errno,'filename':error.filename,'enumerated_index':index},
                            'child_list_sample_valid':False,'collection_started_at_seconds':started}
                        try:
                            after=self.parent_identity(directory,pid,birth,handle)
                            current_names=task_names(task_directory)
                            require(name not in current_names,'the missing task identifier is present again; its generation is unverified')
                            require(directory_identity(directory)==parent_directory and directory_identity(task_directory)==task_directory_identity,
                                'the pinned parent/task directory changed during failed task collection')
                            record.update(parent_after_failure=after,task_roster_after_failure=current_names,
                                retained_live_handle=live_handle_state(self.plan,handle,pid,self.observer),observed_at_seconds=time.time())
                        except BaseException as unverified:
                            self.task_failure(record,unverified)
                            raise ValueError('the vanished task lacks its unchanged live-parent and absent-task proof') from unverified
                        raise EnumeratedTaskDisappeared(record) from error
                    try:
                        thread_directory=directory_identity(thread)
                        require(thread_directory['uid']==self.uid,'the opened enumerated task is not owned by this user')
                        thread_before=base.bounded(thread,'stat').decode();first=base.parsed_identity(thread_before)
                        require(first['pid']==int(name),'the pinned enumerated task changed identity')
                        raw=base.bounded(thread,'children')
                        thread_after=base.bounded(thread,'stat').decode();last=base.parsed_identity(thread_after)
                        require(last['pid']==first['pid'] and last['process_start_time']==first['process_start_time']
                            and directory_identity(thread)==thread_directory,'the pinned task identity changed during its child-list read')
                        count+=len(raw)+len(thread_before.encode())+len(thread_after.encode())
                        require(count<=base.MAX_BYTES and all(value.isdecimal() and str(int(value))==value and int(value)>0
                            for value in raw.decode('ascii').split()),'actual task child evidence is malformed or oversized')
                        children.update(int(value) for value in raw.decode('ascii').split())
                        rows.append({'tid':int(name),'raw':raw.decode('ascii'),'sha256':hashlib.sha256(raw).hexdigest(),
                            'thread_stat_before':thread_before,'thread_stat_after':thread_after,'thread_directory':thread_directory})
                    finally:os.close(thread)
                after=self.parent_identity(directory,pid,birth,handle)
                final_tids=task_names(task_directory)
                require(str(pid) in final_tids,'the live parent leader disappeared from its actual task roster')
                require(directory_identity(directory)==parent_directory and directory_identity(task_directory)==task_directory_identity,
                    'the pinned parent/task directory changed during complete task inspection')
                if final_tids!=tids:
                    # A changed roster invalidates this entire pass even when each earlier open succeeded.
                    initial=self.owner if pid==self.pid else self.nodes[pid]['initial']
                    raise VerifiedTaskRosterChanged({'schema_version':1,'protocol':TASK_PROTOCOL,
                        'outcome':'verified_task_roster_change_requires_fresh_complete_read','parent_pid':pid,'parent_birth':birth,
                        'initial_identity':copy.deepcopy(initial),'parent_directory':parent_directory,'task_directory':task_directory_identity,
                        'parent_before':before,'parent_after_failure':after,'task_roster_before':tids,'task_roster_after_failure':final_tids,
                        'failed_task':None,'failed_task_birth':None,'failed_task_owner_independently_observed':False,
                        'partial_rows':copy.deepcopy(rows),'partial_child_pids':sorted(children),
                        'failure':{'operation':'verified_task_roster_changed_after_complete_pass','error_type':'VerifiedTaskRosterChanged',
                            'errno':None,'filename':None,'enumerated_index':None},'child_list_sample_valid':False,
                        'retained_live_handle':live_handle_state(self.plan,handle,pid,self.observer),
                        'collection_started_at_seconds':started,'observed_at_seconds':time.time()})
                rows[0]['parent_read_identity']={'expected_pid':pid,'expected_birth':birth,
                    'proc_directory_device':parent_directory['device'],'proc_directory_inode':parent_directory['inode'],
                    'before':before,'after':after,'pidfd_live_before':True,'pidfd_live_after':True}
                proof={'parent_pid':pid,'parent_birth':birth,'parent_directory':parent_directory,'task_directory':task_directory_identity,
                    'task_roster_before':tids,'task_roster_after':final_tids,'pids':sorted(children),'rows':copy.deepcopy(rows),
                    'parent_before':before,'parent_after':after,'retained_live_handle':live_handle_state(self.plan,handle,pid,self.observer),
                    'started_at_seconds':started,'finished_at_seconds':time.time()}
                return children,rows,proof
            finally:
                if task_directory is not None:os.close(task_directory)
                os.close(directory)

        def child_lists_with_task_retry(self,pid):
            """Use only a fresh complete collection after a bounded exact task-open disappearance."""
            self.task_roster_collection_sequence+=1
            sequence=self.task_roster_collection_sequence
            pending=[];retry_started=None
            for attempt in range(1,MAX_TASK_ROSTER_ATTEMPTS+1):
                if self.deadline is not None:self.deadline.remaining('total')
                try:
                    children,rows,fresh=self.strict_task_child_lists_once(pid)
                except (EnumeratedTaskDisappeared,VerifiedTaskRosterChanged) as outcome:
                    if retry_started is None:retry_started=time.monotonic()
                    record=outcome.record
                    record.update(collection_sequence=sequence,failed_attempt=attempt,
                        maximum_attempts=MAX_TASK_ROSTER_ATTEMPTS,maximum_retry_seconds=MAX_TASK_ROSTER_RETRY_SECONDS)
                    pending.append(record)
                    if attempt==MAX_TASK_ROSTER_ATTEMPTS or time.monotonic()-retry_started>=MAX_TASK_ROSTER_RETRY_SECONDS:
                        error=ValueError('the exact task disappearance has no bounded fresh complete parent collection')
                        for row in pending:self.task_failure(row,error)
                        raise error from outcome
                    time.sleep(.01)
                    continue
                except BaseException as error:
                    for record in pending:self.task_failure(record,error)
                    raise
                if not pending:return children,rows
                elapsed=time.monotonic()-retry_started
                try:
                    require(elapsed<=MAX_TASK_ROSTER_RETRY_SECONDS,'fresh task collection exceeded the one actual retry budget')
                    require(all(record['failed_task'] is None or str(record['failed_task']) not in fresh['task_roster_before'] for record in pending),
                        'a disappeared task identifier reappeared before reconciliation')
                    fresh_births={row['tid']:base.parsed_identity(row['thread_stat_before'])['process_start_time'] for row in fresh['rows']}
                    for record in pending:
                        for row in record['partial_rows']:
                            require(row['tid'] not in fresh_births or base.parsed_identity(row['thread_stat_before'])['process_start_time']==fresh_births[row['tid']],
                                'a previously observed task identifier changed birth before fresh reconciliation')
                    identities={}
                    for child in sorted(children):
                        actual=self.pin(child,pid)
                        require(actual['parent_pid']==pid,'the fresh complete child list lost its positive owned parent edge')
                        identities[str(child)]=copy.deepcopy(actual)
                    fresh.update(identities=identities,complete_attempt=attempt,bounded_retry_elapsed_seconds=elapsed)
                    for record in pending:
                        require(len(self.task_roster_records)<base.MAX_EVENTS,'too many actual task-roster observations')
                        record.update(outcome='invalid_task_collection_fresh_complete_parent_reconciled',fresh_complete_collection=copy.deepcopy(fresh))
                        path=self.directory/('tool_task_roster_observation_'+str(len(self.task_roster_records)+1)+'.json')
                        self.observer.publish(path,record);reference=self.observer.receipt(path);self.task_roster_records.append(reference)
                        self.event('known_parent_task_roster_reconciled',pid=pid,birth=record['parent_birth'],
                            task=record['failed_task'],collection_sequence=sequence,observation=reference,discarded_child_list_valid=False)
                    return children,rows
                except BaseException as error:
                    for record in pending:self.task_failure(record,error)
                    raise

        def child_lists(self,pid):
            """Retain every original refusal except a separately typed known-handle exit."""
            try:return self.child_lists_with_task_retry(pid)
            except (ValueError,FileNotFoundError,ProcessLookupError) as error:
                eligible=pid!=self.pid and pid in self.nodes and pid in self.handles
                exited=eligible and self.observer.handle_exited(self.handles[pid]) is True
                exact_refusal=type(error) is ValueError and str(error) in EXIT_REFUSALS
                disappeared=isinstance(error,(FileNotFoundError,ProcessLookupError))
                if eligible and exited and (exact_refusal or disappeared):
                    node=self.nodes[pid];initial=node['initial']
                    require(initial['pid']==pid and initial['uids']==[self.uid]*4 and initial['cgroup']==self.cgroup,
                        'the departed parent lacks its positive own-process identity')
                    record={'schema_version':1,'protocol':PROTOCOL,'outcome':'pinned_parent_exit_requires_fresh_root_proof',
                        'child_list_sample_valid':False,'parent_pid':pid,'parent_birth':initial['process_start_time'],
                        'positive_parent_pid':node['positive_parent_pid'],'initial_identity':copy.deepcopy(initial),
                        'handle_state':handle_state(self.plan,self.handles[pid],pid,self.observer),
                        'refusal':{'error_type':type(error).__name__,'message':str(error)},'observed_at_seconds':time.time()}
                    raise KnownPinnedParentExit(record) from error
                failure={'schema_version':1,'protocol':PROTOCOL,'outcome':'unverified_parent_observation',
                    'parent_pid':pid,'parent_known':eligible,'retained_handle_exited':bool(exited),
                    'child_list_sample_valid':False,'error_type':type(error).__name__,'error':str(error),
                    'observed_at_seconds':time.time()}
                self.failed=True
                if len(self.parent_exit_failures)<32:
                    self.parent_exit_failures.append(failure)
                    self.observer.publish(self.directory/('tool_parent_exit_failure_'+str(len(self.parent_exit_failures))+'.json'),failure)
                raise

        def reconcile_parent_exit(self,outcome):
            """Use a new valid producer read and positive adopted identities, never the discarded sample."""
            record=outcome.record
            require(len(self.parent_exit_records)<base.MAX_NODES,'parent-exit observation count exceeded its exact scope bound')
            before=time.time()
            children,rows=super().child_lists(self.pid)
            identities={}
            for pid in sorted(children):
                actual=self.pin(pid,self.pid)
                require(actual['parent_pid']==self.pid,'a fresh producer child is not positively adopted by this producer')
                identities[str(pid)]=copy.deepcopy(actual)
            record.update(outcome='pinned_parent_exited_fresh_root_reconciled',
                fresh_producer_children={'pid':self.pid,'birth':self.birth,'pids':sorted(children),'rows':rows,
                    'identities':identities,'started_at_seconds':before,'finished_at_seconds':time.time()})
            self.observer.publish(self.directory/('tool_parent_exit_observation_'+str(len(self.parent_exit_records)+1)+'.json'),record)
            self.parent_exit_records.append(self.observer.receipt(self.directory/('tool_parent_exit_observation_'+str(len(self.parent_exit_records)+1)+'.json')))
            self.event('known_pinned_parent_exit_reconciled',pid=record['parent_pid'],birth=record['parent_birth'],
                observation=self.parent_exit_records[-1],discarded_child_list_valid=False)
            return children,rows,identities

        def scan(self):
            queue,seen,root_rows,root_children=[self.pid],set(),None,None
            while queue:
                parent=queue.pop(0)
                if parent in seen:continue
                seen.add(parent)
                try:
                    children,rows=self.child_lists(parent)
                except KnownPinnedParentExit as outcome:
                    try:children,rows,identities=self.reconcile_parent_exit(outcome)
                    except BaseException as error:
                        self.failed=True
                        if len(self.parent_exit_failures)<32:
                            self.parent_exit_failures.append({'outcome':'fresh_root_reconciliation_failed','parent_pid':parent,
                                'error_type':type(error).__name__,'error':str(error),'child_list_sample_valid':False})
                            self.observer.publish(self.directory/('tool_parent_exit_failure_'+str(len(self.parent_exit_failures))+'.json'),self.parent_exit_failures[-1])
                        raise
                    root_rows,root_children=rows,children
                    for pid in children:
                        if identities[str(pid)]['state'] not in {'Z','X','x'}:queue.append(pid)
                    continue
                if parent==self.pid:root_rows,root_children=rows,children
                for pid in children:
                    try:
                        actual=self.pin(pid,parent)
                        if actual['state'] not in {'Z','X','x'}:queue.append(pid)
                    except (FileNotFoundError,ProcessLookupError):
                        self.event('positive_child_disappeared_before_pin',pid=pid,parent_pid=parent)
            require(root_rows is not None and root_children is not None,'actual producer child lists were not read')
            return root_children,root_rows

        def finish(self,failed):
            super().finish(failed)
            output=self.directory/'tool_parent_exit_observations.json'
            if not output.exists():
                self.observer.publish(output,{'schema_version':1,'protocol':PROTOCOL,
                    'adapter_sha256':self.plan['helpers']['files'][PREFIX+NAME],
                    'base_sha256':self.plan['helpers']['files'][PREFIX+'direct_tool_descendants.py'],
                    'producer_pid':self.pid,'producer_birth':self.birth,'records':self.parent_exit_records,
                    'failure_count':len(self.parent_exit_failures),'scope_terminal':self.observer.receipt(self.directory/'tool_descendant_scope_terminal.json'),
                    'scope_restored':self.observer.receipt(self.directory/'tool_descendant_scope_restored.json'),
                    'discarded_child_list_samples_accepted_as_valid':False,'scientific_credit':0,'admission_granted':False})
            task_output=self.directory/'tool_task_roster_observations.json'
            if not task_output.exists():
                self.observer.publish(task_output,{'schema_version':1,'protocol':TASK_PROTOCOL,
                    'adapter_sha256':self.plan['helpers']['files'][PREFIX+NAME],
                    'base_sha256':self.plan['helpers']['files'][PREFIX+'direct_tool_descendants.py'],
                    'producer_pid':self.pid,'producer_birth':self.birth,'records':self.task_roster_records,
                    'failure_count':len(self.task_roster_failures),'scope_terminal':self.observer.receipt(self.directory/'tool_descendant_scope_terminal.json'),
                    'scope_restored':self.observer.receipt(self.directory/'tool_descendant_scope_restored.json'),
                    'discarded_child_list_samples_accepted_as_valid':False,'scientific_credit':0,'admission_granted':False})
    _CLASSES[base]=ProfileToolScope
    return ProfileToolScope


def create_scope(directory,plan,observer,owner,check_descriptor,**kwargs):
    return scope_class(base_module(plan))(directory,plan,observer,owner,check_descriptor,**kwargs)


def validate_parent_exit_record(record,terminal,producer,base,observer,open_events,parser,*,producer_terminal_time):
    """Validate the separate typed outcome and fresh source-bound positive adoption proof."""
    require(type(record.get('schema_version')) is int and record['schema_version']==1 and record.get('protocol')==PROTOCOL
        and record.get('outcome')=='pinned_parent_exited_fresh_root_reconciled'
        and record.get('child_list_sample_valid') is False,'an invalid parent observation cannot become child-list evidence')
    refusal=record['refusal']
    require((refusal.get('error_type')=='ValueError' and refusal.get('message') in EXIT_REFUSALS)
        or refusal.get('error_type') in {'FileNotFoundError','ProcessLookupError'},'the accepted typed observation was not a known exited-parent refusal')
    pid,birth=record['parent_pid'],record['parent_birth']
    require(type(pid) is int and pid!=producer['pid'] and str(pid) in terminal['positive_nodes'],
        'only a positively pinned nonproducer may have an exit outcome')
    node=terminal['positive_nodes'][str(pid)]
    require(node['initial']['process_start_time']==birth and same(node['initial'],record['initial_identity'])
        and node['positive_parent_pid']==record['positive_parent_pid'],'the departed pinned generation changed')
    state=record['handle_state'];parsed=parser.parse_pidfd_fdinfo(state['raw_fdinfo'],state['inode'],pid)
    require(type(state['descriptor']) is int and state['descriptor']>=0 and type(state['device']) is int
        and state['handle_exited'] is True and state['close_on_exec'] is True
        and state['pid']==parsed['pid'] and hashlib.sha256(state['raw_fdinfo'].encode()).hexdigest()==state['fdinfo_sha256'],
        'the observed exited pidfd identity is incomplete or changed')
    opened=[row for row in open_events if row.get('operation')=='pidfd_open' and row.get('descriptor')==state['descriptor']
        and row.get('opened_pid')==pid and row['identity']['device']==state['device'] and row['identity']['inode']==state['inode']]
    require(len(opened)==1 and type(opened[0].get('open_generation')) is int and opened[0]['open_generation']>0
        and opened[0]['observed_at_seconds']<=state['observed_at_seconds'],
        'the exit outcome is not joined to its actual retained native-open generation')
    closed=[row for row in open_events if row.get('operation')=='close'
        and row.get('descriptor')==state['descriptor'] and row.get('opened_pid')==pid
        and row.get('open_generation')==opened[0]['open_generation']
        and row['identity']['device']==state['device'] and row['identity']['inode']==state['inode']]
    closure_time=terminal['terminal']['observed_at_seconds']
    require(len(closed)==1 and all(type(value) in (int,float) and math.isfinite(value)
        for value in (state['observed_at_seconds'],closure_time,closed[0]['observed_at_seconds'],producer_terminal_time))
        and state['observed_at_seconds']<=closure_time<=closed[0]['observed_at_seconds']<=producer_terminal_time,
        'the parent observation must precede kernel closure and the same-generation actual close/native terminal')
    fresh=record['fresh_producer_children'];pids=fresh['pids'];rows=fresh['rows']
    require(fresh['pid']==producer['pid'] and fresh['birth']==producer['process_start_time']
        and type(pids) is list and all(type(value) is int and value>0 for value in pids) and pids==sorted(set(pids))
        and set(fresh['identities'])=={str(value) for value in pids} and type(rows) is list and len(rows)==1,
        'the fresh producer child-list proof is incomplete')
    row=rows[0];proof=row['parent_read_identity']
    require(row['tid']==producer['pid'] and sorted(int(value) for value in row['raw'].split())==pids
        and hashlib.sha256(row['raw'].encode()).hexdigest()==row['sha256']
        and proof['expected_pid']==producer['pid'] and proof['expected_birth']==producer['process_start_time']
        and all(type(proof[key]) is int and proof[key]>0 for key in ('proc_directory_device','proc_directory_inode'))
        and proof['pidfd_live_before'] is proof['pidfd_live_after'] is True,
        'the fresh child bytes are not joined to the actual live producer')
    for identity in (proof['before'],proof['after']):
        actual=base.parsed_identity(identity['stat'])
        require(actual['pid']==identity['pid']==producer['pid'] and actual['process_start_time']==identity['process_start_time']==producer['process_start_time']
            and actual['state'] not in {'Z','X','x'} and identity['pidfd_live'] is True
            and observer.parsed_uids(identity['status'])==identity['uids']==producer['uids']
            and identity['cgroup']==producer['cgroup'],'the fresh producer read lost its own identity')
    first,last=base.parsed_identity(row['thread_stat_before']),base.parsed_identity(row['thread_stat_after'])
    require(first['pid']==last['pid']==producer['pid'] and first['process_start_time']==last['process_start_time']==producer['process_start_time'],
        'the fresh producer thread changed during child observation')
    for key,identity in fresh['identities'].items():
        before,after=base.parsed_identity(identity['stat_before']),base.parsed_identity(identity['stat_after'])
        require(before['pid']==after['pid']==identity['pid'] and before['process_start_time']==after['process_start_time']==identity['process_start_time']
            and before['parent_pid']==after['parent_pid']==identity['parent_pid']
            and observer.parsed_uids(identity['status'])==observer.parsed_uids(identity['status_after'])==identity['uids']==producer['uids']
            and identity['cgroup']==producer['cgroup'],'the adopted scoped identity changed raw owner, birth, parent or cgroup')
        require(str(identity['pid'])==key and identity['parent_pid']==producer['pid'] and key in terminal['positive_nodes']
            and identity['process_start_time']==terminal['positive_nodes'][key]['initial']['process_start_time'],
            'a fresh adopted child lacks its exact positive birth/parent edge')
    times=[record['observed_at_seconds'],state['observed_at_seconds'],fresh['started_at_seconds'],fresh['finished_at_seconds']]
    require(all(type(value) in (int,float) and math.isfinite(value) for value in times)
        and times[1]<=times[0]<=times[2]<=times[3]<=closure_time, 'the exit and fresh adoption observations have an invalid chronology')


def validate_directory_record(value,uid):
    """Check the saved positive own-directory identity without opening a historical proc path."""
    require(type(value) is dict and set(value)=={'device','inode','uid','mode','directory'}
        and all(type(value[key]) is int for key in ('device','inode','uid','mode'))
        and value['device']>0 and value['inode']>0 and value['uid']==uid
        and stat.S_ISDIR(value['mode']) and value['directory'] is True,'the saved parent/task directory identity is incomplete')


def validate_parent_read(record,pid,birth,producer,base,observer):
    """Rebuild the exact pinned live parent's PID, birth, four UIDs and cgroup from raw bytes."""
    actual=base.parsed_identity(record['stat'])
    require(type(record['pid']) is int and actual['pid']==record['pid']==pid
        and actual['process_start_time']==record['process_start_time']==birth
        and actual['state'] not in {'Z','X','x'} and record['pidfd_live'] is True
        and observer.parsed_uids(record['status'])==record['uids']==producer['uids']
        and record['cgroup']==producer['cgroup'],'the saved task collection lost its known live parent identity')


def validate_task_rows(rows,names,uid,base):
    """Recompute each exact task generation and its raw child bytes, never a partial collection summary."""
    require(type(rows) is list and len(rows)==len(names)
        and [row['tid'] for row in rows]==[int(name) for name in names],'task rows differ from the exact enumerated roster')
    children=set();count=0;births={}
    for row,name in zip(rows,names):
        require(type(row['tid']) is int and row['tid']==int(name),'a task PID changed type or identity')
        first,last=base.parsed_identity(row['thread_stat_before']),base.parsed_identity(row['thread_stat_after'])
        require(first['pid']==last['pid']==row['tid'] and first['process_start_time']==last['process_start_time'],
            'the saved task generation changed across its child-list read')
        validate_directory_record(row['thread_directory'],uid)
        raw=row['raw'].encode('ascii');values=raw.decode().split()
        require(hashlib.sha256(raw).hexdigest()==row['sha256'] and all(value.isdecimal()
            and str(int(value))==value and int(value)>0 for value in values),'saved task child bytes are malformed or changed')
        children.update(int(value) for value in values)
        count+=len(raw)+len(row['thread_stat_before'].encode())+len(row['thread_stat_after'].encode())
        require(count<=base.MAX_BYTES,'the saved task collection exceeded its actual byte bound')
        births[row['tid']]=first['process_start_time']
    return sorted(children),births


def validate_roster_names(names,pid):
    """Require a canonical bounded roster that includes the known live process leader."""
    require(type(names) is list and 0<len(names)<=4096 and all(type(value) is str and value.isdecimal()
        and str(int(value))==value and int(value)>0 for value in names)
        and names==sorted(set(names),key=int) and str(pid) in names,'the saved actual task roster is malformed or incomplete')


def validate_live_generation(states,pid,events,parser,closure_time,producer_terminal_time):
    """Join both live pidfd observations to the one actual native open/close generation."""
    first=states[0]
    for state in states:
        require(type(state.get('descriptor')) is int and state['descriptor']>=0 and type(state.get('device')) is int
            and type(state.get('inode')) is int and state['inode']>0 and state.get('handle_exited') is False
            and state.get('close_on_exec') is True and state.get('pid')==pid
            and hashlib.sha256(state['raw_fdinfo'].encode()).hexdigest()==state['fdinfo_sha256']
            and parser.parse_pidfd_fdinfo(state['raw_fdinfo'],state['inode'],pid)['pid']==pid
            and all(state[key]==first[key] for key in ('descriptor','device','inode','pid')),
            'the saved task observation lacks its same retained live pidfd')
    windows=[]
    for opened in events:
        if not (opened.get('operation')=='pidfd_open' and opened.get('descriptor')==first['descriptor']
                and opened.get('opened_pid')==pid and opened.get('identity',{}).get('device')==first['device']
                and opened.get('identity',{}).get('inode')==first['inode']):continue
        closed=[row for row in events if row.get('operation')=='close' and row.get('descriptor')==first['descriptor']
            and row.get('opened_pid')==pid and row.get('open_generation')==opened.get('open_generation')
            and row.get('identity',{}).get('device')==first['device'] and row.get('identity',{}).get('inode')==first['inode']]
        if len(closed)!=1:continue
        times=[opened.get('observed_at_seconds'),*(state.get('observed_at_seconds') for state in states),
            closure_time,closed[0].get('observed_at_seconds'),producer_terminal_time]
        if all(type(value) in (int,float) and math.isfinite(value) for value in times) and times==sorted(times):
            windows.append(opened)
    require(len(windows)==1 and type(windows[0].get('open_generation')) is int and windows[0]['open_generation']>0,
        'the task collection is not inside one exact actual native-open/close/terminal generation')


def validate_task_roster_record(record,terminal,scope_begin,producer,base,observer,events,parser,*,producer_terminal_time):
    """Validate one discarded sample followed by a fresh complete same-parent collection."""
    require(type(record.get('schema_version')) is int and record['schema_version']==1 and record.get('protocol')==TASK_PROTOCOL
        and record.get('outcome')=='invalid_task_collection_fresh_complete_parent_reconciled'
        and record.get('child_list_sample_valid') is False and record.get('failed_task_birth') is None
        and record.get('failed_task_owner_independently_observed') is False,'a partial task sample cannot become accepted child evidence')
    pid,birth=record['parent_pid'],record['parent_birth']
    require(type(pid) is int and pid>0 and type(birth) is str and birth.isdecimal()
        and (pid==producer['pid'] or str(pid) in terminal['positive_nodes']),'the task observation names an unknown parent')
    initial=scope_begin['producer'] if pid==producer['pid'] else terminal['positive_nodes'][str(pid)]['initial']
    require(same(record['initial_identity'],initial) and initial['pid']==pid and initial['process_start_time']==birth
        and initial['uids']==producer['uids'] and initial['cgroup']==producer['cgroup'],'the task parent changed its actual positive generation')
    for key in ('parent_directory','task_directory'):validate_directory_record(record[key],producer['uids'][0])
    for key in ('parent_before','parent_after_failure'):validate_parent_read(record[key],pid,birth,producer,base,observer)
    names,after=record['task_roster_before'],record['task_roster_after_failure']
    validate_roster_names(names,pid);validate_roster_names(after,pid)
    refusal=record['failure'];operation=refusal.get('operation')
    if operation=='open_enumerated_task_directory':
        index=refusal['enumerated_index'];task=record['failed_task']
        require(refusal.get('error_type')=='FileNotFoundError' and type(refusal.get('errno')) is int
            and refusal['errno']==errno.ENOENT and type(index) is int and 0<=index<len(names)
            and type(task) is int and names[index]==str(task)==refusal['filename'] and str(task) not in after,
            'only the exact enumerated-task-directory ENOENT can have this typed outcome')
        partial_names=names[:index]
    else:
        require(operation=='verified_task_roster_changed_after_complete_pass'
            and refusal.get('error_type')=='VerifiedTaskRosterChanged' and refusal.get('errno') is None
            and refusal.get('filename') is None and refusal.get('enumerated_index') is None
            and record['failed_task'] is None and names!=after,'an unsupported read failure cannot become a verified roster change')
        partial_names=names
    partial,births=validate_task_rows(record['partial_rows'],partial_names,producer['uids'][0],base)
    require(same(partial,record['partial_child_pids']),'discarded partial child summaries differ from their actual rows')
    fresh=record['fresh_complete_collection'];fresh_names=fresh['task_roster_before']
    validate_roster_names(fresh_names,pid)
    require(fresh_names==fresh['task_roster_after'] and fresh['parent_pid']==pid and fresh['parent_birth']==birth
        and same(fresh['parent_directory'],record['parent_directory']) and same(fresh['task_directory'],record['task_directory'])
        and (record['failed_task'] is None or str(record['failed_task']) not in fresh_names),
        'the retry did not produce a fresh complete roster of the same parent')
    for key in ('parent_before','parent_after'):validate_parent_read(fresh[key],pid,birth,producer,base,observer)
    children,fresh_births=validate_task_rows(fresh['rows'],fresh_names,producer['uids'][0],base)
    require(same(children,fresh['pids']) and set(fresh['identities'])=={str(child) for child in children}
        and all(tid not in fresh_births or fresh_births[tid]==value for tid,value in births.items()),
        'the fresh collection reused a known task generation or changed its complete child union')
    proof=fresh['rows'][0]['parent_read_identity']
    require(proof['expected_pid']==pid and proof['expected_birth']==birth
        and proof['proc_directory_device']==record['parent_directory']['device'] and proof['proc_directory_inode']==record['parent_directory']['inode']
        and same(proof['before'],fresh['parent_before']) and same(proof['after'],fresh['parent_after'])
        and proof['pidfd_live_before'] is proof['pidfd_live_after'] is True,'the fresh whole roster lost its original pinned parent read')
    for key,identity in fresh['identities'].items():
        first,last=base.parsed_identity(identity['stat_before']),base.parsed_identity(identity['stat_after'])
        require(first['pid']==last['pid']==identity['pid'] and str(identity['pid'])==key
            and first['process_start_time']==last['process_start_time']==identity['process_start_time']
            and first['parent_pid']==last['parent_pid']==identity['parent_pid']==pid
            and observer.parsed_uids(identity['status'])==observer.parsed_uids(identity['status_after'])==identity['uids']==producer['uids']
            and identity['cgroup']==producer['cgroup'] and key in terminal['positive_nodes']
            and terminal['positive_nodes'][key]['initial']['process_start_time']==identity['process_start_time'],
            'the fresh collection lacks a positive own-child generation/parent edge')
    require(type(record['collection_sequence']) is int and record['collection_sequence']>0
        and type(record['failed_attempt']) is int and type(fresh['complete_attempt']) is int
        and 1<=record['failed_attempt']<fresh['complete_attempt']<=MAX_TASK_ROSTER_ATTEMPTS
        and type(record['maximum_attempts']) is int and record['maximum_attempts']==MAX_TASK_ROSTER_ATTEMPTS
        and type(record['maximum_retry_seconds']) is float and record['maximum_retry_seconds']==MAX_TASK_ROSTER_RETRY_SECONDS
        and type(fresh['bounded_retry_elapsed_seconds']) in (int,float) and math.isfinite(fresh['bounded_retry_elapsed_seconds'])
        and 0<=fresh['bounded_retry_elapsed_seconds']<=MAX_TASK_ROSTER_RETRY_SECONDS,'the actual fresh-roster retry bound changed')
    states=[record['retained_live_handle'],fresh['retained_live_handle']]
    validate_live_generation(states,pid,events,parser,terminal['terminal']['observed_at_seconds'],producer_terminal_time)
    times=[record['collection_started_at_seconds'],states[0]['observed_at_seconds'],record['observed_at_seconds'],
        fresh['started_at_seconds'],states[1]['observed_at_seconds'],fresh['finished_at_seconds'],terminal['terminal']['observed_at_seconds']]
    require(all(type(value) in (int,float) and math.isfinite(value) for value in times) and times==sorted(times),
        'the discarded sample and complete retry have an invalid actual chronology')


def validate_saved_scope(directory,plan,observer,producer,root_pid,root_birth,check_descriptor,*,diagnostic_deadline=None):
    base=base_module(plan)
    result=base.validate_saved_scope(directory,plan,observer,producer,root_pid,root_birth,check_descriptor,
        diagnostic_deadline=diagnostic_deadline)
    directory=Path(directory)
    manifest_ref=observer.receipt(directory/'tool_parent_exit_observations.json');manifest=observer.read_json(manifest_ref['path'])
    require(manifest.get('protocol')==PROTOCOL and type(manifest.get('schema_version')) is int and manifest['schema_version']==1
        and manifest['adapter_sha256']==plan['helpers']['files'][PREFIX+NAME]
        and manifest['base_sha256']==plan['helpers']['files'][PREFIX+'direct_tool_descendants.py']
        and manifest['producer_pid']==producer['pid'] and manifest['producer_birth']==producer['process_start_time']
        and manifest['scope_terminal']==result['tool_descendant_scope_terminal'] and manifest['scope_restored']==result['tool_descendant_scope_restored']
        and type(manifest['failure_count']) is int and manifest['failure_count']==0
        and manifest['discarded_child_list_samples_accepted_as_valid'] is False,'the new scope source/outcome manifest is incomplete')
    records=manifest['records'];require(type(records) is list and len(records)<=base.MAX_EVENTS,'too many or missing parent-exit records')
    require(not list(directory.glob('tool_parent_exit_failure_*.json')),'a failed parent observation invalidates the whole scope')
    terminal=observer.read_json(result['tool_descendant_scope_terminal']['path'])
    role=observer.read_json(directory/'profile_target_process_producer_terminal.json')
    require(role.get('role')=='producer' and role.get('completed') is True and role.get('kind')=='actual_profile_target_process_role_terminal',
        'the fresh scope requires its actual completed native producer role')
    parser=adjacent(plan,'linux_pidfd_process_control')
    expected_paths={str(directory/('tool_parent_exit_observation_'+str(index)+'.json')) for index in range(1,len(records)+1)}
    require({str(path) for path in directory.glob('tool_parent_exit_observation_*.json')}==expected_paths,
        'an actual parent-exit observation was omitted or added outside the manifest')
    events=[event for event in terminal['events'] if event.get('kind')=='known_pinned_parent_exit_reconciled']
    require([event.get('observation') for event in events]==records,
        'every immutable terminal parent-exit event must match its ordered manifest reference')
    for index,reference in enumerate(records,1):
        require(reference['path']==str(directory/('tool_parent_exit_observation_'+str(index)+'.json'))
            and reference==observer.receipt(reference['path']),'parent-exit record path or bytes changed')
        record=observer.read_json(reference['path'])
        event=events[index-1]
        require(type(event.get('pid')) is int and event['pid']==record['parent_pid']
            and event.get('birth')==record['parent_birth'] and event.get('discarded_child_list_valid') is False
            and type(event.get('at_seconds')) in (int,float) and math.isfinite(event['at_seconds'])
            and record['fresh_producer_children']['finished_at_seconds']<=event['at_seconds']<=terminal['terminal']['observed_at_seconds'],
            'the terminal event changed the observation generation, validity or time')
        validate_parent_exit_record(record,terminal,producer,base,observer,role['provider']['descriptor_events'],parser,
            producer_terminal_time=role['completed_at_seconds'])
    task_manifest_ref=observer.receipt(directory/'tool_task_roster_observations.json')
    task_manifest=observer.read_json(task_manifest_ref['path'])
    require(type(task_manifest.get('schema_version')) is int and task_manifest['schema_version']==1
        and task_manifest.get('protocol')==TASK_PROTOCOL
        and task_manifest['adapter_sha256']==plan['helpers']['files'][PREFIX+NAME]
        and task_manifest['base_sha256']==plan['helpers']['files'][PREFIX+'direct_tool_descendants.py']
        and task_manifest['producer_pid']==producer['pid'] and task_manifest['producer_birth']==producer['process_start_time']
        and task_manifest['scope_terminal']==result['tool_descendant_scope_terminal']
        and task_manifest['scope_restored']==result['tool_descendant_scope_restored']
        and type(task_manifest['failure_count']) is int and task_manifest['failure_count']==0
        and task_manifest['discarded_child_list_samples_accepted_as_valid'] is False,
        'the actual task-roster source/outcome manifest is incomplete')
    task_records=task_manifest['records']
    require(type(task_records) is list and len(task_records)<=base.MAX_EVENTS
        and not list(directory.glob('tool_task_roster_failure_*.json')),'a failed or excessive task collection invalidates the scope')
    expected_paths={str(directory/('tool_task_roster_observation_'+str(index)+'.json')) for index in range(1,len(task_records)+1)}
    require({str(path) for path in directory.glob('tool_task_roster_observation_*.json')}==expected_paths,
        'an actual discarded task collection was omitted or added outside its manifest')
    task_events=[event for event in terminal['events'] if event.get('kind')=='known_parent_task_roster_reconciled']
    require([event.get('observation') for event in task_events]==task_records,
        'every immutable terminal task-roster event must match its ordered manifest reference')
    require(observer.stable(role['owner'])==observer.stable(producer),
        'the task-roster native-open/close evidence belongs to another producer')
    scope_begin=observer.read_json(result['tool_descendant_scope_begin']['path'])
    for index,reference in enumerate(task_records,1):
        require(reference['path']==str(directory/('tool_task_roster_observation_'+str(index)+'.json'))
            and reference==observer.receipt(reference['path']),'a discarded task collection changed its reserved path or bytes')
        record=observer.read_json(reference['path']);event=task_events[index-1]
        require(type(event.get('pid')) is int and event['pid']==record['parent_pid'] and event.get('birth')==record['parent_birth']
            and same(event.get('task'),record['failed_task']) and type(event.get('collection_sequence')) is int
            and event['collection_sequence']==record['collection_sequence'] and event.get('discarded_child_list_valid') is False
            and type(event.get('at_seconds')) in (int,float) and math.isfinite(event['at_seconds'])
            and record['fresh_complete_collection']['finished_at_seconds']<=event['at_seconds']<=terminal['terminal']['observed_at_seconds'],
            'the terminal task-roster event changed its parent, task, validity or chronology')
        validate_task_roster_record(record,terminal,scope_begin,producer,base,observer,role['provider']['descriptor_events'],parser,
            producer_terminal_time=role['completed_at_seconds'])
    return {**result,'profile_parent_exit_observations':manifest_ref,'discarded_parent_samples_used_as_child_lists':False,
        'profile_task_roster_observations':task_manifest_ref,'discarded_task_samples_used_as_child_lists':False}
