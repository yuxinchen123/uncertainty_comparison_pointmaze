"""Discard a failed child-list sample only after a pinned parent exit and fresh adoption proof.

The accepted component scope stays unchanged. New profile-source qualifications
and admitted profiles use this explicit adapter and its additional saved reader.
"""
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

NAME='profile_target_tool_scope.py'
PREFIX='10_jax_exploration_platform/scripts/atari_pixel_campaign/'
PROTOCOL='profile-target-pinned-parent-exit-observation-version-1'
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


def scope_class(base):
    if base in _CLASSES:return _CLASSES[base]
    class ProfileToolScope(base.OwnedToolScope):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self._base_module=base
            self.parent_exit_records=[]
            self.parent_exit_failures=[]

        def child_lists(self,pid):
            """Retain every original refusal except a separately typed known-handle exit."""
            try:return super().child_lists(pid)
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
    return {**result,'profile_parent_exit_observations':manifest_ref,'discarded_parent_samples_used_as_child_lists':False}
