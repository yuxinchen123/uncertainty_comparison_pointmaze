"""Synchronize one own CPU worker-thread exit between task enumeration and directory open."""
import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

S=Path(__file__).resolve().parents[1]
PREFIX='10_jax_exploration_platform/scripts/atari_pixel_campaign/'
PARENT_CODE="""import json,os,sys,threading,time
ready=threading.Event();finish=threading.Event();identity={}
def worker():
    identity['tid']=threading.get_native_id();ready.set();finish.wait()
thread=threading.Thread(target=worker);thread.start();ready.wait()
print(json.dumps({'pid':os.getpid(),'worker_tid':identity['tid']}),flush=True)
for line in sys.stdin.buffer:
    if line.strip()==b'release_thread':
        finish.set();thread.join()
        deadline=time.monotonic()+0.2
        while os.path.exists('/proc/self/task/'+str(identity['tid'])):
            if time.monotonic()>=deadline:raise RuntimeError('own joined task remained visible')
            time.sleep(0.001)
        print(json.dumps({'pid':os.getpid(),'worker_tid':identity['tid'],'worker_joined':not thread.is_alive()}),flush=True)
    elif line.strip()==b'finish':
        finish.set();thread.join();break
"""

def load(path,name):
    """Load declared local process metadata helpers without numerical or profiler code."""
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class LiveParentTaskExitTests(unittest.TestCase):
    """One controlled lifecycle distinguishes an invalid partial sample from a fresh complete read."""
    evidence_directory=None

    def run_case(self,new):
        """Keep the positively owned parent alive while its exact worker task exits normally."""
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            adapter=load(S/'profile_target_tool_scope.py','live_parent_task_exit_adapter_'+str(new))
            reader=load(S/'nsight_process_identity.py','live_parent_task_exit_stdlib_reader_'+str(new))
            owner=reader.ProcReader().snapshot(os.getpid(),os.getuid())
            observer=SimpleNamespace(**{name:getattr(reader,name) for name in
                ('receipt','read_json','publish','parsed_uids','stable','validate_snapshot','handle_exited','signal_handle')})
            events=[];handles={};generation=[0]
            def opened(pid):
                """Record the actual own pidfd generation and its real kernel inode."""
                descriptor=reader.open_process_handle(pid);identity=os.fstat(descriptor);generation[0]+=1
                handles[descriptor]={'pid':pid,'generation':generation[0],'identity':{'device':identity.st_dev,'inode':identity.st_ino}}
                events.append({'operation':'pidfd_open','descriptor':descriptor,'opened_pid':pid,'open_generation':generation[0],
                    'identity':handles[descriptor]['identity'],'observed_at_seconds':time.time()});return descriptor
            def closed(descriptor):
                """Record only a successful close of that same positively retained descriptor."""
                saved=handles[descriptor];identity=os.fstat(descriptor)
                actual={'device':identity.st_dev,'inode':identity.st_ino};self.assertEqual(actual,saved['identity'])
                reader.close_process_handle(descriptor)
                events.append({'operation':'close','descriptor':descriptor,'opened_pid':saved['pid'],'open_generation':saved['generation'],
                    'identity':actual,'observed_at_seconds':time.time()});del handles[descriptor]
            observer.open_process_handle=opened;observer.close_process_handle=closed
            plan={'kind':'full_original_profile_source_target_qualification','profile_tool':{'path':sys.executable},
                'helpers':{'files':{PREFIX+name:hashlib.sha256((S/name).read_bytes()).hexdigest()
                    for name in ('profile_target_tool_scope.py','direct_tool_descendants.py','linux_pidfd_process_control.py')}},
                'configured_native_process_control':{'synthetic_CPU_custody_premise':True}}
            base=adapter.base_module(plan)
            arguments=(directory,plan,observer,owner,lambda:{'CPU_only':True,'synthetic_custody_premise':True})
            options={'grace_seconds':2,'custody_kind':'configured_allocation_process'}
            scope=adapter.create_scope(*arguments,**options) if new else base.OwnedToolScope(*arguments,**options)
            parent=None;worker=None;triggered=[]
            def receive():
                """Read only the exact own child acknowledgment with a bounded wait."""
                self.assertTrue(select.select([parent.stdout],[],[],5)[0],'own parent acknowledgment timed out')
                return json.loads(parent.stdout.readline())
            def exit_worker():
                """Join this actual worker before its enumerated directory is opened; retain its live parent."""
                self.assertIn(parent.pid,scope.handles);self.assertIn(parent.pid,scope.nodes)
                self.assertFalse(observer.handle_exited(scope.handles[parent.pid]))
                before=scope.snapshot(parent.pid)
                self.assertEqual(before['process_start_time'],scope.nodes[parent.pid]['initial']['process_start_time'])
                self.assertEqual(before['uids'],[os.getuid()]*4);self.assertEqual(before['cgroup'],owner['cgroup'])
                parent.stdin.write(b'release_thread\n');parent.stdin.flush();ack=receive()
                self.assertEqual(ack,{'pid':parent.pid,'worker_tid':worker,'worker_joined':True})
                self.assertFalse(observer.handle_exited(scope.handles[parent.pid]));self.assertIsNone(parent.poll())
                self.assertFalse((Path('/proc')/str(parent.pid)/'task'/str(worker)).exists())
                triggered.append({'parent_pid':parent.pid,'parent_birth':before['process_start_time'],'worker_tid':worker,
                    'parent_handle_live':True,'worker_join_acknowledged':True,'at_seconds':time.time()})
            def finish_parent():
                """Close only the exact known CPU parent by its private pipe, then retain its actual wait."""
                if parent is not None:
                    if parent.poll() is None:parent.stdin.write(b'finish\n');parent.stdin.flush()
                    self.assertEqual(scope.wait(parent,5),0)
                    for stream in (parent.stdin,parent.stdout,parent.stderr):stream.close()
            with scope:
                try:
                    command=[sys.executable,'-I','-S','-B','-c',PARENT_CODE]
                    parent=scope.spawn(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
                    ready=receive();self.assertEqual(ready['pid'],parent.pid);worker=ready['worker_tid']
                    self.assertIs(type(worker),int);self.assertGreater(worker,0);self.assertNotEqual(worker,parent.pid)
                    self.assertTrue((Path('/proc')/str(parent.pid)/'task'/str(worker)).is_dir())
                    scope.scan()
                    if new:
                        actual_open=adapter.open_enumerated_task
                        def task_open(task_directory,name):
                            """Cause the actual ENOENT only at this known live parent's enumerated worker."""
                            if str(name)==str(worker) and not triggered:
                                self.assertEqual(os.readlink('/proc/self/fd/'+str(task_directory)),str(Path('/proc')/str(parent.pid)/'task'))
                                exit_worker()
                            return actual_open(task_directory,name)
                        with patch.object(adapter,'open_enumerated_task',new=task_open):scope.child_lists(parent.pid)
                    else:
                        actual_open=os.open
                        def task_open(name,flags,*args,**keywords):
                            """Reproduce the old exact task-directory failure, without fabricating an exception."""
                            descriptor=keywords.get('dir_fd')
                            if str(name)==str(worker) and descriptor is not None and not triggered \
                                    and os.readlink('/proc/self/fd/'+str(descriptor))==str(Path('/proc')/str(parent.pid)/'task'):
                                exit_worker()
                            return actual_open(name,flags,*args,**keywords)
                        with patch.object(base.os,'open',new=task_open):
                            with self.assertRaises(FileNotFoundError) as caught:scope.child_lists(parent.pid)
                        self.assertEqual(caught.exception.errno,errno.ENOENT);self.assertEqual(caught.exception.filename,str(worker))
                    self.assertEqual(len(triggered),1)
                    self.assertFalse(observer.handle_exited(scope.handles[parent.pid]))
                    children,rows=scope.child_lists(parent.pid)
                    self.assertEqual(children,set());self.assertEqual([row['tid'] for row in rows],[parent.pid])
                    proof=rows[0]['parent_read_identity']
                    self.assertEqual(proof['expected_pid'],parent.pid)
                    self.assertEqual(proof['expected_birth'],triggered[0]['parent_birth'])
                    self.assertTrue(proof['pidfd_live_before']);self.assertTrue(proof['pidfd_live_after'])
                finally:finish_parent()
            self.assertTrue(scope.terminal['terminal']['actual_echild'])
            self.assertTrue(scope.terminal['terminal']['all_pinned_handles_exited'])
            self.assertEqual(scope.interventions,[]);self.assertEqual(scope.read_errors,[]);self.assertEqual(handles,{})
            completed_at=time.time()
            if new:
                manifest=reader.read_json(directory/'tool_task_roster_observations.json')
                self.assertEqual(len(manifest['records']),1)
                reference=manifest['records'][0]
                self.assertEqual(reader.receipt(reference['path']),reference)
                record=reader.read_json(reference['path'])
                terminal=reader.read_json(directory/'tool_descendant_scope_terminal.json')
                begin=reader.read_json(directory/'tool_descendant_scope_begin.json')
                parser=adapter.adjacent(plan,'linux_pidfd_process_control').parse_pidfd_fdinfo
                adapter.validate_task_roster_record(record,terminal,begin,owner,base,observer,events,parser,
                    producer_terminal_time=completed_at)
            if self.evidence_directory is not None:
                target=Path(self.evidence_directory)/('fresh_complete_read' if new else 'old_refusal');target.mkdir(parents=True)
                for path in directory.iterdir():
                    if path.is_file() and path.suffix=='.json':(target/path.name).write_bytes(path.read_bytes())
                (target/'fixture_control.json').write_text(json.dumps({'actual_trigger':triggered,'descriptor_events':events,
                    'actual_fixture_completed_at_seconds':completed_at,
                    'scope':'actual own CPU parent and thread only; upstream configured custody is explicitly synthetic',
                    'native_provider_role_claimed':False,'scientific_credit':0,'admission_granted':False},sort_keys=True,indent=2)+'\n')

    def test_real_enumerated_worker_exit_old_refusal_and_fresh_complete_read(self):
        """The exact same controlled event refuses in the old base and requires a complete fresh read in the new scope."""
        self.run_case(False);self.run_case(True)


if __name__=='__main__':unittest.main(verbosity=2)
