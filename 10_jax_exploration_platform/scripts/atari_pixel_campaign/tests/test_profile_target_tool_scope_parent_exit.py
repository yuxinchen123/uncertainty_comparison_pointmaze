"""Exercise only real positively owned CPU children; no Nsight, numerical code or GPU is used."""
import hashlib
import importlib.util
import inspect
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
PARENT_CODE="""import json,os,subprocess,sys
grand=subprocess.Popen([sys.executable,'-I','-S','-B','-c',
    'import os,sys;from pathlib import Path;fd=os.open(sys.argv[1],os.O_RDONLY);Path(sys.argv[2]).write_text(str(os.getpid()));os.read(fd,1);os.close(fd)',sys.argv[1],sys.argv[2]],close_fds=True)
print(json.dumps({'parent_pid':os.getpid(),'grandchild_pid':grand.pid}),flush=True)
sys.stdin.buffer.read(1)
os._exit(0)
"""


def load(path,name):
    """Load only a declared local source file for this zero-credit CPU fixture."""
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class ParentExitTests(unittest.TestCase):
    """Use synchronized normal exits rather than sleeps to create the child-list race."""
    def setUp(self):
        """Construct a private source-pinned observer using actual standard-library pidfds."""
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.directory=Path(self.temporary.name)
        self.adapter=load(S/'profile_target_tool_scope.py','actual_new_profile_scope_fixture')
        self.reader=load(S/'nsight_process_identity.py','actual_stdlib_scope_observer_fixture')
        self.parser=load(S/'linux_pidfd_process_control.py','actual_pidfd_identity_reader_fixture')
        names=('receipt','read_json','publish','parsed_uids','stable','validate_snapshot',
            'open_process_handle','close_process_handle','handle_exited','signal_handle')
        self.observer=SimpleNamespace(**{name:getattr(self.reader,name) for name in names})
        self.open_events=[];self.retained_descriptors={};self.next_open_generation=1
        def opened(pid):
            descriptor=self.reader.open_process_handle(pid);actual=os.fstat(descriptor)
            generation=self.next_open_generation;self.next_open_generation+=1
            self.retained_descriptors[descriptor]={'pid':pid,'generation':generation,
                'identity':{'device':actual.st_dev,'inode':actual.st_ino}}
            self.open_events.append({'operation':'pidfd_open','descriptor':descriptor,'opened_pid':pid,
                'open_generation':generation,'identity':{'device':actual.st_dev,'inode':actual.st_ino},
                'observed_at_seconds':time.time()})
            return descriptor
        def closed(descriptor):
            retained=self.retained_descriptors[descriptor];actual=os.fstat(descriptor)
            identity={'device':actual.st_dev,'inode':actual.st_ino}
            self.assertEqual(identity,retained['identity'])
            self.reader.close_process_handle(descriptor)
            self.open_events.append({'operation':'close','descriptor':descriptor,'opened_pid':retained['pid'],
                'open_generation':retained['generation'],'identity':identity,'observed_at_seconds':time.time()})
            del self.retained_descriptors[descriptor]
        self.observer.open_process_handle=opened
        self.observer.close_process_handle=closed
        self.plan={'kind':'full_original_profile_source_target_qualification',
            'profile_tool':{'path':sys.executable,'scope':'CPU fixture only; never a real profiler'},
            'helpers':{'files':{PREFIX+name:hashlib.sha256((S/name).read_bytes()).hexdigest()
                for name in ('direct_tool_descendants.py','profile_target_tool_scope.py','linux_pidfd_process_control.py')}},
            'configured_native_process_control':{'synthetic_upstream_custody_premise':True}}
        self.base=self.adapter.base_module(self.plan)
        self.owner=self.reader.ProcReader().snapshot(os.getpid(),os.getuid())
        self.parent=None;self.fifo_fd=None

    def scope(self,new):
        """Create one actual closed scope with an explicitly synthetic non-GPU custody premise."""
        arguments=(self.directory,self.plan,self.observer,self.owner,
            lambda:{'configured_allocation_process_custody':True,'synthetic_upstream_premise':True})
        keywords={'grace_seconds':2,'custody_kind':'configured_allocation_process'}
        return self.adapter.create_scope(*arguments,**keywords) if new else self.base.OwnedToolScope(*arguments,**keywords)

    def start_family(self,scope):
        """Start one declared own root whose positively listed child waits on a private FIFO."""
        fifo=self.directory/'grandchild_finish.fifo';ready=self.directory/'grandchild_ready.txt'
        os.mkfifo(fifo,0o600);self.fifo_fd=os.open(fifo,os.O_RDWR|os.O_NONBLOCK)
        command=[sys.executable,'-I','-S','-B','-c',PARENT_CODE,str(fifo),str(ready)]
        self.parent=scope.spawn(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
        self.assertTrue(select.select([self.parent.stdout],[],[],5)[0],'own root did not report its actual child')
        row=json.loads(self.parent.stdout.readline());self.assertEqual(row['parent_pid'],self.parent.pid)
        deadline=time.monotonic()+5
        while not ready.exists():
            self.assertLess(time.monotonic(),deadline,'own grandchild acknowledgment timed out');time.sleep(.01)
        grandchild=int(ready.read_text());self.assertEqual(grandchild,row['grandchild_pid'])
        scope.scan()
        self.assertIn(self.parent.pid,scope.handles);self.assertIn(grandchild,scope.handles)
        self.assertEqual(scope.nodes[grandchild]['initial']['parent_pid'],self.parent.pid)
        return grandchild

    def finish_family(self,scope):
        """Request only these known CPU children to exit normally, then retain actual waits/ECHILD."""
        if self.parent is not None and self.parent.poll() is None:
            self.parent.stdin.write(b'x');self.parent.stdin.flush()
        if self.fifo_fd is not None:
            os.write(self.fifo_fd,b'x');os.close(self.fifo_fd);self.fifo_fd=None
        if self.parent is not None:
            self.assertEqual(scope.wait(self.parent,5),0)
            for stream in (self.parent.stdin,self.parent.stdout,self.parent.stderr):stream.close()
        deadline=time.monotonic()+5
        while any(not self.observer.handle_exited(handle) for handle in scope.handles.values()):
            self.assertLess(time.monotonic(),deadline,'positively owned CPU child did not exit normally')
            scope.reap();time.sleep(.01)
        scope.reap()

    def exit_between_identity_reads(self,scope,new):
        """Exit the exact pinned parent only after its actual cgroup read, before its final pidfd check."""
        grandchild=self.start_family(scope);parent=self.parent
        birth=scope.nodes[parent.pid]['initial']['process_start_time']
        original=self.base.bounded;trigger=[]
        def bounded(directory,name):
            value=original(directory,name)
            if not trigger and name=='cgroup' and inspect.currentframe().f_back.f_code.co_name=='parent_identity':
                actual=self.base.parsed_identity(original(directory,'stat').decode())
                if actual['pid']==parent.pid:
                    self.assertEqual(actual['process_start_time'],birth)
                    self.assertEqual(value.decode(),self.owner['cgroup'])
                    parent.stdin.write(b'x');parent.stdin.flush()
                    self.assertEqual(parent.wait(timeout=5),0)
                    self.assertTrue(self.observer.handle_exited(scope.handles[parent.pid]))
                    trigger.append({'pid':parent.pid,'birth':birth,'wait_returncode':0})
            return value
        with patch.object(self.base,'bounded',new=bounded):
            if new:scope.scan()
            else:
                with self.assertRaisesRegex(ValueError,'known parent exited before its child lists could be accepted'):scope.scan()
        self.assertEqual(len(trigger),1)
        adopted=scope.snapshot(grandchild)
        self.assertEqual(adopted['parent_pid'],os.getpid())
        self.assertEqual(adopted['process_start_time'],scope.nodes[grandchild]['initial']['process_start_time'])
        self.assertEqual(adopted['uids'],[os.getuid()]*4);self.assertEqual(adopted['cgroup'],self.owner['cgroup'])
        return grandchild

    def test_old_scope_reproduces_exact_final_parent_exit_refusal(self):
        """The unchanged old helper refuses the deterministic real own-parent exit."""
        with self.scope(False) as scope:
            try:self.exit_between_identity_reads(scope,False)
            finally:self.finish_family(scope)
        self.assertTrue(scope.terminal['terminal']['actual_echild'])
        self.assertEqual(scope.interventions,[])

    def test_new_scope_reconciles_actual_adoption_and_normal_waits(self):
        """Only positive retained generation exit is reconciled with a fresh producer child list."""
        with self.scope(True) as scope:
            try:grandchild=self.exit_between_identity_reads(scope,True)
            finally:self.finish_family(scope)
        self.assertTrue(scope.terminal['terminal']['actual_echild'])
        self.assertTrue(scope.terminal['terminal']['all_pinned_handles_exited'])
        self.assertTrue(scope.terminal['terminal']['all_actual_tool_roots_waited'])
        self.assertEqual(scope.interventions,[]);self.assertEqual(scope.read_errors,[])
        self.assertTrue(any(row['pid']==grandchild and row['kind']=='reaped_actual_adopted_child' for row in scope.events))
        record=json.loads((self.directory/'tool_parent_exit_observations.json').read_bytes())
        rows=[json.loads(Path(reference['path']).read_bytes()) for reference in record['records']]
        self.assertTrue(rows)
        self.assertTrue(all(row['outcome']=='pinned_parent_exited_fresh_root_reconciled'
            and row['child_list_sample_valid'] is False for row in rows))
        self.assertTrue(any(grandchild in row['fresh_producer_children']['pids'] for row in rows))
        saved_terminal=json.loads((self.directory/'tool_descendant_scope_terminal.json').read_bytes())
        for row in rows:
            self.adapter.validate_parent_exit_record(row,saved_terminal,self.owner,
                self.base,self.observer,self.open_events,self.parser,producer_terminal_time=time.time())
        self.assertEqual(self.retained_descriptors,{})

    def test_live_unknown_birth_and_cgroup_refusals_remain(self):
        """Known live, unpinned and changed-generation/owner-context inputs cannot become exit proof."""
        with self.assertRaisesRegex(RuntimeError,'cleanup intervention or lost evidence'), self.scope(True) as scope:
            try:
                self.start_family(scope);pid=self.parent.pid
                with patch.object(self.base.OwnedToolScope,'parent_identity',side_effect=ValueError('the known parent exited before its child lists could be accepted')):
                    with self.assertRaises(ValueError):scope.child_lists(pid)
                node=scope.nodes.pop(pid)
                try:
                    with self.assertRaisesRegex(ValueError,'already pinned owned parent'):scope.child_lists(pid)
                finally:scope.nodes[pid]=node
                birth=node['initial']['process_start_time'];node['initial']['process_start_time']=str(int(birth)+1)
                try:
                    with self.assertRaisesRegex(ValueError,'pinned birth changed'):scope.child_lists(pid)
                finally:node['initial']['process_start_time']=birth
                group=scope.cgroup;scope.cgroup=group+'synthetic_wrong_context'
                try:
                    with self.assertRaisesRegex(ValueError,'exact owned cgroup'):scope.child_lists(pid)
                finally:scope.cgroup=group
            finally:self.finish_family(scope)
        self.assertTrue(scope.terminal['terminal']['actual_echild'])


if __name__=='__main__':unittest.main(verbosity=2)
