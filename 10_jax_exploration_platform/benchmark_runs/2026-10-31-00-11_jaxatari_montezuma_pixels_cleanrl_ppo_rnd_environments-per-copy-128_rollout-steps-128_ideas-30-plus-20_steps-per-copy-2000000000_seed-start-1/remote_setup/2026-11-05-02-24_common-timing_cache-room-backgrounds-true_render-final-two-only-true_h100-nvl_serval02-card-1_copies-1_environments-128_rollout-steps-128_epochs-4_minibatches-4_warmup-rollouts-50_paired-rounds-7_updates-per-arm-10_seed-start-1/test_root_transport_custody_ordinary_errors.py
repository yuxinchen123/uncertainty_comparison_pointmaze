"""Keep an actual metadata lock held across ordinary observer failures until a real owned child is waited."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from unittest.mock import patch
P=Path(__file__).resolve().parent

def load(name,path):
    """Load the exact frozen metadata source without running its main entry."""
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def main():
    """Inject ordinary poll, wait, report, print and sleep errors around one real positively owned CPU child."""
    starter=load('corrected_custody_for_cpu_fixture',P/'root_start_controller_once.py')
    control=load('unchanged_metadata_control_for_cpu_fixture',P/'direct_qualification_controller.py')
    with tempfile.TemporaryDirectory() as temporary:
        lock=Path(temporary)/'metadata.lock';lock.write_text('');lock.chmod(0o600)
        child=subprocess.Popen(['/usr/bin/python3','-I','-S','-B','-c','import time;time.sleep(0.8)'],close_fds=True)
        identity=control.read_process(child.pid,os.getuid());assert identity['uid']==os.getuid() and identity['pid']==child.pid
        class OwnedChild:
            """Preserve the actual Popen child while injecting only declared observer errors."""
            pid=child.pid
            polls=0
            waits=0
            positive=False
            def poll(self):
                """Reproduce the outer helper's ordinary poll error before custody starts."""
                self.polls+=1;raise OSError('labelled ordinary poll failure')
            def wait(self,timeout=None):
                """Prove the actual barrier remains held on every wait, then wait only our real child."""
                fd=os.open(lock,os.O_RDWR)
                try:
                    try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:pass
                    else:raise AssertionError('metadata barrier released before exact positive child wait')
                finally:os.close(fd)
                self.waits+=1
                if self.waits==1:raise OSError('labelled ordinary wait failure')
                returned=child.wait(timeout=timeout)
                assert type(returned) is int;self.positive=True;return returned
        owned=OwnedChild();real_sleep=time.sleep;sleeps=[]
        def failing_sleep(seconds):
            """Reproduce an ordinary sleep error once, then use the real bounded sleep."""
            sleeps.append(seconds)
            if len(sleeps)==1:raise OSError('labelled ordinary sleep failure')
            return real_sleep(seconds)
        try:
            with control.metadata_lock(lock,os.getuid()):
                with patch.object(starter,'write',side_effect=OSError('labelled report failure')),\
                        patch.object(starter,'print',side_effect=BrokenPipeError('labelled print failure'),create=True),\
                        patch.object(starter.time,'sleep',side_effect=failing_sleep):
                    proof=starter.cleanup_under_barrier(control,owned,None,ValueError('unknown fixture identity'))
                assert owned.positive and proof['kernel_child_exit_observed'] is True and proof['returncode']==0
            fd=os.open(lock,os.O_RDWR)
            try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            finally:os.close(fd)
            assert owned.polls==1 and owned.waits>=2 and sleeps
            phases={row['phase'] for row in proof['ordinary_observer_errors']}
            assert {'initial_custody_report','exact_child_wait','custody_wait_sleep'}<=phases
            assert proof['signal_sent'] is False
            print(json.dumps({'kind':'actual_owned_cpu_child_ordinary_error_custody_regression','passed':True,
                'owned_job':os.environ['SLURM_JOB_ID'],'numeric_step':os.environ['SLURM_STEP_ID'],
                'child_pid':child.pid,'child_birth':identity['process_start_time'],'actual_wait':proof['returncode'],
                'poll_errors':owned.polls,'wait_calls':owned.waits,'retained_error_phases':sorted(phases),
                'barrier_verified_held_on_each_wait':True,'lock_reacquired_only_after_exact_wait':True,
                'unknown_child_signals_sent':False,'actual_target_or_gpu_tool_remote_operations':0}))
        finally:
            # Fixture failure may clean only this positively observed fresh owned CPU child.
            if child.poll() is None:
                control.wait_exact(child,identity,os.getuid(),0)

if __name__=='__main__':main()
