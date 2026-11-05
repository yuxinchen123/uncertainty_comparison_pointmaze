"""Causally test the actual starter on one fresh CPU-only surrogate controller."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

P=Path(__file__).resolve().parent

def load(name,path):
    """Load exact source as metadata without invoking its main entry."""
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def main():
    """Use real Popen, all-UID/birth reads, metadata flock and fsync; no tool or numerical operation."""
    starter=load('actual_root_transport_under_test',P/'root_start_controller_once.py')
    control=load('actual_controller_metadata_for_cpu_fixture',P/'direct_qualification_controller.py')
    with tempfile.TemporaryDirectory() as temporary:
        run=Path(temporary)/'run';packet=run/'remote_setup/cpu-only-controller';packet.mkdir(parents=True)
        logs=run/'slurm/direct_jobs'/packet.name;logs.mkdir(parents=True);(run/'queue').mkdir()
        lock=run/'queue/metadata.lock';lock.write_text('');lock.chmod(0o600)
        stub=packet/'direct_qualification_controller.py'
        stub.write_text("import fcntl,json,os,pathlib\nP=pathlib.Path(__file__).resolve().parent;R=P.parent.parent\n(P/'before_lock').write_text(str(os.getpid()))\nwith (R/'queue/metadata.lock').open('r+') as held:\n fcntl.flock(held,fcntl.LOCK_EX)\n assert (P/'record_writer_completed').exists()\n value=json.loads((P/'root_actual_controller_spawn.json').read_text())\n assert value['actual_pid']==os.getpid()\n (P/'after_lock').write_text(str(os.getpid()))\n")
        publication='d3208df3e8588b29544835fc6e6d242c4be44153'
        command=['/usr/bin/env','-i','PATH=/usr/local/bin:/usr/bin:/bin','LANG=C.UTF-8','TZ=UTC',
            'CUDA_VISIBLE_DEVICES=GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3','PYTHONDONTWRITEBYTECODE=1',
            'PYTHONNOUSERSITE=1','/usr/bin/python3','-I','-S','-B',str(stub),'--launch','--publication-commit',publication]
        (packet/'actual_launch_command_preparation.json').write_text(json.dumps({'host':'serval02','automatic_retry_permitted':False,
            'publication':publication,'command':command,'stdout':str(logs/'controller.stdout'),'stderr':str(logs/'controller.stderr')}))
        (packet/'root_remote_launch_intent.json').write_text('{}')
        real_write=starter.write;events=[]
        def delayed_write(path,value):
            """Wait for the actual child at the held barrier, then complete genuine exclusive file/directory fsync."""
            if path.name=='root_actual_controller_spawn.json':
                deadline=time.monotonic()+10
                while not (packet/'before_lock').exists() and time.monotonic()<deadline:time.sleep(0.01)
                assert (packet/'before_lock').exists() and not (packet/'after_lock').exists()
                events.append('actual_child_waiting_before_spawn_record_fsync')
                real_write(path,value)
                assert not (packet/'after_lock').exists()
                (packet/'record_writer_completed').write_text('actual write returned after file and directory fsync')
                events.append('spawn_record_file_and_directory_fsync_complete_while_barrier_held')
                return
            return real_write(path,value)
        with patch.object(starter,'P',packet),patch.object(starter,'R',run),patch.object(starter,'controller_metadata',return_value=control),\
                patch.object(starter.socket,'gethostname',return_value='serval02'),patch.object(starter,'write',side_effect=delayed_write),contextlib.redirect_stdout(io.StringIO()):
            starter.main()
        saved=json.loads((packet/'root_actual_controller_spawn.json').read_text());pid=saved['actual_pid']
        actual,wait=os.waitpid(pid,0);assert actual==pid and os.waitstatus_to_exitcode(wait)==0
        assert (packet/'after_lock').exists() and int((packet/'after_lock').read_text())==pid
        events.append('surrogate_after_lock_only_after_durable_record_and_barrier_release')
        assert events==['actual_child_waiting_before_spawn_record_fsync','spawn_record_file_and_directory_fsync_complete_while_barrier_held','surrogate_after_lock_only_after_durable_record_and_barrier_release']
        print(json.dumps({'kind':'causal_cpu_only_root_transport_metadata_barrier_fixture','passed':True,'events':events,
            'actual_owned_cpu_job':os.environ['SLURM_JOB_ID'],'actual_numeric_step':os.environ['SLURM_STEP_ID'],
            'real_uid':os.getuid(),'real_child_pid':pid,'real_birth':saved['actual_birth_counter'],'actual_wait_returncode':0,
            'hostname_override':'labelled temporary CPU fixture only','actual_target_operation':False,'numerical_gpu_tool_remote_operations':0}))

if __name__=='__main__':main()
