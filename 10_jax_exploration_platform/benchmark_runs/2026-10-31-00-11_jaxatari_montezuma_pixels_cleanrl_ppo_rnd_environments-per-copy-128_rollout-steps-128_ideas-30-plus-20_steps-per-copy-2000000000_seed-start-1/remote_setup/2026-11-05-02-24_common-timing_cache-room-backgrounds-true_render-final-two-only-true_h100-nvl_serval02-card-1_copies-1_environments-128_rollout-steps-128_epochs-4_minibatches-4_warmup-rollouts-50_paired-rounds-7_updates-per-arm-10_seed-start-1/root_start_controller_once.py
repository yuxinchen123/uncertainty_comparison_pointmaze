"""Start one exact published metadata controller behind its existing owner metadata lock."""
import datetime,hashlib,json,os,pathlib,pwd,socket,subprocess
import importlib.util
import signal
import time
P=pathlib.Path(__file__).resolve(strict=True).parent
R=P.parent.parent

def write(path,x):
    """Publish a new launch record without replacing an earlier attempt."""
    raw=(json.dumps(x,indent=2,sort_keys=True)+'\n').encode()
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)

def controller_metadata():
    """Load only the unchanged staged controller's owned-process and metadata-lock functions."""
    spec=importlib.util.spec_from_file_location('root_transport_controller_metadata',P/'direct_qualification_controller.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.no_numerical_modules()
    return module

def retain_custody(child,error):
    """Retain the barrier through ordinary observer errors until this exact Popen child is waited."""
    # Ignore transport timeout/disconnect signals before any fallible report can run.
    previous={number:signal.getsignal(number) for number in (signal.SIGHUP,signal.SIGINT,signal.SIGTERM)}
    for number in previous:signal.signal(number,signal.SIG_IGN)
    ordinary_errors=[]
    def note(phase,problem):
        """Keep bounded error metadata; a broken report never releases descriptor custody."""
        try:
            if len(ordinary_errors)<32:
                item={'phase':phase,'error_type':type(problem).__name__,'completed':False,
                    'metadata_barrier_retained':True,'signal_sent':False}
                ordinary_errors.append(item)
                try:print(json.dumps(item),flush=True)
                except Exception:pass
        except Exception:pass
    try:
        write(P/'root_transport_custody_inconclusive.json',{
            'kind':'root_transport_cleanup_inconclusive_barrier_retained','completed':False,
            'actual_popen_pid':child.pid,'identity_or_cleanup_error_type':type(error).__name__,
            'metadata_barrier_retained':True,'signal_sent_by_custody_loop':False,'scientific_credit':0})
    except Exception as record_error:note('initial_custody_report',record_error)
    # A direct bounded wait avoids poll as a separate fallible exit claim; only a typed actual wait permits return.
    while True:
        try:
            returned=child.wait(timeout=0.1)
            if type(returned) is int:break
            note('wait_return_type',ValueError('the exact Popen wait did not return an integer status'))
        except subprocess.TimeoutExpired:
            pass
        except Exception as wait_error:
            note('exact_child_wait',wait_error)
        try:time.sleep(0.1)
        except Exception as sleep_error:note('custody_wait_sleep',sleep_error)
    # The exact original child is now positively waited; restoring handlers cannot expose a live unknown child.
    for number,handler in previous.items():signal.signal(number,handler)
    return {'kind':'exact_popen_wait_after_inconclusive_identity','actual_popen_pid':child.pid,
        'returncode':returned,'signal_sent':False,'kernel_child_exit_observed':True,
        'ordinary_observer_errors':ordinary_errors}

def cleanup_under_barrier(control,child,identity,error):
    """Reap only the exact positively verified child, or retain the barrier without signaling it."""
    try:
        polled=child.poll()
    except Exception as poll_error:
        return retain_custody(child,poll_error)
    if polled is not None:
        try:
            returned=child.wait(timeout=0)
            if type(returned) is not int:raise ValueError('the exact Popen wait returned no integer status')
        except Exception as wait_error:
            return retain_custody(child,wait_error)
        return {'kind':'exact_popen_wait_before_barrier_release','actual_popen_pid':child.pid,
            'returncode':returned,'signal_sent':False,'kernel_child_exit_observed':True}
    if identity is not None:
        try:
            returned,actions=control.wait_exact(child,identity,2618919,0)
            if type(returned) is not int:raise ValueError('verified child cleanup returned no integer wait status')
            return {'kind':'verified_owned_child_cleanup_under_metadata_barrier','actual_popen_pid':child.pid,
                'returncode':returned,'interventions':actions,'kernel_child_exit_observed':True}
        except Exception as cleanup_error:
            return retain_custody(child,cleanup_error)
    return retain_custody(child,error)

def main():
    """Fsync the genuine Popen identity before the unchanged controller can enter its launch scope."""
    assert socket.gethostname().split('.')[0]=='serval02' and os.getuid()==os.geteuid()==2618919
    assert os.getresuid()==(2618919,2618919,2618919) and pwd.getpwuid(os.getuid()).pw_name=='sl5nw'
    record=json.loads((P/'actual_launch_command_preparation.json').read_text())
    assert record['host']=='serval02' and record['automatic_retry_permitted'] is False
    assert record['publication']=='d3208df3e8588b29544835fc6e6d242c4be44153'
    command=record['command']
    assert command==['/usr/bin/env','-i','PATH=/usr/local/bin:/usr/bin:/bin','LANG=C.UTF-8','TZ=UTC',
        'CUDA_VISIBLE_DEVICES=GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3','PYTHONDONTWRITEBYTECODE=1',
        'PYTHONNOUSERSITE=1','/usr/bin/python3','-I','-S','-B',
        str(P/'direct_qualification_controller.py'),'--launch','--publication-commit',record['publication']]
    attempts=('root_actual_controller_spawn.json','root_transport_popen_attempt.json','root_transport_failure.json',
        'root_transport_custody_inconclusive.json')
    assert not any(os.path.lexists(P/name) for name in attempts)
    assert (P/'root_remote_launch_intent.json').is_file()
    out=pathlib.Path(record['stdout']);err=pathlib.Path(record['stderr'])
    assert out.parent==err.parent==R/'slurm/direct_jobs'/P.name
    control=controller_metadata();parent=control.read_process(os.getpid(),2618919)
    assert parent['hostname']=='serval02'
    lock=R/'queue/metadata.lock';assert lock.is_file() and not lock.is_symlink()
    child=None;identity=None
    # The unchanged controller takes this same metadata lock before spawning or registering numerical work.
    with control.metadata_lock(lock,2618919):
        write(P/'root_transport_popen_attempt.json',{'kind':'one_root_controller_transport_popen_attempt',
            'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parent':parent,'command':command,
            'actual_child_pid':None,'metadata_barrier_path':str(lock),'automatic_retry_permitted':False})
        try:
            with out.open('xb',buffering=0) as stdout,err.open('xb',buffering=0) as stderr:
                child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                    start_new_session=True,close_fds=True,cwd=P)
                candidate=control.read_process(child.pid,2618919)
                assert candidate['parent_pid']==os.getpid() and candidate['hostname']=='serval02'
                assert candidate['command'] in (command,command[8:])
                # Birth and all UIDs are reconstructed by the proven reader, never guessed from a file name.
                identity=candidate
                after=control.read_process(child.pid,2618919)
                assert after['process_start_time']==identity['process_start_time'] and after['uid']==2618919
                assert child.poll() is None
                value={'kind':'actual_controller_transport_popen_return','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    'actual_pid':child.pid,'actual_birth_counter':identity['process_start_time'],'host':'serval02','uid':2618919,
                    'intended_command':command,'actual_controller_identity':identity,'actual_numerical_child_pid':None,
                    'automatic_retry_permitted':False,'actual_qualification_verified':False,'scientific_credit':0,
                    'metadata_barrier_path':str(lock),'metadata_barrier_held_through_record_fsync':True}
                write(P/'root_actual_controller_spawn.json',value)
        except Exception as error:
            cleanup=None if child is None else cleanup_under_barrier(control,child,identity,error)
            failure={'kind':'failed_root_controller_transport_under_metadata_barrier','error_type':type(error).__name__,
                'error':str(error),'actual_popen_pid':None if child is None else child.pid,'cleanup':cleanup,
                'ordinary_launch_success':False,'automatic_retry_permitted':False,'scientific_credit':0}
            write(P/'root_transport_failure.json',failure)
            raise
    print(json.dumps(value),flush=True)
if __name__=='__main__':main()
