
"""Start the exact published controller once on the approved host; never open a GPU."""
import datetime,hashlib,json,os,pathlib,pwd,socket,subprocess
P=pathlib.Path(__file__).resolve(strict=True).parent
R=P.parent.parent
def write(path,x):
    """Publish a new launch record without replacing an earlier attempt."""
    raw=(json.dumps(x,indent=2,sort_keys=True)+'\n').encode()
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
def main():
    """Use the reviewed exact argv and preserve the real returned process identity."""
    assert socket.gethostname().split('.')[0]=='serval02' and os.getuid()==os.geteuid()==2618919
    assert pwd.getpwuid(os.getuid()).pw_name=='sl5nw'
    record=json.loads((P/'actual_launch_command_preparation.json').read_text())
    assert record['host']=='serval02' and record['automatic_retry_permitted'] is False
    assert record['publication']=='677309d261c223e8b3ed8c9b3eb61edc737a8d82'
    command=record['command']
    assert command==['/usr/bin/env','-i','PATH=/usr/local/bin:/usr/bin:/bin','LANG=C.UTF-8','TZ=UTC',
        'CUDA_VISIBLE_DEVICES=GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3','PYTHONDONTWRITEBYTECODE=1',
        'PYTHONNOUSERSITE=1','/p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python','-I','-B',
        str(P/'direct_qualification_controller.py'),'--launch','--publication-commit',record['publication']]
    assert not (P/'root_actual_controller_spawn.json').exists() and not (P/'root_actual_controller_spawn.json').is_symlink()
    assert (P/'root_remote_launch_intent.json').is_file()
    out=pathlib.Path(record['stdout']);err=pathlib.Path(record['stderr'])
    assert out.parent==err.parent==R/'slurm/direct_jobs'/P.name
    with out.open('xb',buffering=0) as stdout,err.open('xb',buffering=0) as stderr:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
            start_new_session=True,close_fds=True,cwd=P)
        proc=pathlib.Path('/proc')/str(child.pid);assert proc.stat().st_uid==2618919
        birth=proc.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]
        value={'kind':'actual_controller_transport_popen_return','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'actual_pid':child.pid,'actual_birth_counter':birth,'host':'serval02','uid':2618919,
            'intended_command':command,'actual_numerical_child_pid':None,'automatic_retry_permitted':False,
            'actual_qualification_verified':False,'scientific_credit':0}
        write(P/'root_actual_controller_spawn.json',value)
    print(json.dumps(value),flush=True)
if __name__=='__main__':main()
