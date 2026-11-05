"""Observe only own-process libc symbols and local public headers on owned CPU compute."""
import ctypes,hashlib,json,os,platform,re,socket,sys,time
from pathlib import Path
assert os.environ['SLURM_JOB_ID']=='6783003' and os.environ['SLURM_STEP_ID'].isdecimal()
headers={}
for name in ('/usr/include/unistd.h','/usr/include/x86_64-linux-gnu/asm/unistd_64.h','/usr/include/linux/pidfd.h','/usr/include/features.h'):
 p=Path(name);raw=p.read_bytes();assert len(raw)<100000
 headers[name]={'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw),'uid':p.stat().st_uid,'mode':p.stat().st_mode&0o777,'raw':raw.decode()}
libc=ctypes.CDLL(None,use_errno=True)
symbols={}
for name in ('syscall','pidfd_open','pidfd_send_signal','close','prctl','gnu_get_libc_version'):
 try:function=getattr(libc,name)
 except AttributeError:symbols[name]={'present':False}
 else:symbols[name]={'present':True,'pointer':ctypes.cast(function,ctypes.c_void_p).value}
maps=[]
for line in Path('/proc/self/maps').read_text().splitlines():
 parts=line.split(maxsplit=5)
 if len(parts)==6 and Path(parts[5]).name=='libc.so.6':maps.append(line)
paths={str(Path(line.split(maxsplit=5)[5]).resolve(strict=True)) for line in maps};assert len(paths)==1
p=Path(next(iter(paths)));raw=p.read_bytes();assert len(raw)<10000000
v={'kind':'actual_selected_python_native_process_control_capability_observation','job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],'hostname':socket.gethostname(),'pid':os.getpid(),'uids':list(os.getresuid()),'cgroup':Path('/proc/self/cgroup').read_text(),'observed_at_seconds':time.time(),'sys_executable':sys.executable,'resolved_sys_executable':str(Path(sys.executable).resolve(strict=True)),'sys_version':sys.version,'machine':platform.machine(),'kernel':platform.release(),'byteorder':sys.byteorder,'ctypes_sizes':{'long':ctypes.sizeof(ctypes.c_long),'int':ctypes.sizeof(ctypes.c_int),'pointer':ctypes.sizeof(ctypes.c_void_p)},'symbols':symbols,'public_headers':headers,'actual_mapped_libc':{'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw),'uid':p.stat().st_uid,'mode':p.stat().st_mode&0o777,'maps':maps},'kernel_pidfd_syscalls_executed':False,'gpu_or_cuda_qualification':False,'tool_or_child_launched':False,'scientific_credit':0}
out=Path(__file__).parent/('selected_python_native_pidfd_capabilities_'+v['job_id']+'_'+v['step_id']+'.json');out.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps({'record':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'symbols':symbols,'ctypes_sizes':v['ctypes_sizes'],'libc_sha256':v['actual_mapped_libc']['sha256']}))
