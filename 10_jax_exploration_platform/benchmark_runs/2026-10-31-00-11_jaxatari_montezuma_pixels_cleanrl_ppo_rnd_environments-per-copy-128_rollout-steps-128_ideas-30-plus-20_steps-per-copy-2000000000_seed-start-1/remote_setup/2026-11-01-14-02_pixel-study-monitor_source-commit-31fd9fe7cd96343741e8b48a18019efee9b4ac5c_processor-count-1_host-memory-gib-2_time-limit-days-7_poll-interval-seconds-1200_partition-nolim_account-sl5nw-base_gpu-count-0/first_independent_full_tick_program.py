import pathlib,os,json,subprocess,hashlib,datetime,sys
r=pathlib.Path(sys.argv[1]);m=pathlib.Path(sys.argv[2])
assert os.environ['SLURM_JOB_ID']=='6782816' and os.environ['SLURM_STEP_ID'].isdecimal()
source='31fd9fe7cd96343741e8b48a18019efee9b4ac5c'
def save(suffix,raw):
 p=m/('first_independent_full_tick'+suffix)
 with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return {'path':str(p),'sha256':hashlib.sha256(raw).hexdigest()}
checked=subprocess.run(['/usr/bin/python3','-B',str(m/'verify_source.py')],capture_output=True)
save('_source_stdout.raw',checked.stdout);save('_source_stderr.raw',checked.stderr);assert checked.returncode==0
result=subprocess.run(['/usr/bin/python3','-B',str(m/'source_snapshot'/source/'10_jax_exploration_platform/scripts/atari_pixel_campaign/monitor.py'),'--run-dir',str(r)],capture_output=True)
save('_stdout.raw',result.stdout);save('_stderr.raw',result.stderr);assert result.returncode==0
status=(r/'20_mins_monitoring/latest_status.json').read_bytes();report=(r/'20_mins_monitoring/latest_report.md').read_bytes();s=json.loads(status)
fence=bytes([96])*3
assert result.stdout.endswith(b'\n\n')
assert report==b'# Pixel Montezuma study status\n\n'+s['time'].encode()+b'\n\n'+fence+b'text\n'+result.stdout[:-1]+fence+b'\n'
assert s['campaign']['queue_counts']=={'pending':0,'running':0,'completed':120,'failed':0}
assert all(result.stdout.splitlines().count(x.encode())==1 for x in ['TRAINING','WAITING','HELD','FINISHED','PLANNED'])
rows=[x for x in s['jobs'] if x[0]=='6782816'];assert len(rows)==1 and rows[0][1]=='RUNNING'
assert not (r/'20_mins_monitoring/controller_error.json').exists()
proof={'kind':'actual_new_monitor_independently_captured_full_tick','passed':True,'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'monitor_job_id':'6782816','actual_step_id':os.environ['SLURM_STEP_ID'],'source_commit':source,'source_returncode':checked.returncode,'returncode':result.returncode,'queue_counts':s['campaign']['queue_counts'],'all_five_sections':True,'exact_report_stdout_status_framing':True,'initial_startup_status_snapshot_claimed':False,'actual_first_own_log_retained_separately':True,'report':save('_report.md',report),'status':save('_status.json',status),'old_monitor_preserved':'6781231','gpu_actions':0,'release_or_cancel_calls':0,'scientific_credit':0,'admission_granted':False}
record=save('_verification.json',(json.dumps(proof,indent=2)+'\n').encode())
print(json.dumps({'passed':True,'step':proof['actual_step_id'],'verification':record,'counts':proof['queue_counts']}))
