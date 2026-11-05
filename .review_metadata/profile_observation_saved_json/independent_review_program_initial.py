"""Independent narrow A4500 saved-observation caller review, CPU metadata only."""
import ast,copy,datetime,hashlib,importlib.util,io,json,os,pathlib,socket,stat,subprocess,sys,time,unittest
START=time.monotonic();CHECKS=[];FILES={}
W=pathlib.Path('/p/rlprojects/RND_worktrees/2026-11-05-14-20_profile-observation_saved-json-canonical-process-keys_gpu-actions-0');D=W/'.review_metadata/profile_observation_saved_json'
PREFIX='10_jax_exploration_platform/scripts/atari_pixel_campaign/';BASE='ef00da972d4a5c28b6a71a71da6f9c7d31aea18f'
def check(v,m):
 if not v:raise ValueError(m)
 CHECKS.append(m)
def read(p):
 p=pathlib.Path(p);s=p.lstat();check(p.resolve(strict=True)==p and stat.S_ISREG(s.st_mode) and s.st_uid==os.getuid() and s.st_size<=2500000,'bounded ordinary '+str(p))
 data=p.read_bytes();t=p.lstat();check((s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode)==(t.st_dev,t.st_ino,t.st_size,t.st_mtime_ns,t.st_ctime_ns,t.st_mode)and len(data)==s.st_size,'stable '+str(p))
 ref={'path':str(p),'sha256':hashlib.sha256(data).hexdigest(),'size_bytes':len(data)};FILES[str(p)]=ref;return data,ref
def document(p):
 raw,ref=read(p);return json.loads(raw),ref
def git(*args):
 e={k:v for k,v in os.environ.items()if not k.startswith('GIT_')};e.update(GIT_OPTIONAL_LOCKS='0',GIT_TERMINAL_PROMPT='0')
 r=subprocess.run(['git','-C',str(W),*args],capture_output=True,check=True,timeout=30,env=e);check(r.stderr==b'','read-only Git '+args[0]);return r.stdout
def methods(raw):return {n.name:n for n in ast.parse(raw).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
def dump(n):return ast.dump(n,include_attributes=False)
check(os.environ.get('SLURM_JOB_ID')=='6825983','exact current owned CPU allocation')
freeze,FR=document(D/'corrected_source_freeze.json')
check(FR['sha256']=='0655f491eb27c3415916fa9b59505c3708556f4d540a1fdf109107f328af526f' and freeze['base_commit']==BASE and len(freeze['files'])==2,'exact held two-file freeze')
for rel,record in freeze['files'].items():
 raw,ref=read(W/rel);check(ref['sha256']==record['sha256'] and len(raw)==record['size_bytes']and stat.S_IMODE((W/rel).stat().st_mode)==record['mode'],'exact frozen source '+rel)
new=read(W/PREFIX/'profile_selected.py')[0];old=git('show',BASE+':'+PREFIX+'profile_selected.py')
oldmethods,newmethods=methods(old),methods(new)
check(oldmethods.keys()==newmethods.keys(),'same source function roster')
for name in oldmethods:
 if name!='capture_configured_native_processes':check(dump(oldmethods[name])==dump(newmethods[name]),'unchanged function AST '+name)
before=copy.deepcopy(oldmethods['capture_configured_native_processes']);after=newmethods['capture_configured_native_processes']
calls=[n for n in ast.walk(before)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='validate_observation']
check(len(calls)==1 and isinstance(calls[0].args[0],ast.Name)and calls[0].args[0].id=='observation','actual old erroneous live-dict argument')
calls[0].args[0]=ast.parse("observer.read_json(directory / 'capture_process_observation.json')",mode='eval').body
check(dump(before)==dump(after),'only exact canonical published-JSON argument changes in failed function')
changed=git('diff','--name-only',BASE,'--','10_jax_exploration_platform').decode().splitlines()
check(changed==[PREFIX+'profile_selected.py'],'all other tracked runtime, NUM and component helper bytes unchanged')
identity=read(W/PREFIX/'nsight_process_identity.py')[0]
check(identity==git('show',BASE+':'+PREFIX+'nsight_process_identity.py'),'protected strict source remains byte-identical')
author,AUTHORREF=document(D/'author_cpu_source_check.json');check(author['passed']is True and author['methods']==4,'author CPU proof retained')
actual,ACTREF=document(D/'actual_failed_unit_metadata_references.json')
check(ACTREF['sha256']=='1c272169a3dc4c56e31dc9a756fb96355deb1d6dccc3cc3ee9d00c0ed013bcb1'and actual['job_id']=='6826559','exact failed actual unit metadata references')
for reference in list(actual['records'].values())+[actual['staged_failed_caller'],actual['strict_reader']]:
 raw,ref=read(reference['path']);check(ref==reference,'actual failed record exact '+reference['path'])
failure=json.loads(read(pathlib.Path(actual['child_unit'])/'failure.json')[0])
observation=json.loads(read(pathlib.Path(actual['child_unit'])/'capture_process_observation.json')[0])
check("saved process keys must be exact canonical PID strings"in failure['error']and 'line 1293'in failure['traceback'],'actual caller failure diagnosis unchanged')
check(all(type(k)is str and str(int(k))==k for k in observation['nodes']),'actual already-published failed observation has canonical string PID keys')
testpath=W/PREFIX/'tests/test_profile_observation_saved_json.py'
artifacts=D/'independent_actual_cpu_artifacts';artifacts.mkdir(exist_ok=False)
os.environ['PROFILE_OBSERVATION_TEST_ARTIFACT_DIR']=str(artifacts)
spec=importlib.util.spec_from_file_location('independent_saved_observation_boundary_test',testpath);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
suite=unittest.defaultTestLoader.loadTestsFromTestCase(module.SavedObservationBoundaryTests)
log=io.StringIO();result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
logdata=log.getvalue().encode();logpath=D/'independent_focused_tests.log'
with logpath.open('xb')as stream:stream.write(logdata);stream.flush();os.fsync(stream.fileno())
check(result.wasSuccessful()and result.testsRun==4,'four focused methods including one fresh own CPU child pass')
for name in ['actual_cpu_observation.json','actual_cpu_wait.json']:read(artifacts/name)
wait,_=document(artifacts/'actual_cpu_wait.json')
check(type(wait['returncode'])is int and wait['returncode']==0 and wait['stderr']==''and 'canonical PID strings'in wait['old_live_reader_failure'],'actual own CPU child normal0 and old boundary reproduction')
# Independent pure saved-data controls use the real unchanged reader, without another process.
for variant in ['integer-key','signed-key','birth','raw-status-digest']:
 altered=copy.deepcopy(module.SavedObservationBoundaryTests.saved);pid=str(module.SavedObservationBoundaryTests.claim['pid']);reader=module.SavedObservationBoundaryTests.reader
 if variant=='integer-key':altered['nodes'][int(pid)]=altered['nodes'].pop(pid)
 elif variant=='signed-key':altered['nodes']['+'+pid]=altered['nodes'].pop(pid)
 elif variant=='birth':altered['nodes'][pid]['process_start_time']+=1
 elif variant=='raw-status-digest':altered['nodes'][pid]['status_sha256']='0'*64
 try:reader.validate_observation(altered,module.SavedObservationBoundaryTests.claim,module.SavedObservationBoundaryTests.command,module.SavedObservationBoundaryTests.parent)
 except (ValueError,KeyError):check(True,'independent saved mutation refuses '+variant)
 else:raise ValueError('saved mutation accepted '+variant)
for rel,record in freeze['files'].items():check(read(W/rel)[1]['sha256']==record['sha256'],'frozen source unchanged after focused check '+rel)
check(not any(name.split('.')[0]in ('jax','jaxlib','numpy','torch')for name in sys.modules),'no numerical runtime imported')
receipt=dict(schema_version=1,kind='independent_profile_observation_saved_json_source_review',approved=True,passed=True,independent_of_author=True,source_freeze=FR,source_base=BASE,changed_runtime_functions=['capture_configured_native_processes'],runtime_change='Validate exact already-published observation via unchanged read_json; the protected validator and original live collector remain unchanged.',actual_failed_unit_metadata=ACTREF,author_cpu_proof=AUTHORREF,files=FILES,checks=CHECKS,check_count=len(CHECKS),tests_run=result.testsRun,test_log={'path':str(logpath),'sha256':hashlib.sha256(logdata).hexdigest()},actual_job_id=os.environ['SLURM_JOB_ID'],actual_step_id=os.environ['SLURM_STEP_ID'],reviewer_pid=os.getpid(),reviewer_uid=os.getuid(),reviewer_hostname=socket.gethostname(),reviewed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=time.monotonic()-START,CPU_component_scope='One fresh own system-Python child with stdlib process handles; this is not a GPU/native-provider/Nsight qualification or a repeat of the prior scope lifecycle.',original_failed_unit_reexecuted=False,old_component_diagnostics_reexecuted=False,actual_GPU_Nsight_target_actions=False,scientific_credit=0,profile_or_worker_admission=False,unresolved_anomalies=[])
out=D/'independent_corrected_source_review.json';raw=(json.dumps(receipt,indent=2,sort_keys=True)+'\n').encode()
with out.open('xb')as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
fd=os.open(D,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
print(json.dumps(dict(path=str(out),sha256=hashlib.sha256(raw).hexdigest(),check_count=len(CHECKS),step=receipt['actual_step_id'],seconds=receipt['seconds'])))

