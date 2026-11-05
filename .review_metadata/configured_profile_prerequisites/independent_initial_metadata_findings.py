import ast,hashlib,importlib.util,json,os,sys,tempfile,time
from pathlib import Path
from unittest.mock import patch
W=Path('/p/rlprojects/RND_worktrees/2026-11-05-07-34_a4500-profile-preparation_common-gpu-types-2_copies-1-2_reviewed-native-process-contract_admission-closed_gpu-operations-0');D=W/'.review_metadata/configured_profile_prerequisites';S=W/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
assert os.environ['SLURM_JOB_ID']=='6821437'
freeze=D/'source_freeze_3.json';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();assert sha(freeze)=='6bd5bea67787e3f09de1bc36eb4f192a0b8ab4f748ac56aa3b66eba4e3341e6d';rows=json.loads(freeze.read_bytes())['paths']
for name,row in rows.items():assert sha(W/name)==row['sha256'] and (W/name).stat().st_mode&0o777==row['mode']
spec=importlib.util.spec_from_file_location('independent_bounded_profile_prerequisites',S/'configured_profile_prerequisites.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
proof_calls=[];original=Path.read_bytes
with tempfile.TemporaryDirectory(prefix='rnd_profile_metadata_bound_fixture_') as temporary:
 proof=Path(temporary)/'explicit_synthetic_oversized_proof.json';review=Path(temporary)/'explicit_synthetic_review.json'
 with proof.open('wb') as out:out.truncate(helper.MAXIMUM_METADATA_BYTES+1)
 review.write_text('{}')
 def refuse_unbounded_read(path):
  if path==proof:
   proof_calls.append(str(path));raise RuntimeError('synthetic oversized proof read reached before bound refusal')
  return original(path)
 with patch.object(Path,'read_bytes',refuse_unbounded_read):
  for entry,call in [('confirmation_metadata',lambda:helper.confirmation_metadata(proof,review,'jaguar03_rtx_a4500',1)),('configure_plan',lambda:helper.configure_plan({},proof,review))]:
   count=len(proof_calls)
   try:call()
   except RuntimeError as error:assert str(error)=='synthetic oversized proof read reached before bound refusal'
   else:raise AssertionError('fixture did not reach the unbounded read')
   assert len(proof_calls)==count+1
source=ast.parse((S/'profile_selected.py').read_bytes());nodes={n.name:n for n in source.body if isinstance(n,ast.FunctionDef)}
producer=ast.unparse(nodes['run_configured_native_prepared']);assert producer.index('verify_profile_target_qualification(plan)')<producer.index("installed =")
installed=next(n.value for n in ast.walk(nodes['run_configured_native_prepared']) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='installed' for t in n.targets))
assert {n.value for n in installed.keys}=={'jaxtari_source','gpu_libraries'}
assert isinstance(nodes['require_reviewed_profile_process_admission'].body[-1],ast.Raise)
for name,row in rows.items():assert sha(W/name)==row['sha256']
record={'kind':'independent_a4500_profile_prerequisite_initial_findings','approved':False,'held_freeze':{'path':str(freeze),'sha256':sha(freeze)},'required_corrections':[{'finding':'Proof and review reference derivation reads bytes before the declared size/owner/regular-file/stability checks.','reproduced_entries':['confirmation_metadata','configure_plan'],'fixture':'A private synthetic sparse17MiB proof was never read; the read_bytes spy raised before metadata bound validation.'},{'finding':'The new runtime prerequisite route checks historical facts before collecting installed metadata, and its installed record has no package-version field or explicit comparison with the accepted original qualification.','actual_installed_record_keys':['jaxtari_source','gpu_libraries'],'execution_guard_remains_closed':True}],'metadata_fixture_only':True,'actual_large_payload_bytes_read':0,'actual_profile_preparation_or_GPU_or_numerical_or_remote_actions':0,'scientific_credit':0,'job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],'uid':os.getuid(),'pid':os.getpid(),'source_bytes_unchanged':True}
p=D/'independent_initial_metadata_findings.json'
with p.open('x') as out:json.dump(record,out,indent=2);out.write('\n');out.flush();os.fsync(out.fileno())
print(json.dumps({'findings':2,'approved':False,'record':str(p),'sha256':sha(p),'job_id':record['job_id'],'step_id':record['step_id']}))
