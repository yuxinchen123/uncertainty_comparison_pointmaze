"""Inspect one current stage or verify a completed result using the exact staged reader."""
import argparse
import copy
import ctypes
from datetime import datetime,timezone
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

AUDIT=Path(__file__).absolute().parent
PACKET=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-13-50_original-qualification_explicit-pytest-root_jaguar03_gpu-count-1_processor-count-8_host-memory-gib-64_time-limit-hours-3_copies-1_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_reference-updates-1_seed-1')
UNIT=PACKET.parent.parent/'data'/PACKET.name
RUN=PACKET.parent.parent
BLOCKED=[]
CHECKS=[]
assert os.environ.get('SLURM_JOB_ID')=='6781231' and os.environ.get('SLURM_STEP_ID','').isdecimal()

def image(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def doc(path):return json.loads(Path(path).read_bytes())
def ref(path):return {'path':str(path),'sha256':image(path)}
def check(value,message):
 assert value,message
 CHECKS.append(message)
def save(name,value):
 raw=value if isinstance(value,bytes) else (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
 with (AUDIT/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return {'path':str(AUDIT/name),'sha256':hashlib.sha256(raw).hexdigest()}

class NoNumerical(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','jaxtari','scipy','cupy'}:
   BLOCKED.append(name);raise RuntimeError('numerical import forbidden')
sys.meta_path.insert(0,NoNumerical())
ctypes.CDLL=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('shared-library load forbidden'))
sys.addaudithook(lambda event,args: (_ for _ in ()).throw(RuntimeError('network forbidden')) if event in {'socket.connect','socket.bind','socket.getaddrinfo'} else None)

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--independent-review',type=Path)
parser.add_argument('--verify-terminal',action='store_true')
args=parser.parse_args()
if args.verify_terminal:
 assert args.independent_review is not None and args.independent_review.is_absolute() and args.independent_review.resolve(strict=True)==args.independent_review and args.independent_review.is_file()
 AUDIT=AUDIT/'terminal_reader_output'
 AUDIT.mkdir()
assert args.verify_terminal==bool(args.independent_review),'terminal mode requires an actual existing independent review'
try:
 plan=doc(UNIT/'measurement_plan.json')
 # Every imported helper must be the exact frozen staged artifact, before any reader is called.
 for relative,digest in plan['helpers']['files'].items():check(image(UNIT/'code'/Path(relative).name)==digest,'staged helper '+Path(relative).name+' unchanged')
 sys.path.insert(0,str(UNIT/'code'));import configured_measurement as configured
 check(Path(configured.__file__).absolute()==UNIT/'code/configured_measurement.py','reader imported from exact staged path')
 core=configured.core();core.validate_plan(plan)
 configured.validate_original_qualification(plan)
 configured.verify_saved_sources(UNIT,plan)
 configured.verify_qualification_source_receipts(plan)
 check(True,'exact staged full-original source,20helpers,CleanRL/shared-software receipts validate read-only')
 metadata=doc(UNIT/'configured_isolation_metadata.json');binding=doc(UNIT/'resolved_device_binding.json');core.validate_binding(plan,metadata,binding,UNIT)
 check(True,'own numeric step,unmodified selectors,current CUDA/JAX context,physicalUUID/PCI/minor and captured580driver validate')
 tests=doc(UNIT/'original_tests.json');configured.validate_test_collection(plan,tests,(UNIT/'original_tests.stdout').read_text())
 check(len(tests['collected_nodeids'])==48 and len(tests['reports'])==144 and not tests['deselected_nodeids'] and all(row['outcome']=='passed' for row in tests['reports']),'actual complete48tests/144setup-call-teardown outcomes pass without deselection')
 backend=doc(UNIT/'installed_backend.json');matrix=configured.renderer_contract(plan,backend);renderer=doc(UNIT/'original_renderer.json')
 check(matrix==plan['renderer_contract'] and all(type(renderer[k]) is int and renderer[k]==matrix[k] for k in ['rooms','cases','synthetic_pixel_cases']) and renderer['checks']=={'cache':'passed','palette':'passed','fused':'passed'},'actual full33-room132-variant11-synthetic renderer passes source-bound matrix')
 configured.verify_qualification_backend(plan,backend)
 check(True,'actual installed backend equals original shared-software receipt')
 snapshots={}
 for name in ['original_tests.json','original_tests.stdout','original_tests.stderr','original_renderer.json','resolved_device_binding.json','configured_isolation_metadata.json','hardware.json','installed_backend.json','asset_receipt.json','assigned_process_admission.json']:
  f=UNIT/name
  if f.exists():snapshots[name]=save(name+'.snapshot',f.read_bytes())
 resultpath=UNIT/'target_qualification_result.json';failurepath=UNIT/'failure.json'
 result_exists=resultpath.is_file();failure_exists=failurepath.is_file()
 for f in [resultpath,failurepath,UNIT/'progress.json']:
  if f.is_file():snapshots[f.name]=save(f.name+'.snapshot',f.read_bytes())
 command=['sacct','-n','-P','-j','6782778','--format=JobIDRaw,State,ExitCode,NodeList,Elapsed']
 check((RUN/'slurm/submitted_jobids_uva.txt').read_text().splitlines().count('6782778')==1,'exact queried target exists once in own ledger')
 query=subprocess.run(command,capture_output=True,timeout=30,check=False)
 query_ref={'command':command,'returncode':query.returncode,'stdout':save('terminal_accounting.stdout.raw',query.stdout),'stderr':save('terminal_accounting.stderr.raw',query.stderr)}
 check(query.returncode==0,'exact-owned-ID accounting query returned0')
 rows=[line.split('|') for line in query.stdout.decode().splitlines() if line.strip()];job=[row for row in rows if row[0]=='6782778'];check(len(job)==1,'actual job accounting row is unique')
 actual_state,actual_exit=job[0][1:3]
 full_gate_passed=False
 if args.verify_terminal:
  check(actual_state=='COMPLETED' and actual_exit=='0:0','actual target completed0:0 before full-gate review')
  check(all(row[1]=='COMPLETED' and row[2]=='0:0' for row in rows if row[0].startswith('6782778.')) and any(row[0]=='6782778.0' for row in rows),'actual numeric step0 and every saved auxiliary step completed0:0')
  check(result_exists and not failure_exists,'actual dedicated target result exists without failure')
  review=args.independent_review.absolute();check(review==review.resolve(strict=True) and review.is_file(),'actual independent terminal review is exact existing ordinary path')
  consumer=copy.deepcopy(plan)
  consumer['target_qualification_inputs']={'result':ref(resultpath),'review':ref(review)}
  # This exact unmodified reader checks all controlled counters,precision,target/policy,renderer,test,library,asset and full-window facts.
  verified=configured.verify_qualification_inputs(consumer)
  check(verified==doc(resultpath),'exact staged full-original qualification reader passed actual terminal bytes')
  full_gate_passed=True
 check(not BLOCKED,'no numerical import attempted by metadata review')
 report={'schema_version':1,'kind':'full_original_qualification_current_stage_assessment' if not args.verify_terminal else 'full_original_qualification_terminal_reader_verification','at_utc':datetime.now(timezone.utc).isoformat(),'passed':True,'audit_job_id':'6781231','audit_step_id':os.environ['SLURM_STEP_ID'],'target_job_id':'6782778','actual_state':actual_state,'actual_exit_code':actual_exit,'result_present':result_exists,'failure_present':failure_exists,'full_gate_reader_passed':full_gate_passed,'last_completed_gates':['48originaltests/144phases','full33rooms/132variants/11syntheticsrenderer','source/software/assets/current-context/captured-driver'],'controlled_reference_work':'unknown while dedicated target result is absent; do not infer zero','test_fixture_work':'unquantified actual test/renderer work; no whole-job total claimed','snapshots':snapshots,'accounting':query_ref,'checks':CHECKS,'check_count':len(CHECKS),'actual_result':ref(resultpath) if result_exists else None,'source_commit':plan['source_commit'],'helper_commit':plan['helpers']['source_commit'],'plan':ref(UNIT/'measurement_plan.json'),'physical_uuid':binding['gpu_uuid'],'physical_pci':binding['pci_address'],'physical_minor':binding['physical_minor'],'physical_gres_mapping_verified':binding['physical_gres_mapping_verified'],'kernel_device_enforcement_verified':binding['kernel_device_enforcement_verified'],'installed_binary_matches_upstream_verified':binding['installed_binary_matches_upstream_verified'],'scientific_credit':0,'case_credit':0,'idea_credit':0,'final_training_credit':0,'admission_granted':False,'worker_admitted':False,'gpu_actions_by_review':0,'network_actions_by_review':0,'numerical_imports':0}
 print(json.dumps({'passed':True,'receipt':save('assessment.json' if not args.verify_terminal else 'terminal_reader_verification.json',report),'actual_step':os.environ['SLURM_STEP_ID'],'actual_state':actual_state,'result_present':result_exists,'failure_present':failure_exists,'full_gate_reader_passed':full_gate_passed}))
except BaseException:
 save('failure.json',{'traceback':traceback.format_exc(),'checks_before_failure':CHECKS,'audit_job_id':'6781231','audit_step_id':os.environ['SLURM_STEP_ID'],'gpu_actions':0,'admission_granted':False});raise
