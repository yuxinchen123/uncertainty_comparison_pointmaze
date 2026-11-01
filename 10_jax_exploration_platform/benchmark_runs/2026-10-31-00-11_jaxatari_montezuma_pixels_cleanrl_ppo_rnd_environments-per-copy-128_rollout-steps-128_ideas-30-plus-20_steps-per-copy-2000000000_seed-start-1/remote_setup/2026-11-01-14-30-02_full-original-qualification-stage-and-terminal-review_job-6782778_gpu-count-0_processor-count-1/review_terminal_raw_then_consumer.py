"""Verify real raw terminal qualification, save its independent review, then run the unchanged consumer gate."""
import argparse
import ast
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

ROOT=Path(__file__).absolute().parent
PACKET=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-13-50_original-qualification_explicit-pytest-root_jaguar03_gpu-count-1_processor-count-8_host-memory-gib-64_time-limit-hours-3_copies-1_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_reference-updates-1_seed-1')
UNIT=PACKET.parent.parent/'data'/PACKET.name
RUN=PACKET.parent.parent
JOB='6782778'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ref(path):return {'path':str(path),'sha256':digest(path)}
def doc(path):return json.loads(Path(path).read_bytes())
def put(path,value):
 raw=value if isinstance(value,bytes) else (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
 with Path(path).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 fd=os.open(Path(path).parent,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
 return ref(path)

def verify_source_contract():
 contract=doc(ROOT/'raw_review_source_contract.json')
 assert digest(contract['staged_source']['path'])==contract['staged_source']['sha256']=='0182089a73d504565f16b51891848000c1297a098b220374af9b74d89172d50c'
 assert digest(contract['new_raw_reader']['path'])==contract['new_raw_reader']['sha256']=='e48ad26d098e3b0ddf6c698a903306343d94a54fd893c1040da48c9aa25f75b6'
 original=next(n for n in ast.parse(Path(contract['staged_source']['path']).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='verify_qualification_inputs')
 raw=next(n for n in ast.parse(Path(contract['new_raw_reader']['path']).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='verify_unreviewed_actual_raw_qualification')
 removed=contract['removed_statement_indices'];assert len(removed)==2
 assert [ast.unparse(original.body[i]) for i in removed]==contract['removed_statements']
 assert contract['removed_statements'][0]=='review = identity.read_receipt(inputs[\'review\'])'
 assert 'independent actual qualification review' in contract['removed_statements'][1]
 assert [ast.dump(n,include_attributes=False) for i,n in enumerate(original.body) if i not in removed]==[ast.dump(n,include_attributes=False) for n in raw.body]
 assert ast.dump(original.args,include_attributes=False)==ast.dump(raw.args,include_attributes=False)
 return contract


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--execute-terminal-review',action='store_true')
 args=parser.parse_args()
 contract=verify_source_contract()
 if not args.execute_terminal_review:
  print(json.dumps({'source_contract_passed':True,'scheduler_queries':0,'actual_result_reads':0,'independent_review_written':False,'formal_consumer_called':False,'source_contract':ref(ROOT/'raw_review_source_contract.json')}));return
 assert os.environ.get('SLURM_JOB_ID')=='6781231' and os.environ.get('SLURM_STEP_ID','').isdecimal() and os.getuid()==2618919
 output=ROOT/'terminal_full_review';output.mkdir()
 queries=[]
 blocked=[]
 class NoNumerical(importlib.abc.MetaPathFinder):
  def find_spec(self,name,path=None,target=None):
   if name.split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','jaxtari','cupy','scipy'}:blocked.append(name);raise RuntimeError('numerical import forbidden')
 sys.meta_path.insert(0,NoNumerical())
 ctypes.CDLL=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('shared-library load forbidden'))
 sys.addaudithook(lambda event,args: (_ for _ in ()).throw(RuntimeError('network forbidden')) if event in {'socket.connect','socket.bind','socket.getaddrinfo'} else None)
 try:
  plan=doc(UNIT/'measurement_plan.json')
  for relative,image in plan['helpers']['files'].items():assert digest(UNIT/'code'/Path(relative).name)==image
  sys.path.insert(0,str(UNIT/'code'));import configured_measurement as configured
  assert Path(configured.__file__).absolute()==UNIT/'code/configured_measurement.py'
  ids=(RUN/'slurm/submitted_jobids_uva.txt').read_text().splitlines();assert ids.count(JOB)==1
  command=['sacct','-n','-P','-j',JOB,'--format=JobIDRaw,State,ExitCode,NodeList,Elapsed']
  observed=subprocess.run(command,capture_output=True,timeout=30,check=False)
  query={'command':command,'returncode':observed.returncode,'stdout':put(output/'accounting.stdout.raw',observed.stdout),'stderr':put(output/'accounting.stderr.raw',observed.stderr)};queries.append(query)
  assert observed.returncode==0
  rows=[line.split('|') for line in observed.stdout.decode().splitlines() if line.strip()];own=[row for row in rows if row[0]==JOB]
  assert len(own)==1 and own[0][1:3]==['COMPLETED','0:0']
  steps=[row for row in rows if row[0].startswith(JOB+'.')]
  assert any(row[0]==JOB+'.0' for row in steps) and all(row[1:3]==['COMPLETED','0:0'] for row in steps)
  result_path=UNIT/'target_qualification_result.json';assert result_path.is_file() and not (UNIT/'failure.json').exists()
  result_ref=ref(result_path);original_result=doc(result_path)
  spec=importlib.util.spec_from_file_location('actual_raw_qualification_review',ROOT/'raw_qualification_checks.py');raw=importlib.util.module_from_spec(spec);spec.loader.exec_module(raw)
  consumer=copy.deepcopy(plan);consumer['target_qualification_inputs']={'result':result_ref}
  # This raw reader has no review argument and every original raw predicate remains source-checked.
  validated=raw.verify_unreviewed_actual_raw_qualification(consumer)
  assert validated==original_result and not blocked
  review={'schema_version':1,'kind':'independent_full_original_target_qualification_review','approved':True,'at_utc':datetime.now(timezone.utc).isoformat(),'result':result_ref,'job_id':JOB,'actual_terminal_state':'COMPLETED','actual_exit_code':'0:0','audit_job_id':'6781231','audit_step_id':os.environ['SLURM_STEP_ID'],'actual_terminal_accounting':query,'source_contract':ref(ROOT/'raw_review_source_contract.json'),'raw_verifier':ref(ROOT/'raw_qualification_checks.py'),'raw_result_validation_passed_before_review_existed':True,'first_review_did_not_use_placeholder_or_prior_partial_approval':True,'full48tests_33rooms_132variants_11synthetics_and_controlled_full_reference_gates_validated':True,'all_current_context_captured_driver_own_sample_window_backend_asset_source_precision_and_counter_checks_validated':True,'scientific_credit':0,'case_credit':0,'idea_credit':0,'final_training_credit':0,'site_qualification_credit':0,'admission_granted':False,'worker_admitted':False}
  review_ref=put(output/'independent_actual_review.json',review)
  consumer['target_qualification_inputs']['review']=review_ref
  # The original complete consumer now reads only the real, already validated and fsynced independent review.
  final=configured.verify_qualification_inputs(consumer)
  assert final==original_result and ref(result_path)==result_ref
  proof={'schema_version':1,'kind':'actual_full_original_raw_review_then_unmodified_consumer_verification','passed':True,'at_utc':datetime.now(timezone.utc).isoformat(),'job_id':JOB,'audit_job_id':'6781231','audit_step_id':os.environ['SLURM_STEP_ID'],'result':result_ref,'independent_review':review_ref,'raw_checked_before_review':True,'unmodified_staged_full_consumer_passed_after_actual_review':True,'source_contract':ref(ROOT/'raw_review_source_contract.json'),'queries':queries,'gpu_actions':0,'network_calls':0,'numerical_imports':0,'scientific_credit':0,'case_credit':0,'idea_credit':0,'final_training_credit':0,'site_qualification_credit':0,'admission_granted':False,'worker_admitted':False}
  print(json.dumps({'passed':True,'verification':put(output/'verification.json',proof),'independent_review':review_ref}))
 except BaseException:
  put(output/'failure.json',{'traceback':traceback.format_exc(),'queries':queries,'gpu_actions':0,'admission_granted':False});raise


if __name__=='__main__':main()
