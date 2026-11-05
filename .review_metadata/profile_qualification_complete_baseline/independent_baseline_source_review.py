"""Independently check genuine nonwriting baseline derivation; not canonical staging."""
import ast
import copy
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import time

D=Path(__file__).resolve().parent;W=D.parents[1]
FREEZE_SHA='594d97dc0a605d6f9f89c9f58c23062e7417d570de935c23715f46de78f20216'
started=time.monotonic();checks=[]
def require(value,label):
    if not value:raise ValueError(label)
    if time.monotonic()-started>285:raise TimeoutError('independent review exceeded285seconds')
    checks.append(label)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path,key):
    spec=importlib.util.spec_from_file_location(key,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def git(*args):return subprocess.check_output(['git','-C',str(W),*args],env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'},stderr=subprocess.PIPE,timeout=30)
require(os.environ.get('SLURM_JOB_ID')=='6825983' and os.environ.get('SLURM_STEP_ID','').isdecimal(),'current exact ownedCPU')
require(sha(D/'source_freeze.json')==FREEZE_SHA,'exact held sourcefreeze')
freeze=json.loads((D/'source_freeze.json').read_bytes())
for relative,item in freeze['active_files'].items():require(sha(item['path'])==item['sha256'] and Path(item['path']).stat().st_size==item['size_bytes'],'exact active source '+relative)
for count,word in ((1,'one'),(2,'two')):
    active=W/'.review_metadata/profile_target_operational_preparation'/('derive_'+word+'_copy_qualification_request.py')
    before=D/('before_correctness_record_derive_'+word+'_copy_qualification_request.py')
    previous=ast.parse(before.read_text());current=ast.parse(active.read_text())
    found=0
    for node in ast.walk(current):
        if isinstance(node,ast.Dict):
            positions=[i for i,key in enumerate(node.keys) if isinstance(key,ast.Constant) and key.value=='correctness_record']
            for i in reversed(positions):del node.keys[i];del node.values[i];found+=1
    require(found==1 and ast.dump(previous,include_attributes=False)==ast.dump(current,include_attributes=False),
        'only genuine correctness_record field added to '+word+'copy producer')
require(git('diff','--name-only','b45d52cad17f80ad313cd19050aa3a8a8674fa7f','--','10_jax_exploration_platform')==b'',
    'runtime/helper/NUM tree unchanged in authorworktree')
sentinel=load(D/'check_complete_baseline_contract.py','independent_genuine_baseline_sentinel')
require(sha(sentinel.FACT/'four_context_prerequisite_fact.json')=='e75b78e9a9b3fd4a857c2d7c5148de15c8800ef14398c5b56b13f5c78db4d3c7','genuine fourcontext fact')
require(sha(sentinel.FACT/'independent_actual_four_context_fact_review.json')=='5e19415c03e8063275422d6651c5496de9e94fc63c3f1d4d12df27e11a71298d','genuine independent factreview')
require(sha(sentinel.BACKEND)=='19e752ef9b93999e8bec28a0706947935c31f4229a3b33556597de8f86575412','genuine fresh backendrecord')
# This is the reviewed first-mkdir sentinel, not a filesystem map or validator mock.
sys.addaudithook(sentinel.audit)
cases=[sentinel.check(copies,hardware) for hardware in ('h100','jaguar03_rtx_a4500') for copies in (1,2)]
require(len(cases)==4 and all(case['complete_baseline_contract_passed'] and case['output_creation_prevented']
    and not case['canonical_preparation_executed'] for case in cases),'four genuine metadata boundary cases passed without staging')
two=W/'.review_metadata/profile_target_operational_preparation/derive_two_copy_qualification_request.py'
derive=next(node for node in ast.parse(two.read_text()).body if isinstance(node,ast.FunctionDef) and node.name=='derive')
call=next(node.value for node in derive.body if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call)
    and isinstance(node.value.func,ast.Name) and node.value.func.id=='require' and len(node.value.args)>1
    and isinstance(node.value.args[1],ast.Constant) and node.value.args[1].value=='the selected actual common context differs')
predicate=compile(ast.Expression(call.args[0]),str(two),'eval')
fact=json.loads((sentinel.FACT/'four_context_prerequisite_fact.json').read_bytes())
negative_count=0
for hardware in ('h100','jaguar03_rtx_a4500'):
    entry=next(row for row in fact['contexts'] if row['hardware_class']==hardware and row['copies']==2)
    review=json.loads(Path(entry['review']['path']).read_bytes());common=json.loads(Path(review['plan']['path']).read_bytes())
    environment={'common':common,'run':sentinel.R,'args':SimpleNamespace(hardware_class=hardware)}
    require(eval(predicate,environment),'exact real two-copy predicate positive '+hardware)
    for seeds,count in (([1],1),([True,2],2),([1.0,2],2),([1,2],2.0),([1,2],True)):
        changed=copy.deepcopy(common);changed['copy_seeds']=seeds;changed['configuration']['learner']['n_copies']=count
        require(not eval(predicate,{**environment,'common':changed}),'strict two-copy typed count/seed negative '+hardware+' '+repr((seeds,count)));negative_count+=1
for relative,item in freeze['active_files'].items():require(sha(item['path'])==item['sha256'],'held bytes unchanged after '+relative)
record={'schema_version':1,'kind':'independent_profile_qualification_complete_baseline_operational_source_review',
    'approved':True,'passed':True,'independent_of_author':True,'source_freeze':{'path':str(D/'source_freeze.json'),'sha256':FREEZE_SHA},
    'active_files':freeze['active_files'],'actual_job_id':os.environ['SLURM_JOB_ID'],'actual_step_id':os.environ['SLURM_STEP_ID'],
    'checks':checks,'check_count':len(checks),'genuine_context_cases':cases,'independent_typed_two_copy_negative_cases':negative_count,
    'runtime_helpers_or_NUM_changed':False,'four_context_fact_sha256':'e75b78e9a9b3fd4a857c2d7c5148de15c8800ef14398c5b13f5c78db4d3c7',
    'four_context_review_sha256':'5e19415c03e8063275422d6651c5496de9e94fc63c3f1d4d12df27e11a71298d',
    'fresh_backend_record_sha256':'19e752ef9b93999e8bec28a0706947935c31f4229a3b33556597de8f86575412',
    'no_prerequisite_validator_mocked':True,'filesystem_mapping_used':False,'sentinel_only_at_first_output_mkdir':True,
    'canonical_staging_passed':False,'actual_output_directories_created':0,'root_fresh_actual_canonical_preparation_required':True,
    'scientific_credit':0,'admission_granted':False,'seconds':time.monotonic()-started,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
path=D/'independent_source_review.json'
with path.open('x') as stream:json.dump(record,stream,indent=2,sort_keys=True);stream.write('\n');stream.flush();os.fsync(stream.fileno())
print(json.dumps({'passed':True,'review':{'path':str(path),'sha256':sha(path)},'job':os.environ['SLURM_JOB_ID'],
    'step':os.environ['SLURM_STEP_ID'],'checks':len(checks),'seconds':record['seconds']}))
