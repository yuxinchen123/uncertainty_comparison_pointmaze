"""Validate only private request/source metadata; never call prepare/verify/claim/bind."""
import ast,copy,hashlib,json,os,stat,time
from pathlib import Path
W=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent;Q=D/'proposal';S=W/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
if os.environ.get('SLURM_JOB_ID')!='6821437' or not os.environ.get('SLURM_STEP_ID','').isdigit():raise ValueError('exact owned numeric CPU step required')
def read(p):return json.loads(p.read_bytes())
def ref(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
started=time.time();derivation=read(Q/'derivation.json');basepath=Path(derivation['base_request']['path']);base=read(basepath);req=read(Q/'preparation_request.json');procedure=read(Q/'planned_root_metadata_procedure.json')
expected=copy.deepcopy(base)
for key in ('unit_id','packet','data_unit'):expected[key]=req[key]
expected.update(copy_seeds=[1,2],serial_order=2,serial_predecessor_unit=str(basepath.parent.parent.parent/'data'/basepath.parent.name))
for key in ('reference_configuration','configuration'):expected[key]['learner']['n_copies']=2
expected['common_correctness_inputs']={key:expected['accepted_correctness']['2'][key] for key in ('result','review')}
expected.pop('wrapper_source_commit',None)
if req!=expected:raise ValueError('request changed beyond exact two-copy derivation')
if Path(req['packet']).exists() or Path(req['data_unit']).exists():raise ValueError('root paths already exist')
if req['helper_source_commit']!='69f1b5075530418adbe02c1e433c5647848a61ba' or req['numerical_source_commit']!='d28e6735e79788f618acb0ebfde8609db1f776af':raise ValueError('fixed source changed')
reference,candidate=req['reference_configuration'],copy.deepcopy(req['reference_configuration']);candidate['environment']['cache_room_backgrounds']=True;candidate['environment']['render_final_two_only']=True
if candidate!=req['configuration']:raise ValueError('candidate has an extra optimization')
for config in (reference,candidate):
 for key,value in (('n_copies',2),('n_envs',128),('num_steps',128),('update_epochs',4),('num_minibatches',4),('observation_warmup_iterations',50),('total_timesteps',2000000000)):
  if type(config['learner'][key]) is not int or config['learner'][key]!=value:raise ValueError('strict original configuration changed')
if req['copy_seeds']!=[1,2] or any(type(seed) is not int for seed in req['copy_seeds']) or req['rounds']!=7 or req['minimum_updates_per_round']!=10:raise ValueError('measurement protocol changed')
review=Path(derivation['accepted_one_copy_review']['path'])
if ref(review)!=derivation['accepted_one_copy_review']:raise ValueError('accepted H1 review changed')
prior=read(Q/'predecessor_available_records.json');actual_review=read(review)
if prior['result']!=actual_review['result'] or prior['timing_review']!=ref(review):raise ValueError('actual H1 acceptance changed')
wait=read(Path(prior['wait']['path']))
if type(wait['returncode']) is not int or wait['returncode']!=0 or wait['exited_normally'] is not True or wait['timeouts']!=[]:raise ValueError('actual normal wait missing')
oldast=ast.parse((D/'controller_published_d225.py').read_bytes());newast=ast.parse((S/'direct_common_timing_controller.py').read_bytes());fn=lambda tree:{n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)}
old,new=fn(oldast),fn(newast);changed=[name for name in old if old[name]!=new[name]]
if set(old)!=set(new) or changed!=['check_serial_predecessor_inputs']:raise ValueError('controller changed outside its exact serial predicate')
plan=read(Path(prior['plan']['path']));protected={}
for section in ('core','paired_source'):
 for relative,entry in plan[section]['files'].items():
  expected_hash=entry['sha256'] if type(entry) is dict else entry
  source=Path(plan[section]['source_root'])/relative
  if ref(source)['sha256']!=expected_hash or ref(W/relative)['sha256']!=expected_hash:raise ValueError('a protected numerical source changed')
  protected[relative]={'source':ref(source),'current_private_checkout':ref(W/relative)}
if len(protected)!=13:raise ValueError('complete thirteen numerical sources are required')
helpers={}
for relative,expected_hash in plan['helpers']['files'].items():
 source=Path(plan['helpers']['source_root'])/relative
 if ref(source)['sha256']!=expected_hash:raise ValueError('selected private69f helper changed')
 helpers[relative]=ref(source)
if len(helpers)!=29:raise ValueError('complete selected helper closure required')
files=[S/'direct_common_timing_controller.py',S/'tests/test_timing_predecessor_actual_schema.py',S/'prepare_direct_common_timing.py',D/'controller_published_d225.py',D/'author_source_freeze_1.json',D/'author_checks_6821437_134.json',D/'build_private_request.py',D/'validate_and_freeze_request.py',*sorted(Q.iterdir())]
freeze=D/'request_and_source_freeze.json';write(freeze,{'kind':'private_two_copy_h100_request_and_narrow_controller_source_freeze','files':{str(p):ref(p)['sha256'] for p in files},'source_roster':{'protected_numerical_13':protected,'selected_helper_29':helpers},'changed_controller_functions':changed,'unchanged_controller_function_count':len(old)-len(changed),'unchanged_controller_functions':[name for name in old if name not in changed],'entire_controller_unchanged':False,'original_serial1_correctness_contract_unchanged':True,'actual_phase_invocations':0,'future_actual_packet_not_created':True,'source_wrapper_future_publication_commit_not_guessed':True})
step=os.environ['SLURM_STEP_ID'];out=D/('request_validation_6821437_'+step+'.json')
write(out,{'kind':'private_two_copy_request_metadata_validation','passed':True,'source_freeze':ref(freeze),'job_id':os.environ['SLURM_JOB_ID'],'step_id':step,'started_at_seconds':started,'finished_at_seconds':time.time(),'changed_controller_functions':changed,'unchanged_controller_functions':len(old)-len(changed),'protected_numerical_count':len(protected),'helper_count':len(helpers),'request_semantics_exact':True,'actual_prepare_verify_claim_bind_ssh_or_gpu_invocations':0,'actual_result_payload_reread':False,'scientific_credit':0})
print(json.dumps({'receipt':str(out),'sha256':ref(out)['sha256'],'freeze_sha256':ref(freeze)['sha256'],'step':step,'unchanged_controller_functions':len(old)-len(changed),'elapsed_seconds':time.time()-started},sort_keys=True))
