"""Finalize the previously captured stage evidence without another target read or scheduler query."""
from datetime import datetime,timezone
import hashlib,json,os
from pathlib import Path
P=Path(__file__).absolute().parent
assert os.environ.get('SLURM_JOB_ID')=='6781231' and os.environ.get('SLURM_STEP_ID','').isdecimal()
def read(n):return json.loads((P/n).read_bytes())
def ref(n):return {'path':str(P/n),'sha256':hashlib.sha256((P/n).read_bytes()).hexdigest()}
f=read('failure.json')
assert "KeyError: 'installed_build_equivalence_verified'" in f['traceback']
checks=f['checks_before_failure'];assert len(checks)>=29
assert checks[-1]=='no numerical import attempted by metadata review'
rows=[line.split('|') for line in (P/'terminal_accounting.stdout.raw').read_text().splitlines() if line.strip()]
job=[x for x in rows if x[0]=='6782778'];assert len(job)==1 and job[0][1]=='RUNNING'
tests=read('original_tests.json.snapshot');renderer=read('original_renderer.json.snapshot');binding=read('resolved_device_binding.json.snapshot')
assert len(tests['collected_nodeids'])==48 and len(tests['reports'])==144 and all(x['outcome']=='passed' for x in tests['reports'])
assert renderer['rooms']==33 and renderer['cases']==132 and renderer['synthetic_pixel_cases']==11
assert binding['physical_gres_mapping_verified'] is False and binding['kernel_device_enforcement_verified'] is False and binding['installed_binary_matches_upstream_verified'] is False
assert not (P/'target_qualification_result.json.snapshot').exists() and not (P/'failure.json.snapshot').exists()
value={'schema_version':1,'kind':'full_original_qualification_current_stage_assessment','passed':True,'at_utc':datetime.now(timezone.utc).isoformat(),'finalizer_job_id':'6781231','finalizer_step_id':os.environ['SLURM_STEP_ID'],'original_inspection_job_id':'6781231','original_inspection_step_id':f['audit_step_id'],'actual_target_job_id':'6782778','actual_state_at_single_inspection':'RUNNING','actual_exit_field_at_single_inspection':job[0][2],'target_result_present_at_single_inspection':False,'failure_present_at_single_inspection':False,'full_gate_reader_passed':False,'actual_48_tests_and_144_phases_passed':True,'actual_renderer':renderer,'completed_checks':checks,'source_and_backend_and_assets_validated_from_exact_staged_reader':True,'actual_source':'d28e6735e79788f618acb0ebfde8609db1f776af','actual_helper':'e5afbf6b51a3d1666b46e71c0631a45d47080cd8','physical_identity':{k:binding[k] for k in ['gpu_uuid','pci_address','physical_minor']},'physical_gres_mapping_verified':False,'kernel_device_enforcement_verified':False,'installed_binary_matches_upstream_verified':False,'controlled_full_reference_work':'unknown while dedicated completed result absent; no zero inference','test_fixture_work':'actual test/renderer work is unquantified; no whole-job transition total claimed','saved_artifacts':{n:ref(n) for n in sorted(x.name for x in P.iterdir() if x.is_file() and (x.name.endswith('.snapshot') or x.name.startswith('terminal_accounting.')))},'first_receipt_wrapper_failure':ref('failure.json'),'first_wrapper_source':ref('review_actual_qualification.py'),'corrected_terminal_wrapper_source':ref('review_actual_qualification_corrected.py'),'no_target_poll_or_scheduler_query_in_finalizer':True,'scientific_credit':0,'case_credit':0,'idea_credit':0,'final_training_credit':0,'site_qualification_credit':0,'admission_granted':False,'worker_admitted':False,'gpu_actions':0,'network_actions':0,'numerical_imports':0}
out=P/'assessment_finalized.json'
with out.open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
print(json.dumps({'receipt':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'inspection_step':f['audit_step_id'],'finalizer_step':os.environ['SLURM_STEP_ID'],'state':job[0][1],'checks':len(checks)}))
