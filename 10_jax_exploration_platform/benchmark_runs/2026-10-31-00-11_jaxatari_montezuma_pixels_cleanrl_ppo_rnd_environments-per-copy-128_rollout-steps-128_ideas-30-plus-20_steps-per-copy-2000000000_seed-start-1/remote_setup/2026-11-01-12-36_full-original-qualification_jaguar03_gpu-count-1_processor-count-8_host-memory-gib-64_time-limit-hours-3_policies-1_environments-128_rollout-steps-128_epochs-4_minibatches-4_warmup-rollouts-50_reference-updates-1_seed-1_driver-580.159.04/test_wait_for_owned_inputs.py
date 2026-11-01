"""Test the real release-input barrier using synthetic launch records, never an actual job."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

PACKET=Path(__file__).resolve().parent
CONTEXT=json.loads((PACKET/'preparation_context.json').read_text())
SPEC=importlib.util.spec_from_file_location('qualification_startup_gate',PACKET/'wait_for_owned_inputs.py')
GATE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(GATE)

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')

def receipt(path):return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

class BarrierTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='qualification-startup-metadata-')
        self.campaign=Path(self.tmp.name)
        self.unit=self.campaign/'data/full-original-synthetic-qualification'
        self.unit.mkdir(parents=True)
        original=Path(CONTEXT['unit'])
        self.plan=json.loads((original/'measurement_plan.json').read_text())
        self.plan['campaign_run_dir']=str(self.campaign)
        self.plan['unit_id']=self.unit.name
        self.plan['configured_isolation']['owned_inputs_path']=str(self.unit/'owned_allocation_inputs.json')
        shutil.copytree(original/'code',self.unit/'code')
        write(self.unit/'measurement_plan.json',self.plan)
        self.sha=receipt(self.unit/'measurement_plan.json')['sha256']
        write(self.unit/'measurement_prepared.json',{'plan_sha256':self.sha})
        self.job='123456789'
        self.row=[self.job,self.plan['kind'],CONTEXT['helper_commit'],str(self.campaign/'remote_setup'/self.unit.name/'job.sbatch')]
        self.campaign.joinpath('slurm').mkdir()
        ledger=self.campaign/'slurm/submitted_jobids_uva.txt';ledger.write_text(self.job+'\n')
        rows=self.campaign/'slurm/submissions.tsv';rows.write_text('\t'.join(self.row)+'\n')
        ledgersnapshot=self.unit/'configured_isolation_owned_job_ids.txt';shutil.copyfile(ledger,ledgersnapshot)
        rowssnapshot=self.unit/'configured_isolation_owned_submissions.tsv';shutil.copyfile(rows,rowssnapshot)
        common={'schema_version':1,'job_id':self.job,'cluster':'uva','plan_sha256':self.sha,'data_unit':str(self.unit),'returncode':0}
        sub=self.unit/'submission.json';write(sub,{**common,'kind':'owned_uva_configured_isolation_submission','held_command':['sbatch','--hold','--no-requeue',self.row[-1]],'submission_row':self.row})
        rel=self.unit/'release.json';write(rel,{**common,'kind':'owned_uva_configured_isolation_release','command':['scontrol','release',self.job],'submission_receipt_sha256':receipt(sub)['sha256']})
        self.pre={'schema_version':1,'kind':'uva_configured_isolation_pre_release_inputs','job_id':self.job,'plan':receipt(self.unit/'measurement_plan.json'),'ledger':{'snapshot':receipt(ledgersnapshot),'source_path':str(ledger)},'submissions':{'snapshot':receipt(rowssnapshot),'source_path':str(rows)},'submission':receipt(sub),'submission_row':self.row}
        self.final={**self.pre,'kind':'uva_configured_isolation_owned_inputs','release':receipt(rel)}
        write(self.unit/'configured_isolation_pre_release_inputs.json',self.pre)
        self.finalpath=self.unit/'owned_allocation_inputs.json'
        write(self.finalpath,self.final)
        self.calls=[]
    def tearDown(self):self.tmp.cleanup()
    def validate(self,plan):self.calls.append(plan);return self.final
    def run_gate(self,**kwargs):return GATE.wait_for_inputs(self.unit,self.sha,CONTEXT['helper_commit'],self.job,validator=self.validate,**kwargs)
    def test_immediate_complete(self):
        value=self.run_gate();self.assertEqual(value['absence_polls'],0);self.assertEqual(len(self.calls),1)
    def test_delayed_complete(self):
        self.finalpath.unlink();clock=[0.0]
        def pause(seconds):
            clock[0]+=seconds
            if clock[0]==3:write(self.finalpath,self.final)
        value=self.run_gate(clock=lambda:clock[0],pause=pause)
        self.assertEqual(value['waited_seconds'],3);self.assertEqual(value['absence_polls'],3)
    def test_absence_timeout_exact_180(self):
        self.finalpath.unlink();clock=[0.0]
        def pause(seconds):clock[0]+=seconds
        with self.assertRaises(TimeoutError):self.run_gate(clock=lambda:clock[0],pause=pause)
        self.assertEqual(clock[0],180);self.assertEqual(self.calls,[])
    def test_present_partial_is_not_retried(self):
        self.finalpath.write_text('{')
        with self.assertRaises(json.JSONDecodeError):self.run_gate(pause=lambda value:self.fail('invalid existing evidence was retried'))
    def test_changed_actual_job(self):
        with self.assertRaises(ValueError):GATE.wait_for_inputs(self.unit,self.sha,CONTEXT['helper_commit'],'987654321',validator=self.validate)
    def test_changed_pre_release_binding(self):
        self.final['submission_row']=['changed'];write(self.finalpath,self.final)
        with self.assertRaises(ValueError):self.run_gate()
    def test_changed_plan(self):
        self.plan['copy_seeds']=[2];write(self.unit/'measurement_plan.json',self.plan)
        with self.assertRaises(ValueError):self.run_gate()
    def test_changed_staged_helper(self):
        (self.unit/'code/configured_measurement.py').write_text('# changed\n')
        with self.assertRaises(ValueError):self.run_gate()
    def test_boolean_schema_refused(self):
        self.pre['schema_version']=True;write(self.unit/'configured_isolation_pre_release_inputs.json',self.pre)
        with self.assertRaises(ValueError):self.run_gate()
    def test_final_symlink_refused(self):
        other=self.unit/'other.json';self.finalpath.rename(other);self.finalpath.symlink_to(other)
        with self.assertRaises(ValueError):self.run_gate()
    def test_missing_pre_release_is_not_retried(self):
        (self.unit/'configured_isolation_pre_release_inputs.json').unlink()
        with self.assertRaises(FileNotFoundError):self.run_gate(pause=lambda value:self.fail('present invalid launch was retried'))
    def test_real_core_cold_stdlib_startup(self):
        environment={**os.environ,'SLURM_JOB_ID':self.job}
        command=['/usr/bin/python3','-I','-S','-B',str(PACKET/'wait_for_owned_inputs.py'),'--unit',str(self.unit),'--plan-sha256',self.sha,'--helper-commit',CONTEXT['helper_commit']]
        run=subprocess.run(command,capture_output=True,text=True,env=environment,timeout=10)
        self.assertEqual(run.returncode,0,run.stderr)
        value=json.loads((self.unit/'owned_inputs_wait_completed.json').read_text())
        self.assertEqual(value['interpreter_flags'],{'isolated':1,'no_site':1,'dont_write_bytecode':True})
        self.assertEqual(value['observed_actions'],{'numerical_import_attempts':[],'forbidden_operation_attempts':[]})
        self.assertFalse(any(name.split('.')[0] in GATE.NUMERICAL_ROOTS for name in value['imported_module_names']))
    def test_real_core_changed_release_receipt_refused(self):
        path=Path(self.final['release']['path']);document=json.loads(path.read_text());document['job_id']='987654321';write(path,document)
        self.final['release']=receipt(path);write(self.finalpath,self.final)
        command=['/usr/bin/python3','-I','-S','-B',str(PACKET/'wait_for_owned_inputs.py'),'--unit',str(self.unit),'--plan-sha256',self.sha,'--helper-commit',CONTEXT['helper_commit']]
        run=subprocess.run(command,capture_output=True,text=True,env={**os.environ,'SLURM_JOB_ID':self.job},timeout=10)
        self.assertNotEqual(run.returncode,0)
        value=json.loads((self.unit/'owned_inputs_wait_failure.json').read_text())
        self.assertIn('launch receipt identity changed',value['error'])

if __name__=='__main__':
    assert os.environ.get('SLURM_JOB_ID')=='6781231' and os.environ.get('SLURM_STEP_ID','').isdigit()
    source=PACKET/'wait_for_owned_inputs.py';before=receipt(source)
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(BarrierTests)
    started=time.time();result=unittest.TextTestRunner(verbosity=2).run(suite)
    value={'schema_version':1,'kind':'full_original_qualification_startup_barrier_metadata_tests','job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],'source_before':before,'source_after':receipt(source),'test_source':receipt(Path(__file__)),'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'seconds':time.time()-started,'synthetic_launch_only':True,'real_gpu_operations':0,'real_submissions':0,'real_release_actions':0,'returncode':0 if result.wasSuccessful() else 1}
    with (PACKET/'barrier_tests_receipt.json').open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
    raise SystemExit(value['returncode'])
