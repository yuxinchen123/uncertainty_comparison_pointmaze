"""Metadata display regressions using actual small records and private synthetic ledgers only."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

S=Path(__file__).resolve().parents[1];sys.path.insert(0,str(S))
import monitor
R=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1')
N='2026-11-05-06-39_native-process-control_jaguar03_python-3.12_gpu-count-1_processor-count-1_host-memory-gib-2_total-seconds-300_work-seconds-240_cleanup-seconds-60_environment-transitions-0_training-updates-0'
P=R/'remote_setup'/N;U=R/'data'/N

class NativeComponentMonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.run=Path(self.temp.name)/'clearly_synthetic_campaign';self.unit=self.run/'data'/N;self.packet=self.run/'remote_setup'/N
        self.unit.mkdir(parents=True);self.packet.mkdir(parents=True);(self.run/'slurm').mkdir()
        for name in ('measurement_plan.json','result.json'):
            self.write_raw(self.unit/name,(U/name).read_bytes().replace(str(R).encode(),str(self.run).encode()))
        self.plan=self.read(self.unit/'measurement_plan.json');plan_ref=self.ref(self.unit/'measurement_plan.json')
        sub=json.loads((P/'submission_receipt.json').read_bytes().replace(str(R).encode(),str(self.run).encode()))
        sub['plan_sha256']=plan_ref['sha256'];sub['synthetic_fixture']=True;self.job=sub['job_id'];self.write(self.packet/'submission_receipt.json',sub)
        release=json.loads((P/'release_receipt.json').read_bytes().replace(str(R).encode(),str(self.run).encode()))
        release['plan_sha256']=plan_ref['sha256'];release['synthetic_fixture']=True;self.write(self.packet/'release_receipt.json',release)
        self.write(self.unit/'owned_launch_inputs.json',{'schema_version':1,'job_id':self.job,'plan':plan_ref,'submission_row':sub['submission_row'],
            'submission':self.ref(self.packet/'submission_receipt.json'),'release':self.ref(self.packet/'release_receipt.json')})
        result=self.read(self.unit/'result.json');result['plan']=plan_ref;self.write(self.unit/'result.json',result)
        for name in ('root_actual_terminal_accounting_return.json','root_actual_terminal_accounting_stdout.raw'):
            self.write_raw(self.packet/name,(P/name).read_bytes())
        for name in ('submitted_jobids.txt','submitted_jobids_uva.txt'):(self.run/'slurm'/name).write_text(self.job+'\n')
        (self.run/'slurm/submissions.tsv').write_text('\t'.join(sub['submission_row'])+'\n')
    def read(self,p):return json.loads(p.read_bytes())
    def ref(self,p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    def write_raw(self,p,v):p.write_bytes(v)
    def write(self,p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
    def row(self,jobs=()):return monitor.configured_native_component_display(self.run,self.unit,self.plan,jobs)

    def test_actual_small_metadata_component_is_visible_with_zero_authority(self):
        with patch.object(monitor.subprocess,'run',side_effect=AssertionError('no scheduler/network action')):
            row=monitor.configured_native_component_display(R,U,json.loads((U/'measurement_plan.json').read_bytes()),())
        self.assertEqual(row['slurm_job_id'],'6825842');self.assertEqual(row['state'],'completed')
        self.assertEqual(row['scientific_credit'],0);self.assertEqual(row['profile_credit'],0);self.assertFalse(row['admission_authority'])
        self.assertTrue(row['component_metadata_display_only'])

    def test_missing_progress_discovery_regression_and_no_duplicate_after_progress(self):
        self.assertFalse((self.unit/'progress.json').exists())
        rows=monitor.unit_rows(self.run,jobs=())
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['slurm_job_id'],self.job);self.assertIn(self.job,json.dumps(rows))
        self.write(self.unit/'progress.json',{'state':'completed'})
        self.assertEqual(len(monitor.unit_rows(self.run,jobs=())),1)

    def test_original_discovery_omits_same_prepared_result_shape(self):
        baseline=S.parents[2]/'.review_metadata/monitor_configured_native_component/monitor_before_native_display.py'
        spec=importlib.util.spec_from_file_location('preserved_monitor_without_component',baseline);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        self.assertEqual(old.unit_rows(self.run,jobs=()),[])
        self.assertEqual(self.row()['state'],'completed')

    def test_prepared_only_is_planned_but_partial_owned_ledger_is_unknown(self):
        for p in (self.unit/'owned_launch_inputs.json',self.unit/'result.json',self.packet/'submission_receipt.json'):p.unlink()
        with self.assertRaisesRegex(ValueError,'ledger row'):self.row()
        for name in ('submitted_jobids.txt','submitted_jobids_uva.txt','submissions.tsv'):(self.run/'slurm'/name).write_text('')
        self.assertEqual(self.row()['state'],'planned')

    def test_live_running_or_held_does_not_infer_terminal_completion(self):
        self.assertEqual(self.row([[self.job,'RUNNING','jaguar03','gpu:1','None']])['state'],'running')
        self.assertEqual(self.row([[self.job,'PENDING','','gpu:1','JobHeldUser']])['state'],'held')
        self.assertEqual(self.row([[self.job,'PENDING','','gpu:1','Priority']])['state'],'waiting')

    def test_result_without_accounting_cannot_be_reported_finished(self):
        (self.packet/'root_actual_terminal_accounting_return.json').unlink()
        row=self.row();self.assertEqual(row['state'],'unknown');self.assertIn('accounting unavailable',row['progress'])

    def test_current_ledger_missing_duplicate_or_foreign_source_refuses(self):
        path=self.run/'slurm/submitted_jobids_uva.txt';saved=path.read_text()
        for raw in ('',saved+saved):
            path.write_text(raw)
            with self.assertRaisesRegex(ValueError,'uniquely'):self.row()
        path.write_text(saved);p=self.run/'slurm/submissions.tsv';p.write_text(p.read_text().replace(self.plan['helpers']['source_commit'],'f'*40))
        with self.assertRaisesRegex(ValueError,'submission row'):self.row()
        self.assertEqual(monitor.unit_rows(self.run,jobs=())[0]['state'],'unknown')

    def test_failure_priority_and_mismatched_component_results_are_honest(self):
        self.write(self.unit/'failure.json',{'error':'clearly synthetic retained failure'})
        self.assertEqual(self.row()['state'],'failed');(self.unit/'failure.json').unlink()
        result=self.read(self.unit/'result.json');result['source']={};self.write(self.unit/'result.json',result)
        self.assertEqual(self.row()['state'],'unknown')

    def test_foreign_duplicate_or_partial_cached_accounting_refuses(self):
        path=self.packet/'root_actual_terminal_accounting_stdout.raw';raw=path.read_text();lines=raw.splitlines()
        for changed in (raw+lines[1]+'\n',raw.replace(self.job+'.0','9999999.0'),'\n'.join(lines[:2])+'\n'):
            path.write_text(changed)
            with self.assertRaises(ValueError):self.row()
        path.write_text(raw.replace('COMPLETED','RUNNING'))
        self.assertEqual(self.row()['state'],'unknown')
        path.write_text(raw.replace('COMPLETED','FAILED').replace('0:0','1:0'))
        self.assertEqual(self.row()['state'],'failed')

    def test_future_owned_kind_without_progress_is_visible_unknown_without_result_read(self):
        plan=copy.deepcopy(self.plan);plan['kind']='explicitly_synthetic_future_unsupported_kind';self.write(self.unit/'measurement_plan.json',plan)
        path=self.run/'slurm/submissions.tsv';path.write_text(path.read_text().replace(self.plan['kind'],plan['kind']))
        (self.unit/'result.json').write_text('This unsupported result must never be parsed.')
        row=monitor.unit_rows(self.run,jobs=())[0]
        self.assertEqual(row['state'],'unknown');self.assertIn('unsupported',row['progress']);self.assertEqual(row['slurm_job_id'],self.job)
        self.assertFalse(row['admission_authority']);self.assertEqual(row['scientific_credit'],0)

    def test_component_metadata_reads_are_bounded_and_require_exact_snapshot_size(self):
        original = monitor.os.fdopen
        sizes = []
        test = self
        class ObservedStream:
            def __init__(self, stream, add_one=False):self.stream=stream;self.add_one=add_one
            def __enter__(self):self.stream.__enter__();return self
            def __exit__(self,*args):return self.stream.__exit__(*args)
            def read(self,size=-1):
                sizes.append(size)
                test.assertEqual(size,1024*1024+1)
                value=self.stream.read(size)
                return value+b' ' if self.add_one else value
        with patch.object(monitor.os,'fdopen',side_effect=lambda *args,**kwargs:ObservedStream(original(*args,**kwargs))):
            self.assertEqual(self.row()['state'],'completed')
        self.assertTrue(sizes)
        with patch.object(monitor.os,'fdopen',side_effect=lambda *args,**kwargs:ObservedStream(original(*args,**kwargs),True)):
            with self.assertRaisesRegex(ValueError,'changed during its read'):self.row()

    def test_any_present_failure_marker_prevents_completed_display(self):
        for value in ({},None,False,[]):
            self.write(self.unit/'failure.json',value)
            with self.subTest(value=value):self.assertEqual(self.row()['state'],'failed')

    def test_other_existing_monitor_functions_and_scientific_readers_are_unchanged(self):
        baseline=S.parents[2]/'.review_metadata/monitor_configured_native_component/monitor_before_native_display.py'
        funcs=lambda path:{node.name:ast.dump(node,include_attributes=False) for node in ast.parse(path.read_bytes()).body if isinstance(node,ast.FunctionDef)}
        old,new=funcs(baseline),funcs(S/'monitor.py')
        for name in old:
            if name not in ('unit_rows','tick'):self.assertEqual(old[name],new[name],name)
        self.assertEqual(set(new)-set(old),{'configured_native_component_display','unsupported_owned_plan_display'})
        self.assertNotIn('import configured_native_process',(S/'monitor.py').read_text())
        self.assertNotIn('import native_process_evidence',(S/'monitor.py').read_text())

if __name__=='__main__':unittest.main()
