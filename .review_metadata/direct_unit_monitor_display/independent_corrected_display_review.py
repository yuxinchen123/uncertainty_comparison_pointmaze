"""Replay the prior current-ledger failure only, plus narrow typed ownership boundaries."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2];M=Path(__file__).resolve().parent;SCRIPTS=ROOT/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
START=time.monotonic();STEP=os.environ['SLURM_STEP_ID']
assert os.environ['SLURM_JOB_ID']=='6783003' and STEP.isdecimal() and os.getresuid()==(2618919,)*3
assert Path('/proc/self/cgroup').read_text().splitlines()==['0::/system.slice/slurmstepd.scope/job_6783003/step_'+STEP+'/user/task_0']
FP=M/'corrected_frozen_source_for_independent_review.json'
assert hashlib.sha256(FP.read_bytes()).hexdigest()=='28c032588fdef8ab362eb317a1dc4e5c05ea4bda525d028a5bfbb7d696eb4d44'
F=json.loads(FP.read_bytes())
def frozen():
    return all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==v['sha256'] and (ROOT/n).stat().st_size==v['size']
        and ((ROOT/n).stat().st_mode&0o777)==v['mode'] for n,v in F['files'].items())
assert len(F['files'])==50 and frozen()
def audit(event,args):
    if event=='import' and args[0].split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','scipy','cupy','jaxtari','exploration_platform'}:raise RuntimeError('numerical import refused')
    if event in {'subprocess.Popen','ctypes.dlopen','ctypes.dlsym','socket.connect','socket.bind','socket.getaddrinfo','os.system'}:raise RuntimeError('native/process/network operation refused')
    if event=='open' and isinstance(args[0],(str,bytes)):
        name=os.fsdecode(args[0])
        if name.startswith(('/dev/','/tmp/sl5nw_serval02_card1.lock')) or '/direct_history/content/' in name:raise RuntimeError('device/standinglock/payload refused')
sys.addaudithook(audit)
spec=importlib.util.spec_from_file_location('independent_corrected_monitor_fixture',SCRIPTS/'tests/test_monitor_direct_unit_display.py')
A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A);monitor=A.monitor
OBS=[]
class CorrectedCurrentLedgerBoundary(unittest.TestCase):
    def fixture(self):
        f=A.DirectDisplayTests('test_prepared_without_progress_is_visible_once_and_not_running');f.setUp();self.addCleanup(f.doCleanups)
        f.register();f.hardware();f.write(f.unit/'progress.json',{'state':'running','round':1,'total_rounds':7,'variant':'reference'})
        return f
    def test_exact456_three_negatives_and_matching_positive(self):
        for mode in ('missing','other_birth','other_source','matching'):
            with self.subTest(mode=mode):
                f=self.fixture()
                if mode=='missing':rows=[]
                else:
                    row=dict(f.row)
                    if mode!='matching':row['start_time' if mode=='other_birth' else 'source_commit']='other'
                    rows=[row]
                f.write(f.run/'processes.json',rows)
                displayed=monitor.unit_rows(f.run)[0]
                OBS.append({'case':'exact456_current_ledger','mode':mode,'state':displayed['state'],'progress':displayed['progress']})
                self.assertEqual(displayed['state'],'running' if mode=='matching' else 'unknown')
                self.assertEqual(displayed['scientific_credit'],0);self.assertIs(displayed['admission_authority'],False)
    def test_current_helper_UUID_unit_source_and_identity_duplicates_refused(self):
        for field,value in [('helper_commit','other'),('gpu_uuid','GPU-other'),('unit','other_unit'),('source_root','/other_root'),('pid',678)]:
            with self.subTest(field=field):
                f=self.fixture();row=dict(f.row);row[field]=value;f.write(f.run/'processes.json',[row])
                self.assertEqual(monitor.unit_rows(f.run)[0]['state'],'unknown')
        f=self.fixture();progress={'state':'running'};hardware=json.loads((f.unit/'hardware.json').read_bytes())
        for current in (None,{},'bad',[None],[],[dict(f.row),dict(f.row)]):
            with self.subTest(current_type=type(current).__name__),self.assertRaises(ValueError):
                monitor.direct_unit_display(f.run,f.unit,progress,hardware,current)
        alias=dict(f.row);alias['unit']='another_unit'
        with self.assertRaises(ValueError):monitor.direct_unit_display(f.run,f.unit,progress,hardware,[dict(f.row),alias])
        # Retain the existing outer duplicate-current-unit rejection.
        f.write(f.run/'processes.json',[dict(f.row),dict(f.row)])
        with self.assertRaisesRegex(ValueError,'duplicate process unit identity'):
            monitor.unit_rows(f.run)
    def test_current_ledger_is_parsed_exactly_once_and_bound_unlaunched_stays_planned(self):
        f=self.fixture();old=monitor.read_json;reads=[]
        def observed(path):
            if Path(path)==f.run/'processes.json':reads.append(str(path))
            return old(path)
        with patch.object(monitor,'read_json',side_effect=observed):
            self.assertEqual(monitor.unit_rows(f.run)[0]['state'],'running')
        self.assertEqual(len(reads),1)
        f.write(f.unit/'progress.json',{'state':'prepared_not_launched'})
        self.assertEqual(monitor.unit_rows(f.run)[0]['state'],'planned')
    def test_only_the_two_display_functions_changed_after_rejection(self):
        old=M/'independent_rejected_display_freeze_preserved/frozen/10_jax_exploration_platform/scripts/atari_pixel_campaign/monitor.py'
        previous=ast.parse(old.read_bytes());current=ast.parse((SCRIPTS/'monitor.py').read_bytes())
        refs={n.name:n for n in previous.body if isinstance(n,ast.FunctionDef)};changes=[]
        for i,n in enumerate(current.body):
            if isinstance(n,ast.FunctionDef) and n.name in ('direct_unit_display','unit_rows'):
                self.assertNotEqual(ast.dump(n,include_attributes=False),ast.dump(refs[n.name],include_attributes=False))
                current.body[i]=refs[n.name];changes.append(n.name)
        self.assertEqual(sorted(changes),['direct_unit_display','unit_rows'])
        self.assertEqual(ast.dump(current,include_attributes=False),ast.dump(previous,include_attributes=False))
        record=json.loads((M/'source_function_and_closure_check_455.json').read_bytes())
        current_functions={n.name:n for n in ast.parse((SCRIPTS/'monitor.py').read_bytes()).body if isinstance(n,ast.FunctionDef)}
        for name,h in record['all_existing_other_function_asts_unchanged'].items():
            self.assertEqual(hashlib.sha256(ast.dump(current_functions[name],include_attributes=False).encode()).hexdigest(),h)
suite=unittest.defaultTestLoader.loadTestsFromTestCase(CorrectedCurrentLedgerBoundary)
capture=io.StringIO();result=unittest.TextTestRunner(stream=capture,verbosity=2).run(suite)
assert frozen()
campaign=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1')
index=json.loads((campaign/'next_actions.json').read_bytes())['post_initial_reporting_evidence']
assert sum(len(index.get(n,[])) for n in ('common_correctness','common_confirmation','kernel_profiles','packing'))==6
receipt={'schema_version':1,'kind':'independent_corrected_monitor_current_ledger_display_review','reviewer':'/root/implement_direct_h100_profiles/review_direct_profile_source',
    'approved':result.wasSuccessful(),'passed':result.wasSuccessful(),'checks':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),
    'job_id':'6783003','step_id':STEP,'elapsed_seconds':time.monotonic()-START,'observations':OBS,
    'source_freeze':{'path':str(FP),'sha256':hashlib.sha256(FP.read_bytes()).hexdigest()},'all50frozenbytes_modes_unchanged':True,
    'source_sha256':hashlib.sha256((SCRIPTS/'monitor.py').read_bytes()).hexdigest(),
    'only_direct_unit_display_and_unit_rows_changed_after_rejection':True,'all_other_monitor_function_ASTs_unchanged':True,
    'current_campaign_ledger_parsed_once':True,'duplicate_outer_unit_refusal_retained':True,
    'genuine_six_reporting_entries_still_present':True,'full_reporting_join_not_repeated_prior456retained':True,
    'older_eight_display_method_suite_not_repeated':True,'scientific_credit':0,'admission_authority':False,
    'NUM_GPU_native_library_network_tool_operations':0,'monitor_deployed_or_stopped':False}
for suffix,raw in [('.json',(json.dumps(receipt,indent=2,sort_keys=True)+'\n').encode()),('.stdout.raw',capture.getvalue().encode())]:
    with (M/('independent_corrected_display_'+STEP+suffix)).open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
print(json.dumps(receipt,sort_keys=True));print(capture.getvalue(),file=sys.stderr)
raise SystemExit(0 if result.wasSuccessful() else 1)

