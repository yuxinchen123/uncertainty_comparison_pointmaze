"""Independently check narrow direct display ownership/terminal boundaries only."""
import ast
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parents[2];M=Path(__file__).resolve().parent
SCRIPTS=ROOT/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
START=time.monotonic();STEP=os.environ['SLURM_STEP_ID']
assert os.environ['SLURM_JOB_ID']=='6783003' and STEP.isdecimal() and os.getresuid()==(2618919,)*3
assert Path('/proc/self/cgroup').read_text().splitlines()==['0::/system.slice/slurmstepd.scope/job_6783003/step_'+STEP+'/user/task_0']
F=M/'frozen_source_for_independent_review.json'
assert hashlib.sha256(F.read_bytes()).hexdigest()=='852de048d6e76161d0832f39ab21805dd5091a3ef26d57c2667846048d5e71b2'
FREEZE=json.loads(F.read_bytes())
def frozen():
    return all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==v['sha256'] and (ROOT/n).stat().st_size==v['size']
        and ((ROOT/n).stat().st_mode&0o777)==v['mode'] for n,v in FREEZE['files'].items())
assert frozen()
def audit(event,args):
    if event=='import' and args[0].split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','scipy','cupy','jaxtari','exploration_platform'}:raise RuntimeError('numerical import refused')
    if event in {'subprocess.Popen','ctypes.dlopen','ctypes.dlsym','socket.connect','socket.bind','socket.getaddrinfo','os.system'}:raise RuntimeError('native/process/network operation refused')
    if event=='open' and isinstance(args[0],(str,bytes)):
        path=os.fsdecode(args[0])
        if path.startswith(('/dev/','/tmp/sl5nw_serval02_card1.lock')) or '/direct_history/content/' in path:raise RuntimeError('device/standinglock/payload refused')
sys.addaudithook(audit)
spec=importlib.util.spec_from_file_location('independent_monitor_direct_fixture',SCRIPTS/'tests/test_monitor_direct_unit_display.py')
author=importlib.util.module_from_spec(spec);spec.loader.exec_module(author)
monitor=author.monitor
OBS=[]
class ExtraDisplayBoundaries(unittest.TestCase):
    def fixture(self):
        f=author.DirectDisplayTests('test_prepared_without_progress_is_visible_once_and_not_running');f.setUp();self.addCleanup(f.doCleanups)
        return f
    def active(self,f):
        f.register();f.hardware();f.write(f.unit/'progress.json',{'state':'running','round':1,'total_rounds':7,'variant':'reference'})
    def test_current_registration_removed_or_substituted_cannot_be_running(self):
        for mode in ('missing','other_birth','other_source','duplicate'):
            with self.subTest(mode=mode):
                f=self.fixture();self.active(f)
                if mode=='missing':rows=[]
                elif mode=='duplicate':rows=[dict(f.row),dict(f.row)]
                else:
                    row=dict(f.row);row['start_time' if mode=='other_birth' else 'source_commit']='other';rows=[row]
                f.write(f.run/'processes.json',rows)
                row=monitor.unit_rows(f.run)[0]
                OBS.append({'case':'current_campaign_registration_mismatch','mode':mode,'state':row['state'],'progress':row['progress']})
                self.assertNotEqual(row['state'],'running','a missing/substituted/duplicate current campaign registration was promoted to RUNNING')
    def test_failure_and_completed_result_keep_priority(self):
        f=self.fixture();self.active(f)
        f.write(f.unit/'progress.json',{'state':'completed','rounds':7})
        f.write(f.unit/'result.json',{'case_completed':True,'paired_rounds':7,'decision':'accepted','median_speedup':1.1})
        row=monitor.unit_rows(f.run)[0];self.assertEqual(row['state'],'completed');self.assertIn('7/7 pairs',row['progress'])
        f.write(f.unit/'failure.json',{'error':'preserved actual unit failure'})
        row=monitor.unit_rows(f.run)[0];self.assertEqual(row['state'],'failed');self.assertIn('preserved actual unit failure',row['progress'])
    def test_wrong_registered_source_unit_UUID_and_plan_are_unknown(self):
        for field,value in [('source_commit','changed'),('helper_commit','changed'),('unit','other'),('gpu_uuid','GPU-other')]:
            with self.subTest(field=field):
                f=self.fixture();self.active(f);f.row[field]=value;f.register()
                self.assertEqual(monitor.unit_rows(f.run)[0]['state'],'unknown')
    def test_other_function_hashes_and_no_scientific_reader_authority(self):
        declared=json.loads((M/'source_function_and_closure_check_455.json').read_bytes())
        functions={n.name:n for n in ast.parse((SCRIPTS/'monitor.py').read_bytes()).body if isinstance(n,ast.FunctionDef)}
        for name,expected in declared['all_existing_other_function_asts_unchanged'].items():
            self.assertEqual(hashlib.sha256(ast.dump(functions[name],include_attributes=False).encode()).hexdigest(),expected,name)
        self.assertFalse(any(name in sys.modules for name in ('accepted_evidence_snapshot','direct_measurement','measure_selected')))
suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(author.DirectDisplayTests),unittest.defaultTestLoader.loadTestsFromTestCase(ExtraDisplayBoundaries)])
capture=io.StringIO();result=unittest.TextTestRunner(stream=capture,verbosity=2).run(suite)
# Genuine display-only reporting index check; do not import any raw scientific consumer.
sys.path.insert(0,str(SCRIPTS));import controller
campaign=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1')
index=json.loads((campaign/'next_actions.json').read_bytes())['post_initial_reporting_evidence']
assert sum(len(index.get(name,[])) for name in ('common_correctness','common_confirmation','kernel_profiles','packing'))==6
reporting=controller.post_initial_reporting_requirements(campaign,index['source_commit'],[1,2],1)
assert reporting[0]['state']=='reviewed_evidence_present'
assert reporting[1]['missing']==[{'hardware_class':'h100','copies':1},{'hardware_class':'h100','copies':2}]
assert all(row['scientific_credit']==0 and row['admission_authority'] is False for row in reporting)
assert frozen()
receipt={'schema_version':1,'kind':'independent_monitor_direct_unit_display_review','reviewer':'/root/implement_direct_h100_profiles/review_direct_profile_source',
    'approved':result.wasSuccessful(),'passed':result.wasSuccessful(),'checks':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
    'job_id':'6783003','step_id':STEP,'elapsed_seconds':time.monotonic()-START,
    'source_freeze':{'path':str(F),'sha256':hashlib.sha256(F.read_bytes()).hexdigest()},'all17frozenbytes_modes_unchanged':True,
    'observations':OBS,'all_other_existing_monitor_function_ASTs_unchanged':True,
    'genuine_six_metadata_reporting_joins':reporting,'scientific_credit':0,'admission_authority':False,
    'monitor_deployed_or_stopped':False,'NUM_GPU_native_library_network_tool_operations':0,'broad1540suite_repeated':False}
for suffix,raw in [('.json',(json.dumps(receipt,indent=2,sort_keys=True)+'\n').encode()),('.stdout.raw',capture.getvalue().encode())]:
    with (M/('independent_display_'+STEP+suffix)).open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
print(json.dumps({k:v for k,v in receipt.items() if k!='genuine_six_metadata_reporting_joins'},sort_keys=True))
print(capture.getvalue(),file=sys.stderr)
raise SystemExit(0 if result.wasSuccessful() else 1)

