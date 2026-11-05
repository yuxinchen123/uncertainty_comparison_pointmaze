"""Bounded actual-schema projection tests; no current unit, raw replay or scientific admission."""
import ast
import copy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

SCRIPTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SCRIPTS))
import direct_common_timing_controller as control
RUN=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1')
P=RUN/'remote_setup/2026-11-05-05-49_common-timing_cache-room-backgrounds-true_render-final-two-only-true_h100-nvl_serval02-card-1_copies-1_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_paired-rounds-7_updates-per-arm-10_seed-start-1'
U=RUN/'data'/P.name

class ActualTimingPredecessorSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Read/hash the actual53MiB result once, then discard unused payload metadata."""
        cls.review=control.document(P/'independent_actual_common_timing_review.json')
        cls.plan=control.document(U/'measurement_plan.json')
        raw=control.ordinary_bytes(U/'result.json',bound=64*1024*1024)
        cls.actual_result_reference={'path':str(U/'result.json'),'sha256':control.digest(raw)}
        if cls.actual_result_reference!=cls.review['result']:raise ValueError('actual accepted result changed')
        full=json.loads(raw);del raw
        keys=('schema_version','decision','measurement_valid','measurement_mode','correctness_passed','correctness_attempted',
            'implementation_distinct','contaminated','source_commit','copies','copy_seeds','reference','candidate','helpers',
            'runtime_environment','hardware_class','hostname','paired_rounds','protocol')
        cls.result={key:full[key] for key in keys}
        row_keys=('iterations','copies','environments_per_copy','rollout_steps','training_transitions','seconds','report_origin_iteration')
        cls.result['rounds']=[{'round':pair['round'],'order':pair['order'],
            **{arm:{key:pair[arm][key] for key in row_keys} for arm in ('reference','candidate')}} for pair in full['rounds']]
        cls.actual_completed_field_present='completed' in full
        del full
        cls.wait=control.document(U/'direct_process_wait_return.json')
        cls.spawn=control.document(P/'root_actual_controller_spawn.json')
        cls.exit=control.document(cls.review['exit_observation']['path'])
        cls.consumer=control.document(cls.review['unchanged_staged_consumer']['path'])
        cls.tree=ast.parse(Path(control.__file__).read_bytes())
        serial=next(n for n in cls.tree.body if isinstance(n,ast.FunctionDef) and n.name=='check_serial_predecessor_inputs')
        selected=next(n for n in serial.body if isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
            and ast.unparse(n.test)=="request['serial_order'] == 1")
        cls.branch=ast.Module(body=selected.orelse,type_ignores=[])
        helper=U/'code/direct_common_confirmation.py';helper_raw=helper.read_bytes()
        if hashlib.sha256(helper_raw).hexdigest()!=cls.plan['helpers']['files']['10_jax_exploration_platform/scripts/atari_pixel_campaign/direct_common_confirmation.py']:
            raise ValueError('the unchanged seven-by-ten validator changed')
        helper_tree=ast.parse(helper_raw)
        predicate=next(n for n in helper_tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_pair_protocol')
        namespace={'require':control.require,'math':math};exec(compile(ast.Module(body=[predicate],type_ignores=[]),'exact_staged_pair_protocol','exec'),namespace)
        cls.pair_validator=staticmethod(namespace['validate_pair_protocol'])

    def fixture(self):
        request=control.document(P/'preparation_request.json')
        request['serial_order']=2;request['copy_seeds']=[1,2]
        for key in ('configuration','reference_configuration'):request[key]['learner']['n_copies']=2
        inputs={'plan':self.review['plan'],'result':self.review['result'],'terminal':self.review['terminal_process'],
            'timing_review':control.receipt(P/'independent_actual_common_timing_review.json')}
        review=copy.deepcopy(self.review);old_exit=copy.deepcopy(self.exit);consumer=copy.deepcopy(self.consumer)
        docs={inputs['timing_review']['path']:review,review['exit_observation']['path']:old_exit,
            review['unchanged_staged_consumer']['path']:consumer,'/synthetic-current-unit/measurement_plan.json':{'helpers':self.plan['helpers']}}
        expected={'controller':(self.spawn['actual_pid'],self.spawn['actual_birth_counter']),
            'numerical_child':(self.wait['pid'],self.wait['process_start_time'])}
        namespace={'require':control.require,'document':lambda name:docs[str(name)],'receipt':control.receipt,'json':json,'datetime':datetime,
            'SOURCE_COMMIT':control.SOURCE_COMMIT,'GPU_UUID':control.GPU_UUID,'HOST':control.HOST,'UID':control.UID,
            'UNIT':Path('/synthetic-current-unit'),'previous':U,'previous_packet':P,'prior_plan':self.plan,
            'prior_result':copy.deepcopy(self.result),'request':request,'inputs':inputs,'expected':expected,'wait':self.wait,
            'timing_support':lambda plan,unit,name:SimpleNamespace(validate_pair_protocol=self.pair_validator)}
        return namespace,review,old_exit,consumer

    def run_branch(self, namespace):exec(compile(self.branch,'exact_corrected_serial_timing_branch','exec'),namespace)

    def test_genuine_actual_schema_without_completed_is_accepted(self):
        self.assertFalse(self.actual_completed_field_present)
        namespace,_,_,_=self.fixture();self.run_branch(namespace)
        self.assertEqual(namespace['prior_result']['measurement_mode'],'confirmation')
        self.assertEqual(namespace['prior_result']['decision'],'accepted')

    def test_partial_incomplete_or_unaccepted_result_is_refused(self):
        for key,value in (('measurement_valid',False),('measurement_valid',1),('completed',False),('decision','inconclusive'),
                ('decision','rejected'),('contaminated',True),('correctness_passed',False),('schema_version',True),('paired_rounds',True)):
            ns,_,_,_=self.fixture();ns['prior_result'][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):self.run_branch(ns)

    def test_foreign_source_helper_seed_and_configuration_are_refused(self):
        for key,value in (('source_commit','f'*40),('copy_seeds',[2]),('copy_seeds',[True]),('copies',True),
                ('helpers',{}),('candidate',{}),('reference',{}),('runtime_environment',{}),('hardware_class','other')):
            ns,_,_,_=self.fixture();ns['prior_result'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.run_branch(ns)

    def test_seven_alternating_full_ten_update_rounds_are_required(self):
        for variant in ('partial','order','iterations','boolean','origin','protocol'):
            ns,_,_,_=self.fixture();result=ns['prior_result']
            if variant=='partial':result['rounds'].pop()
            elif variant=='order':result['rounds'][0]['order']=['candidate','reference']
            elif variant=='iterations':result['rounds'][0]['reference']['iterations']=9
            elif variant=='boolean':result['rounds'][0]['reference']['iterations']=True
            elif variant=='origin':result['rounds'][1]['candidate']['report_origin_iteration']+=1
            else:result['protocol']['actual_updates_per_round']=True
            with self.subTest(variant=variant),self.assertRaises(ValueError):self.run_branch(ns)

    def test_genuine_review_and_unchanged_consumer_joins_are_required(self):
        for key,value in (('approved',False),('decision','inconclusive'),('copy_seeds',[2]),('source_commit','f'*40),
                ('helper_commit','f'*40),('normal_actual_child_returncode',False),('actual_controller_and_numerical_child_exits_verified',False),
                ('unresolved_anomalies',['unresolved']),('reviewed_at_seconds',float('inf')),('reviewed_at_seconds',True)):
            ns,review,_,_=self.fixture();review[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.run_branch(ns)
        ns,_,_,consumer=self.fixture();consumer['passed']=False
        with self.assertRaises(ValueError):self.run_branch(ns)

    def test_actual_accepted_exit_evidence_is_separate_from_new_observation(self):
        for variant in ('foreign','present','partial','wrongbirth','early'):
            ns,_,old_exit,_=self.fixture()
            if variant=='foreign':old_exit['uid']=0
            elif variant=='present':old_exit['observations'][0]['proc_entry_absent']=False
            elif variant=='partial':old_exit['observations'].pop()
            elif variant=='wrongbirth':old_exit['observations'][0]['expected_birth_counter']='1'
            else:old_exit['observed_at_utc']='2020-01-01T00:00:00+00:00'
            with self.subTest(variant=variant),self.assertRaises(ValueError):self.run_branch(ns)

    def test_other_functions_and_original_correctness_branch_are_preserved(self):
        baseline=SCRIPTS.parents[2]/'.review_metadata/timing_predecessor_schema/controller_published_d225.py'
        old=ast.parse(baseline.read_bytes());functions=lambda t:{n.name:n for n in t.body if isinstance(n,ast.FunctionDef)}
        left,right=functions(old),functions(self.tree)
        self.assertEqual(set(left),set(right))
        for name in left:
            if name!='check_serial_predecessor_inputs':self.assertEqual(ast.dump(left[name],include_attributes=False),ast.dump(right[name],include_attributes=False),name)
        pick=lambda n:next(v for v in n.body if isinstance(v,ast.If) and ast.unparse(v.test)=="request['serial_order'] == 1")
        self.assertEqual(ast.dump(ast.Module(body=pick(left['check_serial_predecessor_inputs']).body,type_ignores=[]),include_attributes=False),
            ast.dump(ast.Module(body=pick(right['check_serial_predecessor_inputs']).body,type_ignores=[]),include_attributes=False))
        self.assertEqual(control.ordinary_bytes.__defaults__,(16*1024*1024,))

if __name__=='__main__':unittest.main()
