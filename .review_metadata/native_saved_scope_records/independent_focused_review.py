"""Review frozen scope-only saved schema using exact preserved actual small JSON."""
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

ROOT = Path(__file__).resolve().parents[2]
M = Path(__file__).resolve().parent
SCRIPTS = ROOT / '10_jax_exploration_platform/scripts/atari_pixel_campaign'
START = time.monotonic()
STEP = os.environ['SLURM_STEP_ID']
assert os.environ['SLURM_JOB_ID'] == '6783003' and STEP.isdecimal()
assert os.getresuid() == (2618919,) * 3
assert Path('/proc/self/cgroup').read_text().splitlines() == ['0::/system.slice/slurmstepd.scope/job_6783003/step_' + STEP + '/user/task_0']
FREEZE = M / 'source_freeze.json'
assert hashlib.sha256(FREEZE.read_bytes()).hexdigest() == 'ca0505be76f23f5f39429845c7ac8799b7579065ace0cca5ba3d7e3ea99fed69'
F = json.loads(FREEZE.read_bytes())
def frozen():
    return all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == value['sha256']
        for name, value in F['files'].items()) and all(hashlib.sha256((Path(name) if Path(name).is_absolute() else ROOT / name).read_bytes()).hexdigest() == value
        for name, value in F['protected_files'].items())
assert frozen()
FORBIDDEN = {'jax', 'jaxlib', 'numpy', 'torch', 'flax', 'optax', 'scipy', 'cupy', 'jaxtari', 'exploration_platform'}
def audit(event, args):
    if event == 'import' and (args[0].split('.')[0] in FORBIDDEN or args[0] == 'linux_pidfd_process_control'):
        raise RuntimeError('numerical or native provider import refused')
    if event in {'subprocess.Popen', 'ctypes.dlopen', 'ctypes.dlsym', 'socket.connect', 'socket.bind', 'socket.getaddrinfo', 'os.system'}:
        raise RuntimeError('process/library/network work refused')
    if event == 'open' and isinstance(args[0], (str, bytes)):
        p = os.fsdecode(args[0])
        if p.startswith(('/dev/nvidia', '/dev/dri', '/tmp/sl5nw_serval02_card1.lock')) or ('/proc/' in p and not p.startswith('/proc/self/')):
            raise RuntimeError('device/standing lock/live process read refused')
sys.addaudithook(audit)
spec = importlib.util.spec_from_file_location('independent_scope_author_tests', SCRIPTS / 'tests/test_native_saved_scope_records.py')
AUTHOR = importlib.util.module_from_spec(spec); spec.loader.exec_module(AUTHOR)
E, O = AUTHOR.evidence, AUTHOR.observer

class ExtraIndependentBoundaries(unittest.TestCase):
    def setUp(self):
        self.raw = AUTHOR.actual('tool_descendant_scope_terminal.json')
        self.intent = AUTHOR.actual('direct_profile_process_intent.json')
        self.owner = self.intent['producer']
        self.uid, self.cgroup = self.owner['uids'][0], self.owner['cgroup']
        self.record = deepcopy(next(iter(self.raw['positive_nodes'].values()))['initial'])

    def test_schema_roles_remain_separate_and_no_raw_fields_are_invented(self):
        self.assertNotIn('cmdline', self.record)
        before = deepcopy(self.record)
        E.validate_scope_snapshot(self.record, self.uid, self.cgroup, O)
        self.assertEqual(self.record, before)
        invalid = deepcopy(self.record); invalid['cmdline'] = 'invented\0'
        with self.assertRaises(ValueError): E.validate_scope_snapshot(invalid, self.uid, self.cgroup, O)
        with self.assertRaises(ValueError): E.validate_scope_snapshot(self.owner, self.uid, self.cgroup, O)
        with self.assertRaises(KeyError): O.validate_snapshot(self.record, self.uid, self.cgroup)
        self.assertEqual(O.validate_snapshot(self.owner, self.uid, self.cgroup), O.stable(self.owner))

    def test_typed_raw_identity_and_command_hash_negative_boundaries(self):
        for key in ('parent_pid', 'process_group', 'session', 'process_start_time', 'state', 'cgroup', 'uids', 'observed_at_seconds'):
            with self.subTest(key=key):
                changed = deepcopy(self.record)
                changed[key] = True
                with self.assertRaises(ValueError): E.validate_scope_snapshot(changed, self.uid, self.cgroup, O)
        empty = next(deepcopy(node['last']) for node in self.raw['positive_nodes'].values() if node['last']['command'] is None)
        self.assertEqual(empty['command_raw_sha256'], hashlib.sha256(b'').hexdigest())
        E.validate_scope_snapshot(empty, self.uid, self.cgroup, O)
        for command in ([''], ['wrong'], [], [True]):
            changed = deepcopy(empty); changed['command'] = command
            with self.assertRaises(ValueError): E.validate_scope_snapshot(changed, self.uid, self.cgroup, O)
        for raw in ('invalid stat', '0 (bad) Z', ''):
            changed = deepcopy(self.record); changed['stat_before'] = raw
            with self.assertRaises(ValueError): E.validate_scope_snapshot(changed, self.uid, self.cgroup, O)

    def test_only_the_single_cross_schema_call_changed_and_all_15_functions_retained(self):
        current = ast.parse((SCRIPTS / 'native_process_evidence.py').read_bytes())
        old = ast.parse((M / 'failed_attempt_preserved/code__native_process_evidence.py').read_bytes())
        def funcs(tree): return {n.name:n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        oldf, newf = funcs(old), funcs(current)
        self.assertEqual(set(newf)-set(oldf), {'scope_stat','validate_scope_snapshot'})
        self.assertFalse(set(oldf)-set(newf))
        for name, expected in F['unchanged_existing_function_ast_hashes'].items():
            raw = ast.dump(newf[name], include_attributes=False).encode()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), expected, name)
            self.assertEqual(ast.dump(newf[name], include_attributes=False), ast.dump(oldf[name], include_attributes=False), name)
        target = deepcopy(oldf['validate_saved'])
        changes = []
        for node in ast.walk(target):
            if isinstance(node, ast.For) and ast.unparse(node.target) == 'node' and ast.unparse(node.iter) == "terminal_scope['positive_nodes'].values()":
                oldcall = node.body[1].value
                self.assertEqual(ast.unparse(oldcall), "observer.validate_snapshot(record, owners['producer']['uids'][0], owners['producer']['cgroup'])")
                node.body[1].value = ast.parse("validate_scope_snapshot(record, owners['producer']['uids'][0], owners['producer']['cgroup'], observer)",mode='eval').body
                changes.append(True)
        self.assertEqual(len(changes),1)
        self.assertEqual(ast.dump(target, include_attributes=False),ast.dump(newf['validate_saved'], include_attributes=False))
        self.assertIn("observer.validate_snapshot(record, owners['producer']['uids'][0], owners['producer']['cgroup'])",ast.unparse(newf['validate_saved']))

suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(AUTHOR.ActualSavedScopeRecordsTests),
    unittest.defaultTestLoader.loadTestsFromTestCase(ExtraIndependentBoundaries)])
capture = io.StringIO()
result = unittest.TextTestRunner(stream=capture, verbosity=2).run(suite)
assert frozen()
receipt={'schema_version':1,'kind':'independent_narrow_native_saved_scope_schema_review','reviewer':'/root/implement_direct_h100_profiles/review_direct_profile_source',
    'source_freeze':{'path':str(FREEZE),'sha256':hashlib.sha256(FREEZE.read_bytes()).hexdigest()},
    'source_sha256':hashlib.sha256((SCRIPTS/'native_process_evidence.py').read_bytes()).hexdigest(),
    'approved':result.wasSuccessful(),'passed':result.wasSuccessful(),'checks':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
    'actual_cpu_job':'6783003','actual_cpu_step':STEP,'elapsed_seconds':time.monotonic()-START,
    'protected_34_hashes_unchanged':True,'protected_function_AST_count':len(F['unchanged_existing_function_ast_hashes']),
    'only_existing_function_change':'validate_saved: scope initial records use strict separate validate_scope_snapshot; all generic full owner/ancestry validators unchanged',
    'scope_zombie_records_are_supported_without_fabricated_cmdline':True,
    'prior_actual_failed_unit_still_failed':True,'actual_original_target_reexecuted':False,
    'numerical_imports':sorted(FORBIDDEN.intersection(sys.modules)),'large_binary_reads':0,'GPU_operations':0,'native_calls':0,'library_loads':0,
    'scientific_credit':0,'profile_admission':False,'worker_admission':False}
for suffix,raw in [('.json',(json.dumps(receipt,indent=2,sort_keys=True)+'\n').encode()),('.stdout.raw',capture.getvalue().encode())]:
    with (M/('independent_focused_'+STEP+suffix)).open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
print(json.dumps(receipt,sort_keys=True))
print(capture.getvalue(),file=sys.stderr)
raise SystemExit(0 if result.wasSuccessful() else 1)

