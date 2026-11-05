"""Independently replay only the rejected actual-size boundary and affected paths."""
import ast
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
ROOT=Path(__file__).resolve().parents[2]
M=Path(__file__).resolve().parent
SCRIPTS=ROOT/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
START=time.monotonic();STEP=os.environ['SLURM_STEP_ID']
assert os.environ['SLURM_JOB_ID']=='6783003' and STEP.isdecimal() and os.getresuid()==(2618919,)*3
assert Path('/proc/self/cgroup').read_text().splitlines()==['0::/system.slice/slurmstepd.scope/job_6783003/step_'+STEP+'/user/task_0']
FP=M/'pinned_size_frozen_source_for_independent_review.json'
assert hashlib.sha256(FP.read_bytes()).hexdigest()=='1febe93dc21dbdf49e96234d0b8a663baa7e4f19d7cd6cccf86179d59dfce054'
F=json.loads(FP.read_bytes())
def frozen():
    return all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==v['sha256'] and oct((ROOT/n).stat().st_mode&0o777)==v['mode'] for n,v in F['files'].items())
assert len(F['files'])==51 and frozen()
def audit(event,args):
    if event=='import' and args[0].split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','scipy','cupy','jaxtari','exploration_platform'}:raise RuntimeError('numerical import refused')
    if event in {'subprocess.Popen','ctypes.dlopen','ctypes.dlsym','socket.connect','socket.bind','socket.getaddrinfo','os.system'}:raise RuntimeError('native/process/network operation refused')
    if event=='open' and isinstance(args[0],(str,bytes)):
        p=os.fsdecode(args[0])
        if p.startswith(('/dev/','/tmp/sl5nw_serval02_card1.lock')) or '/direct_history/content/' in p:raise RuntimeError('device/standinglock/payload refused')
sys.addaudithook(audit)
spec=importlib.util.spec_from_file_location('independent_affected_size_source',SCRIPTS/'tests/test_accepted_evidence_controller_size.py')
author=importlib.util.module_from_spec(spec);spec.loader.exec_module(author)
S=author.snapshot;OBS=[]
# Exact sixth control from .439, no earlier five-control rerun.
old_program=M/'independent_size_review.py'
tree=ast.parse(old_program.read_bytes())
prior_class=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ActualSizeBoundary')
exec(compile(ast.fix_missing_locations(ast.Module(body=[prior_class],type_ignores=[])),str(old_program),'exec'),globals())
class ExtraNarrowChecks(unittest.TestCase):
    def test_bad_size_types_and_zero_mismatch_permanently_refuse(self):
        for value in (True,False,-1,0.0,'0',[],{}):
            with self.subTest(value=value),tempfile.TemporaryDirectory() as tmp,S.ValidationFiles(30) as files:
                path=Path(tmp)/'empty.raw';path.write_bytes(b'')
                with self.assertRaises(ValueError):files.file(path,S.digest(b''),keep_bytes=True,expected_size_bytes=value)
                self.assertTrue(files.failed)
                with self.assertRaises(ValueError):files.file(path,S.digest(b''),keep_bytes=True,expected_size_bytes=0)
        with tempfile.TemporaryDirectory() as tmp,S.ValidationFiles(30) as files:
            path=Path(tmp)/'empty.raw';path.write_bytes(b'')
            with self.assertRaises(ValueError):files.file(path,S.digest(b''),keep_bytes=True,expected_size_bytes=1)
            self.assertTrue(files.failed)
    def test_only_the_two_reviewed_methods_differ_from_rejected_source(self):
        old=M/'independent_rejected_size_freeze_preserved/frozen/10_jax_exploration_platform/scripts/atari_pixel_campaign/accepted_evidence_snapshot.py'
        previous=ast.parse(old.read_bytes());current=ast.parse((SCRIPTS/'accepted_evidence_snapshot.py').read_bytes())
        altered=[]
        old_classes={n.name:n for n in previous.body if isinstance(n,ast.ClassDef)}
        for node in current.body:
            if isinstance(node,ast.ClassDef) and node.name in ('ValidationFiles','AcceptedInventory'):
                name='file' if node.name=='ValidationFiles' else 'add'
                reference=next(f for f in old_classes[node.name].body if isinstance(f,ast.FunctionDef) and f.name==name)
                for index,method in enumerate(node.body):
                    if isinstance(method,ast.FunctionDef) and method.name==name:
                        self.assertNotEqual(ast.dump(method,include_attributes=False),ast.dump(reference,include_attributes=False))
                        node.body[index]=reference;altered.append(node.name+'.'+name)
        self.assertEqual(sorted(altered),['AcceptedInventory.add','ValidationFiles.file'])
        self.assertEqual(ast.dump(current,include_attributes=False),ast.dump(previous,include_attributes=False))
        handoff=json.loads((M/'narrow_source_handoff.json').read_bytes())
        for name,expected in handoff['unchanged_four_dispatch_module_bytes'].items():
            self.assertEqual(hashlib.sha256((SCRIPTS/name).read_bytes()).hexdigest(),expected)
suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(ActualSizeBoundary),
    unittest.defaultTestLoader.loadTestsFromTestCase(author.ControllerSizeTests),
    unittest.defaultTestLoader.loadTestsFromTestCase(ExtraNarrowChecks)])
capture=io.StringIO();result=unittest.TextTestRunner(stream=capture,verbosity=2).run(suite)
# Six genuine streams: only exact previously hashed small output authorities.
classification=json.loads((M/'actual_reference_classification_432.json').read_bytes())
authorities={row['references'][0]['document']['path']:row['references'][0]['document'] for row in classification['all_unresolved']
    if Path(row['original_path']).name in {'direct_controller_stdout.raw','direct_controller_stderr.raw'}}
assert len(authorities)==3
streams=[]
with S.ValidationFiles(30,metadata_only=True) as files:
    inventory=S.AcceptedInventory(files)
    for reference in authorities.values():
        value=files.json(reference);unit=Path(reference['path']).parent
        inventory.accepted_units[unit]={'exact_hashed_actual_output_authority':reference}
        for field in ('stdout','stderr'):
            pair={key:value[field][key] for key in ('path','sha256')}
            inventory.add(pair,metadata=False,shared_metadata_role='accepted_controller_'+field,
                shared_declaration={'authority':reference,'reference_field':[field]})
            entry=files.file(pair['path'],pair['sha256'],keep_bytes=True,expected_size_bytes=value[field]['size_bytes'])
            assert entry['bytes']==entry['identity']['st_size']==value[field]['size_bytes']
            streams.append({'authority':reference,'field':field,'actual':pair,'declared_size':value[field]['size_bytes'],
                'pinned_size':entry['identity']['st_size'],'pinned_bytes':entry['bytes']})
    files.finish(files.paths);small_bytes=files.bytes_hashed
assert len(streams)==6
assert frozen()
receipt={'schema_version':1,'kind':'independent_pinned_controller_size_correction_review','reviewer':'/root/implement_direct_h100_profiles/review_direct_profile_source',
    'approved':result.wasSuccessful(),'passed':result.wasSuccessful(),'checks':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
    'job_id':'6783003','step_id':STEP,'elapsed_seconds':time.monotonic()-START,
    'source_freeze':{'path':str(FP),'sha256':hashlib.sha256(FP.read_bytes()).hexdigest()},
    'source_sha256':hashlib.sha256((SCRIPTS/'accepted_evidence_snapshot.py').read_bytes()).hexdigest(),'all51frozenbytes_modes_unchanged':True,
    'exact439boundary_replayed':OBS,'six_actual_streams':streams,'actual_small_metadata_bytes':small_bytes,
    'only_file_and_add_methods_changed':True,'all1938_capture_reference_and_owner_resolution_logic_AST_unchanged':True,
    'all_four_dispatch_module_bytes_unchanged':True,'original55_history_NUM_gates_retained_from_prior_review':True,
    'earlier_five_controls_repeated':False,'full142_reference_resolution_repeated':False,
    'prior437resolution_receipt_retained':{'path':str(M/'actual_reference_classification_437.json'),
        'sha256':hashlib.sha256((M/'actual_reference_classification_437.json').read_bytes()).hexdigest()},
    'large_payload_bytes_read':0,'actual_full_snapshot':False,'NUM_GPU_native_tool_actions':0,'scientific_credit':0,'worker_admission':False}
for suffix,raw in [('.json',(json.dumps(receipt,indent=2,sort_keys=True)+'\n').encode()),('.stdout.raw',capture.getvalue().encode())]:
    with (M/('independent_pinned_size_'+STEP+suffix)).open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
print(json.dumps({k:v for k,v in receipt.items() if k!='six_actual_streams'},sort_keys=True))
print(capture.getvalue(),file=sys.stderr)
raise SystemExit(0 if result.wasSuccessful() else 1)

