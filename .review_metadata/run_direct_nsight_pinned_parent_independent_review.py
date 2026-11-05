"""Independently verify narrow pinned-parent and native restoration corrections on owned CPU."""
import ast,copy,hashlib,io,json,os,socket,sys,time,unittest
from pathlib import Path
from unittest.mock import patch
W=Path(__file__).resolve().parents[1];S=W/'10_jax_exploration_platform/scripts/atari_pixel_campaign';R=W/'.review_metadata'
F=R/'direct_nsight_producer_source_freeze_author_6_pinned_parent_and_restore.json';frozen=json.loads(F.read_text())
assert hashlib.sha256(F.read_bytes()).hexdigest()=='6911df18d1f263c8c9528aafdc994e506a5d3c1118997720ee92bfe6eb4f1433'
assert os.environ['SLURM_JOB_ID']=='6783003' and os.environ['SLURM_STEP_ID'].isdecimal()
assert all(hashlib.sha256((W/p).read_bytes()).hexdigest()==h for p,h in frozen['files'].items())
label='direct_nsight_pinned_parent_independent_'+os.environ['SLURM_JOB_ID']+'_'+os.environ['SLURM_STEP_ID']
actual=R/('direct_tool_scope_pinned_parent_independent_actual_cpu_'+os.environ['SLURM_JOB_ID']+'_'+os.environ['SLURM_STEP_ID']);actual.mkdir()
os.environ['DIRECT_CPU_SCOPE_REVIEW_OUTPUT']=str(actual)
sys.path.insert(0,str(S/'tests'));sys.path.insert(0,str(S))
import test_direct_tool_descendants as parent_fixtures
import test_direct_profile_process as saved_fixtures
import direct_tool_descendants as source
from test_nsight_process_identity import M

class IndependentNarrowBoundaries(unittest.TestCase):
    """Check additional private metadata boundaries without actual unrelated process inspection."""
    def parent(self):
        """Make a labelled private proc fixture with a pinned non-direct parent."""
        fixture=parent_fixtures.PinnedParentChildListTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        return fixture
    def saved(self):
        """Make a standalone saved-scope fixture so earlier release hashes cannot mask a field check."""
        fixture=saved_fixtures.DirectSavedProcessTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        unit,validate=fixture.scope_reader_fixture();return unit,validate
    def test_wrong_parent_owner_refuses_before_any_thread_child_read(self):
        """An already changed owner cannot introduce or expose another process's child list."""
        fixture=self.parent();(fixture.root/'102/status').write_text('Uid:\t999\t999\t999\t999\n')
        reads=[]
        def read(directory,name):
            reads.append(name);return fixture.original_bounded(directory,name)
        with patch.object(source,'bounded',side_effect=read),self.assertRaises(ValueError):fixture.scope.child_lists(102)
        self.assertNotIn('children',reads)
    def test_wrong_parent_cgroup_refuses_before_any_thread_child_read(self):
        """An already changed allocation cannot authorize reading any child edge."""
        fixture=self.parent();(fixture.root/'102/cgroup').write_text('0::/another-owned-scope\n')
        reads=[]
        def read(directory,name):
            reads.append(name);return fixture.original_bounded(directory,name)
        with patch.object(source,'bounded',side_effect=read),self.assertRaises(ValueError):fixture.scope.child_lists(102)
        self.assertNotIn('children',reads)
    def test_boolean_native_setup_errno_is_rejected(self):
        """A Boolean cannot replace a successful typed native setup return field."""
        unit,validate=self.saved();path=unit/'tool_descendant_scope_begin.json';begin=M.read_json(path)
        begin['set_subreaper']['errno']=False;path.unlink();M.publish(path,begin)
        terminal_path=unit/'tool_descendant_scope_terminal.json';terminal=M.read_json(terminal_path);terminal['begin']=M.receipt(path)
        terminal_path.unlink();M.publish(terminal_path,terminal)
        restored_path=unit/'tool_descendant_scope_restored.json';restored=M.read_json(restored_path);restored['terminal']=M.receipt(terminal_path)
        restored_path.unlink();M.publish(restored_path,restored)
        with self.assertRaises(ValueError):validate()
    def test_consistent_previously_enabled_subreaper_state_is_restored(self):
        """Causal restoration keeps a prior flag1 instead of silently requiring a hard-coded flag0."""
        unit,validate=self.saved();path=unit/'tool_descendant_scope_begin.json';begin=M.read_json(path)
        begin['previous_subreaper']['value']=1;path.unlink();M.publish(path,begin)
        terminal_path=unit/'tool_descendant_scope_terminal.json';terminal=M.read_json(terminal_path);terminal['begin']=M.receipt(path)
        terminal_path.unlink();M.publish(terminal_path,terminal)
        restored_path=unit/'tool_descendant_scope_restored.json';restored=M.read_json(restored_path)
        restored['terminal']=M.receipt(terminal_path);restored['restore']['value']=1;restored['verified']['value']=1
        restored_path.unlink();M.publish(restored_path,restored)
        self.assertTrue(validate()['actual_kernel_descendant_closure_verified'])

names=('test_direct_tool_descendants',
    'test_direct_profile_process.DirectSavedProcessTests.test_saved_direct_waits_accept_positive_edges_without_process_group_equality',
    'test_direct_profile_process.DirectSavedProcessTests.test_restore_request_value_and_native_return_types_must_match_previous_and_verified',
    'test_direct_profile_process.DirectSavedProcessTests.test_native_counts_must_join_actual_before_and_terminal_snapshots',
    'test_direct_profile_process.DirectSavedProcessTests.test_saved_child_list_requires_exact_before_and_after_parent_identity')
suite=unittest.TestSuite()
for name in names:suite.addTests(unittest.defaultTestLoader.loadTestsFromName(name))
suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(IndependentNarrowBoundaries))
output=io.StringIO();started=time.time();result=unittest.TextTestRunner(stream=output,verbosity=2).run(suite)
raw=output.getvalue().encode();(R/(label+'_unittest.raw')).write_bytes(raw)
unchanged=all(hashlib.sha256((W/p).read_bytes()).hexdigest()==h for p,h in frozen['files'].items())
artifacts=[]
for p in sorted(actual.rglob('*')):
    if p.is_file():artifacts.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size_bytes':p.stat().st_size})
before=json.loads((R/'direct_nsight_owned_producer_source_before.json').read_text())
protected={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in before['files'].items()}
core=next(n for n in ast.parse((S/'profile_selected.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_capture_core')
core_unchanged=ast.dump(core,include_attributes=False)==before['capture_core_ast']
record={'kind':'independent_direct_nsight_pinned_parent_and_restore_review','source_freeze':{'path':str(F),'sha256':hashlib.sha256(F.read_bytes()).hexdigest()},
    'base':frozen['base'],'job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],'hostname':socket.gethostname(),'pid':os.getpid(),
    'started_at_seconds':started,'finished_at_seconds':time.time(),'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
    'source_unchanged_before_and_after':unchanged,'protected_file_checks':protected,'capture_core_syntax_unchanged':core_unchanged,
    'unittest_output':{'path':str(R/(label+'_unittest.raw')),'sha256':hashlib.sha256(raw).hexdigest()},'actual_cpu_artifacts':artifacts,
    'approved_narrow_source_corrections':result.wasSuccessful() and unchanged and len(protected)==26 and all(protected.values()) and core_unchanged,
    'actual_tool_or_gpu_execution':False,'standing_gpu_lock_opened':False,'actual_foreign_processes_touched':0,'gpu_operations':0,
    'scientific_credit':0,'admission_granted':False,
    'scope':'Only the pinned-parent and native restore corrections plus actual temporary CPU cleanup regressions. This grants no real tool, CUDA, target, profile or worker admission.'}
(R/(label+'.json')).write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'receipt':str(R/(label+'.json')),'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
    'source_unchanged':unchanged,'protected26_unchanged':len(protected)==26 and all(protected.values()),'core_unchanged':core_unchanged,
    'actual_cpu_artifacts':len(artifacts),'approved_narrow_source_corrections':record['approved_narrow_source_corrections'],
    'output_tail':output.getvalue()[-1600:]}))
assert record['approved_narrow_source_corrections']
