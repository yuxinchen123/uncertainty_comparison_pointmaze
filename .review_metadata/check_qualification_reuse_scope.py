"""Compare exact numerical source and the unchanged UVA execution wrapper without numerical imports."""
import ast,copy,hashlib,json,os,subprocess
from pathlib import Path
W=Path(__file__).absolute().parent.parent
S=W/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
R=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1')
BASE='6066d9ea35fbc9c045855144550971c2ac5895f9'
E5='e5afbf6b51a3d1666b46e71c0631a45d47080cd8'
REL='10_jax_exploration_platform/scripts/atari_pixel_campaign/measure_selected.py'
assert os.environ.get('SLURM_JOB_ID')=='6783003' and os.environ.get('SLURM_STEP_ID','').isdigit()
def sha(raw):
    """Identify exact source bytes."""
    return hashlib.sha256(raw).hexdigest()
def function(raw,name):
    """Select one literal top-level function, refusing ambiguous source."""
    matches=[n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name==name]
    assert len(matches)==1
    return matches[0]
class UvaBranch(ast.NodeTransformer):
    """Remove only explicitly inapplicable direct-host branches for a UVA plan."""
    def visit_Name(self,node):
        """The direct flag is false in the explicitly configured UVA protocol."""
        return ast.Constant(False) if node.id=='direct' and isinstance(node.ctx,ast.Load) else node
    def visit_Compare(self,node):
        """The direct-only plan keys are absent for this one comparison."""
        node=self.generic_visit(node)
        if isinstance(node.left,ast.Constant) and node.left.value in {'direct_binding','direct_qualification'} and len(node.ops)==1 and len(node.comparators)==1 and isinstance(node.comparators[0],ast.Name) and node.comparators[0].id=='plan':
            if isinstance(node.ops[0],ast.In):return ast.Constant(False)
            if isinstance(node.ops[0],ast.NotIn):return ast.Constant(True)
        return node
    def visit_Call(self,node):
        """The reviewed UVA dispatch returns the existing configured support function."""
        node=self.generic_visit(node)
        if isinstance(node.func,ast.Name) and node.func.id=='producer_support':node.func.id='configured_support'
        return node
    def visit_BoolOp(self,node):
        """Simplify only exact Boolean constants introduced by this specialization."""
        node=self.generic_visit(node)
        if isinstance(node.op,ast.And) and any(isinstance(n,ast.Constant) and n.value is False for n in node.values):return ast.Constant(False)
        if isinstance(node.op,ast.Or) and any(isinstance(n,ast.Constant) and n.value is True for n in node.values):return ast.Constant(True)
        node.values=[n for n in node.values if not(isinstance(n,ast.Constant) and (n.value is True if isinstance(node.op,ast.And) else n.value is False))]
        return node.values[0] if len(node.values)==1 else node
    def visit_IfExp(self,node):
        """Choose only branches whose direct-host condition has become a Boolean."""
        node=self.generic_visit(node)
        return (node.body if node.test.value else node.orelse) if isinstance(node.test,ast.Constant) and type(node.test.value) is bool else node
    def visit_If(self,node):
        """Keep every unknown condition and remove only proven direct-host alternatives."""
        node=self.generic_visit(node)
        return (node.body if node.test.value else node.orelse) if isinstance(node.test,ast.Constant) and type(node.test.value) is bool else node
    def visit_Assign(self,node):
        """Remove the false direct flag and the proven no-operation context assignment."""
        if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='direct':
            assert isinstance(node.value,ast.Compare) and isinstance(node.value.left,ast.Constant) and node.value.left.value=='direct_binding'
            return None
        node=self.generic_visit(node)
        if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='whole_scope':
            assert isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='nullcontext'
            assert len(node.value.args)==1 and isinstance(node.value.args[0],ast.Dict) and not node.value.args[0].keys
            return None
        return node
    def visit_With(self,node):
        """Unwrap only the empty direct-only observation context on UVA."""
        node=self.generic_visit(node)
        if len(node.items)==1:
            context=node.items[0].context_expr
            if isinstance(context,ast.Name) and context.id=='whole_scope':return node.body
            if isinstance(context,ast.Call) and isinstance(context.func,ast.Name) and context.func.id=='nullcontext' and not context.args:return node.body
        return node
old=subprocess.check_output(['git','-C',str(W),'show',E5+':'+REL]);base=subprocess.check_output(['git','-C',str(W),'show',BASE+':'+REL]);current=(S/'measure_selected.py').read_bytes()
dump=lambda node:ast.dump(node,include_attributes=False)
assert dump(function(base,'_compute'))==dump(function(current,'_compute')),'new metadata change altered the numerical execution function'
old_uva=UvaBranch().visit(copy.deepcopy(function(old,'_compute')));new_uva=UvaBranch().visit(copy.deepcopy(function(current,'_compute')))
if dump(old_uva)!=dump(new_uva):
    import difflib
    (W/'.review_metadata/uva_compute_specialization_difference.txt').write_text('\n'.join(difflib.unified_diff(ast.unparse(old_uva).splitlines(),ast.unparse(new_uva).splitlines())))
    raise AssertionError('qualified E5 and current UVA-specialized execution functions differ')
qualification=R/'data/2026-11-01-13-50_original-qualification_explicit-pytest-root_jaguar03_gpu-count-1_processor-count-8_host-memory-gib-64_time-limit-hours-3_copies-1_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_reference-updates-1_seed-1'
plan=json.loads((qualification/'measurement_plan.json').read_bytes());result=json.loads((qualification/'target_qualification_result.json').read_bytes());assert sha((qualification/'target_qualification_result.json').read_bytes())=='0c4f94c2d58fdd7277a261481d7a52c4bb5b19e42612a8eae67dff6890c62cd1'
protected={}
for group in ('core','paired_source'):
 record=plan[group]
 for relative,expected in record['files'].items():
    path=Path(record['source_root'])/relative;raw=path.read_bytes();expected=expected['sha256'] if isinstance(expected,dict) else expected
    blob=subprocess.check_output(['git','-C',record['source_root'],'show',record['source_commit']+':'+relative]);assert sha(raw)==expected and raw==blob
    protected[relative]={'source_commit':record['source_commit'],'sha256':expected,'python_ast_sha256':sha(ast.dump(ast.parse(raw),include_attributes=False).encode()) if relative.endswith('.py') else None}
value={'kind':'explicit_e5_qualification_reuse_scope_for_metadata_only_common_confirmation','passed':True,'allocation':'6783003','step':os.environ['SLURM_STEP_ID'],'base_commit':BASE,'qualified_helper_commit':E5,'qualified_result_sha256':'0c4f94c2d58fdd7277a261481d7a52c4bb5b19e42612a8eae67dff6890c62cd1','new_compute_ast_equals_base':True,'uva_specialized_compute_ast_equals_qualified_e5':True,'qualified_e5_full_compute_ast_equals_current':dump(function(old,'_compute'))==dump(function(current,'_compute')),'specialization_scope':'Only absent direct_binding/direct_qualification, false direct flag, reviewed producer_support to configured_support dispatch and direct-only null contexts; every remaining function statement is compared exactly.','protected_numerical_and_executor_files':protected,'runtime_environment':plan['runtime_environment'],'asset_record':plan['asset_record'],'numerical_equations_or_config_changes':False,'gpu_or_numerical_operations':0,'scientific_credit':0,'admission_granted':False}
raw=(json.dumps(value,indent=2)+'\n').encode();out=W/'.review_metadata'/('qualification_reuse_scope_6783003_'+os.environ['SLURM_STEP_ID']+'.json')
with out.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
print(json.dumps({'receipt':str(out),'sha256':sha(raw),'passed':True,'protected_files':len(protected)}))
