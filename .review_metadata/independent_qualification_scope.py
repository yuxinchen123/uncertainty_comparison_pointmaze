WORKTREE='/p/rlprojects/RND_worktrees/2026-11-01-17-55_a4500-common-confirmation_prior-correctness-required_rounds-7_complete-updates-per-arm-10_copies-1-2'
RUN='/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1'
"""Independently compare unchanged numerical files and the fixed UVA execution scope."""
import ast, copy, hashlib, json, os, pathlib, subprocess, datetime
w=pathlib.Path(WORKTREE);s=w/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
r=pathlib.Path(RUN);base='6066d9ea35fbc9c045855144550971c2ac5895f9';qualified='e5afbf6b51a3d1666b46e71c0631a45d47080cd8'
assert os.environ['SLURM_JOB_ID']=='6783003' and os.environ['SLURM_STEP_ID'].isdigit()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def blob(commit,relative):return subprocess.check_output(['git','-C',str(w),'show',commit+':'+relative])
def tree(raw):return ast.parse(raw)
def functions(raw):return {n.name:n for n in tree(raw).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
def dump(node):return ast.dump(node,include_attributes=False)
prefix='10_jax_exploration_platform/scripts/atari_pixel_campaign/'
current=(s/'measure_selected.py').read_bytes();old=blob(qualified,prefix+'measure_selected.py');prior=blob(base,prefix+'measure_selected.py')
assert dump(functions(current)['_compute'])==dump(functions(prior)['_compute'])
# The only dispatch rewritten below is inspected here before specialization.
dispatch=functions(current)['producer_support']
expected_dispatch = ast.parse("def expected(plan):\n    if 'direct_binding' in plan:\n        import direct_measurement as support\n        return support.support(plan)\n    if 'direct_qualification' in plan or 'direct_qualification_inputs' in plan:\n        raise ValueError('direct qualification requires its explicit direct binding profile')\n    return configured_support(plan)\n").body[0]
assert [dump(n) for n in dispatch.body[1:]] == [dump(n) for n in expected_dispatch.body]
author_scope=(w/'.review_metadata/check_qualification_reuse_scope.py').read_bytes()
scope_tree=tree(author_scope)
klass=next(n for n in scope_tree.body if isinstance(n,ast.ClassDef) and n.name=='UvaBranch')
namespace={'ast':ast};exec(compile(ast.fix_missing_locations(ast.Module(body=[klass],type_ignores=[])),'<reviewed_exact_scope_class>','exec'),namespace)
specialize=namespace['UvaBranch']
assert dump(specialize().visit(copy.deepcopy(functions(old)['_compute'])))==dump(specialize().visit(copy.deepcopy(functions(current)['_compute'])))
qdir=r/'data/2026-11-01-13-50_original-qualification_explicit-pytest-root_jaguar03_gpu-count-1_processor-count-8_host-memory-gib-64_time-limit-hours-3_copies-1_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_reference-updates-1_seed-1'
p=json.loads((qdir/'measurement_plan.json').read_bytes())
assert sha((qdir/'target_qualification_result.json').read_bytes())=='0c4f94c2d58fdd7277a261481d7a52c4bb5b19e42612a8eae67dff6890c62cd1'
exact=[]
for group in ['core','paired_source']:
 rec=p[group]
 for rel,expected in rec['files'].items():
  expected=expected['sha256'] if isinstance(expected,dict) else expected
  actual=pathlib.Path(rec['source_root'])/rel;raw=actual.read_bytes()
  assert sha(raw)==expected and raw==subprocess.check_output(['git','-C',rec['source_root'],'show',rec['source_commit']+':'+rel])
  exact.append({'path':str(actual),'sha256':expected})
assert len(exact)==13
unchanged=['train_optimized.py','short_learning.py','uva_slurm_configured_isolation.py']
for name in unchanged:
 assert blob(qualified,prefix+name)==(s/name).read_bytes()
support_changes={k for k,v in functions(blob(qualified,prefix+'configured_measurement.py')).items() if dump(v)!=dump(functions((s/'configured_measurement.py').read_bytes())[k])}
assert support_changes=={'MemoryObservation','qualification_sources','renderer_contract','run_qualification','validate_window','verify_qualification_backend','verify_qualification_source_receipts','verify_qualification_sources','verify_saved_measurement'}
freeze=w/'.review_metadata/common_confirmation_complete_source_freeze.json';f=json.loads(freeze.read_bytes())
assert all(sha(pathlib.Path(path).read_bytes())==expected for path,expected in f['files'].items())
value={'kind':'independent_common_confirmation_numerical_qualification_reuse_scope','passed':True,
 'job_id':'6783003','step_id':os.environ['SLURM_STEP_ID'],'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'source_freeze':{'path':str(freeze),'sha256':sha(freeze.read_bytes())},'base_commit':base,'qualified_helper_commit':qualified,
 'qualified_result_sha256':'0c4f94c2d58fdd7277a261481d7a52c4bb5b19e42612a8eae67dff6890c62cd1',
 'new_compute_AST_identical_to_base':True,'UVA_specialized_compute_AST_identical_to_qualified':True,'whole_qualified_compute_AST_identical':False,
 'reviewed_specializer_sha256':sha(author_scope),'all13_numerical_and_Executor_files':exact,'three_unchanged_support_files':unchanged,
 'configured_support_changes_manually_reviewed':sorted(support_changes),
 'scope':'UVA single-card intermediate common confirmation only; absent direct binding/qualification, existing original configuration and runtime, unchanged D28 numerical files. New common metadata runs before admission and results bind its proof.',
 'monitor79a_is_progress_only_for_this_new_feature':True,'scientific_acceptance_requires_exact_new_staged_consumer_on_owned_compute':True,
 'numerical_work':False,'gpu_or_job_actions':0,'scientific_credit':0,'admission_granted':False}
out=w/'.review_metadata'/('independent_qualification_reuse_scope_6783003_'+os.environ['SLURM_STEP_ID']+'.json')
with out.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
print(json.dumps({'passed':True,'receipt':str(out),'sha256':sha(out.read_bytes())}))
