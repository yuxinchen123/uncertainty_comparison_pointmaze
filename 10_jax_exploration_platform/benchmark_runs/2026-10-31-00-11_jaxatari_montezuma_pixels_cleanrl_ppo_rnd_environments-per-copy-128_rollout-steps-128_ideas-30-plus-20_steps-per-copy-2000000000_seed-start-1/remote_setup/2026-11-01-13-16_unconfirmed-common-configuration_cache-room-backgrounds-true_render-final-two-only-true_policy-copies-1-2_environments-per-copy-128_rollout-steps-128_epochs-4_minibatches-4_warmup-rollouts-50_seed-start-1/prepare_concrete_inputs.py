"""Write only the four explicitly requested common-candidate JSON inputs from frozen original fields."""
import ast
from collections import Counter,defaultdict
import copy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

DIRECTORY=Path(__file__).resolve().parent
R=DIRECTORY.parents[1]
SOURCE=Path('/p/rlprojects/RND_worktrees/atari-cleanrl-source-d28e673')
COMMIT='d28e6735e79788f618acb0ebfde8609db1f776af'
SCRIPTS='10_jax_exploration_platform/scripts/atari_pixel_campaign/'
PIXELS='10_jax_exploration_platform/src/exploration_platform/atari_pixels/'
AUDIT=R/'remote_setup/2026-11-01-12-50_initial-comparisons-completion_ideas-30_gpu-types-2_copies-1-2_comparisons-120_source-d28e6735e797/initial_comparisons_audit.json'
BLOCKED=[]

def guard(event,args):
    """Permit only source metadata commands; no numerical package, GPU, library or network work."""
    if event=='import' and args[0].split('.')[0] in {'jax','jaxlib','numpy','scipy','torch','flax','optax','jaxtari','pynvml'}:
        BLOCKED.append([event,args[0]]);raise RuntimeError('numerical import forbidden during configuration readiness')
    if event in {'ctypes.dlopen','socket.connect','os.system'}:
        BLOCKED.append([event,str(args)]);raise RuntimeError('library/network/device action forbidden during configuration readiness')
    if event=='subprocess.Popen' and Path(str(args[1][0])).name!='git':
        BLOCKED.append([event,list(args[1])]);raise RuntimeError('only read-only Git metadata is permitted')

sys.addaudithook(guard)

def sha(path):
    """Bind one original source, raw result or generated input to exact bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def ref(path):
    """Record a complete absolute input or evidence path and hash."""
    return {'path':str(path),'sha256':sha(path)}

def write(name,value):
    """Exclusively create each concrete readiness artifact; never revise a prior preparation."""
    path=DIRECTORY/name
    with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    return ref(path)

def difference(before,after,prefix=''):
    """Before: two full nested configurations; after: the exact leaf changes only."""
    assert type(before) is type(after)
    if isinstance(before,dict):
        assert set(before)==set(after)
        return [row for key in before for row in difference(before[key],after[key],prefix+'.'+key if prefix else key)]
    return [] if before==after else [{'field':prefix,'before':before,'after':after}]

def main():
    """Check all120 raw hashes and the source's two accepted mappings, then write four full inputs only."""
    assert os.environ['SLURM_JOB_ID']=='6781231' and os.environ['SLURM_STEP_ID'].isdigit()
    print('owned metadata step '+os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],flush=True)
    environment={**os.environ,'GIT_OPTIONAL_LOCKS':'0'}
    def git(*args):
        """Read only the exact frozen numerical checkout's committed metadata."""
        return subprocess.check_output(['git','-C',str(SOURCE),*args],env=environment)
    assert git('rev-parse','HEAD').decode().strip()==COMMIT
    assert not git('status','--porcelain','--untracked-files=all').strip()
    sources=[PIXELS+name for name in sorted(p.name for p in (SOURCE/PIXELS).glob('*.py'))]
    sources.extend(SCRIPTS+name for name in ('ideas.py','paired_benchmark.py','check_rendering.py'))
    source_files={}
    for relative in sources:
        current=(SOURCE/relative).read_bytes();committed=git('show',COMMIT+':'+relative)
        assert current==committed
        source_files[relative]={'sha256':hashlib.sha256(current).hexdigest(),'git_blob':git('rev-parse',COMMIT+':'+relative).decode().strip()}
    audit=json.loads(AUDIT.read_text())
    assert audit['passed'] is True and audit['comparisons']==120 and audit['source_commit']==COMMIT
    assert audit['owned_cpu_job']=='6781231' and audit['step_id']=='72'
    assert audit['decision_counts']=={'accepted':28,'inconclusive':37,'rejected':55}
    groups=defaultdict(list);raw={}
    for row in audit['rows']:
        path=Path(row['result_path']);assert sha(path)==row['result_sha256']
        result=json.loads(path.read_text());assert result['source_commit']==COMMIT and result['decision']==row['decision']
        assert result['hardware_class']==row['hardware_class'] and result['copies']==row['copies']
        assert result['correctness_passed'] is True and result['measurement_valid'] is True
        assert result['copy_seeds']==list(range(1,result['copies']+1))
        groups[row['name']].append(row);raw[str(path)]=result
    expected_cases={('h100',1),('h100',2),('jaguar03_rtx_a4500',1),('jaguar03_rtx_a4500',2)}
    assert len(audit['rows'])==120 and len(groups)==30
    for rows in groups.values():assert len(rows)==4 and {(row['hardware_class'],row['copies']) for row in rows}==expected_cases
    common=sorted(name for name,rows in groups.items() if all(row['decision']=='accepted' and row['correctness_passed'] is True for row in rows))
    assert common==audit['individually_accepted_on_both_gpus_and_both_copy_counts']==['cache_room_backgrounds','render_final_two_game_states']
    # Before: named source hypotheses; after: precisely two environment flags, without a learner or execution change.
    idea_tree=ast.parse((SOURCE/SCRIPTS/'ideas.py').read_text())
    catalog=next(node.value for node in idea_tree.body if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='INITIAL_IDEAS' for target in node.targets))
    changes={}
    for node in catalog.elts:
        name=ast.literal_eval(node.args[1])
        if name in common:
            values={keyword.arg:ast.literal_eval(keyword.value) for keyword in node.keywords if keyword.arg in {'learner','environment','execution'}}
            assert not values.get('learner') and not values.get('execution')
            changes[name]=values['environment']
    assert changes=={'cache_room_backgrounds':{'cache_room_backgrounds':True},'render_final_two_game_states':{'render_final_two_only':True}}
    config_tree=ast.parse((SOURCE/PIXELS/'config.py').read_text());cls=next(node for node in config_tree.body if isinstance(node,ast.ClassDef) and node.name=='PixelPPOConfig')
    learner={node.target.id:ast.literal_eval(node.value) for node in cls.body if isinstance(node,ast.AnnAssign) and node.value is not None}
    env_tree=ast.parse((SOURCE/PIXELS/'environment.py').read_text());cls=next(node for node in env_tree.body if isinstance(node,ast.ClassDef) and node.name=='AtariPixelEnvironment');constructor=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='__init__')
    game={arg.arg:ast.literal_eval(value) for arg,value in zip(constructor.args.args[-len(constructor.args.defaults):],constructor.args.defaults)}
    execution_tree=ast.parse((SOURCE/SCRIPTS/'paired_benchmark.py').read_text());execution=ast.literal_eval(next(node.value for node in execution_tree.body if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='DEFAULT_EXECUTION' for target in node.targets)))
    original=json.loads(json.dumps({'learner':learner,'environment':game,'execution':execution}))
    assert original['environment']['cache_room_backgrounds'] is False and original['environment']['render_final_two_only'] is False
    assert learner['n_envs']==128 and learner['num_steps']==128 and learner['update_epochs']==4 and learner['num_minibatches']==4 and learner['observation_warmup_iterations']==50 and learner['total_timesteps']==2000000000 and learner['anneal_learning_rate'] is True
    assert execution=={'compilation_mode':'whole_iteration','iterations_per_call':1,'report_readout':'none','report_every_iterations':10}
    records=[]
    for copies in (1,2):
        reference=copy.deepcopy(original);reference['learner']['n_copies']=copies
        candidate=copy.deepcopy(reference)
        for values in changes.values():candidate['environment'].update(values)
        diff=difference(reference,candidate)
        assert sorted(diff,key=lambda row:row['field'])==[{'field':'environment.cache_room_backgrounds','before':False,'after':True},{'field':'environment.render_final_two_only','before':False,'after':True}]
        assert candidate['learner']==reference['learner'] and candidate['execution']==reference['execution']
        for name in common:
            for row in groups[name]:
                if row['copies']!=copies:continue
                result=raw[row['result_path']];assert result['reference']==reference
                individual=copy.deepcopy(reference);individual['environment'].update(changes[name]);assert result['candidate']==individual
        reference_file=write('original_policy-copies-'+str(copies)+'_seed-start-1.json',reference)
        candidate_file=write('candidate_cache-room-backgrounds-true_render-final-two-only-true_policy-copies-'+str(copies)+'_seed-start-1.json',candidate)
        records.append({'policy_copies':copies,'seed_start':1,'copy_seeds':list(range(1,copies+1)),'reference':reference_file,'unconfirmed_candidate':candidate_file,'exact_difference':diff,'all_full_fields_retained':True})
    runtime=write('runtime_environment.json',{'JAX_ENABLE_X64':'true','JAX_DEFAULT_MATMUL_PRECISION':'highest','XLA_FLAGS':'--xla_gpu_exclude_nondeterministic_ops=true'})
    protocol=write('comparison_protocol.json',{'schema_version':1,'kind':'unconfirmed_common_candidate_intermediate_readiness','source_root':str(SOURCE),'source_commit':COMMIT,'configuration_inputs':records,'runtime_environment':runtime,'hardware_classes':['h100','jaguar03_rtx_a4500'],'correctness_required_before_timing':True,'timing_rounds':7,'full_updates_per_variant_per_round':10,'variant_orders':[['reference','candidate'] if index%2==0 else ['candidate','reference'] for index in range(7)],'measurement_scope':'training_iteration_execution','initially_selected_mechanisms':changes,'unconfirmed_combination':True,'prerequisites':['Actual target qualification must complete and receive independent review.','The published persistent monitor must support the actual producer records.','The applicable explicit assigned-device producer/binding must be reviewed and published.','Separate full-input and copied-state correctness must pass for this combination before its timing.'],'formal_units_prepared':False,'future_qualification_paths_or_hashes_invented':False,'new_correctness_credit':0,'new_timing_credit':0,'new_case_credit':0,'new_idea_credit':0,'admission_granted':False,'jobs_submitted':False})
    assert not BLOCKED and not git('status','--porcelain','--untracked-files=all').strip()
    receipt={'schema_version':1,'kind':'unconfirmed_common_candidate_full_field_metadata_readiness','created_at_utc':datetime.now(timezone.utc).isoformat(),'owned_cpu_job':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],'executed_source':ref(Path(__file__)),'raw120_audit':ref(AUDIT),'raw120_audit_job_step':'6781231.72','raw_result_hashes_checked':120,'all30_mechanisms_have_exactly_four_cases':True,'only_all_four_accepted_individual_mechanisms':common,'accepted_common_case_evidence':[row for name in common for row in groups[name]],'source_commit':COMMIT,'source_root':str(SOURCE),'source_files':source_files,'source_worktree_clean_before_and_after':True,'source_idea_mapping':changes,'configuration_inputs':records,'runtime_environment':runtime,'comparison_protocol':protocol,'field_counts':{name:len(value) for name,value in original.items()},'precision_intent':{'networks_and_optimizer':'float32 from unchanged original source','running_statistics':'float64 from unchanged original source','matmul':'highest','deterministic_operations':True},'independent_stream_policy':'Unchanged original quantity_keys derives one PRNG stream per copy seed and named quantity; inputs explicitly retain seeds1 and1,2. No runtime stream-independence test was run here.','formal_units_prepared':False,'gpu_operations':0,'numerical_imports':0,'blocked_attempts':BLOCKED,'jobs_submitted':False,'new_correctness_credit':0,'new_timing_credit':0,'new_case_credit':0,'new_idea_credit':0,'admission_granted':False,'returncode':0}
    result=write('readiness_receipt.json',receipt)
    publication=write('publication_roster.json',{'schema_version':1,'kind':'common_configuration_readiness_publication_roster','files':[ref(path) for path in sorted(DIRECTORY.iterdir()) if path.is_file()]})
    print(json.dumps({'directory':str(DIRECTORY),'step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'receipt':result,'publication_roster':publication,'configurations':records}),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as error:
        write('readiness_failure.json',{'job_id':os.environ.get('SLURM_JOB_ID'),'step_id':os.environ.get('SLURM_STEP_ID'),'source':ref(Path(__file__)),'error':str(error),'traceback':traceback.format_exc(),'gpu_operations':0,'formal_units_prepared':False})
        raise
