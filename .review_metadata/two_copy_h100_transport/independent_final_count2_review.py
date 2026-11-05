"""Review only the separate two-copy metadata entry changes; reuse prior custody approval explicitly."""
import ast
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
import traceback
import unittest

D=Path(__file__).resolve().parent;W=D.parents[1]
FREEZE=D/'final_runtime_source_freeze.json'
EXPECTED='764aeaae6018103a9f8264f192ff7a815f41695066133b3b055d8df3afd3595e'
SOURCE='d28e6735e79788f618acb0ebfde8609db1f776af'
CHECKS=[]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def need(value,message):
    """Record a source or metadata conclusion only after its actual predicate passes."""
    if not value:raise ValueError(message)
    CHECKS.append(message)
def publish(path,value):
    """Preserve this independent CPU review exactly once with actual timestamps and returns."""
    with path.open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
def audit(event,args):
    """Reject native, numerical, device, network and actual child-process operations."""
    if event.startswith('ctypes.') or event in ('subprocess.Popen','socket.connect','socket.bind','os.chmod'):raise ValueError('actual native/process/network/permission action refused')
    if event=='import' and args[0].split('.')[0] in {'jax','jaxlib','numpy','flax','optax','torch','jaxtari','cupy','pynvml'}:raise ValueError('numerical/device import refused')
def expression(source,function,message):
    """Use the actual strict count predicate without calling a transport or preparing a unit."""
    function=next(node for node in ast.parse(source).body if isinstance(node,ast.FunctionDef) and node.name==function)
    calls=[node for node in ast.walk(function) if isinstance(node,ast.Call) and any(isinstance(arg,ast.Constant) and arg.value==message for arg in node.args)]
    need(len(calls)==1,'one exact changed count/reference predicate')
    return compile(ast.fix_missing_locations(ast.Expression(calls[0].args[0])),'<actual_count_guard>','eval')

def main():
    """Check held source, private metadata cases and typed boundaries without any live operation."""
    started=time.monotonic();job=os.environ.get('SLURM_JOB_ID');step=os.environ.get('SLURM_STEP_ID')
    need(job=='6825983' and step and step.isdecimal(),'exact current owned CPU allocation and numeric step')
    need('/job_'+job+'/step_'+step+'/' in Path('/proc/self/cgroup').read_text(),'exact current owned CPU cgroup')
    need(sha(FREEZE)==EXPECTED,'the held final three-file source freeze is exact')
    freeze=json.loads(FREEZE.read_bytes())
    def verify_files():
        """Require all frozen bytes, lengths and modes before and after the check."""
        for relative,row in freeze['files'].items():
            path=W/relative
            need(sha(path)==row['sha256'] and path.stat().st_size==row['size_bytes'] and path.stat().st_mode&0o777==row['mode'],
                'held file bytes and mode: '+relative)
    verify_files()
    operations=W/'.review_metadata/profile_target_operational_preparation'
    before={};after={}
    for old,new in (('start_direct_profile_qualification_once.py','start_direct_two_copy_profile_qualification_once.py'),
            ('prepare_direct_transport_configuration.py','prepare_direct_two_copy_transport_configuration.py')):
        reference=freeze['one_copy_sources'][old];path=Path(reference['path'])
        need(sha(path)==reference['sha256'],'actual held one-copy source reference: '+old)
        before[old]=path.read_text();after[old]=(operations/new).read_text()
    original_guard="    original = document(parent['original_qualification_plan'])\n    require(type(original['configuration']['learner']['n_copies']) is int\n        and original['configuration']['learner']['n_copies'] == 1\n        and original['copy_seeds'] == [1] and type(original['copy_seeds'][0]) is int,\n        'the separate original qualification reference must retain exactly one copy and seed one')\n"
    for name,new in after.items():
        normalized=new.replace('start_direct_two_copy_profile_qualification_once.py','start_direct_profile_qualification_once.py')
        block=original_guard if name.startswith('start_') else original_guard.replace('document(', 'transport.document(').replace('    require(', '    transport.require(')
        need(normalized.count(block)==1,'one explicit original-reference-one-copy guard')
        normalized=normalized.replace(block,'')
        normalized=normalized.replace("child['copy_seeds'] == [1, 2]","child['copy_seeds'] == [1]")
        normalized=normalized.replace("and all(type(seed) is int for seed in child['copy_seeds'])\n        and type(child['configuration']['learner']['n_copies']) is int and child['configuration']['learner']['n_copies'] == 2",
            "and type(child['copy_seeds'][0]) is int and child['configuration']['learner']['n_copies'] == 1")
        normalized=normalized.replace('reviewed two-copy H100','reviewed one-copy H100')
        need(normalized==before[name],'all source bytes beyond explicit count/seed/reference/filename changes remain exact: '+name)
    transport_source=after['start_direct_profile_qualification_once.py']
    count=expression(transport_source,'verify','the reviewed two-copy H100 controller/capture plans changed')
    original=expression(transport_source,'verify','the separate original qualification reference must retain exactly one copy and seed one')
    cases=0
    with tempfile.TemporaryDirectory() as temporary:
        run=Path(temporary);controller=run/'data/controller';controller.mkdir(parents=True)
        parent={'unit_id':'controller','kind':'profile_target_qualification_controller','profile_unit':str(run/'data/capture'),'campaign_run_dir':str(run),'source_commit':SOURCE}
        child={'unit_id':'capture','kind':'full_original_profile_source_target_qualification','source_commit':SOURCE,'hardware_class':'h100',
            'copy_seeds':[1,2],'configuration':{'learner':{'n_copies':2}}}
        context={'controller':controller,'R':run,'SOURCE':SOURCE,'parent':parent,'child':child,'record':{'controller_plan':{'path':str(controller/'measurement_plan.json')}}}
        for copies,seeds,expected in ((2,[1,2],True),(1,[1],False),(2.0,[1,2],False),(True,[1,2],False),
                (2,[True,2],False),(2,[1.0,2],False),(2,[1,1],False),(2,[2,1],False),(2,[1,2,3],False)):
            current=copy.deepcopy(context);current['child']['configuration']['learner']['n_copies']=copies;current['child']['copy_seeds']=seeds
            need(eval(count,current) is expected,'strict actual transport count/seed predicate: '+repr((copies,seeds)));cases+=1
        for copies,seeds,expected in ((1,[1],True),(True,[1],False),(1.0,[1],False),(2,[1],False),(1,[True],False),(1,[1.0],False),(1,[2],False)):
            need(eval(original,{'original':{'configuration':{'learner':{'n_copies':copies}},'copy_seeds':seeds}}) is expected,
                'strict actual original-reference predicate: '+repr((copies,seeds)));cases+=1
    sys.addaudithook(audit)
    spec=importlib.util.spec_from_file_location('independent_actual_count2_metadata_tests',D/'test_two_copy_h100_transport.py')
    tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)
    output=io.StringIO();result=unittest.TextTestRunner(stream=output,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
    with (D/'independent_final_count2_tests.stdout.raw').open('x') as stream:stream.write(output.getvalue());stream.flush();os.fsync(stream.fileno())
    need(result.wasSuccessful() and result.testsRun==4,'four independently executed focused metadata methods pass')
    verify_files();need(time.monotonic()-started<285,'whole independent metadata review below five minutes')
    record={'kind':'independent_two_copy_profile_qualification_transport_source_review','approved':True,'passed':True,
        'independent_of_new_count_specific_changes_author':True,'independent_of_producer':True,
        'review_scope':'New count, seed, original-reference and entry filename changes only; unchanged custody and process algorithms reuse their prior independent approval.',
        'reviewer_previously_authored_custody_algorithm_and_one_copy_runtime_pin':True,
        'new_independent_algorithm_approval_claimed':False,
        'prior_independent_algorithm_review':{'path':"/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-05-02-24_common-timing_cache-room-backgrounds-true_render-final-two-only-true_h100-nvl_serval02-card-1_copies-1_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_paired-rounds-7_updates-per-arm-10_seed-start-1/independent_corrected_root_transport_retention_review.json",'sha256':'e0c1b534f43cff08dfa70c8c4c70bb1fb646720577d033c92fceda65f6684d38'},
        'source_freeze':{'path':str(FREEZE),'sha256':EXPECTED},'runtime_commit':freeze['actual_runtime_commit'],
        'one_copy_source_references':freeze['one_copy_sources'],'typed_predicate_cases':cases,'focused_methods':result.testsRun,
        'check_count':len(CHECKS),'checks':CHECKS,'actual_cpu_job_id':job,'actual_cpu_step_id':step,'hostname':socket.gethostname(),
        'seconds':time.monotonic()-started,'synthetic_metadata_premises':'Complete publication/review/Git fixtures are explicitly synthetic; no actual prepared packet is consumed.',
        'no_actual_packet_log_or_target_created':True,'scheduler_native_or_GPU_actions':0,'target_qualification':False,
        'scientific_credit':0,'admission_granted':False,'actual_prepared_transport_review_still_required':True}
    publish(D/'independent_final_count2_source_review.json',record)
    print(output.getvalue(),flush=True);print(json.dumps({'passed':True,'review':{'path':str(D/'independent_final_count2_source_review.json'),
        'sha256':sha(D/'independent_final_count2_source_review.json')},'step_id':step,'checks':len(CHECKS),'seconds':record['seconds']}),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as error:
        publish(D/'independent_final_count2_source_review_failure.json',{'passed':False,'error':repr(error),'traceback':traceback.format_exc(),
            'actual_cpu_job_id':os.environ.get('SLURM_JOB_ID'),'actual_cpu_step_id':os.environ.get('SLURM_STEP_ID'),'source_edited':False,'target_actions':0})
        raise
