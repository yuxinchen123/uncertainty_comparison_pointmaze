"""Check complete baseline metadata before the first output write; do not stage a unit.

Every accepted input and prerequisite validator is genuine. A sentinel prevents the
first requested output directory creation. Root performs real canonical staging later.
"""
import ctypes  # Standard-library own-process Python API bootstrap before the audit hook.
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

D=Path(__file__).resolve().parent
W=D.parents[1]
HELPER=Path('/p/rlprojects/RND_worktrees/2026-11-05-11-56_profile-qualification-helper_source-d1469fd063a4_h100-helpers-44_a4500-helpers-35_gpu-count-0')
R=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1')
FACT=R/'remote_setup/2026-11-05-11-58_common-performance-verification_hardware-types-2_training-copies-1-2_paired-rounds-7_updates-per-arm-10_cpu-count-1_gpu-count-0_time-limit-seconds-300'
BACKEND=next((R/'remote_setup').glob('2026-11-05-12-06_profile-backend-preparation*/profile_qualification_backend_expected.json'))
started=time.monotonic()


def require(value,message):
    if not value:raise ValueError(message)


def audit(event,args):
    if event=='import':require(args[0].split('.')[0] not in {'jax','jaxlib','numpy','torch','flax','optax','jaxtari','cupy','scipy'},'no numerical imports')
    if event in ('ctypes.dlopen','ctypes.dlsym','os.system','os.exec','os.fork') or event.startswith('socket.') and event!='socket.gethostname':raise ValueError('no native-library load, network or nonmetadata process')
    if event=='subprocess.Popen':require(isinstance(args[1],(list,tuple)) and args[1][0]=='git','only read-only source Git children')
    if event=='open' and isinstance(args[0],(str,bytes)):
        path=os.fsdecode(args[0]);require(not path.startswith('/dev/') or path=='/dev/null','no device access')


class BeforeOutput(Exception):pass


def check(copies,hardware):
    name='derive_one_copy_qualification_request.py' if copies==1 else 'derive_two_copy_qualification_request.py'
    source=W/'.review_metadata/profile_target_operational_preparation'/name
    spec=importlib.util.spec_from_file_location('baseline_contract_'+hardware+str(copies),source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    fact=json.loads((FACT/'four_context_prerequisite_fact.json').read_bytes())
    entry=next(row for row in fact['contexts'] if row['hardware_class']==hardware and row['copies']==copies)
    accepted=json.loads(Path(entry['review']['path']).read_bytes())
    common=json.loads(Path(accepted['plan']['path']).read_bytes())
    if hardware=='h100':
        native=json.loads((R/'next_actions.json').read_bytes())['native_direct_process_diagnostic_source']
        result=native['actual_target_result'];review=native['independent_actual_review']
    else:
        native=fact['native_process_qualification'];result=native['diagnostic_result'];review=native['independent_review']
    unit='metadata_contract_inspection_'+hardware+'_copies_'+str(copies)
    packet=R/'remote_setup'/unit
    require(not packet.exists(),'the nonwriting inspection path must be absent')
    args=type('Arguments',(),dict(helper_root=HELPER,run_dir=R,packet=packet,
        common_proof=FACT/'four_context_prerequisite_fact.json',common_review=FACT/'independent_actual_four_context_fact_review.json',
        component_result=Path(result['path']),component_review=Path(review['path']),backend_record=BACKEND,
        hardware_class=hardware,controller_unit_id=unit+'_controller',profile_unit_id=unit))()
    def stop(path,*unused,**ignored):
        require(path==packet,'an unexpected directory creation was requested')
        raise BeforeOutput('stop before any output')
    captured=None
    try:
        with patch.object(Path,'mkdir',stop):module.derive(args)
    except BeforeOutput as caught:
        trace=caught.__traceback__
        while trace:
            frame=trace.tb_frame
            if frame.f_code.co_name=='derive' and frame.f_globals.get('__file__')==str(source):captured=frame.f_locals
            trace=trace.tb_next
    require(captured is not None,'the actual metadata builder did not reach its first output boundary')
    baseline=captured['baseline'];request=captured['request']
    require(baseline['correctness_record']==common['common_correctness_inputs']['result']['path']
        and baseline['confirmation_record']==accepted['result']['path'],
        'the baseline must retain its genuine selected correctness and timing records')
    require(captured['common']['copy_seeds']==list(range(1,copies+1))
        and captured['common']['configuration']['learner']['n_copies']==copies,'selected count/seeds changed')
    require(captured['original']['configuration']['learner']['n_copies']==1
        and captured['original']['copy_seeds']==[1],'the original reference count changed')
    if hardware=='h100':
        argv=request['profile_prepare_arguments']
        require(argv.count('--direct-backend-image-record')==1 and argv[argv.index('--direct-backend-image-record')+1]==request['backend_image_record']['path'],'the parser presence argument changed')
        require(baseline['direct_binding_profile']==common['direct_binding']['profile'],'the direct profile changed')
    else:require(baseline['configured_isolation_profile']==common['configured_isolation']['profile'],'the configured profile changed')
    require(not packet.exists() and not (R/'data'/unit).exists() and not (R/'data'/(unit+'_controller')).exists(),'the inspection created an actual output')
    require(not {'jax','jaxlib','numpy','torch','flax','optax','jaxtari'}&sys.modules.keys(),'a numerical module was imported')
    return {'hardware_class':hardware,'copies':copies,'copy_seeds':common['copy_seeds'],'original_reference_copies':1,
        'complete_baseline_contract_passed':True,'correctness_record':baseline['correctness_record'],
        'confirmation_record':baseline['confirmation_record'],'genuine_prerequisite_validators_executed':True,
        'canonical_preparation_executed':False,'output_creation_prevented':True,'GPU_qualification_or_admission':False}


if __name__=='__main__':
    sys.addaudithook(audit)
    rows=[check(copies,hardware) for hardware in ('h100','jaguar03_rtx_a4500') for copies in (1,2)]
    print(json.dumps({'kind':'nonwriting_complete_profile_baseline_contract_regression','passed':True,'cases':rows,
        'seconds':time.monotonic()-started,'actual_output_directories_created':0,'canonical_staging_passed':False,
        'runtime_helpers_modified':False,'bootstrap':'Standard-library ctypes own-process Python API initialized before audit; every later native loader/symbol event, numerical import, GPU device and network operation is refused.'},indent=2))
