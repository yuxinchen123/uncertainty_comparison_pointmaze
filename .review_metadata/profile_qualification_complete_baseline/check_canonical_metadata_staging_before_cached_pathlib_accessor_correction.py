"""Stage actual qualification metadata with only fresh output paths redirected privately.

All runtime functions, historical readers, facts, reviews, source and backend inputs
are genuine and unchanged. The resulting private files have no GPU or admission credit.
"""
import builtins
import ctypes  # Standard-library own-process Python symbol-table bootstrap before the audit hook.
from contextlib import contextmanager
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

W=Path(__file__).resolve().parents[2]
D=Path(__file__).resolve().parent
HELPER=Path('/p/rlprojects/RND_worktrees/2026-11-05-11-56_profile-qualification-helper_source-d1469fd063a4_h100-helpers-44_a4500-helpers-35_gpu-count-0')
R=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1')
S='10_jax_exploration_platform/scripts/atari_pixel_campaign/'
FACT=R/'remote_setup/2026-11-05-11-58_common-performance-verification_hardware-types-2_training-copies-1-2_paired-rounds-7_updates-per-arm-10_cpu-count-1_gpu-count-0_time-limit-seconds-300'
BACKEND=next((R/'remote_setup').glob('2026-11-05-12-06_profile-backend-preparation*/profile_qualification_backend_expected.json'))
started=time.monotonic()


def require(value,message):
    if not value:raise ValueError(message)


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


@contextmanager
def output_locations(mapping,private):
    """Map only fresh output prefixes; leave every historical/source input unchanged."""
    originals={name:getattr(os,name) for name in ('open','stat','lstat','mkdir','unlink','rmdir','rename','replace','chmod','utime','scandir','listdir','readlink')}
    real_io,real_builtin,real_resolve=io.open,builtins.open,Path.resolve
    pairs=[(str(logical),str(physical)) for logical,physical in mapping.items()]
    def inverse(value):
        value=os.fsdecode(value)
        for logical,physical in pairs:
            if value==physical or value.startswith(physical+os.sep):return logical+value[len(physical):]
        return value
    def mapped(value,dir_fd=None):
        if isinstance(value,int):return value,False
        value=os.fsdecode(value)
        if not os.path.isabs(value):
            base=inverse(originals['readlink']('/proc/self/fd/'+str(dir_fd))) if dir_fd is not None else os.getcwd()
            value=os.path.normpath(os.path.join(base,value))
        for logical,physical in pairs:
            if value==logical or value.startswith(logical+os.sep):return physical+value[len(logical):],True
        return value,False
    def write_allowed(path):
        if isinstance(path,int):
            path=originals['readlink']('/proc/self/fd/'+str(path))
            if path.startswith('pipe:['):return  # Standard-library stdin pipe for an allowed read-only Git child.
        require(isinstance(path,str) and (path==str(private) or path.startswith(str(private)+os.sep)),
            'the staging fixture refuses a write outside its private output directory')
    def opened(path,flags,mode=0o777,*,dir_fd=None):
        actual,changed=mapped(path,dir_fd)
        if flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):write_allowed(actual)
        return originals['open'](actual,flags,mode,**({} if changed or os.path.isabs(os.fspath(actual)) else {'dir_fd':dir_fd}))
    def stated(path,*,dir_fd=None,follow_symlinks=True):
        actual,changed=mapped(path,dir_fd)
        return originals['stat'](actual,follow_symlinks=follow_symlinks)
    def lstated(path,*,dir_fd=None):return stated(path,dir_fd=dir_fd,follow_symlinks=False)
    def simple(name):
        def call(path,*args,**kwargs):
            actual,changed=mapped(path,kwargs.pop('dir_fd',None))
            if name in ('mkdir','unlink','rmdir','chmod','utime'):write_allowed(actual)
            return originals[name](actual,*args,**kwargs)
        return call
    def renamed(name):
        def call(src,dst,*args,**kwargs):
            a,_=mapped(src,kwargs.pop('src_dir_fd',None));b,_=mapped(dst,kwargs.pop('dst_dir_fd',None))
            write_allowed(a);write_allowed(b);return originals[name](a,b,*args,**kwargs)
        return call
    def file_open(real):
        def call(path,mode='r',*args,**kwargs):
            actual,_=mapped(path)
            if any(letter in mode for letter in 'wax+'):write_allowed(actual)
            return real(actual,mode,*args,**kwargs)
        return call
    def resolve(path,*args,**kwargs):
        actual,changed=mapped(path)
        result=real_resolve(Path(actual),*args,**kwargs)
        return Path(inverse(str(result))) if changed else result
    def readlink(path,*args,**kwargs):
        actual,_=mapped(path,kwargs.pop('dir_fd',None));value=originals['readlink'](actual,*args,**kwargs)
        return inverse(value)
    replacements={'open':opened,'stat':stated,'lstat':lstated,'rename':renamed('rename'),'replace':renamed('replace'),'readlink':readlink}
    replacements.update({name:simple(name) for name in ('mkdir','unlink','rmdir','chmod','utime','scandir','listdir')})
    with patch.multiple(os,**replacements),patch.object(io,'open',file_open(real_io)),\
            patch.object(builtins,'open',file_open(real_builtin)),patch.object(Path,'resolve',resolve),\
            patch.dict(os.environ,{'GIT_OPTIONAL_LOCKS':'0'}):
        yield


def audit(event,args):
    if event=='import':require(args[0].split('.')[0] not in {'jax','jaxlib','numpy','torch','flax','optax','jaxtari','cupy','scipy'},'no numerical imports in metadata staging')
    if event in ('ctypes.dlopen','ctypes.dlsym','os.system','os.exec','os.fork') or event.startswith('socket.') and event!='socket.gethostname':
        raise ValueError('no native-library, network or nonmetadata process operation')
    if event=='subprocess.Popen':
        argv=args[1]
        require(isinstance(argv,(list,tuple)) and argv[0]=='git','only read-only Git metadata children are allowed')
    if event=='open' and isinstance(args[0],(str,bytes)):
        path=os.fsdecode(args[0]);require(not path.startswith('/dev/') or path=='/dev/null','no device access')


def run_case(copies,hardware):
    deriver=load(W/'.review_metadata/profile_target_operational_preparation'/('derive_one_copy_qualification_request.py' if copies==1 else 'derive_two_copy_qualification_request.py'),'actual_count_'+str(copies)+'_'+hardware)
    q=deriver.runtime(HELPER)
    n=json.loads((R/'next_actions.json').read_bytes())
    if hardware=='h100':component=n['native_direct_process_diagnostic_source'];result=component['actual_target_result'];review=component['independent_actual_review']
    else:
        proof=json.loads((FACT/'four_context_prerequisite_fact.json').read_bytes());component=proof['native_process_qualification'];result=component['diagnostic_result'];review=component['independent_review']
    label='private_metadata_fixture_'+hardware+'_copies_'+str(copies)
    packet=R/'remote_setup'/label;parent=R/'data'/(label+'_controller');child=R/'data'/label
    require(not any(path.exists() for path in (packet,parent,child)),'fixture destinations must never be real attempted units')
    with tempfile.TemporaryDirectory(prefix='profile-metadata-fixture-') as temporary:
        private=Path(temporary)
        mapping={packet:private/'packet',parent:private/'parent',child:private/'child'}
        args=type('Arguments',(),dict(helper_root=HELPER,run_dir=R,packet=packet,
            common_proof=FACT/'four_context_prerequisite_fact.json',common_review=FACT/'independent_actual_four_context_fact_review.json',
            component_result=Path(result['path']),component_review=Path(review['path']),backend_record=BACKEND,
            hardware_class=hardware,controller_unit_id=parent.name,profile_unit_id=child.name))()
        with output_locations(mapping,private):
            derived=deriver.derive(args)
            request=q.document(packet/'preparation_request.json')
            baseline=q.document(packet/'accepted_baseline.json')
            require(Path(baseline['correctness_record']).is_file(),'the actual accepted correctness pointer is required')
            returned=q.prepare(packet/'preparation_request.json')
            require(returned==parent,'the canonical preparer returned another parent')
            parent_plan=q.document(parent/'measurement_plan.json');child_plan=q.document(child/'measurement_plan.json')
            require(parent_plan['target_fields']['configuration']['learner']['n_copies']==copies
                and child_plan['copy_seeds']==list(range(1,copies+1)), 'count-specific actual staged state changed')
            require(q.checked(parent_plan['original_qualification_plan'])['configuration']['learner']['n_copies']==1,
                'the unchanged original reference must retain one copy')
            count=44 if hardware=='h100' else 35
            require(len(child_plan['helpers']['files'])==count,'the runtime helper closure changed')
            for relative,sha in child_plan['helpers']['files'].items():
                require(hashlib.sha256((child/'code'/Path(relative).name).read_bytes()).hexdigest()==sha,'staged helper bytes changed')
            require((parent/'prepared_child_binding.json').is_file(),'the full canonical parent/child binding is missing')
            for directory in (packet,parent,child):
                require(all((directory/name).is_file() for name in ('manifest.yaml','config_resolved.yaml','command.txt','experiment_background.md')),'complete standard records missing')
            record={'hardware_class':hardware,'copies':copies,'copy_seeds':child_plan['copy_seeds'],'reference_copies':1,
                'helper_count':count,'full_canonical_preparation_passed':True,'all_runtime_validators_unchanged':True,
                'only_fresh_output_paths_redirected':True,'correctness_record':baseline['correctness_record'],
                'confirmation_record':baseline['confirmation_record'],'real_campaign_units_created':False,
                'GPU_qualification_established':False,'admission_granted':False}
        require(not any(path.exists() for path in (packet,parent,child)),'the fixture wrote a real campaign output')
        return record


if __name__=='__main__':
    sys.addaudithook(audit)
    records=[]
    for hardware,copies in [('h100',1),('h100',2),('jaguar03_rtx_a4500',2)]:
        records.append(run_case(copies,hardware))
    output={'kind':'full_canonical_profile_qualification_metadata_staging_fixture','passed':True,
        'cases':records,'seconds':time.monotonic()-started,'GPU_or_numerical_operations':0,
        'actual_campaign_units_created':False,'bootstrap':'Before the audit hook, standard-library ctypes initializes only its own-process Python API handle with dlopen(None). All subsequent native-library loads/symbol calls are refused; no numerical runtime, GPU device or profiling tool is called.'}
    print(json.dumps(output,indent=2))
