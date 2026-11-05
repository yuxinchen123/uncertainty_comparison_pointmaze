"""Complete the preserved incoming review using Git features installed on this cluster."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

OUTPUT = Path(__file__).resolve().parent
specification = importlib.util.spec_from_file_location('incoming_review_functions',OUTPUT/'review_incoming.py')
review = importlib.util.module_from_spec(specification)
specification.loader.exec_module(review)


def main():
    """Resolve only the declared shared source in a private index and check the resulting tree."""
    assert os.environ['SLURM_JOB_ID']=='6783003' and os.environ['SLURM_STEP_ID'].isdigit()
    assert review.git(['rev-parse','HEAD']).stdout.decode().strip()==review.CURRENT
    index = Path(review.git(['rev-parse','--git-path','index']).stdout.decode().strip())
    if not index.is_absolute():
        index=review.INTEGRATION/index
    index_before=review.reference(index)
    roster_path=review.WORKTREE/'.review_metadata/direct_common_historical_consumer/reviewed_publication_roster.json'
    roster=json.loads(roster_path.read_bytes())
    incoming_paths=list(roster['publication_files'])+[str(roster_path.relative_to(review.WORKTREE))]
    assert len(incoming_paths)==196
    incoming=review.files(review.INCOMING,incoming_paths)
    for name,row in roster['publication_files'].items():
        entry=incoming[name]
        assert entry['mode']==row['git_mode'] and review.digest(entry['raw'])==row['sha256'] and len(entry['raw'])==row['bytes']
    assert incoming[str(roster_path.relative_to(review.WORKTREE))]['raw']==roster_path.read_bytes()
    collisions={}
    for name,arguments in [('dirty',['diff','--name-only','-z','HEAD']),('untracked',['ls-files','--others','--exclude-standard','-z']),
            ('ignored',['ls-files','--others','--ignored','--exclude-standard','-z'])]:
        raw=review.git([*arguments,'--',*incoming_paths]).stdout
        collisions[name]=[item for item in raw.decode().split('\0') if item]
        assert not collisions[name],collisions
    for name in incoming_paths:
        target=review.INTEGRATION/name
        assert not target.is_dir()
        assert all(not parent.exists() or parent.is_dir() for parent in target.parents)
    current=review.files(review.CURRENT,[review.PREFIX])
    base=review.files(review.BASE,[review.SHARED])[review.SHARED]
    with tempfile.TemporaryDirectory(prefix='incoming_review_private_git_') as directory:
        temporary=Path(directory)
        environment=dict(os.environ,GIT_INDEX_FILE=str(temporary/'three-way-index'))
        review.git(['read-tree','-m',review.BASE,review.CURRENT,review.INCOMING],env=environment)
        unmerged=review.git(['ls-files','--unmerged','-z'],env=environment).stdout
        rows=[row for row in unmerged.split(b'\0') if row]
        assert len(rows)==3 and all(row.split(b'\t')[1].decode()==review.SHARED for row in rows)
        for name,raw in [('current.py',current[review.SHARED]['raw']),('base.py',base['raw']),('incoming.py',incoming[review.SHARED]['raw'])]:
            (temporary/name).write_bytes(raw)
        command=['git','merge-file','-p',str(temporary/'current.py'),str(temporary/'base.py'),str(temporary/'incoming.py')]
        result=subprocess.run(command,capture_output=True)
        review.publish('supported_source_merge_return.json',{'command':command,'returncode':result.returncode,
            'stdout_sha256':review.digest(result.stdout),'stderr':result.stderr.decode()})
        assert result.returncode==0
        object_id=review.git(['hash-object','-w','--stdin'],input=result.stdout).stdout.decode().strip()
        review.git(['update-index','--add','--cacheinfo','100644',object_id,review.SHARED],env=environment)
        assert not review.git(['ls-files','--unmerged','-z'],env=environment).stdout
        predicted=review.git(['write-tree'],env=environment).stdout.decode().strip()
        for count in range(2):
            path=temporary/('actual-index-copy-'+str(count))
            shutil.copyfile(index,path)
            review.git(['read-tree','-m','-u','--dry-run',review.CURRENT,predicted],env=dict(os.environ,GIT_INDEX_FILE=str(path)))
        predicted_incoming=review.files(predicted,incoming_paths)
        for name,entry in incoming.items():
            assert predicted_incoming[name]['mode']==entry['mode']
            if name!=review.SHARED:
                assert predicted_incoming[name]['object_id']==entry['object_id']
        changes=set(review.git(['diff','--name-only','-z',review.CURRENT,predicted]).stdout.decode().strip('\0').split('\0'))
        assert changes==set(incoming_paths)
        predicted_files=review.files(predicted,[review.PREFIX])
        for name,entry in current.items():
            if name not in incoming_paths:
                assert predicted_files[name]['mode']==entry['mode'] and predicted_files[name]['object_id']==entry['object_id']
        old_functions=review.syntax(current[review.SHARED]['raw']);new_functions=review.syntax(predicted_files[review.SHARED]['raw'])
        incoming_functions=review.syntax(incoming[review.SHARED]['raw'])
        assert all(new_functions[name]==value for name,value in old_functions.items() if name!='verify_qualification_inputs')
        assert all(new_functions[name]==incoming_functions[name] for name in ('historical_support','verify_qualification_inputs'))
        for filename,names in [('profile_selected.py',('_capture_core',)),('measure_selected.py',('_compute','run_prepared','save_measurement_result','producer_support','card_admission'))]:
            assert all(review.syntax(current[review.PREFIX+filename]['raw'])[name]==review.syntax(predicted_files[review.PREFIX+filename]['raw'])[name] for name in names)
        exported=temporary/'exported'
        for name,entry in predicted_files.items():
            if name.endswith('.py'):
                path=exported/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(entry['raw'])
        code="""import sys,json,ctypes
from pathlib import Path
forbidden={'jax','jaxlib','numpy','torch','flax','optax','cupy','scipy','exploration_platform','jaxtari','nvidia','jax_cuda12_plugin','ml_dtypes'}
def audit(event,values):
    if event=='import' and values[0].split('.')[0] in forbidden: raise RuntimeError('numerical import refused')
    if event in {'ctypes.dlopen','ctypes.dlsym','socket.connect','socket.bind','socket.getaddrinfo','subprocess.Popen'}: raise RuntimeError('native, network or child operation refused')
sys.addaudithook(audit)
sys.path.insert(0,sys.argv[1])
import measure_selected as m
import profile_selected as p
common=m.required_helpers(direct=True,common=True,backend_image=True)
profile=p.required_helpers(direct=True,direct_profile=True)
assert len(common)==28 and len(profile)==32
assert all((Path(sys.argv[1])/name).is_file() for name in set(common)|set(profile))
assert not forbidden.intersection(sys.modules)
print(json.dumps({'common':list(common),'profile':list(profile),'numerical_imports':[],'stdlib_own_process_ctypes_bootstrap_before_audit':True}))
"""
        cold=subprocess.run(['/usr/bin/python3','-I','-S','-B','-c',code,str(exported/review.PREFIX)],capture_output=True)
        review.publish('supported_merged_cold_import.json',{'returncode':cold.returncode,'stdout':cold.stdout.decode(),'stderr':cold.stderr.decode()})
        assert cold.returncode==0 and cold.stderr==b''
    canonical_path=review.RUN/'remote_setup/direct_nsight_incoming_independent_2026-11-04-20-01-56_protected_before.json'
    canonical=json.loads(canonical_path.read_bytes());after=review.protected_snapshot(canonical)
    before=json.loads((OUTPUT/'protected_before.json').read_bytes())['files']
    assert before==after and len(after)==662
    assert review.reference(index)==index_before and review.git(['rev-parse','HEAD']).stdout.decode().strip()==review.CURRENT
    receipt={'schema_version':1,'kind':'independent_incoming_historical_consumer_merge_review','passed':True,
        'reviewer':'/root/review_historical_consumer_bridge','actual_job':'6783003','actual_step':os.environ['SLURM_STEP_ID'],
        'current_head':review.CURRENT,'incoming_commit':review.INCOMING,'incoming_tree':'a7fc055afe3fb3743a3e7d2003e28a6bb3229e3b','predicted_tree':predicted,
        'incoming_blobs_modes_and_roster_verified':196,'source_freeze':roster['source_freeze'],'genuine_source_approval':roster['genuine_independent_approval'],
        'fresh_scoped_dirty_untracked_ignored_collisions':collisions,'ancestor_file_collisions':[],
        'actual_private_three_way_index_shared_source_stages':3,'shared_source_only':review.SHARED,
        'actual_fresh_real_index_copy_update_worktree_dry_runs_passed':2,'actual_index_before_and_after':index_before,
        'protected_source_files_unchanged':662,'protected_before':review.reference(OUTPUT/'protected_before.json'),
        'all_nonincoming_current_tree_paths_unchanged':True,'all_current_source_functions_preserved_except_exact_reviewed_qualification_dispatch':True,
        'current_direct_profile_source_and_process_api_source_preserved':True,'profile_capture_core_and_numerical_functions_preserved':True,
        'merged_common_helpers':28,'merged_current_profile_helpers':32,'merged_cold_metadata_import_passed':True,
        'future_unadmitted_profile_gap':'direct_profile_prerequisites.py lines222 and389 require historical confirmation helpers27. New common path uses28; separately review exact historical consumer and dependencies before ANY direct-profile preparation/admission. Current profile32 remains unadmitted; no invented28-helper confirmation evidence.',
        'preserved_review_setup_failures':['failed_attempt.json','failed_attempt_compatible_git.json'],
        'commands':review.COMMANDS,'saved_scientific_evidence_recomputed':False,'production_guard_success_established':False,
        'scientific_credit':0,'admission_granted':False,'fresh_disjoint_paths_head_index_and_collision_recheck_required_before_actual_merge':True}
    review.publish('independent_supported_incoming_review.json',receipt)
    print(json.dumps(review.reference(OUTPUT/'independent_supported_incoming_review.json'),sort_keys=True))


if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        review.publish('supported_resume_failure.json',{'type':type(error).__name__,'message':str(error),'commands':review.COMMANDS})
        raise
