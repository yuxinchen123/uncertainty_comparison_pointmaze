import ast,datetime,hashlib,importlib.util,json,os,pathlib,stat,sys,tempfile,unittest
from unittest.mock import patch
D=pathlib.Path(__file__).resolve().parent;W=D.parent.parent;S=W/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
checks=[]
def ck(v,s):
 if not v:raise AssertionError(s)
 checks.append(s)
def capture(p):
 st=p.lstat();ck(stat.S_ISREG(st.st_mode) and not p.is_symlink() and st.st_uid==2618919 and st.st_size<1024*1024,'owned bounded '+p.name);raw=p.read_bytes();ck(p.stat().st_size==len(raw),'stable '+p.name);return raw
fraw=capture(D/'source_freeze_4.json');ck(hashlib.sha256(fraw).hexdigest()=='ea60d48221e71308de9c2d8db723d512cd60b502130ad9b5f8e81ed57bddaafb','exact held freeze4');freeze=json.loads(fraw)
def verify_source():
 for name,item in freeze['paths'].items():
  p=W/name;ck(hashlib.sha256(capture(p)).hexdigest()==item['sha256'] and stat.S_IMODE(p.stat().st_mode)==item['mode'],'actual frozen source '+name)
verify_source();ck(os.environ.get('SLURM_JOB_ID')=='6821437','exact current owned CPU allocation')
# These are pure metadata fixtures. No external process or native library may be loaded.
def guard(event,args):
 if event in {'subprocess.Popen','os.system','os.posix_spawn','ctypes.dlopen','ctypes.dlsym','socket.connect'}:raise RuntimeError('independent metadata review forbids '+event)
 if event=='import' and args[0].split('.')[0] in {'jax','jaxlib','numpy','torch','cupy'}:raise RuntimeError('numerical import refused')
sys.addaudithook(guard)
spec=importlib.util.spec_from_file_location('independent_profile_prerequisite_tests',S/'tests/test_configured_profile_prerequisites.py');tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)
suite=unittest.defaultTestLoader.loadTestsFromModule(tests);result=unittest.TextTestRunner(verbosity=2).run(suite);ck(result.wasSuccessful() and result.testsRun==13,'all thirteen held metadata tests pass')
helper=tests.HELPER;profile=tests.PROFILE
with tempfile.TemporaryDirectory() as temp:
 root=pathlib.Path(temp);small=root/'small.json';small.write_text('{}');large=root/'too_large.json'
 with large.open('wb') as f:f.truncate(helper.MAXIMUM_METADATA_BYTES+1)
 for entry in ['confirmation_metadata','configure_plan']:
  for placement in ['proof','review']:
   with patch.object(helper.os,'open',side_effect=AssertionError('oversized metadata must not be opened')) as opened:
    # A valid proof may be opened when the oversized file is the review, so force the first reference result to skip that unrelated read.
    if placement=='review':
     original=helper.MetadataFiles.reference
     def reference(self,path):
      if pathlib.Path(path)==small:return {'path':str(small),'sha256':hashlib.sha256(b'{}').hexdigest()}
      return original(self,path)
     manager=patch.object(helper.MetadataFiles,'reference',reference)
    else:manager=patch.object(helper.MetadataFiles,'reference',helper.MetadataFiles.reference)
    with manager:
     try:
      if entry=='confirmation_metadata':
       if placement=='review':
        with patch.object(helper.MetadataFiles,'document',return_value={}):helper.confirmation_metadata(small,large,'jaguar03_rtx_a4500',1)
       else:helper.confirmation_metadata(large,small,'jaguar03_rtx_a4500',1)
      else:helper.configure_plan({},large if placement=='proof' else small,large if placement=='review' else small)
     except ValueError as e:ck('bounded ordinary' in str(e),'oversized '+placement+' refused before open in '+entry)
     else:raise AssertionError('oversized input accepted')
   ck(opened.call_count==0,'no payload open for oversized '+entry+'/'+placement)
 sym=root/'linked.json';sym.symlink_to(small)
 with patch.object(helper.os,'open',side_effect=AssertionError('symlink input must not be opened')) as opened:
  try:helper.MetadataFiles().reference(sym)
  except ValueError:pass
  else:raise AssertionError('symlink accepted')
 ck(opened.call_count==0,'symlink refused before file open')
# Preserve actual loaded-library, context and numerical capture contracts as source, without invoking them.
old=D/'preserved_third_source_before_independent_corrections/source/10_jax_exploration_platform/scripts/atari_pixel_campaign/profile_selected.py'
def functions(p):return {x.name:ast.dump(x,include_attributes=False) for x in ast.parse(capture(p)) .body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef))}
a,b=functions(old),functions(S/'profile_selected.py')
for name in ['_capture_core','capture_direct_owned_processes','run_direct_owned_waiter','capture_direct_owned_child','direct_profile_reader','direct_process_reader','require_reviewed_profile_process_admission','verify_prepared']:
 ck(a[name]==b[name],'unchanged capture/direct/execution source '+name)
node=next(x for x in ast.parse(capture(S/'profile_selected.py')).body if isinstance(x,ast.FunctionDef) and x.name=='run_configured_native') if False else None
# Inspect the actual native producer order using literal call locations.
source=(S/'profile_selected.py').read_text();start=source.index('            qualified = verify_profile_target_qualification(plan)');backend=source.index('            installed, backend_validation = verify_current_profile_backend',start);tool=source.index('            with native.tool_scope',start)
ck(start<backend<tool,'exact original qualification metadata and current backend join precede first tool scope')
ck('reader.validate(plan)\n        return reader.qualification_metadata(plan)' in source,'pure prerequisite and exact original qualification both required')
ck('support.verify_qualification_inputs(plan)' in source,'legacy backend verification preserved')
ck(not any(x.split('.')[0] in {'jax','jaxlib','numpy','torch','cupy'} for x in sys.modules),'no numerical runtime imported')
verify_source()
x={'schema_version':1,'kind':'independent_configured_profile_preparation_source_review','approved':True,'passed':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'allocation':os.environ.get('SLURM_JOB_ID'),'step':os.environ.get('SLURM_STEP_ID'),'freeze':{'path':str(D/'source_freeze_4.json'),'sha256':hashlib.sha256(fraw).hexdigest()},'checks':checks,'check_count':len(checks),'methods':result.testsRun,'initial_findings_preserved':'5727fcf1299725db1032cf40b6a38a8968d895254613011394d77f3fbcafd986','bounded_reference_before_hash_verified':True,'current_package_source_and_library_metadata_join_verified':True,'all_four_actual_common_contexts_required':True,'current_h100_two_copy_context_is_not_accepted':True,'execution_admission_granted':False,'actual_profile_preparation_or_tool_operations':0,'scientific_credit':0,'scope':'Pure metadata preparation and exact current-software checks only. Operation remains unconditionally closed; actual target qualification and a separate reviewed admission remain required.'}
out=D/'independent_corrected_preparation_review.json'
with out.open('x') as f:json.dump(x,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps({'passed':True,'checks':len(checks),'methods':result.testsRun,'review':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'step':os.environ.get('SLURM_STEP_ID')}))
