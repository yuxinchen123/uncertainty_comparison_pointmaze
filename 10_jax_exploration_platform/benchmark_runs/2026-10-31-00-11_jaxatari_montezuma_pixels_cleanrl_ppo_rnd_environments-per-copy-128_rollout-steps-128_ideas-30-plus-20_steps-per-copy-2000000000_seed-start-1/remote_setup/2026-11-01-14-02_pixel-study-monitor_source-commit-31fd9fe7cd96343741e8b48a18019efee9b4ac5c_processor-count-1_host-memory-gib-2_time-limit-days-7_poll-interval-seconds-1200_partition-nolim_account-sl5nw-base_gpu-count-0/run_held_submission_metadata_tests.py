"""Run only submission-control metadata tests on the already owned monitor allocation."""
import ctypes,datetime,hashlib,importlib.abc,json,os,pathlib,subprocess,sys,unittest
from unittest.mock import patch
P=pathlib.Path(__file__).resolve().parent
assert os.environ.get("SLURM_JOB_ID")=="6781231" and os.environ.get("SLURM_STEP_ID","").isdecimal()
freeze=json.loads((P/"held_submission_source_freeze.json").read_bytes())
images=lambda:{n:hashlib.sha256(pathlib.Path(n).read_bytes()).hexdigest() for n in freeze["files"]}
before=images();assert before==freeze["files"];blocked=[]
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split(".")[0] in {"jax","jaxlib","numpy","torch","flax","optax","cupy","scipy"}:blocked.append(name);raise RuntimeError("numerical import forbidden")
sys.meta_path.insert(0,Block())
ctypes.CDLL=lambda *a,**k:(_ for _ in ()).throw(RuntimeError("shared libraries forbidden"))
subprocess.Popen=lambda *a,**k:(_ for _ in ()).throw(RuntimeError("all live processes forbidden"))
sys.path.insert(0,str(P))
import test_submit_monitor_held as tests
suite=unittest.defaultTestLoader.loadTestsFromModule(tests)
result=unittest.TextTestRunner(verbosity=2).run(suite)
module=tests.submit
inputs,plan=module.validate_inputs()
repo=pathlib.Path(inputs["repository"]);show=[]
def git_blob(command):
 assert command[:3]==["git","-C",str(repo)] and command[3]=="show"
 relative=command[4].split(":",1)[1];show.append(relative);return (repo/relative).read_bytes()
with patch.object(module.subprocess,"run") as branch,patch.object(module.subprocess,"check_output",side_effect=git_blob):
 module.validate_publication(inputs,"a"*40)
 branch.assert_called_once_with(["git","-C",str(repo),"merge-base","--is-ancestor","a"*40,"origin/merge-atari-cleanrl"],check=True)
 assert len(show)==len(inputs["required_files"])+4
with patch.object(module.subprocess,"run"),patch.object(module.subprocess,"check_output",return_value=b"wrong object"):
 try:module.validate_publication(inputs,"a"*40)
 except ValueError:pass
 else:raise AssertionError("changed published blob accepted")
after=images();assert after==before and not blocked
raw=pathlib.Path("/proc/self/stat").read_text();step=os.environ["SLURM_STEP_ID"]
v={"schema_version":1,"kind":"monitor_held_submission_author_metadata_verification","at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"job_id":"6781231","step_id":step,"pid":os.getpid(),"birth":raw[raw.rfind(")")+2:].split()[19],"cgroup":pathlib.Path("/proc/self/cgroup").read_text(),"tests":result.testsRun,"errors":len(result.errors),"failures":len(result.failures),"passed":result.wasSuccessful(),"failure_details":[{"test":t.id(),"traceback":s} for t,s in result.failures],"error_details":[{"test":t.id(),"traceback":s} for t,s in result.errors],"actual_packet_local_input_check":True,"synthetic_publication_full_object_check":True,"changed_publication_object_refused":True,"publication_object_checks":len(show),"actual_publication_claimed":False,"source_before":before,"source_after":after,"numerical_import_attempts":blocked,"live_commands":0,"new_jobs":0,"release_calls":0,"scientific_credit":0}
q=P/("held_submission_author_6781231_"+step+".json")
with q.open("x") as o:json.dump(v,o,sort_keys=True,indent=2);o.write("\n");o.flush();os.fsync(o.fileno())
print(json.dumps({"receipt":str(q),"sha256":hashlib.sha256(q.read_bytes()).hexdigest(),"tests":result.testsRun,"passed":result.wasSuccessful()}))
sys.exit(0 if result.wasSuccessful() else 1)

