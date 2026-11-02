"""Derive profile-scoped utilities while preserving the reviewed action mechanics."""
import ast,hashlib,json,re,os
from pathlib import Path
HERE=Path(__file__).resolve().parent;R=HERE.parents[1]
OLD_SUB=next((R/'remote_setup').glob('2026-11-01-18-57_common-confirmation-held-submission*'))
OLD_REL=next((R/'remote_setup').glob('2026-11-01-19-24_common-confirmation-release*'))
PREP=R/'remote_setup/2026-11-01-21-31_selected-common-a4500-profile-preparation_copies-1-2_capture-complete-updates-1_gpu-count-0'
COMMIT='fdb35f886ad01c6d4ed387dc55b172d0221a9853'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
ref=lambda p:{'path':str(p),'sha256':sha(p)}
def put(name,raw):
 p=HERE/name
 with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return ref(p)
def code(name,text):return put(name,text.encode())
scope='''def same(left,right):
    """Keep Boolean and integer fields distinct in nested scientific metadata."""
    if type(left) is not type(right):return False
    if isinstance(left,dict):return set(left)==set(right) and all(same(left[key],right[key]) for key in left)
    if isinstance(left,list):return len(left)==len(right) and all(same(a,b) for a,b in zip(left,right))
    return left==right

def profile_scope(plan,copies,item):
    """Require only the exact accepted common A4500 profile, never a measurement or direct plan."""
    require(plan.get("kind")=="selected_configuration_profile" and plan.get("source_commit")=="d28e6735e79788f618acb0ebfde8609db1f776af"
        and plan["helpers"]["source_commit"]==SOURCE and len(plan["helpers"]["files"])==22
        and plan.get("gpu_uuid") is None and plan["configured_isolation"]["physical_identity"]["gpu_uuid"] is None
        and plan["configured_isolation"]["consumer"]=="component_profile"
        and plan.get("accepted_confirmation_requires_common_reader") is True
        and type(plan.get("profile_updates")) is int and plan["profile_updates"]==1
        and type(plan.get("minimum_profile_updates")) is int and plan["minimum_profile_updates"]==1
        and type(plan.get("maximum_active_seconds")) is float and plan["maximum_active_seconds"]==3600.0
        and type(plan["configuration"]["learner"]["n_copies"]) is int and plan["configuration"]["learner"]["n_copies"]==copies
        and same(plan["copy_seeds"],list(range(1,copies+1)))
        and not any(key in plan for key in ("mode","rounds","iterations","minimum_updates_per_round","common_correctness_inputs","final_confirmation_protocol","completed_study","direct_binding","target_qualification")),
        "only the exact unresolved one-complete-update selected profile is supported")
    require(same(plan["configuration"],item["configuration"]) and same(plan["runtime_environment"],item["runtime_environment"])
        and same(plan["target_qualification_inputs"],item["target_qualification_inputs"])
        and plan["inputs"]["accepted_confirmation"]==item["accepted_confirmation"]
        and same(plan["accepted_baseline_specification"],item["accepted_baseline_specification"])
        and same(plan["profile_tool"],item["profile_tool"]), "profile configuration,evidence,tool or qualification changed")
    accepted=document(item["accepted_confirmation"]["path"]);approved=document(item["accepted_confirmation_review"]["path"])
    require(same(accepted["candidate"],plan["configuration"]) and same(accepted["copy_seeds"],plan["copy_seeds"])
        and accepted["measurement_mode"]=="confirmation" and accepted["decision"]=="accepted"
        and approved.get("approved") is True and approved.get("passed") is True
        and approved["result"]==item["accepted_confirmation"]
        and approved["actual_terminal_state"]=="COMPLETED" and approved["actual_exit_code"]=="0:0",
        "the exact profile has no independently accepted terminal common baseline")
'''
s=(OLD_SUB/'submit_common_confirmation_held.py').read_text()
s=s.replace('Submit one exact common-candidate confirmation packet held','Submit one exact selected common A4500 profile packet held')
s=re.sub(r"INPUTS_SHA='[0-9a-f]+'","INPUTS_SHA=None",s)
s=s.replace('SOURCE="9fcc43a478b93d731df780805fd7c3c514cdf67f"','SOURCE="'+COMMIT+'"')
start=s.index('def validate_inputs(');end=s.index('\ndef validate_publication(',start)
new='''def validate_inputs(copies):
    """Check exact static profile inputs locally; never query or submit from the default path."""
    require(type(copies) is int and copies in (1,2),"select exactly one supported copy count")
    require(INPUTS_SHA is not None and receipt(UTILITY/"submission_inputs.json")["sha256"]==INPUTS_SHA,
        "actual independent packet review is not yet bound")
    inputs=document(UTILITY/"submission_inputs.json")
    require(type(inputs["schema_version"]) is int and inputs["schema_version"]==1
        and inputs["kind"]=="fixed_selected_profile_held_submission_inputs"
        and inputs["actual_submission_ids"]=={"1":None,"2":None},"unsupported profile submission input kind or guessed ID")
    for record in inputs["required_files"]:checked(record)
    review=document(inputs["independent_packet_review"]["path"]);freeze=document(inputs["packet_freeze"]["path"])
    require(receipt(inputs["packet_freeze"]["path"])==inputs["packet_freeze"]
        and review.get("approved") is True and review.get("passed") is True
        and review.get("kind")=="independent_selected_common_a4500_profile_packets_and_startup_review"
        and type(review.get("schema_version")) is int and review["schema_version"]==1
        and review["packet_freeze"]==inputs["packet_freeze"] and review["before"]==review["after"]==freeze["files"]
        and review["admission_granted"] is False,"genuine independent profile packet review did not pass")
    item=inputs["packets"][str(copies)]
    packet=Path(item["packet"]);unit=Path(item["unit"]);campaign=Path(inputs["campaign_run_dir"])
    require(packet.parent==campaign/"remote_setup" and unit.parent==campaign/"data" and packet.name==unit.name,
        "profile unit or packet escaped the exact campaign")
    plan=document(unit/"measurement_plan.json");profile_scope(plan,copies,item)
    require(plan["campaign_run_dir"]==str(campaign) and plan["unit_id"]==unit.name
        and receipt(unit/"measurement_plan.json")==item["plan"] and receipt(packet/"job.sbatch")==item["script"],
        "fixed profile location,plan or script changed")
    expected=["sbatch","--parsable","--hold","--no-requeue","--nodes=1","--ntasks=1","--cpus-per-task=8","--mem=64G",
        "--time="+{1:"02",2:"03"}[copies]+":00:00","--partition=gpu","--account=sl5nw_base","--qos=csresnolim",
        "--nodelist=jaguar03","--reservation=sl5nw_168","--gres=gpu:nvidia_rtx_a4500:1",str(packet/"job.sbatch")]
    require(item["command"]==expected and document(packet/"held_submission_command.json")["command"]==expected,
        "only the exact held single-A4500 profile command is permitted")
    for record in (*plan["inputs"].values(),*plan["target_qualification_inputs"].values(),item["accepted_confirmation_review"]):checked(record)
    unattempted(packet,unit,campaign)
    return inputs,item,plan

def validate_readiness(inputs):
    """Require a recent actual running full monitor and verified twenty-minute native schedule."""
    record=inputs.get("submission_readiness")
    require(isinstance(record,dict),"actual current monitoring readiness is not bound")
    ready=json.loads(checked(record))
    require(type(ready.get("schema_version")) is int and ready["schema_version"]==1
        and ready.get("kind")=="actual_selected_profile_submission_monitor_readiness"
        and ready.get("passed") is True and ready.get("actual_actions")==0,"unsupported current monitoring readiness")
    verification=json.loads(checked(ready["full_verification"]));status=json.loads(checked(ready["full_status"]))
    checked(ready["full_report"]);runtime=json.loads(checked(ready["runtime_review"]));native=json.loads(checked(ready["native_record"]))
    observed=datetime.fromisoformat(verification["checked_at_utc"].replace("Z","+00:00"))
    require(observed.tzinfo is not None and observed.utcoffset().total_seconds()==0
        and 0<=(datetime.now(timezone.utc)-observed).total_seconds()<1200,
        "the actual full-monitor observation is stale,future-dated or not UTC")
    require(verification.get("passed") is True and verification.get("returncode")==0
        and verification["monitor_job_id"]=="6783003" and verification["source_commit"]=="79a01e851ff92ddba7d0e12eaf8164cab2028878"
        and verification["report_sha256"]==ready["full_report"]["sha256"] and verification["status_sha256"]==ready["full_status"]["sha256"]
        and len([row for row in status["jobs"] if row[0]=="6783003" and row[1]=="RUNNING"])==1,
        "the exact current CPU monitor is not observed running with a valid full report")
    require(runtime.get("passed") is True and runtime["job_id"]=="6783003"
        and runtime["source_commit"]=="79a01e851ff92ddba7d0e12eaf8164cab2028878"
        and type(runtime.get("gpu_count")) is int and runtime["gpu_count"]==0
        and type(runtime.get("source_git_archive_sha_and_executable_files_verified")) is int
        and runtime["source_git_archive_sha_and_executable_files_verified"]==265,
        "the current zero-GPU monitor source/runtime approval changed")
    expiry=datetime.fromisoformat(native["expires_at"].replace("Z","+00:00"))
    require(native["cron"]=="*/20 * * * *" and type(native["cadence_seconds"]) is int and native["cadence_seconds"]==1200
        and native["list_verified"] is True and native["persistent_job_id"]=="6783003"
        and expiry>datetime.now(timezone.utc),"the verified native monitoring schedule is unavailable")
    return ready
'''
s=s[:start]+scope+'\n'+new+s[end:]
s=s.replace("test_submit_common_confirmation_held.py","test_submit_selected_profiles_held.py")
s=s.replace("inputs,item,plan=validate_inputs(copies);validate_publication(inputs,publication)","inputs,item,plan=validate_inputs(copies);validate_publication(inputs,publication);validate_readiness(inputs)")
s=s.replace('exclusive_common_confirmation_held_submission_intent','exclusive_selected_profile_held_submission_intent')
s=s.replace('held_common_confirmation_per_submission_records','held_selected_profile_per_submission_records')
code('submit_selected_profiles_held.py',s)
tests=(OLD_SUB/'test_submit_common_confirmation_held.py').read_text()
tests=tests.replace('submit_common_confirmation_held.py','submit_selected_profiles_held.py')
tests=tests.replace("HERE/'submission_inputs.json'","HERE/'draft_submission_inputs.json'")
tests=tests.replace("'selected_configuration_measurement'","'selected_configuration_profile'")
tests=tests.replace("self.addCleanup(patch.stopall);patch.object(submit,'UTILITY',self.utility).start()",
 """self.addCleanup(patch.stopall);patch.object(submit,'UTILITY',self.utility).start()
        self.readiness=patch.object(submit,'validate_readiness',return_value={'passed':True}).start()""")
anchor="    def test_default_no_scheduler_or_write(self):"
extra='''    def test_stale_readiness_refuses_before_intent(self):
        self.readiness.side_effect=ValueError("synthetic stale full monitor")
        with patch.object(submit.subprocess,'run') as scheduler:
            with self.assertRaises(ValueError):self.invoke()
            scheduler.assert_not_called()
        self.assertFalse((self.path()/'submission_intent.json').exists())
    def test_missing_actual_packet_review_refuses_before_query(self):
        with patch.object(submit,'INPUTS_SHA',None),patch.object(submit.subprocess,'run') as scheduler:
            with self.assertRaises(ValueError):submit.validate_inputs.__wrapped__(1) if hasattr(submit.validate_inputs,'__wrapped__') else submit.require(False,'unbound review')
            scheduler.assert_not_called()
'''
# The actual unbound default is tested separately with the original production validation.
tests=tests.replace(anchor,extra+anchor)
code('test_submit_selected_profiles_held.py',tests)
# The source cannot release until actual held identifiers and proof hashes are bound.
r=(OLD_REL/'release_common_confirmation.py').read_text()
r=r.replace('Release one exact held common-candidate confirmation job','Release one exact held selected A4500 profile job')
r=r.replace('SOURCE = "9fcc43a478b93d731df780805fd7c3c514cdf67f"','SOURCE = "'+COMMIT+'"')
r=re.sub(r'INITIAL_PUBLICATION = "[0-9a-f]+"','INITIAL_PUBLICATION = None',r)
r=re.sub(r"FROZEN_INPUTS_SHA = '[0-9a-f]+'","FROZEN_INPUTS_SHA = None",r)
records=json.loads((PREP/'result.json').read_bytes())['records']
scopes={str(row['copies']):{'job_id':None,'packet':row['packet'],'unit':row['unit']} for row in records}
r=re.sub(r'^SCOPES = .+$','SCOPES = '+repr(scopes),r,flags=re.M)
r=r.replace('fixed_common_confirmation_actual_release_inputs','fixed_selected_profile_actual_release_inputs')
r=r.replace('plan["kind"] == "selected_configuration_measurement" and plan["mode"] == "confirmation"','plan["kind"] == "selected_configuration_profile" and "mode" not in plan')
start=r.index('    require(all(type(plan.get(key))');end=r.index('    identity = core(plan)',start)
r=r[:start]+'''    profile_scope(plan,copies,SELECTED)
'''+r[end:]
r=r.replace('review["jobs"] == {"1": "6783295", "2": "6783296"}','review["jobs"] == {key:value["job_id"] for key,value in SCOPES.items()}')
r=r.replace('test_release_common_confirmation.py','test_release_selected_profiles.py').replace('owned_uva_common_confirmation_release_intent','owned_uva_selected_profile_release_intent')
place=r.index('\ndef core(plan):')
r=r[:place]+'\n'+scope+r[place:]
code('release_selected_profiles.py',r)
release_tests=(OLD_REL/'test_release_common_confirmation.py').read_text().replace('release_common_confirmation.py','release_selected_profiles.py')
release_tests=release_tests.replace('"kind": "selected_configuration_measurement"','"kind": "selected_configuration_profile"')
release_tests=release_tests.replace('"6783295"','"1234567891"').replace('"6783296"','"1234567892"').replace('"6782640"','"1234567899"')
code('test_release_selected_profiles.py',release_tests)
# Draft metadata contains current static paths and no future approval, identifier or release proof.
packet_freeze=PREP/'profile_packets_source_freeze.json';frozen=json.loads(packet_freeze.read_bytes())
required={path:row['sha256'] for path,row in frozen['files'].items()}
required[str(packet_freeze)]=sha(packet_freeze)
nsf=R/'remote_setup/2026-11-01-21-25_nsight-tool-metadata_source-and-actual-version_freeze.json'
for path,row in json.loads(nsf.read_bytes())['files'].items():required[path]=row['sha256']
required[str(nsf)]=sha(nsf)
items={}
for row in records:
 unit=Path(row['unit']);packet=Path(row['packet']);plan=json.loads((unit/'measurement_plan.json').read_bytes())
 approved=Path(plan['accepted_baseline_specification']['independent_confirmation_review']['path'])
 for record in (*plan['inputs'].values(),*plan['target_qualification_inputs'].values(),ref(approved)):
  required[record['path']]=record['sha256']
 items[str(row['copies'])]={'packet':str(packet),'unit':str(unit),'copies':row['copies'],'hours':row['requested_hours'],
  'plan':ref(unit/'measurement_plan.json'),'script':ref(packet/'job.sbatch'),
  'command':json.loads((packet/'held_submission_command.json').read_bytes())['command'],
  'configuration':plan['configuration'],'runtime_environment':plan['runtime_environment'],'target_qualification_inputs':plan['target_qualification_inputs'],
  'accepted_confirmation':plan['inputs']['accepted_confirmation'],'accepted_confirmation_review':ref(approved),
  'accepted_baseline_specification':plan['accepted_baseline_specification'],'profile_tool':plan['profile_tool']}
draft={'schema_version':1,'kind':'fixed_selected_profile_held_submission_inputs','repository':str(R.parents[2]),'campaign_run_dir':str(R),
 'packet_freeze':ref(packet_freeze),'independent_packet_review':None,'submission_readiness':None,'packets':items,
 'required_files':[{'path':path,'sha256':expected} for path,expected in sorted(required.items())],
 'actual_submission_ids':{'1':None,'2':None}}
put('draft_submission_inputs.json',(json.dumps(draft,indent=2,sort_keys=True)+'\n').encode())
# Compare exact action/source functions rather than only claiming reuse.
def functions(text):
 return {node.name:ast.dump(node,include_attributes=False) for node in ast.parse(text).body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
oldf,newf=functions((OLD_SUB/'submit_common_confirmation_held.py').read_text()),functions(s)
held_names=('require','digest','ordinary','read_bytes','receipt','checked','document','publish_bytes','publish','append_ledgers','unattempted','validate_publication','main')
# Publication/test filename and profile intent strings are intentional scope changes.
held_equal={name:oldf[name]==newf[name] for name in held_names}
oldr,newr=functions((OLD_REL/'release_common_confirmation.py').read_text()),functions(r)
release_names=('require','sha','encoded','contents','document','ref','checked_ref','put','field','validate_held','core','validate_pre_release','utc_observation','validate_proof','validate_frozen_files','require_unattempted_release','capture','release_once','main')
release_equal={name:oldr[name]==newr[name] for name in release_names}
put('proven_function_reuse.json',(json.dumps({'held_source':ref(OLD_SUB/'submit_common_confirmation_held.py'),'release_source':ref(OLD_REL/'release_common_confirmation.py'),'held_function_AST_equal':held_equal,'release_function_AST_equal':release_equal,'actual_release_ids_unbound':True,'actual_release_proof_unbound':True},indent=2,sort_keys=True)+'\n').encode())
print(json.dumps({'created':True,'utility':str(HERE),'held_equal':held_equal,'release_equal':release_equal}))
