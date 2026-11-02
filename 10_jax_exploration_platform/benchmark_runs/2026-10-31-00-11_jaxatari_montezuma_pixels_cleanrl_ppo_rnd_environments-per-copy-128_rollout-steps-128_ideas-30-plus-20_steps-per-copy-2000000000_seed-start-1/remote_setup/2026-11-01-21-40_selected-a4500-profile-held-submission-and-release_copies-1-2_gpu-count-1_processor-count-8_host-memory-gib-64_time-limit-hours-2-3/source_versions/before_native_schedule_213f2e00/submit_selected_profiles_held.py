"""Submit one exact selected common A4500 profile packet held, once, without release."""
import argparse
from contextlib import ExitStack
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess

UTILITY=Path(__file__).absolute().parent
INPUTS_SHA='39d9af6e53b680edf084fba97a50dd8b1e2454569ac469bd4d79849665bad4f2'
SOURCE="fdb35f886ad01c6d4ed387dc55b172d0221a9853"
ATTEMPTS=("submission_intent.json","submission_stdout.raw","submission_stderr.raw","submission_return.json","submission_failure.json","submission_receipt.json","held_binding.json")
BOUNDARY={"scientific_credit":0,"case_credit":0,"idea_credit":0,"admission_granted":False,"worker_admitted":False}

def require(value, message):
    if not value:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def ordinary(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path, "exact ordinary path required")
    require(stat.S_ISREG(path.lstat().st_mode), "ordinary file required")
    return path


def read_bytes(path):
    path = ordinary(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        require(stat.S_ISREG(os.fstat(fd).st_mode), "ordinary opened file required")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(64 * 1024 * 1024 + 1)
        require(len(raw) <= 64 * 1024 * 1024, "metadata/source artifact exceeds bound")
        return raw
    finally:
        os.close(fd)


def receipt(path):
    return {"path": str(path), "sha256": digest(read_bytes(path))}


def checked(value):
    require(isinstance(value, dict) and set(value) == {"path", "sha256"}
        and receipt(value["path"]) == value, "exact input bytes changed")
    return read_bytes(value["path"])


def document(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(read_bytes(path).decode("utf-8", errors="strict"), object_pairs_hook=pairs)


def publish_bytes(path, raw):
    path = Path(path); require(path.parent.resolve(strict=True) == path.parent, "ordinary output parent required")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as out:
            out.write(raw); out.flush(); os.fsync(out.fileno())
    finally:
        os.close(fd)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return receipt(path)


def publish(path, value):
    return publish_bytes(path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())



def append_ledgers(campaign,job,row):
    """Synchronize both identifier ledgers and the exact row before returning immutable copies."""
    paths=[campaign/'slurm/submitted_jobids.txt',campaign/'slurm/submitted_jobids_uva.txt',campaign/'slurm/submissions.tsv']
    with ExitStack() as stack:
        streams=[]
        for path in paths:
            ordinary(path);fd=os.open(path,os.O_RDWR|os.O_NOFOLLOW|os.O_NONBLOCK)
            require(stat.S_ISREG(os.fstat(fd).st_mode),'ordinary ledger required')
            stream=stack.enter_context(os.fdopen(fd,'r+',encoding='utf-8'));fcntl.flock(fd,fcntl.LOCK_EX);streams.append(stream)
        before=[stream.read() for stream in streams]
        require(all(job not in text.splitlines() for text in before[:2]) and not any(line.split('\t')[0]==job or line.split('\t')[-1]==row[-1] for line in before[2].splitlines()),'returned identifier or exact packet already recorded; preserve partial attempt')
        for stream,text,line in zip(streams,before,[job,job,'\t'.join(row)]):
            require(not text or text.endswith('\n'),'ledger final newline missing')
            stream.seek(0,os.SEEK_END);stream.write(line+'\n');stream.flush();os.fsync(stream.fileno())
        after=[read_bytes(path) for path in paths]
        require(all(raw.decode().splitlines().count(job)==1 for raw in after[:2]) and after[2].decode().splitlines().count('\t'.join(row))==1,'durable actual identifier or row is missing or duplicated')
        return [{'path':str(path),'before_sha256':digest(text.encode()),'after_sha256':digest(raw)} for path,text,raw in zip(paths,before,after)],after[1],after[2]


def unattempted(packet,unit,campaign):
    require(not any((packet/name).exists() or (packet/name).is_symlink() for name in ATTEMPTS),'packet already has a submission attempt; never retry')
    require(not any((unit/name).exists() or (unit/name).is_symlink() for name in ('configured_isolation_owned_job_ids.txt','configured_isolation_owned_submissions.tsv','configured_isolation_pre_release_inputs.json','owned_allocation_inputs.json','owned_inputs_wait_completed.json','owned_inputs_wait_failure.json')),'actual launch ownership inputs already exist; never retry')
    require(not any(line.split('\t')[-1]==str(packet/'job.sbatch') for line in read_bytes(campaign/'slurm/submissions.tsv').decode().splitlines()),'an owned submission row already names this packet')


def same(left,right):
    """Keep Boolean and integer fields distinct in nested scientific metadata."""
    if type(left) is not type(right):return False
    if isinstance(left,dict):return set(left)==set(right) and all(same(left[key],right[key]) for key in left)
    if isinstance(left,list):return len(left)==len(right) and all(same(a,b) for a,b in zip(left,right))
    return left==right

def profile_scope(plan,copies,item):
    """Require only the exact accepted common A4500 profile, never a measurement or direct plan."""
    require(plan.get("kind")=="selected_configuration_profile" and plan.get("source_commit")=="d28e6735e79788f618acb0ebfde8609db1f776af"
        and plan["helpers"]["source_commit"]==SOURCE and len(plan["helpers"]["files"])==22
        and all("10_jax_exploration_platform/scripts/atari_pixel_campaign/"+name in plan["helpers"]["files"]
            for name in ("profile_selected.py","common_confirmation.py","uva_slurm_configured_isolation.py"))
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

def require_profile_unstarted(unit,item):
    """Refuse additional profile files or directories before creating a held-job intent."""
    paths=list(unit.rglob("*"))
    require(all(not path.is_symlink() for path in paths)
        and sorted(str(path.relative_to(unit)) for path in paths)==item["initial_unit_paths"],
        "retain the previous or partial profile attempt; use a fresh unit")


def validate_inputs(copies):
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
    require_profile_unstarted(unit,item)
    unattempted(packet,unit,campaign)
    return inputs,item,plan

def validate_readiness(inputs):
    """Require a recent actual running full monitor and verified twenty-minute native schedule."""
    record=inputs.get("submission_readiness")
    require(isinstance(record,dict),"actual current monitoring readiness is not bound")
    ready=json.loads(checked(record))
    require(type(ready.get("schema_version")) is int and ready["schema_version"]==1
        and ready.get("kind")=="actual_selected_profile_submission_monitor_readiness"
        and ready.get("passed") is True and type(ready.get("actual_actions")) is int and ready["actual_actions"]==0,"unsupported current monitoring readiness")
    verification=json.loads(checked(ready["full_verification"]));status=json.loads(checked(ready["full_status"]))
    checked(ready["full_report"]);runtime=json.loads(checked(ready["runtime_review"]));native=json.loads(checked(ready["native_record"]))
    observed=datetime.fromisoformat(verification["checked_at_utc"].replace("Z","+00:00"))
    require(observed.tzinfo is not None and observed.utcoffset().total_seconds()==0
        and 0<=(datetime.now(timezone.utc)-observed).total_seconds()<1200,
        "the actual full-monitor observation is stale,future-dated or not UTC")
    require(verification.get("passed") is True and type(verification.get("returncode")) is int and verification["returncode"]==0
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

def validate_publication(inputs,commit):
    require(re.fullmatch(r'[0-9a-f]{40}',commit or ''),'exact published preparation commit required')
    repository=Path(inputs['repository'])
    subprocess.run(['git','-C',str(repository),'merge-base','--is-ancestor',commit,'origin/merge-atari-cleanrl'],check=True)
    required=[*inputs['required_files'],receipt(UTILITY/'submission_inputs.json'),receipt(Path(__file__).absolute()),receipt(UTILITY/'test_submit_selected_profiles_held.py'),receipt(UTILITY/'author_review.json')]
    for value in required:
        path=Path(value['path']);raw=subprocess.check_output(['git','-C',str(repository),'show',commit+':'+str(path.relative_to(repository))])
        require(digest(raw)==value['sha256'] and receipt(path)==value,'required exact bytes are absent from the published commit')
    return commit


def submit_once(copies,publication):
    """One durable intent permits one sbatch; every partial attempt permanently stops retries."""
    inputs,item,plan=validate_inputs(copies);validate_publication(inputs,publication);validate_readiness(inputs)
    require(os.getuid()==os.geteuid()==2618919 and socket.gethostname().split('.')[0]=='portal05','only the owner onportal05 may submit')
    packet=Path(item['packet']);unit=Path(item['unit']);campaign=Path(inputs['campaign_run_dir'])
    intent=publish(packet/'submission_intent.json',{'schema_version':1,'kind':'exclusive_selected_profile_held_submission_intent','at_utc':datetime.now(timezone.utc).isoformat(),'copies':copies,'command':item['command'],'publication_commit':publication,'inputs':receipt(UTILITY/'submission_inputs.json'),'automatic_retry_permitted':False,**BOUNDARY})
    job=None
    try:
        returned={'command':item['command'],'at_utc':datetime.now(timezone.utc).isoformat()}
        try:
            process=subprocess.run(item['command'],capture_output=True,timeout=90)
            stdout,stderr=process.stdout,process.stderr;returned.update(returncode=process.returncode,completed=True)
        except BaseException as error:
            stdout,stderr=getattr(error,'stdout',None) or b'',getattr(error,'stderr',None) or b''
            returned.update(returncode=None,completed=False,error_type=type(error).__name__,error=str(error))
        out=publish_bytes(packet/'submission_stdout.raw',stdout);err=publish_bytes(packet/'submission_stderr.raw',stderr)
        returned.update(stdout=out,stderr=err);raw_return=publish(packet/'submission_return.json',returned)
        match=re.fullmatch(rb'([1-9][0-9]*)\n?',stdout)
        require(returned['completed'] is True and type(returned['returncode']) is int and returned['returncode']==0 and stderr==b'' and match is not None,'sbatch returned failure or an ambiguous identifier; never retry')
        job=match[1].decode('ascii');row=[job,plan['kind'],SOURCE,item['command'][-1]]
        ledgers,ids_snapshot,rows_snapshot=append_ledgers(campaign,job,row)
        ids_ref=publish_bytes(unit/'configured_isolation_owned_job_ids.txt',ids_snapshot);rows_ref=publish_bytes(unit/'configured_isolation_owned_submissions.tsv',rows_snapshot)
        record={'schema_version':1,'kind':'owned_uva_configured_isolation_submission','cluster':'uva','job_id':job,'at_utc':datetime.now(timezone.utc).isoformat(),'returncode':0,'packet':str(packet),'data_unit':str(unit),'source_commit':SOURCE,'publication_commit':publication,'plan_sha256':item['plan']['sha256'],'script_sha256':item['script']['sha256'],'held_command':item['command'],'submission_row':row,'submission_intent':intent,'submission_return':raw_return,'ledger_appends':ledgers,'ledgers_and_submission_row_fsynced_before_this_receipt':True,'released':False,'automatic_retry_permitted':False,**BOUNDARY}
        submitted=publish(packet/'submission_receipt.json',record)
        pre={'schema_version':1,'kind':'uva_configured_isolation_pre_release_inputs','job_id':job,'plan':item['plan'],'ledger':{'snapshot':ids_ref,'source_path':str(campaign/'slurm/submitted_jobids_uva.txt')},'submissions':{'snapshot':rows_ref,'source_path':str(campaign/'slurm/submissions.tsv')},'submission':submitted,'submission_row':row}
        pre_ref=publish(unit/'configured_isolation_pre_release_inputs.json',pre)
        held=publish(packet/'held_binding.json',{'schema_version':1,'kind':'actual_owned_uva_configured_isolation_held_binding','job_id':job,'at_utc':datetime.now(timezone.utc).isoformat(),'submission_receipt':submitted,'plan':item['plan'],'pre_release_inputs':pre_ref,'copied_ledgers':{key:pre[key] for key in ('ledger','submissions')},'no_frozen_plan_change':True,'physical_gpu_identity_unresolved':True})
        jobs_root=campaign/'slurm/jobs';require(jobs_root.resolve(strict=True)==jobs_root,'exact per-submission parent directory required')
        job_directory=jobs_root/job;job_directory.mkdir()
        parent_fd=os.open(jobs_root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:os.fsync(parent_fd)
        finally:os.close(parent_fd)
        publish_bytes(job_directory/'job.sbatch',read_bytes(packet/'job.sbatch'))
        publish_bytes(job_directory/'config_resolved.yaml',read_bytes(unit/'config_resolved.yaml'))
        publish_bytes(job_directory/'experiment_background.md',read_bytes(packet/'experiment_background.md'))
        publish(job_directory/'manifest.yaml',{'schema_version':1,'kind':'held_selected_profile_per_submission_records','job_id':job,'copies':copies,'submission_receipt':submitted,'held_binding':held,'source_commit':plan['source_commit'],'helper_commit':SOURCE,'observed_host':None,'observed_gpu_uuid':None,'requested_host':'jaguar03','released':False,**BOUNDARY})
        publish(job_directory/'assignment.json',{'job_id':job,'packet':str(packet),'unit':str(unit),'submission_row':row,'owned_inputs_pre_release':pre_ref,'physical_identity_unresolved':True,'released':False,**BOUNDARY})
        publish_bytes(job_directory/'submission_receipt.json',read_bytes(packet/'submission_receipt.json'))
        return {'job_id':job,'copies':copies,'held':True,'released':False,'ledgers_fsynced':True,'pre_release_inputs':pre_ref,'held_binding':held,'per_submission_directory':str(job_directory)}
    except BaseException as error:
        publish(packet/'submission_failure.json',{'schema_version':1,'error_type':type(error).__name__,'error':str(error),'intent':intent,'actual_returned_job_id_if_validated':job,'automatic_retry_permitted':False,'released':False,**BOUNDARY});raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--copies',type=int,choices=(1,2),required=True);parser.add_argument('--publication-commit');parser.add_argument('--submit-held',action='store_true');args=parser.parse_args(argv)
    require(not args.submit_held or args.publication_commit is not None,'publication required before submission')
    if args.submit_held:result=submit_once(args.copies,args.publication_commit)
    else:
        inputs,item,plan=validate_inputs(args.copies)
        if args.publication_commit is not None:validate_publication(inputs,args.publication_commit)
        result={'validated':True,'copies':args.copies,'command':item['command'],'scheduler_queries':0,'submissions':0,'releases':0,'owned_launch_inputs_written':False}
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
