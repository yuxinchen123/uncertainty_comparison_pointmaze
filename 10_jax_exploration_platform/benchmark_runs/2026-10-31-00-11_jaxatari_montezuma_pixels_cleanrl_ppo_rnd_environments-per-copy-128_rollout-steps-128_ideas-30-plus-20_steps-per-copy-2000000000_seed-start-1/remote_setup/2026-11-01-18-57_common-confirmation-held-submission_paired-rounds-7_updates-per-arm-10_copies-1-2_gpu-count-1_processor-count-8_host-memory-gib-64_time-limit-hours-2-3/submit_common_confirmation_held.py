"""Submit one exact common-candidate confirmation packet held, once, without release."""
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
INPUTS_SHA='035353595780d30bd12b5ecb7ed3989af42e005dcd65473dfbee8b41210139e2'
SOURCE="9fcc43a478b93d731df780805fd7c3c514cdf67f"
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


def validate_inputs(copies):
    """Read the exact current reviewed packet bytes, without any scheduler query or action."""
    require(type(copies) is int and copies in (1,2),'select exactly one supported copy count')
    require(INPUTS_SHA is not None and receipt(UTILITY/'submission_inputs.json')['sha256']==INPUTS_SHA,'actual independent packet review is not yet bound')
    inputs=document(UTILITY/'submission_inputs.json');require(type(inputs['schema_version']) is int and inputs['schema_version']==1 and inputs['kind']=='fixed_common_confirmation_held_submission_inputs','unsupported held-submission input kind')
    for record in inputs['required_files']:checked(record)
    review=document(inputs['independent_packet_review']['path'])
    freeze=document(inputs['packet_freeze']['path'])
    require(receipt(inputs['packet_freeze']['path'])==inputs['packet_freeze'],'static packet freeze changed')
    require(review.get('approved') is True and review.get('kind')=='independent_common_a4500_confirmation_packets_and_startup_review'
        and type(review.get('schema_version')) is int and review['schema_version']==1
        and review.get('packet_freeze')==inputs['packet_freeze']
        and review.get('before')==review.get('after')==freeze['files']
        and review.get('admission_granted') is False,'actual independent packet review did not pass')
    item=inputs['packets'][str(copies)];packet=Path(item['packet']);unit=Path(item['unit']);campaign=Path(inputs['campaign_run_dir']);plan=document(unit/'measurement_plan.json')
    require(plan['mode']=='confirmation' and plan['kind']=='selected_configuration_measurement' and plan['source_commit']=='d28e6735e79788f618acb0ebfde8609db1f776af' and plan['helpers']['source_commit']==SOURCE and plan['gpu_uuid'] is None and plan['configured_isolation']['physical_identity']['gpu_uuid'] is None and type(plan['configuration']['learner']['n_copies']) is int and plan['configuration']['learner']['n_copies']==copies and plan['copy_seeds']==list(range(1,copies+1)) and all(type(seed) is int for seed in plan['copy_seeds']),'the exact independently reviewed confirmation plan changed')
    require(all(type(plan.get(key)) is int and plan[key]==value for key,value in
        (('rounds',7),('iterations',10),('minimum_updates_per_round',10)))
        and not any(key in plan for key in ('final_confirmation_protocol','completed_study','direct_binding','target_qualification'))
        and plan['common_correctness_inputs']==item['common_correctness_inputs'],
        'only the exact intermediate seven-pair ten-update prior-correctness protocol is supported')
    for record in plan['common_correctness_inputs'].values():checked(record)
    require(receipt(unit/'measurement_plan.json')==item['plan'] and receipt(packet/'job.sbatch')==item['script'],'fixed plan or script changed')
    expected=['sbatch','--parsable','--hold','--no-requeue','--nodes=1','--ntasks=1','--cpus-per-task=8','--mem=64G','--time='+{1:'02',2:'03'}[copies]+':00:00','--partition=gpu','--account=sl5nw_base','--qos=csresnolim','--nodelist=jaguar03','--reservation=sl5nw_168','--gres=gpu:nvidia_rtx_a4500:1',str(packet/'job.sbatch')]
    require(item['command']==expected and document(packet/'held_submission_command.json')['command']==expected,'only this exact held single-A4500 command is permitted')
    for record in plan['target_qualification_inputs'].values():checked(record)
    unattempted(packet,unit,campaign)
    return inputs,item,plan


def validate_publication(inputs,commit):
    require(re.fullmatch(r'[0-9a-f]{40}',commit or ''),'exact published preparation commit required')
    repository=Path(inputs['repository'])
    subprocess.run(['git','-C',str(repository),'merge-base','--is-ancestor',commit,'origin/merge-atari-cleanrl'],check=True)
    required=[*inputs['required_files'],receipt(UTILITY/'submission_inputs.json'),receipt(Path(__file__).absolute()),receipt(UTILITY/'test_submit_common_confirmation_held.py'),receipt(UTILITY/'author_review.json')]
    for value in required:
        path=Path(value['path']);raw=subprocess.check_output(['git','-C',str(repository),'show',commit+':'+str(path.relative_to(repository))])
        require(digest(raw)==value['sha256'] and receipt(path)==value,'required exact bytes are absent from the published commit')
    return commit


def submit_once(copies,publication):
    """One durable intent permits one sbatch; every partial attempt permanently stops retries."""
    inputs,item,plan=validate_inputs(copies);validate_publication(inputs,publication)
    require(os.getuid()==os.geteuid()==2618919 and socket.gethostname().split('.')[0]=='portal05','only the owner onportal05 may submit')
    packet=Path(item['packet']);unit=Path(item['unit']);campaign=Path(inputs['campaign_run_dir'])
    intent=publish(packet/'submission_intent.json',{'schema_version':1,'kind':'exclusive_common_confirmation_held_submission_intent','at_utc':datetime.now(timezone.utc).isoformat(),'copies':copies,'command':item['command'],'publication_commit':publication,'inputs':receipt(UTILITY/'submission_inputs.json'),'automatic_retry_permitted':False,**BOUNDARY})
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
        publish(job_directory/'manifest.yaml',{'schema_version':1,'kind':'held_common_confirmation_per_submission_records','job_id':job,'copies':copies,'submission_receipt':submitted,'held_binding':held,'source_commit':plan['source_commit'],'helper_commit':SOURCE,'observed_host':None,'observed_gpu_uuid':None,'requested_host':'jaguar03','released':False,**BOUNDARY})
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
