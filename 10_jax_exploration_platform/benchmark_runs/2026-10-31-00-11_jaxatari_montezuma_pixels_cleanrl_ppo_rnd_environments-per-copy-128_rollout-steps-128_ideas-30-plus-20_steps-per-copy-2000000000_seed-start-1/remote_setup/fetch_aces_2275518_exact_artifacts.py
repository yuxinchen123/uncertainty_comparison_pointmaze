"""Fetch six already completed ACES job artifacts, with a five-minute limit per file."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

ACTIVE_DIRECTORY = None
ACTIVE_FILE_RECORD = None

PLAN_SHA256 = "25112e056985f8ca2e4488b9683ca3d7a45edcd9c6bda6689e9dcd725968bda2"
REMOTE = r'''
import json,os,pwd,stat,sys,time
from pathlib import Path
request=json.loads(sys.argv[1])
assert pwd.getpwuid(os.getuid()).pw_name=="u.sl359075"
path=Path(request["remote_path"])
root=Path(request["native_data_unit"])
assert path.is_absolute() and str(path).startswith(str(root)+"/") and path.resolve(strict=True)==path
fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
keys=("st_dev","st_ino","st_size","st_mtime_ns","st_ctime_ns","st_mode","st_uid","st_gid","st_nlink")
started=time.monotonic()
try:
    before=os.fstat(fd)
    assert stat.S_ISREG(before.st_mode) and before.st_uid==os.getuid()
    assert before.st_size<=request["maximum_permitted_bytes"]
    assert request["expected_size_bytes"] is None or before.st_size==request["expected_size_bytes"]
    length=0
    while True:
        assert time.monotonic()-started<240
        chunk=os.read(fd,1024*1024)
        if not chunk:break
        sys.stdout.buffer.write(chunk)
        length+=len(chunk)
    sys.stdout.buffer.flush()
    after=os.fstat(fd)
    assert path.resolve(strict=True)==path and not path.is_symlink()
    assert length==before.st_size
    assert all(getattr(before,k)==getattr(after,k)==getattr(path.stat(),k) for k in keys)
    print(json.dumps({"path":str(path),"bytes":length,"identity":{k:getattr(after,k) for k in keys},
        "source_identity_unchanged":True,"image_executed":False}),file=sys.stderr,flush=True)
finally:
    os.close(fd)
'''
HASH = r'''
import hashlib,json,os,stat,sys,time
from pathlib import Path
assert os.environ.get("SLURM_JOB_ID")=="6783003" and os.environ.get("SLURM_STEP_ID","").isdigit()
path=Path(sys.argv[1]);assert path.resolve(strict=True)==path and not path.is_symlink()
start_path=Path(sys.argv[3])
with start_path.open("x") as receipt:
    receipt.write(json.dumps({"job_id":os.environ["SLURM_JOB_ID"],"step_id":os.environ["SLURM_STEP_ID"],
        "pid":os.getpid(),"uid":os.getuid(),"cgroup":Path("/proc/self/cgroup").read_text(),
        "path":str(path),"hash_process_started":True})+"\n")
    receipt.flush();os.fsync(receipt.fileno())
fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
try:
    before=os.fstat(fd);assert stat.S_ISREG(before.st_mode) and before.st_uid==2618919
    h=hashlib.sha256()
    while True:
        part=os.read(fd,1024*1024)
        if not part:break
        h.update(part)
    after=os.fstat(fd)
    assert path.resolve(strict=True)==path and not path.is_symlink()
    keys=("st_dev","st_ino","st_size","st_mtime_ns","st_ctime_ns","st_uid","st_mode")
    assert all(getattr(before,k)==getattr(after,k)==getattr(path.stat(),k) for k in keys)
    assert h.hexdigest()==sys.argv[2]
    print(json.dumps({"sha256":h.hexdigest(),"bytes":after.st_size,"job_id":os.environ["SLURM_JOB_ID"],
        "step_id":os.environ["SLURM_STEP_ID"],"identity":{k:getattr(after,k) for k in keys},
        "image_loaded_or_executed":False}))
finally:
    os.close(fd)
'''

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_once(path, value):
    raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
    with path.open("xb") as out:
        out.write(raw);out.flush();os.fsync(out.fileno())

def main():
    global ACTIVE_DIRECTORY,ACTIVE_FILE_RECORD
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    assert digest(args.plan)==PLAN_SHA256
    plan=json.loads(args.plan.read_text())
    assert str(plan["job_id"])=="2275518" and len(plan["files"])==6
    ledger=Path(plan["matching_local_owned_ledger"]["path"])
    assert "2275518" in ledger.read_text().splitlines()
    for key in ("actual_snapshot","reported_site_result","submission_receipt","release_receipt"):
        ref=plan[key];assert digest(Path(ref["path"]))==ref["sha256"],key
    destination=Path(plan["destination_directory_to_create_only_at_root_fetch"])
    assert destination.is_relative_to(args.plan.parent)
    for row in plan["files"]:
        assert Path(row["local_destination"]).is_relative_to(destination)
        assert Path(row["remote_path"]).is_relative_to(Path(plan["native_data_unit"]))
    if not args.execute:
        print(json.dumps({"reviewed_plan_sha256":PLAN_SHA256,"files":6,"executed":False}))
        return
    destination.mkdir(exist_ok=False)
    ACTIVE_DIRECTORY=destination
    receipts=[]
    for number,row in enumerate(plan["files"]):
        started=time.monotonic();deadline=started+300
        path=Path(row["local_destination"]);path.parent.mkdir(parents=True,exist_ok=True)
        request={key:row[key] for key in ("remote_path","maximum_permitted_bytes","expected_size_bytes")}
        request["native_data_unit"]=plan["native_data_unit"]
        command=["ssh","-T","-o","ControlMaster=no","-o","ProxyCommand=false","-o","BatchMode=yes",
            "-o","ConnectTimeout=15","aces",
            "python3 -I -S -B -c "+shlex.quote(REMOTE)+" "+shlex.quote(json.dumps(request))]
        write_once(destination/(str(number)+"_fetch_intent.json"),{"role":row["role"],"command":command,
            "expected_sha256":row["expected_sha256"],"started_at_utc":datetime.now(timezone.utc).isoformat(),
            "maximum_total_seconds":300,"no_job_action":True})
        record={"role":row["role"],"local_path":str(path),"verified":False,"no_job_action":True}
        ACTIVE_FILE_RECORD=record
        error=None
        with path.open("xb") as out,(destination/(str(number)+"_fetch_stderr.raw")).open("xb") as err:
            os.fchmod(out.fileno(),0o600)
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=out,stderr=err)
            try:
                code=child.wait(timeout=240)
            except BaseException as failure:
                error=failure
                if child.poll() is None:child.kill()
                child.wait(timeout=max(0.1,deadline-time.monotonic()))
                code=child.returncode
            out.flush();err.flush();os.fsync(out.fileno());os.fsync(err.fileno())
            record["transport_returncode"]=code
            if error is None and code==0:
                remote=json.loads((destination/(str(number)+"_fetch_stderr.raw")).read_text())
                assert remote["source_identity_unchanged"] is True and remote["bytes"]==path.stat().st_size
                remaining=deadline-time.monotonic()
                # Leave one full minute for step exit/accounting and file receipt publication.
                hash_minutes=int((remaining-60)//60)
                assert hash_minutes>=1, "download retained; verify it in a separate bounded check without fetching again"
                run_dir=args.plan.parent.parent
                hash_start=destination/(str(number)+"_hash_process_start.json")
                command_hash=["srun","--jobid=6783003","--overlap","--exact","--immediate=5",
                    "--time="+str(hash_minutes),"-n1","-c1","--mem=256M","flock","-n","-x",
                    str(run_dir/"queue/cpu_monitor_large_artifact_read.lock"),
                    "/usr/bin/python3","-I","-S","-B","-c",HASH,str(path),row["expected_sha256"],str(hash_start)]
                record.update(hash_command=command_hash,hash_step_limit_seconds=hash_minutes*60,
                    hash_start_record=str(hash_start),hash_lock_nonblocking=True)
                try:
                    result=subprocess.run(command_hash,capture_output=True,text=True,
                        timeout=min(hash_minutes*60+35,deadline-time.monotonic()-10))
                except subprocess.TimeoutExpired:
                    record["hash_step_exit_verified"]=False
                    record["hash_controller_timed_out"]=True
                    raise
                record.update(hash_command=command_hash,hash_returncode=result.returncode,
                    hash_stdout=result.stdout,hash_stderr=result.stderr,remote_identity=remote)
                assert result.returncode==0,result.stderr
                proof=json.loads(result.stdout);assert proof["sha256"]==row["expected_sha256"]
                assert proof["bytes"]==remote["bytes"]
                os.fchmod(out.fileno(),0o440);os.fsync(out.fileno())
                record["verified"]=True
        record["elapsed_seconds"]=time.monotonic()-started
        if error is not None:record["error"]={"type":type(error).__name__,"message":str(error)}
        write_once(destination/(str(number)+"_fetch_receipt.json"),record)
        receipts.append(record)
        print(json.dumps({"role":row["role"],"verified":record["verified"],
            "bytes":path.stat().st_size,"elapsed_seconds":record["elapsed_seconds"]}),flush=True)
        if not record["verified"]:
            write_once(destination/"fetch_summary.json",{"verified_files":sum(x["verified"] for x in receipts),
                "requested_files":6,"all_six_verified":False,"scientific_credit":0,"worker_admission":False})
            raise RuntimeError("artifact transfer failed; preserve files without retry")
    write_once(destination/"fetch_summary.json",{"verified_files":sum(x["verified"] for x in receipts),
        "requested_files":6,"all_six_verified":len(receipts)==6 and all(x["verified"] for x in receipts),
        "scientific_credit":0,"worker_admission":False,"binary_bytes_allowed_in_git":False})

if __name__=="__main__":
    try:
        main()
    except BaseException as error:
        if ACTIVE_DIRECTORY is not None:
            failure=ACTIVE_DIRECTORY/"fetch_failure.json"
            if not failure.exists():
                write_once(failure,{"error_type":type(error).__name__,"message":str(error),
                    "all_six_verified":False,"scientific_credit":0,"worker_admission":False,
                    "automatic_retry":False,"active_file_record":ACTIVE_FILE_RECORD})
        raise
