"""Release only the saved ACES site job after its actual held monitor proof."""
import base64
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys

RUN = Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1")
JOB = "2274632"
REMOTE = r'''
import base64,datetime,hashlib,json,os,pathlib,pwd,re,socket,subprocess,sys
x=json.load(sys.stdin)
assert os.getuid()==56626 and pwd.getpwuid(os.getuid()).pw_name=="u.sl359075"
assert socket.gethostname().split(".")[0]=="alogin1"
p=pathlib.Path(x["packet"]);r=pathlib.Path(x["run"])
assert p.resolve(strict=True)==p and r.resolve(strict=True)==r and p.parent==r/"remote_setup"
job=x["job_id"];assert job=="2274632"
def sha(raw):return hashlib.sha256(raw).hexdigest()
def put(name,raw):
 fd=os.open(p/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,"wb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
 fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
def save(name,v):put(name,(json.dumps(v,indent=2,allow_nan=False)+"\n").encode())
raw=(p/"submission_receipt.json").read_bytes();s=json.loads(raw)
assert sha(raw)==x["submission_receipt_sha256"] and s["job_id"]==job
assert s["kind"]=="owned_aces_full_site_qualification_submission" and s["cluster"]=="aces"
assert (r/"slurm/submitted_jobids_aces.txt").read_text().splitlines().count(job)==1
assert [v for v in (r/"slurm/submissions_aces.tsv").read_text().splitlines() if v.split("\t",1)[0]==job]==["\t".join(s["submission_row"])]
assert all(not (p/n).exists() and not (p/n).is_symlink() for n in ["release_intent.json","release_receipt.json","release_return.json"])
for n,expected in x["native_hashes"].items():assert sha((p/n).read_bytes())==expected
save("release_intent.json",x)
query=subprocess.run(["scontrol","show","job","-o",job],capture_output=True,timeout=30,check=True)
put("release_owned_job_before.raw",query.stdout)
detail=query.stdout.decode()
assert re.search(r"(?:^|\s)JobId=2274632(?:\s|$)",detail)
assert "UserId=u.sl359075(56626)" in detail and "JobState=PENDING" in detail and "Reason=JobHeldUser" in detail
assert "Account=158590754315" in detail and "QOS=normal" in detail
cmd=["scontrol","release",job]
with (p/"release_stdout.raw").open("xb") as out,(p/"release_stderr.raw").open("xb") as err:
 try:
  result=subprocess.run(cmd,stdout=out,stderr=err,timeout=30)
 except BaseException as e:
  out.flush();os.fsync(out.fileno());err.flush();os.fsync(err.fileno())
  save("release_return.json",{"returncode":None,"error_type":type(e).__name__,"error":str(e),"automatic_retry_permitted":False})
  raise
 out.flush();os.fsync(out.fileno());err.flush();os.fsync(err.fileno())
save("release_return.json",{"returncode":result.returncode,"automatic_retry_permitted":False})
assert result.returncode==0
v={"schema_version":1,"kind":"owned_aces_full_site_qualification_release","cluster":"aces","job_id":job,"command":cmd,"returncode":0,"stdout":(p/"release_stdout.raw").read_text(),"stderr":(p/"release_stderr.raw").read_text(),"released_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"submission_receipt_sha256":sha(raw),"persistent_monitor_job_id":"6780635","persistent_monitor_source_commit":"f9a9c1672f788f52ba06836cee26f2dd6314cec7","held_monitor_verification_sha256":x["held_monitor_verification_sha256"],"owned_job_detail_before_release":detail,"automatic_retry_permitted":False,"workers_permitted":False,"campaign_claims_permitted":False,"scientific_hypothesis_credit":0,"comparison_credit":0,"idea_credit":0,"admission_credit":0}
save("release_receipt.json",v)
print(json.dumps({"files":{n:base64.b64encode((p/n).read_bytes()).decode() for n in ["release_receipt.json","release_intent.json","release_owned_job_before.raw","release_stdout.raw","release_stderr.raw","release_return.json"]}}))
'''
def sha(raw):
    return hashlib.sha256(raw).hexdigest()
def write(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    os.fsync(descriptor)
    os.close(descriptor)
def main():
    assert socket.gethostname().split(".")[0]=="portal05"
    plan=json.loads((RUN/"next_actions.json").read_text())["aces_full_site_qualification"]
    packet=Path(plan["packet"])
    native=json.loads((packet/"site_validation_plan.json").read_text())
    raw=(packet/"submission_receipt.json").read_bytes()
    receipt=json.loads(raw)
    proof_raw=(packet/"held_monitor_verification.json").read_bytes()
    proof=json.loads(proof_raw)
    assert receipt["job_id"]==JOB and proof["job_id"]==JOB and proof["passed"] is True
    assert proof["monitor_job_id"]=="6780635" and proof["monitor_source_commit"]=="f9a9c1672f788f52ba06836cee26f2dd6314cec7"
    assert proof["kind"]=="owned_aces_full_site_actual_held_monitor_verification"
    assert type(proof["actual_monitor_returncode"]) is int and proof["actual_monitor_returncode"]==0
    assert proof["zero_scientific_credit"] is True and len(proof["held_rows"])==1
    assert (RUN/"slurm/submitted_jobids_aces.txt").read_text().splitlines().count(JOB)==1
    assert [v for v in (RUN/"slurm/submissions_aces.tsv").read_text().splitlines() if v.split("\t",1)[0]==JOB]==["\t".join(receipt["submission_row"])]
    assert json.loads((packet/"submission_adoption_receipt.json").read_text())["passed"] is True
    row=proof["held_rows"][0]["row"]
    assert row["state"]=="held" and row["slurm_job_id"]==JOB and row["workers_permitted"] is False
    relative="10_jax_exploration_platform/scripts/atari_pixel_campaign/aces_site_monitor.py"
    pinned=RUN/"data/code_snapshots"/proof["monitor_source_commit"]/relative
    expected=subprocess.check_output(["git","-C",str(RUN.parents[2]),"cat-file","blob",proof["monitor_source_commit"]+":"+relative])
    assert pinned.read_bytes()==expected
    specification=importlib.util.spec_from_file_location("reviewed_f9_aces_release_monitor",pinned)
    adapter=importlib.util.module_from_spec(specification)
    specification.loader.exec_module(adapter)
    inputs=adapter.packet_inputs(RUN,packet)
    assert inputs["release"] is None and inputs["identifier"]==JOB
    checked_row,checked_record=adapter.site_row(inputs)
    assert checked_record==proof["held_rows"][0] and checked_row==row
    observed=datetime.fromisoformat(checked_row["observed_at_utc"].replace("Z","+00:00"))
    assert observed.tzinfo is not None and observed.utcoffset().total_seconds()==0
    assert 0 <= (datetime.now(timezone.utc)-observed).total_seconds()<1200
    assert 0 <= (datetime.now(timezone.utc)-datetime.fromisoformat(proof["at_utc"])).total_seconds()<600
    assert sha((packet/"remote_monitor_snapshot.json").read_bytes())==proof["snapshot_sha256"]
    hashes={n:sha((packet/n).read_bytes()) for n in ["site_validation_plan.json","manifest.yaml","job.sbatch"]}
    value={"schema_version":1,"kind":"owned_aces_full_site_release_intent","job_id":JOB,"packet":plan["native_packet"],"run":native["campaign_run_dir"],"at_utc":datetime.now(timezone.utc).isoformat(),"submission_receipt_sha256":sha(raw),"held_monitor_verification_sha256":sha(proof_raw),"native_hashes":hashes,"release_source_sha256":sha(Path(__file__).read_bytes()),"remote_source_sha256":sha(REMOTE.encode()),"automatic_retry_permitted":False}
    if sys.argv[1:]!=["--release"]:
        assert not sys.argv[1:]
        print(json.dumps({"validated":True,"job_id":JOB,"ssh_calls":0,"release_calls":0}))
        return
    write(packet/"release_intent.json",(json.dumps(value,indent=2)+"\n").encode())
    cmd=["ssh","-T","-o","ControlMaster=no","-o","ProxyCommand=false","-o","BatchMode=yes","-o","NumberOfPasswordPrompts=0","-o","ConnectionAttempts=1","-o","ConnectTimeout=10","aces","exec "+shlex.join([native["python"],"-I","-B","-c",REMOTE])]
    with (packet/"release_transport_stdout.raw").open("xb") as out,(packet/"release_transport_stderr.raw").open("xb") as err:
        try:
            result=subprocess.run(cmd,input=(json.dumps(value)+"\n").encode(),stdout=out,stderr=err,timeout=90)
        except BaseException as error:
            out.flush();os.fsync(out.fileno());err.flush();os.fsync(err.fileno())
            write(packet/"release_transport_result.json",(json.dumps({"returncode":None,"error_type":type(error).__name__,"error":str(error),"automatic_retry_permitted":False})+"\n").encode())
            raise
        out.flush();os.fsync(out.fileno());err.flush();os.fsync(err.fileno())
    write(packet/"release_transport_result.json",(json.dumps({"returncode":result.returncode,"automatic_retry_permitted":False})+"\n").encode())
    assert result.returncode==0
    response=json.loads((packet/"release_transport_stdout.raw").read_bytes())
    names={"release_receipt.json","release_intent.json","release_owned_job_before.raw","release_stdout.raw","release_stderr.raw","release_return.json"}
    assert set(response)=={"files"} and set(response["files"])==names
    values={n:base64.b64decode(v,validate=True) for n,v in response["files"].items()}
    saved=json.loads(values["release_receipt.json"])
    assert type(saved["schema_version"]) is int and saved["schema_version"]==1
    assert saved["kind"]=="owned_aces_full_site_qualification_release" and saved["cluster"]=="aces" and saved["job_id"]==JOB
    assert type(saved["returncode"]) is int and saved["returncode"]==0 and saved["command"]==["scontrol","release",JOB]
    adapter.boundary(saved)
    assert saved["automatic_retry_permitted"] is False
    assert saved["submission_receipt_sha256"]==sha(raw) and saved["held_monitor_verification_sha256"]==sha(proof_raw)
    assert json.loads(values["release_intent.json"])==value
    returned=json.loads(values["release_return.json"])
    assert type(returned["returncode"]) is int and returned["returncode"]==0 and returned["automatic_retry_permitted"] is False
    assert values["release_stdout.raw"]==saved["stdout"].encode() and values["release_stderr.raw"]==saved["stderr"].encode()
    assert values["release_owned_job_before.raw"]==saved["owned_job_detail_before_release"].encode()
    for name,contents in values.items():
        write(packet/("remote_release_intent.json" if name=="release_intent.json" else name),contents)
    print(json.dumps({"released":True,"job_id":JOB,"remote_and_local_receipt_bytes_saved":True,"workers_permitted":False}))
if __name__=="__main__":
    main()

