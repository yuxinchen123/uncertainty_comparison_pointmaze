"""Stage one published helper commit over the existing ACES master without starting jobs."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import shlex
import subprocess


ROOT = Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl")
DIRECTORY = Path(__file__).resolve().parent
TIP = "11ceab5c640b484d18ca40e05785bbf55c6de86f"
BASIS = "8118d12e7924d0c2978db5391b32c258a0a5d20f"
LOCAL_REF = "refs/transfer/aces_helpers_" + TIP
REMOTE_REF = "refs/heads/study-helpers-" + TIP
REMOTE_BASE = "/scratch/user/u.sl359075/rnd_atari_pixels/git"
REMOTE_PYTHON = "/scratch/user/u.sl359075/envs/rnd_atari_pixels_jax/bin/python"
BUNDLE_NAME = "helpers-" + TIP + "-since-" + BASIS + ".bundle"
REMOTE_BUNDLE = REMOTE_BASE + "/bundles/" + BUNDLE_NAME
REMOTE_CHECKOUT = REMOTE_BASE + "/checkouts/study_helpers_" + TIP[:12]
SPARSE = ["10_jax_exploration_platform/src", "10_jax_exploration_platform/configs",
          "10_jax_exploration_platform/scripts/atari_pixel_campaign"]
COMMANDS = []
SSH = ["ssh", "-T", "-o", "ControlMaster=no", "-o", "ProxyCommand=false", "-o", "BatchMode=yes",
       "-o", "ConnectTimeout=5", "aces"]


def save(name, value):
    """Save finite complete transport metadata without replacing an earlier record."""
    path = DIRECTORY / name
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def limit():
    """Bound Git metadata work to two gigabytes of address space."""
    resource.setrlimit(resource.RLIMIT_AS, (2 * 2 ** 30, 2 * 2 ** 30))


def invoke(command, *, stdin=None, bounded=False, timeout=180):
    """Record each exact argument list and require its successful exit before continuing."""
    COMMANDS.append(command)
    result = subprocess.run(command, stdin=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, timeout=timeout, preexec_fn=limit if bounded else None)
    if result.returncode:
        raise RuntimeError(json.dumps({"command": command, "exit_code": result.returncode,
                                       "stdout": result.stdout, "stderr": result.stderr}))
    return result


def remote(program, *, stdin=None):
    """Use only the checked existing master and modern project Python for standard-library metadata."""
    return invoke(SSH + [shlex.join([REMOTE_PYTHON, "-c", program])], stdin=stdin, timeout=300)


def digest(path):
    """Hash bundle bytes incrementally instead of loading them into memory."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2 ** 20), b""):
            value.update(block)
    return value.hexdigest()


REMOTE_FUNCTIONS = '''import hashlib,json,os,resource,subprocess,sys
from pathlib import Path
base=Path(REMOTE_BASE)
repository=base/"repository.git"
commands=[]
def limit():
    resource.setrlimit(resource.RLIMIT_AS,(2*2**30,2*2**30))
def invoke(command,allowed=(0,)):
    commands.append(command)
    result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=180,preexec_fn=limit)
    if result.returncode not in allowed:
        raise RuntimeError(json.dumps({"command":command,"exit_code":result.returncode,"stdout":result.stdout,"stderr":result.stderr}))
    return result
def old_roots():
    records=[]
    expected=[("numerical_source_d28e6735e797","d28e6735e79788f618acb0ebfde8609db1f776af"),
              ("study_helpers_535efed3d25c","535efed3d25c9276bba5f503b3825b1a68c5183d"),("study_helpers_8118d12e7924",BASIS),("jaxtari_source_04ab9aec318c","04ab9aec318c72162e2e53503a594a489d7e162b")]
    for name,commit in expected:
        root=base/"checkouts"/name
        top=invoke(["git","-C",str(root),"rev-parse","--show-toplevel"]).stdout.strip()
        actual=invoke(["git","-C",str(root),"rev-parse","HEAD"]).stdout.strip()
        branch=invoke(["git","-C",str(root),"symbolic-ref","--quiet","HEAD"],(0,1))
        status=invoke(["git","-C",str(root),"status","--porcelain","--untracked-files=all"]).stdout
        if top!=str(root) or actual!=commit or branch.returncode!=1 or status:
            raise RuntimeError("an existing pinned checkout is changed; preserve it: "+str(root))
        records.append({"root":str(root),"commit":commit,"tree":invoke(["git","-C",str(root),"rev-parse","HEAD^{tree}"]).stdout.strip(),
                        "detached_head":True,"git_status_porcelain":status})
    return records
def sha(path):
    value=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(2**20),b""):value.update(block)
    return value.hexdigest()
'''


def main():
    """Check, incrementally transfer and verify only the explicitly published helper source."""
    master_command = ["ssh", "-o", "BatchMode=yes", "-O", "check", "aces"]
    master = subprocess.run(master_command, capture_output=True, text=True, timeout=10)
    COMMANDS.append(master_command)
    save("master_transport_check.json", {"exit_code": master.returncode, "stdout": master.stdout,
        "stderr": master.stderr, "purpose": "necessary existing-master check immediately before source transport",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(), "authentication_attempted": False})
    if master.returncode:
        save("skipped.json", {"reason": "existing ACES master is closed", "authentication_attempted": False,
                              "jobs_submitted": 0, "remote_files_changed": False})
        return

    # Verify the existing repository and three old detached roots before any remote metadata mutation.
    definitions = "REMOTE_BASE=" + repr(REMOTE_BASE) + "\nBASIS=" + repr(BASIS) + "\n"
    preflight = remote(definitions + REMOTE_FUNCTIONS + '''
if invoke(["git","--git-dir",str(repository),"rev-parse","--is-bare-repository"]).stdout.strip()!="true":raise RuntimeError("destination is not the dedicated bare repository")
invoke(["git","--git-dir",str(repository),"cat-file","-e",BASIS+"^{commit}"])
print(json.dumps({"old_roots":old_roots(),"python":sys.executable,"python_version":sys.version,"commands":commands}))
''')
    before = json.loads(preflight.stdout)
    save("remote_before.json", before)

    # Bundle only new objects reachable from TIP after the exact already staged BASIS.
    invoke(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", BASIS, TIP])
    tip_tree = invoke(["git", "-C", str(ROOT), "rev-parse", TIP + "^{tree}"]).stdout.strip()
    origin_tip = invoke(["git", "-C", str(ROOT), "rev-parse", "refs/remotes/origin/merge-atari-cleanrl"]).stdout.strip()
    invoke(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", TIP, origin_tip])
    created = False
    try:
        invoke(["git", "-C", str(ROOT), "update-ref", LOCAL_REF, TIP, "0" * 40])
        created = True
        temporary = DIRECTORY / (BUNDLE_NAME + ".incomplete")
        invoke(["git", "-C", str(ROOT), "-c", "pack.threads=2", "-c", "pack.windowMemory=64m",
                "-c", "pack.deltaCacheSize=32m", "-c", "pack.window=5", "-c", "pack.depth=20",
                "bundle", "create", str(temporary), LOCAL_REF, "^" + BASIS], bounded=True)
        verified = invoke(["git", "-C", str(ROOT), "bundle", "verify", str(temporary)], bounded=True)
        heads = invoke(["git", "bundle", "list-heads", str(temporary)]).stdout
        if heads.strip() != TIP + " " + LOCAL_REF:
            raise RuntimeError("incremental bundle contains an unexpected reference")
        bundle = DIRECTORY / BUNDLE_NAME
        temporary.rename(bundle)
    finally:
        if created:
            invoke(["git", "-C", str(ROOT), "update-ref", "-d", LOCAL_REF, TIP])
    expected, size = digest(bundle), bundle.stat().st_size
    save("local_bundle.json", {"kind": "incremental_published_helper_bundle", "source_commit": TIP,
        "source_tree": tip_tree, "basis_commit": BASIS, "published_origin_tip": origin_tip,
        "bundle": str(bundle), "bundle_sha256": expected, "bundle_bytes": size, "bundle_heads": heads,
        "bundle_verify_stdout": verified.stdout, "bundle_verify_stderr": verified.stderr,
        "local_temporary_reference": LOCAL_REF, "local_temporary_reference_removed": True,
        "packing_threads": 2, "address_space_limit_bytes": 2 * 2 ** 30})

    # Stream the exact bundle, preserving any conflicting old target or incomplete transfer.
    transfer_program = '''from pathlib import Path
import hashlib,json,os,sys,tempfile
target=Path(TARGET)
target.parent.mkdir(parents=True,exist_ok=True)
value=hashlib.sha256();count=0
with tempfile.NamedTemporaryFile(prefix="incoming-helpers-11ce-",dir=target.parent,delete=False) as stream:
    temporary=Path(stream.name)
    for block in iter(lambda:sys.stdin.buffer.read(2**20),b""):
        stream.write(block);value.update(block);count+=len(block)
    stream.flush();os.fsync(stream.fileno())
if value.hexdigest()!=EXPECTED or count!=SIZE:raise RuntimeError("transported bundle bytes differ; incomplete file preserved")
try:os.link(temporary,target)
except FileExistsError:
    check=hashlib.sha256()
    with target.open("rb") as stream:
        for block in iter(lambda:stream.read(2**20),b""):check.update(block)
    if check.hexdigest()!=EXPECTED:raise RuntimeError("existing target differs; both files preserved")
temporary.unlink()
directory=os.open(target.parent,os.O_RDONLY);os.fsync(directory);os.close(directory)
print(json.dumps({"bundle":str(target),"sha256":value.hexdigest(),"bytes":count,"authentication_attempted":False}))
'''
    constants = "TARGET=" + repr(REMOTE_BUNDLE) + "\nEXPECTED=" + repr(expected) + "\nSIZE=" + repr(size) + "\n"
    with bundle.open("rb") as stream:
        transported = remote(constants + transfer_program, stdin=stream)
    save("transport_receipt.json", json.loads(transported.stdout))

    # Import into a new named reference and create only the new sparse detached helper root.
    constants = definitions + "TIP=" + repr(TIP) + "\nREF=" + repr(LOCAL_REF) + "\nREMOTE_REF=" + repr(REMOTE_REF)
    constants += "\nBUNDLE=" + repr(REMOTE_BUNDLE) + "\nEXPECTED=" + repr(expected) + "\nROOT=" + repr(REMOTE_CHECKOUT)
    constants += "\nTREE=" + repr(tip_tree) + "\nSPARSE=" + repr(SPARSE) + "\nBEFORE=" + repr(before["old_roots"]) + "\n"
    checkout_program = constants + REMOTE_FUNCTIONS + '''
bundle=Path(BUNDLE)
if sha(bundle)!=EXPECTED:raise RuntimeError("bundle changed before import")
verified=invoke(["git","--git-dir",str(repository),"bundle","verify",str(bundle)])
heads=invoke(["git","bundle","list-heads",str(bundle)]).stdout
if heads.strip()!=TIP+" "+REF:raise RuntimeError("unexpected transported reference")
invoke(["git","--git-dir",str(repository),"-c","pack.threads=2","-c","pack.windowMemory=64m","-c","pack.deltaCacheSize=32m","fetch","--no-tags",str(bundle),REF+":"+REMOTE_REF])
invoke(["git","--git-dir",str(repository),"merge-base","--is-ancestor",BASIS,TIP])
root=Path(ROOT)
created=not root.exists()
if created:
    invoke(["git","--git-dir",str(repository),"worktree","add","--detach","--no-checkout",str(root),TIP])
    invoke(["git","-C",str(root),"sparse-checkout","init","--cone"])
    invoke(["git","-C",str(root),"sparse-checkout","set"]+SPARSE)
    invoke(["git","-C",str(root),"checkout","--detach",TIP])
top=invoke(["git","-C",str(root),"rev-parse","--show-toplevel"]).stdout.strip()
commit=invoke(["git","-C",str(root),"rev-parse","HEAD"]).stdout.strip()
tree=invoke(["git","-C",str(root),"rev-parse","HEAD^{tree}"]).stdout.strip()
branch=invoke(["git","-C",str(root),"symbolic-ref","--quiet","HEAD"],(0,1))
status=invoke(["git","-C",str(root),"status","--porcelain","--untracked-files=all"]).stdout
sparse=invoke(["git","-C",str(root),"sparse-checkout","list"]).stdout.splitlines()
if top!=str(root) or commit!=TIP or tree!=TREE or branch.returncode!=1 or status or set(sparse)!=set(SPARSE):raise RuntimeError("new source checkout differs from the exact clean detached request")
after=old_roots()
if after!=BEFORE:raise RuntimeError("an old pinned root changed during staging")
record={"kind":"verified_incremental_helper_git_checkout","source_commit":commit,"source_tree":tree,"basis_commit":BASIS,
        "bare_repository":str(repository),"remote_reference":REMOTE_REF,"checkout_root":str(root),"created_new_checkout":created,
        "detached_head":True,"git_status_porcelain":status,"sparse_paths":sparse,"bundle":str(bundle),"bundle_sha256":EXPECTED,
        "bundle_verify_stdout":verified.stdout,"bundle_verify_stderr":verified.stderr,"old_roots_before":BEFORE,"old_roots_after":after,
        "commands":commands,"numerical_imports":False,"jobs_submitted":0,"source_admission_changed":False}
receipt=base/("helpers-"+TIP+".receipt.json")
contents=(json.dumps(record,indent=2)+"\\n").encode()
with receipt.open("xb") as stream:stream.write(contents);stream.flush();os.fsync(stream.fileno())
print(json.dumps(record))
'''
    checked = remote(checkout_program)
    save("remote_checkout_receipt.json", json.loads(checked.stdout))
    save("transport_commands.json", {"kind": "incremental_aces_helper_transport", "commands": COMMANDS,
        "existing_master_only": True, "authentication_attempted": False, "jobs_submitted": 0,
        "numerical_imports": False, "source_admission_changed": False,
        "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"prepared_directory": str(DIRECTORY), "bundle_sha256": expected, "bundle_bytes": size,
                      "remote_checkout": REMOTE_CHECKOUT, "source_commit": TIP, "source_tree": tip_tree}), flush=True)


if __name__ == "__main__":
    main()
